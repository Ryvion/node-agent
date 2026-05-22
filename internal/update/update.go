package update

import (
	"archive/tar"
	"archive/zip"
	"bufio"
	"compress/gzip"
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// releasePublicKeyB64 is the pinned Ed25519 public key used to verify
// SHA256SUMS signatures for update artifacts. It is the SOLE trust anchor for
// auto-update and is intentionally NOT overridable at runtime/env — key
// rotation ships as a new signed release binary. (A prior RYV_UPDATE_PUBKEY_B64
// env override let any env-write foothold replace the entire trust anchor.)
const releasePublicKeyB64 = "KZWGe+VQWPy2ypCNpGwPEYlc8FnFVadufGXnbGAk2nE="

// releaseAssetBaseURL is the GitHub Releases download root. Update artifacts
// (SHA256SUMS, SHA256SUMS.sig, per-platform archives) are fetched from here
// keyed by the release tag — NOT from the hub. The hub only advertises a
// version number (an untrusted hint), so it cannot substitute or downgrade
// signed binaries. This is a package var ONLY so in-package tests can point it
// at a local server; there is no runtime/env path to change it.
var releaseAssetBaseURL = "https://github.com/Ryvion/ryvion-node/releases/download"

// testSigningPublicKeyB64, when set by an in-package test, overrides the pinned
// verification key. There is no runtime/env path to set it (not an attack
// surface); production always uses releasePublicKeyB64.
var testSigningPublicKeyB64 string

// NeedsUpdate compares semver strings (with optional "v" prefix).
// Returns true if latest is strictly newer than current.
//
// Special cases:
//   - latest == "" → no update info → false
//   - current == "" or "dev" → unstamped local build → update IF the user
//     hasn't explicitly opted out via RYV_DISABLE_AUTO_UPDATE=1. This is the
//     fix for operator nodes installed via `go install` (which doesn't
//     stamp main.version), which previously got stuck on whatever they
//     compiled against, silently missing every release.
func NeedsUpdate(current, latest string) bool {
	if latest == "" {
		return false
	}
	if current == "" || current == "dev" {
		if strings.EqualFold(strings.TrimSpace(os.Getenv("RYV_DISABLE_AUTO_UPDATE")), "1") ||
			strings.EqualFold(strings.TrimSpace(os.Getenv("RYV_DISABLE_AUTO_UPDATE")), "true") {
			return false
		}
		// Treat dev / unstamped builds as older than any signed release tag.
		// Active developers can opt out via the env flag above.
		return parseSemver(latest) != nil
	}
	cur := parseSemver(current)
	lat := parseSemver(latest)
	if cur == nil || lat == nil {
		return false
	}
	for i := 0; i < 3; i++ {
		if lat[i] > cur[i] {
			return true
		}
		if lat[i] < cur[i] {
			return false
		}
	}
	return false
}

func parseSemver(s string) []int {
	s = strings.TrimPrefix(s, "v")
	parts := strings.SplitN(s, ".", 3)
	if len(parts) != 3 {
		return nil
	}
	out := make([]int, 3)
	for i, p := range parts {
		// Strip pre-release suffix (e.g. "1-beta")
		if idx := strings.IndexByte(p, '-'); idx >= 0 {
			p = p[:idx]
		}
		v, err := strconv.Atoi(p)
		if err != nil {
			return nil
		}
		out[i] = v
	}
	return out
}

// Apply downloads the signed release binary for the given version from GitHub
// Releases (verified against the pinned key, version-bound by tag) and replaces
// the current executable. The hub is NOT in the artifact trust path — it only
// advertises the version number, which is validated and used as the release tag.
func Apply(ctx context.Context, version string) error {
	if !isValidReleaseVersion(version) {
		return fmt.Errorf("refusing update: invalid release version %q", version)
	}
	expectedFile := expectedArchiveFilename()
	if expectedFile == "" {
		return fmt.Errorf("unsupported platform for updates: %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	expectedSHA, err := fetchExpectedChecksum(ctx, version, expectedFile)
	if err != nil {
		return fmt.Errorf("fetch checksums: %w", err)
	}

	downloadURL := releaseAssetURL(version, expectedFile)
	slog.Info("downloading update", "url", downloadURL, "version", releaseTag(version))

	req, err := http.NewRequestWithContext(ctx, "GET", downloadURL, nil)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	resp, err := (&http.Client{Timeout: 5 * time.Minute}).Do(req)
	if err != nil {
		return fmt.Errorf("download: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download returned %d", resp.StatusCode)
	}

	// Save archive to temp file
	tmpArchive, err := os.CreateTemp("", "ryvion-update-*")
	if err != nil {
		return fmt.Errorf("create temp: %w", err)
	}
	defer os.Remove(tmpArchive.Name())
	h := sha256.New()
	if _, err := io.Copy(io.MultiWriter(tmpArchive, h), resp.Body); err != nil {
		tmpArchive.Close()
		return fmt.Errorf("save archive: %w", err)
	}
	tmpArchive.Close()
	gotSHA := hex.EncodeToString(h.Sum(nil))
	if !secureHexEqual(gotSHA, expectedSHA) {
		return fmt.Errorf("checksum mismatch for %s: got %s expected %s", expectedFile, gotSHA, expectedSHA)
	}

	// Extract binary
	var binaryData []byte
	if runtime.GOOS == "windows" {
		binaryData, err = extractFromZip(tmpArchive.Name(), "ryvion-node.exe")
	} else {
		binaryData, err = extractFromTarGz(tmpArchive.Name(), "ryvion-node")
	}
	if err != nil {
		return fmt.Errorf("extract: %w", err)
	}

	// Get current executable path
	exePath, err := os.Executable()
	if err != nil {
		return fmt.Errorf("resolve executable: %w", err)
	}
	exePath, err = filepath.EvalSymlinks(exePath)
	if err != nil {
		return fmt.Errorf("resolve symlinks: %w", err)
	}

	if runtime.GOOS == "windows" {
		return replaceWindows(exePath, binaryData)
	}
	return replaceUnix(exePath, binaryData)
}

func replaceUnix(exePath string, data []byte) error {
	dir := filepath.Dir(exePath)
	tmp, err := os.CreateTemp(dir, ".ryvion-node-update-*")
	if err != nil {
		if runtime.GOOS == "darwin" && os.IsPermission(err) {
			return replaceDarwinUserManaged(exePath, data)
		}
		return fmt.Errorf("create temp binary: %w", err)
	}
	tmpPath := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmpPath)
		if runtime.GOOS == "darwin" && os.IsPermission(err) {
			return replaceDarwinUserManaged(exePath, data)
		}
		return fmt.Errorf("write temp binary: %w", err)
	}
	tmp.Close()
	if err := os.Chmod(tmpPath, 0755); err != nil {
		os.Remove(tmpPath)
		if runtime.GOOS == "darwin" && os.IsPermission(err) {
			return replaceDarwinUserManaged(exePath, data)
		}
		return fmt.Errorf("chmod: %w", err)
	}
	if err := os.Rename(tmpPath, exePath); err != nil {
		os.Remove(tmpPath)
		if runtime.GOOS == "darwin" && os.IsPermission(err) {
			return replaceDarwinUserManaged(exePath, data)
		}
		return fmt.Errorf("rename: %w", err)
	}
	return nil
}

func replaceDarwinUserManaged(previousExePath string, data []byte) error {
	home, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("resolve home for user-managed update: %w", err)
	}
	binDir := filepath.Join(home, ".ryvion", "bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		return fmt.Errorf("create user-managed binary dir: %w", err)
	}
	target := filepath.Join(binDir, "ryvion-node")
	tmp, err := os.CreateTemp(binDir, ".ryvion-node-update-*")
	if err != nil {
		return fmt.Errorf("create user-managed temp binary: %w", err)
	}
	tmpPath := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		_ = os.Remove(tmpPath)
		return fmt.Errorf("write user-managed binary: %w", err)
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("close user-managed binary: %w", err)
	}
	if err := os.Chmod(tmpPath, 0o755); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("chmod user-managed binary: %w", err)
	}
	if err := os.Rename(tmpPath, target); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("install user-managed binary: %w", err)
	}
	if err := rewriteDarwinLaunchAgentBinary(home, previousExePath, target); err != nil {
		return err
	}
	slog.Info("installed update into user-managed macOS path", "path", target)
	return nil
}

func rewriteDarwinLaunchAgentBinary(home, previousExePath, target string) error {
	plistPath := filepath.Join(home, "Library", "LaunchAgents", "com.ryvion.node.plist")
	body, err := os.ReadFile(plistPath)
	if err != nil {
		if os.IsNotExist(err) {
			slog.Warn("macOS LaunchAgent plist not found after user-managed update", "path", plistPath)
			return nil
		}
		return fmt.Errorf("read macOS LaunchAgent plist: %w", err)
	}
	next, changed := rewriteLaunchAgentBinaryContent(string(body), previousExePath, target)
	if !changed {
		slog.Warn("macOS LaunchAgent plist did not contain previous binary path", "path", plistPath, "previous", previousExePath, "target", target)
		return nil
	}
	if err := os.WriteFile(plistPath, []byte(next), 0o644); err != nil {
		return fmt.Errorf("write macOS LaunchAgent plist: %w", err)
	}
	return nil
}

func rewriteLaunchAgentBinaryContent(content, previousExePath, target string) (string, bool) {
	previousExePath = strings.TrimSpace(previousExePath)
	target = strings.TrimSpace(target)
	if content == "" || target == "" {
		return content, false
	}
	if previousExePath != "" && strings.Contains(content, previousExePath) {
		return strings.Replace(content, previousExePath, target, 1), true
	}
	const programArgs = "<key>ProgramArguments</key>"
	idx := strings.Index(content, programArgs)
	if idx < 0 {
		return content, false
	}
	rest := content[idx+len(programArgs):]
	startRel := strings.Index(rest, "<string>")
	if startRel < 0 {
		return content, false
	}
	valueStart := idx + len(programArgs) + startRel + len("<string>")
	endRel := strings.Index(content[valueStart:], "</string>")
	if endRel < 0 {
		return content, false
	}
	valueEnd := valueStart + endRel
	return content[:valueStart] + target + content[valueEnd:], true
}

func replaceWindows(exePath string, data []byte) error {
	installRoot := windowsInstallRootFromExe(exePath)
	canonicalTarget := windowsCanonicalExePath(installRoot)
	updateDir := filepath.Join(installRoot, "updates")
	if err := os.MkdirAll(updateDir, 0755); err != nil {
		return fmt.Errorf("create update dir: %w", err)
	}

	sum := sha256.Sum256(data)
	target := filepath.Join(updateDir, "ryvion-node-"+hex.EncodeToString(sum[:8])+".exe")
	tmp, err := os.CreateTemp(updateDir, ".ryvion-node-update-*.exe")
	if err != nil {
		return fmt.Errorf("create staged binary: %w", err)
	}
	tmpPath := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		_ = os.Remove(tmpPath)
		return fmt.Errorf("write staged binary: %w", err)
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("close staged binary: %w", err)
	}
	if err := os.Rename(tmpPath, target); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("install staged binary: %w", err)
	}

	serviceName := strings.TrimSpace(os.Getenv("RYVION_WINDOWS_SERVICE"))
	if serviceName == "" {
		serviceName = "RyvionNode"
	}
	currentImagePath, err := queryWindowsServiceImagePath(serviceName)
	if err != nil {
		return fmt.Errorf("query Windows service image path: %w", err)
	}
	args := splitWindowsServiceImageArgs(currentImagePath)
	nextImagePath := quoteWindowsArg(target) + args
	if err := setWindowsServiceImagePath(serviceName, nextImagePath); err != nil {
		return fmt.Errorf("set Windows service image path: %w", err)
	}
	if err := scheduleWindowsCanonicalBinaryRefresh(target, canonicalTarget); err != nil {
		slog.Warn("could not schedule Windows canonical binary refresh", "target", canonicalTarget, "staged", target, "error", err)
	}
	slog.Info("staged Windows update and rewired service path", "service", serviceName, "target", target)
	return nil
}

// ReconcileWindowsCanonicalBinary repairs the PATH-visible Program Files binary
// after a staged Windows auto-update. Older updaters can restart the service
// from Program Files\Ryvion\updates\ryvion-node-*.exe while leaving
// Program Files\Ryvion\ryvion-node.exe stale; this lets the newly started
// version complete that first-hop migration itself.
func ReconcileWindowsCanonicalBinary() error {
	if runtime.GOOS != "windows" {
		return nil
	}
	exePath, err := os.Executable()
	if err != nil {
		return fmt.Errorf("resolve executable: %w", err)
	}
	if resolved, err := filepath.EvalSymlinks(exePath); err == nil && strings.TrimSpace(resolved) != "" {
		exePath = resolved
	}
	if !windowsRunningFromStagedUpdate(exePath) {
		return nil
	}
	installRoot := windowsInstallRootFromExe(exePath)
	canonicalTarget := windowsCanonicalExePath(installRoot)
	if strings.EqualFold(strings.TrimSpace(exePath), strings.TrimSpace(canonicalTarget)) {
		return nil
	}
	data, err := os.ReadFile(exePath)
	if err != nil {
		return fmt.Errorf("read staged Windows binary: %w", err)
	}
	if err := writeWindowsCanonicalBinary(canonicalTarget, data); err != nil {
		if scheduleErr := scheduleWindowsCanonicalBinaryRefresh(exePath, canonicalTarget); scheduleErr != nil {
			slog.Warn("could not schedule delayed Windows canonical binary refresh", "target", canonicalTarget, "staged", exePath, "error", scheduleErr)
		}
		return fmt.Errorf("write canonical Windows binary: %w", err)
	}
	serviceName := windowsServiceName()
	currentImagePath, err := queryWindowsServiceImagePath(serviceName)
	if err != nil {
		slog.Warn("could not query Windows service path while reconciling canonical binary", "service", serviceName, "error", err)
		return nil
	}
	nextImagePath := windowsCanonicalServiceImagePath(currentImagePath, canonicalTarget)
	if strings.TrimSpace(nextImagePath) == "" {
		return nil
	}
	if !strings.EqualFold(strings.TrimSpace(currentImagePath), strings.TrimSpace(nextImagePath)) {
		if err := setWindowsServiceImagePath(serviceName, nextImagePath); err != nil {
			return fmt.Errorf("set Windows service path to canonical binary: %w", err)
		}
	}
	slog.Info("reconciled Windows canonical node binary", "service", serviceName, "target", canonicalTarget)
	return nil
}

func writeWindowsCanonicalBinary(target string, data []byte) error {
	dir := windowsDirName(target)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("create canonical binary dir: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".ryvion-node-canonical-*.exe")
	if err != nil {
		return fmt.Errorf("create canonical temp binary: %w", err)
	}
	tmpPath := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		_ = os.Remove(tmpPath)
		return fmt.Errorf("write canonical temp binary: %w", err)
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("close canonical temp binary: %w", err)
	}
	_ = os.Remove(target)
	if err := os.Rename(tmpPath, target); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("install canonical binary: %w", err)
	}
	return nil
}

func windowsCanonicalExePath(installRoot string) string {
	installRoot = strings.TrimRight(strings.TrimSpace(installRoot), `\/`)
	if installRoot == "" {
		return "ryvion-node.exe"
	}
	return installRoot + `\ryvion-node.exe`
}

func windowsRunningFromStagedUpdate(exePath string) bool {
	dir := windowsDirName(exePath)
	return strings.EqualFold(windowsBaseName(dir), "updates")
}

func windowsCanonicalServiceImagePath(currentImagePath, canonicalTarget string) string {
	canonicalTarget = strings.TrimSpace(canonicalTarget)
	if canonicalTarget == "" {
		return ""
	}
	return quoteWindowsArg(canonicalTarget) + splitWindowsServiceImageArgs(currentImagePath)
}

func windowsInstallRootFromExe(exePath string) string {
	dir := windowsDirName(exePath)
	if strings.EqualFold(windowsBaseName(dir), "updates") {
		parent := windowsDirName(dir)
		if parent != "." && parent != "" {
			return parent
		}
	}
	return dir
}

func windowsDirName(path string) string {
	path = strings.TrimRight(strings.TrimSpace(path), `\/`)
	idx := strings.LastIndexAny(path, `\/`)
	if idx < 0 {
		return filepath.Dir(path)
	}
	return path[:idx]
}

func windowsBaseName(path string) string {
	path = strings.TrimRight(strings.TrimSpace(path), `\/`)
	idx := strings.LastIndexAny(path, `\/`)
	if idx < 0 {
		return filepath.Base(path)
	}
	return path[idx+1:]
}

func queryWindowsServiceImagePath(serviceName string) (string, error) {
	key := `HKLM\SYSTEM\CurrentControlSet\Services\` + serviceName
	out, err := exec.Command("reg.exe", "query", key, "/v", "ImagePath").CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("reg query failed: %w: %s", err, strings.TrimSpace(string(out)))
	}
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(strings.ToLower(line), "imagepath") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 3 {
			continue
		}
		return strings.TrimSpace(strings.Join(fields[2:], " ")), nil
	}
	return "", fmt.Errorf("ImagePath value not found")
}

func setWindowsServiceImagePath(serviceName, imagePath string) error {
	key := `HKLM\SYSTEM\CurrentControlSet\Services\` + serviceName
	out, err := exec.Command("reg.exe", "add", key, "/v", "ImagePath", "/t", "REG_EXPAND_SZ", "/d", imagePath, "/f").CombinedOutput()
	if err != nil {
		return fmt.Errorf("reg add failed: %w: %s", err, strings.TrimSpace(string(out)))
	}
	return nil
}

func splitWindowsServiceImageArgs(imagePath string) string {
	imagePath = strings.TrimSpace(imagePath)
	if imagePath == "" {
		return ""
	}
	if strings.HasPrefix(imagePath, `"`) {
		end := strings.Index(imagePath[1:], `"`)
		if end >= 0 {
			return strings.TrimRight(imagePath[end+2:], " \t")
		}
	}
	lower := strings.ToLower(imagePath)
	if idx := strings.Index(lower, ".exe"); idx >= 0 {
		return strings.TrimRight(imagePath[idx+4:], " \t")
	}
	return ""
}

func quoteWindowsArg(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return `""`
	}
	escaped := strings.ReplaceAll(value, `"`, `\"`)
	return `"` + escaped + `"`
}

// Restart restarts the service using the platform's service manager.
// On Windows, we exit with code 1 so the SCM failure-recovery policy
// (restart after 5 s) relaunches the service with the new binary.
// Spawning detached processes from within a service is unreliable
// because Windows terminates child processes when the service stops.
func Restart() error {
	switch runtime.GOOS {
	case "linux":
		return exec.Command("systemctl", "restart", "ryvion-node").Run()
	case "darwin":
		targets := []string{
			fmt.Sprintf("gui/%d/com.ryvion.node", os.Getuid()),
			"system/com.ryvion.node",
		}
		var lastErr error
		for _, target := range targets {
			if err := exec.Command("launchctl", "kickstart", "-k", target).Run(); err == nil {
				return nil
			} else {
				lastErr = err
			}
		}
		return lastErr
	case "windows":
		serviceName := windowsServiceName()
		if err := configureWindowsServiceRecovery(serviceName); err != nil {
			slog.Warn("could not confirm Windows service recovery before update restart", "service", serviceName, "error", err)
		}
		if err := scheduleWindowsServiceStart(serviceName); err != nil {
			slog.Warn("could not schedule Windows service start after update restart", "service", serviceName, "error", err)
		}
		slog.Info("exiting for Windows service restart", "service", serviceName)
		os.Exit(1)
		return nil // unreachable
	default:
		return fmt.Errorf("unsupported platform for restart: %s", runtime.GOOS)
	}
}

func windowsServiceName() string {
	serviceName := strings.TrimSpace(os.Getenv("RYVION_WINDOWS_SERVICE"))
	if serviceName == "" {
		serviceName = "RyvionNode"
	}
	return serviceName
}

func configureWindowsServiceRecovery(serviceName string) error {
	if runtime.GOOS != "windows" {
		return nil
	}
	if err := runWindowsServiceCommand("sc.exe", "failure", serviceName, "reset= 86400", "actions= restart/5000/restart/10000/restart/30000"); err != nil {
		return err
	}
	return runWindowsServiceCommand("sc.exe", "failureflag", serviceName, "1")
}

func scheduleWindowsServiceStart(serviceName string) error {
	if runtime.GOOS != "windows" {
		return nil
	}
	return startWindowsServiceCommand(windowsServiceStartCommand(serviceName)...)
}

func scheduleWindowsCanonicalBinaryRefresh(stagedPath, canonicalPath string) error {
	if runtime.GOOS != "windows" {
		return nil
	}
	return startWindowsCanonicalRefreshCommand(windowsCanonicalRefreshCommand(stagedPath, canonicalPath)...)
}

func windowsServiceStartCommand(serviceName string) []string {
	command := fmt.Sprintf("Start-Sleep -Seconds 3; Start-Service -Name %s", quotePowerShellSingle(serviceName))
	return []string{"cmd.exe", "/C", "start", "", "powershell.exe", "-NoProfile", "-WindowStyle", "Hidden", "-Command", command}
}

func windowsCanonicalRefreshCommand(stagedPath, canonicalPath string) []string {
	command := fmt.Sprintf("$src = %s; $dst = %s; Start-Sleep -Seconds 6; for ($i = 0; $i -lt 12; $i++) { try { Copy-Item -LiteralPath $src -Destination $dst -Force; exit 0 } catch { Start-Sleep -Seconds 2 } }; exit 1",
		quotePowerShellSingle(stagedPath),
		quotePowerShellSingle(canonicalPath),
	)
	return []string{"cmd.exe", "/C", "start", "", "powershell.exe", "-NoProfile", "-WindowStyle", "Hidden", "-Command", command}
}

func quotePowerShellSingle(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "''") + "'"
}

var runWindowsServiceCommand = func(name string, args ...string) error {
	out, err := exec.Command(name, args...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("%s failed: %w: %s", name, err, strings.TrimSpace(string(out)))
	}
	return nil
}

var startWindowsServiceCommand = func(args ...string) error {
	if len(args) == 0 {
		return fmt.Errorf("missing Windows service start command")
	}
	cmd := exec.Command(args[0], args[1:]...)
	if err := cmd.Start(); err != nil {
		return err
	}
	if cmd.Process != nil {
		return cmd.Process.Release()
	}
	return nil
}

var startWindowsCanonicalRefreshCommand = startWindowsServiceCommand

// isValidReleaseVersion rejects anything that is not a clean semver, which also
// blocks path/URL injection via a hub-advertised version string.
func isValidReleaseVersion(version string) bool {
	v := strings.TrimSpace(version)
	if v == "" || strings.ContainsAny(v, "/\\ \t\r\n?#:@") {
		return false
	}
	return parseSemver(v) != nil
}

func releaseTag(version string) string {
	v := strings.TrimSpace(version)
	if !strings.HasPrefix(v, "v") {
		v = "v" + v
	}
	return v
}

// releaseAssetURL builds a version-bound GitHub Releases download URL. The
// release tag is in the path, so an attacker who controls the advertised
// version number still cannot substitute a different version's signed
// artifacts, and the pinned key verifies SHA256SUMS.
func releaseAssetURL(version, asset string) string {
	return strings.TrimRight(releaseAssetBaseURL, "/") + "/" + releaseTag(version) + "/" + asset
}

func expectedArchiveFilename() string {
	switch runtime.GOOS {
	case "windows":
		return fmt.Sprintf("ryvion-node-%s-%s.zip", runtime.GOOS, runtime.GOARCH)
	case "darwin", "linux":
		return fmt.Sprintf("ryvion-node-%s-%s.tar.gz", runtime.GOOS, runtime.GOARCH)
	default:
		return ""
	}
}

func fetchExpectedChecksum(ctx context.Context, version, archiveName string) (string, error) {
	checksums, err := fetchText(ctx, releaseAssetURL(version, "SHA256SUMS"))
	if err != nil {
		return "", fmt.Errorf("download checksums: %w", err)
	}
	sigB64, err := fetchText(ctx, releaseAssetURL(version, "SHA256SUMS.sig"))
	if err != nil {
		return "", fmt.Errorf("download checksums signature: %w", err)
	}
	pub, err := resolveUpdatePublicKey()
	if err != nil {
		return "", err
	}
	if err := verifyChecksumsSignature(pub, []byte(checksums), sigB64); err != nil {
		return "", err
	}

	target := strings.TrimSpace(archiveName)
	if target == "" {
		return "", fmt.Errorf("missing archive name")
	}
	sc := bufio.NewScanner(strings.NewReader(checksums))
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		sum := strings.ToLower(strings.TrimSpace(fields[0]))
		name := strings.TrimSpace(fields[len(fields)-1])
		if filepath.Base(name) != target {
			continue
		}
		if _, err := hex.DecodeString(sum); err != nil {
			return "", fmt.Errorf("invalid checksum format for %s", target)
		}
		return sum, nil
	}
	if err := sc.Err(); err != nil {
		return "", err
	}
	return "", fmt.Errorf("checksum for %s not found", target)
}

func fetchText(ctx context.Context, url string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return "", err
	}
	resp, err := (&http.Client{Timeout: 30 * time.Second}).Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("endpoint %s returned %d", url, resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(body), nil
}

func resolveUpdatePublicKey() (ed25519.PublicKey, error) {
	keyB64 := strings.TrimSpace(testSigningPublicKeyB64)
	if keyB64 == "" {
		keyB64 = strings.TrimSpace(releasePublicKeyB64)
	}
	if keyB64 == "" {
		return nil, fmt.Errorf("missing update signing public key")
	}
	key, err := base64.StdEncoding.DecodeString(keyB64)
	if err != nil {
		return nil, fmt.Errorf("invalid update signing public key encoding")
	}
	if len(key) != ed25519.PublicKeySize {
		return nil, fmt.Errorf("invalid update signing public key size")
	}
	return ed25519.PublicKey(key), nil
}

func verifyChecksumsSignature(pub ed25519.PublicKey, checksums []byte, sigB64 string) error {
	sigRaw, err := base64.StdEncoding.DecodeString(strings.TrimSpace(sigB64))
	if err != nil {
		return fmt.Errorf("invalid checksums signature encoding")
	}
	if len(sigRaw) != ed25519.SignatureSize {
		return fmt.Errorf("invalid checksums signature size")
	}
	if !ed25519.Verify(pub, checksums, sigRaw) {
		return fmt.Errorf("invalid checksums signature")
	}
	return nil
}

func secureHexEqual(a, b string) bool {
	ab, errA := hex.DecodeString(strings.TrimSpace(a))
	bb, errB := hex.DecodeString(strings.TrimSpace(b))
	if errA != nil || errB != nil || len(ab) == 0 || len(ab) != len(bb) {
		return false
	}
	return subtle.ConstantTimeCompare(ab, bb) == 1
}

func extractFromTarGz(archivePath, binaryName string) ([]byte, error) {
	f, err := os.Open(archivePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, err
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		// Match the binary name anywhere in the archive path
		if filepath.Base(hdr.Name) == binaryName && hdr.Typeflag == tar.TypeReg {
			return io.ReadAll(tr)
		}
	}
	return nil, fmt.Errorf("binary %q not found in archive", binaryName)
}

func extractFromZip(archivePath, binaryName string) ([]byte, error) {
	r, err := zip.OpenReader(archivePath)
	if err != nil {
		return nil, err
	}
	defer r.Close()
	for _, f := range r.File {
		if filepath.Base(f.Name) == binaryName {
			rc, err := f.Open()
			if err != nil {
				return nil, err
			}
			defer rc.Close()
			return io.ReadAll(rc)
		}
	}
	return nil, fmt.Errorf("binary %q not found in archive", binaryName)
}
