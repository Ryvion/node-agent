package update

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
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

// NeedsUpdate compares semver strings (with optional "v" prefix).
// Returns true if latest is strictly newer than current.
func NeedsUpdate(current, latest string) bool {
	if latest == "" || current == "" || current == "dev" {
		return false
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

// Apply downloads the latest binary from the hub and replaces the current executable.
func Apply(ctx context.Context, hubBaseURL string) error {
	downloadURL := buildDownloadURL(hubBaseURL)
	slog.Info("downloading update", "url", downloadURL)

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
	if _, err := io.Copy(tmpArchive, resp.Body); err != nil {
		tmpArchive.Close()
		return fmt.Errorf("save archive: %w", err)
	}
	tmpArchive.Close()

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
		return fmt.Errorf("create temp binary: %w", err)
	}
	tmpPath := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		os.Remove(tmpPath)
		return fmt.Errorf("write temp binary: %w", err)
	}
	tmp.Close()
	if err := os.Chmod(tmpPath, 0755); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("chmod: %w", err)
	}
	if err := os.Rename(tmpPath, exePath); err != nil {
		os.Remove(tmpPath)
		return fmt.Errorf("rename: %w", err)
	}
	return nil
}

func replaceWindows(exePath string, data []byte) error {
	oldPath := exePath + ".old"
	_ = os.Remove(oldPath)
	if err := os.Rename(exePath, oldPath); err != nil {
		return fmt.Errorf("rename old binary: %w", err)
	}
	if err := os.WriteFile(exePath, data, 0755); err != nil {
		// Try to restore old binary
		os.Rename(oldPath, exePath)
		return fmt.Errorf("write new binary: %w", err)
	}
	return nil
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
		return exec.Command("launchctl", "kickstart", "-k", "system/com.ryvion.node").Run()
	case "windows":
		slog.Info("exiting for Windows service recovery restart")
		os.Exit(1)
		return nil // unreachable
	default:
		return fmt.Errorf("unsupported platform for restart: %s", runtime.GOOS)
	}
}

func buildDownloadURL(hubBase string) string {
	hubBase = strings.TrimRight(hubBase, "/")
	switch runtime.GOOS {
	case "windows":
		return hubBase + "/download/windows/binary"
	case "darwin":
		return hubBase + "/download/macos/binary?arch=" + runtime.GOARCH
	default:
		if runtime.GOARCH == "arm64" {
			return hubBase + "/download/linux/arm64"
		}
		return hubBase + "/download/linux/binary"
	}
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
