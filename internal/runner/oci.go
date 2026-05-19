package runner

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
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

	"github.com/Ryvion/ryvion-node/internal/runtimeexec"
)

type Result struct {
	Hash            string
	ExitCode        int
	Logs            string
	OutputPath      string
	Duration        time.Duration
	Metrics         map[string]any
	Metadata        map[string]any
	ReceiptComplete bool
}

// Run executes an OCI image with /work mounted and specJSON written to /work/job.json.
// The container is expected to write receipt.json and optional metrics.json/output artifact files.
func Run(ctx context.Context, image, specJSON, gpus string) (*Result, error) {
	if strings.TrimSpace(image) == "" {
		return nil, fmt.Errorf("image required")
	}

	if strings.TrimSpace(specJSON) == "" {
		specJSON = `{}`
	}

	workBase := resolveWorkBase(runtime.GOOS, os.Getenv)
	if workBase != "" {
		if err := os.MkdirAll(workBase, 0o755); err != nil {
			return nil, fmt.Errorf("create work dir: %w", err)
		}
	}
	workDir, err := os.MkdirTemp(workBase, "ryv_job_*")
	if err != nil {
		return nil, fmt.Errorf("create temp work dir: %w", err)
	}
	defer os.RemoveAll(workDir)

	if err := os.WriteFile(filepath.Join(workDir, "job.json"), []byte(specJSON), 0o644); err != nil {
		return nil, fmt.Errorf("write job.json: %w", err)
	}

	// Pre-download payload URL into the work directory so the container
	// (which runs with --network=none) can access it as a local file.
	if err := prefetchPayloadURL(ctx, specJSON, workDir); err != nil {
		slog.Warn("payload prefetch failed (non-fatal)", "error", err)
	}

	ociExec, err := resolveOCIExecutor()
	if err != nil {
		return nil, fmt.Errorf("OCI runtime not found: %w", err)
	}

	name := fmt.Sprintf("ryv_%s", filepath.Base(workDir))
	defer exec.Command(ociExec.command, ociCommandArgs(ociExec, "rm", "-f", name)...).Run()

	memLimit := strings.TrimSpace(os.Getenv("RYV_CONTAINER_MEMORY"))
	if memLimit == "" {
		memLimit = "8g"
	}
	cpuLimit := strings.TrimSpace(os.Getenv("RYV_CONTAINER_CPUS"))
	if cpuLimit == "" {
		cpuLimit = "4"
	}
	// Managed OCI jobs run with network isolation by default. Inputs are
	// prefetched into /work before the container starts.
	networkMode := "--network=none"
	if needsNetwork(specJSON) {
		networkMode = "--network=bridge"
	}

	// Pull the latest OCI image before running so cached stale images are refreshed.
	pullCtx, pullCancel := context.WithTimeout(ctx, 15*time.Minute)
	pullCmd := exec.CommandContext(pullCtx, ociExec.command, ociCommandArgs(ociExec, "pull", image)...)
	if pullOut, pullErr := pullCmd.CombinedOutput(); pullErr != nil {
		slog.Warn("managed OCI image pull failed (will try cached image)", "image", image, "error", pullErr, "output", string(pullOut[:min(len(pullOut), 200)]))
	} else {
		slog.Info("managed OCI image pull succeeded", "image", image)
	}
	pullCancel()

	args := baseOCIRunArgs(name, workDir, memLimit, cpuLimit, networkMode)
	if gpuArg := resolveGPUFlag(gpus); gpuArg != "" {
		args = append(args, "--gpus", gpuArg)
	} else if gpus == "auto" && isROCmAvailable() {
		// AMD ROCm GPU passthrough
		args = append(args, "--device=/dev/kfd", "--device=/dev/dri", "--group-add=video")
	}
	args = append(args, image)

	start := time.Now()
	cmd := exec.CommandContext(ctx, ociExec.command, ociCommandArgs(ociExec, args...)...)
	var out cappedBuffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	runErr := cmd.Run()
	duration := time.Since(start)

	if ctx.Err() != nil {
		stopContainerGracefully(ociExec, name, abortGracePeriod())
	}

	exitCode := 0
	if runErr != nil {
		if ee, ok := runErr.(*exec.ExitError); ok {
			exitCode = ee.ExitCode()
		} else {
			exitCode = -1
		}
	}

	receiptHash := readReceiptHash(
		filepath.Join(workDir, "receipt.json"),
		filepath.Join(workDir, "receipt.partial.json"),
	)
	receiptComplete := receiptFileHasHash(filepath.Join(workDir, "receipt.json"))
	metrics := readMetrics(filepath.Join(workDir, "metrics.json"), duration)
	artifactPath, _ := copyArtifact(workDir, workBase)

	hash := receiptHash
	if hash == "" {
		sum := sha256.Sum256(out.Bytes())
		hash = hex.EncodeToString(sum[:])
	}

	result := &Result{
		Hash:            hash,
		ExitCode:        exitCode,
		Logs:            out.Tail(32768),
		OutputPath:      artifactPath,
		Duration:        duration,
		Metrics:         metrics,
		ReceiptComplete: receiptComplete,
	}
	return result, runErr
}

func sha256String(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}

func baseOCIRunArgs(name, workDir, memLimit, cpuLimit, networkMode string) []string {
	graceSeconds := int(abortGracePeriod().Seconds())
	if graceSeconds <= 0 {
		graceSeconds = 10
	}
	return []string{"run", "--name", name, "--rm", "-v", workDir + ":/work",
		"--memory", memLimit, "--memory-swap", memLimit, "--cpus", cpuLimit, "--pids-limit", "256",
		"--cpu-shares", "256",
		"--cap-drop=ALL",
		"--security-opt=no-new-privileges:true",
		"--env", "RYV_RECEIPT_PATH=/work/receipt.json",
		"--env", "RYV_PARTIAL_RECEIPT_PATH=/work/receipt.partial.json",
		"--env", fmt.Sprintf("RYV_ABORT_GRACE_SECONDS=%d", graceSeconds),
		networkMode}
}

func abortGracePeriod() time.Duration {
	raw := strings.TrimSpace(os.Getenv("RYV_ABORT_GRACE_PERIOD"))
	if raw != "" {
		if d, err := time.ParseDuration(raw); err == nil && d > 0 {
			return clampAbortGracePeriod(d)
		}
	}
	rawSeconds := strings.TrimSpace(os.Getenv("RYV_ABORT_GRACE_SECONDS"))
	if rawSeconds != "" {
		if seconds, err := strconv.Atoi(rawSeconds); err == nil && seconds > 0 {
			return clampAbortGracePeriod(time.Duration(seconds) * time.Second)
		}
	}
	return 10 * time.Second
}

func clampAbortGracePeriod(d time.Duration) time.Duration {
	if d < time.Second {
		return time.Second
	}
	if d > 60*time.Second {
		return 60 * time.Second
	}
	return d
}

func stopContainerGracefully(ociExec ociExecutor, name string, grace time.Duration) {
	seconds := int(grace.Seconds())
	if seconds <= 0 {
		seconds = 10
	}
	stopCtx, stopCancel := context.WithTimeout(context.Background(), grace+5*time.Second)
	stopErr := exec.CommandContext(stopCtx, ociExec.command, ociCommandArgs(ociExec, "stop", "--time", strconv.Itoa(seconds), name)...).Run()
	stopCancel()
	if stopErr == nil {
		return
	}
	killCtx, killCancel := context.WithTimeout(context.Background(), 10*time.Second)
	if err := exec.CommandContext(killCtx, ociExec.command, ociCommandArgs(ociExec, "kill", name)...).Run(); err != nil {
		slog.Warn("failed to kill timed-out container", "name", name, "stop_error", stopErr, "error", err)
	}
	killCancel()
}

// needsNetwork is an explicit escape hatch for trusted operator-controlled
// workloads. Managed jobs should rely on prefetched inputs and remain
// network-isolated.
func needsNetwork(specJSON string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("RYV_ALLOW_JOB_NETWORK"))) {
	case "1", "true", "yes", "on":
	default:
		return false
	}
	var spec map[string]any
	if json.Unmarshal([]byte(specJSON), &spec) != nil {
		return false
	}
	value, _ := spec["network"].(bool)
	return value
}

// prefetchPayloadURL parses specJSON for trusted workload input URL fields and
// downloads them into workDir so the container (which
// runs with --network=none) can access them as local files.
func prefetchPayloadURL(ctx context.Context, specJSON, workDir string) error {
	var spec map[string]any
	if err := json.Unmarshal([]byte(specJSON), &spec); err != nil {
		return nil // not JSON, skip
	}
	downloads := map[string]string{
		"payload_url": "payload.bin",
		"audio_url":   "input_audio",
		"input_url":   "input.bin",
	}
	for field, filename := range downloads {
		rawURL, ok := spec[field].(string)
		if !ok || strings.TrimSpace(rawURL) == "" {
			continue
		}
		dest := filepath.Join(workDir, filename)
		if err := downloadToFile(ctx, rawURL, dest); err != nil {
			slog.Warn("prefetch download failed", "field", field, "url", rawURL[:min(len(rawURL), 80)], "error", err)
			continue
		}
		slog.Info("prefetched input file", "field", field, "dest", filename, "size", fileSize(dest))
	}
	return nil
}

func downloadToFile(ctx context.Context, rawURL, dest string) error {
	if err := validateDownloadURL(rawURL, allowLoopbackDownloads()); err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return err
	}
	client := restrictedHTTPClient(10*time.Minute, allowLoopbackDownloads())
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, resp.Body)
	return err
}

func fileSize(path string) int64 {
	fi, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return fi.Size()
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func resolveWorkBase(goos string, getenv func(string) string) string {
	if getenv == nil {
		getenv = os.Getenv
	}
	if workBase := strings.TrimSpace(getenv("RYV_WORK_DIR")); workBase != "" {
		return workBase
	}
	if goos != "windows" {
		return ""
	}
	programData := strings.TrimSpace(getenv("ProgramData"))
	if programData == "" {
		programData = `C:\ProgramData`
	}
	return filepath.Join(programData, "Ryvion", "work")
}

type ociExecutor struct {
	command    string
	prefixArgs []string
}

func resolveOCIExecutor() (ociExecutor, error) {
	execConfig, err := runtimeexec.ResolveExecutor(runtime.GOOS, os.Getenv)
	if err != nil {
		return ociExecutor{}, err
	}
	return ociExecutor{command: execConfig.Command, prefixArgs: execConfig.PrefixArgs}, nil
}

func ociCommandArgs(executor ociExecutor, args ...string) []string {
	out := append([]string{}, executor.prefixArgs...)
	return append(out, args...)
}

func resolveGPUFlag(gpus string) string {
	gpus = strings.TrimSpace(strings.ToLower(gpus))
	switch gpus {
	case "", "none", "off":
		return ""
	case "auto":
		if _, err := exec.LookPath("nvidia-smi"); err == nil {
			return "all"
		}
		return ""
	default:
		return gpus
	}
}

func isROCmAvailable() bool {
	_, err := os.Stat("/dev/kfd")
	return err == nil
}

func readReceiptHash(paths ...string) string {
	for _, path := range paths {
		if hash := receiptFileOutputHash(path); hash != "" {
			return hash
		}
	}
	return ""
}

func receiptFileHasHash(path string) bool {
	return receiptFileOutputHash(path) != ""
}

func receiptFileOutputHash(path string) string {
	if strings.HasSuffix(path, ".tmp") {
		return ""
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	var rec struct {
		OutputHash string `json:"output_hash"`
	}
	if err := json.Unmarshal(b, &rec); err != nil {
		return ""
	}
	return trimDigestPrefix(strings.TrimSpace(rec.OutputHash))
}

func readMetrics(path string, duration time.Duration) map[string]any {
	b, err := os.ReadFile(path)
	if err != nil {
		return map[string]any{"duration_ms": duration.Milliseconds()}
	}
	var metrics map[string]any
	if err := json.Unmarshal(b, &metrics); err != nil {
		return map[string]any{"duration_ms": duration.Milliseconds()}
	}
	if metrics == nil {
		metrics = map[string]any{}
	}
	if _, ok := metrics["duration_ms"]; !ok {
		metrics["duration_ms"] = duration.Milliseconds()
	}
	return metrics
}

func copyArtifact(workDir, workBase string) (string, error) {
	workRoot := canonicalPath(workDir)
	candidates := artifactCandidates(workDir)
	for _, src := range candidates {
		fi, err := os.Stat(src)
		if err != nil || fi.IsDir() {
			continue
		}
		resolved, err := filepath.EvalSymlinks(src)
		if err != nil {
			continue
		}
		if !isPathWithin(workRoot, canonicalPath(resolved)) {
			slog.Warn("artifact path traversal blocked", "path", src, "resolved", resolved)
			continue
		}
		targetDir := workBase
		if targetDir == "" {
			targetDir = os.TempDir()
		}
		dst, err := os.CreateTemp(targetDir, "ryv_artifact_*")
		if err != nil {
			return "", err
		}
		defer dst.Close()
		in, err := os.Open(src)
		if err != nil {
			return "", err
		}
		_, copyErr := io.Copy(dst, in)
		_ = in.Close()
		if copyErr != nil {
			_ = os.Remove(dst.Name())
			return "", copyErr
		}
		return dst.Name(), nil
	}
	return "", nil
}

func artifactCandidates(workDir string) []string {
	controlFiles := map[string]bool{
		"job.json":                 true,
		"receipt.json":             true,
		"receipt.partial.json":     true,
		"receipt.partial.json.tmp": true,
		"metrics.json":             true,
		"metrics.partial.json":     true,
		"metrics.partial.json.tmp": true,
	}
	seen := map[string]bool{}
	candidates := []string{}
	add := func(path string) {
		path = strings.TrimSpace(path)
		if path == "" {
			return
		}
		cleaned := filepath.Clean(path)
		if seen[cleaned] {
			return
		}
		seen[cleaned] = true
		candidates = append(candidates, cleaned)
	}

	add(filepath.Join(workDir, "output"))
	add(filepath.Join(workDir, "output.bin"))
	add(filepath.Join(workDir, "result.bin"))

	if outputName := metricsOutputName(filepath.Join(workDir, "metrics.json")); outputName != "" {
		add(filepath.Join(workDir, filepath.Base(outputName)))
	}

	entries, err := os.ReadDir(workDir)
	if err != nil {
		return candidates
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if controlFiles[strings.ToLower(entry.Name())] {
			continue
		}
		add(filepath.Join(workDir, entry.Name()))
	}
	return candidates
}

func metricsOutputName(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	var metrics map[string]any
	if err := json.Unmarshal(b, &metrics); err != nil {
		return ""
	}
	v, _ := metrics["output_name"].(string)
	return strings.TrimSpace(v)
}

func canonicalPath(path string) string {
	if resolved, err := filepath.EvalSymlinks(path); err == nil {
		return filepath.Clean(resolved)
	}
	return filepath.Clean(path)
}

func isPathWithin(root, target string) bool {
	if root == "" || target == "" {
		return false
	}
	rel, err := filepath.Rel(root, target)
	if err != nil {
		return false
	}
	if rel == "." {
		return true
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(os.PathSeparator))
}

func trimDigestPrefix(v string) string {
	if i := strings.IndexByte(v, ':'); i > 0 {
		return v[i+1:]
	}
	return v
}

type cappedBuffer struct {
	buf []byte
}

func (c *cappedBuffer) Write(p []byte) (int, error) {
	c.buf = append(c.buf, p...)
	const limit = 1 << 20
	if len(c.buf) > limit {
		c.buf = c.buf[len(c.buf)-limit:]
	}
	return len(p), nil
}

func (c *cappedBuffer) Bytes() []byte {
	return c.buf
}

func (c *cappedBuffer) Tail(n int) string {
	if n <= 0 || len(c.buf) == 0 {
		return ""
	}
	if len(c.buf) <= n {
		return string(c.buf)
	}
	return string(c.buf[len(c.buf)-n:])
}
