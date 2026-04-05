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
	"strings"
	"time"
)

type Result struct {
	Hash       string
	ExitCode   int
	Logs       string
	OutputPath string
	Duration   time.Duration
	Metrics    map[string]any
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

	dockerBin, err := resolveDocker()
	if err != nil {
		return nil, fmt.Errorf("docker not found: %w", err)
	}

	name := fmt.Sprintf("ryv_%s", filepath.Base(workDir))
	defer exec.Command(dockerBin, "rm", "-f", name).Run()

	memLimit := strings.TrimSpace(os.Getenv("RYV_CONTAINER_MEMORY"))
	if memLimit == "" {
		memLimit = "8g"
	}
	cpuLimit := strings.TrimSpace(os.Getenv("RYV_CONTAINER_CPUS"))
	if cpuLimit == "" {
		cpuLimit = "4"
	}
	// Determine network mode: finetune/training jobs need network access to
	// download base models from HuggingFace. All other jobs run isolated.
	networkMode := "--network=none"
	if needsNetwork(specJSON) {
		networkMode = "--network=bridge"
	}

	args := []string{"run", "--name", name, "--rm", "-v", workDir + ":/work",
		"--memory", memLimit, "--memory-swap", memLimit, "--cpus", cpuLimit, "--pids-limit", "256",
		"--cpu-shares", "256",
		"--cap-drop=ALL",
		"--security-opt=no-new-privileges:true",
		networkMode}
	if gpuArg := resolveGPUFlag(gpus); gpuArg != "" {
		args = append(args, "--gpus", gpuArg)
	} else if gpus == "auto" && isROCmAvailable() {
		// AMD ROCm GPU passthrough
		args = append(args, "--device=/dev/kfd", "--device=/dev/dri", "--group-add=video")
	}
	args = append(args, image)

	start := time.Now()
	cmd := exec.CommandContext(ctx, dockerBin, args...)
	var out cappedBuffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	runErr := cmd.Run()
	duration := time.Since(start)

	if ctx.Err() != nil {
		killCtx, killCancel := context.WithTimeout(context.Background(), 10*time.Second)
		if err := exec.CommandContext(killCtx, dockerBin, "kill", name).Run(); err != nil {
			slog.Warn("failed to kill timed-out container", "name", name, "error", err)
		}
		killCancel()
	}

	exitCode := 0
	if runErr != nil {
		if ee, ok := runErr.(*exec.ExitError); ok {
			exitCode = ee.ExitCode()
		} else {
			exitCode = -1
		}
	}

	receiptHash := readReceiptHash(filepath.Join(workDir, "receipt.json"))
	metrics := readMetrics(filepath.Join(workDir, "metrics.json"), duration)
	artifactPath, _ := copyArtifact(workDir, workBase)

	hash := receiptHash
	if hash == "" {
		sum := sha256.Sum256(out.Bytes())
		hash = hex.EncodeToString(sum[:])
	}

	result := &Result{
		Hash:       hash,
		ExitCode:   exitCode,
		Logs:       out.Tail(4096),
		OutputPath: artifactPath,
		Duration:   duration,
		Metrics:    metrics,
	}
	return result, runErr
}

// needsNetwork checks if a job spec requires network access inside the container.
// Currently only finetune jobs need this (to download HuggingFace base models).
func needsNetwork(specJSON string) bool {
	var spec map[string]any
	if json.Unmarshal([]byte(specJSON), &spec) != nil {
		return false
	}
	task, _ := spec["task"].(string)
	return task == "finetune"
}

// prefetchPayloadURL parses specJSON for payload_url, training_data_url, or
// audio_url fields and downloads them into workDir so the container (which
// runs with --network=none) can access them as local files.
func prefetchPayloadURL(ctx context.Context, specJSON, workDir string) error {
	var spec map[string]any
	if err := json.Unmarshal([]byte(specJSON), &spec); err != nil {
		return nil // not JSON, skip
	}
	downloads := map[string]string{
		"payload_url":       "payload.bin",
		"training_data_url": "training.jsonl",
		"audio_url":         "input_audio",
		"input_url":         "input.bin",
		"model_url":         "model.bin",
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
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return err
	}
	client := &http.Client{Timeout: 10 * time.Minute}
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

// resolveDocker finds the docker binary, checking well-known paths if it's
// not on the current PATH (common when running as a background service).
func resolveDocker() (string, error) {
	if p, err := exec.LookPath("docker"); err == nil {
		return p, nil
	}
	for _, p := range []string{
		"/usr/local/bin/docker",
		"/opt/homebrew/bin/docker",
		"/usr/bin/docker",
		"/snap/bin/docker",
	} {
		if _, err := os.Stat(p); err == nil {
			return p, nil
		}
	}
	return "", fmt.Errorf("docker not found in PATH or common locations")
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

func readReceiptHash(path string) string {
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
		"job.json":     true,
		"receipt.json": true,
		"metrics.json": true,
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
