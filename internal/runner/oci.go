package runner

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
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

	workBase := strings.TrimSpace(os.Getenv("RYV_WORK_DIR"))
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
	args := []string{"run", "--name", name, "--rm", "-v", workDir + ":/work",
		"--memory", memLimit, "--memory-swap", memLimit, "--cpus", cpuLimit, "--pids-limit", "256",
		"--cpu-shares", "256",
		"--cap-drop=ALL",
		"--security-opt=no-new-privileges:true"}
	if gpuArg := resolveGPUFlag(gpus); gpuArg != "" {
		args = append(args, "--gpus", gpuArg)
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
	candidates := []string{
		filepath.Join(workDir, "output"),
		filepath.Join(workDir, "output.bin"),
		filepath.Join(workDir, "result.bin"),
	}
	for _, src := range candidates {
		fi, err := os.Stat(src)
		if err != nil || fi.IsDir() {
			continue
		}
		resolved, err := filepath.EvalSymlinks(src)
		if err != nil {
			continue
		}
		if !strings.HasPrefix(resolved, workDir) {
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
