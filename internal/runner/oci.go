package runner

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

type RunResult struct {
	ResultHash string
	Duration   time.Duration
	ExitCode   int
	LogsTail   string
	Metrics    map[string]any
	OutputPath string
}

// RunOCI executes a runner image with a mounted work dir containing job.json.
// It expects the container to read /work/job.json and write /work/receipt.json, /work/metrics.json.
func RunOCI(ctx context.Context, image string, jobJSON []byte, gpus string) (*RunResult, error) {
	if image == "" {
		return nil, fmt.Errorf("image required")
	}
	if gpus == "" {
		gpus = "all"
	}
	workdir, err := os.MkdirTemp("", "ak_work_*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(workdir)
	if err := os.WriteFile(filepath.Join(workdir, "job.json"), jobJSON, 0644); err != nil {
		return nil, err
	}

	args := []string{"run", "--rm"}
	useGPU := false
	if gpus == "auto" {
		if _, err := exec.LookPath("nvidia-smi"); err == nil {
			useGPU = true
		}
	} else if gpus != "" {
		useGPU = true
	}
	if useGPU {
		args = append(args, "--gpus", map[string]string{"auto": "all"}[gpus])
		if args[len(args)-1] == "" {
			args[len(args)-1] = gpus
		}
	}
	args = append(args, "-v", workdir+":/work", image)
	start := time.Now()
	cmd := exec.CommandContext(ctx, "docker", args...)
	var outBuf limitedBuffer
	cmd.Stdout = &outBuf
	cmd.Stderr = &outBuf
	err = cmd.Run()
	exitCode := 0
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			exitCode = ee.ExitCode()
		} else {
			exitCode = -1
		}
	}
	dur := time.Since(start)

	receiptPath := filepath.Join(workdir, "receipt.json")
	metricsPath := filepath.Join(workdir, "metrics.json")
	var resultHash string
	var metrics map[string]any
	if b, rerr := os.ReadFile(receiptPath); rerr == nil {
		var rec struct {
			OutputHash  string `json:"output_hash"`
			ImageDigest string `json:"image_digest"`
		}
		if jerr := json.Unmarshal(b, &rec); jerr == nil && rec.OutputHash != "" {
			resultHash = trimAlgo(rec.OutputHash)
		}
	}
	if b, merr := os.ReadFile(metricsPath); merr == nil {
		_ = json.Unmarshal(b, &metrics)
	} else {
		metrics = map[string]any{"duration_ms": dur.Milliseconds()}
	}
	if resultHash == "" {
		sum := sha256.Sum256(outBuf.Bytes())
		resultHash = hex.EncodeToString(sum[:])
	}
	outPath := filepath.Join(workdir, "output")
	return &RunResult{ResultHash: resultHash, Duration: dur, ExitCode: exitCode, LogsTail: outBuf.Tail(2048), Metrics: metrics, OutputPath: outPath}, nil
}

type limitedBuffer struct{ buf []byte }

func (l *limitedBuffer) Write(p []byte) (int, error) {
	l.buf = append(l.buf, p...)
	if len(l.buf) > 1<<20 {
		l.buf = l.buf[len(l.buf)-(1<<20):]
	}
	return len(p), nil
}
func (l *limitedBuffer) Bytes() []byte { return l.buf }
func (l *limitedBuffer) Tail(n int) string {
	if n <= 0 {
		return ""
	}
	if len(l.buf) <= n {
		return string(l.buf)
	}
	return string(l.buf[len(l.buf)-n:])
}

func trimAlgo(s string) string {
	for i := 0; i < len(s); i++ {
		if s[i] == ':' {
			return s[i+1:]
		}
	}
	return s
}
