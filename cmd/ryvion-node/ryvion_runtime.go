package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/Ryvion/node-agent/internal/blob"
	"github.com/Ryvion/node-agent/internal/hub"
)

const (
	runtimeTaskImageGeneration = "image_generation"
	flux2Klein4BLocalModel     = "flux-2-klein-4b-local"
	flux2Klein4BMinVRAMMB      = 13000
	flux2Klein4BMinDiskGB      = 40
)

type ryvionRuntimeSpec struct {
	ExecutorKind string `json:"executor_kind"`
	RuntimeTask  string `json:"runtime_task"`
	Model        string `json:"model"`
	Prompt       string `json:"prompt"`
	Width        int    `json:"width"`
	Height       int    `json:"height"`
	OutputFile   string `json:"output_file"`
	Index        int    `json:"index"`
}

func (ryvionRuntimeEngine) Execute(ctx context.Context, work *hub.WorkAssignment, execCtx executionContext) (*runnerResultSnapshot, error) {
	spec, err := parseRyvionRuntimeSpec(work.SpecJSON)
	if err != nil {
		return submitRyvionRuntimeFailure(ctx, work, execCtx, "invalid_runtime_spec", err)
	}
	switch strings.TrimSpace(spec.RuntimeTask) {
	case runtimeTaskImageGeneration:
		return runRyvionRuntimeImageGeneration(ctx, work, execCtx, spec)
	default:
		return submitRyvionRuntimeFailure(ctx, work, execCtx, "unsupported_runtime_task", fmt.Errorf("unsupported runtime task %q", spec.RuntimeTask))
	}
}

func parseRyvionRuntimeSpec(specJSON string) (ryvionRuntimeSpec, error) {
	var spec ryvionRuntimeSpec
	if strings.TrimSpace(specJSON) == "" {
		return spec, fmt.Errorf("missing runtime spec")
	}
	if err := json.Unmarshal([]byte(specJSON), &spec); err != nil {
		return spec, err
	}
	spec.ExecutorKind = strings.TrimSpace(spec.ExecutorKind)
	spec.RuntimeTask = strings.TrimSpace(spec.RuntimeTask)
	spec.Model = strings.TrimSpace(spec.Model)
	spec.Prompt = strings.TrimSpace(spec.Prompt)
	spec.OutputFile = strings.TrimSpace(spec.OutputFile)
	if spec.Width <= 0 {
		spec.Width = 1024
	}
	if spec.Height <= 0 {
		spec.Height = 1024
	}
	if spec.Width < 256 || spec.Width > 2048 || spec.Height < 256 || spec.Height > 2048 {
		return spec, fmt.Errorf("invalid image size %dx%d", spec.Width, spec.Height)
	}
	if spec.RuntimeTask == "" {
		return spec, fmt.Errorf("runtime_task required")
	}
	return spec, nil
}

func runRyvionRuntimeImageGeneration(ctx context.Context, work *hub.WorkAssignment, execCtx executionContext, spec ryvionRuntimeSpec) (*runnerResultSnapshot, error) {
	if !strings.EqualFold(spec.Model, flux2Klein4BLocalModel) {
		return submitRyvionRuntimeFailure(ctx, work, execCtx, "unsupported_image_model", fmt.Errorf("unsupported local image model %q", spec.Model))
	}
	start := time.Now()
	workDir, err := os.MkdirTemp("", "ryvion-runtime-image-*")
	if err != nil {
		return submitRyvionRuntimeFailure(ctx, work, execCtx, "runtime_workdir_failed", err)
	}
	defer os.RemoveAll(workDir)
	outputPath := filepath.Join(workDir, "output.png")

	var logs string
	if helper, ok := resolveFlux2LocalHelper(); ok {
		logs, err = runFlux2LocalHelper(ctx, helper, spec, outputPath)
	} else if runtimeFixturesEnabled() {
		logs, err = writeFixturePNG(outputPath, spec)
	} else {
		err = fmt.Errorf("local FLUX.2 helper is not installed; node should not advertise runtime:image:%s yet", flux2Klein4BLocalModel)
	}
	if err != nil {
		return submitRyvionRuntimeFailure(ctx, work, execCtx, "runtime_task_failed", err)
	}

	uploadRes, uploadErr := blob.Upload(ctx, execCtx.client, work.JobID, outputPath)
	if uploadErr != nil {
		return submitRyvionRuntimeFailure(ctx, work, execCtx, "artifact_upload_failed", uploadErr)
	}
	resultHash := uploadRes.Hash
	if resultHash == "" {
		body, _ := os.ReadFile(outputPath)
		sum := sha256.Sum256(body)
		resultHash = hex.EncodeToString(sum[:])
	}
	metadata := receiptMetadataBase(
		work,
		execCtx.runtimeManager.ReceiptMetadata(execCtx.gpuDetected),
		map[string]any{
			"executor":          executorKindRyvionRuntime,
			"runtime_task":      runtimeTaskImageGeneration,
			"model":             spec.Model,
			"duration_ms":       time.Since(start).Milliseconds(),
			"exit_code":         0,
			"stdout_tail":       tailString(logs, 2048),
			"blob_url":          uploadRes.URL,
			"object_key":        uploadRes.Key,
			"artifact_sha256":   resultHash,
			"artifact_mime":     "image/png",
			"width":             spec.Width,
			"height":            spec.Height,
			"runtime_event_log": []string{"runtime.task_started", "artifact.uploaded", "receipt.submitted"},
		},
	)
	if strings.TrimSpace(uploadRes.Key) != "" {
		metadata["manifest_key"] = uploadRes.Key + ".manifest.json"
	}
	receipt := hub.Receipt{
		JobID:         work.JobID,
		ResultHashHex: resultHash,
		MeteringUnits: unitsForWork(work),
		Metadata:      metadata,
	}
	if err := submitReceiptWithRetry(ctx, execCtx.client, receipt); err != nil {
		return &runnerResultSnapshot{
			DurationMs:    time.Since(start).Milliseconds(),
			ResultHashHex: resultHash,
			MeteringUnits: unitsForWork(work),
			BlobURL:       uploadRes.URL,
			ObjectKey:     uploadRes.Key,
			Metadata:      metadata,
		}, err
	}
	return &runnerResultSnapshot{
		DurationMs:    time.Since(start).Milliseconds(),
		ResultHashHex: resultHash,
		MeteringUnits: unitsForWork(work),
		BlobURL:       uploadRes.URL,
		ObjectKey:     uploadRes.Key,
		Metadata:      metadata,
	}, nil
}

func submitRyvionRuntimeFailure(ctx context.Context, work *hub.WorkAssignment, execCtx executionContext, code string, runErr error) (*runnerResultSnapshot, error) {
	msg := code
	if runErr != nil && strings.TrimSpace(runErr.Error()) != "" {
		msg = runErr.Error()
	}
	sum := sha256.Sum256([]byte(work.JobID + ":" + code + ":" + msg))
	hash := hex.EncodeToString(sum[:])
	metadata := receiptMetadataBase(
		work,
		execCtx.runtimeManager.ReceiptMetadata(execCtx.gpuDetected),
		map[string]any{
			"executor":          executorKindRyvionRuntime,
			"runtime_task":      "unknown",
			"exit_code":         1,
			"error_code":        code,
			"error":             msg,
			"runtime_event_log": []string{"runtime.task_started", "runtime.task_failed", "receipt.submitted"},
		},
	)
	_ = submitReceiptWithRetry(ctx, execCtx.client, hub.Receipt{
		JobID:         work.JobID,
		ResultHashHex: hash,
		MeteringUnits: 0,
		Metadata:      metadata,
	})
	return &runnerResultSnapshot{ResultHashHex: hash, Metadata: metadata, ExitCode: 1}, runErr
}

func resolveFlux2LocalHelper() (string, bool) {
	if value := strings.TrimSpace(os.Getenv("RYV_FLUX2_HELPER")); value != "" {
		if executableExists(value) {
			return value, true
		}
		if path, err := exec.LookPath(value); err == nil && strings.TrimSpace(path) != "" {
			return path, true
		}
	}
	for _, name := range []string{"ryvion-flux2-klein", "ryvion-image-runtime"} {
		if path, err := exec.LookPath(name); err == nil && strings.TrimSpace(path) != "" {
			return path, true
		}
	}
	for _, path := range defaultImageRuntimeHelperPaths() {
		if executableExists(path) {
			return path, true
		}
	}
	return "", false
}

func defaultImageRuntimeHelperPaths() []string {
	switch runtime.GOOS {
	case "windows":
		root := strings.TrimSpace(os.Getenv("ProgramFiles"))
		if root == "" {
			root = `C:\Program Files`
		}
		return []string{
			filepath.Join(root, "Ryvion", "runtime", "helpers", "ryvion-image-runtime.exe"),
			filepath.Join(root, "Ryvion", "runtime", "helpers", "ryvion-image-runtime.cmd"),
		}
	default:
		return []string{
			"/opt/ryvion/runtime/helpers/ryvion-image-runtime",
			"/usr/local/bin/ryvion-image-runtime",
		}
	}
}

func runFlux2LocalHelper(ctx context.Context, helper string, spec ryvionRuntimeSpec, outputPath string) (string, error) {
	args := []string{
		"--model", flux2Klein4BLocalModel,
		"--prompt", spec.Prompt,
		"--output", outputPath,
		"--width", fmt.Sprintf("%d", spec.Width),
		"--height", fmt.Sprintf("%d", spec.Height),
	}
	command := helper
	commandArgs := args
	lowerHelper := strings.ToLower(helper)
	if strings.HasSuffix(lowerHelper, ".ps1") {
		command = "powershell"
		commandArgs = append([]string{"-NoProfile", "-ExecutionPolicy", "Bypass", "-File", helper}, args...)
	}
	if strings.HasSuffix(lowerHelper, ".py") {
		command = "python"
		commandArgs = append([]string{helper}, args...)
	}
	cmd := exec.CommandContext(ctx, command, commandArgs...)
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	err := cmd.Run()
	return buf.String(), err
}

func runtimeFixturesEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("RYV_ENABLE_RUNTIME_FIXTURES"))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func localFlux2KleinReady(capsVRAMBytes uint64, diskGB uint64, gpuReady bool) bool {
	if !gpuReady {
		return false
	}
	if capsVRAMBytes/1024/1024 < flux2Klein4BMinVRAMMB {
		return false
	}
	if diskGB < flux2Klein4BMinDiskGB {
		return false
	}
	if _, ok := resolveFlux2LocalHelper(); ok {
		return true
	}
	return runtimeFixturesEnabled()
}

func executableExists(path string) bool {
	if runtime.GOOS == "windows" && !strings.Contains(filepath.Base(path), ".") {
		for _, ext := range []string{".exe", ".cmd", ".bat", ".ps1"} {
			if executableExists(path + ext) {
				return true
			}
		}
	}
	info, err := os.Stat(path)
	if err != nil || info.IsDir() {
		return false
	}
	return info.Mode()&0o111 != 0 || runtime.GOOS == "windows"
}

func writeFixturePNG(outputPath string, spec ryvionRuntimeSpec) (string, error) {
	img := image.NewRGBA(image.Rect(0, 0, spec.Width, spec.Height))
	sum := sha256.Sum256([]byte(spec.Prompt + spec.Model))
	for y := 0; y < spec.Height; y++ {
		for x := 0; x < spec.Width; x++ {
			i := byte((x + y) % 256)
			c := color.RGBA{
				R: sum[0] ^ i,
				G: sum[7] ^ byte(x%256),
				B: sum[15] ^ byte(y%256),
				A: 255,
			}
			img.SetRGBA(x, y, c)
		}
	}
	f, err := os.Create(outputPath)
	if err != nil {
		return "", err
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		return "", err
	}
	return "runtime fixture image generated; not a production FLUX output", nil
}

func unitsForWork(work *hub.WorkAssignment) uint64 {
	if work != nil && work.Units > 0 {
		return uint64(work.Units)
	}
	return 1
}

func tailString(value string, max int) string {
	value = strings.TrimSpace(value)
	if max <= 0 || len(value) <= max {
		return value
	}
	return value[len(value)-max:]
}
