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
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/Ryvion/node-agent/internal/blob"
	"github.com/Ryvion/node-agent/internal/hub"
	"github.com/Ryvion/node-agent/internal/hw"
)

const (
	runtimeTaskImageGeneration = "image_generation"
	runtimeTaskPrepare         = "runtime_prepare"
	flux2Klein4BLocalModel     = "flux-2-klein-4b-local"
	flux2Klein4BMinVRAMMB      = 13000
	flux2Klein4BMinRAMGB       = 16
	flux2Klein4BMinCPUCores    = 4
	flux2Klein4BMinDiskGB      = 40
	flux2Klein4BReadyMarker    = ".model-flux2-klein-ready-v2"
	flux2Klein4BWarmingMarker  = ".model-flux2-klein-warming"
)

type ryvionRuntimeSpec struct {
	ExecutorKind  string `json:"executor_kind"`
	RuntimeTask   string `json:"runtime_task"`
	RuntimeFamily string `json:"runtime_family"`
	Model         string `json:"model"`
	Prompt        string `json:"prompt"`
	Width         int    `json:"width"`
	Height        int    `json:"height"`
	OutputFile    string `json:"output_file"`
	Index         int    `json:"index"`
}

func (ryvionRuntimeEngine) Execute(ctx context.Context, work *hub.WorkAssignment, execCtx executionContext) (*runnerResultSnapshot, error) {
	spec, err := parseRyvionRuntimeSpec(work.SpecJSON)
	if err != nil {
		return submitRyvionRuntimeFailure(ctx, work, execCtx, "invalid_runtime_spec", err)
	}
	switch strings.TrimSpace(spec.RuntimeTask) {
	case runtimeTaskImageGeneration:
		return runRyvionRuntimeImageGeneration(ctx, work, execCtx, spec)
	case runtimeTaskPrepare, "prepare":
		return runRyvionRuntimePrepare(ctx, work, execCtx, spec)
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
	spec.RuntimeFamily = strings.TrimSpace(spec.RuntimeFamily)
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

func runRyvionRuntimePrepare(ctx context.Context, work *hub.WorkAssignment, execCtx executionContext, spec ryvionRuntimeSpec) (*runnerResultSnapshot, error) {
	if !strings.EqualFold(spec.Model, flux2Klein4BLocalModel) {
		return submitRyvionRuntimeFailure(ctx, work, execCtx, "unsupported_prepare_model", fmt.Errorf("unsupported runtime prepare model %q", spec.Model))
	}
	start := time.Now()
	helper, ok := resolveFlux2LocalHelper()
	if !ok {
		return submitRyvionRuntimeFailure(ctx, work, execCtx, "runtime_helper_missing", fmt.Errorf("local FLUX.2 helper is not installed"))
	}
	logs, err := runFlux2LocalPrepareHelper(ctx, helper, spec.Model)
	if err != nil {
		return submitRyvionRuntimeFailureWithLogs(ctx, work, execCtx, "runtime_prepare_failed", err, logs, runtimeTaskPrepare)
	}
	if !localFlux2KleinModelCacheReady() && !runtimeFixturesEnabled() {
		err := fmt.Errorf("runtime prepare finished but readiness marker %s is missing", flux2Klein4BReadyMarker)
		return submitRyvionRuntimeFailureWithLogs(ctx, work, execCtx, "runtime_prepare_not_ready", err, logs, runtimeTaskPrepare)
	}
	resultHash := hashRuntimePrepareResult(spec.Model, logs)
	metadata := receiptMetadataBase(
		work,
		execCtx.runtimeManager.ReceiptMetadata(execCtx.gpuDetected),
		map[string]any{
			"executor":          executorKindRyvionRuntime,
			"runtime_task":      runtimeTaskPrepare,
			"runtime_family":    "image_generation",
			"model":             spec.Model,
			"duration_ms":       time.Since(start).Milliseconds(),
			"exit_code":         0,
			"stdout_tail":       tailString(logs, 4096),
			"cache_ready":       true,
			"ready_marker":      filepath.Join(imageRuntimeRoot(), flux2Klein4BReadyMarker),
			"runtime_event_log": []string{"runtime.prepare_started", "model.download_completed", "runtime.smoke_tested", "receipt.submitted"},
		},
	)
	receipt := hub.Receipt{
		JobID:         work.JobID,
		ResultHashHex: resultHash,
		MeteringUnits: 0,
		Metadata:      metadata,
	}
	if err := submitReceiptWithRetry(ctx, execCtx.client, receipt); err != nil {
		return &runnerResultSnapshot{
			DurationMs:    time.Since(start).Milliseconds(),
			ResultHashHex: resultHash,
			MeteringUnits: 0,
			Metadata:      metadata,
		}, err
	}
	return &runnerResultSnapshot{
		DurationMs:    time.Since(start).Milliseconds(),
		ResultHashHex: resultHash,
		MeteringUnits: 0,
		Metadata:      metadata,
	}, nil
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
		return submitRyvionRuntimeFailureWithLogs(ctx, work, execCtx, "runtime_task_failed", err, logs, runtimeTaskImageGeneration)
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
	return submitRyvionRuntimeFailureWithLogs(ctx, work, execCtx, code, runErr, "", "unknown")
}

func submitRyvionRuntimeFailureWithLogs(ctx context.Context, work *hub.WorkAssignment, execCtx executionContext, code string, runErr error, logs, runtimeTask string) (*runnerResultSnapshot, error) {
	msg := code
	if runErr != nil && strings.TrimSpace(runErr.Error()) != "" {
		msg = runErr.Error()
	}
	runtimeTask = strings.TrimSpace(runtimeTask)
	if runtimeTask == "" {
		runtimeTask = "unknown"
	}
	sum := sha256.Sum256([]byte(work.JobID + ":" + code + ":" + msg))
	hash := hex.EncodeToString(sum[:])
	extra := map[string]any{
		"executor":          executorKindRyvionRuntime,
		"runtime_task":      runtimeTask,
		"exit_code":         1,
		"error_code":        code,
		"error":             msg,
		"runtime_event_log": []string{"runtime.task_started", "runtime.task_failed", "receipt.submitted"},
	}
	if strings.TrimSpace(logs) != "" {
		extra["stdout_tail"] = tailString(logs, 4096)
		extra["stderr_tail"] = tailString(logs, 4096)
	}
	metadata := receiptMetadataBase(
		work,
		execCtx.runtimeManager.ReceiptMetadata(execCtx.gpuDetected),
		extra,
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
	paths := []string{}
	if home, err := os.UserHomeDir(); err == nil && strings.TrimSpace(home) != "" {
		switch runtime.GOOS {
		case "windows":
			paths = append(paths,
				filepath.Join(home, ".ryvion", "runtime", "helpers", "ryvion-image-runtime.cmd"),
				filepath.Join(home, ".ryvion", "runtime", "helpers", "ryvion-image-runtime.ps1"),
			)
		default:
			paths = append(paths, filepath.Join(home, ".ryvion", "runtime", "helpers", "ryvion-image-runtime"))
		}
	}
	switch runtime.GOOS {
	case "windows":
		root := strings.TrimSpace(os.Getenv("ProgramFiles"))
		if root == "" {
			root = `C:\Program Files`
		}
		paths = append(paths,
			filepath.Join(root, "Ryvion", "runtime", "helpers", "ryvion-image-runtime.exe"),
			filepath.Join(root, "Ryvion", "runtime", "helpers", "ryvion-image-runtime.cmd"),
		)
	default:
		paths = append(paths,
			"/opt/ryvion/runtime/helpers/ryvion-image-runtime",
			"/usr/local/bin/ryvion-image-runtime",
		)
	}
	return paths
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

func runFlux2LocalPrepareHelper(ctx context.Context, helper string, model string) (string, error) {
	args := []string{"--prepare", "--model", model}
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

func hashRuntimePrepareResult(model string, logs string) string {
	parts := []string{executorKindRyvionRuntime, runtimeTaskPrepare, strings.TrimSpace(model)}
	if marker, err := os.ReadFile(filepath.Join(imageRuntimeRoot(), flux2Klein4BReadyMarker)); err == nil {
		parts = append(parts, string(marker))
	}
	parts = append(parts, logs)
	sum := sha256.Sum256([]byte(strings.Join(parts, "\n")))
	return hex.EncodeToString(sum[:])
}

func runtimeFixturesEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("RYV_ENABLE_RUNTIME_FIXTURES"))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func localFlux2KleinReady(caps hw.CapSet, diskGB uint64, gpuReady bool) bool {
	if diskGB < flux2Klein4BMinDiskGB {
		return false
	}
	if runtimeFixturesEnabled() && localFlux2KleinHardwareEligible(caps, gpuReady) {
		return true
	}
	if _, ok := resolveFlux2LocalHelper(); !ok {
		return false
	}
	return localFlux2KleinHardwareEligible(caps, gpuReady) && localFlux2KleinModelCacheReady()
}

func localFlux2KleinPreparing(caps hw.CapSet, diskGB uint64, gpuReady bool) bool {
	if diskGB < flux2Klein4BMinDiskGB || !localFlux2KleinHardwareEligible(caps, gpuReady) {
		return false
	}
	if _, ok := resolveFlux2LocalHelper(); !ok {
		return false
	}
	if localFlux2KleinModelCacheReady() {
		return false
	}
	if _, err := os.Stat(filepath.Join(imageRuntimeRoot(), flux2Klein4BWarmingMarker)); err == nil {
		return true
	}
	return false
}

func localFlux2KleinPrepareEligible(caps hw.CapSet, diskGB uint64, gpuReady bool) bool {
	if diskGB < flux2Klein4BMinDiskGB {
		return false
	}
	if !localFlux2KleinFastGPUEligible(caps, gpuReady) {
		return false
	}
	if _, ok := resolveFlux2LocalHelper(); !ok {
		return false
	}
	return !localFlux2KleinModelCacheReady()
}

func localFlux2KleinFastGPUEligible(caps hw.CapSet, gpuReady bool) bool {
	return gpuReady && caps.VRAMBytes/1024/1024 >= flux2Klein4BMinVRAMMB
}

func localFlux2KleinModelCacheReady() bool {
	info, err := os.Stat(filepath.Join(imageRuntimeRoot(), flux2Klein4BReadyMarker))
	return err == nil && !info.IsDir()
}

func imageRuntimeRoot() string {
	if root := strings.TrimSpace(os.Getenv("RYVION_IMAGE_RUNTIME_ROOT")); root != "" {
		return root
	}
	switch runtime.GOOS {
	case "windows":
		root := strings.TrimSpace(os.Getenv("ProgramData"))
		if root == "" {
			root = `C:\ProgramData`
		}
		return filepath.Join(root, "Ryvion", "image-runtime")
	case "linux":
		return "/var/lib/ryvion/image-runtime"
	}
	home, err := os.UserHomeDir()
	if err != nil || strings.TrimSpace(home) == "" {
		return filepath.Join(os.TempDir(), "ryvion-image-runtime")
	}
	return filepath.Join(home, ".ryvion", "image-runtime")
}

func localFlux2KleinHardwareEligible(caps hw.CapSet, gpuReady bool) bool {
	if gpuReady && caps.VRAMBytes/1024/1024 >= flux2Klein4BMinVRAMMB {
		return true
	}
	if caps.RAMBytes/1024/1024/1024 < flux2Klein4BMinRAMGB {
		return false
	}
	if caps.CPUCores < flux2Klein4BMinCPUCores {
		return false
	}
	if runtime.GOOS == "darwin" {
		return true
	}
	return caps.RAMBytes/1024/1024/1024 >= 32 && caps.CPUCores >= 8
}

func ensureUserImageRuntimeHelper() error {
	if strings.TrimSpace(os.Getenv("RYV_FLUX2_HELPER")) != "" {
		if _, ok := resolveFlux2LocalHelper(); ok {
			return nil
		}
	}
	if runtime.GOOS != "darwin" {
		return nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	path := filepath.Join(home, ".ryvion", "runtime", "helpers", "ryvion-image-runtime")
	desired := []byte(darwinImageRuntimeHelperScript())
	if current, err := os.ReadFile(path); err == nil && bytes.Equal(current, desired) {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	return os.WriteFile(path, desired, 0o700)
}

func startUserImageRuntimePrewarm(ctx context.Context, caps hw.CapSet, diskGB uint64, gpuReady bool) {
	if !publicAIOptInEnabled() {
		return
	}
	if diskGB < flux2Klein4BMinDiskGB || !localFlux2KleinFastGPUEligible(caps, gpuReady) || localFlux2KleinModelCacheReady() {
		return
	}
	helper, ok := resolveFlux2LocalHelper()
	if !ok {
		return
	}
	go func() {
		root := imageRuntimeRoot()
		if err := os.MkdirAll(root, 0o700); err != nil {
			slog.Warn("image runtime prewarm skipped; failed to create runtime root", "error", err)
			return
		}
		warming := filepath.Join(root, flux2Klein4BWarmingMarker)
		logPath := filepath.Join(root, "prewarm.log")
		_ = os.WriteFile(warming, []byte(time.Now().UTC().Format(time.RFC3339)), 0o600)
		defer os.Remove(warming)

		prewarmCtx, cancel := context.WithTimeout(ctx, 3*time.Hour)
		defer cancel()
		cmd := exec.CommandContext(prewarmCtx, helper, "--prepare", "--model", flux2Klein4BLocalModel)
		var buf bytes.Buffer
		cmd.Stdout = &buf
		cmd.Stderr = &buf
		start := time.Now()
		slog.Info("image runtime prewarm started", "model", flux2Klein4BLocalModel)
		err := cmd.Run()
		_ = os.WriteFile(logPath, []byte(tailString(buf.String(), 65536)), 0o600)
		if err != nil {
			slog.Warn("image runtime prewarm failed", "model", flux2Klein4BLocalModel, "duration", time.Since(start), "error", err)
			return
		}
		slog.Info("image runtime prewarm completed", "model", flux2Klein4BLocalModel, "duration", time.Since(start))
	}()
}

func darwinImageRuntimeHelperScript() string {
	return `#!/usr/bin/env bash
set -euo pipefail

MODEL="flux-2-klein-4b-local"
PROMPT=""
OUTPUT=""
WIDTH="1024"
HEIGHT="1024"
PREPARE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="${2:-}"; shift 2 ;;
    --prompt) PROMPT="${2:-}"; shift 2 ;;
    --output) OUTPUT="${2:-}"; shift 2 ;;
    --width) WIDTH="${2:-1024}"; shift 2 ;;
    --height) HEIGHT="${2:-1024}"; shift 2 ;;
    --prepare) PREPARE="1"; shift ;;
    *) shift ;;
  esac
done

if [[ "$PREPARE" != "1" && ( -z "$PROMPT" || -z "$OUTPUT" ) ]]; then
  echo "prompt and output are required" >&2
  exit 2
fi
if [[ "$MODEL" != "flux-2-klein-4b-local" ]]; then
  echo "unsupported model: $MODEL" >&2
  exit 2
fi

ROOT="${RYVION_IMAGE_RUNTIME_ROOT:-$HOME/.ryvion/image-runtime}"
VENV="$ROOT/venv"
BOOTSTRAP="$ROOT/bootstrap"
CACHE="$ROOT/hf-cache"
MARKER="$ROOT/.deps-flux2-klein-v2"
READY_MARKER="$ROOT/.model-flux2-klein-ready-v2"
mkdir -p "$ROOT" "$CACHE"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

python_ok() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
}

find_python() {
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      path="$(command -v "$candidate")"
      if python_ok "$path"; then
        echo "$path"
        return 0
      fi
    fi
  done
  return 1
}

ensure_uv() {
  if [[ -x "$BOOTSTRAP/bin/uv" ]]; then
    echo "$BOOTSTRAP/bin/uv"
    return 0
  fi
  if command -v uv >/dev/null 2>&1; then
    echo "$(command -v uv)"
    return 0
  fi
  bootstrap_python=""
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      bootstrap_python="$(command -v "$candidate")"
      break
    fi
  done
  if [[ -z "$bootstrap_python" ]]; then
    echo "Python is required to bootstrap the Ryvion image runtime." >&2
    exit 127
  fi
  rm -rf "$BOOTSTRAP"
  "$bootstrap_python" -m venv "$BOOTSTRAP"
  "$BOOTSTRAP/bin/python" -m pip install --upgrade pip uv
  echo "$BOOTSTRAP/bin/uv"
}

create_runtime_venv() {
  rm -rf "$VENV"
  if PYTHON="$(find_python)"; then
    "$PYTHON" -m venv "$VENV"
    return 0
  fi
  echo "runtime.image: provisioning Python 3.12 in Ryvion-managed runtime"
  UV="$(ensure_uv)"
  "$UV" python install 3.12
  "$UV" venv "$VENV" --python 3.12
}

if [[ -x "$VENV/bin/python" ]] && ! python_ok "$VENV/bin/python"; then
  echo "runtime.image: replacing Python <3.10 environment"
  rm -rf "$VENV"
  rm -f "$ROOT"/.deps-flux2-klein-*
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "runtime.image: creating Python environment"
  create_runtime_venv
fi
PY="$VENV/bin/python"
if ! python_ok "$PY"; then
  echo "runtime.image: existing Python environment is incompatible; recreating"
  create_runtime_venv
  PY="$VENV/bin/python"
fi

if [[ ! -f "$MARKER" ]]; then
  echo "runtime.image: installing FLUX.2 klein runtime dependencies"
  rm -f "$ROOT"/.deps-flux2-klein-*
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install --upgrade torch torchvision
  "$PY" -m pip install --upgrade git+https://github.com/huggingface/diffusers.git transformers accelerate safetensors pillow protobuf sentencepiece huggingface_hub
  touch "$MARKER"
fi

RUN_SCRIPT="$ROOT/run_flux2_klein.py"
cat > "$RUN_SCRIPT" <<'PY'
import os
import sys
import torch
from diffusers import Flux2KleinPipeline
from huggingface_hub import snapshot_download

model, prompt, output, width, height, cache_dir = sys.argv[1:7]
width = int(width)
height = int(height)
if model != "flux-2-klein-4b-local":
    raise SystemExit(f"unsupported model {model}")
if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32
repo_id = "black-forest-labs/FLUX.2-klein-4B"
local_dir = snapshot_download(
    repo_id,
    cache_dir=cache_dir,
    token=os.environ.get("HF_TOKEN") or None,
    resume_download=True,
)
pipe = Flux2KleinPipeline.from_pretrained(
    local_dir,
    torch_dtype=dtype,
    cache_dir=cache_dir,
)
pipe = pipe.to(device)
generator = torch.Generator(device="cpu").manual_seed(0)
image = pipe(
    prompt=prompt,
    height=height,
    width=width,
    guidance_scale=1.0,
    num_inference_steps=4 if device == "mps" else 2,
    generator=generator,
).images[0]
image.save(output)
print(f"runtime.image: wrote {output} on {device}")
PY

PREPARE_SCRIPT="$ROOT/prepare_flux2_klein.py"
cat > "$PREPARE_SCRIPT" <<'PY'
import os
import sys
from pathlib import Path
import torch
from diffusers import Flux2KleinPipeline
from huggingface_hub import snapshot_download

model, cache_dir, ready_marker = sys.argv[1:4]
if model != "flux-2-klein-4b-local":
    raise SystemExit(f"unsupported model {model}")
local_dir = snapshot_download(
    "black-forest-labs/FLUX.2-klein-4B",
    cache_dir=cache_dir,
    token=os.environ.get("HF_TOKEN") or None,
    resume_download=True,
)
if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32
pipe = Flux2KleinPipeline.from_pretrained(local_dir, torch_dtype=dtype, cache_dir=cache_dir)
pipe = pipe.to(device)
probe_path = str(Path(ready_marker).with_name("readiness_probe.png"))
generator = torch.Generator(device="cpu").manual_seed(1)
image = pipe(
    prompt="ryvion runtime readiness probe",
    height=256,
    width=256,
    guidance_scale=1.0,
    num_inference_steps=1,
    generator=generator,
).images[0]
image.save(probe_path)
Path(ready_marker).write_text(f"{local_dir}\nprobe={probe_path}\n", encoding="utf-8")
print(f"runtime.image: smoke-tested model cache at {local_dir}")
PY

if [[ "$PREPARE" == "1" ]]; then
  "$PY" "$PREPARE_SCRIPT" "$MODEL" "$CACHE" "$READY_MARKER"
  exit 0
fi

if [[ ! -f "$READY_MARKER" ]]; then
  echo "runtime.image: model cache not ready; preparing now"
  "$PY" "$PREPARE_SCRIPT" "$MODEL" "$CACHE" "$READY_MARKER"
fi

"$PY" "$RUN_SCRIPT" "$MODEL" "$PROMPT" "$OUTPUT" "$WIDTH" "$HEIGHT" "$CACHE"
`
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
