package main

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/Ryvion/node-agent/internal/hub"
	"github.com/Ryvion/node-agent/internal/hw"
)

func TestExecutorKindForRyvionRuntimeAssignment(t *testing.T) {
	work := &hub.WorkAssignment{
		Image:    "ryvion-runtime",
		SpecJSON: `{"executor_kind":"ryvion_runtime","runtime_task":"image_generation","model":"flux-2-klein-4b-local"}`,
	}
	if got := executorKindForAssignment(work); got != executorKindRyvionRuntime {
		t.Fatalf("expected %s, got %s", executorKindRyvionRuntime, got)
	}
	if got := selectExecutionEngine(work).Kind(); got != executorKindRyvionRuntime {
		t.Fatalf("expected runtime engine, got %s", got)
	}
}

func TestBuildHealthReportAdvertisesLocalFluxOnlyWhenRuntimeReady(t *testing.T) {
	t.Setenv("RYV_PUBLIC_AI", "1")
	t.Setenv("RYV_DISABLE_OCI", "1")
	t.Setenv("RYV_ENABLE_RUNTIME_FIXTURES", "1")
	if disk := detectAvailableDiskGB(); disk < flux2Klein4BMinDiskGB {
		t.Skipf("host test disk below local FLUX readiness threshold: %dGB", disk)
	}

	caps := hw.CapSet{
		GPUModel:  "RTX 4070 Ti Super",
		VRAMBytes: 16 * 1024 * 1024 * 1024,
		CPUCores:  16,
		RAMBytes:  32 * 1024 * 1024 * 1024,
	}
	report := buildHealthReport(caps, nil, newRuntimeManager("test", runtimeContractMetadata{}))
	for _, want := range []string{
		"cap:ryvion_runtime:1",
		"cap:image_gen:1",
		"runtime:image:flux-2-klein-4b-local",
		"model:flux-2-klein-4b-local",
	} {
		if !strings.Contains(report.Message, want) {
			t.Fatalf("expected health report to contain %q, got %s", want, report.Message)
		}
	}
}

func TestBuildHealthReportDoesNotAdvertiseLocalFluxOnSmallGPU(t *testing.T) {
	t.Setenv("RYV_PUBLIC_AI", "1")
	t.Setenv("RYV_DISABLE_OCI", "1")
	t.Setenv("RYV_ENABLE_RUNTIME_FIXTURES", "1")

	caps := hw.CapSet{
		GPUModel:  "RTX 3060",
		VRAMBytes: 8 * 1024 * 1024 * 1024,
		CPUCores:  8,
	}
	report := buildHealthReport(caps, nil, newRuntimeManager("test", runtimeContractMetadata{}))
	if strings.Contains(report.Message, "runtime:image:flux-2-klein-4b-local") {
		t.Fatalf("small GPU should not advertise local FLUX, got %s", report.Message)
	}
}

func TestBuildHealthReportDoesNotAdvertiseLocalFluxUntilCacheReady(t *testing.T) {
	t.Setenv("RYV_PUBLIC_AI", "1")
	t.Setenv("RYV_DISABLE_OCI", "1")
	t.Setenv("RYV_ENABLE_RUNTIME_FIXTURES", "0")
	root := t.TempDir()
	t.Setenv("RYVION_IMAGE_RUNTIME_ROOT", root)
	helper := filepath.Join(root, "ryvion-image-runtime")
	if err := os.WriteFile(helper, []byte("#!/bin/sh\nexit 0\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("RYV_FLUX2_HELPER", helper)
	if disk := detectAvailableDiskGB(); disk < flux2Klein4BMinDiskGB {
		t.Skipf("host test disk below local FLUX readiness threshold: %dGB", disk)
	}

	caps := hw.CapSet{
		GPUModel:  "RTX 4070 Ti Super",
		VRAMBytes: 16 * 1024 * 1024 * 1024,
		CPUCores:  16,
		RAMBytes:  32 * 1024 * 1024 * 1024,
	}
	report := buildHealthReport(caps, nil, newRuntimeManager("test", runtimeContractMetadata{}))
	if strings.Contains(report.Message, "runtime:image:flux-2-klein-4b-local,") ||
		strings.HasSuffix(report.Message, "runtime:image:flux-2-klein-4b-local") {
		t.Fatalf("unprepared model cache should not advertise ready local FLUX, got %s", report.Message)
	}
	if !strings.Contains(report.Message, "runtime:image:flux-2-klein-4b-local:eligible:1") {
		t.Fatalf("unprepared fast GPU should advertise prepare eligibility, got %s", report.Message)
	}
	if !strings.Contains(report.Message, "cap:ryvion_runtime:0") {
		t.Fatalf("unprepared model cache should disable ryvion runtime, got %s", report.Message)
	}

	if err := os.WriteFile(filepath.Join(root, flux2Klein4BWarmingMarker), []byte("warming"), 0o600); err != nil {
		t.Fatal(err)
	}
	report = buildHealthReport(caps, nil, newRuntimeManager("test", runtimeContractMetadata{}))
	if !strings.Contains(report.Message, "runtime:image:flux-2-klein-4b-local:preparing:1") {
		t.Fatalf("warming model cache should report preparing token, got %s", report.Message)
	}

	if err := os.WriteFile(filepath.Join(root, flux2Klein4BReadyMarker), []byte("ready"), 0o600); err != nil {
		t.Fatal(err)
	}
	report = buildHealthReport(caps, nil, newRuntimeManager("test", runtimeContractMetadata{}))
	for _, want := range []string{
		"cap:ryvion_runtime:1",
		"runtime:image:flux-2-klein-4b-local",
		"model:flux-2-klein-4b-local",
	} {
		if !strings.Contains(report.Message, want) {
			t.Fatalf("prepared model cache should contain %q, got %s", want, report.Message)
		}
	}
}

func TestBuildHealthReportAdvertisesPreparedHighMemoryCPUFluxPreview(t *testing.T) {
	t.Setenv("RYV_PUBLIC_AI", "1")
	t.Setenv("RYV_DISABLE_OCI", "1")
	t.Setenv("RYV_ENABLE_RUNTIME_FIXTURES", "1")
	if disk := detectAvailableDiskGB(); disk < flux2Klein4BMinDiskGB {
		t.Skipf("host test disk below local FLUX readiness threshold: %dGB", disk)
	}
	if runtimeGOOS() == "darwin" {
		t.Skip("darwin path has a lower memory threshold; covered by production helper bootstrap")
	}

	caps := hw.CapSet{
		CPUCores: 8,
		RAMBytes: 32 * 1024 * 1024 * 1024,
	}
	report := buildHealthReport(caps, nil, newRuntimeManager("test", runtimeContractMetadata{}))
	for _, want := range []string{
		"cap:ryvion_runtime:1",
		"cap:image_gen:1",
		"runtime:image:flux-2-klein-4b-local",
		"model:flux-2-klein-4b-local",
		"runtime:image:flux-2-klein-4b-local:mode:cpu-preview",
	} {
		if !strings.Contains(report.Message, want) {
			t.Fatalf("expected health report to contain %q, got %s", want, report.Message)
		}
	}
	if strings.Contains(report.Message, "runtime:image:flux-2-klein-4b-local:min_vram_mb") {
		t.Fatalf("CPU preview node must not advertise VRAM-gated FLUX readiness, got %s", report.Message)
	}
}

func TestRunFlux2LocalPrepareHelperInvokesPrepare(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell helper fixture is POSIX-only")
	}
	root := t.TempDir()
	t.Setenv("RYVION_IMAGE_RUNTIME_ROOT", root)
	helper := filepath.Join(root, "ryvion-image-runtime")
	script := `#!/usr/bin/env sh
set -eu
prepare=0
model=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --prepare) prepare=1; shift ;;
    --model) model="$2"; shift 2 ;;
    *) shift ;;
  esac
done
if [ "$prepare" != "1" ]; then
  echo "missing prepare flag" >&2
  exit 2
fi
if [ "$model" != "flux-2-klein-4b-local" ]; then
  echo "bad model: $model" >&2
  exit 2
fi
printf "ready\n" > "$RYVION_IMAGE_RUNTIME_ROOT/.model-flux2-klein-ready-v2"
echo "prepared $model"
`
	if err := os.WriteFile(helper, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	logs, err := runFlux2LocalPrepareHelper(context.Background(), helper, flux2Klein4BLocalModel)
	if err != nil {
		t.Fatalf("prepare helper failed: %v logs=%s", err, logs)
	}
	if !strings.Contains(logs, "prepared flux-2-klein-4b-local") {
		t.Fatalf("unexpected logs: %s", logs)
	}
	if !localFlux2KleinModelCacheReady() {
		t.Fatal("prepare helper fixture should create ready marker")
	}
	if hashRuntimePrepareResult(flux2Klein4BLocalModel, logs) == "" {
		t.Fatal("prepare hash should not be empty")
	}
}

func runtimeGOOS() string {
	return strings.ToLower(strings.TrimSpace(runtime.GOOS))
}
