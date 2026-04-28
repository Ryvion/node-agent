package main

import (
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
