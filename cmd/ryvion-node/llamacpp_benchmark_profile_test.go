package main

import (
	"testing"

	capshardware "github.com/Ryvion/ryvion-node/internal/capabilities/hardware"
	llamacpp "github.com/Ryvion/ryvion-node/internal/runtimes/llamacpp"
)

func TestLlamaCppBenchmarkAccelerationModeReportsMetal(t *testing.T) {
	cfg := llamacpp.LlamaCppSidecarConfig{GPULayers: 99}
	hardware := capshardware.CapacityInventory{
		GPUDetected:    true,
		GPUVendor:      capshardware.GPUVendorApple,
		GPUName:        "Apple M4",
		MetalAvailable: true,
	}

	if got := llamaCppBenchmarkAccelerationMode(cfg, hardware); got != "metal" {
		t.Fatalf("acceleration = %q, want metal", got)
	}
}

func TestLlamaCppBenchmarkAccelerationModeHonorsExplicitCPU(t *testing.T) {
	cfg := llamacpp.LlamaCppSidecarConfig{GPULayers: 0}
	hardware := capshardware.CapacityInventory{
		GPUDetected:    true,
		GPUVendor:      capshardware.GPUVendorApple,
		GPUName:        "Apple M4",
		MetalAvailable: true,
	}

	if got := llamaCppBenchmarkAccelerationMode(cfg, hardware); got != "cpu" {
		t.Fatalf("acceleration = %q, want cpu", got)
	}
}
