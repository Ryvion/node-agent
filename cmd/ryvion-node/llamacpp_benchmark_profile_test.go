package main

import (
	"testing"

	v7hardware "github.com/Ryvion/node-agent/internal/v7/hardware"
	v7llamacpp "github.com/Ryvion/node-agent/internal/v7/llamacpp"
)

func TestLlamaCppBenchmarkAccelerationModeReportsMetal(t *testing.T) {
	cfg := v7llamacpp.LlamaCppSidecarConfig{GPULayers: 99}
	hardware := v7hardware.CapacityInventory{
		GPUDetected:    true,
		GPUVendor:      v7hardware.GPUVendorApple,
		GPUName:        "Apple M4",
		MetalAvailable: true,
	}

	if got := llamaCppBenchmarkAccelerationMode(cfg, hardware); got != "metal" {
		t.Fatalf("acceleration = %q, want metal", got)
	}
}

func TestLlamaCppBenchmarkAccelerationModeHonorsExplicitCPU(t *testing.T) {
	cfg := v7llamacpp.LlamaCppSidecarConfig{GPULayers: 0}
	hardware := v7hardware.CapacityInventory{
		GPUDetected:    true,
		GPUVendor:      v7hardware.GPUVendorApple,
		GPUName:        "Apple M4",
		MetalAvailable: true,
	}

	if got := llamaCppBenchmarkAccelerationMode(cfg, hardware); got != "cpu" {
		t.Fatalf("acceleration = %q, want cpu", got)
	}
}
