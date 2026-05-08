package modelpolicy

import (
	"strings"
	"testing"

	v7hardware "github.com/Ryvion/node-agent/internal/v7/hardware"
)

func TestBuildDerivedPolicyBlocksPhiOnAppleMac(t *testing.T) {
	t.Parallel()

	policy := BuildDerivedPolicy(DerivedPolicyInput{
		BasePolicy: Policy{
			CacheDir:            "/models",
			MaxSingleModelBytes: 16 * bytesPerGiB,
			MaxCacheBytes:       64 * bytesPerGiB,
			AllowedFamilies:     []string{"llama", "phi"},
			AllowedFormats:      []string{"gguf"},
			RuntimePolicy: RuntimePolicy{
				AllowRuntimeExecution:            true,
				MaxRuntimeModelBytes:             8 * bytesPerGiB,
				MaxRuntimeParameterCountBillions: 8,
				AllowCPUOffload:                  true,
				AllowFamilies:                    []string{"llama", "phi"},
			},
		},
		Hardware: v7hardware.NormalizeInventory(v7hardware.CapacityInventory{
			OS:              "darwin",
			Arch:            "arm64",
			CPULogicalCores: 10,
			SystemRAMBytes:  32 * bytesPerGiB,
			GPUDetected:     true,
			GPUVendor:       v7hardware.GPUVendorApple,
			GPUName:         "Apple M4 Pro",
			UnifiedMemory:   true,
			MetalAvailable:  true,
		}),
	})

	if got := strings.Join(policy.RuntimePolicy.DenyFamilies, ","); !strings.Contains(got, "phi") {
		t.Fatalf("deny_families = %q, want phi", got)
	}
	decision := EvaluateRuntimeRequest(policy, RuntimeRequest{
		ModelID:        "phi-4-Q5_K_M.gguf",
		ModelSizeBytes: 5 * bytesPerGiB,
		Family:         "phi",
	})
	if decision.Allowed || decision.Reason != RuntimeDecisionFamilyDenied {
		t.Fatalf("decision = %+v, want Mac Phi blocked by derived policy", decision)
	}
}

func TestBuildDerivedPolicyBoundsRuntimeBytesByVRAM(t *testing.T) {
	t.Parallel()

	policy := BuildDerivedPolicy(DerivedPolicyInput{
		BasePolicy: Policy{
			CacheDir:            "/models",
			MaxSingleModelBytes: 64 * bytesPerGiB,
			MaxCacheBytes:       128 * bytesPerGiB,
			AllowedFamilies:     []string{"llama"},
			AllowedFormats:      []string{"gguf"},
			RuntimePolicy: RuntimePolicy{
				AllowRuntimeExecution:            true,
				MaxRuntimeModelBytes:             64 * bytesPerGiB,
				MaxRuntimeParameterCountBillions: 70,
				AllowCPUOffload:                  true,
				AllowFamilies:                    []string{"llama"},
			},
		},
		Hardware: v7hardware.NormalizeInventory(v7hardware.CapacityInventory{
			OS:              "windows",
			Arch:            "amd64",
			CPULogicalCores: 24,
			SystemRAMBytes:  64 * bytesPerGiB,
			GPUDetected:     true,
			GPUVendor:       v7hardware.GPUVendorNVIDIA,
			GPUName:         "NVIDIA GeForce RTX 4090",
			GPUVRAMBytes:    24 * bytesPerGiB,
			CUDAAvailable:   true,
		}),
	})

	if policy.RuntimePolicy.MaxRuntimeModelBytes != 18*bytesPerGiB {
		t.Fatalf("max_runtime_model_bytes = %d, want 75%% of 24GiB VRAM", policy.RuntimePolicy.MaxRuntimeModelBytes)
	}
	if policy.RuntimePolicy.MaxConcurrentInferenceJobs < 2 || policy.RuntimePolicy.MaxWarmModels < 2 {
		t.Fatalf("runtime concurrency/warm = %+v, want larger RTX defaults", policy.RuntimePolicy)
	}
}

func TestBuildDerivedPolicyAllowsPhiOnCapableWindowsRTX(t *testing.T) {
	t.Parallel()

	policy := BuildDerivedPolicy(DerivedPolicyInput{
		BasePolicy: FromConfigSource(ConfigSource{
			Getenv: func(string) string { return "" },
			UserHomeDir: func() (string, error) {
				return `C:\Users\operator`, nil
			},
			GOOS: "windows",
		}),
		Hardware: v7hardware.NormalizeInventory(v7hardware.CapacityInventory{
			OS:                "windows",
			Arch:              "amd64",
			CPULogicalCores:   24,
			SystemRAMBytes:    64 * bytesPerGiB,
			AvailableRAMBytes: 48 * bytesPerGiB,
			GPUDetected:       true,
			GPUVendor:         v7hardware.GPUVendorNVIDIA,
			GPUName:           "NVIDIA GeForce RTX 4090",
			GPUVRAMBytes:      24 * bytesPerGiB,
			CUDAAvailable:     true,
		}),
	})

	if got := strings.Join(policy.RuntimePolicy.AllowFamilies, ","); !strings.Contains(got, "phi") {
		t.Fatalf("runtime allow_families = %q, want phi derived from capable Windows hardware", got)
	}
	if policy.RuntimePolicy.MaxRuntimeModelBytes != 18*bytesPerGiB ||
		policy.RuntimePolicy.MaxRuntimeParameterCountBillions < 14 ||
		!policy.RuntimePolicy.AllowLargeModels {
		t.Fatalf("runtime policy = %+v, want Windows RTX large-model policy derived from capacity", policy.RuntimePolicy)
	}
	decision := EvaluateRuntimeRequest(policy, RuntimeRequest{
		ModelID:                "phi-4-Q5_K_M.gguf",
		ModelSizeBytes:         10 * bytesPerGiB,
		ParameterCountBillions: 14,
		Family:                 "phi",
	})
	if !decision.Allowed {
		t.Fatalf("decision = %+v, want Phi runnable on capable Windows RTX policy", decision)
	}
}

func TestBuildDerivedPolicyKeepsWindowsUnknownHardwareConservative(t *testing.T) {
	t.Parallel()

	policy := BuildDerivedPolicy(DerivedPolicyInput{
		BasePolicy: FromConfigSource(ConfigSource{
			Getenv: func(string) string { return "" },
			UserHomeDir: func() (string, error) {
				return `C:\Users\operator`, nil
			},
			GOOS: "windows",
		}),
		Hardware: v7hardware.NormalizeInventory(v7hardware.CapacityInventory{
			OS:   "windows",
			Arch: "amd64",
		}),
	})

	decision := EvaluateRuntimeRequest(policy, RuntimeRequest{
		ModelID:                "phi-4-Q5_K_M.gguf",
		ModelSizeBytes:         10 * bytesPerGiB,
		ParameterCountBillions: 14,
		Family:                 "phi",
	})
	if decision.Allowed {
		t.Fatalf("decision = %+v, want unknown Windows hardware to stay conservative for Phi", decision)
	}
}
