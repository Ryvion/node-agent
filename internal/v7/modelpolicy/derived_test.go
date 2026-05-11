package modelpolicy

import (
	"strings"
	"testing"

	v7hardware "github.com/Ryvion/node-agent/internal/v7/hardware"
)

func TestBuildDerivedPolicyBlocksPhiOnConstrainedAppleMac(t *testing.T) {
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
			CPULogicalCores: 8,
			SystemRAMBytes:  16 * bytesPerGiB,
			GPUDetected:     true,
			GPUVendor:       v7hardware.GPUVendorApple,
			GPUName:         "Apple M4",
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

func TestBuildDerivedPolicyAllowsPhiOnLargeAppleUnifiedMemory(t *testing.T) {
	t.Parallel()

	policy := BuildDerivedPolicy(DerivedPolicyInput{
		BasePolicy: FromConfigSource(ConfigSource{
			Getenv: func(string) string { return "" },
			UserHomeDir: func() (string, error) {
				return "/Users/operator", nil
			},
			GOOS: "darwin",
		}),
		Hardware: v7hardware.NormalizeInventory(v7hardware.CapacityInventory{
			OS:                "darwin",
			Arch:              "arm64",
			CPULogicalCores:   14,
			SystemRAMBytes:    48 * bytesPerGiB,
			AvailableRAMBytes: 36 * bytesPerGiB,
			GPUDetected:       true,
			GPUVendor:         v7hardware.GPUVendorApple,
			GPUName:           "Apple M4 Max",
			UnifiedMemory:     true,
			MetalAvailable:    true,
		}),
	})

	if got := strings.Join(policy.RuntimePolicy.DenyFamilies, ","); strings.Contains(got, "phi") {
		t.Fatalf("deny_families = %q, want phi allowed on large Apple unified memory", got)
	}
	if got := strings.Join(policy.RuntimePolicy.AllowFamilies, ","); !strings.Contains(got, "phi") {
		t.Fatalf("runtime allow_families = %q, want phi derived from large Apple unified memory", got)
	}
	decision := EvaluateRuntimeRequest(policy, RuntimeRequest{
		ModelID:                "phi-4-Q4_K_M.gguf",
		ModelSizeBytes:         10 * bytesPerGiB,
		ParameterCountBillions: 14,
		Family:                 "phi",
	})
	if !decision.Allowed {
		t.Fatalf("decision = %+v, want Phi runnable on large Apple unified memory", decision)
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

	if want := (64 * bytesPerGiB * 3) / 5; policy.RuntimePolicy.MaxRuntimeModelBytes != want {
		t.Fatalf("max_runtime_model_bytes = %d, want CPU-offload cap %d", policy.RuntimePolicy.MaxRuntimeModelBytes, want)
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
	if want := (64 * bytesPerGiB * 3) / 5; policy.RuntimePolicy.MaxRuntimeModelBytes != want ||
		policy.RuntimePolicy.MaxRuntimeParameterCountBillions < 16 ||
		!policy.RuntimePolicy.AllowLargeModels {
		t.Fatalf("runtime policy = %+v, want Windows RTX large-model policy derived from capacity %d", policy.RuntimePolicy, want)
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

func TestBuildDerivedPolicyAllowsPhiOnMidrangeAcceleratedGPU(t *testing.T) {
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
			CPULogicalCores:   16,
			SystemRAMBytes:    34 * bytesPerGiB,
			AvailableRAMBytes: 20 * bytesPerGiB,
			GPUDetected:       true,
			GPUVendor:         v7hardware.GPUVendorAMD,
			GPUName:           "AMD Radeon RX 7900 XTX",
			GPUVRAMBytes:      16 * bytesPerGiB,
			VulkanAvailable:   true,
		}),
	})

	if got := strings.Join(policy.RuntimePolicy.AllowFamilies, ","); !strings.Contains(got, "phi") {
		t.Fatalf("runtime allow_families = %q, want Phi auto-allowed on accelerated GPU", got)
	}
	decision := EvaluateRuntimeRequest(policy, RuntimeRequest{
		ModelID:                "phi-4-Q4_K_M.gguf",
		ModelSizeBytes:         9 * bytesPerGiB,
		ParameterCountBillions: 15,
		Family:                 "phi",
	})
	if !decision.Allowed {
		t.Fatalf("decision = %+v, want resident Phi runnable on midrange accelerated GPU policy", decision)
	}
}

func TestBuildDerivedPolicyAllowsPhiOnWindowsThirtyTwoGBClassRTX(t *testing.T) {
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
			CPULogicalCores:   16,
			SystemRAMBytes:    33934004224,
			AvailableRAMBytes: 19450638336,
			GPUDetected:       true,
			GPUVendor:         v7hardware.GPUVendorNVIDIA,
			GPUName:           "NVIDIA GeForce RTX 4070 Ti SUPER",
			GPUVRAMBytes:      17171480576,
			CUDAAvailable:     true,
		}),
	})

	decision := EvaluateRuntimeRequest(policy, RuntimeRequest{
		ModelID:                "phi-4-Q4_K_M.gguf",
		ModelSizeBytes:         9053114816,
		ParameterCountBillions: 15,
		Family:                 "phi",
	})
	if !decision.Allowed {
		t.Fatalf("decision = %+v policy = %+v, want live RTX 4070 Ti SUPER node to allow Phi", decision, policy.RuntimePolicy)
	}
}

func TestBuildDerivedPolicyAllowsResidentGemmaOnWindowsSixteenGBClassRTX(t *testing.T) {
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
			CPULogicalCores:   16,
			SystemRAMBytes:    33934004224,
			AvailableRAMBytes: 19450638336,
			GPUDetected:       true,
			GPUVendor:         v7hardware.GPUVendorNVIDIA,
			GPUName:           "NVIDIA GeForce RTX 4070 Ti SUPER",
			GPUVRAMBytes:      17171480576,
			CUDAAvailable:     true,
			VulkanAvailable:   true,
		}),
	})

	if policy.RuntimePolicy.MaxRuntimeModelBytes < 18*bytesPerGiB ||
		policy.MaxSingleModelBytes < 18*bytesPerGiB ||
		policy.RuntimePolicy.MaxRuntimeParameterCountBillions < 32 ||
		!policy.RuntimePolicy.AllowLargeModels ||
		!policy.AllowLicenseRestricted {
		t.Fatalf("policy = %+v, runtime = %+v, want 16GB-class RTX resident Gemma policy", policy, policy.RuntimePolicy)
	}
	decision := EvaluateRuntimeRequest(policy, RuntimeRequest{
		ModelID:                "gemma-3-27b-it-Q4_K_M.gguf",
		ModelSizeBytes:         18 * bytesPerGiB,
		ParameterCountBillions: 27,
		Family:                 "gemma",
		CPUOffload:             true,
	})
	if !decision.Allowed {
		t.Fatalf("decision = %+v policy = %+v, want resident Gemma runnable on 16GB-class RTX with CPU offload", decision, policy.RuntimePolicy)
	}
}

func TestBuildDerivedPolicyAllowsPhiOnCapableLinuxGPU(t *testing.T) {
	t.Parallel()

	policy := BuildDerivedPolicy(DerivedPolicyInput{
		BasePolicy: FromConfigSource(ConfigSource{
			Getenv: func(string) string { return "" },
			UserHomeDir: func() (string, error) {
				return "/home/operator", nil
			},
			GOOS: "linux",
		}),
		Hardware: v7hardware.NormalizeInventory(v7hardware.CapacityInventory{
			OS:                "linux",
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

	if got := strings.Join(policy.RuntimePolicy.AllowFamilies, ","); !strings.Contains(got, "phi") || !strings.Contains(got, "gemma") {
		t.Fatalf("runtime allow_families = %q, want accelerated Linux policy to inherit catalog families", got)
	}
	if want := (64 * bytesPerGiB * 3) / 5; policy.RuntimePolicy.MaxRuntimeModelBytes != want ||
		policy.RuntimePolicy.MaxRuntimeParameterCountBillions < 32 ||
		!policy.RuntimePolicy.AllowLargeModels {
		t.Fatalf("runtime policy = %+v, want Linux RTX large-model policy derived from capacity %d", policy.RuntimePolicy, want)
	}
	decision := EvaluateRuntimeRequest(policy, RuntimeRequest{
		ModelID:                "phi-4-Q4_K_M.gguf",
		ModelSizeBytes:         10 * bytesPerGiB,
		ParameterCountBillions: 14,
		Family:                 "phi",
	})
	if !decision.Allowed {
		t.Fatalf("decision = %+v, want Phi runnable on capable Linux GPU policy", decision)
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
