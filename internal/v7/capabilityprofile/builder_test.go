package capabilityprofile

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/Ryvion/ryvion-node/internal/models/cache"
	"github.com/Ryvion/ryvion-node/internal/models/policy"
	"github.com/Ryvion/ryvion-node/internal/runtimes/llamacpp"
	"github.com/Ryvion/ryvion-node/internal/v7/backendprobe"
	v7hardware "github.com/Ryvion/ryvion-node/internal/v7/hardware"
	"github.com/Ryvion/ryvion-node/internal/v7/inferenceconfig"
	"github.com/Ryvion/ryvion-node/internal/v7/tensoraccess"
)

const gib = uint64(1024 * 1024 * 1024)

func TestBuildProfileFreshMacLlamaRunnablePhiResidentBlocked(t *testing.T) {
	t.Parallel()

	policy := modelpolicy.BuildDerivedPolicy(modelpolicy.DerivedPolicyInput{
		BasePolicy: modelpolicy.Policy{
			AutoDownload:        false,
			MaxSingleModelBytes: 8 * gib,
			MaxCacheBytes:       50 * gib,
			CacheDir:            "/Users/operator/.ryvion/models",
			AllowedFamilies:     []string{"llama", "phi", "qwen", "gemma"},
			AllowedFormats:      []string{"gguf"},
			RuntimePolicy: modelpolicy.RuntimePolicy{
				AllowRuntimeExecution:              true,
				MaxRuntimeModelBytes:               8 * gib,
				MaxRuntimeParameterCountBillions:   8,
				AllowCPUOffload:                    true,
				AllowFamilies:                      []string{"llama", "phi"},
				RequireExplicitAllowForLargeModels: true,
			},
		},
		Hardware: macHardware(),
	})
	profile := BuildProfile(BuildInput{
		Hardware:        macHardware(),
		Policy:          policy,
		ModelCache:      macModelCache(),
		BackendProbes:   llamaProbe(true),
		BackendRuntimes: llamacpp.NormalizeBackendRuntimes(llamacpp.BackendRuntimes{}),
		TensorAccess:    tensoraccess.NewNoopProvider(tensoraccess.NoopProviderConfig{}).Capability(nil),
	})

	if !profile.V7DashboardInference || !profile.TextOutput || !profile.Streaming || !profile.HashMetricsReceipts || !profile.BackendTextGeneration || !profile.BackendWarm || !profile.WarmBackend || !profile.Ready {
		t.Fatalf("profile = %+v, want default inference capability on fresh Mac with llama.cpp and cached model", profile)
	}
	if profile.SpeculativeDecoding == nil ||
		!profile.SpeculativeDecoding.Supported ||
		!profile.SpeculativeDecoding.Enabled ||
		profile.SpeculativeDecoding.DefaultMethod != "ngram" {
		t.Fatalf("speculative_decoding = %+v, want enabled ngram support for Mac llama.cpp", profile.SpeculativeDecoding)
	}
	llama := modelByID(t, profile.Models, "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
	if !llama.Resident || !llama.Runnable {
		t.Fatalf("llama model capability = %+v, want resident runnable", llama)
	}
	phi := modelByID(t, profile.Models, "phi-4-Q5_K_M.gguf")
	if !phi.Resident || phi.Runnable || phi.Reason != modelpolicy.RuntimeDecisionFamilyDenied {
		t.Fatalf("phi model capability = %+v, want resident but Mac policy blocked", phi)
	}
	if got := strings.Join(profile.Policy.DeniedFamilies, ","); !strings.Contains(got, "phi") {
		t.Fatalf("denied families = %q, want Mac policy to deny phi", got)
	}
	assertProfileJSONSafe(t, profile)
}

func TestBuildProfileWindowsRTXIncludesHardwareBackendAndInventory(t *testing.T) {
	t.Parallel()

	hardware := v7hardware.NormalizeInventory(v7hardware.CapacityInventory{
		OS:                "windows",
		Arch:              "amd64",
		CPULogicalCores:   24,
		SystemRAMBytes:    64 * gib,
		AvailableRAMBytes: 48 * gib,
		GPUDetected:       true,
		GPUVendor:         v7hardware.GPUVendorNVIDIA,
		GPUName:           "NVIDIA GeForce RTX 4090",
		GPUVRAMBytes:      24 * gib,
		CUDAAvailable:     true,
		VulkanAvailable:   true,
	})
	policy := modelpolicy.BuildDerivedPolicy(modelpolicy.DerivedPolicyInput{
		BasePolicy: modelpolicy.FromConfigSource(modelpolicy.ConfigSource{Getenv: func(string) string { return "" }, UserHomeDir: func() (string, error) { return `C:\Users\operator`, nil }, GOOS: "windows"}),
		Hardware:   hardware,
	})
	profile := BuildProfile(BuildInput{
		Hardware:        hardware,
		Policy:          policy,
		ModelCache:      windowsModelCacheWithPhi(),
		BackendProbes:   llamaProbe(true),
		BackendRuntimes: llamacpp.NormalizeBackendRuntimes(llamacpp.BackendRuntimes{}),
		TensorAccess:    tensoraccess.NewNoopProvider(tensoraccess.NoopProviderConfig{}).Capability(nil),
	})

	if profile.Hardware.OS != "windows" || profile.Hardware.GPUVendor != "nvidia" || profile.Hardware.GPUVRAMBytes != 24*gib || !profile.Hardware.CUDAAvailable || !profile.Hardware.VulkanAvailable {
		t.Fatalf("hardware = %+v, want Windows RTX CUDA/Vulkan inventory", profile.Hardware)
	}
	if !profile.BackendRuntime.SupportsTextGeneration || !profile.BackendRuntime.SupportsStreaming || len(profile.Models) != 2 {
		t.Fatalf("profile = %+v, want backend and runnable GGUF inventory", profile)
	}
	llama := modelByID(t, profile.Models, "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
	phi := modelByID(t, profile.Models, "phi-4-Q5_K_M.gguf")
	if !llama.Runnable || !phi.Runnable || len(phi.BlockedReasons) != 0 {
		t.Fatalf("windows model capabilities llama=%+v phi=%+v, want both runnable from derived RTX policy", llama, phi)
	}
	assertProfileJSONSafe(t, profile)
}

func TestBuildProfileTextOutputOptOutKeepsHashMetricsCapability(t *testing.T) {
	t.Parallel()

	profile := BuildProfile(BuildInput{
		Hardware:        macHardware(),
		Policy:          modelpolicy.FromConfigSource(modelpolicy.ConfigSource{Getenv: func(string) string { return "" }, UserHomeDir: func() (string, error) { return "/Users/operator", nil }, GOOS: "darwin"}),
		ModelCache:      modelcache.Status{CacheDir: "/models", Models: []modelcache.Model{llamaModel("/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")}},
		BackendProbes:   llamaProbe(true),
		BackendRuntimes: llamacpp.NormalizeBackendRuntimes(llamacpp.BackendRuntimes{}),
		TensorAccess:    tensoraccess.NewNoopProvider(tensoraccess.NoopProviderConfig{}).Capability(nil),
		Getenv: func(key string) string {
			if key == inferenceconfig.EnvDisableTextOutput {
				return "1"
			}
			return ""
		},
	})

	if profile.TextOutput || profile.Streaming {
		t.Fatalf("text/streaming = %t/%t, want disabled by text opt-out", profile.TextOutput, profile.Streaming)
	}
	if !profile.HashMetricsReceipts || !profile.V7DashboardInference {
		t.Fatalf("profile = %+v, want hash-only dashboard inference still capable", profile)
	}
}

func TestBuildProfileBackendUnavailableDoesNotClaimRunCapability(t *testing.T) {
	t.Parallel()

	profile := BuildProfile(BuildInput{
		Hardware:        macHardware(),
		Policy:          modelpolicy.FromConfigSource(modelpolicy.ConfigSource{Getenv: func(string) string { return "" }, UserHomeDir: func() (string, error) { return "/Users/operator", nil }, GOOS: "darwin"}),
		ModelCache:      modelcache.Status{CacheDir: "/models", Models: []modelcache.Model{llamaModel("/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")}},
		BackendProbes:   llamaProbe(false),
		BackendRuntimes: llamacpp.NormalizeBackendRuntimes(llamacpp.BackendRuntimes{}),
		TensorAccess:    tensoraccess.NewNoopProvider(tensoraccess.NoopProviderConfig{}).Capability(nil),
	})

	if profile.TextOutput || profile.Streaming || profile.Ready || profile.V7DashboardInference || profile.BackendTextGeneration {
		t.Fatalf("profile = %+v, want no run capability without backend", profile)
	}
	model := modelByID(t, profile.Models, "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
	if model.Runnable || model.Reason != "backend_text_generation_unavailable" {
		t.Fatalf("model = %+v, want backend unavailable reason", model)
	}
}

func TestBuildProfileMissingHardwareDoesNotClaimCapability(t *testing.T) {
	t.Parallel()

	profile := BuildProfile(BuildInput{
		Hardware:        v7hardware.NormalizeInventory(v7hardware.CapacityInventory{OS: "darwin", Arch: "arm64"}),
		Policy:          modelpolicy.FromConfigSource(modelpolicy.ConfigSource{Getenv: func(string) string { return "" }, UserHomeDir: func() (string, error) { return "/Users/operator", nil }, GOOS: "darwin"}),
		ModelCache:      modelcache.Status{CacheDir: "/models", Models: []modelcache.Model{llamaModel("/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")}},
		BackendProbes:   llamaProbe(true),
		BackendRuntimes: llamacpp.NormalizeBackendRuntimes(llamacpp.BackendRuntimes{}),
		TensorAccess:    tensoraccess.NewNoopProvider(tensoraccess.NoopProviderConfig{}).Capability(nil),
	})

	if profile.Ready || profile.V7DashboardInference || profile.TextOutput || profile.HashMetricsReceipts {
		t.Fatalf("profile = %+v, want missing hardware to block capability", profile)
	}
	if profile.Reason != "hardware_capacity_missing" {
		t.Fatalf("reason = %q, want hardware_capacity_missing", profile.Reason)
	}
}

func macHardware() v7hardware.CapacityInventory {
	return v7hardware.NormalizeInventory(v7hardware.CapacityInventory{
		OS:                "darwin",
		Arch:              "arm64",
		CPULogicalCores:   10,
		SystemRAMBytes:    16 * gib,
		AvailableRAMBytes: 10 * gib,
		GPUDetected:       true,
		GPUVendor:         v7hardware.GPUVendorApple,
		GPUName:           "Apple M4",
		UnifiedMemory:     true,
		MetalAvailable:    true,
	})
}

func macModelCache() modelcache.Status {
	return modelcache.NormalizeStatus(modelcache.Status{
		CacheDir: "/Users/operator/.ryvion/models",
		Models: []modelcache.Model{
			llamaModel("/Users/operator/.ryvion/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
			{
				ModelID:          "phi-4-Q5_K_M.gguf",
				Filename:         "phi-4-Q5_K_M.gguf",
				Path:             "/Users/operator/.ryvion/models/phi-4-Q5_K_M.gguf",
				SizeBytes:        int64(5 * gib),
				FamilyHint:       "phi",
				QuantizationHint: "Q5_K_M",
				Format:           "gguf",
				Installed:        true,
				LastSeenAt:       time.Unix(100, 0),
			},
		},
	})
}

func windowsModelCache() modelcache.Status {
	return modelcache.NormalizeStatus(modelcache.Status{
		CacheDir: `C:\Users\operator\.ryvion\models`,
		Models:   []modelcache.Model{llamaModel(`C:\Users\operator\.ryvion\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf`)},
	})
}

func windowsModelCacheWithPhi() modelcache.Status {
	cache := windowsModelCache()
	cache.Models = append(cache.Models, modelcache.Model{
		ModelID:                "phi-4-Q5_K_M.gguf",
		Filename:               "phi-4-Q5_K_M.gguf",
		Path:                   `C:\Users\operator\.ryvion\models\phi-4-Q5_K_M.gguf`,
		SizeBytes:              int64(10 * gib),
		FamilyHint:             "phi",
		QuantizationHint:       "Q5_K_M",
		ParameterCountBillions: 14,
		Format:                 "gguf",
		Installed:              true,
		Resident:               true,
		LastSeenAt:             time.Unix(100, 0),
	})
	return modelcache.NormalizeStatus(cache)
}

func llamaModel(path string) modelcache.Model {
	return modelcache.Model{
		ModelID:          "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		Filename:         "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		Path:             path,
		SizeBytes:        int64(3 * gib),
		FamilyHint:       "llama",
		QuantizationHint: "Q4_K_M",
		Format:           "gguf",
		Installed:        true,
		LastSeenAt:       time.Unix(100, 0),
	}
}

func llamaProbe(available bool) backendprobe.Probes {
	probe := backendprobe.LlamaCPPProbe{
		Available:                      available,
		BinaryPath:                     "/opt/ryvion/bin/llama-cli",
		ServerBinaryPath:               "/opt/ryvion/bin/llama-server",
		Version:                        "llama.cpp build 456",
		GGUFModelsDetected:             available,
		SupportsTextGeneration:         available,
		SupportsStreaming:              available,
		SupportsOpenAICompatibleServer: available,
		CandidateForFastTextRuntime:    available,
		Reason:                         "llama.cpp detected; real KV/tensor hooks require adapter implementation",
	}
	if !available {
		probe = backendprobe.LlamaCPPProbe{Reason: "llama.cpp binary not detected"}
	}
	return backendprobe.NormalizeProbes(backendprobe.Probes{LlamaCPP: probe})
}

func modelByID(t *testing.T, models []ModelCapability, modelID string) ModelCapability {
	t.Helper()
	for _, model := range models {
		if model.ModelID == modelID {
			return model
		}
	}
	t.Fatalf("model %q not found in %+v", modelID, models)
	return ModelCapability{}
}

func assertProfileJSONSafe(t *testing.T, profile Profile) {
	t.Helper()
	raw, err := json.Marshal(profile)
	if err != nil {
		t.Fatalf("json.Marshal(profile) error = %v", err)
	}
	body := strings.ToLower(string(raw))
	for _, forbidden := range []string{"raw_prompt", "prompt_text", "model_output", "output_text", "generated_text", "key_data", "value_data", "query_vector", "tensor_bytes", "raw_tensor", "secret"} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("capability profile JSON contains forbidden marker %q: %s", forbidden, raw)
		}
	}
}
