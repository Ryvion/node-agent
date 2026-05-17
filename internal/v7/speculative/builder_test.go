package speculative

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/Ryvion/ryvion-node/internal/v7/backendprobe"
	v7hardware "github.com/Ryvion/ryvion-node/internal/v7/hardware"
	"github.com/Ryvion/ryvion-node/internal/v7/llamacpp"
	"github.com/Ryvion/ryvion-node/internal/v7/modelcache"
	"github.com/Ryvion/ryvion-node/internal/v7/modelpolicy"
)

const gib = uint64(1024 * 1024 * 1024)

func TestBuildReportMacLlamaNgramRunnableDraftReasonPhiBlocked(t *testing.T) {
	t.Parallel()

	report := BuildReport(BuildInput{
		Hardware:        macHardware(),
		Policy:          macPolicy(),
		ModelCache:      macModelCacheNoDraft(),
		BackendProbes:   llamaProbe(true),
		BackendRuntimes: llamacpp.NormalizeBackendRuntimes(llamacpp.BackendRuntimes{}),
		Getenv:          func(string) string { return "" },
	})

	if !report.SpeculativeDecoding.Supported || !report.SpeculativeDecoding.Enabled {
		t.Fatalf("speculative_decoding = %+v, want enabled ngram support", report.SpeculativeDecoding)
	}
	if !containsString(report.SpeculativeDecoding.Methods, MethodNgram) {
		t.Fatalf("methods = %+v, want ngram", report.SpeculativeDecoding.Methods)
	}
	ngram := profileByTargetMethod(t, report.SpeculativeProfiles, "Llama-3.2-3B-Instruct-Q4_K_M.gguf", MethodNgram)
	if !ngram.Runnable || !ngram.TargetResident || ngram.MemoryEstimateBytes == 0 {
		t.Fatalf("llama ngram profile = %+v, want runnable resident target", ngram)
	}
	draft := profileByTargetMethod(t, report.SpeculativeProfiles, "Llama-3.2-3B-Instruct-Q4_K_M.gguf", MethodDraftModel)
	if draft.Runnable || !containsString(draft.BlockedReasons, "compatible_draft_model_not_found") {
		t.Fatalf("llama draft_model profile = %+v, want blocked missing compatible draft reason", draft)
	}
	for _, profile := range report.SpeculativeProfiles {
		if strings.Contains(strings.ToLower(profile.TargetModelID), "phi") && profile.Runnable {
			t.Fatalf("phi speculative profile should not be runnable: %+v", profile)
		}
	}
	assertJSONSafe(t, report)
}

func TestBuildReportDraftModelRequiresDraftPolicyAndPairMemory(t *testing.T) {
	t.Parallel()

	cache := modelcache.NormalizeStatus(modelcache.Status{
		CacheDir: "/Users/operator/.ryvion/models",
		Models: []modelcache.Model{
			llamaModel("/Users/operator/.ryvion/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf", 3*gib),
			llamaModel("/Users/operator/.ryvion/models/Llama-1B-Draft-Q4_K_M.gguf", 1*gib),
		},
	})
	report := BuildReport(BuildInput{
		Hardware:        macHardware(),
		Policy:          macPolicy(),
		ModelCache:      cache,
		BackendProbes:   llamaProbe(true),
		BackendRuntimes: llamacpp.NormalizeBackendRuntimes(llamacpp.BackendRuntimes{}),
		Getenv:          func(string) string { return "" },
	})

	profile := profileByTargetMethod(t, report.SpeculativeProfiles, "Llama-3.2-3B-Instruct-Q4_K_M.gguf", MethodDraftModel)
	if !profile.Runnable ||
		profile.DraftModelID != "Llama-1B-Draft-Q4_K_M.gguf" ||
		!profile.DraftResident ||
		!profile.TokenizerCompatible ||
		profile.MemoryEstimateBytes != 4*gib {
		t.Fatalf("draft_model profile = %+v, want runnable compatible draft pair", profile)
	}
}

func TestBuildReportSpeculativeOptOutDisablesRunnableProfiles(t *testing.T) {
	t.Parallel()

	report := BuildReport(BuildInput{
		Hardware:        macHardware(),
		Policy:          macPolicy(),
		ModelCache:      macModelCacheNoDraft(),
		BackendProbes:   llamaProbe(true),
		BackendRuntimes: llamacpp.NormalizeBackendRuntimes(llamacpp.BackendRuntimes{}),
		Getenv: func(key string) string {
			if key == EnvDisableSpeculativeDecoding {
				return "1"
			}
			return ""
		},
	})

	if !report.SpeculativeDecoding.Supported || report.SpeculativeDecoding.Enabled {
		t.Fatalf("speculative_decoding = %+v, want supported but disabled by operator opt-out", report.SpeculativeDecoding)
	}
	ngram := profileByTargetMethod(t, report.SpeculativeProfiles, "Llama-3.2-3B-Instruct-Q4_K_M.gguf", MethodNgram)
	if ngram.Runnable || !containsString(ngram.BlockedReasons, "speculative_decoding_disabled") {
		t.Fatalf("ngram profile = %+v, want opt-out blocked reason", ngram)
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

func macPolicy() modelpolicy.Policy {
	return modelpolicy.BuildDerivedPolicy(modelpolicy.DerivedPolicyInput{
		BasePolicy: modelpolicy.FromConfigSource(modelpolicy.ConfigSource{
			Getenv:      func(string) string { return "" },
			UserHomeDir: func() (string, error) { return "/Users/operator", nil },
			GOOS:        "darwin",
		}),
		Hardware: macHardware(),
	})
}

func macModelCacheNoDraft() modelcache.Status {
	return modelcache.NormalizeStatus(modelcache.Status{
		CacheDir: "/Users/operator/.ryvion/models",
		Models: []modelcache.Model{
			llamaModel("/Users/operator/.ryvion/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf", 3*gib),
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

func llamaModel(path string, sizeBytes uint64) modelcache.Model {
	parts := strings.Split(path, "/")
	filename := parts[len(parts)-1]
	return modelcache.Model{
		ModelID:          filename,
		Filename:         filename,
		Path:             path,
		SizeBytes:        int64(sizeBytes),
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
		Reason:                         "llama.cpp detected",
	}
	if !available {
		probe = backendprobe.LlamaCPPProbe{Reason: "llama.cpp binary not detected"}
	}
	return backendprobe.NormalizeProbes(backendprobe.Probes{LlamaCPP: probe})
}

func profileByTargetMethod(t *testing.T, profiles []Profile, targetModelID string, method string) Profile {
	t.Helper()
	for _, profile := range profiles {
		if profile.TargetModelID == targetModelID && profile.Method == method {
			return profile
		}
	}
	t.Fatalf("profile %s/%s not found in %+v", targetModelID, method, profiles)
	return Profile{}
}

func assertJSONSafe(t *testing.T, value any) {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	body := strings.ToLower(string(raw))
	for _, forbidden := range []string{"raw_prompt", "prompt_text", "model_output", "output_text", "generated_text", "key_data", "value_data", "query_vector", "tensor_bytes", "raw_tensor", "secret"} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("JSON contains forbidden marker %q: %s", forbidden, raw)
		}
	}
}
