package modelpolicy

import "testing"

func TestEvaluateRuntimeRequestBlocksDeniedModelID(t *testing.T) {
	t.Parallel()

	policy := NormalizePolicy(Policy{
		AutoDownload:        true,
		CacheDir:            "/cache",
		MaxSingleModelBytes: 8 * bytesPerGiB,
		MaxCacheBytes:       50 * bytesPerGiB,
		AllowedFamilies:     []string{"llama", "phi"},
		AllowedFormats:      []string{"gguf"},
		RuntimePolicy: RuntimePolicy{
			AllowRuntimeExecution:            true,
			MaxRuntimeModelBytes:             8 * bytesPerGiB,
			MaxRuntimeParameterCountBillions: 8,
			AllowCPUOffload:                  true,
			DenyModelIDs:                     []string{"phi-4-Q4_K_M.gguf"},
			AllowFamilies:                    []string{"llama"},
		},
	})

	decision := EvaluateRuntimeRequest(policy, RuntimeRequest{
		ModelID:                "phi-4-Q4_K_M.gguf",
		ModelSizeBytes:         4 * bytesPerGiB,
		ParameterCountBillions: 14,
		Family:                 "phi",
	})
	if decision.Allowed || decision.Reason != RuntimeDecisionModelDenied {
		t.Fatalf("decision = %+v, want denied model id", decision)
	}
}

func TestEvaluateRuntimeRequestDistinguishesInstalledFromRuntimeAllowed(t *testing.T) {
	t.Parallel()

	policy := NormalizePolicy(Policy{
		AutoDownload:        true,
		CacheDir:            "/cache",
		MaxSingleModelBytes: 8 * bytesPerGiB,
		MaxCacheBytes:       50 * bytesPerGiB,
		AllowedFamilies:     []string{"llama", "phi"},
		AllowedFormats:      []string{"gguf"},
		RuntimePolicy: RuntimePolicy{
			AllowRuntimeExecution:            true,
			MaxRuntimeModelBytes:             8 * bytesPerGiB,
			MaxRuntimeParameterCountBillions: 8,
			AllowCPUOffload:                  true,
			AllowFamilies:                    []string{"llama"},
		},
	})

	installDecision := EvaluatePrepareRequest(policy, PrepareRequest{
		ModelID:        "phi-4-Q4_K_M.gguf",
		ModelSizeBytes: 4 * bytesPerGiB,
		CacheUsedBytes: 0,
		Family:         "phi",
		Format:         "gguf",
	})
	if !installDecision.Allowed {
		t.Fatalf("prepare decision = %+v, want installed/cache policy to allow phi", installDecision)
	}

	runtimeDecision := EvaluateRuntimeRequest(policy, RuntimeRequest{
		ModelID:                "phi-4-Q4_K_M.gguf",
		ModelSizeBytes:         4 * bytesPerGiB,
		ParameterCountBillions: 4,
		Family:                 "phi",
	})
	if runtimeDecision.Allowed || runtimeDecision.Reason != RuntimeDecisionFamilyNotAllowed {
		t.Fatalf("runtime decision = %+v, want runtime family block", runtimeDecision)
	}
}

func TestEvaluateRuntimeRequestBlocksLargeModelWithoutLargePolicy(t *testing.T) {
	t.Parallel()

	policy := NormalizePolicy(Policy{
		CacheDir:            "/cache",
		MaxSingleModelBytes: 16 * bytesPerGiB,
		MaxCacheBytes:       50 * bytesPerGiB,
		AllowedFamilies:     []string{"llama"},
		AllowedFormats:      []string{"gguf"},
		RuntimePolicy: RuntimePolicy{
			AllowRuntimeExecution:            true,
			MaxRuntimeModelBytes:             8 * bytesPerGiB,
			MaxRuntimeParameterCountBillions: 8,
			AllowCPUOffload:                  true,
			AllowLargeModels:                 false,
			AllowFamilies:                    []string{"llama"},
		},
	})

	decision := EvaluateRuntimeRequest(policy, RuntimeRequest{
		ModelID:                "Llama-70B.Q4_K_M.gguf",
		ModelSizeBytes:         40 * bytesPerGiB,
		ParameterCountBillions: 70,
		Family:                 "llama",
	})
	if decision.Allowed || decision.Reason != RuntimeDecisionLargeModelNotAllowed {
		t.Fatalf("decision = %+v, want large model block", decision)
	}
}
