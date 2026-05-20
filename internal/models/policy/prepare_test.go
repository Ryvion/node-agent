package modelpolicy

import "testing"

func TestEvaluatePrepareRequestBlocksUnsafePolicy(t *testing.T) {
	t.Parallel()

	base := Policy{
		AutoDownload:        true,
		MaxSingleModelBytes: 10,
		MaxCacheBytes:       20,
		CacheDir:            "/cache",
		AllowedFamilies:     []string{"llama"},
		AllowedFormats:      []string{"gguf"},
	}
	cases := []struct {
		name    string
		policy  Policy
		request PrepareRequest
		reason  string
	}{
		{
			name:    "auto download disabled",
			policy:  Policy{AutoDownload: false, MaxSingleModelBytes: 10, MaxCacheBytes: 20, CacheDir: "/cache", AllowedFamilies: []string{"llama"}, AllowedFormats: []string{"gguf"}},
			request: PrepareRequest{ModelSizeBytes: 1, Format: "gguf"},
			reason:  PrepareDecisionAutoDownloadDisabled,
		},
		{
			name:    "single model too large",
			policy:  base,
			request: PrepareRequest{ModelSizeBytes: 11, Format: "gguf"},
			reason:  PrepareDecisionModelTooLarge,
		},
		{
			name:    "cache capacity exceeded",
			policy:  base,
			request: PrepareRequest{ModelSizeBytes: 8, CacheUsedBytes: 14, Format: "gguf"},
			reason:  PrepareDecisionCacheCapacity,
		},
		{
			name:    "family blocked",
			policy:  base,
			request: PrepareRequest{ModelSizeBytes: 8, Family: "mistral", Format: "gguf"},
			reason:  PrepareDecisionFamilyNotAllowed,
		},
		{
			name:    "format blocked",
			policy:  base,
			request: PrepareRequest{ModelSizeBytes: 8, Family: "llama", Format: "bin"},
			reason:  PrepareDecisionFormatNotAllowed,
		},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			decision := EvaluatePrepareRequest(tc.policy, tc.request)
			if decision.Allowed || decision.Reason != tc.reason {
				t.Fatalf("decision = %+v, want blocked %q", decision, tc.reason)
			}
		})
	}
}

func TestEvaluatePrepareRequestAllowsSafeRequest(t *testing.T) {
	t.Parallel()

	decision := EvaluatePrepareRequest(Policy{
		AutoDownload:        true,
		MaxSingleModelBytes: 10,
		MaxCacheBytes:       20,
		CacheDir:            "/cache",
		AllowedFamilies:     []string{"llama"},
		AllowedFormats:      []string{"gguf"},
	}, PrepareRequest{
		ModelID:        "tinyllama.Q4_K_M.gguf",
		ModelSizeBytes: 8,
		CacheUsedBytes: 2,
		Family:         "llama",
		Format:         "gguf",
	})
	if !decision.Allowed || decision.Reason != PrepareDecisionAllowed {
		t.Fatalf("decision = %+v, want allowed", decision)
	}
}
