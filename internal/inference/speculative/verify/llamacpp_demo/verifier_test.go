package llamacppdemo

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/Ryvion/ryvion-node/internal/hub"
	nodespec "github.com/Ryvion/ryvion-node/internal/inference/speculative"
	llamacpp "github.com/Ryvion/ryvion-node/internal/runtimes/llamacpp"
)

func TestVerifierVerifyWaveUsesMeasuredCompletion(t *testing.T) {
	sidecar := &fakeSidecar{status: readyStatus()}
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{
		Output:                   []byte("Verified local text."),
		OutputBytes:              int64(len("Verified local text.")),
		TokensGenerated:          4,
		CompletionTokens:         4,
		RequestedMaxTokens:       8,
		FinishReason:             "stop",
		RuntimeMeasurementStatus: llamacpp.RuntimeMeasurementStatusMeasured,
		MetadataParseStatus:      llamacpp.MetadataParseStatusOK,
		TotalTimeMs:              25,
		Streamed:                 true,
	}}
	verifier := Verifier{Sidecar: sidecar, Client: client}

	result, err := verifier.VerifyWave(context.Background(), nodespec.HotSessionSpec{
		RunID:            "flab_llama",
		SessionID:        "sess_llama",
		WorkGraphID:      "wg_llama",
		ModelID:          "tinyllama",
		Prompt:           "Write one short sentence.",
		ParentPrefixHash: "sha256:prefix",
		MaxTokens:        8,
	}, hub.SpeculativeLiveLabSessionCommand{
		CommandID:   "cmd_1",
		WindowID:    "win_llama",
		WaveIndex:   1,
		WorkGraphID: "wg_llama",
		Tree: map[string]any{
			"tree_cid": "sha256:tree",
			"branches": []any{
				map[string]any{"candidate_tokens": []any{float64(1), float64(2), float64(3), float64(4), float64(5)}},
			},
		},
	}, 0)
	if err != nil {
		t.Fatalf("VerifyWave() error = %v, want nil", err)
	}
	if !sidecar.started {
		t.Fatal("VerifyWave() did not start llama.cpp sidecar")
	}
	if len(client.requests) != 1 {
		t.Fatalf("completion requests = %d, want 1", len(client.requests))
	}
	req := client.requests[0]
	if req.BaseURL != readyStatus().BaseURL || req.ModelID != "tinyllama" || !req.Stream {
		t.Fatalf("completion request = %+v, want hot llama.cpp request", req)
	}
	if req.Prompt == "" || !strings.Contains(req.Prompt, "sha256:tree") || !strings.Contains(req.Prompt, "branch_count=1") {
		t.Fatalf("completion prompt = %q, want verifier tree summary", req.Prompt)
	}
	if result.AcceptedLen != 4 || result.TreeCID != "sha256:tree" || result.AcceptedText != "Verified local text." || !result.AcceptedTextPublic {
		t.Fatalf("verifier result = %+v, want measured llama.cpp acceptance", result)
	}
	if result.StopReason != "" || result.EOS {
		t.Fatalf("verifier result stop = %q eos=%v, want non-terminal finish_reason=stop", result.StopReason, result.EOS)
	}
	if result.ProbeSummary["source"] != Executor ||
		result.ProbeSummary["backend"] != llamacpp.BackendName ||
		result.ProbeSummary["output_hash"] == "" {
		t.Fatalf("probe summary = %#v, want llama.cpp measured metadata", result.ProbeSummary)
	}
	if encoded, _ := json.Marshal(result.ProbeSummary); strings.Contains(string(encoded), "Write one short sentence") || strings.Contains(string(encoded), "Verified local text") {
		t.Fatalf("probe summary leaked raw prompt/output: %s", encoded)
	}
}

func TestVerifierUnavailableDoesNotUseSyntheticFallback(t *testing.T) {
	sidecar := &fakeSidecar{status: llamacpp.LlamaCppSidecarStatus{
		Enabled:   true,
		Available: false,
		Backend:   llamacpp.BackendName,
		Reason:    "llama-server binary not detected",
	}}
	client := &fakeCompletionClient{}
	verifier := Verifier{Sidecar: sidecar, Client: client}

	_, err := verifier.VerifyWave(context.Background(), nodespec.HotSessionSpec{
		RunID:       "flab_llama",
		WorkGraphID: "wg_llama",
		ModelID:     "tinyllama",
	}, hub.SpeculativeLiveLabSessionCommand{
		CommandID: "cmd_1",
		WindowID:  "win_llama",
		Tree:      map[string]any{"tree_cid": "sha256:tree"},
	}, 0)
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("VerifyWave() error = %v, want ErrUnavailable", err)
	}
	if len(client.requests) != 0 {
		t.Fatalf("completion requests = %d, want 0 when llama.cpp unavailable", len(client.requests))
	}
}

func TestLabStopReason(t *testing.T) {
	tests := []struct {
		name       string
		completion llamacpp.CompletionResult
		wantReason string
		wantEOS    bool
	}{
		{
			name:       "backend stop is not lab eos",
			completion: llamacpp.CompletionResult{FinishReason: llamacpp.FinishReasonStop},
		},
		{
			name:       "length maps to max tokens",
			completion: llamacpp.CompletionResult{FinishReason: llamacpp.FinishReasonLength},
			wantReason: "max_tokens",
		},
		{
			name:       "max tokens flag wins",
			completion: llamacpp.CompletionResult{FinishReason: llamacpp.FinishReasonStop, MaxTokensReached: true},
			wantReason: "max_tokens",
		},
		{
			name:       "explicit eos remains terminal",
			completion: llamacpp.CompletionResult{BackendStopReason: "eos"},
			wantReason: "eos",
			wantEOS:    true,
		},
		{
			name:       "timeout remains terminal",
			completion: llamacpp.CompletionResult{FinishReason: llamacpp.FinishReasonTimeout},
			wantReason: "timeout",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotReason, gotEOS := LabStopReason(tt.completion)
			if gotReason != tt.wantReason || gotEOS != tt.wantEOS {
				t.Fatalf("LabStopReason() = %q/%v, want %q/%v", gotReason, gotEOS, tt.wantReason, tt.wantEOS)
			}
		})
	}
}

func readyStatus() llamacpp.LlamaCppSidecarStatus {
	return llamacpp.LlamaCppSidecarStatus{
		Enabled:                true,
		Available:              true,
		Running:                true,
		Healthy:                true,
		BaseURL:                "http://127.0.0.1:45910",
		ModelFilename:          "tinyllama.Q4_K_M.gguf",
		Backend:                llamacpp.BackendName,
		OpenAICompatible:       true,
		SupportsTextGeneration: true,
		SupportsStreaming:      true,
	}
}

type fakeSidecar struct {
	status  llamacpp.LlamaCppSidecarStatus
	started bool
}

func (f *fakeSidecar) Start(context.Context) llamacpp.LlamaCppSidecarStatus {
	f.started = true
	return f.status
}

func (f *fakeSidecar) Status(context.Context) llamacpp.LlamaCppSidecarStatus {
	return f.status
}

type fakeCompletionClient struct {
	result   llamacpp.CompletionResult
	err      error
	requests []llamacpp.CompletionRequest
}

func (f *fakeCompletionClient) Complete(_ context.Context, req llamacpp.CompletionRequest) (llamacpp.CompletionResult, error) {
	f.requests = append(f.requests, req)
	if f.err != nil {
		return llamacpp.CompletionResult{}, f.err
	}
	return f.result, nil
}
