package modelbench

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestNativeInferenceRunnerUnavailableWithoutNativeRuntime(t *testing.T) {
	spec := testModelBenchmarkSelfTestSpec()
	runner := NativeInferenceModelBenchmarkRunner{
		AgentVersion:     "test",
		RuntimeAvailable: func() bool { return false },
	}

	result, err := runner.RunModelBenchmark(context.Background(), spec)
	if err != nil {
		t.Fatalf("RunModelBenchmark() error = %v", err)
	}
	if result.ProofStatus != ModelBenchmarkProofStatusUnavailable {
		t.Fatalf("proof_status = %q, want %q", result.ProofStatus, ModelBenchmarkProofStatusUnavailable)
	}
	if result.Metrics.ErrorCode != "native_runtime_unavailable" {
		t.Fatalf("error_code = %q, want native_runtime_unavailable", result.Metrics.ErrorCode)
	}
	if result.OutputHash != "" || result.OutputBytes != 0 {
		t.Fatalf("unavailable result should not contain output hash/bytes: %+v", result)
	}
	if err := ValidateModelBenchmarkResult(result); err != nil {
		t.Fatalf("ValidateModelBenchmarkResult() error = %v", err)
	}
}

func TestNativeInferenceRunnerUnavailableWithoutManager(t *testing.T) {
	spec := testModelBenchmarkSelfTestSpec()
	runner := NativeInferenceModelBenchmarkRunner{
		AgentVersion:     "test",
		RuntimeAvailable: func() bool { return true },
	}

	result, err := runner.RunModelBenchmark(context.Background(), spec)
	if err != nil {
		t.Fatalf("RunModelBenchmark() error = %v", err)
	}
	if result.ProofStatus != ModelBenchmarkProofStatusUnavailable {
		t.Fatalf("proof_status = %q, want unavailable", result.ProofStatus)
	}
	if result.Metrics.ErrorCode != "native_inference_manager_unavailable" {
		t.Fatalf("error_code = %q, want manager unavailable", result.Metrics.ErrorCode)
	}
}

func TestNativeInferenceRunnerUnavailableWhenNotReady(t *testing.T) {
	spec := testModelBenchmarkSelfTestSpec()
	native := &fakeNativeInference{
		healthy:   false,
		serverURL: "http://127.0.0.1:1",
		modelName: spec.ModelID,
	}
	runner := NativeInferenceModelBenchmarkRunner{
		Native:           native,
		AgentVersion:     "test",
		RuntimeAvailable: func() bool { return true },
	}

	result, err := runner.RunModelBenchmark(context.Background(), spec)
	if err != nil {
		t.Fatalf("RunModelBenchmark() error = %v", err)
	}
	if result.ProofStatus != ModelBenchmarkProofStatusUnavailable {
		t.Fatalf("proof_status = %q, want unavailable", result.ProofStatus)
	}
	if result.Metrics.ErrorCode != "native_inference_not_ready" {
		t.Fatalf("error_code = %q, want native_inference_not_ready", result.Metrics.ErrorCode)
	}
	if native.ensureModel != spec.ModelID {
		t.Fatalf("EnsureModel called with %q, want %q", native.ensureModel, spec.ModelID)
	}
}

func TestNativeInferenceRunnerStreamsAndHashesMeasuredOutput(t *testing.T) {
	spec := testModelBenchmarkSelfTestSpec()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("path = %q, want /v1/chat/completions", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"ready\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\" now\"}}],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":2}}\n\n"))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer ts.Close()

	native := &fakeNativeInference{
		healthy:   true,
		serverURL: ts.URL,
		modelName: spec.ModelID,
	}
	runner := NativeInferenceModelBenchmarkRunner{
		Native:           native,
		AgentVersion:     "test",
		GPUDetected:      true,
		GPUModel:         "test-gpu",
		RuntimeAvailable: func() bool { return true },
	}

	result, err := runner.RunModelBenchmark(context.Background(), spec)
	if err != nil {
		t.Fatalf("RunModelBenchmark() error = %v", err)
	}
	if result.ProofStatus != ModelBenchmarkProofStatusMeasured {
		t.Fatalf("proof_status = %q, want measured", result.ProofStatus)
	}
	if result.OutputHash == "" || !strings.HasPrefix(result.OutputHash, "sha256:") {
		t.Fatalf("output_hash = %q, want sha256 hash", result.OutputHash)
	}
	if result.OutputBytes != int64(len("ready now")) {
		t.Fatalf("output_bytes = %d, want %d", result.OutputBytes, len("ready now"))
	}
	if result.Metrics.TokensGenerated != 2 {
		t.Fatalf("tokens_generated = %d, want completion token count", result.Metrics.TokensGenerated)
	}
	if result.Metrics.PromptTokens == nil || *result.Metrics.PromptTokens != 9 {
		t.Fatalf("prompt_tokens = %v, want 9", result.Metrics.PromptTokens)
	}
	if result.Metrics.CompletionTokens == nil || *result.Metrics.CompletionTokens != 2 {
		t.Fatalf("completion_tokens = %v, want 2", result.Metrics.CompletionTokens)
	}
	if result.Metrics.TimeToFirstTokenMs < 0 {
		t.Fatalf("time_to_first_token_ms = %d, want non-negative", result.Metrics.TimeToFirstTokenMs)
	}
	if err := ValidateModelBenchmarkResult(result); err != nil {
		t.Fatalf("ValidateModelBenchmarkResult() error = %v", err)
	}

	encoded, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("json.Marshal(result) error = %v", err)
	}
	if strings.Contains(string(encoded), "ready now") {
		t.Fatalf("result JSON leaked raw output: %s", encoded)
	}
}

func testModelBenchmarkSelfTestSpec() ModelBenchmarkSpec {
	return BuildModelBenchmarkSelfTestSpec(
		ModelBenchmarkSelfTestConfig{ModelID: "ryvion-llama-3.2-3b", MaxTokens: 16, TimeoutMs: 60_000},
		time.UnixMilli(1_800_000_000_000),
	)
}

type fakeNativeInference struct {
	healthy     bool
	serverURL   string
	modelName   string
	ensureModel string
	ensureErr   error
}

func (f *fakeNativeInference) Healthy() bool {
	return f.healthy
}

func (f *fakeNativeInference) EnsureModel(_ context.Context, model string) error {
	f.ensureModel = model
	return f.ensureErr
}

func (f *fakeNativeInference) ServerURL() string {
	return f.serverURL
}

func (f *fakeNativeInference) ModelName() string {
	return f.modelName
}
