package inferencebench

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/Ryvion/node-agent/internal/v7/llamacpp"
)

func TestExecuteBenchmarkAssignmentRunsMockedLlamaCppClient(t *testing.T) {
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{
		Output:           []byte("secret measured output"),
		OutputBytes:      int64(len("secret measured output")),
		TokensGenerated:  12,
		CompletionTokens: 12,
		TTFTMs:           100,
		TotalTimeMs:      700,
		Streamed:         true,
	}}
	runner := LlamaCppBenchmarkRunner{
		Sidecar: healthyInferenceSidecar(),
		Client:  client,
		Getenv:  inferenceBenchmarkEnv,
	}

	receipt, handled, err := ExecuteBenchmarkAssignment(context.Background(), validInferenceBenchmarkSpecJSON(t), ExecuteOptions{
		Getenv: inferenceBenchmarkEnv,
		Runner: runner,
	})
	if err != nil {
		t.Fatalf("ExecuteBenchmarkAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if client.calls != 1 {
		t.Fatalf("client calls = %d, want 1", client.calls)
	}
	if len(client.requests) != 1 || !client.requests[0].Stream {
		t.Fatalf("request stream flag = %+v, want streaming request", client.requests)
	}
	if strings.TrimSpace(client.requests[0].Prompt) == "" {
		t.Fatal("client prompt is empty, want internal fixed prompt")
	}
	metadata := receipt.Metadata[BenchmarkTask].(map[string]any)
	if metadata["proof_status"] != ProofStatusMeasured {
		t.Fatalf("proof_status = %v, want measured", metadata["proof_status"])
	}
	if metadata["backend"] != llamacpp.BackendName || metadata["model_id"] != "tinyllama.Q4_K_M.gguf" {
		t.Fatalf("backend/model metadata = %+v", metadata)
	}
	if metadata["tokens_generated"] != int64(12) || metadata["p50_ttft_ms"] != int64(100) || metadata["p95_ttft_ms"] != int64(100) {
		t.Fatalf("metrics metadata = %+v", metadata)
	}
	if metadata["p50_decode_tps"] != 20.0 || metadata["p50_end_to_end_tps"] != 17.143 {
		t.Fatalf("tps metadata = %+v", metadata)
	}
	for _, key := range []string{"prompt_hash", "output_hash"} {
		value, ok := metadata[key].(string)
		if !ok || !strings.HasPrefix(value, "sha256:") {
			t.Fatalf("%s = %#v, want sha256 hash", key, metadata[key])
		}
	}
	assertInferenceBenchmarkReceiptSafe(t, receipt, "secret measured output")
}

func TestExecuteBenchmarkAssignmentSidecarUnavailableBuildsSafeFailedReceipt(t *testing.T) {
	client := &fakeCompletionClient{}
	runner := LlamaCppBenchmarkRunner{
		Sidecar: &fakeInferenceSidecar{status: llamacpp.LlamaCppSidecarStatus{
			Enabled:   true,
			Available: false,
			Running:   false,
			Healthy:   false,
			BaseURL:   "http://127.0.0.1:45910",
			Backend:   llamacpp.BackendName,
			Reason:    "llama-server binary not detected",
		}},
		Client: client,
		Getenv: inferenceBenchmarkEnv,
	}

	receipt, handled, err := ExecuteBenchmarkAssignment(context.Background(), validInferenceBenchmarkSpecJSON(t), ExecuteOptions{
		Getenv: inferenceBenchmarkEnv,
		Runner: runner,
	})
	if err != nil {
		t.Fatalf("ExecuteBenchmarkAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if receipt.MeteringUnits != 0 {
		t.Fatalf("metering_units = %d, want 0 for failed benchmark", receipt.MeteringUnits)
	}
	if client.calls != 0 {
		t.Fatalf("client calls = %d, want zero when sidecar unavailable", client.calls)
	}
	metadata := receipt.Metadata[BenchmarkTask].(map[string]any)
	if metadata["proof_status"] != ProofStatusRejected {
		t.Fatalf("proof_status = %v, want rejected", metadata["proof_status"])
	}
	if _, ok := metadata["output_hash"]; ok {
		t.Fatalf("rejection metadata includes output_hash: %+v", metadata)
	}
	if metadata["error_code"] != "llamacpp_sidecar_unavailable" {
		t.Fatalf("error_code = %v, want safe sidecar code", metadata["error_code"])
	}
	assertInferenceBenchmarkReceiptSafe(t, receipt)
}

func TestExecuteBenchmarkAssignmentFlagOffDoesNotHandle(t *testing.T) {
	runner := LlamaCppBenchmarkRunner{
		Sidecar: healthyInferenceSidecar(),
		Client:  &fakeCompletionClient{},
	}
	receipt, handled, err := ExecuteBenchmarkAssignment(context.Background(), validInferenceBenchmarkSpecJSON(t), ExecuteOptions{
		Getenv: func(string) string { return "" },
		Runner: runner,
	})
	if err != nil {
		t.Fatalf("ExecuteBenchmarkAssignment() error = %v", err)
	}
	if handled {
		t.Fatal("handled = true, want false when flag is off")
	}
	if receipt.ResultHashHex != "" {
		t.Fatalf("receipt = %+v, want zero receipt", receipt)
	}
}

func TestLlamaCppRunnerUsesKeepWarmCheckWhenEnabled(t *testing.T) {
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{
		Output:          []byte("keep warm measured"),
		OutputBytes:     int64(len("keep warm measured")),
		TokensGenerated: 3,
		TTFTMs:          10,
		TotalTimeMs:     100,
		Streamed:        true,
	}}
	sidecar := healthyInferenceSidecar()
	keeper := &fakeKeepWarmChecker{status: sidecar.status}
	runner := LlamaCppBenchmarkRunner{
		Sidecar:  sidecar,
		KeepWarm: keeper,
		Client:   client,
		Getenv: func(key string) string {
			if key == llamacpp.EnvKeepWarm {
				return "1"
			}
			return ""
		},
	}

	result, err := runner.RunBackendInferenceBenchmark(context.Background(), validInferenceBenchmarkSpec())
	if err != nil {
		t.Fatalf("RunBackendInferenceBenchmark() error = %v", err)
	}
	if result.ProofStatus != ProofStatusMeasured {
		t.Fatalf("proof_status = %q, want measured", result.ProofStatus)
	}
	if keeper.calls != 1 {
		t.Fatalf("keep warm calls = %d, want 1", keeper.calls)
	}
	if sidecar.startCalls != 0 {
		t.Fatalf("sidecar start calls = %d, want 0 when keep warm returns healthy running", sidecar.startCalls)
	}
}

type fakeInferenceSidecar struct {
	status      llamacpp.LlamaCppSidecarStatus
	startCalls  int
	statusCalls int
}

func healthyInferenceSidecar() *fakeInferenceSidecar {
	return &fakeInferenceSidecar{status: llamacpp.LlamaCppSidecarStatus{
		Enabled:           true,
		Available:         true,
		Running:           true,
		Healthy:           true,
		BaseURL:           "http://127.0.0.1:45910",
		ModelPath:         "/models/tinyllama.Q4_K_M.gguf",
		ModelFilename:     "tinyllama.Q4_K_M.gguf",
		Backend:           llamacpp.BackendName,
		SupportsStreaming: true,
	}}
}

func (f *fakeInferenceSidecar) Start(context.Context) llamacpp.LlamaCppSidecarStatus {
	f.startCalls++
	return f.status
}

func (f *fakeInferenceSidecar) Status(context.Context) llamacpp.LlamaCppSidecarStatus {
	f.statusCalls++
	return f.status
}

type fakeKeepWarmChecker struct {
	status llamacpp.LlamaCppSidecarStatus
	calls  int
}

func (f *fakeKeepWarmChecker) CheckOnce(context.Context) llamacpp.LlamaCppSidecarStatus {
	f.calls++
	return f.status
}

type fakeCompletionClient struct {
	result   llamacpp.CompletionResult
	err      error
	requests []llamacpp.CompletionRequest
	calls    int
}

func (f *fakeCompletionClient) Complete(_ context.Context, req llamacpp.CompletionRequest) (llamacpp.CompletionResult, error) {
	f.calls++
	f.requests = append(f.requests, req)
	if f.err != nil {
		return llamacpp.CompletionResult{}, f.err
	}
	return f.result, nil
}

func validInferenceBenchmarkSpec() BenchmarkSpec {
	return BenchmarkSpec{
		Task:            BenchmarkTask,
		RequestID:       "request-backend-inference-local",
		BenchmarkID:     "benchmark-backend-inference-local",
		JobID:           "job-backend-inference-local",
		Backend:         llamacpp.BackendName,
		ModelID:         "tinyllama.Q4_K_M.gguf",
		TargetNodeID:    "node-backend-inference-local",
		PromptHash:      llamacpp.HashBenchmarkPrompt(),
		PromptProfileID: BenchmarkPromptProfileID,
		MaxTokens:       16,
		TimeoutMs:       30_000,
		CreatedAtUnixMs: 1_800_000_000_123,
	}
}

func validInferenceBenchmarkSpecJSON(t *testing.T) string {
	t.Helper()
	encoded, err := json.Marshal(validInferenceBenchmarkSpec())
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	return string(encoded)
}

func inferenceBenchmarkEnv(key string) string {
	if key == BenchmarkFlagEnv {
		return "1"
	}
	return ""
}
