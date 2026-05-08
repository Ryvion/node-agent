package dashboardinference

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"testing"

	"github.com/Ryvion/node-agent/internal/v7/llamacpp"
)

type fakeSidecar struct {
	status llamacpp.LlamaCppSidecarStatus
	starts int
}

func (f *fakeSidecar) Start(context.Context) llamacpp.LlamaCppSidecarStatus {
	f.starts++
	return f.status
}

func (f *fakeSidecar) Status(context.Context) llamacpp.LlamaCppSidecarStatus {
	return f.status
}

type fakeCompletionClient struct {
	result llamacpp.CompletionResult
	err    error
	reqs   []llamacpp.CompletionRequest
}

func (f *fakeCompletionClient) Complete(_ context.Context, req llamacpp.CompletionRequest) (llamacpp.CompletionResult, error) {
	f.reqs = append(f.reqs, req)
	if f.err != nil {
		return llamacpp.CompletionResult{}, f.err
	}
	return f.result, nil
}

func TestExecuteAssignmentRunsDashboardInferenceWithMockedLlamaCpp(t *testing.T) {
	sidecar := &fakeSidecar{status: healthySidecarStatus()}
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{
		Output:          []byte("dashboard inference measured output"),
		OutputBytes:     int64(len("dashboard inference measured output")),
		TokensGenerated: 7,
		TTFTMs:          80,
		TotalTimeMs:     430,
		Streamed:        true,
	}}
	receipt, handled, err := ExecuteAssignment(context.Background(), validSpecJSON(t), ExecuteOptions{
		Getenv: getenvEnabled,
		Runner: LlamaCppRunner{
			Sidecar: sidecar,
			Client:  client,
		},
	})
	if err != nil {
		t.Fatalf("ExecuteAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if sidecar.starts != 1 {
		t.Fatalf("sidecar starts = %d, want 1", sidecar.starts)
	}
	if len(client.reqs) != 1 {
		t.Fatalf("client calls = %d, want 1", len(client.reqs))
	}
	if !client.reqs[0].Stream {
		t.Fatal("completion request stream = false, want true")
	}
	if client.reqs[0].ModelID != "Llama-3.2-3B-Instruct-Q4_K_M.gguf" {
		t.Fatalf("completion model = %q", client.reqs[0].ModelID)
	}
	if receipt.JobID != "v7dashboardinfer_job" || receipt.MeteringUnits != 1 {
		t.Fatalf("receipt = %+v, want measured job receipt", receipt)
	}
	metadata, ok := receipt.Metadata[Task].(map[string]any)
	if !ok {
		t.Fatalf("receipt metadata missing %q: %+v", Task, receipt.Metadata)
	}
	if metadata["proof_status"] != ProofStatusMeasured {
		t.Fatalf("proof_status = %v", metadata["proof_status"])
	}
	if metadata["output_hash"] == "" || metadata["tokens_generated"] != int64(7) {
		t.Fatalf("metadata missing hash/metrics: %+v", metadata)
	}
	if metadata["ttft_ms"] != int64(80) || metadata["total_time_ms"] != int64(430) {
		t.Fatalf("timing metadata = %+v", metadata)
	}
	if metadata["decode_tps"] != float64(20) || metadata["end_to_end_tps"] != float64(16.279) {
		t.Fatalf("tps metadata = %+v", metadata)
	}
	if !ReceiptJSONContainsNoRawText(receipt) {
		raw, _ := json.Marshal(receipt.Metadata)
		t.Fatalf("metadata leaked raw text: %s", raw)
	}
}

func TestExecuteAssignmentFlagDisabledReturnsSafeRejection(t *testing.T) {
	client := &fakeCompletionClient{}
	receipt, handled, err := ExecuteAssignment(context.Background(), validSpecJSON(t), ExecuteOptions{
		Getenv: func(string) string { return "" },
		Runner: LlamaCppRunner{
			Sidecar: &fakeSidecar{status: healthySidecarStatus()},
			Client:  client,
		},
	})
	if err == nil {
		t.Fatal("ExecuteAssignment() error = nil, want disabled error")
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if len(client.reqs) != 0 {
		t.Fatalf("client calls = %d, want 0", len(client.reqs))
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["proof_status"] != ProofStatusRejected || metadata["error_code"] != "dashboard_inference_disabled" {
		t.Fatalf("disabled metadata = %+v", metadata)
	}
	if !ReceiptJSONContainsNoRawText(receipt) {
		raw, _ := json.Marshal(receipt.Metadata)
		t.Fatalf("disabled metadata leaked raw text: %s", raw)
	}
}

func TestExecuteAssignmentSidecarUnavailableReturnsSafeRejection(t *testing.T) {
	status := healthySidecarStatus()
	status.Available = false
	status.Running = false
	status.Healthy = false
	receipt, handled, err := ExecuteAssignment(context.Background(), validSpecJSON(t), ExecuteOptions{
		Getenv: getenvEnabled,
		Runner: LlamaCppRunner{
			Sidecar: &fakeSidecar{status: status},
			Client:  &fakeCompletionClient{},
		},
	})
	if err != nil {
		t.Fatalf("ExecuteAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["proof_status"] != ProofStatusRejected || metadata["error_code"] != "llamacpp_sidecar_unavailable" {
		t.Fatalf("sidecar unavailable metadata = %+v", metadata)
	}
	if metadata["output_hash"] == "" || metadata["output_bytes"] != int64(0) {
		t.Fatalf("safe rejection output metadata = %+v", metadata)
	}
}

func TestExecuteAssignmentUnsupportedBackendRejectedSafely(t *testing.T) {
	spec := validSpec(t)
	spec.Backend = "openai"
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	receipt, handled, err := ExecuteAssignment(context.Background(), string(raw), ExecuteOptions{Getenv: getenvEnabled})
	if err == nil {
		t.Fatal("ExecuteAssignment() error = nil, want unsupported backend error")
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["proof_status"] != ProofStatusRejected {
		t.Fatalf("proof_status = %v", metadata["proof_status"])
	}
	if _, ok := metadata["error_code"].(string); !ok {
		t.Fatalf("metadata missing error_code: %+v", metadata)
	}
}

func TestExecuteAssignmentNonDashboardTaskIgnored(t *testing.T) {
	receipt, handled, err := ExecuteAssignment(context.Background(), `{"task":"other"}`, ExecuteOptions{Getenv: getenvEnabled})
	if err != nil {
		t.Fatalf("ExecuteAssignment() error = %v", err)
	}
	if handled {
		t.Fatal("handled = true, want false")
	}
	if receipt.JobID != "" {
		t.Fatalf("receipt = %+v, want empty", receipt)
	}
}

func TestLlamaCppRunnerFallsBackWhenStreamingUnsupported(t *testing.T) {
	client := &fakeCompletionClient{err: llamacpp.ClientError{Code: "llamacpp_stream_unavailable"}}
	runner := LlamaCppRunner{
		Sidecar: &fakeSidecar{status: healthySidecarStatus()},
		Client:  client,
	}
	spec := validSpec(t)
	result, err := runner.RunDashboardInference(context.Background(), spec)
	if err != nil {
		t.Fatalf("RunDashboardInference() error = %v", err)
	}
	if result.ProofStatus != ProofStatusRejected || result.ErrorCode != "llamacpp_stream_unavailable" {
		t.Fatalf("result = %+v, want safe rejection", result)
	}
	if len(client.reqs) != 2 {
		t.Fatalf("client calls = %d, want stream attempt and fallback", len(client.reqs))
	}
	if !client.reqs[0].Stream || client.reqs[1].Stream {
		t.Fatalf("stream flags = %+v", []bool{client.reqs[0].Stream, client.reqs[1].Stream})
	}
}

func TestErrorCodeRedactsUnsafeErrorText(t *testing.T) {
	got := ErrorCode(errors.New("secret raw_output should not leak"))
	if got != "dashboard_inference_error_redacted" {
		t.Fatalf("ErrorCode() = %q", got)
	}
}

func validSpecJSON(t *testing.T) string {
	t.Helper()
	raw, err := json.Marshal(validSpec(t))
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	return string(raw)
}

func validSpec(t *testing.T) Spec {
	t.Helper()
	return Spec{
		Task:            Task,
		RequestID:       "dashboardinfer_request",
		RunID:           "dashboardinfer_run",
		JobID:           "v7dashboardinfer_job",
		Backend:         llamacpp.BackendName,
		ModelID:         "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		TargetNodeID:    "node-local",
		MaxTokens:       32,
		Stream:          true,
		CreatedAtUnixMs: 1_800_000_000_123,
		PromptHash:      testSHA256("dashboard prompt profile"),
		PromptProfileID: "dashboard_default",
	}
}

func healthySidecarStatus() llamacpp.LlamaCppSidecarStatus {
	return llamacpp.LlamaCppSidecarStatus{
		Enabled:                true,
		Available:              true,
		Running:                true,
		Healthy:                true,
		BaseURL:                "http://127.0.0.1:45910",
		ModelPath:              "/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		ModelFilename:          "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		Backend:                llamacpp.BackendName,
		SupportsTextGeneration: true,
		SupportsStreaming:      true,
	}
}

func getenvEnabled(key string) string {
	if key == FlagEnv {
		return "1"
	}
	return ""
}

func testSHA256(value string) string {
	sum := sha256.Sum256([]byte(value))
	return "sha256:" + hex.EncodeToString(sum[:])
}
