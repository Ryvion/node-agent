package dashboardinference

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/Ryvion/node-agent/internal/v7/llamacpp"
	"github.com/Ryvion/node-agent/internal/v7/modelpolicy"
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
	deltas []string
	reqs   []llamacpp.CompletionRequest
}

func (f *fakeCompletionClient) Complete(_ context.Context, req llamacpp.CompletionRequest) (llamacpp.CompletionResult, error) {
	f.reqs = append(f.reqs, req)
	if f.err != nil {
		return llamacpp.CompletionResult{}, f.err
	}
	for _, delta := range f.deltas {
		if req.OnDelta == nil {
			continue
		}
		if err := req.OnDelta(llamacpp.CompletionDelta{Text: delta}); err != nil {
			return llamacpp.CompletionResult{}, err
		}
	}
	return f.result, nil
}

type fakeProgressSender struct {
	err     error
	batches []ProgressBatch
}

func (f *fakeProgressSender) SendDashboardInferenceProgress(_ context.Context, batch ProgressBatch) error {
	f.batches = append(f.batches, batch)
	if f.err != nil {
		return f.err
	}
	return nil
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
	if client.reqs[0].Stream {
		t.Fatal("completion request stream = true without return_text/text/stream flags, want false")
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
	if _, ok := metadata["generated_text"]; ok {
		t.Fatalf("metadata included generated_text without opt-in: %+v", metadata)
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

func TestExecuteAssignmentUsesPromptWithoutReturningText(t *testing.T) {
	sidecar := &fakeSidecar{status: healthySidecarStatus()}
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{
		Output:          []byte("private completion text"),
		OutputBytes:     int64(len("private completion text")),
		TokensGenerated: 3,
		TTFTMs:          20,
		TotalTimeMs:     120,
	}}
	spec := validSpec(t)
	spec.Prompt = "private dashboard prompt"
	spec.ReturnText = false
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	receipt, handled, err := ExecuteAssignment(context.Background(), string(raw), ExecuteOptions{
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
	if len(client.reqs) != 1 || client.reqs[0].Prompt != "private dashboard prompt" {
		t.Fatal("completion request did not use supplied prompt")
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if _, ok := metadata["generated_text"]; ok {
		t.Fatalf("metadata included generated_text without return_text: %+v", metadata)
	}
	encoded, _ := json.Marshal(receipt.Metadata)
	for _, forbidden := range []string{"private dashboard prompt", "private completion text", "generated_text"} {
		if strings.Contains(string(encoded), forbidden) {
			t.Fatalf("metadata leaked %q: %s", forbidden, encoded)
		}
	}
}

func TestExecuteAssignmentReturnTextFlagDisabledRejects(t *testing.T) {
	client := &fakeCompletionClient{}
	spec := validSpec(t)
	spec.Prompt = "private dashboard prompt"
	spec.ReturnText = true
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	receipt, handled, err := ExecuteAssignment(context.Background(), string(raw), ExecuteOptions{
		Getenv: getenvEnabled,
		Runner: LlamaCppRunner{
			Sidecar: &fakeSidecar{status: healthySidecarStatus()},
			Client:  client,
		},
	})
	if err == nil {
		t.Fatal("ExecuteAssignment() error = nil, want text output disabled error")
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if len(client.reqs) != 0 {
		t.Fatalf("client calls = %d, want 0", len(client.reqs))
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["proof_status"] != ProofStatusRejected || metadata["error_code"] != "text_output_disabled" {
		t.Fatalf("disabled text metadata = %+v", metadata)
	}
	encoded, _ := json.Marshal(receipt.Metadata)
	if strings.Contains(string(encoded), "private dashboard prompt") || strings.Contains(string(encoded), "generated_text") {
		t.Fatalf("disabled text receipt leaked prompt/text field: %s", encoded)
	}
}

func TestExecuteAssignmentReturnTextFlagEnabledIncludesGeneratedText(t *testing.T) {
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{
		Output:          []byte("Ryvion routes AI work to warm, ready nodes."),
		OutputBytes:     int64(len("Ryvion routes AI work to warm, ready nodes.")),
		TokensGenerated: 8,
		TTFTMs:          123,
		TotalTimeMs:     456,
	}}
	spec := validSpec(t)
	spec.Prompt = "Write one short sentence about Ryvion."
	spec.ReturnText = true
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	receipt, handled, err := ExecuteAssignment(context.Background(), string(raw), ExecuteOptions{
		Getenv: getenvTextOutputEnabled,
		Runner: LlamaCppRunner{
			Sidecar: &fakeSidecar{status: healthySidecarStatus()},
			Client:  client,
		},
	})
	if err != nil {
		t.Fatalf("ExecuteAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["generated_text"] != "Ryvion routes AI work to warm, ready nodes." {
		t.Fatalf("generated_text = %#v", metadata["generated_text"])
	}
	if metadata["generated_text_truncated"] != false {
		t.Fatalf("generated_text_truncated = %#v", metadata["generated_text_truncated"])
	}
	if metadata["output_hash"] == "" || metadata["tokens_generated"] != int64(8) || metadata["ttft_ms"] != int64(123) {
		t.Fatalf("metadata missing hash/timing metrics: %+v", metadata)
	}
	if len(receipt.ResultHashHex) != 64 {
		t.Fatalf("result hash = %q, want 64 hex chars", receipt.ResultHashHex)
	}
}

func TestLlamaCppRunnerStreamsReturnTextWithProgressBatches(t *testing.T) {
	output := "Ryvion routes tokens"
	client := &fakeCompletionClient{
		result: llamacpp.CompletionResult{
			Output:          []byte(output),
			OutputBytes:     int64(len(output)),
			TokensGenerated: 3,
			TTFTMs:          90,
			TotalTimeMs:     390,
		},
		deltas: []string{"Ryvion", " routes", " tokens"},
	}
	progress := &fakeProgressSender{}
	spec := validSpec(t)
	spec.Prompt = "Write a short dashboard sentence."
	spec.ReturnText = true
	runner := LlamaCppRunner{
		Sidecar: &fakeSidecar{status: healthySidecarStatus()},
		Client:  client,
		Getenv:  getenvTextAndStreamingEnabled,
	}
	result, err := runner.RunDashboardInferenceWithProgress(context.Background(), spec, progress)
	if err != nil {
		t.Fatalf("RunDashboardInferenceWithProgress() error = %v", err)
	}
	if len(client.reqs) != 1 || !client.reqs[0].Stream {
		t.Fatalf("stream requests = %+v, want one streaming request", client.reqs)
	}
	if result.ProofStatus != ProofStatusMeasured || result.GeneratedText != output {
		t.Fatalf("result = %+v, want measured generated text", result)
	}
	if len(progress.batches) != 1 {
		t.Fatalf("progress batches = %d, want 1", len(progress.batches))
	}
	batch := progress.batches[0]
	if batch.RunID != spec.RunID || batch.JobID != spec.JobID || batch.NodeID != spec.TargetNodeID || batch.SeqStart != 1 {
		t.Fatalf("batch identity = %+v", batch)
	}
	if len(batch.Chunks) != 3 {
		t.Fatalf("batch chunks = %+v, want 3", batch.Chunks)
	}
	for idx, chunk := range batch.Chunks {
		wantSeq := int64(idx + 1)
		if chunk.Seq != wantSeq || chunk.Type != "delta" {
			t.Fatalf("chunk[%d] = %+v, want seq %d delta", idx, chunk, wantSeq)
		}
	}
	if got := batch.Chunks[0].Text + batch.Chunks[1].Text + batch.Chunks[2].Text; got != output {
		t.Fatalf("concatenated chunks = %q, want %q", got, output)
	}
}

func TestLlamaCppRunnerDoesNotPostChunksWhenStreamingFlagDisabled(t *testing.T) {
	client := &fakeCompletionClient{
		result: llamacpp.CompletionResult{
			Output:          []byte("non streaming output"),
			OutputBytes:     int64(len("non streaming output")),
			TokensGenerated: 3,
			TTFTMs:          300,
			TotalTimeMs:     300,
		},
		deltas: []string{"should not post"},
	}
	progress := &fakeProgressSender{}
	spec := validSpec(t)
	spec.Prompt = "Write a short dashboard sentence."
	spec.ReturnText = true
	runner := LlamaCppRunner{
		Sidecar: &fakeSidecar{status: healthySidecarStatus()},
		Client:  client,
		Getenv:  getenvTextOutputEnabled,
	}
	result, err := runner.RunDashboardInferenceWithProgress(context.Background(), spec, progress)
	if err != nil {
		t.Fatalf("RunDashboardInferenceWithProgress() error = %v", err)
	}
	if len(client.reqs) != 1 || client.reqs[0].Stream {
		t.Fatalf("stream requests = %+v, want one non-streaming request", client.reqs)
	}
	if len(progress.batches) != 0 {
		t.Fatalf("progress batches = %+v, want none", progress.batches)
	}
	if result.ProofStatus != ProofStatusMeasured {
		t.Fatalf("result = %+v, want measured result", result)
	}
}

func TestLlamaCppRunnerDoesNotPostChunksWhenSpecStreamFalse(t *testing.T) {
	client := &fakeCompletionClient{
		result: llamacpp.CompletionResult{
			Output:          []byte("non streaming output"),
			OutputBytes:     int64(len("non streaming output")),
			TokensGenerated: 3,
			TTFTMs:          300,
			TotalTimeMs:     300,
		},
		deltas: []string{"should not post"},
	}
	progress := &fakeProgressSender{}
	spec := validSpec(t)
	spec.Prompt = "Write a short dashboard sentence."
	spec.ReturnText = true
	spec.Stream = false
	runner := LlamaCppRunner{
		Sidecar: &fakeSidecar{status: healthySidecarStatus()},
		Client:  client,
		Getenv:  getenvTextAndStreamingEnabled,
	}
	result, err := runner.RunDashboardInferenceWithProgress(context.Background(), spec, progress)
	if err != nil {
		t.Fatalf("RunDashboardInferenceWithProgress() error = %v", err)
	}
	if len(client.reqs) != 1 || client.reqs[0].Stream {
		t.Fatalf("stream requests = %+v, want one non-streaming request", client.reqs)
	}
	if len(progress.batches) != 0 {
		t.Fatalf("progress batches = %+v, want none", progress.batches)
	}
	if result.ProofStatus != ProofStatusMeasured {
		t.Fatalf("result = %+v, want measured result", result)
	}
}

func TestLlamaCppRunnerProgressFailureReturnsSafeRejection(t *testing.T) {
	client := &fakeCompletionClient{
		result: llamacpp.CompletionResult{
			Output:          []byte("private streamed output"),
			OutputBytes:     int64(len("private streamed output")),
			TokensGenerated: 3,
			TTFTMs:          40,
			TotalTimeMs:     240,
		},
		deltas: []string{"private", " streamed", " output"},
	}
	progress := &fakeProgressSender{err: errors.New("secret private streamed output")}
	spec := validSpec(t)
	spec.Prompt = "private prompt"
	spec.ReturnText = true
	runner := LlamaCppRunner{
		Sidecar: &fakeSidecar{status: healthySidecarStatus()},
		Client:  client,
		Getenv:  getenvTextAndStreamingEnabled,
	}
	result, err := runner.RunDashboardInferenceWithProgress(context.Background(), spec, progress)
	if err != nil {
		t.Fatalf("RunDashboardInferenceWithProgress() error = %v", err)
	}
	if result.ProofStatus != ProofStatusRejected || result.ErrorCode != "dashboard_inference_stream_progress_failed" {
		t.Fatalf("result = %+v, want safe progress rejection", result)
	}
	if strings.Contains(result.ErrorCode, "private") || strings.Contains(result.GeneratedText, "private") {
		t.Fatalf("result leaked private text: %+v", result)
	}
}

func TestExecuteAssignmentReturnTextTruncatesDeterministically(t *testing.T) {
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{
		Output:          []byte("abcdef"),
		OutputBytes:     6,
		TokensGenerated: 1,
		TTFTMs:          1,
		TotalTimeMs:     2,
	}}
	spec := validSpec(t)
	spec.Prompt = "Write six letters."
	spec.ReturnText = true
	spec.MaxReturnChars = 3
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	receipt, _, err := ExecuteAssignment(context.Background(), string(raw), ExecuteOptions{
		Getenv: getenvTextOutputEnabled,
		Runner: LlamaCppRunner{
			Sidecar: &fakeSidecar{status: healthySidecarStatus()},
			Client:  client,
		},
	})
	if err != nil {
		t.Fatalf("ExecuteAssignment() error = %v", err)
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["generated_text"] != "abc" || metadata["generated_text_truncated"] != true {
		t.Fatalf("truncated generated metadata = %+v", metadata)
	}
	if metadata["output_hash"] != HashOutput(spec.JobID, []byte("abcdef")) {
		t.Fatalf("output_hash = %v, want hash over full output", metadata["output_hash"])
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

func TestLlamaCppRunnerRejectsPhi4WhenRuntimePolicyBlocksFamily(t *testing.T) {
	status := healthySidecarStatus()
	status.ModelPath = "/models/phi-4-Q4_K_M.gguf"
	status.ModelFilename = "phi-4-Q4_K_M.gguf"
	status.ModelFamilyHint = "phi"
	status.ModelSizeBytes = 4 * 1024 * 1024 * 1024
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{Output: []byte("should not run")}}
	spec := validSpec(t)
	spec.ModelID = "phi-4-Q4_K_M.gguf"
	spec.Prompt = "Write a short dashboard sentence."
	spec.ReturnText = true
	progress := &fakeProgressSender{}
	runner := LlamaCppRunner{
		Sidecar: &fakeSidecar{status: status},
		Client:  client,
		Getenv:  getenvTextAndStreamingEnabled,
		Policy: modelpolicy.Policy{
			CacheDir:            "/models",
			MaxSingleModelBytes: 8 * 1024 * 1024 * 1024,
			MaxCacheBytes:       50 * 1024 * 1024 * 1024,
			AllowedFamilies:     []string{"llama", "phi"},
			AllowedFormats:      []string{"gguf"},
			RuntimePolicy: modelpolicy.RuntimePolicy{
				AllowRuntimeExecution:            true,
				MaxRuntimeModelBytes:             8 * 1024 * 1024 * 1024,
				MaxRuntimeParameterCountBillions: 8,
				AllowCPUOffload:                  true,
				AllowFamilies:                    []string{"llama"},
			},
		},
	}
	result, err := runner.RunDashboardInferenceWithProgress(context.Background(), spec, progress)
	if err != nil {
		t.Fatalf("RunDashboardInference() error = %v", err)
	}
	if result.ProofStatus != ProofStatusRejected || result.ErrorCode != modelpolicy.RuntimeDecisionFamilyNotAllowed {
		t.Fatalf("result = %+v, want runtime policy family rejection", result)
	}
	if len(client.reqs) != 0 {
		t.Fatalf("client calls = %d, want 0", len(client.reqs))
	}
	if len(progress.batches) != 0 {
		t.Fatalf("progress batches = %+v, want none", progress.batches)
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
		Getenv:  getenvTextAndStreamingEnabled,
	}
	spec := validSpec(t)
	spec.Prompt = "Write a short dashboard sentence."
	spec.ReturnText = true
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

func getenvTextOutputEnabled(key string) string {
	switch key {
	case FlagEnv, TextOutputFlagEnv:
		return "1"
	default:
		return ""
	}
}

func getenvTextAndStreamingEnabled(key string) string {
	switch key {
	case FlagEnv, TextOutputFlagEnv, StreamingFlagEnv:
		return "1"
	default:
		return ""
	}
}

func testSHA256(value string) string {
	sum := sha256.Sum256([]byte(value))
	return "sha256:" + hex.EncodeToString(sum[:])
}
