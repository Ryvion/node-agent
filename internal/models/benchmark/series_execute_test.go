package modelbench

import (
	"context"
	"encoding/json"
	"testing"
)

func TestExecuteModelBenchmarkSeriesAssignmentFlagOffDoesNotHandle(t *testing.T) {
	runner := &fakeSeriesRunner{
		results: []ModelBenchmarkResult{
			seriesRunResult(ModelBenchmarkProofStatusMeasured, 100, 1_100, 11, 10, ""),
		},
	}

	receipt, handled, err := ExecuteModelBenchmarkSeriesAssignment(context.Background(), validModelBenchmarkSeriesSpecJSON(t), runner, func(string) string {
		return ""
	})
	if err != nil {
		t.Fatalf("ExecuteModelBenchmarkSeriesAssignment() error = %v", err)
	}
	if handled {
		t.Fatal("handled = true, want false when feature flag is off")
	}
	if runner.calls != 0 {
		t.Fatalf("runner calls = %d, want 0", runner.calls)
	}
	if receipt.ResultHashHex != "" {
		t.Fatalf("receipt = %+v, want empty", receipt)
	}
}

func TestExecuteModelBenchmarkSeriesAssignmentValidSpecHandled(t *testing.T) {
	spec := validModelBenchmarkSeriesSpec()
	spec.WarmupRuns = 1
	spec.MeasuredRuns = 2
	runner := &fakeSeriesRunner{
		results: []ModelBenchmarkResult{
			seriesRunResult(ModelBenchmarkProofStatusMeasured, 9_999, 10_999, 99, 99, ""),
			seriesRunResult(ModelBenchmarkProofStatusMeasured, 100, 1_100, 11, 10, ""),
			seriesRunResult(ModelBenchmarkProofStatusMeasured, 200, 1_200, 11, 9.17, ""),
		},
	}

	receipt, handled, err := ExecuteModelBenchmarkSeriesAssignment(context.Background(), mustMarshalModelBenchmarkSeriesAssignment(t, spec), runner, enabledModelBenchmarkEnv)
	if err != nil {
		t.Fatalf("ExecuteModelBenchmarkSeriesAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if runner.calls != 3 {
		t.Fatalf("runner calls = %d, want 3", runner.calls)
	}
	if receipt.JobID != spec.JobID {
		t.Fatalf("job_id = %q, want %q", receipt.JobID, spec.JobID)
	}
	if len(receipt.ResultHashHex) != 64 {
		t.Fatalf("result_hash_hex = %q, want 64 hex chars", receipt.ResultHashHex)
	}
	metadata := receipt.Metadata[ModelBenchmarkSeriesTask].(map[string]any)
	if metadata["proof_status"] != ModelBenchmarkSeriesProofStatusMeasured {
		t.Fatalf("proof_status = %v, want measured series", metadata["proof_status"])
	}
	summary := metadata["summary"].(map[string]any)
	if summary["successful_measured_runs"] != 2 {
		t.Fatalf("successful_measured_runs = %v, want 2", summary["successful_measured_runs"])
	}
	if summary["p50_ttft_ms"] != int64(100) || summary["p95_ttft_ms"] != int64(200) {
		t.Fatalf("summary ttft = %+v, want measured-only 100/200", summary)
	}
}

func TestExecuteModelBenchmarkSeriesAssignmentUnavailableRunnerProducesUnavailableReceipt(t *testing.T) {
	spec := validModelBenchmarkSeriesSpec()
	spec.WarmupRuns = 1
	spec.MeasuredRuns = 2
	runner := &fakeSeriesRunner{
		results: []ModelBenchmarkResult{
			seriesRunResult(ModelBenchmarkProofStatusUnavailable, 0, 25, 0, 0, "native_runtime_unavailable"),
			seriesRunResult(ModelBenchmarkProofStatusUnavailable, 0, 30, 0, 0, "native_runtime_unavailable"),
			seriesRunResult(ModelBenchmarkProofStatusUnavailable, 0, 20, 0, 0, "native_runtime_unavailable"),
		},
	}

	receipt, handled, err := ExecuteModelBenchmarkSeriesAssignment(context.Background(), mustMarshalModelBenchmarkSeriesAssignment(t, spec), runner, enabledModelBenchmarkEnv)
	if err != nil {
		t.Fatalf("ExecuteModelBenchmarkSeriesAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	metadata := receipt.Metadata[ModelBenchmarkSeriesTask].(map[string]any)
	if metadata["proof_status"] != ModelBenchmarkSeriesProofStatusUnavailable {
		t.Fatalf("proof_status = %v, want unavailable series", metadata["proof_status"])
	}
	summary := metadata["summary"].(map[string]any)
	if summary["successful_measured_runs"] != 0 {
		t.Fatalf("successful_measured_runs = %v, want 0", summary["successful_measured_runs"])
	}
	trials := metadata["trials"].([]map[string]any)
	if trials[1]["output_hash"] != "" || trials[1]["output_bytes"] != int64(0) {
		t.Fatalf("unavailable trial exposed output fields: %+v", trials[1])
	}
}

func TestExecuteModelBenchmarkSeriesAssignmentNormalJobDoesNotHandle(t *testing.T) {
	runner := &fakeSeriesRunner{}

	_, handled, err := ExecuteModelBenchmarkSeriesAssignment(context.Background(), `{"task":"native_report","job_id":"job-normal"}`, runner, enabledModelBenchmarkEnv)
	if err != nil {
		t.Fatalf("ExecuteModelBenchmarkSeriesAssignment() error = %v", err)
	}
	if handled {
		t.Fatal("handled = true, want false for normal work")
	}
	if runner.calls != 0 {
		t.Fatalf("runner calls = %d, want 0", runner.calls)
	}
}

func TestModelBenchmarkSeriesAssignmentIdentityFromJSON(t *testing.T) {
	identity, ok := ModelBenchmarkSeriesAssignmentIdentityFromJSON(`{"task":"v7_model_benchmark_series","job_id":" job-1 ","request_id":" request-1 "}`)
	if !ok {
		t.Fatal("ok = false, want true")
	}
	if identity.JobID != "job-1" || identity.RequestID != "request-1" {
		t.Fatalf("identity = %+v", identity)
	}
	if _, ok := ModelBenchmarkSeriesAssignmentIdentityFromJSON(`{"task":"v7_model_benchmark","job_id":"job-model"}`); ok {
		t.Fatal("ok = true, want false for single benchmark task")
	}
}

func validModelBenchmarkSeriesSpecJSON(t *testing.T) string {
	t.Helper()
	return mustMarshalModelBenchmarkSeriesAssignment(t, validModelBenchmarkSeriesSpec())
}

func mustMarshalModelBenchmarkSeriesAssignment(t *testing.T, spec ModelBenchmarkSeriesSpec) string {
	t.Helper()
	encoded, err := json.Marshal(modelBenchmarkSeriesAssignmentSpec{
		Task:            ModelBenchmarkSeriesTask,
		RequestID:       spec.RequestID,
		JobID:           spec.JobID,
		ModelID:         spec.ModelID,
		PromptProfileID: spec.PromptProfileID,
		PromptHash:      spec.PromptHash,
		MaxTokens:       spec.MaxTokens,
		TimeoutMs:       spec.TimeoutMs,
		WarmupRuns:      spec.WarmupRuns,
		MeasuredRuns:    spec.MeasuredRuns,
		CreatedAtUnixMs: spec.CreatedAtUnixMs,
	})
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	return string(encoded)
}
