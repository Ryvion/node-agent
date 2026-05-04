package modelbench

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestExecuteModelBenchmarkAssignmentFlagOffDoesNotHandle(t *testing.T) {
	runner := &fakeAssignmentRunner{result: validMeasuredModelBenchmarkResult()}

	receipt, handled, err := ExecuteModelBenchmarkAssignment(context.Background(), validModelBenchmarkSpecJSON(t), runner, func(string) string {
		return ""
	})
	if err != nil {
		t.Fatalf("ExecuteModelBenchmarkAssignment() error = %v", err)
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

func TestExecuteModelBenchmarkAssignmentValidSpecHandled(t *testing.T) {
	runner := &fakeAssignmentRunner{result: validMeasuredModelBenchmarkResult()}

	receipt, handled, err := ExecuteModelBenchmarkAssignment(context.Background(), validModelBenchmarkSpecJSON(t), runner, enabledModelBenchmarkEnv)
	if err != nil {
		t.Fatalf("ExecuteModelBenchmarkAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if runner.calls != 1 {
		t.Fatalf("runner calls = %d, want 1", runner.calls)
	}
	if receipt.JobID != "job-modelbench-1" {
		t.Fatalf("job_id = %q, want job-modelbench-1", receipt.JobID)
	}
	if receipt.MeteringUnits != 1 {
		t.Fatalf("metering_units = %d, want 1", receipt.MeteringUnits)
	}
	if len(receipt.ResultHashHex) != 64 {
		t.Fatalf("result_hash_hex = %q, want 64 hex chars", receipt.ResultHashHex)
	}
	if _, ok := receipt.Metadata[ModelBenchmarkTask].(map[string]any); !ok {
		t.Fatalf("metadata[%q] type = %T, want map[string]any", ModelBenchmarkTask, receipt.Metadata[ModelBenchmarkTask])
	}
}

func TestExecuteModelBenchmarkAssignmentUnavailableRunnerProducesUnavailableReceipt(t *testing.T) {
	runner := &fakeAssignmentRunner{result: validUnavailableModelBenchmarkResult()}

	receipt, handled, err := ExecuteModelBenchmarkAssignment(context.Background(), validModelBenchmarkSpecJSON(t), runner, enabledModelBenchmarkEnv)
	if err != nil {
		t.Fatalf("ExecuteModelBenchmarkAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	metadata := receipt.Metadata[ModelBenchmarkTask].(map[string]any)
	if metadata["proof_status"] != string(ModelBenchmarkProofStatusUnavailable) {
		t.Fatalf("proof_status = %v, want unavailable", metadata["proof_status"])
	}
	if _, ok := metadata["output_hash"]; ok {
		t.Fatalf("unavailable metadata contains output_hash: %+v", metadata)
	}
	metrics := metadata["metrics"].(map[string]any)
	if metrics["error_code"] != "native_model_unavailable" {
		t.Fatalf("error_code = %v, want native_model_unavailable", metrics["error_code"])
	}
}

func TestExecuteModelBenchmarkAssignmentNonBenchmarkTaskDoesNotHandleNormalWork(t *testing.T) {
	runner := &fakeAssignmentRunner{result: validMeasuredModelBenchmarkResult()}

	_, handled, err := ExecuteModelBenchmarkAssignment(context.Background(), `{"task":"native_report","job_id":"job-normal"}`, runner, enabledModelBenchmarkEnv)
	if err != nil {
		t.Fatalf("ExecuteModelBenchmarkAssignment() error = %v", err)
	}
	if handled {
		t.Fatal("handled = true, want false for normal work")
	}
	if runner.calls != 0 {
		t.Fatalf("runner calls = %d, want 0", runner.calls)
	}
}

func TestExecuteModelBenchmarkAssignmentRejectsInvalidSpec(t *testing.T) {
	spec := validModelBenchmarkSpec()
	spec.PromptHash = ""

	_, handled, err := ExecuteModelBenchmarkAssignment(context.Background(), mustMarshalModelBenchmarkSpec(t, spec), &fakeAssignmentRunner{}, enabledModelBenchmarkEnv)
	if !handled {
		t.Fatal("handled = false, want true for benchmark task")
	}
	if !errors.Is(err, ErrInvalidModelBenchmarkSpec) || !strings.Contains(err.Error(), "prompt_hash") {
		t.Fatalf("error = %v, want invalid prompt_hash", err)
	}
}

func TestModelBenchmarkLocalStatusRecordsLifecycle(t *testing.T) {
	status := NewLocalStatus()
	status.RecordSeen(" job-1 ", " request-1 ")
	status.RecordExecuted("job-1")
	status.RecordReceiptSubmitted("job-1")
	status.RecordReceiptFailed("job-2", errors.New("submit failed\nwith newline"))

	snapshot := status.Snapshot()
	if snapshot.LastSeenBenchmarkJobID != "job-1" || snapshot.LastSeenRequestID != "request-1" {
		t.Fatalf("seen snapshot = %+v", snapshot)
	}
	if snapshot.LastExecutedJobID != "job-1" {
		t.Fatalf("last executed = %q", snapshot.LastExecutedJobID)
	}
	if snapshot.LastReceiptSubmittedJobID != "job-1" {
		t.Fatalf("last receipt submitted = %q", snapshot.LastReceiptSubmittedJobID)
	}
	if snapshot.Counters.Seen != 1 || snapshot.Counters.Executed != 1 || snapshot.Counters.ReceiptSubmitted != 1 || snapshot.Counters.ReceiptFailed != 1 {
		t.Fatalf("counters = %+v", snapshot.Counters)
	}
	if !strings.Contains(snapshot.LastError, "submit failed with newline") {
		t.Fatalf("last_error = %q", snapshot.LastError)
	}
}

func TestModelBenchmarkAssignmentIdentityFromJSON(t *testing.T) {
	identity, ok := ModelBenchmarkAssignmentIdentityFromJSON(`{"task":"v7_model_benchmark","job_id":" job-1 ","request_id":" request-1 "}`)
	if !ok {
		t.Fatal("ok = false, want true")
	}
	if identity.JobID != "job-1" || identity.RequestID != "request-1" {
		t.Fatalf("identity = %+v", identity)
	}
	if _, ok := ModelBenchmarkAssignmentIdentityFromJSON(`{"task":"native_report","job_id":"job-normal"}`); ok {
		t.Fatal("ok = true, want false for normal task")
	}
}

type fakeAssignmentRunner struct {
	result ModelBenchmarkResult
	err    error
	calls  int
}

func (f *fakeAssignmentRunner) RunModelBenchmark(_ context.Context, _ ModelBenchmarkSpec) (ModelBenchmarkResult, error) {
	f.calls++
	return f.result, f.err
}

func validModelBenchmarkSpecJSON(t *testing.T) string {
	t.Helper()
	return mustMarshalModelBenchmarkSpec(t, validModelBenchmarkSpec())
}

func mustMarshalModelBenchmarkSpec(t *testing.T, spec ModelBenchmarkSpec) string {
	t.Helper()
	encoded, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	return string(encoded)
}

func enabledModelBenchmarkEnv(name string) string {
	if name == ModelBenchmarkFlagEnv {
		return "1"
	}
	return ""
}

func validUnavailableModelBenchmarkResult() ModelBenchmarkResult {
	result := validMeasuredModelBenchmarkResult()
	result.RuntimeInfo.NativeInferenceReady = false
	result.RuntimeInfo.ModelLoaded = false
	result.Metrics.TimeToFirstTokenMs = 0
	result.Metrics.TokensGenerated = 0
	result.Metrics.TokensPerSecond = 0
	result.Metrics.ModelLoadState = ModelBenchmarkModelLoadStateUnavailable
	result.Metrics.ErrorCode = "native_model_unavailable"
	result.OutputHash = ""
	result.OutputBytes = 0
	result.ProofStatus = ModelBenchmarkProofStatusUnavailable
	return result
}
