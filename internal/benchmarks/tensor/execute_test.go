package tensorplane

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestExecuteBenchmarkAssignmentFlagOffDoesNotHandle(t *testing.T) {
	_, handled, err := ExecuteBenchmarkAssignment(context.Background(), validTensorPlaneBenchmarkSpecJSON(t), ExecuteOptions{
		Getenv: func(string) string { return "" },
	})
	if err != nil {
		t.Fatalf("ExecuteBenchmarkAssignment() error = %v", err)
	}
	if handled {
		t.Fatal("handled = true, want false when feature flag is off")
	}
}

func TestExecuteBenchmarkAssignmentValidSpecExecutes(t *testing.T) {
	receipt, handled, err := ExecuteBenchmarkAssignment(context.Background(), validTensorPlaneBenchmarkSpecJSON(t), enabledTensorPlaneBenchmarkOptions())
	if err != nil {
		t.Fatalf("ExecuteBenchmarkAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if receipt.JobID != "job-tensorplane-1" {
		t.Fatalf("job id = %q, want job-tensorplane-1", receipt.JobID)
	}
	if receipt.MeteringUnits != 1 {
		t.Fatalf("metering units = %d, want 1", receipt.MeteringUnits)
	}
	if len(receipt.ResultHashHex) != 64 {
		t.Fatalf("result hash length = %d, want 64", len(receipt.ResultHashHex))
	}
	metadata, ok := receipt.Metadata[BenchmarkTask].(map[string]any)
	if !ok {
		t.Fatalf("metadata[%q] type = %T, want map[string]any", BenchmarkTask, receipt.Metadata[BenchmarkTask])
	}
	if metadata["proof_status"] != ProofStatusTensorPlaneMeasured {
		t.Fatalf("proof_status = %v", metadata["proof_status"])
	}
	if metadata["correctness_status"] != CorrectnessStatusMatched {
		t.Fatalf("correctness_status = %v", metadata["correctness_status"])
	}
}

func TestRunBenchmarkSpecCorrectnessMatchedForSmallFixture(t *testing.T) {
	spec := validTensorPlaneBenchmarkSpec()
	spec.Tokens = 4
	spec.HeadDim = 2
	spec.ValueDim = 2

	result, err := RunBenchmarkSpec(context.Background(), spec)
	if err != nil {
		t.Fatalf("RunBenchmarkSpec() error = %v", err)
	}
	if result.CorrectnessStatus != CorrectnessStatusMatched {
		t.Fatalf("correctness_status = %q", result.CorrectnessStatus)
	}
	if result.MaxAbsDiffVsReference > TensorPlaneProbeTolerance(spec.DType) {
		t.Fatalf("max_abs_diff_vs_reference = %.17g", result.MaxAbsDiffVsReference)
	}
	if result.Summary.SummaryHash == "" || result.Summary.PageHash == "" || result.QueryHash == "" {
		t.Fatalf("missing hashes: summary=%q page=%q query=%q", result.Summary.SummaryHash, result.Summary.PageHash, result.QueryHash)
	}
}

func TestDecodeBenchmarkSpecRejectsInvalidDType(t *testing.T) {
	spec := validTensorPlaneBenchmarkSpec()
	spec.DType = TensorDType("int8")

	_, err := DecodeBenchmarkSpec(mustMarshalTensorPlaneBenchmarkSpec(t, spec))
	if !errors.Is(err, ErrInvalidTensorDType) {
		t.Fatalf("DecodeBenchmarkSpec() error = %v, want invalid dtype", err)
	}
}

func TestExecuteBenchmarkAssignmentResultHashDeterministic(t *testing.T) {
	specJSON := validTensorPlaneBenchmarkSpecJSON(t)
	first, handled, err := ExecuteBenchmarkAssignment(context.Background(), specJSON, enabledTensorPlaneBenchmarkOptions())
	if err != nil || !handled {
		t.Fatalf("first ExecuteBenchmarkAssignment() handled=%v error=%v", handled, err)
	}
	second, handled, err := ExecuteBenchmarkAssignment(context.Background(), specJSON, enabledTensorPlaneBenchmarkOptions())
	if err != nil || !handled {
		t.Fatalf("second ExecuteBenchmarkAssignment() handled=%v error=%v", handled, err)
	}
	if first.ResultHashHex != second.ResultHashHex {
		t.Fatalf("result hashes differ\nfirst:  %s\nsecond: %s", first.ResultHashHex, second.ResultHashHex)
	}
	firstMetadata := first.Metadata[BenchmarkTask].(map[string]any)
	secondMetadata := second.Metadata[BenchmarkTask].(map[string]any)
	for _, key := range []string{"page_hash", "query_hash", "summary_hash", "local_max", "exp_sum", "weighted_value_length", "correctness_status", "proof_status"} {
		if firstMetadata[key] != secondMetadata[key] {
			t.Fatalf("metadata[%q] differs: %v vs %v", key, firstMetadata[key], secondMetadata[key])
		}
	}
}

func TestBenchmarkAssignmentIdentityFromJSON(t *testing.T) {
	identity, ok := BenchmarkAssignmentIdentityFromJSON(`{"task":"v7_tensorplane_benchmark","job_id":" job-1 ","request_id":" req-1 ","query_vector":[1,2,3]}`)
	if !ok {
		t.Fatal("ok = false, want true")
	}
	if identity.JobID != "job-1" || identity.RequestID != "req-1" {
		t.Fatalf("identity = %+v", identity)
	}
	if _, ok := BenchmarkAssignmentIdentityFromJSON(`{"task":"v7_memory_benchmark","job_id":"job-1"}`); ok {
		t.Fatal("ok = true for non-tensorplane task")
	}
}

func validTensorPlaneBenchmarkSpec() BenchmarkSpec {
	return BenchmarkSpec{
		Task:            BenchmarkTask,
		RequestID:       "request-tensorplane-1",
		JobID:           "job-tensorplane-1",
		ModelID:         "model-fixture",
		LayerIndex:      2,
		DType:           TensorDTypeFloat32,
		Tokens:          8,
		HeadDim:         4,
		ValueDim:        3,
		Seed:            99,
		TimeoutMs:       5_000,
		CreatedAtUnixMs: 1_800_000_000_123,
	}
}

func validTensorPlaneBenchmarkSpecJSON(t *testing.T) string {
	t.Helper()
	return mustMarshalTensorPlaneBenchmarkSpec(t, validTensorPlaneBenchmarkSpec())
}

func mustMarshalTensorPlaneBenchmarkSpec(t *testing.T, spec BenchmarkSpec) string {
	t.Helper()
	encoded, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	return string(encoded)
}

func enabledTensorPlaneBenchmarkOptions() ExecuteOptions {
	return ExecuteOptions{
		Getenv: func(key string) string {
			if strings.TrimSpace(key) == BenchmarkFlagEnv {
				return "1"
			}
			return ""
		},
	}
}
