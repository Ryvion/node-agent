package memorybench

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestExecuteBenchmarkAssignmentValidSpecExecutes(t *testing.T) {
	receipt, handled, err := ExecuteBenchmarkAssignment(context.Background(), validBenchmarkSpecJSON(t), enabledBenchmarkOptions())
	if err != nil {
		t.Fatalf("ExecuteBenchmarkAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if receipt.JobID != "job-bench-1" {
		t.Fatalf("job id = %q, want job-bench-1", receipt.JobID)
	}
	if receipt.MeteringUnits != 1 {
		t.Fatalf("metering units = %d, want 1", receipt.MeteringUnits)
	}
	if len(receipt.ResultHashHex) != 64 {
		t.Fatalf("result hash length = %d, want 64", len(receipt.ResultHashHex))
	}
	if _, ok := receipt.Metadata[BenchmarkTask].(map[string]any); !ok {
		t.Fatalf("metadata[%q] type = %T, want map[string]any", BenchmarkTask, receipt.Metadata[BenchmarkTask])
	}
}

func TestExecuteBenchmarkAssignmentRejectsInvalidSpec(t *testing.T) {
	spec := validBenchmarkSpec(t)
	spec.TokenCount = 0
	specJSON := mustMarshalBenchmarkSpec(t, spec)

	_, handled, err := ExecuteBenchmarkAssignment(context.Background(), specJSON, enabledBenchmarkOptions())
	if !handled {
		t.Fatal("handled = false, want true for benchmark task")
	}
	if !errors.Is(err, ErrInvalidBenchmarkSpec) || !strings.Contains(err.Error(), "token_count") {
		t.Fatalf("error = %v, want token_count invalid spec", err)
	}
}

func TestExecuteBenchmarkAssignmentFlagOffDoesNotExecuteBenchmarkPath(t *testing.T) {
	_, handled, err := ExecuteBenchmarkAssignment(context.Background(), validBenchmarkSpecJSON(t), ExecuteOptions{
		Getenv: func(string) string { return "" },
		Sleep:  noSleep,
	})
	if handled {
		t.Fatal("handled = true, want false when feature flag is off")
	}
	if err != nil {
		t.Fatalf("error = %v, want nil when feature flag is off", err)
	}
}

func TestExecuteBenchmarkAssignmentNonBenchmarkTaskDoesNotHandleNormalWork(t *testing.T) {
	_, handled, err := ExecuteBenchmarkAssignment(context.Background(), `{"task":"native_report","job_id":"job-normal"}`, enabledBenchmarkOptions())
	if err != nil {
		t.Fatalf("ExecuteBenchmarkAssignment() error = %v", err)
	}
	if handled {
		t.Fatal("handled = true, want false for non-benchmark task")
	}
}

func TestExecuteBenchmarkAssignmentDeterministicForSameSeed(t *testing.T) {
	specJSON := validBenchmarkSpecJSON(t)
	first, handled, err := ExecuteBenchmarkAssignment(context.Background(), specJSON, enabledBenchmarkOptions())
	if err != nil || !handled {
		t.Fatalf("first ExecuteBenchmarkAssignment() handled=%v error=%v", handled, err)
	}
	second, handled, err := ExecuteBenchmarkAssignment(context.Background(), specJSON, enabledBenchmarkOptions())
	if err != nil || !handled {
		t.Fatalf("second ExecuteBenchmarkAssignment() handled=%v error=%v", handled, err)
	}
	if first.ResultHashHex != second.ResultHashHex {
		t.Fatalf("result hashes differ\nfirst:  %s\nsecond: %s", first.ResultHashHex, second.ResultHashHex)
	}
	if !reflect.DeepEqual(first.Metadata, second.Metadata) {
		t.Fatalf("metadata differs\nfirst:  %+v\nsecond: %+v", first.Metadata, second.Metadata)
	}
}

func TestBenchmarkReceiptMetadataContainsSummary(t *testing.T) {
	receipt, handled, err := ExecuteBenchmarkAssignment(context.Background(), validBenchmarkSpecJSON(t), enabledBenchmarkOptions())
	if err != nil || !handled {
		t.Fatalf("ExecuteBenchmarkAssignment() handled=%v error=%v", handled, err)
	}
	metadata := receipt.Metadata[BenchmarkTask].(map[string]any)
	for _, key := range []string{
		"request_id",
		"shard_id",
		"local_max",
		"exp_sum",
		"weighted_value",
		"token_count",
		"value_dim",
		"compute_time_ms",
		"output_bytes_estimate",
		"proof_status",
	} {
		if _, ok := metadata[key]; !ok {
			t.Fatalf("metadata missing key %q: %+v", key, metadata)
		}
	}
	if metadata["proof_status"] != "synthetic_measured" {
		t.Fatalf("proof_status = %v, want synthetic_measured", metadata["proof_status"])
	}
	if metadata["token_count"] != 12 {
		t.Fatalf("token_count = %v, want 12", metadata["token_count"])
	}
	if metadata["value_dim"] != 5 {
		t.Fatalf("value_dim = %v, want 5", metadata["value_dim"])
	}
	if weighted, ok := metadata["weighted_value"].([]float64); !ok || len(weighted) != 5 {
		t.Fatalf("weighted_value = %T %+v, want []float64 length 5", metadata["weighted_value"], metadata["weighted_value"])
	}
}

func TestBenchmarkReceiptMetadataContainsNoRawPromptTranscriptOrTensorInputs(t *testing.T) {
	receipt, handled, err := ExecuteBenchmarkAssignment(context.Background(), validBenchmarkSpecJSON(t), enabledBenchmarkOptions())
	if err != nil || !handled {
		t.Fatalf("ExecuteBenchmarkAssignment() handled=%v error=%v", handled, err)
	}
	encoded, err := json.Marshal(receipt.Metadata)
	if err != nil {
		t.Fatalf("json.Marshal(metadata) error = %v", err)
	}
	text := strings.ToLower(string(encoded))
	for _, forbidden := range []string{"prompt", "transcript", "logits", "values"} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("metadata contains forbidden raw field %q: %s", forbidden, encoded)
		}
	}
}

func validBenchmarkSpec(t *testing.T) BenchmarkSpec {
	t.Helper()
	return BenchmarkSpec{
		Task:             BenchmarkTask,
		RequestID:        "request-bench-1",
		JobID:            "job-bench-1",
		ShardID:          "shard-a",
		Seed:             99,
		TokenCount:       12,
		ValueDim:         5,
		SimulatedDelayMs: 3,
		CreatedAtUnixMs:  1_800_000_000_321,
	}
}

func validBenchmarkSpecJSON(t *testing.T) string {
	t.Helper()
	return mustMarshalBenchmarkSpec(t, validBenchmarkSpec(t))
}

func mustMarshalBenchmarkSpec(t *testing.T, spec BenchmarkSpec) string {
	t.Helper()
	encoded, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	return string(encoded)
}

func enabledBenchmarkOptions() ExecuteOptions {
	return ExecuteOptions{
		Getenv: func(name string) string {
			if name == BenchmarkFlagEnv {
				return "1"
			}
			return ""
		},
		Sleep: noSleep,
	}
}

func noSleep(context.Context, time.Duration) error {
	return nil
}
