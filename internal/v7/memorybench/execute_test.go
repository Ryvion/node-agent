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
	firstMetadata := first.Metadata[BenchmarkTask].(map[string]any)
	secondMetadata := second.Metadata[BenchmarkTask].(map[string]any)
	if !reflect.DeepEqual(semanticBenchmarkMetadata(firstMetadata), semanticBenchmarkMetadata(secondMetadata)) {
		t.Fatalf("semantic metadata differs\nfirst:  %+v\nsecond: %+v", firstMetadata, secondMetadata)
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
		"node_started_at_unix_ms",
		"node_completed_at_unix_ms",
		"compute_time_ms",
		"compute_time_us",
		"simulated_delay_ms",
		"total_node_wall_time_ms",
		"total_node_wall_time_us",
		"summary_payload_bytes_estimate",
		"output_bytes_estimate",
		"receipt_metadata_json_bytes",
		"receipt_envelope_json_bytes",
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
	if got := benchmarkMetadataInt64(t, metadata, "simulated_delay_ms"); got != 3 {
		t.Fatalf("simulated_delay_ms = %d, want 3", got)
	}
	if got := benchmarkMetadataInt64(t, metadata, "compute_time_us"); got < 0 {
		t.Fatalf("compute_time_us = %d, want non-negative", got)
	}
	if got := benchmarkMetadataInt64(t, metadata, "summary_payload_bytes_estimate"); got <= 0 {
		t.Fatalf("summary_payload_bytes_estimate = %d, want positive", got)
	}
	if got := benchmarkMetadataInt64(t, metadata, "output_bytes_estimate"); got <= 0 {
		t.Fatalf("output_bytes_estimate = %d, want positive", got)
	}
	if got := benchmarkMetadataInt64(t, metadata, "receipt_metadata_json_bytes"); got <= 0 {
		t.Fatalf("receipt_metadata_json_bytes = %d, want positive", got)
	}
	if got := benchmarkMetadataInt64(t, metadata, "receipt_envelope_json_bytes"); got <= 0 {
		t.Fatalf("receipt_envelope_json_bytes = %d, want positive", got)
	}
}

func TestExecuteBenchmarkSpecSeparatesSimulatedDelayFromComputeTime(t *testing.T) {
	spec := validBenchmarkSpec(t)
	spec.SimulatedDelayMs = 250

	receipt, err := ExecuteBenchmarkSpec(context.Background(), spec, ExecuteOptions{})
	if err != nil {
		t.Fatalf("ExecuteBenchmarkSpec() error = %v", err)
	}
	metadata := receipt.Metadata[BenchmarkTask].(map[string]any)

	if got := benchmarkMetadataInt64(t, metadata, "simulated_delay_ms"); got != spec.SimulatedDelayMs {
		t.Fatalf("simulated_delay_ms = %d, want %d", got, spec.SimulatedDelayMs)
	}
	if got := benchmarkMetadataInt64(t, metadata, "compute_time_ms"); got == spec.SimulatedDelayMs {
		t.Fatalf("compute_time_ms = %d, want actual compute time not simulated delay", got)
	}
	if got := benchmarkMetadataInt64(t, metadata, "compute_time_us"); got < 0 {
		t.Fatalf("compute_time_us = %d, want non-negative", got)
	}
	if got := benchmarkMetadataInt64(t, metadata, "total_node_wall_time_ms"); got < spec.SimulatedDelayMs {
		t.Fatalf("total_node_wall_time_ms = %d, want at least simulated delay %d", got, spec.SimulatedDelayMs)
	}
	if got := benchmarkMetadataInt64(t, metadata, "total_node_wall_time_us"); got < spec.SimulatedDelayMs*1000 {
		t.Fatalf("total_node_wall_time_us = %d, want at least simulated delay %d", got, spec.SimulatedDelayMs*1000)
	}
}

func TestBenchmarkReceiptMetadataByteEstimatesMatchJSON(t *testing.T) {
	receipt, handled, err := ExecuteBenchmarkAssignment(context.Background(), validBenchmarkSpecJSON(t), enabledBenchmarkOptions())
	if err != nil || !handled {
		t.Fatalf("ExecuteBenchmarkAssignment() handled=%v error=%v", handled, err)
	}
	metadata := receipt.Metadata[BenchmarkTask].(map[string]any)

	encodedMetadata, err := json.Marshal(metadata)
	if err != nil {
		t.Fatalf("json.Marshal(metadata) error = %v", err)
	}
	if got := benchmarkMetadataInt64(t, metadata, "receipt_metadata_json_bytes"); got != int64(len(encodedMetadata)) {
		t.Fatalf("receipt_metadata_json_bytes = %d, want %d", got, len(encodedMetadata))
	}

	encodedEnvelope, err := json.Marshal(benchmarkReceiptJSONEnvelope{
		JobID:         receipt.JobID,
		ResultHashHex: receipt.ResultHashHex,
		MeteringUnits: receipt.MeteringUnits,
		Metadata: map[string]any{
			BenchmarkTask: metadata,
		},
	})
	if err != nil {
		t.Fatalf("json.Marshal(envelope) error = %v", err)
	}
	if got := benchmarkMetadataInt64(t, metadata, "receipt_envelope_json_bytes"); got != int64(len(encodedEnvelope)) {
		t.Fatalf("receipt_envelope_json_bytes = %d, want %d", got, len(encodedEnvelope))
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
	for _, forbidden := range []string{"prompt", "transcript", "logits", "values", "raw_output", "output_text"} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("metadata contains forbidden raw field %q: %s", forbidden, encoded)
		}
	}
}

func TestBenchmarkResultHashIgnoresTimingMetadata(t *testing.T) {
	spec := validBenchmarkSpec(t)
	spec.SimulatedDelayMs = 0
	request := GenerateSyntheticAttentionRequest(spec.Seed, spec.ShardID, spec.TokenCount, spec.ValueDim)
	request.RequestID = spec.RequestID
	request.JobID = spec.JobID
	request.ShardID = spec.ShardID
	request.CreatedAtUnixMs = spec.CreatedAtUnixMs

	response, err := ComputePartialAttentionSummary(request)
	if err != nil {
		t.Fatalf("ComputePartialAttentionSummary() error = %v", err)
	}
	response.ComputeTimeMs = 1
	response.ComputeTimeUs = 1_500
	response.NodeStartedAtUnixMs = 1_900_000_000_000
	response.NodeCompletedAtUnixMs = 1_900_000_000_001
	response.TotalNodeWallTimeMs = 1
	response.TotalNodeWallTimeUs = 1_500

	first, err := BuildBenchmarkReceipt(spec, response)
	if err != nil {
		t.Fatalf("first BuildBenchmarkReceipt() error = %v", err)
	}

	response.ComputeTimeMs = 7
	response.ComputeTimeUs = 7_500
	response.NodeStartedAtUnixMs = 1_900_000_000_100
	response.NodeCompletedAtUnixMs = 1_900_000_000_108
	response.TotalNodeWallTimeMs = 8
	response.TotalNodeWallTimeUs = 8_000

	second, err := BuildBenchmarkReceipt(spec, response)
	if err != nil {
		t.Fatalf("second BuildBenchmarkReceipt() error = %v", err)
	}
	if first.ResultHashHex != second.ResultHashHex {
		t.Fatalf("result hash changed after timing-only metadata changes\nfirst:  %s\nsecond: %s", first.ResultHashHex, second.ResultHashHex)
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

func benchmarkMetadataInt64(t *testing.T, metadata map[string]any, key string) int64 {
	t.Helper()
	value, ok := metadata[key]
	if !ok {
		t.Fatalf("metadata missing key %q: %+v", key, metadata)
	}
	switch typed := value.(type) {
	case int64:
		return typed
	case int:
		return int64(typed)
	case float64:
		return int64(typed)
	default:
		t.Fatalf("metadata[%q] type = %T, want integer", key, value)
		return 0
	}
}

func semanticBenchmarkMetadata(metadata map[string]any) map[string]any {
	omit := map[string]bool{
		"node_started_at_unix_ms":     true,
		"node_completed_at_unix_ms":   true,
		"compute_time_ms":             true,
		"compute_time_us":             true,
		"total_node_wall_time_ms":     true,
		"total_node_wall_time_us":     true,
		"receipt_metadata_json_bytes": true,
		"receipt_envelope_json_bytes": true,
	}
	semantic := make(map[string]any, len(metadata))
	for key, value := range metadata {
		if omit[key] {
			continue
		}
		semantic[key] = value
	}
	return semantic
}
