package memorybench

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestLocalStatusRecordsSeenExecutedSubmitted(t *testing.T) {
	status := NewLocalStatus()

	status.RecordSeen(" job-bench-1 ", " request-bench-1 ")
	status.RecordExecuted("job-bench-1")
	status.RecordReceiptSubmitted("job-bench-1")

	snapshot := status.Snapshot()
	if snapshot.LastSeenBenchmarkJobID != "job-bench-1" {
		t.Fatalf("last seen job id = %q, want job-bench-1", snapshot.LastSeenBenchmarkJobID)
	}
	if snapshot.LastSeenRequestID != "request-bench-1" {
		t.Fatalf("last seen request id = %q, want request-bench-1", snapshot.LastSeenRequestID)
	}
	if snapshot.LastSeenAt == nil {
		t.Fatal("last_seen_at is nil")
	}
	if snapshot.LastExecutedJobID != "job-bench-1" {
		t.Fatalf("last executed job id = %q, want job-bench-1", snapshot.LastExecutedJobID)
	}
	if snapshot.LastExecutedAt == nil {
		t.Fatal("last_executed_at is nil")
	}
	if snapshot.LastReceiptSubmittedJobID != "job-bench-1" {
		t.Fatalf("last receipt submitted job id = %q, want job-bench-1", snapshot.LastReceiptSubmittedJobID)
	}
	if snapshot.LastReceiptSubmittedAt == nil {
		t.Fatal("last_receipt_submitted_at is nil")
	}
	if snapshot.LastError != "" {
		t.Fatalf("last error = %q, want empty", snapshot.LastError)
	}
	if snapshot.Counters.Seen != 1 || snapshot.Counters.Executed != 1 || snapshot.Counters.ReceiptSubmitted != 1 {
		t.Fatalf("unexpected counters: %+v", snapshot.Counters)
	}
	if snapshot.Counters.ReceiptFailed != 0 || snapshot.Counters.Rejected != 0 {
		t.Fatalf("unexpected failure counters: %+v", snapshot.Counters)
	}
}

func TestLocalStatusRecordsError(t *testing.T) {
	status := NewLocalStatus()

	status.RecordSeen("job-bench-1", "request-bench-1")
	status.RecordReceiptFailed("job-bench-1", errors.New(" receipt submit failed\nretry exhausted "))

	snapshot := status.Snapshot()
	if snapshot.Counters.ReceiptFailed != 1 {
		t.Fatalf("receipt_failed counter = %d, want 1", snapshot.Counters.ReceiptFailed)
	}
	if snapshot.LastError != "receipt submit failed retry exhausted" {
		t.Fatalf("last error = %q", snapshot.LastError)
	}

	status.RecordSeen("job-bench-2", "request-bench-2")
	status.RecordRejected("job-bench-2", errors.New("memorybench: invalid benchmark spec: token_count must be greater than zero"))

	snapshot = status.Snapshot()
	if snapshot.Counters.Seen != 2 || snapshot.Counters.Rejected != 1 {
		t.Fatalf("unexpected counters after rejection: %+v", snapshot.Counters)
	}
	if !strings.Contains(snapshot.LastError, "token_count") {
		t.Fatalf("last error = %q, want token_count detail", snapshot.LastError)
	}
}

func TestLocalStatusJSONDoesNotIncludeWeightedValue(t *testing.T) {
	status := NewLocalStatus()
	status.RecordSeen("job-bench-1", "request-bench-1")
	status.RecordExecuted("job-bench-1")
	status.RecordReceiptSubmitted("job-bench-1")

	encoded, err := json.Marshal(status.Snapshot())
	if err != nil {
		t.Fatalf("json.Marshal(snapshot) error = %v", err)
	}
	if strings.Contains(string(encoded), "weighted_value") {
		t.Fatalf("local status includes weighted_value: %s", encoded)
	}
}

func TestSanitizeLocalReceiptMetadataRemovesWeightedValue(t *testing.T) {
	metadata := map[string]any{
		"executor": BenchmarkTask,
		BenchmarkTask: map[string]any{
			"request_id":     "request-bench-1",
			"weighted_value": []float64{1, 2, 3},
			"proof_status":   "synthetic_measured",
		},
	}

	sanitized := SanitizeLocalReceiptMetadata(metadata)
	benchmark, ok := sanitized[BenchmarkTask].(map[string]any)
	if !ok {
		t.Fatalf("benchmark metadata type = %T", sanitized[BenchmarkTask])
	}
	if _, ok := benchmark["weighted_value"]; ok {
		t.Fatalf("sanitized metadata still contains weighted_value: %+v", benchmark)
	}
	if benchmark["request_id"] != "request-bench-1" || benchmark["proof_status"] != "synthetic_measured" {
		t.Fatalf("sanitized metadata lost small fields: %+v", benchmark)
	}
	if _, ok := metadata[BenchmarkTask].(map[string]any)["weighted_value"]; !ok {
		t.Fatalf("source metadata was mutated: %+v", metadata)
	}
}

func TestBenchmarkAssignmentIdentityFromJSON(t *testing.T) {
	identity, ok := BenchmarkAssignmentIdentityFromJSON(`{"task":"v7_memory_benchmark","job_id":" job-1 ","request_id":" request-1 ","weighted_value":[1,2,3]}`)
	if !ok {
		t.Fatal("ok = false, want true")
	}
	if identity.JobID != "job-1" || identity.RequestID != "request-1" {
		t.Fatalf("identity = %+v", identity)
	}

	if _, ok := BenchmarkAssignmentIdentityFromJSON(`{"task":"native_report","job_id":"job-normal"}`); ok {
		t.Fatal("normal job returned benchmark identity")
	}
}
