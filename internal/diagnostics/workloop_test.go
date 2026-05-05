package diagnostics

import (
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestWorkLoopDiagnosticsRecordsCounters(t *testing.T) {
	recorder := NewWorkLoopDiagnostics()

	recorder.RecordPollStart()
	recorder.RecordPollEnd(nil)
	recorder.RecordWorkDecode(12 * time.Millisecond)
	recorder.RecordWorkSeen(" job-1 ", " benchmark ", " v7_memory_benchmark ")
	recorder.RecordExecutionStart("job-1")
	recorder.RecordExecutionEnd(34*time.Millisecond, nil)
	recorder.RecordReceiptBuild(3 * time.Millisecond)
	recorder.RecordReceiptSubmitStart("job-1", 2)
	recorder.RecordReceiptSubmitEnd(56*time.Millisecond, nil)

	snapshot := recorder.Snapshot()
	if snapshot.PollCount != 1 {
		t.Fatalf("poll_count = %d, want 1", snapshot.PollCount)
	}
	if snapshot.WorkSeenCount != 1 || snapshot.WorkCompletedCount != 1 || snapshot.ReceiptSubmittedCount != 1 || snapshot.ReceiptFailedCount != 0 {
		t.Fatalf("unexpected counters: %+v", snapshot)
	}
	if snapshot.LastWorkJobID != "job-1" || snapshot.LastWorkKind != "benchmark" || snapshot.LastWorkSpecTask != "v7_memory_benchmark" {
		t.Fatalf("unexpected work identity: %+v", snapshot)
	}
	if snapshot.LastWorkDecodeMs != 12 || snapshot.LastExecutionDurationMs != 34 || snapshot.LastReceiptBuildMs != 3 || snapshot.LastReceiptSubmitDurationMs != 56 {
		t.Fatalf("unexpected timings: %+v", snapshot)
	}
	if snapshot.LastReceiptAttempts != 2 {
		t.Fatalf("last_receipt_attempts = %d, want 2", snapshot.LastReceiptAttempts)
	}
	if snapshot.LastPollStartedAt == "" || snapshot.LastPollCompletedAt == "" || snapshot.LastWorkSeenAt == "" || snapshot.LastExecutionStartedAt == "" || snapshot.LastReceiptSubmitStartedAt == "" {
		t.Fatalf("expected timestamp fields to be populated: %+v", snapshot)
	}
}

func TestWorkLoopDiagnosticsJSONShape(t *testing.T) {
	recorder := NewWorkLoopDiagnostics()
	recorder.RecordPollStart()
	recorder.RecordPollEnd(nil)

	encoded, err := json.Marshal(recorder.Snapshot())
	if err != nil {
		t.Fatalf("json.Marshal(snapshot) error = %v", err)
	}
	text := string(encoded)
	for _, key := range []string{
		"last_poll_started_at",
		"last_work_spec_task",
		"last_receipt_submit_duration_ms",
		"receipt_submitted_count",
	} {
		if !strings.Contains(text, `"`+key+`"`) {
			t.Fatalf("snapshot JSON missing %q: %s", key, text)
		}
	}
}

func TestWorkLoopDiagnosticsSanitizesErrors(t *testing.T) {
	recorder := NewWorkLoopDiagnostics()
	errText := " fetch failed\n\t" + strings.Repeat("x", maxWorkLoopErrorLen+50)

	recorder.RecordPollStart()
	recorder.RecordPollEnd(errors.New(errText))
	recorder.RecordReceiptSubmitStart("job-1", 1)
	recorder.RecordReceiptSubmitEnd(time.Millisecond, errors.New(" submit failed\r\nretry exhausted "))

	snapshot := recorder.Snapshot()
	if strings.ContainsAny(snapshot.LastPollError, "\n\r\t") {
		t.Fatalf("last_poll_error was not whitespace-sanitized: %q", snapshot.LastPollError)
	}
	if len([]rune(snapshot.LastPollError)) > maxWorkLoopErrorLen {
		t.Fatalf("last_poll_error length = %d, want <= %d", len([]rune(snapshot.LastPollError)), maxWorkLoopErrorLen)
	}
	if snapshot.LastReceiptSubmitError != "submit failed retry exhausted" {
		t.Fatalf("last_receipt_submit_error = %q", snapshot.LastReceiptSubmitError)
	}
	if snapshot.ReceiptFailedCount != 1 {
		t.Fatalf("receipt_failed_count = %d, want 1", snapshot.ReceiptFailedCount)
	}
}

func TestWorkLoopDiagnosticsDoesNotExposeRawPayloadFields(t *testing.T) {
	recorder := NewWorkLoopDiagnostics()
	recorder.RecordWorkSeen("job-1", "benchmark", WorkSpecTaskFromJSON(`{"task":"v7_memory_benchmark","prompt":"raw prompt","output":"raw output","weighted_value":[1,2,3]}`))

	encoded, err := json.Marshal(recorder.Snapshot())
	if err != nil {
		t.Fatalf("json.Marshal(snapshot) error = %v", err)
	}
	text := string(encoded)
	for _, forbidden := range []string{"raw prompt", "raw output", "weighted_value"} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("work loop diagnostics leaked forbidden payload material %q: %s", forbidden, text)
		}
	}
	if !strings.Contains(text, `"last_work_spec_task":"v7_memory_benchmark"`) {
		t.Fatalf("snapshot did not keep safe spec task: %s", text)
	}
}

func TestWorkLoopDiagnosticsConcurrentUse(t *testing.T) {
	recorder := NewWorkLoopDiagnostics()

	const goroutines = 8
	const iterations = 50
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				recorder.RecordPollStart()
				recorder.RecordPollEnd(nil)
				recorder.RecordWorkSeen("job", "kind", "task")
				recorder.RecordExecutionStart("job")
				recorder.RecordExecutionEnd(time.Millisecond, nil)
				recorder.RecordReceiptSubmitStart("job", 1)
				recorder.RecordReceiptSubmitEnd(time.Millisecond, nil)
			}
		}()
	}
	wg.Wait()

	snapshot := recorder.Snapshot()
	want := uint64(goroutines * iterations)
	if snapshot.PollCount != want || snapshot.WorkSeenCount != want || snapshot.WorkCompletedCount != want || snapshot.ReceiptSubmittedCount != want {
		t.Fatalf("unexpected concurrent counters: got %+v want %d", snapshot, want)
	}
}

func TestWorkSpecTaskFromJSON(t *testing.T) {
	if got := WorkSpecTaskFromJSON(`{"task":" v7_model_benchmark_series ","prompt":"do not expose"}`); got != "v7_model_benchmark_series" {
		t.Fatalf("WorkSpecTaskFromJSON() = %q, want v7_model_benchmark_series", got)
	}
	if got := WorkSpecTaskFromJSON(`not json`); got != "" {
		t.Fatalf("WorkSpecTaskFromJSON(invalid) = %q, want empty", got)
	}
}
