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
	if snapshot.LastExecutionDurationUs != 34_000 || snapshot.LastReceiptSubmitDurationUs != 56_000 {
		t.Fatalf("unexpected microsecond timings: %+v", snapshot)
	}
	if snapshot.LastReceiptTotalBuildMs != 3 || snapshot.LastReceiptEnvelopeBuildMs != 3 || snapshot.LastReceiptTotalBuildUs != 3_000 || snapshot.LastReceiptEnvelopeBuildUs != 3_000 {
		t.Fatalf("unexpected receipt build aliases: %+v", snapshot)
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
		"last_poll_cycle_duration_ms",
		"last_work_spec_task",
		"last_receipt_total_build_ms",
		"last_receipt_hash_us",
		"last_receipt_submit_duration_ms",
		"last_receipt_submit_duration_us",
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

func TestWorkLoopDiagnosticsPollTimestampOrdering(t *testing.T) {
	recorder := NewWorkLoopDiagnostics()

	recorder.RecordPollStart()
	inFlight := recorder.Snapshot()
	assertPollSnapshotOrdered(t, inFlight)
	if inFlight.LastPollDurationMs != 0 || inFlight.LastPollCycleDurationMs != 0 {
		t.Fatalf("in-flight poll duration = %d/%d, want 0", inFlight.LastPollDurationMs, inFlight.LastPollCycleDurationMs)
	}

	time.Sleep(3 * time.Millisecond)
	recorder.RecordPollEnd(nil)

	snapshot := recorder.Snapshot()
	assertPollSnapshotOrdered(t, snapshot)
	startedAt := mustParseWorkLoopTestTime(t, snapshot.LastPollStartedAt)
	completedAt := mustParseWorkLoopTestTime(t, snapshot.LastPollCompletedAt)
	wantDurationMs := durationMilliseconds(completedAt.Sub(startedAt))
	if snapshot.LastPollDurationMs != wantDurationMs {
		t.Fatalf("last_poll_duration_ms = %d, want %d from timestamp pair", snapshot.LastPollDurationMs, wantDurationMs)
	}
	if snapshot.LastPollCycleDurationMs != snapshot.LastPollDurationMs {
		t.Fatalf("last_poll_cycle_duration_ms = %d, want last_poll_duration_ms %d", snapshot.LastPollCycleDurationMs, snapshot.LastPollDurationMs)
	}
}

func TestWorkLoopDiagnosticsRecordsReceiptBuildTimings(t *testing.T) {
	recorder := NewWorkLoopDiagnostics()

	recorder.RecordReceiptBuildTimings(ReceiptBuildTimingsFromMicroseconds(1_250, 2_500, 3_750, 4_000, 11_500))

	snapshot := recorder.Snapshot()
	if snapshot.LastReceiptBuildMs != 11 || snapshot.LastReceiptTotalBuildMs != 11 {
		t.Fatalf("receipt total ms aliases = %d/%d, want 11", snapshot.LastReceiptBuildMs, snapshot.LastReceiptTotalBuildMs)
	}
	if snapshot.LastReceiptMetadataBuildMs != 1 || snapshot.LastReceiptHashMs != 2 || snapshot.LastReceiptJSONMeasureMs != 3 || snapshot.LastReceiptEnvelopeBuildMs != 4 {
		t.Fatalf("unexpected receipt split ms timings: %+v", snapshot)
	}
	if snapshot.LastReceiptMetadataBuildUs != 1_250 || snapshot.LastReceiptHashUs != 2_500 || snapshot.LastReceiptJSONMeasureUs != 3_750 || snapshot.LastReceiptEnvelopeBuildUs != 4_000 || snapshot.LastReceiptTotalBuildUs != 11_500 {
		t.Fatalf("unexpected receipt split us timings: %+v", snapshot)
	}

	encoded, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatalf("json.Marshal(snapshot) error = %v", err)
	}
	text := string(encoded)
	for _, key := range []string{
		"last_receipt_metadata_build_ms",
		"last_receipt_hash_ms",
		"last_receipt_json_measure_ms",
		"last_receipt_envelope_build_ms",
		"last_receipt_total_build_us",
	} {
		if !strings.Contains(text, `"`+key+`"`) {
			t.Fatalf("snapshot JSON missing %q: %s", key, text)
		}
	}
	for _, forbidden := range []string{"raw prompt", "raw output", "weighted_value"} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("receipt timing diagnostics leaked forbidden material %q: %s", forbidden, text)
		}
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
				recorder.RecordReceiptBuildTimings(ReceiptBuildTimingsFromMicroseconds(1, 1, 1, 1, 4))
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
	if snapshot.LastReceiptBuildMs != snapshot.LastReceiptTotalBuildMs || snapshot.LastReceiptTotalBuildUs < 0 {
		t.Fatalf("unexpected concurrent receipt timings: %+v", snapshot)
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

func assertPollSnapshotOrdered(t *testing.T, snapshot WorkLoopSnapshot) {
	t.Helper()
	if snapshot.LastPollStartedAt == "" || snapshot.LastPollCompletedAt == "" {
		t.Fatalf("poll timestamps empty: %+v", snapshot)
	}
	startedAt := mustParseWorkLoopTestTime(t, snapshot.LastPollStartedAt)
	completedAt := mustParseWorkLoopTestTime(t, snapshot.LastPollCompletedAt)
	if completedAt.Before(startedAt) {
		t.Fatalf("last_poll_completed_at %s before last_poll_started_at %s", snapshot.LastPollCompletedAt, snapshot.LastPollStartedAt)
	}
	if snapshot.LastPollDurationMs < 0 {
		t.Fatalf("last_poll_duration_ms = %d, want non-negative", snapshot.LastPollDurationMs)
	}
}

func mustParseWorkLoopTestTime(t *testing.T, value string) time.Time {
	t.Helper()
	parsed, err := time.Parse(time.RFC3339Nano, value)
	if err != nil {
		t.Fatalf("time.Parse(%q) error = %v", value, err)
	}
	return parsed
}
