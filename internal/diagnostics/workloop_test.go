package diagnostics

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	v7memorybench "github.com/Ryvion/ryvion-node/internal/v7/memorybench"
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
	recorder.RecordReceiptBuildTimings(ReceiptBuildTimings{
		MetadataStructUs:    1,
		WeightedValueCopyUs: 1,
		MetadataDefaultsUs:  1,
		MetadataValidateUs:  1,
		MetadataTotalUs:     4,
		TotalBuildUs:        4,
	})

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
		"last_receipt_metadata_struct_us",
		"last_receipt_weighted_value_copy_us",
		"last_receipt_metadata_defaults_us",
		"last_receipt_metadata_validate_us",
		"last_receipt_metadata_gap_us",
		"last_receipt_metadata_total_us",
		"last_receipt_hash_us",
		"last_receipt_ready_at",
		"last_receipt_ready_to_submit_ms",
		"last_receipt_ready_to_submit_us",
		"last_receipt_submit_queue_gap_ms",
		"last_receipt_submit_queue_gap_us",
		"last_receipt_submit_duration_ms",
		"last_receipt_submit_duration_us",
		"receipt_submitted_count",
		"recent_events",
	} {
		if !strings.Contains(text, `"`+key+`"`) {
			t.Fatalf("snapshot JSON missing %q: %s", key, text)
		}
	}
}

func TestWorkLoopDiagnosticsSeparatesReceiptReadyToSubmitGap(t *testing.T) {
	recorder := NewWorkLoopDiagnostics()
	recorder.RecordWorkSeen("job-gap", "benchmark", "v7_memory_benchmark")
	recorder.RecordReceiptBuildTimings(ReceiptBuildTimings{
		MetadataStructUs:    10,
		WeightedValueCopyUs: 10,
		MetadataDefaultsUs:  10,
		MetadataValidateUs:  10,
		MetadataTotalUs:     40,
		HashUs:              100,
		JSONMeasureUs:       200,
		EnvelopeBuildUs:     300,
		TotalBuildUs:        1_000,
	})
	recorder.RecordReceiptReady("job-gap", "benchmark", time.Now(), map[string]string{
		"spec_task": "v7_memory_benchmark",
		"prompt":    "raw prompt",
		"output":    "raw output",
	})
	time.Sleep(2 * time.Millisecond)
	recorder.RecordReceiptSubmitStart("job-gap", 1)

	snapshot := recorder.Snapshot()
	if snapshot.LastReceiptBuildMs != 1 || snapshot.LastReceiptTotalBuildMs != 1 || snapshot.LastReceiptTotalBuildUs != 1_000 {
		t.Fatalf("receipt build timing changed by pre-submit gap: %+v", snapshot)
	}
	if snapshot.LastReceiptMetadataGapUs != 0 {
		t.Fatalf("metadata gap = %d us, want only actual metadata substep gap", snapshot.LastReceiptMetadataGapUs)
	}
	if snapshot.LastReceiptReadyAt == "" || snapshot.LastReceiptSubmitStartedAt == "" {
		t.Fatalf("receipt ready/submit timestamps not populated: %+v", snapshot)
	}
	if snapshot.LastReceiptReadyToSubmitUs <= 0 {
		t.Fatalf("ready-to-submit gap = %d us, want positive", snapshot.LastReceiptReadyToSubmitUs)
	}
	if snapshot.LastReceiptSubmitQueueGapUs != snapshot.LastReceiptReadyToSubmitUs || snapshot.LastReceiptSubmitQueueGapMs != snapshot.LastReceiptReadyToSubmitMs {
		t.Fatalf("ready gap aliases differ: %+v", snapshot)
	}

	events := snapshot.RecentEvents
	if countWorkLoopEvents(events, "receipt_ready_to_submit_gap") != 1 || countWorkLoopEvents(events, "receipt_submit_start") != 1 {
		t.Fatalf("unexpected ready/submit events: %+v", events)
	}
	gapEvent := findWorkLoopEvent(events, "receipt_ready_to_submit_gap")
	if gapEvent.SafeContext["gap_us"] == "" || gapEvent.SafeContext["job_id"] != "job-gap" || gapEvent.SafeContext["kind"] != "benchmark" || gapEvent.SafeContext["spec_task"] != "v7_memory_benchmark" {
		t.Fatalf("gap event safe context missing expected fields: %+v", gapEvent.SafeContext)
	}
	encoded, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatalf("json.Marshal(snapshot) error = %v", err)
	}
	if strings.Contains(string(encoded), "raw prompt") || strings.Contains(string(encoded), "raw output") {
		t.Fatalf("ready-to-submit diagnostics leaked raw payload: %s", encoded)
	}
}

func TestWorkLoopDiagnosticsImmediateReceiptSubmitGapIsSmall(t *testing.T) {
	recorder := NewWorkLoopDiagnostics()
	recorder.RecordWorkSeen("job-immediate", "benchmark", "v7_memory_benchmark")
	recorder.RecordReceiptReady("job-immediate", "benchmark", time.Now(), map[string]string{
		"spec_task": "v7_memory_benchmark",
	})
	recorder.RecordReceiptSubmitStart("job-immediate", 1)

	snapshot := recorder.Snapshot()
	if snapshot.LastReceiptReadyToSubmitUs < 0 {
		t.Fatalf("ready-to-submit gap = %d us, want non-negative", snapshot.LastReceiptReadyToSubmitUs)
	}
	if snapshot.LastReceiptReadyToSubmitUs > 100_000 {
		t.Fatalf("ready-to-submit gap = %d us, want near-zero immediate submit", snapshot.LastReceiptReadyToSubmitUs)
	}
}

func TestWorkLoopDiagnosticsReceiptReadyTimestampIsMonotonic(t *testing.T) {
	recorder := NewWorkLoopDiagnostics()
	recorder.RecordEvent("v7_fast_path_receipt_ready", "job-ready", "benchmark", map[string]string{
		"spec_task": "v7_memory_benchmark",
	})
	previous := mustParseWorkLoopTestTime(t, recorder.EventTimeline()[0].At)
	recorder.RecordReceiptReady("job-ready", "benchmark", previous.Add(-time.Second), map[string]string{
		"spec_task": "v7_memory_benchmark",
	})
	recorder.RecordReceiptSubmitStart("job-ready", 1)

	snapshot := recorder.Snapshot()
	events := snapshot.RecentEvents
	if len(events) < 3 {
		t.Fatalf("events = %+v, want receipt ready and submit events", events)
	}
	for i := 1; i < len(events); i++ {
		prev := mustParseWorkLoopTestTime(t, events[i-1].At)
		next := mustParseWorkLoopTestTime(t, events[i].At)
		if next.Before(prev) {
			t.Fatalf("event[%d] time %s before previous %s; events=%+v", i, events[i].At, events[i-1].At, events)
		}
		if events[i].SincePrevUs < 0 {
			t.Fatalf("event[%d].since_prev_us = %d, want non-negative", i, events[i].SincePrevUs)
		}
	}
	if snapshot.LastReceiptReadyToSubmitUs < 0 || snapshot.LastReceiptSubmitQueueGapUs < 0 {
		t.Fatalf("ready-to-submit gap went negative after monotonic clamp: %+v", snapshot)
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

	recorder.RecordReceiptBuildTimings(ReceiptBuildTimings{
		MetadataStructUs:    100,
		WeightedValueCopyUs: 200,
		MetadataDefaultsUs:  300,
		MetadataValidateUs:  400,
		MetadataGapUs:       250,
		MetadataTotalUs:     1_250,
		HashUs:              2_500,
		JSONMeasureUs:       3_750,
		EnvelopeBuildUs:     4_000,
		TotalBuildUs:        11_500,
	})

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
	if snapshot.LastReceiptMetadataStructUs != 100 || snapshot.LastReceiptWeightedValueCopyUs != 200 || snapshot.LastReceiptMetadataDefaultsUs != 300 || snapshot.LastReceiptMetadataValidateUs != 400 || snapshot.LastReceiptMetadataGapUs != 250 || snapshot.LastReceiptMetadataTotalUs != 1_250 {
		t.Fatalf("unexpected receipt metadata substep timings: %+v", snapshot)
	}
	if snapshot.LastReceiptMetadataBuildUs != snapshot.LastReceiptMetadataTotalUs {
		t.Fatalf("metadata build/total aliases = %d/%d, want equal", snapshot.LastReceiptMetadataBuildUs, snapshot.LastReceiptMetadataTotalUs)
	}

	encoded, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatalf("json.Marshal(snapshot) error = %v", err)
	}
	text := string(encoded)
	for _, key := range []string{
		"last_receipt_metadata_build_ms",
		"last_receipt_metadata_struct_us",
		"last_receipt_weighted_value_copy_us",
		"last_receipt_metadata_defaults_us",
		"last_receipt_metadata_validate_us",
		"last_receipt_metadata_gap_us",
		"last_receipt_metadata_total_us",
		"last_receipt_hash_ms",
		"last_receipt_json_measure_ms",
		"last_receipt_envelope_build_ms",
		"last_receipt_total_build_us",
	} {
		if !strings.Contains(text, `"`+key+`"`) {
			t.Fatalf("snapshot JSON missing %q: %s", key, text)
		}
	}
	for _, forbidden := range []string{"raw prompt", "raw output", `"weighted_value":[`} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("receipt timing diagnostics leaked forbidden material %q: %s", forbidden, text)
		}
	}
}

func TestWorkLoopDiagnosticsMetadataGapClampsWhenSubstepsExceedTotal(t *testing.T) {
	recorder := NewWorkLoopDiagnostics()

	recorder.RecordReceiptBuildTimings(ReceiptBuildTimings{
		MetadataStructUs:    7,
		WeightedValueCopyUs: 7,
		MetadataDefaultsUs:  7,
		MetadataValidateUs:  7,
		MetadataTotalUs:     10,
	})

	snapshot := recorder.Snapshot()
	if snapshot.LastReceiptMetadataGapUs != 0 {
		t.Fatalf("metadata gap = %d us, want clamp to 0: %+v", snapshot.LastReceiptMetadataGapUs, snapshot)
	}
	if snapshot.LastReceiptMetadataBuildUs != 10 || snapshot.LastReceiptMetadataTotalUs != 10 {
		t.Fatalf("metadata build/total = %d/%d us, want 10/10", snapshot.LastReceiptMetadataBuildUs, snapshot.LastReceiptMetadataTotalUs)
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
	for _, forbidden := range []string{"raw prompt", "raw output", `"weighted_value":[`} {
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
	if got := len(recorder.EventTimeline()); got != defaultWorkLoopEventLimit {
		t.Fatalf("event timeline length = %d, want capped at %d", got, defaultWorkLoopEventLimit)
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

func TestWorkLoopDiagnosticsEventRingCapsAndOrdersOldestFirst(t *testing.T) {
	recorder := newWorkLoopDiagnostics(3)

	for i := 0; i < 5; i++ {
		recorder.RecordEvent("work_seen", fmt.Sprintf("job-%d", i), "benchmark", map[string]string{
			"spec_task":   "v7_memory_benchmark",
			"token_count": fmt.Sprintf("%d", i+1),
		})
		time.Sleep(time.Millisecond)
	}

	events := recorder.EventTimeline()
	if len(events) != 3 {
		t.Fatalf("event count = %d, want 3: %+v", len(events), events)
	}
	for i, wantJobID := range []string{"job-2", "job-3", "job-4"} {
		if events[i].JobID != wantJobID {
			t.Fatalf("event[%d].job_id = %q, want %q; events=%+v", i, events[i].JobID, wantJobID, events)
		}
		if i > 0 {
			prev := mustParseWorkLoopTestTime(t, events[i-1].At)
			next := mustParseWorkLoopTestTime(t, events[i].At)
			if next.Before(prev) {
				t.Fatalf("events not sorted oldest to newest: %+v", events)
			}
			if events[i].SincePrevUs <= 0 {
				t.Fatalf("event[%d].since_prev_us = %d, want positive", i, events[i].SincePrevUs)
			}
		}
	}
	if events[0].SincePrevUs != 0 {
		t.Fatalf("oldest retained event since_prev_us = %d, want 0", events[0].SincePrevUs)
	}
}

func TestWorkLoopDiagnosticsEventContextSanitized(t *testing.T) {
	recorder := NewWorkLoopDiagnostics()

	recorder.RecordEvent("work_seen", "job-unsafe", "benchmark", map[string]string{
		"spec_task":          "v7_memory_benchmark",
		"token_count":        "64",
		"value_dim":          "not-a-number",
		"metadata_gap_us":    "123",
		"receipt_body_bytes": "456",
		"prompt":             "raw prompt",
		"output":             "raw output",
		"weighted_value":     "[1,2,3]",
	})
	recorder.RecordEvent("prompt_dump", "job-unsafe", "benchmark", map[string]string{
		"spec_task": "v7_memory_benchmark",
	})

	events := recorder.EventTimeline()
	if len(events) != 1 {
		t.Fatalf("event count = %d, want 1 allowed event: %+v", len(events), events)
	}
	context := events[0].SafeContext
	if context["spec_task"] != "v7_memory_benchmark" || context["token_count"] != "64" || context["metadata_gap_us"] != "123" || context["receipt_body_bytes"] != "456" {
		t.Fatalf("safe context missing expected allowed values: %+v", context)
	}
	if _, ok := context["value_dim"]; ok {
		t.Fatalf("non-numeric value_dim was retained: %+v", context)
	}

	encoded, err := json.Marshal(recorder.Snapshot())
	if err != nil {
		t.Fatalf("json.Marshal(snapshot) error = %v", err)
	}
	text := string(encoded)
	for _, forbidden := range []string{"raw prompt", "raw output", "weighted_value", "prompt_dump"} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("event diagnostics leaked forbidden material %q: %s", forbidden, text)
		}
	}
	if !strings.Contains(text, `"recent_events"`) {
		t.Fatalf("snapshot JSON missing recent_events: %s", text)
	}
}

func TestWorkLoopDiagnosticsConcurrentEventWrites(t *testing.T) {
	recorder := newWorkLoopDiagnostics(10)

	const goroutines = 6
	const iterations = 25
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func(worker int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				recorder.RecordEvent("poll_start", fmt.Sprintf("worker-%d-job-%d", worker, j), "benchmark", map[string]string{
					"spec_task": "v7_memory_benchmark",
				})
			}
		}(i)
	}
	wg.Wait()

	events := recorder.EventTimeline()
	if len(events) != 10 {
		t.Fatalf("event count = %d, want 10", len(events))
	}
	for i, event := range events {
		if event.Name != "poll_start" || event.At == "" || event.SafeContext == nil {
			t.Fatalf("event[%d] malformed after concurrent writes: %+v", i, event)
		}
		if i > 0 && event.SincePrevUs < 0 {
			t.Fatalf("event[%d].since_prev_us = %d, want non-negative", i, event.SincePrevUs)
		}
	}
}

func TestMemoryBenchReceiptEmitsSubstepEvents(t *testing.T) {
	spec := v7memorybench.BenchmarkSpec{
		Task:            v7memorybench.BenchmarkTask,
		RequestID:       "request-substeps",
		JobID:           "job-substeps",
		ShardID:         "shard-a",
		Seed:            7,
		TokenCount:      4,
		ValueDim:        2,
		CreatedAtUnixMs: 1_800_000_000_123,
	}
	request := v7memorybench.GenerateSyntheticAttentionRequest(spec.Seed, spec.ShardID, spec.TokenCount, spec.ValueDim)
	request.RequestID = spec.RequestID
	request.JobID = spec.JobID
	request.ShardID = spec.ShardID
	request.CreatedAtUnixMs = spec.CreatedAtUnixMs
	response, err := v7memorybench.ComputePartialAttentionSummary(request)
	if err != nil {
		t.Fatalf("ComputePartialAttentionSummary() error = %v", err)
	}

	recorder := &receiptSubstepCapture{}
	restore := v7memorybench.SetReceiptSubstepEventRecorder(recorder)
	defer restore()

	if _, _, err := v7memorybench.BuildBenchmarkReceiptWithTimings(spec, response); err != nil {
		t.Fatalf("BuildBenchmarkReceiptWithTimings() error = %v", err)
	}

	wantNames := []string{
		"receipt_build_start",
		"receipt_metadata_start",
		"receipt_metadata_struct_end",
		"receipt_weighted_copy_end",
		"receipt_defaults_end",
		"receipt_validate_end",
		"receipt_metadata_end",
		"receipt_hash_end",
		"receipt_json_measure_end",
		"receipt_envelope_end",
		"receipt_build_end",
	}
	if len(recorder.events) != len(wantNames) {
		t.Fatalf("receipt substep event count = %d, want %d: %+v", len(recorder.events), len(wantNames), recorder.events)
	}
	for i, wantName := range wantNames {
		event := recorder.events[i]
		if event.Name != wantName {
			t.Fatalf("event[%d].name = %q, want %q; events=%+v", i, event.Name, wantName, recorder.events)
		}
		if event.JobID != spec.JobID || event.Kind != v7memorybench.BenchmarkTask {
			t.Fatalf("event[%d] identity = %q/%q, want %q/%q", i, event.JobID, event.Kind, spec.JobID, v7memorybench.BenchmarkTask)
		}
	}
	lastContext := recorder.events[len(recorder.events)-1].SafeContext
	if lastContext["metadata_total_us"] == "" || lastContext["metadata_gap_us"] == "" || lastContext["receipt_body_bytes"] == "" {
		t.Fatalf("receipt_build_end context missing timing/body fields: %+v", lastContext)
	}
	encoded, err := json.Marshal(recorder.events)
	if err != nil {
		t.Fatalf("json.Marshal(events) error = %v", err)
	}
	if strings.Contains(string(encoded), `"weighted_value":[`) || strings.Contains(string(encoded), "raw prompt") || strings.Contains(string(encoded), "raw output") {
		t.Fatalf("receipt substep events leaked unsafe fields: %s", encoded)
	}
}

type receiptSubstepCapture struct {
	events []WorkLoopEvent
}

func (c *receiptSubstepCapture) RecordReceiptSubstepEvent(name, jobID, kind string, safeContext map[string]string) {
	context := make(map[string]string, len(safeContext))
	for key, value := range safeContext {
		context[key] = value
	}
	c.events = append(c.events, WorkLoopEvent{
		Name:        name,
		JobID:       jobID,
		Kind:        kind,
		SafeContext: context,
	})
}

func countWorkLoopEvents(events []WorkLoopEvent, name string) int {
	count := 0
	for _, event := range events {
		if event.Name == name {
			count++
		}
	}
	return count
}

func findWorkLoopEvent(events []WorkLoopEvent, name string) WorkLoopEvent {
	for _, event := range events {
		if event.Name == name {
			return event
		}
	}
	return WorkLoopEvent{}
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
