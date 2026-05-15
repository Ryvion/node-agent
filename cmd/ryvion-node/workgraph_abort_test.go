package main

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Ryvion/node-agent/internal/hub"
)

type fakeWorkGraphAbortChecker struct {
	calls atomic.Int32
	abort *hub.WorkGraphAbort
}

func (f *fakeWorkGraphAbortChecker) FetchWorkGraphAbort(context.Context, string) (*hub.WorkGraphAbort, error) {
	f.calls.Add(1)
	return f.abort, nil
}

func TestWorkGraphIDForAssignmentReadsTopLevelAndNestedSpec(t *testing.T) {
	if got := workGraphIDForAssignment(&hub.WorkAssignment{WorkGraphID: "wg-field"}); got != "wg-field" {
		t.Fatalf("direct workgraph id = %q, want wg-field", got)
	}
	if got := workGraphIDForAssignment(&hub.WorkAssignment{SpecJSON: `{"workgraph_id":"wg-spec"}`}); got != "wg-spec" {
		t.Fatalf("top-level workgraph id = %q, want wg-spec", got)
	}
	if got := workGraphIDForAssignment(&hub.WorkAssignment{SpecJSON: `{"workgraph":{"graph_id":"wg-nested"}}`}); got != "wg-nested" {
		t.Fatalf("nested workgraph id = %q, want wg-nested", got)
	}
}

func TestWorkGraphAbortMonitorCancelsActiveContext(t *testing.T) {
	parent := context.Background()
	runCtx, cancel := context.WithCancel(parent)
	defer cancel()
	checker := &fakeWorkGraphAbortChecker{abort: &hub.WorkGraphAbort{
		WorkGraphHash: "sha256:test",
		AbortEpoch:    3,
		Reason:        "user_cancel",
	}}
	stop := startWorkGraphAbortMonitorWithInterval(runCtx, checker, &hub.WorkAssignment{
		JobID:    "job-1",
		SpecJSON: `{"workgraph_id":"wg-abort"}`,
	}, cancel, time.Millisecond)
	defer stop()

	select {
	case <-runCtx.Done():
	case <-time.After(time.Second):
		t.Fatal("abort monitor did not cancel active context")
	}
	if checker.calls.Load() == 0 {
		t.Fatal("abort checker was not called")
	}
}

func TestWorkGraphAbortPollIntervalClampsOperatorConfig(t *testing.T) {
	if got := workGraphAbortPollInterval(func(string) string { return "10ms" }); got != 250*time.Millisecond {
		t.Fatalf("short interval clamp = %s, want 250ms", got)
	}
	if got := workGraphAbortPollInterval(func(key string) string {
		if key == "RYV_WORKGRAPH_ABORT_POLL_INTERVAL_MS" {
			return "90000"
		}
		return ""
	}); got != 30*time.Second {
		t.Fatalf("long interval clamp = %s, want 30s", got)
	}
}
