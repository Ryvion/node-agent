package main

import (
	"context"
	"errors"
	"testing"

	"github.com/Ryvion/node-agent/internal/runner"
)

func TestAnnotateManagedOCIAbortReceiptMarksOrphanedComputeNonBillable(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	metadata := map[string]any{
		"executor": "oci",
	}

	got := annotateManagedOCIAbortReceipt(ctx, metadata, errors.New("context canceled"), nil)

	if got["status"] != "aborted" || got["execution_status"] != "aborted" {
		t.Fatalf("abort status missing: %#v", got)
	}
	if got["billing_status"] != "not_billable_orphaned_compute" {
		t.Fatalf("billing status = %#v, want not_billable_orphaned_compute", got["billing_status"])
	}
	if got["accepted_before_abort"] != false || got["committed_before_abort"] != false {
		t.Fatalf("abort should default to no committed accepted work: %#v", got)
	}
	if got["abort_reason"] == "" {
		t.Fatalf("abort reason missing: %#v", got)
	}
}

func TestManagedOCIExecutionAbortedIgnoresLateContextCancelAfterCompleteReceipt(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	result := &runner.Result{
		ExitCode:        0,
		ReceiptComplete: true,
	}

	if managedOCIExecutionAborted(ctx, nil, result) {
		t.Fatal("complete exit-0 runner receipt must not be reclassified as aborted orphaned compute")
	}
}
