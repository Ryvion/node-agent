package main

import (
	"context"
	"errors"
	"testing"

	"github.com/Ryvion/ryvion-node/internal/runner"
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

func TestWorkCapsuleDisabledByDefaultIsNotAdvertised(t *testing.T) {
	t.Setenv("RYV_ENABLE_WORK_CAPSULE", "")

	kinds := v7SupportedRunnerKinds(false, true, false)
	if containsString(kinds, executorKindWorkCapsule) {
		t.Fatalf("work_capsule advertised by default: %v", kinds)
	}
}

func TestWorkCapsuleOptInIsAdvertisedWhenGitExists(t *testing.T) {
	if !commandExists("git") {
		t.Skip("git not available in test environment")
	}
	t.Setenv("RYV_ENABLE_WORK_CAPSULE", "1")

	kinds := v7SupportedRunnerKinds(false, true, false)
	if !containsString(kinds, executorKindWorkCapsule) {
		t.Fatalf("work_capsule not advertised after explicit opt-in: %v", kinds)
	}
}

func TestUsesVerifierSessionRunnerImageRecognizesVerifierRuntimeNames(t *testing.T) {
	accepted := []string{
		"ghcr.io/ryvion/ryvion-verifier-sglang:0.1.0",
		"ghcr.io/ryvion/ryvion-verifier-contract-test:0.1.0",
		"registry.example.com/ryvion/runtimes/verifier/sglang:dev",
		"registry.example.com/ryvion/runtimes/verifier/contract-test:dev",
	}
	for _, image := range accepted {
		if !usesVerifierSessionRunnerImage(image) {
			t.Fatalf("usesVerifierSessionRunnerImage(%q) = false, want true", image)
		}
	}

	if usesVerifierSessionRunnerImage("ghcr.io/ryvion/ryvion-draft-small-model:0.1.0") {
		t.Fatal("draft runner image must not use verifier session runner")
	}
}

func containsString(values []string, needle string) bool {
	for _, value := range values {
		if value == needle {
			return true
		}
	}
	return false
}
