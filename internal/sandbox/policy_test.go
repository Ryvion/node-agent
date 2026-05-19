package sandbox

import "testing"

func TestEvaluateSandboxManagedOCIAllowed(t *testing.T) {
	result := EvaluateSandbox(DefaultSandboxPolicy(), SandboxRequest{
		RunnerKind: RunnerKindManagedOCI,
	})

	assertSandboxDecision(t, result, SandboxDecisionAllow)
	assertHasReason(t, result, SandboxReasonManagedOCIAllowed)
}

func TestEvaluateSandboxBlenderRunnerAllowed(t *testing.T) {
	result := EvaluateSandbox(DefaultSandboxPolicy(), SandboxRequest{
		RunnerKind: RunnerKindBlender,
	})

	assertSandboxDecision(t, result, SandboxDecisionAllow)
	assertHasReason(t, result, SandboxReasonBlenderRunnerAllowed)
}

func TestEvaluateSandboxMediaToolRunnerAllowed(t *testing.T) {
	result := EvaluateSandbox(DefaultSandboxPolicy(), SandboxRequest{
		RunnerKind: RunnerKindMediaTool,
	})

	assertSandboxDecision(t, result, SandboxDecisionAllow)
	assertHasReason(t, result, SandboxReasonMediaToolRunnerAllowed)
}

func TestEvaluateSandboxCustomAllowlistedAllowed(t *testing.T) {
	result := EvaluateSandbox(DefaultSandboxPolicy(), SandboxRequest{
		RunnerKind:          RunnerKindCustom,
		IsAllowlistedRunner: true,
	})

	assertSandboxDecision(t, result, SandboxDecisionAllow)
	assertHasReason(t, result, SandboxReasonCustomRunnerAllowlisted)
}

func TestEvaluateSandboxCustomNonAllowlistedRejected(t *testing.T) {
	result := EvaluateSandbox(DefaultSandboxPolicy(), SandboxRequest{
		RunnerKind:      RunnerKindCustom,
		IsTrustedSource: true,
	})

	assertSandboxDecision(t, result, SandboxDecisionReject)
	assertHasReason(t, result, SandboxReasonCustomRunnerNotAllowlisted)
}

func TestEvaluateSandboxTrustedCustomCanBeAllowedByPolicy(t *testing.T) {
	policy := DefaultSandboxPolicy()
	policy.AllowTrustedCustomRunners = true

	result := EvaluateSandbox(policy, SandboxRequest{
		RunnerKind:      RunnerKindCustom,
		IsTrustedSource: true,
	})

	assertSandboxDecision(t, result, SandboxDecisionAllow)
	assertHasReason(t, result, SandboxReasonCustomRunnerTrustedAllowed)
}

func TestEvaluateSandboxNetworkRequirementTriggersIsolation(t *testing.T) {
	result := EvaluateSandbox(DefaultSandboxPolicy(), SandboxRequest{
		RunnerKind:      RunnerKindManagedOCI,
		RequiresNetwork: true,
	})

	assertSandboxDecision(t, result, SandboxDecisionRequireIsolation)
	assertHasReason(t, result, SandboxReasonNetworkRequiresIsolation)
	assertHasReason(t, result, SandboxReasonManagedOCIAllowed)
}

func TestEvaluateSandboxNetworkAllowedForRunner(t *testing.T) {
	policy := DefaultSandboxPolicy()
	policy.NetworkAllowedRunnerKinds = []RunnerKind{RunnerKindMediaTool}

	result := EvaluateSandbox(policy, SandboxRequest{
		RunnerKind:      RunnerKindMediaTool,
		RequiresNetwork: true,
	})

	assertSandboxDecision(t, result, SandboxDecisionAllow)
	assertHasReason(t, result, SandboxReasonNetworkAllowedForRunner)
}

func TestEvaluateSandboxFilesystemWriteRequiresIsolation(t *testing.T) {
	result := EvaluateSandbox(DefaultSandboxPolicy(), SandboxRequest{
		RunnerKind:              RunnerKindManagedOCI,
		RequiresFilesystemWrite: true,
	})

	assertSandboxDecision(t, result, SandboxDecisionRequireIsolation)
	assertHasReason(t, result, SandboxReasonFilesystemWriteRequiresIsolation)
}

func TestEvaluateSandboxReasonCodesIncluded(t *testing.T) {
	result := EvaluateSandbox(DefaultSandboxPolicy(), SandboxRequest{
		RunnerKind:      RunnerKindManagedOCI,
		RequiresNetwork: true,
	})

	if len(result.ReasonCodes) < 2 {
		t.Fatalf("reason codes = %v, want runner and isolation reasons", result.ReasonCodes)
	}
}

func assertSandboxDecision(t *testing.T, result SandboxDecisionResult, want SandboxDecision) {
	t.Helper()

	if result.Decision != want {
		t.Fatalf("decision = %q, want %q; reasons = %v", result.Decision, want, result.ReasonCodes)
	}
}

func assertHasReason(t *testing.T, result SandboxDecisionResult, want SandboxReasonCode) {
	t.Helper()

	for _, reason := range result.ReasonCodes {
		if reason == want {
			return
		}
	}
	t.Fatalf("reason codes = %v, want %q", result.ReasonCodes, want)
}
