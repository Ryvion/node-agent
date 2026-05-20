package sandbox

import "testing"

func TestEvaluateSandboxGGUFNativeLlamaAllowed(t *testing.T) {
	result := EvaluateSandbox(DefaultSandboxPolicy(), SandboxRequest{
		ModelPath:  "llama.gguf",
		RunnerKind: RunnerKindNativeLlama,
	})

	assertSandboxDecision(t, result, SandboxDecisionAllow)
	assertHasReason(t, result, SandboxReasonGGUFNativeLlamaAllowed)
}

func TestEvaluateSandboxSafetensorsAllowlistedAllowed(t *testing.T) {
	result := EvaluateSandbox(DefaultSandboxPolicy(), SandboxRequest{
		ModelPath:           "model.safetensors",
		RunnerKind:          RunnerKindRyvionRuntime,
		IsAllowlistedRunner: true,
	})

	assertSandboxDecision(t, result, SandboxDecisionAllow)
	assertHasReason(t, result, SandboxReasonSafetensorsAllowlistedRunnerAllowed)
}

func TestEvaluateSandboxPyTorchPickleRejected(t *testing.T) {
	result := EvaluateSandbox(DefaultSandboxPolicy(), SandboxRequest{
		ModelPath:           "checkpoint.pt",
		RunnerKind:          RunnerKindManagedOCI,
		IsAllowlistedRunner: true,
	})

	assertSandboxDecision(t, result, SandboxDecisionReject)
	assertHasReason(t, result, SandboxReasonPyTorchPickleRejected)
}

func TestEvaluateSandboxPythonSourceRequiresIsolationByDefault(t *testing.T) {
	result := EvaluateSandbox(DefaultSandboxPolicy(), SandboxRequest{
		ModelPath:           "model.py",
		RunnerKind:          RunnerKindManagedOCI,
		IsAllowlistedRunner: true,
	})

	assertSandboxDecision(t, result, SandboxDecisionRequireIsolation)
	assertHasReason(t, result, SandboxReasonPythonSourceRequiresIsolation)
}

func TestEvaluateSandboxPythonSourceCanBeRejectedByPolicy(t *testing.T) {
	policy := DefaultSandboxPolicy()
	policy.PythonSourceDecision = SandboxDecisionReject

	result := EvaluateSandbox(policy, SandboxRequest{
		ModelPath:           "model.py",
		RunnerKind:          RunnerKindManagedOCI,
		IsAllowlistedRunner: true,
	})

	assertSandboxDecision(t, result, SandboxDecisionReject)
	assertHasReason(t, result, SandboxReasonPythonSourceRejected)
}

func TestEvaluateSandboxUnknownRejected(t *testing.T) {
	result := EvaluateSandbox(DefaultSandboxPolicy(), SandboxRequest{
		ModelPath:           "model.weights",
		RunnerKind:          RunnerKindRyvionRuntime,
		IsAllowlistedRunner: true,
		IsTrustedSource:     true,
	})

	assertSandboxDecision(t, result, SandboxDecisionReject)
	assertHasReason(t, result, SandboxReasonUnknownFormatRejected)
}

func TestEvaluateSandboxCustomNonAllowlistedRejected(t *testing.T) {
	result := EvaluateSandbox(DefaultSandboxPolicy(), SandboxRequest{
		ModelPath:       "model.safetensors",
		RunnerKind:      RunnerKindCustom,
		IsTrustedSource: true,
	})

	assertSandboxDecision(t, result, SandboxDecisionReject)
	assertHasReason(t, result, SandboxReasonCustomRunnerNotAllowlisted)
}

func TestEvaluateSandboxNetworkRequirementTriggersIsolation(t *testing.T) {
	result := EvaluateSandbox(DefaultSandboxPolicy(), SandboxRequest{
		ModelPath:           "model.safetensors",
		RunnerKind:          RunnerKindRyvionRuntime,
		IsAllowlistedRunner: true,
		RequiresNetwork:     true,
	})

	assertSandboxDecision(t, result, SandboxDecisionRequireIsolation)
	assertHasReason(t, result, SandboxReasonNetworkRequiresIsolation)
	assertHasReason(t, result, SandboxReasonSafetensorsAllowlistedRunnerAllowed)
}

func TestEvaluateSandboxNetworkAllowedForAgentHosting(t *testing.T) {
	result := EvaluateSandbox(DefaultSandboxPolicy(), SandboxRequest{
		ModelPath:           "model.safetensors",
		RunnerKind:          RunnerKindAgentHosting,
		IsAllowlistedRunner: true,
		RequiresNetwork:     true,
	})

	assertSandboxDecision(t, result, SandboxDecisionAllow)
	assertHasReason(t, result, SandboxReasonNetworkAllowedForRunner)
}

func TestEvaluateSandboxTrustedSourceAloneDoesNotAllowPickle(t *testing.T) {
	result := EvaluateSandbox(DefaultSandboxPolicy(), SandboxRequest{
		ModelPath:           "trusted.pt",
		RunnerKind:          RunnerKindManagedOCI,
		IsAllowlistedRunner: true,
		IsTrustedSource:     true,
	})

	assertSandboxDecision(t, result, SandboxDecisionReject)
	assertHasReason(t, result, SandboxReasonPyTorchPickleRejected)
}

func TestEvaluateSandboxPolicyCanRequireIsolationForTrustedAllowlistedPickle(t *testing.T) {
	policy := DefaultSandboxPolicy()
	policy.AllowPyTorchPickleTrustedAllowlisted = true

	result := EvaluateSandbox(policy, SandboxRequest{
		ModelPath:           "trusted.pt",
		RunnerKind:          RunnerKindManagedOCI,
		IsAllowlistedRunner: true,
		IsTrustedSource:     true,
	})

	assertSandboxDecision(t, result, SandboxDecisionRequireIsolation)
	assertHasReason(t, result, SandboxReasonPyTorchPickleRequiresIsolation)
}

func TestEvaluateSandboxReasonCodesIncluded(t *testing.T) {
	result := EvaluateSandbox(DefaultSandboxPolicy(), SandboxRequest{
		ModelPath:           "model.safetensors",
		RunnerKind:          RunnerKindRyvionRuntime,
		IsAllowlistedRunner: true,
		RequiresNetwork:     true,
	})

	if len(result.ReasonCodes) < 2 {
		t.Fatalf("reason codes = %v, want model and network reasons", result.ReasonCodes)
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
