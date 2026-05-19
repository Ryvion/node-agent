package sandbox

import "testing"

func TestValidateRunnerRequestAllowlistedRunnerAccepted(t *testing.T) {
	result := ValidateRunnerRequest(DefaultSandboxPolicy(), runnerAllowlistForTest(), RunnerRequest{
		RunnerKind:          RunnerKindCustom,
		RunnerImageOrBinary: "ghcr.io/ryvion/runner-safe:v1",
	})

	assertSandboxDecision(t, result, SandboxDecisionAllow)
	assertHasReason(t, result, SandboxReasonCustomRunnerAllowlisted)
}

func TestValidateRunnerRequestCustomNonAllowlistedRejected(t *testing.T) {
	result := ValidateRunnerRequest(DefaultSandboxPolicy(), RunnerAllowlist{}, RunnerRequest{
		RunnerKind:          RunnerKindCustom,
		RunnerImageOrBinary: "ghcr.io/example/custom-runner:v1",
	})

	assertSandboxDecision(t, result, SandboxDecisionReject)
	assertHasReason(t, result, SandboxReasonCustomRunnerNotAllowlisted)
}

func TestValidateRunnerRequestBuiltInRunnerAccepted(t *testing.T) {
	result := ValidateRunnerRequest(DefaultSandboxPolicy(), RunnerAllowlist{}, RunnerRequest{
		RunnerKind: RunnerKindBlender,
	})

	assertSandboxDecision(t, result, SandboxDecisionAllow)
	assertHasReason(t, result, SandboxReasonBlenderRunnerAllowed)
}

func TestValidateRunnerRequestNetworkRequiresIsolation(t *testing.T) {
	result := ValidateRunnerRequest(DefaultSandboxPolicy(), runnerAllowlistForTest(), RunnerRequest{
		RunnerKind:          RunnerKindCustom,
		RunnerImageOrBinary: "ghcr.io/ryvion/runner-safe:v1",
		RequiresNetwork:     true,
	})

	assertSandboxDecision(t, result, SandboxDecisionRequireIsolation)
	assertHasReason(t, result, SandboxReasonNetworkRequiresIsolation)
	assertHasReason(t, result, SandboxReasonCustomRunnerAllowlisted)
}

func TestValidateRunnerRequestUsesSourceTrustLevel(t *testing.T) {
	policy := DefaultSandboxPolicy()
	policy.AllowTrustedCustomRunners = true

	result := ValidateRunnerRequest(policy, RunnerAllowlist{}, RunnerRequest{
		RunnerKind:       RunnerKindCustom,
		SourceTrustLevel: "verified",
	})

	assertSandboxDecision(t, result, SandboxDecisionAllow)
	assertHasReason(t, result, SandboxReasonCustomRunnerTrustedAllowed)
}

func TestValidateRunnerRequestReasonCodesPresent(t *testing.T) {
	result := ValidateRunnerRequest(DefaultSandboxPolicy(), runnerAllowlistForTest(), RunnerRequest{
		RunnerKind:          RunnerKindCustom,
		RunnerImageOrBinary: "ghcr.io/ryvion/runner-safe:v1",
		RequiresNetwork:     true,
	})

	if len(result.ReasonCodes) == 0 {
		t.Fatalf("ReasonCodes = nil, want at least one reason code")
	}
}

func runnerAllowlistForTest() RunnerAllowlist {
	return RunnerAllowlist{
		Entries: []RunnerAllowlistEntry{
			{
				RunnerKind:          RunnerKindCustom,
				RunnerImageOrBinary: "ghcr.io/ryvion/runner-safe:v1",
			},
		},
	}
}
