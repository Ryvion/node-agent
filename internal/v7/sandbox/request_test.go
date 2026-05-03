package sandbox

import "testing"

func TestValidateRunnerRequestAllowlistedRunnerAccepted(t *testing.T) {
	result := ValidateRunnerRequest(DefaultSandboxPolicy(), runnerAllowlistForTest(), RunnerRequest{
		RunnerKind:          RunnerKindCustom,
		RunnerImageOrBinary: "ghcr.io/ryvion/runner-safe:v1",
		ModelPath:           "model.safetensors",
	})

	assertSandboxDecision(t, result, SandboxDecisionAllow)
	assertHasReason(t, result, SandboxReasonSafetensorsAllowlistedRunnerAllowed)
}

func TestValidateRunnerRequestCustomNonAllowlistedRejected(t *testing.T) {
	result := ValidateRunnerRequest(DefaultSandboxPolicy(), RunnerAllowlist{}, RunnerRequest{
		RunnerKind:          RunnerKindCustom,
		RunnerImageOrBinary: "ghcr.io/example/custom-runner:v1",
		ModelPath:           "model.safetensors",
	})

	assertSandboxDecision(t, result, SandboxDecisionReject)
	assertHasReason(t, result, SandboxReasonCustomRunnerNotAllowlisted)
}

func TestValidateRunnerRequestPyTorchPickleRejected(t *testing.T) {
	result := ValidateRunnerRequest(DefaultSandboxPolicy(), runnerAllowlistForTest(), RunnerRequest{
		RunnerKind:          RunnerKindCustom,
		RunnerImageOrBinary: "ghcr.io/ryvion/runner-safe:v1",
		ModelPath:           "checkpoint.pt",
		SourceTrustLevel:    SourceTrustLevelTrusted,
	})

	assertSandboxDecision(t, result, SandboxDecisionReject)
	assertHasReason(t, result, SandboxReasonPyTorchPickleRejected)
}

func TestValidateRunnerRequestGGUFNativeAccepted(t *testing.T) {
	result := ValidateRunnerRequest(DefaultSandboxPolicy(), RunnerAllowlist{}, RunnerRequest{
		RunnerKind: RunnerKindNativeLlama,
		ModelPath:  "llama.gguf",
	})

	assertSandboxDecision(t, result, SandboxDecisionAllow)
	assertHasReason(t, result, SandboxReasonGGUFNativeLlamaAllowed)
}

func TestValidateRunnerRequestNetworkRequiresIsolation(t *testing.T) {
	result := ValidateRunnerRequest(DefaultSandboxPolicy(), runnerAllowlistForTest(), RunnerRequest{
		RunnerKind:          RunnerKindCustom,
		RunnerImageOrBinary: "ghcr.io/ryvion/runner-safe:v1",
		ModelPath:           "model.safetensors",
		RequiresNetwork:     true,
	})

	assertSandboxDecision(t, result, SandboxDecisionRequireIsolation)
	assertHasReason(t, result, SandboxReasonNetworkRequiresIsolation)
	assertHasReason(t, result, SandboxReasonSafetensorsAllowlistedRunnerAllowed)
}

func TestValidateRunnerRequestDeclaredModelFormatUsed(t *testing.T) {
	result := ValidateRunnerRequest(DefaultSandboxPolicy(), runnerAllowlistForTest(), RunnerRequest{
		RunnerKind:          RunnerKindCustom,
		RunnerImageOrBinary: "ghcr.io/ryvion/runner-safe:v1",
		ModelPath:           "model.bin",
		DeclaredModelFormat: "safetensors",
		SourceTrustLevel:    SourceTrustLevelUntrusted,
	})

	assertSandboxDecision(t, result, SandboxDecisionAllow)
	if result.ModelFormat != ModelFormatSafetensors {
		t.Fatalf("ModelFormat = %q, want %q", result.ModelFormat, ModelFormatSafetensors)
	}
}

func TestValidateRunnerRequestUsesSourceTrustLevel(t *testing.T) {
	policy := DefaultSandboxPolicy()
	policy.AllowUnknownTrustedAllowlisted = true

	result := ValidateRunnerRequest(policy, runnerAllowlistForTest(), RunnerRequest{
		RunnerKind:          RunnerKindCustom,
		RunnerImageOrBinary: "ghcr.io/ryvion/runner-safe:v1",
		ModelPath:           "model.weights",
		SourceTrustLevel:    "verified",
	})

	assertSandboxDecision(t, result, SandboxDecisionRequireIsolation)
	assertHasReason(t, result, SandboxReasonUnknownFormatTrustedAllowed)
}

func TestValidateRunnerRequestReasonCodesPresent(t *testing.T) {
	result := ValidateRunnerRequest(DefaultSandboxPolicy(), runnerAllowlistForTest(), RunnerRequest{
		RunnerKind:          RunnerKindCustom,
		RunnerImageOrBinary: "ghcr.io/ryvion/runner-safe:v1",
		ModelPath:           "model.safetensors",
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
