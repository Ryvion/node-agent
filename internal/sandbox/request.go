package sandbox

import "strings"

type SourceTrustLevel string

const (
	SourceTrustLevelUnknown    SourceTrustLevel = ""
	SourceTrustLevelUntrusted  SourceTrustLevel = "untrusted"
	SourceTrustLevelTrusted    SourceTrustLevel = "trusted"
	SourceTrustLevelVerified   SourceTrustLevel = "verified"
	SourceTrustLevelFirstParty SourceTrustLevel = "first_party"
)

type RunnerRequest struct {
	RunnerKind              RunnerKind       `json:"runner_kind"`
	RunnerImageOrBinary     string           `json:"runner_image_or_binary,omitempty"`
	RequiresNetwork         bool             `json:"requires_network"`
	RequiresFilesystemWrite bool             `json:"requires_filesystem_write"`
	SourceTrustLevel        SourceTrustLevel `json:"source_trust_level,omitempty"`
}

func ValidateRunnerRequest(policy SandboxPolicy, allowlist RunnerAllowlist, request RunnerRequest) SandboxDecisionResult {
	return EvaluateSandbox(policy, SandboxRequest{
		RunnerKind:              request.RunnerKind,
		RequiresNetwork:         request.RequiresNetwork,
		RequiresFilesystemWrite: request.RequiresFilesystemWrite,
		IsAllowlistedRunner:     allowlist.Allows(request),
		IsTrustedSource:         request.SourceTrustLevel.IsTrusted(),
	})
}

func (level SourceTrustLevel) IsTrusted() bool {
	switch normalizeSourceTrustLevel(level) {
	case SourceTrustLevelTrusted,
		SourceTrustLevelVerified,
		SourceTrustLevelFirstParty:
		return true
	default:
		return false
	}
}

func normalizeSourceTrustLevel(level SourceTrustLevel) SourceTrustLevel {
	value := strings.ToLower(strings.TrimSpace(string(level)))
	value = strings.ReplaceAll(value, "-", "_")
	value = strings.ReplaceAll(value, " ", "_")
	return SourceTrustLevel(value)
}
