package sandbox

type SandboxDecision string

const (
	SandboxDecisionAllow            SandboxDecision = "allow"
	SandboxDecisionReject           SandboxDecision = "reject"
	SandboxDecisionRequireIsolation SandboxDecision = "require_isolation"
)

type SandboxReasonCode string

const (
	SandboxReasonAllowed                          SandboxReasonCode = "allowed"
	SandboxReasonManagedOCIAllowed                SandboxReasonCode = "managed_oci_allowed"
	SandboxReasonRyvionRuntimeAllowed             SandboxReasonCode = "ryvion_runtime_allowed"
	SandboxReasonLlamaCPPRunnerAllowed            SandboxReasonCode = "llama_cpp_runner_allowed"
	SandboxReasonCustomRunnerAllowlisted          SandboxReasonCode = "custom_runner_allowlisted"
	SandboxReasonCustomRunnerTrustedAllowed       SandboxReasonCode = "custom_runner_trusted_allowed"
	SandboxReasonCustomRunnerNotAllowlisted       SandboxReasonCode = "custom_runner_not_allowlisted"
	SandboxReasonUnknownRunnerRejected            SandboxReasonCode = "unknown_runner_rejected"
	SandboxReasonNetworkAllowedForRunner          SandboxReasonCode = "network_allowed_for_runner"
	SandboxReasonNetworkRequiresIsolation         SandboxReasonCode = "network_requires_isolation"
	SandboxReasonFilesystemWriteAllowedForRunner  SandboxReasonCode = "filesystem_write_allowed_for_runner"
	SandboxReasonFilesystemWriteRequiresIsolation SandboxReasonCode = "filesystem_write_requires_isolation"
)

type SandboxPolicy struct {
	AllowTrustedCustomRunners         bool         `json:"allow_trusted_custom_runners"`
	NetworkAllowedRunnerKinds         []RunnerKind `json:"network_allowed_runner_kinds,omitempty"`
	FilesystemWriteAllowedRunnerKinds []RunnerKind `json:"filesystem_write_allowed_runner_kinds,omitempty"`
}

type SandboxRequest struct {
	RunnerKind              RunnerKind `json:"runner_kind"`
	RequiresNetwork         bool       `json:"requires_network"`
	RequiresFilesystemWrite bool       `json:"requires_filesystem_write"`
	IsAllowlistedRunner     bool       `json:"is_allowlisted_runner"`
	IsTrustedSource         bool       `json:"is_trusted_source"`
}

type SandboxDecisionResult struct {
	Decision    SandboxDecision     `json:"decision"`
	RunnerKind  RunnerKind          `json:"runner_kind"`
	ReasonCodes []SandboxReasonCode `json:"reason_codes"`
}

func DefaultSandboxPolicy() SandboxPolicy {
	return SandboxPolicy{
		NetworkAllowedRunnerKinds:         nil,
		FilesystemWriteAllowedRunnerKinds: nil,
		AllowTrustedCustomRunners:         false,
	}
}

func EvaluateSandbox(policy SandboxPolicy, request SandboxRequest) SandboxDecisionResult {
	policy = normalizeSandboxPolicy(policy)

	evaluation := sandboxEvaluation{decision: SandboxDecisionAllow}
	evaluateRunner(&evaluation, policy, request)
	evaluateIsolationRequirements(&evaluation, policy, request)

	if len(evaluation.reasons) == 0 {
		evaluation.reasons = append(evaluation.reasons, SandboxReasonAllowed)
	}

	return SandboxDecisionResult{
		Decision:    evaluation.decision,
		RunnerKind:  request.RunnerKind,
		ReasonCodes: evaluation.reasons,
	}
}

type sandboxEvaluation struct {
	decision SandboxDecision
	reasons  []SandboxReasonCode
}

func (e *sandboxEvaluation) reject(reason SandboxReasonCode) {
	e.decision = SandboxDecisionReject
	e.reasons = append(e.reasons, reason)
}

func (e *sandboxEvaluation) requireIsolation(reason SandboxReasonCode) {
	if e.decision != SandboxDecisionReject {
		e.decision = SandboxDecisionRequireIsolation
	}
	e.reasons = append(e.reasons, reason)
}

func (e *sandboxEvaluation) allow(reason SandboxReasonCode) {
	e.reasons = append(e.reasons, reason)
}

func evaluateRunner(evaluation *sandboxEvaluation, policy SandboxPolicy, request SandboxRequest) {
	if !validRunnerKind(request.RunnerKind) {
		evaluation.reject(SandboxReasonUnknownRunnerRejected)
		return
	}
	switch request.RunnerKind {
	case RunnerKindManagedOCI:
		evaluation.allow(SandboxReasonManagedOCIAllowed)
	case RunnerKindRyvionRuntime:
		evaluation.allow(SandboxReasonRyvionRuntimeAllowed)
	case RunnerKindLlamaCPP:
		evaluation.allow(SandboxReasonLlamaCPPRunnerAllowed)
	case RunnerKindCustom:
		if request.IsAllowlistedRunner {
			evaluation.allow(SandboxReasonCustomRunnerAllowlisted)
			return
		}
		if policy.AllowTrustedCustomRunners && request.IsTrustedSource {
			evaluation.allow(SandboxReasonCustomRunnerTrustedAllowed)
			return
		}
		evaluation.reject(SandboxReasonCustomRunnerNotAllowlisted)
	}
}

func evaluateIsolationRequirements(evaluation *sandboxEvaluation, policy SandboxPolicy, request SandboxRequest) {
	if request.RequiresNetwork {
		if runnerKindIn(request.RunnerKind, policy.NetworkAllowedRunnerKinds) {
			evaluation.allow(SandboxReasonNetworkAllowedForRunner)
		} else {
			evaluation.requireIsolation(SandboxReasonNetworkRequiresIsolation)
		}
	}
	if request.RequiresFilesystemWrite {
		if runnerKindIn(request.RunnerKind, policy.FilesystemWriteAllowedRunnerKinds) {
			evaluation.allow(SandboxReasonFilesystemWriteAllowedForRunner)
		} else {
			evaluation.requireIsolation(SandboxReasonFilesystemWriteRequiresIsolation)
		}
	}
}

func normalizeSandboxPolicy(policy SandboxPolicy) SandboxPolicy {
	return policy
}
