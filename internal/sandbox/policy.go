package sandbox

type SandboxDecision string

const (
	SandboxDecisionAllow            SandboxDecision = "allow"
	SandboxDecisionReject           SandboxDecision = "reject"
	SandboxDecisionRequireIsolation SandboxDecision = "require_isolation"
)

type SandboxReasonCode string

const (
	SandboxReasonAllowed                             SandboxReasonCode = "allowed"
	SandboxReasonGGUFNativeLlamaAllowed              SandboxReasonCode = "gguf_native_llama_allowed"
	SandboxReasonGGUFAllowlistedRunnerAllowed        SandboxReasonCode = "gguf_allowlisted_runner_allowed"
	SandboxReasonGGUFRequiresNativeOrAllowlisted     SandboxReasonCode = "gguf_requires_native_or_allowlisted_runner"
	SandboxReasonSafetensorsAllowlistedRunnerAllowed SandboxReasonCode = "safetensors_allowlisted_runner_allowed"
	SandboxReasonSafetensorsRequiresAllowlisted      SandboxReasonCode = "safetensors_requires_allowlisted_runner"
	SandboxReasonONNXAllowlistedRunnerAllowed        SandboxReasonCode = "onnx_allowlisted_runner_allowed"
	SandboxReasonONNXRequiresAllowlisted             SandboxReasonCode = "onnx_requires_allowlisted_runner"
	SandboxReasonTorchScriptRequiresIsolation        SandboxReasonCode = "torchscript_requires_isolation"
	SandboxReasonTorchScriptRequiresAllowlisted      SandboxReasonCode = "torchscript_requires_allowlisted_runner"
	SandboxReasonPyTorchPickleRejected               SandboxReasonCode = "pytorch_pickle_rejected"
	SandboxReasonPyTorchPickleRequiresIsolation      SandboxReasonCode = "pytorch_pickle_requires_isolation"
	SandboxReasonPythonSourceRequiresIsolation       SandboxReasonCode = "python_source_requires_isolation"
	SandboxReasonPythonSourceTrustedAllowed          SandboxReasonCode = "python_source_trusted_allowlisted"
	SandboxReasonPythonSourceRejected                SandboxReasonCode = "python_source_rejected"
	SandboxReasonUnknownFormatRejected               SandboxReasonCode = "unknown_format_rejected"
	SandboxReasonUnknownFormatTrustedAllowed         SandboxReasonCode = "unknown_format_trusted_allowlisted"
	SandboxReasonCustomRunnerNotAllowlisted          SandboxReasonCode = "custom_runner_not_allowlisted"
	SandboxReasonUnknownRunnerRejected               SandboxReasonCode = "unknown_runner_rejected"
	SandboxReasonNetworkAllowedForRunner             SandboxReasonCode = "network_allowed_for_runner"
	SandboxReasonNetworkRequiresIsolation            SandboxReasonCode = "network_requires_isolation"
	SandboxReasonFilesystemWriteAllowedForRunner     SandboxReasonCode = "filesystem_write_allowed_for_runner"
	SandboxReasonFilesystemWriteRequiresIsolation    SandboxReasonCode = "filesystem_write_requires_isolation"
)

type SandboxPolicy struct {
	AllowUnknownTrustedAllowlisted       bool            `json:"allow_unknown_trusted_allowlisted"`
	AllowPyTorchPickleTrustedAllowlisted bool            `json:"allow_pytorch_pickle_trusted_allowlisted"`
	PythonSourceDecision                 SandboxDecision `json:"python_source_decision,omitempty"`
	NetworkAllowedRunnerKinds            []RunnerKind    `json:"network_allowed_runner_kinds,omitempty"`
	FilesystemWriteAllowedRunnerKinds    []RunnerKind    `json:"filesystem_write_allowed_runner_kinds,omitempty"`
}

type SandboxRequest struct {
	ModelPath               string     `json:"model_path,omitempty"`
	DeclaredFormat          string     `json:"declared_format,omitempty"`
	RunnerKind              RunnerKind `json:"runner_kind"`
	RequiresNetwork         bool       `json:"requires_network"`
	RequiresFilesystemWrite bool       `json:"requires_filesystem_write"`
	IsAllowlistedRunner     bool       `json:"is_allowlisted_runner"`
	IsTrustedSource         bool       `json:"is_trusted_source"`
}

type SandboxDecisionResult struct {
	Decision    SandboxDecision     `json:"decision"`
	ModelFormat ModelFormat         `json:"model_format"`
	RunnerKind  RunnerKind          `json:"runner_kind"`
	ReasonCodes []SandboxReasonCode `json:"reason_codes"`
}

func DefaultSandboxPolicy() SandboxPolicy {
	return SandboxPolicy{
		PythonSourceDecision:              SandboxDecisionRequireIsolation,
		NetworkAllowedRunnerKinds:         []RunnerKind{RunnerKindAgentHosting},
		FilesystemWriteAllowedRunnerKinds: nil,
	}
}

func EvaluateSandbox(policy SandboxPolicy, request SandboxRequest) SandboxDecisionResult {
	policy = normalizeSandboxPolicy(policy)
	format := EvaluateModelFormat(request.ModelPath, request.DeclaredFormat)

	evaluation := sandboxEvaluation{decision: SandboxDecisionAllow}
	evaluateRunner(&evaluation, request)
	evaluateModelFormat(&evaluation, policy, request, format)
	evaluateIsolationRequirements(&evaluation, policy, request)

	if len(evaluation.reasons) == 0 {
		evaluation.reasons = append(evaluation.reasons, SandboxReasonAllowed)
	}

	return SandboxDecisionResult{
		Decision:    evaluation.decision,
		ModelFormat: format,
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

func evaluateRunner(evaluation *sandboxEvaluation, request SandboxRequest) {
	if !validRunnerKind(request.RunnerKind) {
		evaluation.reject(SandboxReasonUnknownRunnerRejected)
		return
	}
	if request.RunnerKind == RunnerKindCustom && !request.IsAllowlistedRunner {
		evaluation.reject(SandboxReasonCustomRunnerNotAllowlisted)
	}
}

func evaluateModelFormat(evaluation *sandboxEvaluation, policy SandboxPolicy, request SandboxRequest, format ModelFormat) {
	switch format {
	case ModelFormatGGUF:
		if request.RunnerKind == RunnerKindNativeLlama {
			evaluation.allow(SandboxReasonGGUFNativeLlamaAllowed)
			return
		}
		if request.IsAllowlistedRunner {
			evaluation.allow(SandboxReasonGGUFAllowlistedRunnerAllowed)
			return
		}
		evaluation.reject(SandboxReasonGGUFRequiresNativeOrAllowlisted)
	case ModelFormatSafetensors:
		if request.IsAllowlistedRunner {
			evaluation.allow(SandboxReasonSafetensorsAllowlistedRunnerAllowed)
			return
		}
		evaluation.reject(SandboxReasonSafetensorsRequiresAllowlisted)
	case ModelFormatONNX:
		if request.IsAllowlistedRunner {
			evaluation.allow(SandboxReasonONNXAllowlistedRunnerAllowed)
			return
		}
		evaluation.reject(SandboxReasonONNXRequiresAllowlisted)
	case ModelFormatTorchScript:
		if request.IsAllowlistedRunner {
			evaluation.requireIsolation(SandboxReasonTorchScriptRequiresIsolation)
			return
		}
		evaluation.reject(SandboxReasonTorchScriptRequiresAllowlisted)
	case ModelFormatPyTorchPickle:
		if policy.AllowPyTorchPickleTrustedAllowlisted && request.IsTrustedSource && request.IsAllowlistedRunner {
			evaluation.requireIsolation(SandboxReasonPyTorchPickleRequiresIsolation)
			return
		}
		evaluation.reject(SandboxReasonPyTorchPickleRejected)
	case ModelFormatPythonSource:
		switch policy.PythonSourceDecision {
		case SandboxDecisionRequireIsolation:
			evaluation.requireIsolation(SandboxReasonPythonSourceRequiresIsolation)
		case SandboxDecisionAllow:
			if request.IsTrustedSource && request.IsAllowlistedRunner {
				evaluation.allow(SandboxReasonPythonSourceTrustedAllowed)
				return
			}
			evaluation.requireIsolation(SandboxReasonPythonSourceRequiresIsolation)
		default:
			evaluation.reject(SandboxReasonPythonSourceRejected)
		}
	case ModelFormatUnknown:
		if policy.AllowUnknownTrustedAllowlisted && request.IsTrustedSource && request.IsAllowlistedRunner {
			evaluation.requireIsolation(SandboxReasonUnknownFormatTrustedAllowed)
			return
		}
		evaluation.reject(SandboxReasonUnknownFormatRejected)
	default:
		evaluation.reject(SandboxReasonUnknownFormatRejected)
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
	defaultPolicy := DefaultSandboxPolicy()
	if policy.PythonSourceDecision == "" {
		policy.PythonSourceDecision = defaultPolicy.PythonSourceDecision
	}
	if policy.NetworkAllowedRunnerKinds == nil {
		policy.NetworkAllowedRunnerKinds = defaultPolicy.NetworkAllowedRunnerKinds
	}
	return policy
}
