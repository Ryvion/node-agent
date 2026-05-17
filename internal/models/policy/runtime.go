package modelpolicy

import "strings"

const (
	RuntimeDecisionAllowed                    = "allowed"
	RuntimeDecisionExecutionDisabled          = "runtime_policy_execution_off"
	RuntimeDecisionModelDenied                = "runtime_policy_model_denied"
	RuntimeDecisionFamilyDenied               = "runtime_policy_family_denied"
	RuntimeDecisionModelNotAllowed            = "runtime_policy_model_not_allowed"
	RuntimeDecisionFamilyNotAllowed           = "runtime_policy_family_blocked"
	RuntimeDecisionCPUOffloadNotAllowed       = "runtime_policy_cpu_offload_off"
	RuntimeDecisionLargeModelNotAllowed       = "runtime_policy_large_blocked"
	RuntimeDecisionLargeModelExplicitRequired = "runtime_policy_large_needs_allow"
)

type RuntimeRequest struct {
	ModelID                string  `json:"model_id"`
	ModelSizeBytes         uint64  `json:"model_size_bytes,omitempty"`
	ParameterCountBillions float64 `json:"parameter_count_billions,omitempty"`
	Family                 string  `json:"family,omitempty"`
	CPUOffload             bool    `json:"cpu_offload,omitempty"`
}

type RuntimeDecision struct {
	Allowed bool   `json:"allowed"`
	Reason  string `json:"reason"`
}

func EvaluateRuntimeRequest(policy Policy, request RuntimeRequest) RuntimeDecision {
	runtimePolicy := NormalizePolicy(policy).RuntimePolicy
	modelID := cleanPolicyText(request.ModelID, maxPolicyItemLen)
	family := cleanPolicyText(strings.ToLower(request.Family), maxPolicyCompactLen)

	if !runtimePolicy.AllowRuntimeExecution {
		return blockedRuntimeDecision(RuntimeDecisionExecutionDisabled)
	}
	if listContains(runtimePolicy.DenyModelIDs, modelID) {
		return blockedRuntimeDecision(RuntimeDecisionModelDenied)
	}
	if family != "" && listContains(runtimePolicy.DenyFamilies, family) {
		return blockedRuntimeDecision(RuntimeDecisionFamilyDenied)
	}
	if request.CPUOffload && !runtimePolicy.AllowCPUOffload {
		return blockedRuntimeDecision(RuntimeDecisionCPUOffloadNotAllowed)
	}

	explicitlyAllowed := listContains(runtimePolicy.AllowModelIDs, modelID)
	if !explicitlyAllowed && len(runtimePolicy.AllowModelIDs) > 0 && family == "" {
		return blockedRuntimeDecision(RuntimeDecisionModelNotAllowed)
	}
	if family != "" && !explicitlyAllowed && len(runtimePolicy.AllowFamilies) > 0 && !listContains(runtimePolicy.AllowFamilies, family) {
		return blockedRuntimeDecision(RuntimeDecisionFamilyNotAllowed)
	}
	if !explicitlyAllowed && len(runtimePolicy.AllowModelIDs) > 0 && len(runtimePolicy.AllowFamilies) == 0 {
		return blockedRuntimeDecision(RuntimeDecisionModelNotAllowed)
	}

	large := request.ModelSizeBytes > runtimePolicy.MaxRuntimeModelBytes ||
		request.ParameterCountBillions > runtimePolicy.MaxRuntimeParameterCountBillions
	if large {
		if !runtimePolicy.AllowLargeModels {
			return blockedRuntimeDecision(RuntimeDecisionLargeModelNotAllowed)
		}
		if runtimePolicy.RequireExplicitAllowForLargeModels && !explicitlyAllowed && (family == "" || !listContains(runtimePolicy.AllowFamilies, family)) {
			return blockedRuntimeDecision(RuntimeDecisionLargeModelExplicitRequired)
		}
	}

	return RuntimeDecision{Allowed: true, Reason: RuntimeDecisionAllowed}
}

func blockedRuntimeDecision(reason string) RuntimeDecision {
	reason = cleanPolicyText(reason, maxPolicyCompactLen)
	if reason == "" {
		reason = "runtime_policy_blocked"
	}
	return RuntimeDecision{Allowed: false, Reason: reason}
}

func listContains(values []string, value string) bool {
	value = strings.TrimSpace(value)
	if value == "" {
		return false
	}
	for _, item := range values {
		if strings.EqualFold(strings.TrimSpace(item), value) {
			return true
		}
	}
	return false
}
