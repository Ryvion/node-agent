package modelpolicy

import "strings"

const (
	PrepareDecisionAllowed              = "allowed"
	PrepareDecisionAutoDownloadDisabled = "policy_auto_download_disabled"
	PrepareDecisionModelSizeRequired    = "policy_model_size_required"
	PrepareDecisionModelTooLarge        = "policy_model_too_large"
	PrepareDecisionCacheCapacity        = "policy_cache_capacity_exceeded"
	PrepareDecisionFamilyNotAllowed     = "policy_family_not_allowed"
	PrepareDecisionFormatNotAllowed     = "policy_format_not_allowed"
)

type PrepareRequest struct {
	ModelID        string `json:"model_id"`
	ModelSizeBytes uint64 `json:"model_size_bytes"`
	CacheUsedBytes uint64 `json:"cache_used_bytes"`
	Family         string `json:"family,omitempty"`
	Format         string `json:"format,omitempty"`
}

type PrepareDecision struct {
	Allowed bool   `json:"allowed"`
	Reason  string `json:"reason"`
}

func EvaluatePrepareRequest(policy Policy, request PrepareRequest) PrepareDecision {
	policy = NormalizePolicy(policy)
	request.Family = cleanPolicyText(strings.ToLower(request.Family), maxPolicyCompactLen)
	request.Format = cleanPolicyText(strings.ToLower(request.Format), maxPolicyCompactLen)

	if !policy.AutoDownload {
		return blockedPrepareDecision(PrepareDecisionAutoDownloadDisabled)
	}
	if request.ModelSizeBytes == 0 {
		return blockedPrepareDecision(PrepareDecisionModelSizeRequired)
	}
	if request.ModelSizeBytes > policy.MaxSingleModelBytes {
		return blockedPrepareDecision(PrepareDecisionModelTooLarge)
	}
	if policy.MaxCacheBytes > 0 && (request.CacheUsedBytes > policy.MaxCacheBytes || request.ModelSizeBytes > policy.MaxCacheBytes-request.CacheUsedBytes) {
		return blockedPrepareDecision(PrepareDecisionCacheCapacity)
	}
	if request.Family != "" && !policyAllowsValue(policy.AllowedFamilies, request.Family) {
		return blockedPrepareDecision(PrepareDecisionFamilyNotAllowed)
	}
	if request.Format != "" && !policyAllowsValue(policy.AllowedFormats, request.Format) {
		return blockedPrepareDecision(PrepareDecisionFormatNotAllowed)
	}
	return PrepareDecision{Allowed: true, Reason: PrepareDecisionAllowed}
}

func blockedPrepareDecision(reason string) PrepareDecision {
	reason = cleanPolicyText(reason, maxPolicyCompactLen)
	if reason == "" {
		reason = "policy_prepare_blocked"
	}
	return PrepareDecision{Allowed: false, Reason: reason}
}

func policyAllowsValue(allowed []string, value string) bool {
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "" {
		return true
	}
	for _, item := range allowed {
		if strings.EqualFold(strings.TrimSpace(item), value) {
			return true
		}
	}
	return false
}
