package modelpolicy

import (
	"errors"
	"strings"
	"unicode"
)

var ErrInvalidPolicy = errors.New("invalid model policy")

func NormalizePolicy(policy Policy) Policy {
	policy.CacheDir = cleanPolicyPath("", policy.CacheDir, nil)
	if policy.CacheDir == "" {
		policy.CacheDir = ".ryvion/models"
	}
	if policy.MaxSingleModelBytes == 0 {
		policy.MaxSingleModelBytes = DefaultMaxSingleModelGB * bytesPerGiB
	}
	if policy.MaxCacheBytes == 0 {
		policy.MaxCacheBytes = DefaultMaxCacheGB * bytesPerGiB
	}
	maxSingleCap := uint64(maxPolicySingleSizeGB) * bytesPerGiB
	if policy.MaxSingleModelBytes > maxSingleCap {
		policy.MaxSingleModelBytes = maxSingleCap
	}
	maxCacheCap := uint64(maxPolicyCacheSizeGB) * bytesPerGiB
	if policy.MaxCacheBytes > maxCacheCap {
		policy.MaxCacheBytes = maxCacheCap
	}
	if policy.MaxSingleModelBytes > policy.MaxCacheBytes {
		policy.MaxSingleModelBytes = policy.MaxCacheBytes
	}
	policy.AllowedFamilies = normalizeList(policy.AllowedFamilies, maxPolicyCompactLen, DefaultAllowedFamilies)
	policy.AllowedFormats = normalizeList(policy.AllowedFormats, maxPolicyCompactLen, DefaultAllowedFormats)
	policy.KeepWarmModelIDs = normalizeModelIDs(policy.KeepWarmModelIDs)
	policy.EvictionPolicy = cleanPolicyText(strings.ToLower(policy.EvictionPolicy), maxPolicyCompactLen)
	if policy.EvictionPolicy == "" {
		policy.EvictionPolicy = DefaultEvictionPolicy
	}
	return policy
}

func ValidatePolicy(policy Policy) error {
	policy = NormalizePolicy(policy)
	if policy.CacheDir == "" {
		return ErrInvalidPolicy
	}
	if policy.MaxSingleModelBytes == 0 || policy.MaxCacheBytes == 0 || policy.MaxSingleModelBytes > policy.MaxCacheBytes {
		return ErrInvalidPolicy
	}
	if len(policy.AllowedFamilies) == 0 || len(policy.AllowedFormats) == 0 {
		return ErrInvalidPolicy
	}
	return nil
}

func normalizeList(values []string, maxRunes int, fallback []string) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, len(values))
	for _, value := range values {
		if len(out) >= maxPolicyListItems {
			break
		}
		value = cleanPolicyText(strings.ToLower(value), maxRunes)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	if len(out) == 0 && len(fallback) > 0 {
		return cloneStrings(fallback)
	}
	if out == nil {
		return []string{}
	}
	return out
}

func normalizeModelIDs(values []string) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, len(values))
	for _, value := range values {
		if len(out) >= maxPolicyListItems {
			break
		}
		value = cleanPolicyText(value, maxPolicyItemLen)
		if value == "" {
			continue
		}
		key := strings.ToLower(value)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, value)
	}
	if out == nil {
		return []string{}
	}
	return out
}

func cleanPolicyText(value string, maxRunes int) string {
	value = strings.TrimSpace(value)
	if value == "" || maxRunes <= 0 {
		return ""
	}
	var b strings.Builder
	b.Grow(len(value))
	written := 0
	for _, r := range value {
		if unicode.IsControl(r) {
			continue
		}
		if written >= maxRunes {
			break
		}
		b.WriteRune(r)
		written++
	}
	return strings.TrimSpace(b.String())
}
