package modelpolicy

import (
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

const (
	bytesPerGiB           = uint64(1024 * 1024 * 1024)
	maxPolicyPathLen      = 512
	maxPolicyListItems    = 32
	maxPolicyItemLen      = 128
	maxPolicyCompactLen   = 32
	maxPolicyCacheSizeGB  = 4096
	maxPolicySingleSizeGB = 1024
	maxPolicyParamsB      = 1024
)

func FromEnv() Policy {
	return FromConfigSource(ConfigSource{})
}

func FromConfigSource(source ConfigSource) Policy {
	source = normalizeConfigSource(source)
	policy := defaultPolicy(source)

	if raw := strings.TrimSpace(source.Getenv(EnvModelAutoDownload)); raw != "" {
		policy.AutoDownload = parseBool(raw)
	}
	if raw := strings.TrimSpace(source.Getenv(EnvModelMaxSingleGB)); raw != "" {
		policy.MaxSingleModelBytes = parseGiB(raw, DefaultMaxSingleModelGB, maxPolicySingleSizeGB)
	}
	if raw := strings.TrimSpace(source.Getenv(EnvModelMaxCacheGB)); raw != "" {
		policy.MaxCacheBytes = parseGiB(raw, DefaultMaxCacheGB, maxPolicyCacheSizeGB)
	}
	if raw := strings.TrimSpace(source.Getenv(EnvModelCacheDir)); raw != "" {
		if dir := cleanPolicyPath(source.GOOS, raw, source.UserHomeDir); dir != "" {
			policy.CacheDir = dir
		}
	}
	if raw := strings.TrimSpace(source.Getenv(EnvModelAllowedFamilies)); raw != "" {
		policy.AllowedFamilies = parseList(raw, maxPolicyCompactLen, true)
	}
	if raw := strings.TrimSpace(source.Getenv(EnvModelAllowedFormats)); raw != "" {
		policy.AllowedFormats = parseList(raw, maxPolicyCompactLen, true)
	}
	if raw := strings.TrimSpace(source.Getenv(EnvModelKeepWarmIDs)); raw != "" {
		policy.KeepWarmModelIDs = parseList(raw, maxPolicyItemLen, false)
	}
	if raw := strings.TrimSpace(source.Getenv(EnvModelEvictionPolicy)); raw != "" {
		policy.EvictionPolicy = cleanPolicyText(strings.ToLower(raw), maxPolicyCompactLen)
	}
	if raw := strings.TrimSpace(source.Getenv(EnvModelAllowLicenseRestricted)); raw != "" {
		policy.AllowLicenseRestricted = parseBool(raw)
	}
	if raw := strings.TrimSpace(source.Getenv(EnvModelRuntimeMaxSingleGB)); raw != "" {
		policy.RuntimePolicy.MaxRuntimeModelBytes = parseGiB(raw, DefaultRuntimeMaxModelGB, maxPolicySingleSizeGB)
	}
	if raw := strings.TrimSpace(source.Getenv(EnvModelRuntimeMaxParamsB)); raw != "" {
		policy.RuntimePolicy.MaxRuntimeParameterCountBillions = parseBillions(raw, DefaultRuntimeMaxParamsB, maxPolicyParamsB)
	}
	if raw := strings.TrimSpace(source.Getenv(EnvModelDenyIDs)); raw != "" {
		policy.RuntimePolicy.DenyModelIDs = parseList(raw, maxPolicyItemLen, false)
	}
	if raw := strings.TrimSpace(source.Getenv(EnvModelAllowIDs)); raw != "" {
		policy.RuntimePolicy.AllowModelIDs = parseList(raw, maxPolicyItemLen, false)
	}
	if raw := strings.TrimSpace(source.Getenv(EnvModelRuntimeAllowLarge)); raw != "" {
		policy.RuntimePolicy.AllowLargeModels = parseBool(raw)
	}
	if raw := strings.TrimSpace(source.Getenv(EnvModelRequireExplicitLarge)); raw != "" {
		policy.RuntimePolicy.RequireExplicitAllowForLargeModels = parseBool(raw)
	}

	return NormalizePolicy(policy)
}

func defaultPolicy(source ConfigSource) Policy {
	cacheDir := ""
	if home, err := source.UserHomeDir(); err == nil && strings.TrimSpace(home) != "" {
		cacheDir = joinPolicyPath(source.GOOS, strings.TrimSpace(home), ".ryvion", "models")
	}
	if cacheDir == "" {
		cacheDir = ".ryvion/models"
	}
	return Policy{
		AutoDownload:           false,
		MaxSingleModelBytes:    DefaultMaxSingleModelGB * bytesPerGiB,
		MaxCacheBytes:          DefaultMaxCacheGB * bytesPerGiB,
		CacheDir:               cleanPolicyPath(source.GOOS, cacheDir, source.UserHomeDir),
		AllowedFamilies:        cloneStrings(DefaultAllowedFamilies),
		AllowedFormats:         cloneStrings(DefaultAllowedFormats),
		KeepWarmModelIDs:       []string{},
		EvictionPolicy:         DefaultEvictionPolicy,
		AllowLicenseRestricted: false,
		RuntimePolicy:          defaultRuntimePolicy(),
	}
}

func normalizeConfigSource(source ConfigSource) ConfigSource {
	if source.Getenv == nil {
		source.Getenv = os.Getenv
	}
	if source.UserHomeDir == nil {
		source.UserHomeDir = os.UserHomeDir
	}
	if strings.TrimSpace(source.GOOS) == "" {
		source.GOOS = runtime.GOOS
	}
	return source
}

func parseBool(raw string) bool {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func parseGiB(raw string, fallbackGB uint64, maxGB uint64) uint64 {
	value, err := strconv.ParseFloat(strings.TrimSpace(raw), 64)
	if err != nil || value <= 0 {
		return fallbackGB * bytesPerGiB
	}
	gb := uint64(value)
	if gb == 0 {
		return fallbackGB * bytesPerGiB
	}
	if maxGB > 0 && gb > maxGB {
		gb = maxGB
	}
	return gb * bytesPerGiB
}

func parseBillions(raw string, fallback float64, maxValue float64) float64 {
	value, err := strconv.ParseFloat(strings.TrimSpace(raw), 64)
	if err != nil || value <= 0 {
		return fallback
	}
	if maxValue > 0 && value > maxValue {
		value = maxValue
	}
	return value
}

func parseList(raw string, maxRunes int, lower bool) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, 8)
	for _, item := range strings.Split(raw, ",") {
		if len(out) >= maxPolicyListItems {
			break
		}
		if lower {
			item = strings.ToLower(item)
		}
		value := cleanPolicyText(item, maxRunes)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	if out == nil {
		return []string{}
	}
	return out
}

func cleanPolicyPath(goos, value string, userHomeDir func() (string, error)) string {
	value = cleanPolicyText(value, maxPolicyPathLen)
	if value == "" {
		return ""
	}
	if strings.HasPrefix(value, "~") {
		home := ""
		if userHomeDir != nil {
			if found, err := userHomeDir(); err == nil {
				home = strings.TrimSpace(found)
			}
		}
		if home != "" {
			rest := strings.TrimLeft(strings.TrimPrefix(value, "~"), `/\`)
			value = joinPolicyPath(goos, home, rest)
		}
	}
	if isWindowsPath(goos, value) {
		value = strings.TrimRight(value, `\/`)
		if isUnsafeWindowsRoot(value) {
			return ""
		}
		return value
	}
	cleaned := filepath.Clean(value)
	if cleaned == "." || cleaned == string(filepath.Separator) {
		return ""
	}
	return cleaned
}

func joinPolicyPath(goos, dir string, elems ...string) string {
	dir = strings.TrimSpace(dir)
	if dir == "" {
		return ""
	}
	if isWindowsPath(goos, dir) {
		path := strings.TrimRight(dir, `\/`)
		for _, elem := range elems {
			elem = strings.Trim(strings.TrimSpace(elem), `\/`)
			if elem == "" {
				continue
			}
			path += `\` + elem
		}
		return path
	}
	parts := append([]string{dir}, elems...)
	return filepath.Join(parts...)
}

func isWindowsPath(goos, value string) bool {
	return strings.EqualFold(strings.TrimSpace(goos), "windows") || strings.Contains(value, `\`)
}

func isUnsafeWindowsRoot(value string) bool {
	trimmed := strings.TrimSpace(value)
	if trimmed == `\` || trimmed == `/` {
		return true
	}
	if len(trimmed) == 2 && trimmed[1] == ':' {
		return true
	}
	if len(trimmed) == 3 && trimmed[1] == ':' && (trimmed[2] == '\\' || trimmed[2] == '/') {
		return true
	}
	return false
}

func cloneStrings(values []string) []string {
	if len(values) == 0 {
		return []string{}
	}
	out := make([]string, len(values))
	copy(out, values)
	return out
}
