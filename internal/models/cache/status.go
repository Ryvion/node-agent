package modelcache

import (
	"strings"

	"github.com/Ryvion/ryvion-node/internal/models/policy"
)

const canonicalTinyLlamaDrafterModelID = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

func BuildStatus(cacheDir string) Status {
	return Scan(cacheDir)
}

func BuildStatusFromDirs(cacheDirs []string) Status {
	if len(cacheDirs) == 0 {
		return BuildStatus("")
	}
	merged := Status{
		CacheDir: cleanCachePath("", cacheDirs[0]),
		Models:   []Model{},
	}
	seenModelIDs := map[string]struct{}{}
	seenPaths := map[string]struct{}{}
	for _, cacheDir := range cacheDirs {
		status := Scan(cacheDir)
		for _, model := range status.Models {
			modelKey := strings.ToLower(strings.TrimSpace(firstNonEmptyString(model.ModelID, model.Filename)))
			pathKey := strings.ToLower(strings.TrimSpace(model.Path))
			if modelKey == "" || pathKey == "" {
				continue
			}
			if _, ok := seenModelIDs[modelKey]; ok {
				continue
			}
			if _, ok := seenPaths[pathKey]; ok {
				continue
			}
			seenModelIDs[modelKey] = struct{}{}
			seenPaths[pathKey] = struct{}{}
			merged.Models = append(merged.Models, model)
		}
	}
	return NormalizeStatus(merged)
}

func NormalizeStatus(status Status) Status {
	status.CacheDir = cleanCachePath("", status.CacheDir)
	if status.TotalBytes < 0 {
		status.TotalBytes = 0
	}
	if len(status.Models) == 0 {
		status.Models = []Model{}
		return status
	}
	models := make([]Model, 0, min(len(status.Models), DefaultMaxModels))
	total := int64(0)
	for _, model := range status.Models {
		if len(models) >= DefaultMaxModels {
			break
		}
		model.ModelID = cleanCacheText(model.ModelID, maxCacheTextLen)
		model.Filename = cleanCacheText(model.Filename, maxCacheTextLen)
		model.Path = cleanCachePath("", model.Path)
		model.FamilyHint = normalizeFamily(model.FamilyHint)
		model.QuantizationHint = cleanCacheText(strings.ToUpper(model.QuantizationHint), maxCacheCompactLen)
		if model.QuantizationHint == "" {
			model.QuantizationHint = "unknown"
		}
		if model.ParameterCountBillions < 0 {
			model.ParameterCountBillions = 0
		}
		if model.ParameterCountBillions == 0 {
			model.ParameterCountBillions = InferParameterCountBillions(firstNonEmptyString(model.Filename, model.ModelID, model.Path))
		}
		model.Format = cleanCacheText(strings.ToLower(model.Format), maxCacheCompactLen)
		if model.Format == "" {
			model.Format = DefaultFormat
		}
		if model.SizeBytes < 0 {
			model.SizeBytes = 0
		}
		if model.ModelID == "" {
			model.ModelID = model.Filename
		}
		model.ModelID = canonicalModelID(model)
		if model.Filename == "" || model.Path == "" {
			continue
		}
		model.Installed = true
		model.Resident = true
		model.HashVerified = false
		model.BlockedReasons = normalizeBlockedReasons(model.BlockedReasons)
		total += model.SizeBytes
		models = append(models, model)
	}
	status.Models = models
	status.TotalBytes = total
	return status
}

func canonicalModelID(model Model) string {
	modelID := strings.TrimSpace(model.ModelID)
	name := strings.ToLower(firstNonEmptyString(model.Filename, model.ModelID, model.Path))
	if strings.Contains(name, "tinyllama") &&
		strings.EqualFold(model.FamilyHint, "llama") &&
		strings.EqualFold(model.Format, DefaultFormat) &&
		(model.ParameterCountBillions == 0 || model.ParameterCountBillions <= 1.2) {
		return canonicalTinyLlamaDrafterModelID
	}
	return modelID
}

func AnnotateRuntimeStatus(input RuntimeAnnotationInput) Status {
	status := NormalizeStatus(input.Status)
	policy := modelpolicy.NormalizePolicy(input.Policy)
	for i := range status.Models {
		model := &status.Models[i]
		decision := modelpolicy.EvaluateRuntimeRequest(policy, modelpolicy.RuntimeRequest{
			ModelID:                model.ModelID,
			ModelSizeBytes:         uint64NonNegative(model.SizeBytes),
			ParameterCountBillions: model.ParameterCountBillions,
			Family:                 model.FamilyHint,
		})
		reasons := make([]string, 0, 4)
		if !input.V7InferenceEnabled {
			reasons = append(reasons, "v7_inference_disabled")
		}
		if !input.HardwareCapacityAvailable {
			reasons = append(reasons, "hardware_capacity_missing")
		}
		if !input.BackendTextGenerationAvailable {
			reasons = append(reasons, "backend_text_generation_unavailable")
		}
		if !decision.Allowed {
			reasons = append(reasons, decision.Reason)
		}
		model.Runnable = len(reasons) == 0
		model.BlockedReasons = normalizeBlockedReasons(reasons)
	}
	return NormalizeStatus(status)
}

func normalizeFamily(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "llama":
		return "llama"
	case "phi":
		return "phi"
	case "qwen":
		return "qwen"
	case "gpt-oss", "gptoss":
		return "gpt-oss"
	case "gemma":
		return "gemma"
	default:
		return "unknown"
	}
}

func normalizeBlockedReasons(reasons []string) []string {
	if len(reasons) == 0 {
		return []string{}
	}
	seen := map[string]struct{}{}
	out := make([]string, 0, min(len(reasons), 8))
	for _, reason := range reasons {
		if len(out) >= 8 {
			break
		}
		reason = cleanCacheText(strings.ToLower(reason), maxCacheCompactLen)
		if reason == "" || reason == "allowed" {
			continue
		}
		if _, ok := seen[reason]; ok {
			continue
		}
		seen[reason] = struct{}{}
		out = append(out, reason)
	}
	if out == nil {
		return []string{}
	}
	return out
}

func uint64NonNegative(value int64) uint64 {
	if value <= 0 {
		return 0
	}
	return uint64(value)
}

func firstNonEmptyString(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}
