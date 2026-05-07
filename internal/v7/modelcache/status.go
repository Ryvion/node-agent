package modelcache

import "strings"

func BuildStatus(cacheDir string) Status {
	return Scan(cacheDir)
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
		if model.Filename == "" || model.Path == "" {
			continue
		}
		model.Installed = true
		model.HashVerified = false
		total += model.SizeBytes
		models = append(models, model)
	}
	status.Models = models
	status.TotalBytes = total
	return status
}

func normalizeFamily(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "llama":
		return "llama"
	case "phi":
		return "phi"
	case "qwen":
		return "qwen"
	case "gemma":
		return "gemma"
	default:
		return "unknown"
	}
}
