package modelcache

import (
	"path/filepath"
	"regexp"
	"strings"
)

var ggufQuantizationTokenPattern = regexp.MustCompile(`(?i)^(?:(?:IQ|Q)[0-9](?:_[A-Z0-9]+){0,3}|BF16|F16|F32)$`)

func ModelMatches(model Model, modelID string) bool {
	for _, value := range []string{model.ModelID, model.Filename, model.Path} {
		if ModelIDMatches(value, modelID) {
			return true
		}
	}
	return false
}

func ModelIDMatches(value, modelID string) bool {
	want := modelAliasToken(modelID)
	if want == "" {
		return false
	}
	return modelAliasToken(value) == want
}

func modelAliasToken(value string) string {
	token := normalizeModelComparable(value)
	token = strings.TrimSuffix(token, ".gguf")
	if strings.HasPrefix(token, "gemma-") {
		return gemmaAliasToken(token)
	}
	return token
}

func normalizeModelComparable(value string) string {
	value = strings.TrimSpace(value)
	if value == "" || value == "." || value == string(filepath.Separator) {
		return ""
	}
	value = strings.ReplaceAll(value, "\\", "/")
	return strings.ToLower(filepath.Base(value))
}

func gemmaAliasToken(token string) string {
	parts := strings.Split(token, "-")
	if len(parts) < 3 {
		return token
	}
	stripped := false
	if len(parts) > 0 && parts[len(parts)-1] == "gguf" {
		parts = parts[:len(parts)-1]
		stripped = true
	}
	if len(parts) > 0 && ggufQuantizationTokenPattern.MatchString(parts[len(parts)-1]) {
		parts = parts[:len(parts)-1]
		stripped = true
	}
	if stripped && len(parts) > 0 && parts[len(parts)-1] == "qat" {
		parts = parts[:len(parts)-1]
	}
	if len(parts) < 3 {
		return token
	}
	return strings.Join(parts, "-")
}
