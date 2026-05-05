package modelbench

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
)

type ModelBenchmarkSpec struct {
	Task            string  `json:"task"`
	RequestID       string  `json:"request_id"`
	JobID           string  `json:"job_id"`
	ModelID         string  `json:"model_id"`
	PromptProfileID string  `json:"prompt_profile_id,omitempty"`
	PromptLabel     string  `json:"prompt_label,omitempty"`
	PromptHash      string  `json:"prompt_hash"`
	MaxTokens       int     `json:"max_tokens"`
	Temperature     float64 `json:"temperature"`
	TimeoutMs       int64   `json:"timeout_ms"`
	CreatedAtUnixMs int64   `json:"created_at_unix_ms"`
}

func HashBenchmarkPrompt(prompt ModelBenchmarkPrompt) string {
	hash := sha256.New()
	hash.Write([]byte("ryvion:v7:model_benchmark_prompt:v1\n"))
	hash.Write([]byte("label:"))
	hash.Write([]byte(strings.TrimSpace(prompt.Label)))
	hash.Write([]byte("\ncontent:"))
	hash.Write(prompt.Content)
	return "sha256:" + hex.EncodeToString(hash.Sum(nil))
}

func normalizeModelBenchmarkSpec(spec ModelBenchmarkSpec) ModelBenchmarkSpec {
	spec.Task = strings.TrimSpace(spec.Task)
	spec.RequestID = strings.TrimSpace(spec.RequestID)
	spec.JobID = strings.TrimSpace(spec.JobID)
	spec.ModelID = strings.TrimSpace(spec.ModelID)
	spec.PromptProfileID = strings.TrimSpace(spec.PromptProfileID)
	spec.PromptLabel = strings.TrimSpace(spec.PromptLabel)
	spec.PromptHash = strings.TrimSpace(spec.PromptHash)
	if !benchmarkPromptBindingValid(spec) {
		spec.PromptHash = ""
	}
	return spec
}
