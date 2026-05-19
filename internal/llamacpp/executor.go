package llamacpp

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type Execution struct {
	ResultHashHex string
	MeteringUnits uint64
	OutputPath    string
	Metadata      map[string]any
}

func Execute(ctx context.Context, specJSON string, cfg Config, client Client, workMeta map[string]any) (Execution, error) {
	spec, err := DecodeSpec(specJSON)
	if err != nil {
		return Execution{}, err
	}
	timeout := cfg.HTTPTimeout
	if spec.TimeoutSeconds > 0 {
		timeout = time.Duration(spec.TimeoutSeconds) * time.Second
	}
	if timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
		cfg.HTTPTimeout = timeout
	}
	result, err := client.Complete(ctx, cfg, spec)
	if err != nil {
		return Execution{}, err
	}
	metadata := BuildReceiptMetadata(result, workMeta)
	outputPath := ""
	if spec.OutputArtifact {
		outputPath, err = writeOutputArtifact(result.Text)
		if err != nil {
			metadata["artifact_error"] = err.Error()
		}
	}
	return Execution{
		ResultHashHex: result.OutputHash,
		MeteringUnits: meteringUnits(result),
		OutputPath:    outputPath,
		Metadata:      metadata,
	}, nil
}

func BuildReceiptMetadata(result Result, workMeta map[string]any) map[string]any {
	metadata := map[string]any{
		"schema":            Schema,
		"executor":          ExecutorKind,
		"model":             result.Model,
		"prompt_hash":       result.PromptHash,
		"output_hash":       result.OutputHash,
		"output_bytes":      result.OutputBytes,
		"prompt_tokens":     result.PromptTokens,
		"completion_tokens": result.CompletionTokens,
		"total_tokens":      result.TotalTokens,
		"finish_reason":     result.FinishReason,
		"duration_ms":       result.DurationMs,
	}
	for key, value := range workMeta {
		if key != "" && value != nil {
			metadata[key] = value
		}
	}
	return SanitizeMetadata(metadata)
}

func SanitizeMetadata(metadata map[string]any) map[string]any {
	out := make(map[string]any, len(metadata))
	for key, value := range metadata {
		if unsafeMetadataKey(key) {
			continue
		}
		switch typed := value.(type) {
		case map[string]any:
			out[key] = SanitizeMetadata(typed)
		default:
			out[key] = value
		}
	}
	return out
}

func unsafeMetadataKey(key string) bool {
	key = strings.ToLower(strings.TrimSpace(key))
	switch key {
	case "prompt", "messages", "output", "output_text", "generated_text", "content", "response", "raw", "raw_output", "logprobs":
		return true
	default:
		return false
	}
}

func meteringUnits(result Result) uint64 {
	if result.CompletionTokens > 0 {
		return uint64(result.CompletionTokens)
	}
	if result.TotalTokens > 0 {
		return uint64(result.TotalTokens)
	}
	return 1
}

func writeOutputArtifact(text string) (string, error) {
	dir, err := os.MkdirTemp("", "ryvion-llamacpp-*")
	if err != nil {
		return "", err
	}
	path := filepath.Join(dir, "output.json")
	body := struct {
		Schema     string `json:"schema"`
		OutputText string `json:"output_text"`
		OutputHash string `json:"output_hash"`
	}{
		Schema:     Schema,
		OutputText: text,
		OutputHash: OutputHash(text),
	}
	encoded, err := json.Marshal(body)
	if err != nil {
		return "", err
	}
	if err := os.WriteFile(path, encoded, 0o600); err != nil {
		return "", err
	}
	return path, nil
}

func FailureHash(jobID, reason string) string {
	sum := sha256.Sum256([]byte(fmt.Sprintf("%s:%s", jobID, reason)))
	return hex.EncodeToString(sum[:])
}
