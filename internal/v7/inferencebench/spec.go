package inferencebench

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/Ryvion/node-agent/internal/v7/llamacpp"
)

const (
	BenchmarkTask    = "v7_backend_inference_benchmark"
	BenchmarkFlagEnv = "RYV_NODE_V7_INFERENCE_BENCH"

	maxBenchmarkIDLen      = 256
	maxBenchmarkModelIDLen = 256
	maxBenchmarkMaxTokens  = 8_192
	maxBenchmarkTimeoutMs  = int64(10 * 60 * 1000)
)

var ErrInvalidBenchmarkSpec = errors.New("inferencebench: invalid benchmark spec")

type BenchmarkSpec struct {
	Task            string `json:"task"`
	RequestID       string `json:"request_id"`
	JobID           string `json:"job_id"`
	Backend         string `json:"backend"`
	ModelID         string `json:"model_id"`
	PromptHash      string `json:"prompt_hash"`
	MaxTokens       int    `json:"max_tokens"`
	TimeoutMs       int64  `json:"timeout_ms"`
	CreatedAtUnixMs int64  `json:"created_at_unix_ms"`
}

type BenchmarkAssignmentIdentity struct {
	JobID     string
	RequestID string
}

func IsBenchmarkSpecJSON(specJSON string) bool {
	var header struct {
		Task string `json:"task"`
	}
	if json.Unmarshal([]byte(strings.TrimSpace(specJSON)), &header) != nil {
		return false
	}
	return strings.TrimSpace(header.Task) == BenchmarkTask
}

func BenchmarkEnabledFromEnv(getenv func(string) string) bool {
	if getenv == nil {
		return false
	}
	return strings.TrimSpace(getenv(BenchmarkFlagEnv)) == "1"
}

func BenchmarkAssignmentIdentityFromJSON(specJSON string) (BenchmarkAssignmentIdentity, bool) {
	var header struct {
		Task      string `json:"task"`
		JobID     string `json:"job_id"`
		RequestID string `json:"request_id"`
	}
	if json.Unmarshal([]byte(strings.TrimSpace(specJSON)), &header) != nil {
		return BenchmarkAssignmentIdentity{}, false
	}
	if strings.TrimSpace(header.Task) != BenchmarkTask {
		return BenchmarkAssignmentIdentity{}, false
	}
	return BenchmarkAssignmentIdentity{
		JobID:     cleanBenchmarkText(header.JobID, maxBenchmarkIDLen),
		RequestID: cleanBenchmarkText(header.RequestID, maxBenchmarkIDLen),
	}, true
}

func DecodeBenchmarkSpec(specJSON string) (BenchmarkSpec, error) {
	raw := strings.TrimSpace(specJSON)
	if raw == "" {
		return BenchmarkSpec{}, fmt.Errorf("%w: spec_json required", ErrInvalidBenchmarkSpec)
	}

	var spec BenchmarkSpec
	if err := json.Unmarshal([]byte(raw), &spec); err != nil {
		return BenchmarkSpec{}, fmt.Errorf("%w: decode spec_json: %v", ErrInvalidBenchmarkSpec, err)
	}
	spec = normalizeBenchmarkSpec(spec)
	if spec.Task != BenchmarkTask {
		return BenchmarkSpec{}, fmt.Errorf("%w: task must be %q", ErrInvalidBenchmarkSpec, BenchmarkTask)
	}
	if err := ValidateBenchmarkSpec(spec); err != nil {
		return BenchmarkSpec{}, err
	}
	return spec, nil
}

func ValidateBenchmarkSpec(spec BenchmarkSpec) error {
	spec = normalizeBenchmarkSpec(spec)

	var errs []error
	if spec.Task == "" {
		errs = append(errs, fmt.Errorf("%w: task required", ErrInvalidBenchmarkSpec))
	} else if spec.Task != BenchmarkTask {
		errs = append(errs, fmt.Errorf("%w: task must be %q", ErrInvalidBenchmarkSpec, BenchmarkTask))
	}
	if spec.RequestID == "" {
		errs = append(errs, fmt.Errorf("%w: request_id required", ErrInvalidBenchmarkSpec))
	}
	if spec.JobID == "" {
		errs = append(errs, fmt.Errorf("%w: job_id required", ErrInvalidBenchmarkSpec))
	}
	if spec.Backend == "" {
		errs = append(errs, fmt.Errorf("%w: backend required", ErrInvalidBenchmarkSpec))
	} else if spec.Backend != llamacpp.BackendName {
		errs = append(errs, fmt.Errorf("%w: backend must be %q", ErrInvalidBenchmarkSpec, llamacpp.BackendName))
	}
	if spec.ModelID == "" {
		errs = append(errs, fmt.Errorf("%w: model_id required", ErrInvalidBenchmarkSpec))
	}
	if err := validateSHA256ID(spec.PromptHash, "prompt_hash", ErrInvalidBenchmarkSpec); err != nil {
		errs = append(errs, err)
	} else if spec.PromptHash != llamacpp.HashBenchmarkPrompt() {
		errs = append(errs, fmt.Errorf("%w: prompt_hash must match node-agent internal benchmark prompt", ErrInvalidBenchmarkSpec))
	}
	if spec.MaxTokens <= 0 {
		errs = append(errs, fmt.Errorf("%w: max_tokens must be greater than zero", ErrInvalidBenchmarkSpec))
	} else if spec.MaxTokens > maxBenchmarkMaxTokens {
		errs = append(errs, fmt.Errorf("%w: max_tokens exceeds maximum %d", ErrInvalidBenchmarkSpec, maxBenchmarkMaxTokens))
	}
	if spec.TimeoutMs <= 0 {
		errs = append(errs, fmt.Errorf("%w: timeout_ms must be greater than zero", ErrInvalidBenchmarkSpec))
	} else if spec.TimeoutMs > maxBenchmarkTimeoutMs {
		errs = append(errs, fmt.Errorf("%w: timeout_ms exceeds maximum %d", ErrInvalidBenchmarkSpec, maxBenchmarkTimeoutMs))
	}
	if spec.CreatedAtUnixMs <= 0 {
		errs = append(errs, fmt.Errorf("%w: created_at_unix_ms must be greater than zero", ErrInvalidBenchmarkSpec))
	}
	return errors.Join(errs...)
}

func normalizeBenchmarkSpec(spec BenchmarkSpec) BenchmarkSpec {
	spec.Task = strings.TrimSpace(spec.Task)
	spec.RequestID = cleanBenchmarkText(spec.RequestID, maxBenchmarkIDLen)
	spec.JobID = cleanBenchmarkText(spec.JobID, maxBenchmarkIDLen)
	spec.Backend = normalizeBackendName(spec.Backend)
	spec.ModelID = cleanBenchmarkText(spec.ModelID, maxBenchmarkModelIDLen)
	spec.PromptHash = strings.TrimSpace(spec.PromptHash)
	return spec
}

func normalizeBackendName(value string) string {
	value = strings.TrimSpace(value)
	normalized := strings.ToLower(value)
	normalized = strings.NewReplacer(".", "", "_", "", "-", "", " ", "").Replace(normalized)
	if normalized == "llamacpp" {
		return llamacpp.BackendName
	}
	return cleanBenchmarkText(value, maxBenchmarkIDLen)
}

func validateSHA256ID(value string, field string, base error) error {
	value = strings.TrimSpace(value)
	if value == "" {
		return fmt.Errorf("%w: %s required", base, field)
	}
	if !strings.HasPrefix(value, "sha256:") || len(value) != len("sha256:")+64 {
		return fmt.Errorf("%w: %s must be sha256:<64 hex>", base, field)
	}
	for _, r := range strings.TrimPrefix(value, "sha256:") {
		if (r < '0' || r > '9') && (r < 'a' || r > 'f') {
			return fmt.Errorf("%w: %s must be lowercase hex", base, field)
		}
	}
	return nil
}

func cleanBenchmarkText(value string, maxRunes int) string {
	value = strings.Join(strings.Fields(value), " ")
	if maxRunes <= 0 {
		return value
	}
	runes := []rune(value)
	if len(runes) <= maxRunes {
		return value
	}
	return string(runes[:maxRunes])
}
