package llamacpp

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

const (
	BackendBenchmarkTask    = "v7_llamacpp_backend_benchmark"
	BackendBenchmarkFlagEnv = "RYV_NODE_V7_BACKEND_BENCH"

	maxBackendBenchmarkIDLen      = 256
	maxBackendBenchmarkRuns       = 100
	maxBackendBenchmarkMaxTokens  = 8_192
	maxBackendBenchmarkTimeoutMs  = int64(10 * 60 * 1000)
	maxBackendBenchmarkModelIDLen = 256
)

var ErrInvalidBackendBenchmarkSpec = errors.New("llamacpp: invalid backend benchmark spec")

type BackendBenchmarkSpec struct {
	Task            string `json:"task"`
	RequestID       string `json:"request_id"`
	JobID           string `json:"job_id"`
	Backend         string `json:"backend"`
	ModelID         string `json:"model_id"`
	MaxTokens       int    `json:"max_tokens"`
	WarmupRuns      int    `json:"warmup_runs"`
	MeasuredRuns    int    `json:"measured_runs"`
	TimeoutMs       int64  `json:"timeout_ms"`
	CreatedAtUnixMs int64  `json:"created_at_unix_ms"`
}

type BackendBenchmarkAssignmentIdentity struct {
	JobID     string
	RequestID string
}

func IsBackendBenchmarkSpecJSON(specJSON string) bool {
	var header struct {
		Task string `json:"task"`
	}
	if json.Unmarshal([]byte(strings.TrimSpace(specJSON)), &header) != nil {
		return false
	}
	return strings.TrimSpace(header.Task) == BackendBenchmarkTask
}

func BackendBenchmarkEnabledFromEnv(getenv func(string) string) bool {
	if getenv == nil {
		return true
	}
	switch strings.ToLower(strings.TrimSpace(getenv(BackendBenchmarkFlagEnv))) {
	case "0", "false", "no", "off", "disabled":
		return false
	default:
		return true
	}
}

func BackendBenchmarkAssignmentIdentityFromJSON(specJSON string) (BackendBenchmarkAssignmentIdentity, bool) {
	var header struct {
		Task      string `json:"task"`
		JobID     string `json:"job_id"`
		RequestID string `json:"request_id"`
	}
	if json.Unmarshal([]byte(strings.TrimSpace(specJSON)), &header) != nil {
		return BackendBenchmarkAssignmentIdentity{}, false
	}
	if strings.TrimSpace(header.Task) != BackendBenchmarkTask {
		return BackendBenchmarkAssignmentIdentity{}, false
	}
	return BackendBenchmarkAssignmentIdentity{
		JobID:     cleanStatusText(header.JobID, maxBackendBenchmarkIDLen),
		RequestID: cleanStatusText(header.RequestID, maxBackendBenchmarkIDLen),
	}, true
}

func DecodeBackendBenchmarkSpec(specJSON string) (BackendBenchmarkSpec, error) {
	raw := strings.TrimSpace(specJSON)
	if raw == "" {
		return BackendBenchmarkSpec{}, fmt.Errorf("%w: spec_json required", ErrInvalidBackendBenchmarkSpec)
	}

	var spec BackendBenchmarkSpec
	if err := json.Unmarshal([]byte(raw), &spec); err != nil {
		return BackendBenchmarkSpec{}, fmt.Errorf("%w: decode spec_json: %v", ErrInvalidBackendBenchmarkSpec, err)
	}
	spec = normalizeBackendBenchmarkSpec(spec)
	if spec.Task != BackendBenchmarkTask {
		return BackendBenchmarkSpec{}, fmt.Errorf("%w: task must be %q", ErrInvalidBackendBenchmarkSpec, BackendBenchmarkTask)
	}
	if err := ValidateBackendBenchmarkSpec(spec); err != nil {
		return BackendBenchmarkSpec{}, err
	}
	return spec, nil
}

func ValidateBackendBenchmarkSpec(spec BackendBenchmarkSpec) error {
	spec = normalizeBackendBenchmarkSpec(spec)

	var errs []error
	if spec.Task == "" {
		errs = append(errs, fmt.Errorf("%w: task required", ErrInvalidBackendBenchmarkSpec))
	} else if spec.Task != BackendBenchmarkTask {
		errs = append(errs, fmt.Errorf("%w: task must be %q", ErrInvalidBackendBenchmarkSpec, BackendBenchmarkTask))
	}
	if spec.RequestID == "" {
		errs = append(errs, fmt.Errorf("%w: request_id required", ErrInvalidBackendBenchmarkSpec))
	}
	if spec.JobID == "" {
		errs = append(errs, fmt.Errorf("%w: job_id required", ErrInvalidBackendBenchmarkSpec))
	}
	if spec.Backend == "" {
		errs = append(errs, fmt.Errorf("%w: backend required", ErrInvalidBackendBenchmarkSpec))
	} else if spec.Backend != BackendName {
		errs = append(errs, fmt.Errorf("%w: backend must be %q", ErrInvalidBackendBenchmarkSpec, BackendName))
	}
	if spec.ModelID == "" {
		errs = append(errs, fmt.Errorf("%w: model_id required", ErrInvalidBackendBenchmarkSpec))
	}
	if spec.MaxTokens <= 0 {
		errs = append(errs, fmt.Errorf("%w: max_tokens must be greater than zero", ErrInvalidBackendBenchmarkSpec))
	} else if spec.MaxTokens > maxBackendBenchmarkMaxTokens {
		errs = append(errs, fmt.Errorf("%w: max_tokens exceeds maximum %d", ErrInvalidBackendBenchmarkSpec, maxBackendBenchmarkMaxTokens))
	}
	if spec.WarmupRuns < 0 {
		errs = append(errs, fmt.Errorf("%w: warmup_runs must be non-negative", ErrInvalidBackendBenchmarkSpec))
	} else if spec.WarmupRuns > maxBackendBenchmarkRuns {
		errs = append(errs, fmt.Errorf("%w: warmup_runs exceeds maximum %d", ErrInvalidBackendBenchmarkSpec, maxBackendBenchmarkRuns))
	}
	if spec.MeasuredRuns <= 0 {
		errs = append(errs, fmt.Errorf("%w: measured_runs must be greater than zero", ErrInvalidBackendBenchmarkSpec))
	} else if spec.MeasuredRuns > maxBackendBenchmarkRuns {
		errs = append(errs, fmt.Errorf("%w: measured_runs exceeds maximum %d", ErrInvalidBackendBenchmarkSpec, maxBackendBenchmarkRuns))
	}
	if spec.TimeoutMs <= 0 {
		errs = append(errs, fmt.Errorf("%w: timeout_ms must be greater than zero", ErrInvalidBackendBenchmarkSpec))
	} else if spec.TimeoutMs > maxBackendBenchmarkTimeoutMs {
		errs = append(errs, fmt.Errorf("%w: timeout_ms exceeds maximum %d", ErrInvalidBackendBenchmarkSpec, maxBackendBenchmarkTimeoutMs))
	}
	if spec.CreatedAtUnixMs <= 0 {
		errs = append(errs, fmt.Errorf("%w: created_at_unix_ms must be greater than zero", ErrInvalidBackendBenchmarkSpec))
	}
	return errors.Join(errs...)
}

func normalizeBackendBenchmarkSpec(spec BackendBenchmarkSpec) BackendBenchmarkSpec {
	spec.Task = strings.TrimSpace(spec.Task)
	spec.RequestID = cleanStatusText(spec.RequestID, maxBackendBenchmarkIDLen)
	spec.JobID = cleanStatusText(spec.JobID, maxBackendBenchmarkIDLen)
	spec.Backend = normalizeBackendBenchmarkName(spec.Backend)
	spec.ModelID = cleanStatusText(spec.ModelID, maxBackendBenchmarkModelIDLen)
	return spec
}

func normalizeBackendBenchmarkName(value string) string {
	value = strings.TrimSpace(value)
	normalized := strings.ToLower(value)
	normalized = strings.NewReplacer(".", "", "_", "", "-", "", " ", "").Replace(normalized)
	if normalized == "llamacpp" {
		return BackendName
	}
	return cleanStatusText(value, maxStatusReasonLen)
}
