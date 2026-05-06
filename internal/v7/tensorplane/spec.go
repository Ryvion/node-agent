package tensorplane

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

const (
	BenchmarkTask    = "v7_tensorplane_benchmark"
	BenchmarkFlagEnv = "RYV_NODE_V7_TENSORPLANE_BENCH"

	maxBenchmarkTokens         = 1_000_000
	maxBenchmarkDim            = 4_096
	maxBenchmarkTensorElements = 16_000_000
	maxBenchmarkTimeoutMs      = 60_000
)

var (
	ErrInvalidBenchmarkSpec = errors.New("tensorplane: invalid benchmark spec")
	ErrUnsupportedBenchmark = errors.New("tensorplane: unsupported benchmark task")
)

type BenchmarkSpec struct {
	Task            string      `json:"task"`
	RequestID       string      `json:"request_id"`
	JobID           string      `json:"job_id"`
	ModelID         string      `json:"model_id"`
	LayerIndex      int         `json:"layer_index"`
	DType           TensorDType `json:"dtype"`
	Tokens          int         `json:"tokens"`
	HeadDim         int         `json:"head_dim"`
	ValueDim        int         `json:"value_dim"`
	Seed            int64       `json:"seed"`
	TimeoutMs       int64       `json:"timeout_ms"`
	CreatedAtUnixMs int64       `json:"created_at_unix_ms"`
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
		JobID:     cleanTensorPlaneLocalStatusText(header.JobID, maxLocalStatusIDLen),
		RequestID: cleanTensorPlaneLocalStatusText(header.RequestID, maxLocalStatusIDLen),
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
		return BenchmarkSpec{}, fmt.Errorf("%w: task must be %q", ErrUnsupportedBenchmark, BenchmarkTask)
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
		errs = append(errs, fmt.Errorf("%w: task must be %q", ErrUnsupportedBenchmark, BenchmarkTask))
	}
	if spec.RequestID == "" {
		errs = append(errs, fmt.Errorf("%w: request_id required", ErrInvalidBenchmarkSpec))
	}
	if spec.JobID == "" {
		errs = append(errs, fmt.Errorf("%w: job_id required", ErrInvalidBenchmarkSpec))
	}
	if spec.ModelID == "" {
		errs = append(errs, fmt.Errorf("%w: model_id required", ErrInvalidBenchmarkSpec))
	}
	if spec.LayerIndex < 0 {
		errs = append(errs, fmt.Errorf("%w: layer_index must be non-negative", ErrInvalidBenchmarkSpec))
	}
	if err := ValidateTensorDType(spec.DType); err != nil {
		errs = append(errs, err)
	}
	if spec.Tokens <= 0 {
		errs = append(errs, fmt.Errorf("%w: tokens must be greater than zero", ErrInvalidBenchmarkSpec))
	} else if spec.Tokens > maxBenchmarkTokens {
		errs = append(errs, fmt.Errorf("%w: tokens exceeds maximum %d", ErrInvalidBenchmarkSpec, maxBenchmarkTokens))
	}
	if spec.HeadDim <= 0 {
		errs = append(errs, fmt.Errorf("%w: head_dim must be greater than zero", ErrInvalidBenchmarkSpec))
	} else if spec.HeadDim > maxBenchmarkDim {
		errs = append(errs, fmt.Errorf("%w: head_dim exceeds maximum %d", ErrInvalidBenchmarkSpec, maxBenchmarkDim))
	}
	if spec.ValueDim <= 0 {
		errs = append(errs, fmt.Errorf("%w: value_dim must be greater than zero", ErrInvalidBenchmarkSpec))
	} else if spec.ValueDim > maxBenchmarkDim {
		errs = append(errs, fmt.Errorf("%w: value_dim exceeds maximum %d", ErrInvalidBenchmarkSpec, maxBenchmarkDim))
	}
	if spec.TimeoutMs <= 0 {
		errs = append(errs, fmt.Errorf("%w: timeout_ms must be greater than zero", ErrInvalidBenchmarkSpec))
	} else if spec.TimeoutMs > maxBenchmarkTimeoutMs {
		errs = append(errs, fmt.Errorf("%w: timeout_ms exceeds maximum %d", ErrInvalidBenchmarkSpec, maxBenchmarkTimeoutMs))
	}
	if spec.CreatedAtUnixMs <= 0 {
		errs = append(errs, fmt.Errorf("%w: created_at_unix_ms must be greater than zero", ErrInvalidBenchmarkSpec))
	}
	if spec.Tokens > 0 && spec.HeadDim > 0 && spec.Tokens > maxBenchmarkTensorElements/spec.HeadDim {
		errs = append(errs, fmt.Errorf("%w: tokens * head_dim exceeds maximum %d", ErrInvalidBenchmarkSpec, maxBenchmarkTensorElements))
	}
	if spec.Tokens > 0 && spec.ValueDim > 0 && spec.Tokens > maxBenchmarkTensorElements/spec.ValueDim {
		errs = append(errs, fmt.Errorf("%w: tokens * value_dim exceeds maximum %d", ErrInvalidBenchmarkSpec, maxBenchmarkTensorElements))
	}
	if spec.Tokens > 0 && spec.HeadDim > 0 {
		if elements, ok := checkedMultiply(spec.Tokens, spec.HeadDim); !ok {
			errs = append(errs, fmt.Errorf("%w: key tensor element count overflow", ErrInvalidBenchmarkSpec))
		} else if elementBytes, err := tensorDTypeElementBytes(spec.DType); err == nil {
			if _, ok := checkedMultiply(elements, elementBytes); !ok {
				errs = append(errs, fmt.Errorf("%w: key tensor byte count overflow", ErrInvalidBenchmarkSpec))
			}
		}
	}
	if spec.Tokens > 0 && spec.ValueDim > 0 {
		if elements, ok := checkedMultiply(spec.Tokens, spec.ValueDim); !ok {
			errs = append(errs, fmt.Errorf("%w: value tensor element count overflow", ErrInvalidBenchmarkSpec))
		} else if elementBytes, err := tensorDTypeElementBytes(spec.DType); err == nil {
			if _, ok := checkedMultiply(elements, elementBytes); !ok {
				errs = append(errs, fmt.Errorf("%w: value tensor byte count overflow", ErrInvalidBenchmarkSpec))
			}
		}
	}
	return errors.Join(errs...)
}

func normalizeBenchmarkSpec(spec BenchmarkSpec) BenchmarkSpec {
	spec.Task = strings.TrimSpace(spec.Task)
	spec.RequestID = strings.TrimSpace(spec.RequestID)
	spec.JobID = strings.TrimSpace(spec.JobID)
	spec.ModelID = strings.TrimSpace(spec.ModelID)
	spec.DType = NormalizeTensorDType(spec.DType)
	return spec
}
