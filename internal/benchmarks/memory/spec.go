package memorybench

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

const (
	BenchmarkTask    = "v7_memory_benchmark"
	BenchmarkFlagEnv = "RYV_NODE_V7_MEMORY_BENCH"

	maxBenchmarkTokenCount       = 1_000_000
	maxBenchmarkValueDim         = 4_096
	maxBenchmarkValueElements    = 16_000_000
	maxBenchmarkSimulatedDelayMs = 60_000
)

var (
	ErrInvalidBenchmarkSpec  = errors.New("memorybench: invalid benchmark spec")
	ErrUnsupportedBenchmark  = errors.New("memorybench: unsupported benchmark task")
	ErrBenchmarkExecutionOff = errors.New("memorybench: benchmark execution disabled")
)

type BenchmarkSpec struct {
	Task             string `json:"task"`
	RequestID        string `json:"request_id"`
	JobID            string `json:"job_id"`
	ShardID          string `json:"shard_id"`
	Seed             int64  `json:"seed"`
	TokenCount       int    `json:"token_count"`
	ValueDim         int    `json:"value_dim"`
	SimulatedDelayMs int64  `json:"simulated_delay_ms"`
	CreatedAtUnixMs  int64  `json:"created_at_unix_ms"`
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
	if spec.ShardID == "" {
		errs = append(errs, fmt.Errorf("%w: shard_id required", ErrInvalidBenchmarkSpec))
	}
	if spec.TokenCount <= 0 {
		errs = append(errs, fmt.Errorf("%w: token_count must be greater than zero", ErrInvalidBenchmarkSpec))
	} else if spec.TokenCount > maxBenchmarkTokenCount {
		errs = append(errs, fmt.Errorf("%w: token_count exceeds maximum %d", ErrInvalidBenchmarkSpec, maxBenchmarkTokenCount))
	}
	if spec.ValueDim <= 0 {
		errs = append(errs, fmt.Errorf("%w: value_dim must be greater than zero", ErrInvalidBenchmarkSpec))
	} else if spec.ValueDim > maxBenchmarkValueDim {
		errs = append(errs, fmt.Errorf("%w: value_dim exceeds maximum %d", ErrInvalidBenchmarkSpec, maxBenchmarkValueDim))
	}
	if spec.TokenCount > 0 && spec.ValueDim > 0 && spec.TokenCount > maxBenchmarkValueElements/spec.ValueDim {
		errs = append(errs, fmt.Errorf("%w: token_count * value_dim exceeds maximum %d", ErrInvalidBenchmarkSpec, maxBenchmarkValueElements))
	}
	if spec.SimulatedDelayMs < 0 {
		errs = append(errs, fmt.Errorf("%w: simulated_delay_ms must be non-negative", ErrInvalidBenchmarkSpec))
	} else if spec.SimulatedDelayMs > maxBenchmarkSimulatedDelayMs {
		errs = append(errs, fmt.Errorf("%w: simulated_delay_ms exceeds maximum %d", ErrInvalidBenchmarkSpec, maxBenchmarkSimulatedDelayMs))
	}
	if spec.CreatedAtUnixMs <= 0 {
		errs = append(errs, fmt.Errorf("%w: created_at_unix_ms must be greater than zero", ErrInvalidBenchmarkSpec))
	}
	return errors.Join(errs...)
}

func normalizeBenchmarkSpec(spec BenchmarkSpec) BenchmarkSpec {
	spec.Task = strings.TrimSpace(spec.Task)
	spec.RequestID = strings.TrimSpace(spec.RequestID)
	spec.JobID = strings.TrimSpace(spec.JobID)
	spec.ShardID = strings.TrimSpace(spec.ShardID)
	return spec
}
