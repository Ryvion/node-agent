//go:build !containers

package executor

import (
	"context"
	"errors"
	"time"
)

// Stub types to satisfy callers when built without container support.

type WorkloadExecutor struct{}

type WorkResult struct {
	JobID      string
	ResultHash string
	OutputData []byte
	Metrics    ExecutionMetrics
	Error      error
}

type ExecutionMetrics struct {
	Duration       time.Duration
	GPUUtilization float64
	PowerUsage     float64
	TokensPerSec   float64
}

func NewWorkloadExecutor() (*WorkloadExecutor, error) {
	return nil, errors.New("containers build tag not enabled")
}

func (e *WorkloadExecutor) ExecuteInference(ctx context.Context, jobID, jobType, payloadURL string) (*WorkResult, error) {
	return nil, errors.New("containers build tag not enabled")
}
