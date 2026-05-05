package memorybench

import (
	"context"
	"fmt"
	"time"
)

type ExecuteOptions struct {
	Getenv func(string) string
	Sleep  func(context.Context, time.Duration) error
}

func ExecuteBenchmarkAssignment(ctx context.Context, specJSON string, opts ExecuteOptions) (BenchmarkReceipt, bool, error) {
	if !IsBenchmarkSpecJSON(specJSON) {
		return BenchmarkReceipt{}, false, nil
	}
	if !BenchmarkEnabledFromEnv(opts.Getenv) {
		return BenchmarkReceipt{}, false, nil
	}
	spec, err := DecodeBenchmarkSpec(specJSON)
	if err != nil {
		return BenchmarkReceipt{}, true, err
	}
	receipt, err := ExecuteBenchmarkSpec(ctx, spec, opts)
	if err != nil {
		return BenchmarkReceipt{}, true, err
	}
	return receipt, true, nil
}

func ExecuteBenchmarkSpec(ctx context.Context, spec BenchmarkSpec, opts ExecuteOptions) (BenchmarkReceipt, error) {
	spec = normalizeBenchmarkSpec(spec)
	if err := ValidateBenchmarkSpec(spec); err != nil {
		return BenchmarkReceipt{}, err
	}
	nodeStartedAt := time.Now()
	if spec.SimulatedDelayMs > 0 {
		sleep := opts.Sleep
		if sleep == nil {
			sleep = sleepWithContext
		}
		if err := sleep(ctx, time.Duration(spec.SimulatedDelayMs)*time.Millisecond); err != nil {
			return BenchmarkReceipt{}, fmt.Errorf("%w: simulated delay: %v", ErrInvalidBenchmarkSpec, err)
		}
	}

	request := GenerateSyntheticAttentionRequest(spec.Seed, spec.ShardID, spec.TokenCount, spec.ValueDim)
	request.RequestID = spec.RequestID
	request.JobID = spec.JobID
	request.ShardID = spec.ShardID
	request.CreatedAtUnixMs = spec.CreatedAtUnixMs

	computeStartedAt := time.Now()
	response, err := ComputePartialAttentionSummary(request)
	if err != nil {
		return BenchmarkReceipt{}, err
	}
	computeCompletedAt := time.Now()
	computeDuration := computeCompletedAt.Sub(computeStartedAt)
	nodeWallDuration := computeCompletedAt.Sub(nodeStartedAt)
	summaryPayloadBytes := estimatePartialAttentionSummaryBytes(response.Summary)

	response.NodeStartedAtUnixMs = nodeStartedAt.UnixMilli()
	response.NodeCompletedAtUnixMs = computeCompletedAt.UnixMilli()
	response.ComputeTimeMs = nonNegativeDurationMilliseconds(computeDuration)
	response.ComputeTimeUs = nonNegativeDurationMicroseconds(computeDuration)
	response.SimulatedDelayMs = spec.SimulatedDelayMs
	response.TotalNodeWallTimeMs = nonNegativeDurationMilliseconds(nodeWallDuration)
	response.TotalNodeWallTimeUs = nonNegativeDurationMicroseconds(nodeWallDuration)
	response.SummaryPayloadBytesEstimate = summaryPayloadBytes
	response.CreatedAtUnixMs = spec.CreatedAtUnixMs
	response.OutputBytesEstimate = summaryPayloadBytes

	return BuildBenchmarkReceipt(spec, response)
}

func sleepWithContext(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return nil
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
