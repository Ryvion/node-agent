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
	receipt, _, handled, err := ExecuteBenchmarkAssignmentWithReceiptTimings(ctx, specJSON, opts)
	return receipt, handled, err
}

func ExecuteBenchmarkAssignmentWithReceiptTimings(ctx context.Context, specJSON string, opts ExecuteOptions) (BenchmarkReceipt, ReceiptBuildTimings, bool, error) {
	if !IsBenchmarkSpecJSON(specJSON) {
		return BenchmarkReceipt{}, ReceiptBuildTimings{}, false, nil
	}
	if !BenchmarkEnabledFromEnv(opts.Getenv) {
		return BenchmarkReceipt{}, ReceiptBuildTimings{}, false, nil
	}
	spec, err := DecodeBenchmarkSpec(specJSON)
	if err != nil {
		return BenchmarkReceipt{}, ReceiptBuildTimings{}, true, err
	}
	receipt, timings, err := ExecuteBenchmarkSpecWithReceiptTimings(ctx, spec, opts)
	if err != nil {
		return BenchmarkReceipt{}, timings, true, err
	}
	return receipt, timings, true, nil
}

func ExecuteBenchmarkSpec(ctx context.Context, spec BenchmarkSpec, opts ExecuteOptions) (BenchmarkReceipt, error) {
	receipt, _, err := ExecuteBenchmarkSpecWithReceiptTimings(ctx, spec, opts)
	return receipt, err
}

func ExecuteBenchmarkSpecWithReceiptTimings(ctx context.Context, spec BenchmarkSpec, opts ExecuteOptions) (BenchmarkReceipt, ReceiptBuildTimings, error) {
	spec = normalizeBenchmarkSpec(spec)
	if err := ValidateBenchmarkSpec(spec); err != nil {
		return BenchmarkReceipt{}, ReceiptBuildTimings{}, err
	}
	nodeStartedAt := time.Now()
	if spec.SimulatedDelayMs > 0 {
		sleep := opts.Sleep
		if sleep == nil {
			sleep = sleepWithContext
		}
		if err := sleep(ctx, time.Duration(spec.SimulatedDelayMs)*time.Millisecond); err != nil {
			return BenchmarkReceipt{}, ReceiptBuildTimings{}, fmt.Errorf("%w: simulated delay: %v", ErrInvalidBenchmarkSpec, err)
		}
	}

	computeStartedAt := time.Now()
	response, err := ComputeSyntheticPartialAttentionStreaming(spec)
	if err != nil {
		return BenchmarkReceipt{}, ReceiptBuildTimings{}, err
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

	return BuildBenchmarkReceiptWithTimings(spec, response)
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
