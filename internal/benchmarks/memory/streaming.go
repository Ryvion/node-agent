package memorybench

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"time"
)

type computeMemoryDeltas struct {
	allocBytesDelta      int64
	totalAllocBytesDelta int64
	mallocsDelta         int64
	numGCDelta           int64
	gcPauseTotalUsDelta  int64
}

func ComputePartialAttentionSummaryFromSeed(seed int64, shardID string, tokenCount int, valueDim int) (SyntheticAttentionResponse, error) {
	return computePartialAttentionSummaryFromSeedWithMetadata(
		seed,
		shardID,
		tokenCount,
		valueDim,
		fmt.Sprintf("synthetic-attention-%d-%s-%d-%d", seed, shardID, tokenCount, valueDim),
		fmt.Sprintf("synthetic-attention-job-%d", seed),
		syntheticCreatedAtUnixMs(seed),
	)
}

func ComputeSyntheticPartialAttentionStreaming(spec BenchmarkSpec) (SyntheticAttentionResponse, error) {
	spec = normalizeBenchmarkSpec(spec)
	if err := ValidateBenchmarkSpec(spec); err != nil {
		return SyntheticAttentionResponse{}, err
	}
	return computePartialAttentionSummaryFromSeedWithMetadata(
		spec.Seed,
		spec.ShardID,
		spec.TokenCount,
		spec.ValueDim,
		spec.RequestID,
		spec.JobID,
		spec.CreatedAtUnixMs,
	)
}

func computePartialAttentionSummaryFromSeedWithMetadata(seed int64, shardID string, tokenCount int, valueDim int, requestID string, jobID string, createdAtUnixMs int64) (SyntheticAttentionResponse, error) {
	startedAt := time.Now()
	if err := validateSyntheticAttentionSeedInput(tokenCount, valueDim); err != nil {
		return SyntheticAttentionResponse{}, err
	}

	before := readComputeMemStats()
	summary, err := computePartialAttentionSummaryFromSeed(seed, tokenCount, valueDim)
	after := readComputeMemStats()
	if err != nil {
		return SyntheticAttentionResponse{}, err
	}

	completedAt := time.Now()
	computeDuration := completedAt.Sub(startedAt)
	computeTimeMs := nonNegativeDurationMilliseconds(computeDuration)
	computeTimeUs := nonNegativeDurationMicroseconds(computeDuration)
	if createdAtUnixMs <= 0 {
		createdAtUnixMs = startedAt.UnixMilli()
	}
	summaryPayloadBytes := estimatePartialAttentionSummaryBytes(summary)
	deltas := computeMemStatsDelta(before, after)

	return SyntheticAttentionResponse{
		RequestID:                   requestID,
		JobID:                       jobID,
		ShardID:                     shardID,
		Summary:                     summary,
		NodeStartedAtUnixMs:         startedAt.UnixMilli(),
		NodeCompletedAtUnixMs:       completedAt.UnixMilli(),
		ComputeTimeMs:               computeTimeMs,
		ComputeTimeUs:               computeTimeUs,
		TotalNodeWallTimeMs:         computeTimeMs,
		TotalNodeWallTimeUs:         computeTimeUs,
		SummaryPayloadBytesEstimate: summaryPayloadBytes,
		OutputBytesEstimate:         summaryPayloadBytes,
		CreatedAtUnixMs:             createdAtUnixMs,
		ComputeAllocBytesDelta:      deltas.allocBytesDelta,
		ComputeTotalAllocBytesDelta: deltas.totalAllocBytesDelta,
		ComputeMallocsDelta:         deltas.mallocsDelta,
		ComputeNumGCDelta:           deltas.numGCDelta,
		ComputeGCPauseTotalUsDelta:  deltas.gcPauseTotalUsDelta,
	}, nil
}

func validateSyntheticAttentionSeedInput(tokenCount int, valueDim int) error {
	var errs []error
	if tokenCount <= 0 {
		errs = append(errs, fmt.Errorf("%w: token_count must be greater than zero", ErrInvalidSyntheticAttentionRequest))
	}
	if valueDim <= 0 {
		errs = append(errs, fmt.Errorf("%w: value_dim must be greater than zero", ErrInvalidSyntheticAttentionRequest))
	}
	return errors.Join(errs...)
}

func computePartialAttentionSummaryFromSeed(seed int64, tokenCount int, valueDim int) (PartialAttentionSummary, error) {
	rng := rand.New(rand.NewSource(seed))
	localMax := math.Inf(-1)
	for token := 0; token < tokenCount; token++ {
		logit := nextSyntheticAttentionLogit(rng)
		if logit > localMax {
			localMax = logit
		}
		for dim := 0; dim < valueDim; dim++ {
			_ = nextSyntheticAttentionValue(rng)
		}
	}

	rng = rand.New(rand.NewSource(seed))
	expSum := 0.0
	weightedValue := make([]float64, valueDim)
	for token := 0; token < tokenCount; token++ {
		logit := nextSyntheticAttentionLogit(rng)
		weight := math.Exp(logit - localMax)
		if !finiteFloat64(weight) {
			return PartialAttentionSummary{}, fmt.Errorf("%w: exponent overflow at token %d", ErrInvalidSyntheticAttentionRequest, token)
		}
		expSum += weight
		for dim := 0; dim < valueDim; dim++ {
			value := nextSyntheticAttentionValue(rng)
			weightedValue[dim] += weight * value
			if !finiteFloat64(weightedValue[dim]) {
				return PartialAttentionSummary{}, fmt.Errorf("%w: weighted value overflow at value_dim %d", ErrInvalidSyntheticAttentionRequest, dim)
			}
		}
	}
	if expSum <= 0 || !finiteFloat64(expSum) {
		return PartialAttentionSummary{}, fmt.Errorf("%w: exp_sum must be positive and finite", ErrInvalidSyntheticAttentionRequest)
	}

	return PartialAttentionSummary{
		LocalMax:      localMax,
		ExpSum:        expSum,
		WeightedValue: weightedValue,
		TokenCount:    tokenCount,
		ValueDim:      valueDim,
		DType:         SyntheticAttentionDTypeFloat64,
	}, nil
}

func nextSyntheticAttentionLogit(rng *rand.Rand) float64 {
	return (rng.Float64() * 8) - 4
}

func nextSyntheticAttentionValue(rng *rand.Rand) float64 {
	return (rng.Float64() * 2) - 1
}

func readComputeMemStats() runtime.MemStats {
	var stats runtime.MemStats
	runtime.ReadMemStats(&stats)
	return stats
}

func computeMemStatsDelta(before, after runtime.MemStats) computeMemoryDeltas {
	return computeMemoryDeltas{
		allocBytesDelta:      signedUint64Delta(after.Alloc, before.Alloc),
		totalAllocBytesDelta: monotonicUint64Delta(after.TotalAlloc, before.TotalAlloc),
		mallocsDelta:         monotonicUint64Delta(after.Mallocs, before.Mallocs),
		numGCDelta:           monotonicUint32Delta(after.NumGC, before.NumGC),
		gcPauseTotalUsDelta:  monotonicUint64Delta(after.PauseTotalNs, before.PauseTotalNs) / 1000,
	}
}

func signedUint64Delta(after, before uint64) int64 {
	if after >= before {
		return uint64ToInt64(after - before)
	}
	return -uint64ToInt64(before - after)
}

func monotonicUint64Delta(after, before uint64) int64 {
	if after <= before {
		return 0
	}
	return uint64ToInt64(after - before)
}

func monotonicUint32Delta(after, before uint32) int64 {
	if after <= before {
		return 0
	}
	return int64(after - before)
}

func uint64ToInt64(value uint64) int64 {
	const maxInt64 = uint64(^uint64(0) >> 1)
	if value > maxInt64 {
		return int64(maxInt64)
	}
	return int64(value)
}
