package memorybench

import (
	"fmt"
	"math/rand"
)

const (
	syntheticRequestBaseUnixMs   int64  = 1_800_000_000_000
	syntheticTimestampModuloUnix uint64 = 86_400_000
)

func GenerateSyntheticAttentionRequest(seed int64, shardID string, tokenCount int, valueDim int) SyntheticAttentionRequest {
	request := SyntheticAttentionRequest{
		RequestID:       fmt.Sprintf("synthetic-attention-%d-%s-%d-%d", seed, shardID, tokenCount, valueDim),
		JobID:           fmt.Sprintf("synthetic-attention-job-%d", seed),
		ShardID:         shardID,
		Seed:            seed,
		ValueDim:        valueDim,
		CreatedAtUnixMs: syntheticCreatedAtUnixMs(seed),
	}
	if tokenCount <= 0 || valueDim <= 0 {
		return request
	}

	rng := rand.New(rand.NewSource(seed))
	request.Logits = make([]float64, tokenCount)
	request.Values = make([][]float64, tokenCount)
	for token := 0; token < tokenCount; token++ {
		request.Logits[token] = (rng.Float64() * 8) - 4
		value := make([]float64, valueDim)
		for dim := 0; dim < valueDim; dim++ {
			value[dim] = (rng.Float64() * 2) - 1
		}
		request.Values[token] = value
	}
	return request
}

func estimatePartialAttentionSummaryBytes(summary PartialAttentionSummary) int64 {
	const float64Bytes int64 = 8
	const intBytes int64 = 8
	return (2 * float64Bytes) +
		(int64(len(summary.WeightedValue)) * float64Bytes) +
		(2 * intBytes) +
		int64(len(summary.DType))
}

func syntheticCreatedAtUnixMs(seed int64) int64 {
	return syntheticRequestBaseUnixMs + int64(uint64(seed)%syntheticTimestampModuloUnix)
}
