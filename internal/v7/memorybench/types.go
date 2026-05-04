package memorybench

import "errors"

const SyntheticAttentionDTypeFloat64 = "float64"

var ErrInvalidSyntheticAttentionRequest = errors.New("memorybench: invalid synthetic attention request")

type SyntheticAttentionRequest struct {
	RequestID       string      `json:"request_id"`
	JobID           string      `json:"job_id"`
	ShardID         string      `json:"shard_id"`
	Seed            int64       `json:"seed"`
	Logits          []float64   `json:"logits"`
	Values          [][]float64 `json:"values"`
	ValueDim        int         `json:"value_dim"`
	CreatedAtUnixMs int64       `json:"created_at_unix_ms"`
}

type SyntheticAttentionResponse struct {
	RequestID           string                  `json:"request_id"`
	JobID               string                  `json:"job_id"`
	ShardID             string                  `json:"shard_id"`
	Summary             PartialAttentionSummary `json:"summary"`
	ComputeTimeMs       int64                   `json:"compute_time_ms"`
	OutputBytesEstimate int64                   `json:"output_bytes_estimate"`
	CreatedAtUnixMs     int64                   `json:"created_at_unix_ms"`
}

type PartialAttentionSummary struct {
	LocalMax      float64   `json:"local_max"`
	ExpSum        float64   `json:"exp_sum"`
	WeightedValue []float64 `json:"weighted_value"`
	TokenCount    int       `json:"token_count"`
	ValueDim      int       `json:"value_dim"`
	DType         string    `json:"dtype"`
}
