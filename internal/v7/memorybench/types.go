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
	RequestID                   string                  `json:"request_id"`
	JobID                       string                  `json:"job_id"`
	ShardID                     string                  `json:"shard_id"`
	Summary                     PartialAttentionSummary `json:"summary"`
	NodeStartedAtUnixMs         int64                   `json:"node_started_at_unix_ms"`
	NodeCompletedAtUnixMs       int64                   `json:"node_completed_at_unix_ms"`
	ComputeTimeMs               int64                   `json:"compute_time_ms"`
	ComputeTimeUs               int64                   `json:"compute_time_us"`
	SimulatedDelayMs            int64                   `json:"simulated_delay_ms"`
	TotalNodeWallTimeMs         int64                   `json:"total_node_wall_time_ms"`
	TotalNodeWallTimeUs         int64                   `json:"total_node_wall_time_us"`
	SummaryPayloadBytesEstimate int64                   `json:"summary_payload_bytes_estimate"`
	OutputBytesEstimate         int64                   `json:"output_bytes_estimate"`
	CreatedAtUnixMs             int64                   `json:"created_at_unix_ms"`
}

type PartialAttentionSummary struct {
	LocalMax      float64   `json:"local_max"`
	ExpSum        float64   `json:"exp_sum"`
	WeightedValue []float64 `json:"weighted_value"`
	TokenCount    int       `json:"token_count"`
	ValueDim      int       `json:"value_dim"`
	DType         string    `json:"dtype"`
}
