package memorybench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

type BenchmarkReceipt struct {
	JobID         string
	ResultHashHex string
	MeteringUnits uint64
	Metadata      map[string]any
}

type BenchmarkReceiptMetadata struct {
	RequestID           string    `json:"request_id"`
	ShardID             string    `json:"shard_id"`
	LocalMax            float64   `json:"local_max"`
	ExpSum              float64   `json:"exp_sum"`
	WeightedValue       []float64 `json:"weighted_value"`
	TokenCount          int       `json:"token_count"`
	ValueDim            int       `json:"value_dim"`
	ComputeTimeMs       int64     `json:"compute_time_ms"`
	OutputBytesEstimate int64     `json:"output_bytes_estimate"`
	ProofStatus         string    `json:"proof_status"`
}

func BuildBenchmarkReceipt(spec BenchmarkSpec, response SyntheticAttentionResponse) (BenchmarkReceipt, error) {
	spec = normalizeBenchmarkSpec(spec)
	if err := ValidateBenchmarkSpec(spec); err != nil {
		return BenchmarkReceipt{}, err
	}

	metadata := BenchmarkReceiptMetadata{
		RequestID:           response.RequestID,
		ShardID:             response.ShardID,
		LocalMax:            response.Summary.LocalMax,
		ExpSum:              response.Summary.ExpSum,
		WeightedValue:       append([]float64(nil), response.Summary.WeightedValue...),
		TokenCount:          response.Summary.TokenCount,
		ValueDim:            response.Summary.ValueDim,
		ComputeTimeMs:       response.ComputeTimeMs,
		OutputBytesEstimate: response.OutputBytesEstimate,
		ProofStatus:         "synthetic_measured",
	}
	if metadata.OutputBytesEstimate <= 0 {
		metadata.OutputBytesEstimate = estimatePartialAttentionSummaryBytes(response.Summary)
	}
	if metadata.RequestID == "" {
		metadata.RequestID = spec.RequestID
	}
	if metadata.ShardID == "" {
		metadata.ShardID = spec.ShardID
	}
	if err := validateBenchmarkReceiptMetadata(spec, metadata); err != nil {
		return BenchmarkReceipt{}, err
	}

	hashHex, err := HashBenchmarkReceiptMetadata(spec.JobID, metadata)
	if err != nil {
		return BenchmarkReceipt{}, err
	}
	return BenchmarkReceipt{
		JobID:         spec.JobID,
		ResultHashHex: hashHex,
		MeteringUnits: 1,
		Metadata: map[string]any{
			BenchmarkTask: metadata.Map(),
		},
	}, nil
}

func BuildBenchmarkRejectionReceipt(jobID string, runErr error) BenchmarkReceipt {
	jobID = strings.TrimSpace(jobID)
	if jobID == "" {
		jobID = "v7-memory-benchmark-rejected"
	}
	reason := "benchmark rejected"
	if runErr != nil {
		reason = strings.TrimSpace(runErr.Error())
	}
	if len(reason) > 256 {
		reason = reason[:256]
	}
	payload := struct {
		Task        string `json:"task"`
		JobID       string `json:"job_id"`
		ProofStatus string `json:"proof_status"`
		Error       string `json:"error"`
	}{
		Task:        BenchmarkTask,
		JobID:       jobID,
		ProofStatus: "rejected",
		Error:       reason,
	}
	encoded, _ := json.Marshal(payload)
	sum := sha256.Sum256(encoded)
	return BenchmarkReceipt{
		JobID:         jobID,
		ResultHashHex: hex.EncodeToString(sum[:]),
		MeteringUnits: 0,
		Metadata: map[string]any{
			BenchmarkTask: map[string]any{
				"proof_status": "rejected",
				"error":        reason,
			},
		},
	}
}

func HashBenchmarkReceiptMetadata(jobID string, metadata BenchmarkReceiptMetadata) (string, error) {
	jobID = strings.TrimSpace(jobID)
	if jobID == "" {
		return "", fmt.Errorf("%w: job_id required for result hash", ErrInvalidBenchmarkSpec)
	}
	envelope := struct {
		Task     string                   `json:"task"`
		JobID    string                   `json:"job_id"`
		Metadata BenchmarkReceiptMetadata `json:"v7_memory_benchmark"`
	}{
		Task:     BenchmarkTask,
		JobID:    jobID,
		Metadata: metadata.clone(),
	}
	encoded, err := json.Marshal(envelope)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:]), nil
}

func (m BenchmarkReceiptMetadata) Map() map[string]any {
	return map[string]any{
		"request_id":            m.RequestID,
		"shard_id":              m.ShardID,
		"local_max":             m.LocalMax,
		"exp_sum":               m.ExpSum,
		"weighted_value":        append([]float64(nil), m.WeightedValue...),
		"token_count":           m.TokenCount,
		"value_dim":             m.ValueDim,
		"compute_time_ms":       m.ComputeTimeMs,
		"output_bytes_estimate": m.OutputBytesEstimate,
		"proof_status":          m.ProofStatus,
	}
}

func (m BenchmarkReceiptMetadata) clone() BenchmarkReceiptMetadata {
	m.WeightedValue = append([]float64(nil), m.WeightedValue...)
	return m
}

func validateBenchmarkReceiptMetadata(spec BenchmarkSpec, metadata BenchmarkReceiptMetadata) error {
	var errs []error
	if strings.TrimSpace(metadata.RequestID) == "" {
		errs = append(errs, fmt.Errorf("%w: receipt request_id required", ErrInvalidBenchmarkSpec))
	}
	if strings.TrimSpace(metadata.ShardID) == "" {
		errs = append(errs, fmt.Errorf("%w: receipt shard_id required", ErrInvalidBenchmarkSpec))
	}
	if metadata.TokenCount != spec.TokenCount {
		errs = append(errs, fmt.Errorf("%w: receipt token_count mismatch", ErrInvalidBenchmarkSpec))
	}
	if metadata.ValueDim != spec.ValueDim {
		errs = append(errs, fmt.Errorf("%w: receipt value_dim mismatch", ErrInvalidBenchmarkSpec))
	}
	if len(metadata.WeightedValue) != spec.ValueDim {
		errs = append(errs, fmt.Errorf("%w: receipt weighted_value length mismatch", ErrInvalidBenchmarkSpec))
	}
	if metadata.ComputeTimeMs < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt compute_time_ms must be non-negative", ErrInvalidBenchmarkSpec))
	}
	if metadata.OutputBytesEstimate <= 0 {
		errs = append(errs, fmt.Errorf("%w: receipt output_bytes_estimate must be positive", ErrInvalidBenchmarkSpec))
	}
	if strings.TrimSpace(metadata.ProofStatus) != "synthetic_measured" {
		errs = append(errs, fmt.Errorf("%w: receipt proof_status must be synthetic_measured", ErrInvalidBenchmarkSpec))
	}
	return errors.Join(errs...)
}
