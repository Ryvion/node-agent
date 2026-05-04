package memorybench

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"time"
)

func ComputePartialAttentionSummary(request SyntheticAttentionRequest) (SyntheticAttentionResponse, error) {
	startedAt := time.Now()
	request = normalizeSyntheticAttentionRequest(request)
	if err := validateSyntheticAttentionRequest(request); err != nil {
		return SyntheticAttentionResponse{}, err
	}

	summary, err := computePartialAttentionSummary(request.Logits, request.Values, request.ValueDim)
	if err != nil {
		return SyntheticAttentionResponse{}, err
	}

	computeTimeMs := time.Since(startedAt).Milliseconds()
	if computeTimeMs < 0 {
		computeTimeMs = 0
	}
	createdAtUnixMs := request.CreatedAtUnixMs
	if createdAtUnixMs <= 0 {
		createdAtUnixMs = startedAt.UnixMilli()
	}

	return SyntheticAttentionResponse{
		RequestID:           request.RequestID,
		JobID:               request.JobID,
		ShardID:             request.ShardID,
		Summary:             summary,
		ComputeTimeMs:       computeTimeMs,
		OutputBytesEstimate: estimatePartialAttentionSummaryBytes(summary),
		CreatedAtUnixMs:     createdAtUnixMs,
	}, nil
}

func NaiveAttentionOutput(logits []float64, values [][]float64) ([]float64, error) {
	valueDim := 0
	if len(values) > 0 {
		valueDim = len(values[0])
	}
	request := SyntheticAttentionRequest{
		RequestID: "naive-attention-output",
		JobID:     "naive-attention-output",
		Logits:    logits,
		Values:    values,
		ValueDim:  valueDim,
	}
	if err := validateSyntheticAttentionRequest(request); err != nil {
		return nil, err
	}

	localMax := logits[0]
	for _, logit := range logits[1:] {
		if logit > localMax {
			localMax = logit
		}
	}

	denominator := 0.0
	for _, logit := range logits {
		denominator += math.Exp(logit - localMax)
	}
	if denominator <= 0 || !finiteFloat64(denominator) {
		return nil, fmt.Errorf("%w: exp_sum must be positive and finite", ErrInvalidSyntheticAttentionRequest)
	}

	output := make([]float64, valueDim)
	for i, logit := range logits {
		weight := math.Exp(logit-localMax) / denominator
		for dim := 0; dim < valueDim; dim++ {
			output[dim] += weight * values[i][dim]
			if !finiteFloat64(output[dim]) {
				return nil, fmt.Errorf("%w: attention output overflow at value_dim %d", ErrInvalidSyntheticAttentionRequest, dim)
			}
		}
	}
	return output, nil
}

func normalizeSyntheticAttentionRequest(request SyntheticAttentionRequest) SyntheticAttentionRequest {
	request.RequestID = strings.TrimSpace(request.RequestID)
	request.JobID = strings.TrimSpace(request.JobID)
	request.ShardID = strings.TrimSpace(request.ShardID)
	return request
}

func validateSyntheticAttentionRequest(request SyntheticAttentionRequest) error {
	var errs []error
	if request.RequestID == "" {
		errs = append(errs, fmt.Errorf("%w: request_id required", ErrInvalidSyntheticAttentionRequest))
	}
	if request.JobID == "" {
		errs = append(errs, fmt.Errorf("%w: job_id required", ErrInvalidSyntheticAttentionRequest))
	}
	if len(request.Logits) == 0 {
		errs = append(errs, fmt.Errorf("%w: token_count must be greater than zero", ErrInvalidSyntheticAttentionRequest))
	}
	if request.ValueDim <= 0 {
		errs = append(errs, fmt.Errorf("%w: value_dim must be greater than zero", ErrInvalidSyntheticAttentionRequest))
	}
	if len(request.Logits) != len(request.Values) {
		errs = append(errs, fmt.Errorf("%w: logits and values length mismatch", ErrInvalidSyntheticAttentionRequest))
	}
	for i, logit := range request.Logits {
		if !finiteFloat64(logit) {
			errs = append(errs, fmt.Errorf("%w: logit %d must be finite", ErrInvalidSyntheticAttentionRequest, i))
		}
	}
	for i, value := range request.Values {
		if len(value) != request.ValueDim {
			errs = append(errs, fmt.Errorf("%w: value %d length must equal value_dim", ErrInvalidSyntheticAttentionRequest, i))
			continue
		}
		for dim, component := range value {
			if !finiteFloat64(component) {
				errs = append(errs, fmt.Errorf("%w: value %d component %d must be finite", ErrInvalidSyntheticAttentionRequest, i, dim))
			}
		}
	}
	return errors.Join(errs...)
}

func computePartialAttentionSummary(logits []float64, values [][]float64, valueDim int) (PartialAttentionSummary, error) {
	localMax := logits[0]
	for _, logit := range logits[1:] {
		if logit > localMax {
			localMax = logit
		}
	}

	expSum := 0.0
	weightedValue := make([]float64, valueDim)
	for i, logit := range logits {
		weight := math.Exp(logit - localMax)
		if !finiteFloat64(weight) {
			return PartialAttentionSummary{}, fmt.Errorf("%w: exponent overflow at token %d", ErrInvalidSyntheticAttentionRequest, i)
		}
		expSum += weight
		for dim := 0; dim < valueDim; dim++ {
			weightedValue[dim] += weight * values[i][dim]
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
		TokenCount:    len(logits),
		ValueDim:      valueDim,
		DType:         SyntheticAttentionDTypeFloat64,
	}, nil
}

func finiteFloat64(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
