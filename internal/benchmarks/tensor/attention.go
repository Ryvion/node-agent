package tensorplane

import (
	"fmt"
	"math"
	"time"
)

func ComputeTensorPartialAttentionSummary(query AttentionQuery, page TensorPage) (TensorPartialAttentionSummary, error) {
	startedAt := time.Now()
	page = normalizeTensorPage(page)
	query = normalizeAttentionQuery(query)

	if err := ValidateTensorPage(page); err != nil {
		return TensorPartialAttentionSummary{}, err
	}
	if err := ValidateAttentionQuery(query, page); err != nil {
		return TensorPartialAttentionSummary{}, err
	}
	localHead, err := resolveQueryLocalHead(query, page)
	if err != nil {
		return TensorPartialAttentionSummary{}, err
	}
	pageHash, err := HashTensorPage(page)
	if err != nil {
		return TensorPartialAttentionSummary{}, err
	}

	logits := make([]float64, page.Shape.Tokens)
	localMax := math.Inf(-1)
	for token := 0; token < page.Shape.Tokens; token++ {
		logit, err := tensorAttentionLogit(query, page, localHead, token)
		if err != nil {
			return TensorPartialAttentionSummary{}, err
		}
		logits[token] = logit
		if logit > localMax {
			localMax = logit
		}
	}
	if !finiteFloat64(localMax) {
		return TensorPartialAttentionSummary{}, fmt.Errorf("%w: local_max must be finite", ErrInvalidAttentionQuery)
	}

	expSum := 0.0
	weightedValue := make([]float64, page.Shape.ValueDim)
	for token, logit := range logits {
		weight := math.Exp(logit - localMax)
		if !finiteFloat64(weight) {
			return TensorPartialAttentionSummary{}, fmt.Errorf("%w: exponent overflow at token %d", ErrInvalidAttentionQuery, token)
		}
		expSum += weight
		for dim := 0; dim < page.Shape.ValueDim; dim++ {
			value, err := decodeTensorFloat(page.DType, page.ValueData, valueElementIndex(page.Shape, localHead, token, dim))
			if err != nil {
				return TensorPartialAttentionSummary{}, err
			}
			weightedValue[dim] += weight * float64(value)
			if !finiteFloat64(weightedValue[dim]) {
				return TensorPartialAttentionSummary{}, fmt.Errorf("%w: weighted value overflow at value_dim %d", ErrInvalidAttentionQuery, dim)
			}
		}
	}
	if expSum <= 0 || !finiteFloat64(expSum) {
		return TensorPartialAttentionSummary{}, fmt.Errorf("%w: exp_sum must be positive and finite", ErrInvalidAttentionQuery)
	}

	summary := TensorPartialAttentionSummary{
		RequestID:            query.RequestID,
		JobID:                query.JobID,
		PageID:               page.PageID,
		LocalMax:             localMax,
		ExpSum:               expSum,
		WeightedValue:        weightedValue,
		TokenCount:           page.Shape.Tokens,
		ValueDim:             page.Shape.ValueDim,
		DType:                page.DType,
		PageHash:             pageHash,
		ComputeTimeUs:        nonNegativeDurationMicroseconds(time.Since(startedAt)),
		PayloadBytesEstimate: EstimateTensorPartialAttentionPayloadBytes(page.Shape.ValueDim),
	}
	summaryHash, err := HashTensorPartialAttentionSummary(summary)
	if err != nil {
		return TensorPartialAttentionSummary{}, err
	}
	summary.SummaryHash = summaryHash
	return summary, nil
}

func tensorAttentionLogit(query AttentionQuery, page TensorPage, localHead int, token int) (float64, error) {
	dot := 0.0
	for dim, queryValue := range query.QueryVector {
		keyValue, err := decodeTensorFloat(page.DType, page.KeyData, keyElementIndex(page.Shape, localHead, token, dim))
		if err != nil {
			return 0, err
		}
		dot += float64(queryValue) * float64(keyValue)
		if !finiteFloat64(dot) {
			return 0, fmt.Errorf("%w: dot product overflow at token %d head_dim %d", ErrInvalidAttentionQuery, token, dim)
		}
	}
	logit := dot * query.Scale
	if !finiteFloat64(logit) {
		return 0, fmt.Errorf("%w: logit at token %d must be finite", ErrInvalidAttentionQuery, token)
	}
	return logit, nil
}

func keyElementIndex(shape TensorShape, localHead int, token int, dim int) int {
	return (localHead*shape.Tokens*shape.HeadDim + token*shape.HeadDim + dim)
}

func valueElementIndex(shape TensorShape, localHead int, token int, dim int) int {
	return (localHead*shape.Tokens*shape.ValueDim + token*shape.ValueDim + dim)
}

func nonNegativeDurationMicroseconds(duration time.Duration) int64 {
	if duration < 0 {
		return 0
	}
	return duration.Microseconds()
}
