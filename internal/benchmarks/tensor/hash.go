package tensorplane

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

func HashTensorPage(page TensorPage) (string, error) {
	page = normalizeTensorPage(page)
	if err := ValidateTensorPage(page); err != nil {
		return "", err
	}
	payload := struct {
		PageID    TensorPageID `json:"page_id"`
		DType     TensorDType  `json:"dtype"`
		Shape     TensorShape  `json:"shape"`
		KeyData   []byte       `json:"key_data"`
		ValueData []byte       `json:"value_data"`
	}{
		PageID:    page.PageID,
		DType:     page.DType,
		Shape:     page.Shape,
		KeyData:   append([]byte(nil), page.KeyData...),
		ValueData: append([]byte(nil), page.ValueData...),
	}
	return sha256HexJSON(payload)
}

func HashTensorPartialAttentionSummary(summary TensorPartialAttentionSummary) (string, error) {
	summary.PageID = normalizeTensorPageID(summary.PageID)
	summary.DType = NormalizeTensorDType(summary.DType)
	if err := validateTensorPartialAttentionSummaryForHash(summary); err != nil {
		return "", err
	}
	payload := tensorPartialAttentionHashPayload{
		RequestID:            summary.RequestID,
		JobID:                summary.JobID,
		PageID:               summary.PageID,
		LocalMax:             summary.LocalMax,
		ExpSum:               summary.ExpSum,
		WeightedValue:        append([]float64(nil), summary.WeightedValue...),
		TokenCount:           summary.TokenCount,
		ValueDim:             summary.ValueDim,
		DType:                summary.DType,
		PageHash:             summary.PageHash,
		PayloadBytesEstimate: summary.PayloadBytesEstimate,
	}
	return sha256HexJSON(payload)
}

func EstimateTensorPartialAttentionPayloadBytes(valueDim int) int64 {
	if valueDim <= 0 {
		return 0
	}
	const fixedSummaryBytes = 256
	return fixedSummaryBytes + int64(valueDim)*8
}

type tensorPartialAttentionHashPayload struct {
	RequestID            string       `json:"request_id"`
	JobID                string       `json:"job_id"`
	PageID               TensorPageID `json:"page_id"`
	LocalMax             float64      `json:"local_max"`
	ExpSum               float64      `json:"exp_sum"`
	WeightedValue        []float64    `json:"weighted_value"`
	TokenCount           int          `json:"token_count"`
	ValueDim             int          `json:"value_dim"`
	DType                TensorDType  `json:"dtype"`
	PageHash             string       `json:"page_hash"`
	PayloadBytesEstimate int64        `json:"payload_bytes_estimate"`
}

func validateTensorPartialAttentionSummaryForHash(summary TensorPartialAttentionSummary) error {
	if summary.RequestID == "" {
		return fmt.Errorf("%w: request_id required", ErrInvalidPartialAttentionHash)
	}
	if summary.JobID == "" {
		return fmt.Errorf("%w: job_id required", ErrInvalidPartialAttentionHash)
	}
	if err := ValidateTensorPageID(summary.PageID); err != nil {
		return err
	}
	if !finiteFloat64(summary.LocalMax) {
		return fmt.Errorf("%w: local_max must be finite", ErrInvalidPartialAttentionHash)
	}
	if summary.ExpSum <= 0 || !finiteFloat64(summary.ExpSum) {
		return fmt.Errorf("%w: exp_sum must be positive and finite", ErrInvalidPartialAttentionHash)
	}
	if summary.TokenCount <= 0 {
		return fmt.Errorf("%w: token_count must be greater than zero", ErrInvalidPartialAttentionHash)
	}
	if summary.ValueDim <= 0 {
		return fmt.Errorf("%w: value_dim must be greater than zero", ErrInvalidPartialAttentionHash)
	}
	if len(summary.WeightedValue) != summary.ValueDim {
		return fmt.Errorf("%w: weighted_value length must equal value_dim", ErrInvalidPartialAttentionHash)
	}
	for i, value := range summary.WeightedValue {
		if !finiteFloat64(value) {
			return fmt.Errorf("%w: weighted_value[%d] must be finite", ErrInvalidPartialAttentionHash, i)
		}
	}
	if summary.DType == "" {
		return fmt.Errorf("%w: dtype required", ErrInvalidPartialAttentionHash)
	}
	if err := ValidateTensorDType(summary.DType); err != nil {
		return err
	}
	if summary.PageHash == "" {
		return fmt.Errorf("%w: page_hash required", ErrInvalidPartialAttentionHash)
	}
	if summary.PayloadBytesEstimate <= 0 {
		return fmt.Errorf("%w: payload_bytes_estimate must be greater than zero", ErrInvalidPartialAttentionHash)
	}
	return nil
}

func sha256HexJSON(value any) (string, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:]), nil
}
