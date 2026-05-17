package tensorplane

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

const ProofStatusTensorPlaneMeasured = "tensorplane_measured"

type BenchmarkReceipt struct {
	JobID         string
	ResultHashHex string
	MeteringUnits uint64
	Metadata      map[string]any
}

type BenchmarkReceiptMetadata struct {
	RequestID             string      `json:"request_id"`
	JobID                 string      `json:"job_id"`
	ModelID               string      `json:"model_id"`
	LayerIndex            int         `json:"layer_index"`
	DType                 TensorDType `json:"dtype"`
	Tokens                int         `json:"tokens"`
	HeadDim               int         `json:"head_dim"`
	ValueDim              int         `json:"value_dim"`
	PageHash              string      `json:"page_hash"`
	QueryHash             string      `json:"query_hash"`
	SummaryHash           string      `json:"summary_hash"`
	LocalMax              float64     `json:"local_max"`
	ExpSum                float64     `json:"exp_sum"`
	WeightedValue         []float64   `json:"weighted_value"`
	WeightedValueLength   int         `json:"weighted_value_length"`
	ComputeTimeUs         int64       `json:"compute_time_us"`
	PayloadBytesEstimate  int64       `json:"payload_bytes_estimate"`
	MaxAbsDiffVsReference float64     `json:"max_abs_diff_vs_reference"`
	CorrectnessStatus     string      `json:"correctness_status"`
	ProofStatus           string      `json:"proof_status"`
}

func BuildBenchmarkReceipt(result BenchmarkExecutionResult) (BenchmarkReceipt, error) {
	result.Spec = normalizeBenchmarkSpec(result.Spec)
	if err := ValidateBenchmarkSpec(result.Spec); err != nil {
		return BenchmarkReceipt{}, err
	}
	if strings.TrimSpace(result.QueryHash) == "" {
		queryHash, err := HashBenchmarkQuery(result.Query)
		if err != nil {
			return BenchmarkReceipt{}, err
		}
		result.QueryHash = queryHash
	}

	metadata := BenchmarkReceiptMetadata{
		RequestID:             result.Spec.RequestID,
		JobID:                 result.Spec.JobID,
		ModelID:               result.Spec.ModelID,
		LayerIndex:            result.Spec.LayerIndex,
		DType:                 result.Spec.DType,
		Tokens:                result.Spec.Tokens,
		HeadDim:               result.Spec.HeadDim,
		ValueDim:              result.Spec.ValueDim,
		PageHash:              sha256ID(result.Summary.PageHash),
		QueryHash:             sha256ID(result.QueryHash),
		SummaryHash:           sha256ID(result.Summary.SummaryHash),
		LocalMax:              result.Summary.LocalMax,
		ExpSum:                result.Summary.ExpSum,
		WeightedValue:         append([]float64(nil), result.Summary.WeightedValue...),
		WeightedValueLength:   len(result.Summary.WeightedValue),
		ComputeTimeUs:         result.Summary.ComputeTimeUs,
		PayloadBytesEstimate:  result.Summary.PayloadBytesEstimate,
		MaxAbsDiffVsReference: result.MaxAbsDiffVsReference,
		CorrectnessStatus:     strings.TrimSpace(result.CorrectnessStatus),
		ProofStatus:           ProofStatusTensorPlaneMeasured,
	}
	if metadata.CorrectnessStatus == "" {
		metadata.CorrectnessStatus = CorrectnessStatusNotChecked
	}
	if err := validateBenchmarkReceiptMetadata(result.Spec, metadata); err != nil {
		return BenchmarkReceipt{}, err
	}
	hashHex, err := HashBenchmarkReceiptMetadata(result.Spec.JobID, metadata)
	if err != nil {
		return BenchmarkReceipt{}, err
	}
	return BenchmarkReceipt{
		JobID:         result.Spec.JobID,
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
		jobID = "v7-tensorplane-benchmark-rejected"
	}
	reason := "tensorplane benchmark rejected"
	if runErr != nil {
		reason = cleanTensorPlaneLocalStatusError(runErr)
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

func HashBenchmarkQuery(query AttentionQuery) (string, error) {
	query = normalizeAttentionQuery(query)
	var errs []error
	if query.RequestID == "" {
		errs = append(errs, fmt.Errorf("%w: query request_id required", ErrInvalidBenchmarkSpec))
	}
	if query.JobID == "" {
		errs = append(errs, fmt.Errorf("%w: query job_id required", ErrInvalidBenchmarkSpec))
	}
	if query.ModelID == "" {
		errs = append(errs, fmt.Errorf("%w: query model_id required", ErrInvalidBenchmarkSpec))
	}
	if query.LayerIndex < 0 {
		errs = append(errs, fmt.Errorf("%w: query layer_index must be non-negative", ErrInvalidBenchmarkSpec))
	}
	if err := ValidateTensorDType(query.DType); err != nil {
		errs = append(errs, err)
	}
	if len(query.QueryVector) == 0 {
		errs = append(errs, fmt.Errorf("%w: query_vector required for query hash", ErrInvalidBenchmarkSpec))
	}
	for i, value := range query.QueryVector {
		if !finiteFloat64(float64(value)) {
			errs = append(errs, fmt.Errorf("%w: query_vector[%d] must be finite", ErrInvalidBenchmarkSpec, i))
		}
	}
	if !finiteFloat64(query.Scale) {
		errs = append(errs, fmt.Errorf("%w: query scale must be finite", ErrInvalidBenchmarkSpec))
	}
	if query.CreatedAtUnixMs <= 0 {
		errs = append(errs, fmt.Errorf("%w: query created_at_unix_ms must be positive", ErrInvalidBenchmarkSpec))
	}
	if err := errors.Join(errs...); err != nil {
		return "", err
	}
	payload := struct {
		RequestID       string      `json:"request_id"`
		JobID           string      `json:"job_id"`
		ModelID         string      `json:"model_id"`
		LayerIndex      int         `json:"layer_index"`
		HeadIndex       int         `json:"head_index"`
		HeadStart       int         `json:"head_start"`
		HeadCount       int         `json:"head_count"`
		QueryVector     []float32   `json:"query_vector"`
		Scale           float64     `json:"scale"`
		DType           TensorDType `json:"dtype"`
		CreatedAtUnixMs int64       `json:"created_at_unix_ms"`
	}{
		RequestID:       query.RequestID,
		JobID:           query.JobID,
		ModelID:         query.ModelID,
		LayerIndex:      query.LayerIndex,
		HeadIndex:       query.HeadIndex,
		HeadStart:       query.HeadStart,
		HeadCount:       query.HeadCount,
		QueryVector:     append([]float32(nil), query.QueryVector...),
		Scale:           query.Scale,
		DType:           query.DType,
		CreatedAtUnixMs: query.CreatedAtUnixMs,
	}
	hash, err := sha256HexJSON(payload)
	if err != nil {
		return "", err
	}
	return sha256ID(hash), nil
}

func HashBenchmarkReceiptMetadata(jobID string, metadata BenchmarkReceiptMetadata) (string, error) {
	jobID = strings.TrimSpace(jobID)
	if jobID == "" {
		return "", fmt.Errorf("%w: job_id required for result hash", ErrInvalidBenchmarkSpec)
	}
	metadata = metadata.clone()
	if err := validateBenchmarkReceiptMetadataForHash(metadata); err != nil {
		return "", err
	}
	envelope := struct {
		Task     string                       `json:"task"`
		JobID    string                       `json:"job_id"`
		Metadata benchmarkReceiptHashMetadata `json:"v7_tensorplane_benchmark"`
	}{
		Task:     BenchmarkTask,
		JobID:    jobID,
		Metadata: benchmarkReceiptHashMetadataFromMetadata(metadata),
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
		"request_id":                m.RequestID,
		"job_id":                    m.JobID,
		"model_id":                  m.ModelID,
		"layer_index":               m.LayerIndex,
		"dtype":                     string(m.DType),
		"tokens":                    m.Tokens,
		"head_dim":                  m.HeadDim,
		"value_dim":                 m.ValueDim,
		"page_hash":                 m.PageHash,
		"query_hash":                m.QueryHash,
		"summary_hash":              m.SummaryHash,
		"local_max":                 m.LocalMax,
		"exp_sum":                   m.ExpSum,
		"weighted_value":            append([]float64(nil), m.WeightedValue...),
		"weighted_value_length":     m.WeightedValueLength,
		"compute_time_us":           m.ComputeTimeUs,
		"payload_bytes_estimate":    m.PayloadBytesEstimate,
		"max_abs_diff_vs_reference": m.MaxAbsDiffVsReference,
		"correctness_status":        m.CorrectnessStatus,
		"proof_status":              m.ProofStatus,
	}
}

func (m BenchmarkReceiptMetadata) clone() BenchmarkReceiptMetadata {
	m.DType = NormalizeTensorDType(m.DType)
	m.RequestID = strings.TrimSpace(m.RequestID)
	m.JobID = strings.TrimSpace(m.JobID)
	m.ModelID = strings.TrimSpace(m.ModelID)
	m.PageHash = sha256ID(m.PageHash)
	m.QueryHash = sha256ID(m.QueryHash)
	m.SummaryHash = sha256ID(m.SummaryHash)
	m.CorrectnessStatus = strings.TrimSpace(m.CorrectnessStatus)
	m.ProofStatus = strings.TrimSpace(m.ProofStatus)
	m.WeightedValue = append([]float64(nil), m.WeightedValue...)
	if m.WeightedValueLength == 0 {
		m.WeightedValueLength = len(m.WeightedValue)
	}
	return m
}

type benchmarkReceiptHashMetadata struct {
	RequestID             string      `json:"request_id"`
	JobID                 string      `json:"job_id"`
	ModelID               string      `json:"model_id"`
	LayerIndex            int         `json:"layer_index"`
	DType                 TensorDType `json:"dtype"`
	Tokens                int         `json:"tokens"`
	HeadDim               int         `json:"head_dim"`
	ValueDim              int         `json:"value_dim"`
	PageHash              string      `json:"page_hash"`
	QueryHash             string      `json:"query_hash"`
	SummaryHash           string      `json:"summary_hash"`
	LocalMax              float64     `json:"local_max"`
	ExpSum                float64     `json:"exp_sum"`
	WeightedValue         []float64   `json:"weighted_value"`
	WeightedValueLength   int         `json:"weighted_value_length"`
	PayloadBytesEstimate  int64       `json:"payload_bytes_estimate"`
	MaxAbsDiffVsReference float64     `json:"max_abs_diff_vs_reference"`
	CorrectnessStatus     string      `json:"correctness_status"`
	ProofStatus           string      `json:"proof_status"`
}

func benchmarkReceiptHashMetadataFromMetadata(metadata BenchmarkReceiptMetadata) benchmarkReceiptHashMetadata {
	metadata = metadata.clone()
	return benchmarkReceiptHashMetadata{
		RequestID:             metadata.RequestID,
		JobID:                 metadata.JobID,
		ModelID:               metadata.ModelID,
		LayerIndex:            metadata.LayerIndex,
		DType:                 metadata.DType,
		Tokens:                metadata.Tokens,
		HeadDim:               metadata.HeadDim,
		ValueDim:              metadata.ValueDim,
		PageHash:              metadata.PageHash,
		QueryHash:             metadata.QueryHash,
		SummaryHash:           metadata.SummaryHash,
		LocalMax:              metadata.LocalMax,
		ExpSum:                metadata.ExpSum,
		WeightedValue:         append([]float64(nil), metadata.WeightedValue...),
		WeightedValueLength:   metadata.WeightedValueLength,
		PayloadBytesEstimate:  metadata.PayloadBytesEstimate,
		MaxAbsDiffVsReference: metadata.MaxAbsDiffVsReference,
		CorrectnessStatus:     metadata.CorrectnessStatus,
		ProofStatus:           metadata.ProofStatus,
	}
}

func validateBenchmarkReceiptMetadata(spec BenchmarkSpec, metadata BenchmarkReceiptMetadata) error {
	spec = normalizeBenchmarkSpec(spec)
	metadata = metadata.clone()
	var errs []error
	if err := ValidateBenchmarkSpec(spec); err != nil {
		errs = append(errs, err)
	}
	if metadata.RequestID != spec.RequestID {
		errs = append(errs, fmt.Errorf("%w: receipt request_id mismatch", ErrInvalidBenchmarkSpec))
	}
	if metadata.JobID != spec.JobID {
		errs = append(errs, fmt.Errorf("%w: receipt job_id mismatch", ErrInvalidBenchmarkSpec))
	}
	if metadata.ModelID != spec.ModelID {
		errs = append(errs, fmt.Errorf("%w: receipt model_id mismatch", ErrInvalidBenchmarkSpec))
	}
	if metadata.LayerIndex != spec.LayerIndex {
		errs = append(errs, fmt.Errorf("%w: receipt layer_index mismatch", ErrInvalidBenchmarkSpec))
	}
	if metadata.DType != spec.DType {
		errs = append(errs, fmt.Errorf("%w: receipt dtype mismatch", ErrInvalidBenchmarkSpec))
	}
	if metadata.Tokens != spec.Tokens {
		errs = append(errs, fmt.Errorf("%w: receipt tokens mismatch", ErrInvalidBenchmarkSpec))
	}
	if metadata.HeadDim != spec.HeadDim {
		errs = append(errs, fmt.Errorf("%w: receipt head_dim mismatch", ErrInvalidBenchmarkSpec))
	}
	if metadata.ValueDim != spec.ValueDim {
		errs = append(errs, fmt.Errorf("%w: receipt value_dim mismatch", ErrInvalidBenchmarkSpec))
	}
	if err := validateBenchmarkReceiptMetadataForHash(metadata); err != nil {
		errs = append(errs, err)
	}
	if metadata.ComputeTimeUs < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt compute_time_us must be non-negative", ErrInvalidBenchmarkSpec))
	}
	return errors.Join(errs...)
}

func validateBenchmarkReceiptMetadataForHash(metadata BenchmarkReceiptMetadata) error {
	metadata = metadata.clone()
	var errs []error
	if metadata.RequestID == "" {
		errs = append(errs, fmt.Errorf("%w: receipt request_id required", ErrInvalidBenchmarkSpec))
	}
	if metadata.JobID == "" {
		errs = append(errs, fmt.Errorf("%w: receipt job_id required", ErrInvalidBenchmarkSpec))
	}
	if metadata.ModelID == "" {
		errs = append(errs, fmt.Errorf("%w: receipt model_id required", ErrInvalidBenchmarkSpec))
	}
	if metadata.LayerIndex < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt layer_index must be non-negative", ErrInvalidBenchmarkSpec))
	}
	if err := ValidateTensorDType(metadata.DType); err != nil {
		errs = append(errs, err)
	}
	if metadata.Tokens <= 0 {
		errs = append(errs, fmt.Errorf("%w: receipt tokens must be positive", ErrInvalidBenchmarkSpec))
	}
	if metadata.HeadDim <= 0 {
		errs = append(errs, fmt.Errorf("%w: receipt head_dim must be positive", ErrInvalidBenchmarkSpec))
	}
	if metadata.ValueDim <= 0 {
		errs = append(errs, fmt.Errorf("%w: receipt value_dim must be positive", ErrInvalidBenchmarkSpec))
	}
	if err := validateSHA256ID(metadata.PageHash, "receipt page_hash"); err != nil {
		errs = append(errs, err)
	}
	if err := validateSHA256ID(metadata.QueryHash, "receipt query_hash"); err != nil {
		errs = append(errs, err)
	}
	if err := validateSHA256ID(metadata.SummaryHash, "receipt summary_hash"); err != nil {
		errs = append(errs, err)
	}
	if !finiteFloat64(metadata.LocalMax) {
		errs = append(errs, fmt.Errorf("%w: receipt local_max must be finite", ErrInvalidBenchmarkSpec))
	}
	if metadata.ExpSum <= 0 || !finiteFloat64(metadata.ExpSum) {
		errs = append(errs, fmt.Errorf("%w: receipt exp_sum must be positive and finite", ErrInvalidBenchmarkSpec))
	}
	if metadata.WeightedValueLength != len(metadata.WeightedValue) {
		errs = append(errs, fmt.Errorf("%w: receipt weighted_value_length mismatch", ErrInvalidBenchmarkSpec))
	}
	if metadata.WeightedValueLength != metadata.ValueDim {
		errs = append(errs, fmt.Errorf("%w: receipt weighted_value_length must equal value_dim", ErrInvalidBenchmarkSpec))
	}
	for i, value := range metadata.WeightedValue {
		if !finiteFloat64(value) {
			errs = append(errs, fmt.Errorf("%w: receipt weighted_value[%d] must be finite", ErrInvalidBenchmarkSpec, i))
		}
	}
	if metadata.PayloadBytesEstimate <= 0 {
		errs = append(errs, fmt.Errorf("%w: receipt payload_bytes_estimate must be positive", ErrInvalidBenchmarkSpec))
	}
	if metadata.MaxAbsDiffVsReference < 0 || !finiteFloat64(metadata.MaxAbsDiffVsReference) {
		errs = append(errs, fmt.Errorf("%w: receipt max_abs_diff_vs_reference must be finite and non-negative", ErrInvalidBenchmarkSpec))
	}
	switch metadata.CorrectnessStatus {
	case CorrectnessStatusMatched, CorrectnessStatusMismatch, CorrectnessStatusNotChecked:
	default:
		errs = append(errs, fmt.Errorf("%w: receipt correctness_status unknown %q", ErrInvalidBenchmarkSpec, metadata.CorrectnessStatus))
	}
	if metadata.ProofStatus != ProofStatusTensorPlaneMeasured {
		errs = append(errs, fmt.Errorf("%w: receipt proof_status must be %s", ErrInvalidBenchmarkSpec, ProofStatusTensorPlaneMeasured))
	}
	return errors.Join(errs...)
}

func sha256ID(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	if strings.HasPrefix(value, "sha256:") {
		return value
	}
	return "sha256:" + value
}

func validateSHA256ID(value, field string) error {
	value = strings.TrimSpace(value)
	if !strings.HasPrefix(value, "sha256:") {
		return fmt.Errorf("%w: %s must use sha256:<hex>", ErrInvalidBenchmarkSpec, field)
	}
	hexPart := strings.TrimPrefix(value, "sha256:")
	if len(hexPart) != 64 {
		return fmt.Errorf("%w: %s must contain 64 hex chars", ErrInvalidBenchmarkSpec, field)
	}
	if _, err := hex.DecodeString(hexPart); err != nil {
		return fmt.Errorf("%w: %s must contain valid hex", ErrInvalidBenchmarkSpec, field)
	}
	return nil
}
