package tensorplane

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"
)

const (
	CorrectnessStatusMatched    = "matched"
	CorrectnessStatusMismatch   = "mismatch"
	CorrectnessStatusNotChecked = "not_checked"

	maxLocalStatusIDLen    = 256
	maxLocalStatusErrorLen = 512
)

type ExecuteOptions struct {
	Getenv func(string) string
}

type BenchmarkExecutionResult struct {
	Spec                  BenchmarkSpec
	Page                  TensorPage
	Query                 AttentionQuery
	QueryHash             string
	Summary               TensorPartialAttentionSummary
	MaxAbsDiffVsReference float64
	CorrectnessStatus     string
}

type LocalStatusCounters struct {
	Seen             uint64 `json:"seen"`
	Executed         uint64 `json:"executed"`
	ReceiptSubmitted uint64 `json:"receipt_submitted"`
	ReceiptFailed    uint64 `json:"receipt_failed"`
}

type LocalStatusSnapshot struct {
	LastJobID  string              `json:"last_job_id,omitempty"`
	LastError  string              `json:"last_error,omitempty"`
	Counters   LocalStatusCounters `json:"counters"`
	LastSeenAt *time.Time          `json:"last_seen_at,omitempty"`
}

type LocalStatus struct {
	mu       sync.RWMutex
	snapshot LocalStatusSnapshot
}

func NewLocalStatus() *LocalStatus {
	return &LocalStatus{}
}

func ExecuteBenchmarkAssignment(ctx context.Context, specJSON string, opts ExecuteOptions) (BenchmarkReceipt, bool, error) {
	if !IsBenchmarkSpecJSON(specJSON) {
		return BenchmarkReceipt{}, false, nil
	}
	if !BenchmarkEnabledFromEnv(opts.Getenv) {
		return BenchmarkReceipt{}, false, nil
	}
	spec, err := DecodeBenchmarkSpec(specJSON)
	if err != nil {
		return BenchmarkReceipt{}, true, err
	}
	receipt, err := ExecuteBenchmarkSpec(ctx, spec, opts)
	return receipt, true, err
}

func ExecuteBenchmarkSpec(ctx context.Context, spec BenchmarkSpec, _ ExecuteOptions) (BenchmarkReceipt, error) {
	result, err := RunBenchmarkSpec(ctx, spec)
	if err != nil {
		return BenchmarkReceipt{}, err
	}
	return BuildBenchmarkReceipt(result)
}

func RunBenchmarkSpec(ctx context.Context, spec BenchmarkSpec) (BenchmarkExecutionResult, error) {
	spec = normalizeBenchmarkSpec(spec)
	if err := ValidateBenchmarkSpec(spec); err != nil {
		return BenchmarkExecutionResult{}, err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	runCtx, cancel := context.WithTimeout(ctx, time.Duration(spec.TimeoutMs)*time.Millisecond)
	defer cancel()
	if err := runCtx.Err(); err != nil {
		return BenchmarkExecutionResult{}, fmt.Errorf("%w: benchmark context unavailable: %v", ErrInvalidBenchmarkSpec, err)
	}

	page, query, err := BuildBenchmarkFixture(spec)
	if err != nil {
		return BenchmarkExecutionResult{}, err
	}
	if err := runCtx.Err(); err != nil {
		return BenchmarkExecutionResult{}, fmt.Errorf("%w: fixture build exceeded timeout: %v", ErrInvalidBenchmarkSpec, err)
	}

	summary, err := ComputeTensorPartialAttentionSummary(query, page)
	if err != nil {
		return BenchmarkExecutionResult{}, err
	}
	if err := runCtx.Err(); err != nil {
		return BenchmarkExecutionResult{}, fmt.Errorf("%w: attention benchmark exceeded timeout: %v", ErrInvalidBenchmarkSpec, err)
	}

	queryHash, err := HashBenchmarkQuery(query)
	if err != nil {
		return BenchmarkExecutionResult{}, err
	}
	reference, err := computeNaiveTensorPartialAttentionReference(query, page)
	if err != nil {
		return BenchmarkExecutionResult{}, err
	}
	maxDiff := maxAbsDiffVsReference(summary, reference)
	correctness := CorrectnessStatusMatched
	if maxDiff > TensorPlaneProbeTolerance(summary.DType) {
		correctness = CorrectnessStatusMismatch
	}

	return BenchmarkExecutionResult{
		Spec:                  spec,
		Page:                  page,
		Query:                 query,
		QueryHash:             queryHash,
		Summary:               summary,
		MaxAbsDiffVsReference: maxDiff,
		CorrectnessStatus:     correctness,
	}, nil
}

func BuildBenchmarkFixture(spec BenchmarkSpec) (TensorPage, AttentionQuery, error) {
	spec = normalizeBenchmarkSpec(spec)
	if err := ValidateBenchmarkSpec(spec); err != nil {
		return TensorPage{}, AttentionQuery{}, err
	}

	keyElements, ok := checkedMultiply(spec.Tokens, spec.HeadDim)
	if !ok {
		return TensorPage{}, AttentionQuery{}, fmt.Errorf("%w: key tensor element count overflow", ErrInvalidBenchmarkSpec)
	}
	valueElements, ok := checkedMultiply(spec.Tokens, spec.ValueDim)
	if !ok {
		return TensorPage{}, AttentionQuery{}, fmt.Errorf("%w: value tensor element count overflow", ErrInvalidBenchmarkSpec)
	}
	keyValues := make([]float32, keyElements)
	valueValues := make([]float32, valueElements)
	queryVector := make([]float32, spec.HeadDim)
	for i := range keyValues {
		keyValues[i] = deterministicTensorPlaneFloat(spec.Seed, 11, i, 0.75)
	}
	for i := range valueValues {
		valueValues[i] = deterministicTensorPlaneFloat(spec.Seed, 13, i, 1.25)
	}
	for i := range queryVector {
		queryVector[i] = deterministicTensorPlaneFloat(spec.Seed, 17, i, 0.5)
	}

	keyData, err := encodeBenchmarkFloats(spec.DType, keyValues)
	if err != nil {
		return TensorPage{}, AttentionQuery{}, err
	}
	valueData, err := encodeBenchmarkFloats(spec.DType, valueValues)
	if err != nil {
		return TensorPage{}, AttentionQuery{}, err
	}
	pageID, err := benchmarkPageID(spec)
	if err != nil {
		return TensorPage{}, AttentionQuery{}, err
	}
	page := TensorPage{
		PageID: TensorPageID{
			ModelID:       spec.ModelID,
			LayerIndex:    spec.LayerIndex,
			HeadStart:     0,
			HeadCount:     1,
			TokenStart:    0,
			TokenCount:    spec.Tokens,
			PageID:        pageID,
			DType:         spec.DType,
			LayoutVersion: TensorLayoutSimpleContiguousV1,
		},
		DType: spec.DType,
		Shape: TensorShape{
			Heads:    1,
			Tokens:   spec.Tokens,
			HeadDim:  spec.HeadDim,
			ValueDim: spec.ValueDim,
			PageSize: spec.Tokens,
		},
		KeyData:   keyData,
		ValueData: valueData,
	}
	query := AttentionQuery{
		RequestID:       spec.RequestID,
		JobID:           spec.JobID,
		ModelID:         spec.ModelID,
		LayerIndex:      spec.LayerIndex,
		HeadIndex:       0,
		QueryVector:     queryVector,
		Scale:           1 / math.Sqrt(float64(spec.HeadDim)),
		DType:           spec.DType,
		CreatedAtUnixMs: spec.CreatedAtUnixMs,
	}
	if err := ValidateTensorPage(page); err != nil {
		return TensorPage{}, AttentionQuery{}, err
	}
	if err := ValidateAttentionQuery(query, page); err != nil {
		return TensorPage{}, AttentionQuery{}, err
	}
	return page, query, nil
}

func encodeBenchmarkFloats(dtype TensorDType, values []float32) ([]byte, error) {
	dtype = NormalizeTensorDType(dtype)
	elementBytes, err := tensorDTypeElementBytes(dtype)
	if err != nil {
		return nil, err
	}
	out := make([]byte, len(values)*elementBytes)
	for i, value := range values {
		if !finiteFloat64(float64(value)) {
			return nil, fmt.Errorf("%w: tensor value %d must be finite", ErrInvalidTensorPage, i)
		}
		switch dtype {
		case TensorDTypeFloat32:
			binary.LittleEndian.PutUint32(out[i*4:i*4+4], math.Float32bits(value))
		case TensorDTypeFloat16:
			binary.LittleEndian.PutUint16(out[i*2:i*2+2], float32ToTensorPlaneFloat16Bits(value))
		case TensorDTypeBFloat16:
			binary.LittleEndian.PutUint16(out[i*2:i*2+2], float32ToBFloat16Bits(value))
		default:
			return nil, fmt.Errorf("%w: unsupported dtype %q", ErrInvalidTensorDType, dtype)
		}
	}
	return out, nil
}

func benchmarkPageID(spec BenchmarkSpec) (string, error) {
	payload := struct {
		Task            string      `json:"task"`
		RequestID       string      `json:"request_id"`
		JobID           string      `json:"job_id"`
		ModelID         string      `json:"model_id"`
		LayerIndex      int         `json:"layer_index"`
		DType           TensorDType `json:"dtype"`
		Tokens          int         `json:"tokens"`
		HeadDim         int         `json:"head_dim"`
		ValueDim        int         `json:"value_dim"`
		Seed            int64       `json:"seed"`
		CreatedAtUnixMs int64       `json:"created_at_unix_ms"`
	}{
		Task:            spec.Task,
		RequestID:       spec.RequestID,
		JobID:           spec.JobID,
		ModelID:         spec.ModelID,
		LayerIndex:      spec.LayerIndex,
		DType:           spec.DType,
		Tokens:          spec.Tokens,
		HeadDim:         spec.HeadDim,
		ValueDim:        spec.ValueDim,
		Seed:            spec.Seed,
		CreatedAtUnixMs: spec.CreatedAtUnixMs,
	}
	hash, err := sha256HexJSON(payload)
	if err != nil {
		return "", err
	}
	return "tensorplane-benchmark-" + hash[:24], nil
}

func float32ToBFloat16Bits(value float32) uint16 {
	bits := math.Float32bits(value)
	roundingBias := uint32(0x7fff) + ((bits >> 16) & 1)
	return uint16((bits + roundingBias) >> 16)
}

func (s *LocalStatus) RecordSeen(jobID string) {
	s.recordSeenAt(jobID, time.Now())
}

func (s *LocalStatus) RecordExecuted(jobID string) {
	s.recordExecutedAt(jobID)
}

func (s *LocalStatus) RecordReceiptSubmitted(jobID string) {
	s.recordReceiptSubmittedAt(jobID)
}

func (s *LocalStatus) RecordReceiptFailed(jobID string, err error) {
	s.recordReceiptFailedAt(jobID, err)
}

func (s *LocalStatus) RecordError(jobID string, err error) {
	s.recordError(jobID, err)
}

func (s *LocalStatus) Snapshot() LocalStatusSnapshot {
	if s == nil {
		return LocalStatusSnapshot{}
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := s.snapshot
	if s.snapshot.LastSeenAt != nil {
		cp := *s.snapshot.LastSeenAt
		out.LastSeenAt = &cp
	}
	return out
}

func (s *LocalStatus) recordSeenAt(jobID string, at time.Time) {
	if s == nil {
		return
	}
	if at.IsZero() {
		at = time.Now()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastJobID = cleanTensorPlaneLocalStatusText(jobID, maxLocalStatusIDLen)
	s.snapshot.LastError = ""
	s.snapshot.LastSeenAt = &at
	s.snapshot.Counters.Seen++
}

func (s *LocalStatus) recordExecutedAt(jobID string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastJobID = cleanTensorPlaneLocalStatusText(jobID, maxLocalStatusIDLen)
	s.snapshot.Counters.Executed++
}

func (s *LocalStatus) recordReceiptSubmittedAt(jobID string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastJobID = cleanTensorPlaneLocalStatusText(jobID, maxLocalStatusIDLen)
	s.snapshot.LastError = ""
	s.snapshot.Counters.ReceiptSubmitted++
}

func (s *LocalStatus) recordReceiptFailedAt(jobID string, err error) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastJobID = cleanTensorPlaneLocalStatusText(jobID, maxLocalStatusIDLen)
	s.snapshot.LastError = cleanTensorPlaneLocalStatusError(err)
	s.snapshot.Counters.ReceiptFailed++
}

func (s *LocalStatus) recordError(jobID string, err error) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastJobID = cleanTensorPlaneLocalStatusText(jobID, maxLocalStatusIDLen)
	s.snapshot.LastError = cleanTensorPlaneLocalStatusError(err)
}

func cleanTensorPlaneLocalStatusError(err error) string {
	if err == nil {
		return ""
	}
	return cleanTensorPlaneLocalStatusText(err.Error(), maxLocalStatusErrorLen)
}

func cleanTensorPlaneLocalStatusText(value string, maxLen int) string {
	value = strings.TrimSpace(value)
	value = strings.ReplaceAll(value, "\r", " ")
	value = strings.ReplaceAll(value, "\n", " ")
	value = strings.Join(strings.Fields(value), " ")
	if maxLen > 0 && len(value) > maxLen {
		return value[:maxLen]
	}
	return value
}
