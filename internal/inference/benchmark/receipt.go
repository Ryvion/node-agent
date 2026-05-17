package inferencebench

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/Ryvion/ryvion-node/internal/runtimes/llamacpp"
)

var ErrInvalidBenchmarkReceipt = errors.New("inferencebench: invalid benchmark receipt")

type BenchmarkReceipt struct {
	JobID         string
	ResultHashHex string
	MeteringUnits uint64
	Metadata      map[string]any
}

type BenchmarkReceiptMetadata struct {
	RequestID       string  `json:"request_id"`
	JobID           string  `json:"job_id"`
	Backend         string  `json:"backend"`
	ModelID         string  `json:"model_id"`
	PromptHash      string  `json:"prompt_hash"`
	OutputHash      string  `json:"output_hash,omitempty"`
	TokensGenerated int64   `json:"tokens_generated"`
	P50TTFTMs       int64   `json:"p50_ttft_ms"`
	P95TTFTMs       int64   `json:"p95_ttft_ms"`
	P50DecodeTPS    float64 `json:"p50_decode_tps"`
	P50EndToEndTPS  float64 `json:"p50_end_to_end_tps"`
	ProofStatus     string  `json:"proof_status"`
	ErrorCode       string  `json:"error_code,omitempty"`
}

func BuildBenchmarkReceipt(result BenchmarkExecutionResult) (BenchmarkReceipt, error) {
	result = normalizeExecutionResult(result)
	if err := ValidateBenchmarkSpec(result.Spec); err != nil {
		return BenchmarkReceipt{}, err
	}
	if result.ProofStatus == ProofStatusMeasured {
		if err := ensureMeasuredMetrics(result); err != nil {
			return BenchmarkReceipt{}, err
		}
	}

	metadata := benchmarkReceiptMetadataFromResult(result)
	if err := validateBenchmarkReceiptMetadata(result.Spec, metadata); err != nil {
		return BenchmarkReceipt{}, err
	}
	hashHex, err := HashBenchmarkReceiptMetadata(result.Spec.JobID, metadata)
	if err != nil {
		return BenchmarkReceipt{}, err
	}
	meteringUnits := uint64(0)
	if metadata.ProofStatus == ProofStatusMeasured {
		meteringUnits = 1
	}
	return BenchmarkReceipt{
		JobID:         result.Spec.JobID,
		ResultHashHex: hashHex,
		MeteringUnits: meteringUnits,
		Metadata: map[string]any{
			BenchmarkTask: metadata.Map(),
		},
	}, nil
}

func BuildBenchmarkRejectionReceipt(jobID string, runErr error) BenchmarkReceipt {
	jobID = cleanBenchmarkText(jobID, maxBenchmarkIDLen)
	if jobID == "" {
		jobID = "v7-backend-inference-benchmark-rejected"
	}
	reason := "backend_inference_benchmark_rejected"
	if runErr != nil {
		reason = cleanBenchmarkErrorCode(runErr.Error())
		if reason == "" {
			reason = "backend_inference_benchmark_rejected"
		}
	}
	metadata := map[string]any{
		"job_id":       jobID,
		"backend":      llamacpp.BackendName,
		"proof_status": ProofStatusRejected,
		"error":        reason,
		"error_code":   reason,
	}
	payload := struct {
		Task     string         `json:"task"`
		JobID    string         `json:"job_id"`
		Metadata map[string]any `json:"v7_backend_inference_benchmark"`
	}{
		Task:     BenchmarkTask,
		JobID:    jobID,
		Metadata: metadata,
	}
	encoded, _ := json.Marshal(payload)
	sum := sha256.Sum256(encoded)
	return BenchmarkReceipt{
		JobID:         jobID,
		ResultHashHex: hex.EncodeToString(sum[:]),
		MeteringUnits: 0,
		Metadata: map[string]any{
			BenchmarkTask: metadata,
		},
	}
}

func HashBenchmarkReceiptMetadata(jobID string, metadata BenchmarkReceiptMetadata) (string, error) {
	jobID = cleanBenchmarkText(jobID, maxBenchmarkIDLen)
	if jobID == "" {
		return "", fmt.Errorf("%w: job_id required for result hash", ErrInvalidBenchmarkReceipt)
	}
	if err := validateBenchmarkReceiptMetadataForHash(metadata); err != nil {
		return "", err
	}
	envelope := struct {
		Task     string         `json:"task"`
		JobID    string         `json:"job_id"`
		Metadata map[string]any `json:"v7_backend_inference_benchmark"`
	}{
		Task:     BenchmarkTask,
		JobID:    jobID,
		Metadata: metadata.Map(),
	}
	encoded, err := json.Marshal(envelope)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:]), nil
}

func (m BenchmarkReceiptMetadata) Map() map[string]any {
	m = m.clone()
	if m.ProofStatus != ProofStatusMeasured {
		out := map[string]any{
			"job_id":       m.JobID,
			"backend":      m.Backend,
			"prompt_hash":  m.PromptHash,
			"proof_status": ProofStatusRejected,
		}
		if m.RequestID != "" {
			out["request_id"] = m.RequestID
		}
		if m.ModelID != "" {
			out["model_id"] = m.ModelID
		}
		if m.ErrorCode != "" {
			out["error"] = m.ErrorCode
			out["error_code"] = m.ErrorCode
		}
		return out
	}
	out := map[string]any{
		"request_id":         m.RequestID,
		"job_id":             m.JobID,
		"backend":            m.Backend,
		"model_id":           m.ModelID,
		"prompt_hash":        m.PromptHash,
		"output_hash":        m.OutputHash,
		"tokens_generated":   m.TokensGenerated,
		"p50_ttft_ms":        m.P50TTFTMs,
		"p95_ttft_ms":        m.P95TTFTMs,
		"p50_decode_tps":     m.P50DecodeTPS,
		"p50_end_to_end_tps": m.P50EndToEndTPS,
		"proof_status":       m.ProofStatus,
	}
	return out
}

func (m BenchmarkReceiptMetadata) clone() BenchmarkReceiptMetadata {
	m.RequestID = cleanBenchmarkText(m.RequestID, maxBenchmarkIDLen)
	m.JobID = cleanBenchmarkText(m.JobID, maxBenchmarkIDLen)
	m.Backend = normalizeBackendName(m.Backend)
	m.ModelID = cleanBenchmarkText(m.ModelID, maxBenchmarkModelIDLen)
	m.PromptHash = strings.TrimSpace(m.PromptHash)
	m.OutputHash = strings.TrimSpace(m.OutputHash)
	m.ProofStatus = cleanBenchmarkText(m.ProofStatus, maxBenchmarkIDLen)
	if m.ProofStatus == ProofStatusFailed {
		m.ProofStatus = ProofStatusRejected
	}
	m.ErrorCode = cleanBenchmarkErrorCode(m.ErrorCode)
	if m.TokensGenerated < 0 {
		m.TokensGenerated = 0
	}
	if m.P50TTFTMs < 0 {
		m.P50TTFTMs = 0
	}
	if m.P95TTFTMs < 0 {
		m.P95TTFTMs = 0
	}
	if m.P95TTFTMs == 0 {
		m.P95TTFTMs = m.P50TTFTMs
	}
	if m.P95TTFTMs < m.P50TTFTMs {
		m.P95TTFTMs = m.P50TTFTMs
	}
	m.P50DecodeTPS = roundTPS(m.P50DecodeTPS)
	m.P50EndToEndTPS = roundTPS(m.P50EndToEndTPS)
	return m
}

func benchmarkReceiptMetadataFromResult(result BenchmarkExecutionResult) BenchmarkReceiptMetadata {
	result = normalizeExecutionResult(result)
	return BenchmarkReceiptMetadata{
		RequestID:       result.Spec.RequestID,
		JobID:           result.Spec.JobID,
		Backend:         firstNonEmpty(result.Backend, result.Spec.Backend),
		ModelID:         firstNonEmpty(result.ModelID, result.Spec.ModelID),
		PromptHash:      firstNonEmpty(result.PromptHash, result.Spec.PromptHash),
		OutputHash:      result.OutputHash,
		TokensGenerated: result.TokensGenerated,
		P50TTFTMs:       result.TTFTMs,
		P95TTFTMs:       firstPositiveInt64(result.P95TTFTMs, result.TTFTMs),
		P50DecodeTPS:    result.DecodeTPS,
		P50EndToEndTPS:  result.EndToEndTPS,
		ProofStatus:     result.ProofStatus,
		ErrorCode:       result.ErrorCode,
	}.clone()
}

func normalizeExecutionResult(result BenchmarkExecutionResult) BenchmarkExecutionResult {
	result.Spec = normalizeBenchmarkSpec(result.Spec)
	result.Backend = normalizeBackendName(firstNonEmpty(result.Backend, result.Spec.Backend))
	result.ModelID = cleanBenchmarkText(firstNonEmpty(result.ModelID, result.Spec.ModelID), maxBenchmarkModelIDLen)
	result.PromptHash = strings.TrimSpace(firstNonEmpty(result.PromptHash, result.Spec.PromptHash))
	result.OutputHash = strings.TrimSpace(result.OutputHash)
	result.ProofStatus = cleanBenchmarkText(result.ProofStatus, maxBenchmarkIDLen)
	if result.ProofStatus == ProofStatusFailed {
		result.ProofStatus = ProofStatusRejected
	}
	if result.ProofStatus == "" {
		result.ProofStatus = ProofStatusRejected
	}
	result.ErrorCode = cleanBenchmarkErrorCode(result.ErrorCode)
	if result.OutputBytes < 0 {
		result.OutputBytes = 0
	}
	if result.TokensGenerated < 0 {
		result.TokensGenerated = 0
	}
	if result.TTFTMs < 0 {
		result.TTFTMs = 0
	}
	if result.P95TTFTMs < 0 {
		result.P95TTFTMs = 0
	}
	if result.TotalTimeMs < 0 {
		result.TotalTimeMs = 0
	}
	result.DecodeTPS = roundTPS(result.DecodeTPS)
	result.EndToEndTPS = roundTPS(result.EndToEndTPS)
	return result
}

func validateBenchmarkReceiptMetadata(spec BenchmarkSpec, metadata BenchmarkReceiptMetadata) error {
	spec = normalizeBenchmarkSpec(spec)
	metadata = metadata.clone()
	var errs []error
	if metadata.RequestID != spec.RequestID {
		errs = append(errs, fmt.Errorf("%w: receipt request_id must match spec", ErrInvalidBenchmarkReceipt))
	}
	if metadata.JobID != spec.JobID {
		errs = append(errs, fmt.Errorf("%w: receipt job_id must match spec", ErrInvalidBenchmarkReceipt))
	}
	if metadata.Backend != spec.Backend {
		errs = append(errs, fmt.Errorf("%w: receipt backend must match spec", ErrInvalidBenchmarkReceipt))
	}
	if metadata.ModelID != spec.ModelID {
		errs = append(errs, fmt.Errorf("%w: receipt model_id must match spec", ErrInvalidBenchmarkReceipt))
	}
	if metadata.PromptHash != spec.PromptHash {
		errs = append(errs, fmt.Errorf("%w: receipt prompt_hash must match spec", ErrInvalidBenchmarkReceipt))
	}
	if err := validateBenchmarkReceiptMetadataForHash(metadata); err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

func validateBenchmarkReceiptMetadataForHash(metadata BenchmarkReceiptMetadata) error {
	metadata = metadata.clone()
	var errs []error
	if metadata.RequestID == "" {
		errs = append(errs, fmt.Errorf("%w: receipt request_id required", ErrInvalidBenchmarkReceipt))
	}
	if metadata.JobID == "" {
		errs = append(errs, fmt.Errorf("%w: receipt job_id required", ErrInvalidBenchmarkReceipt))
	}
	if metadata.Backend == "" {
		errs = append(errs, fmt.Errorf("%w: receipt backend required", ErrInvalidBenchmarkReceipt))
	} else if metadata.Backend != llamacpp.BackendName {
		errs = append(errs, fmt.Errorf("%w: receipt backend must be %q", ErrInvalidBenchmarkReceipt, llamacpp.BackendName))
	}
	if metadata.ModelID == "" {
		errs = append(errs, fmt.Errorf("%w: receipt model_id required", ErrInvalidBenchmarkReceipt))
	}
	if err := validateSHA256ID(metadata.PromptHash, "receipt prompt_hash", ErrInvalidBenchmarkReceipt); err != nil {
		errs = append(errs, err)
	}
	if metadata.OutputHash != "" {
		if err := validateSHA256ID(metadata.OutputHash, "receipt output_hash", ErrInvalidBenchmarkReceipt); err != nil {
			errs = append(errs, err)
		}
	}
	if metadata.TokensGenerated < 0 || metadata.P50TTFTMs < 0 || metadata.P95TTFTMs < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt metrics must be non-negative", ErrInvalidBenchmarkReceipt))
	}
	if !finiteTPS(metadata.P50DecodeTPS) || !finiteTPS(metadata.P50EndToEndTPS) {
		errs = append(errs, fmt.Errorf("%w: receipt tps metrics must be finite", ErrInvalidBenchmarkReceipt))
	}
	switch metadata.ProofStatus {
	case ProofStatusMeasured:
		if metadata.OutputHash == "" {
			errs = append(errs, fmt.Errorf("%w: measured receipt requires output_hash", ErrInvalidBenchmarkReceipt))
		}
		if metadata.TokensGenerated <= 0 {
			errs = append(errs, fmt.Errorf("%w: measured receipt requires tokens_generated", ErrInvalidBenchmarkReceipt))
		}
		if metadata.P50TTFTMs < 0 || metadata.P95TTFTMs < metadata.P50TTFTMs {
			errs = append(errs, fmt.Errorf("%w: measured receipt requires p95_ttft_ms >= p50_ttft_ms", ErrInvalidBenchmarkReceipt))
		}
	case ProofStatusRejected:
		if metadata.ErrorCode == "" {
			errs = append(errs, fmt.Errorf("%w: rejected receipt requires error", ErrInvalidBenchmarkReceipt))
		}
	default:
		errs = append(errs, fmt.Errorf("%w: receipt proof_status unknown %q", ErrInvalidBenchmarkReceipt, metadata.ProofStatus))
	}
	return errors.Join(errs...)
}

func hashBenchmarkOutput(spec BenchmarkSpec, output []byte) string {
	hash := sha256.New()
	hash.Write([]byte("ryvion:v7:backend_inference_benchmark_output:v1\n"))
	hash.Write([]byte("job_id:"))
	hash.Write([]byte(spec.JobID))
	hash.Write([]byte("\nbytes:"))
	hash.Write([]byte(fmt.Sprintf("%d\n", len(output))))
	hash.Write(output)
	return "sha256:" + hex.EncodeToString(hash.Sum(nil))
}

func BenchmarkReceiptJSONContainsNoRawText(receipt BenchmarkReceipt) bool {
	raw, err := json.Marshal(receipt.Metadata)
	if err != nil {
		return false
	}
	lower := bytes.ToLower(raw)
	for _, forbidden := range [][]byte{
		[]byte("write one short sentence"),
		[]byte("distributed computing"),
		[]byte("raw_prompt"),
		[]byte("prompt_text"),
		[]byte("messages"),
		[]byte("input_text"),
		[]byte("output_text"),
		[]byte("generated_text"),
		[]byte("raw_output"),
		[]byte("model_output"),
		[]byte("completion"),
		[]byte("logprobs"),
		[]byte("key_data"),
		[]byte("value_data"),
		[]byte("query_vector"),
		[]byte("tensor_bytes"),
		[]byte("raw_tensor"),
	} {
		if bytes.Contains(lower, bytes.ToLower(forbidden)) {
			return false
		}
	}
	return true
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			return trimmed
		}
	}
	return ""
}

func firstPositiveInt64(values ...int64) int64 {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func errorsJoin(errs ...error) error {
	var flat []error
	for _, err := range errs {
		if err != nil {
			flat = append(flat, err)
		}
	}
	return errors.Join(flat...)
}
