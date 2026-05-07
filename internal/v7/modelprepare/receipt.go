package modelprepare

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/Ryvion/node-agent/internal/v7/llamacpp"
)

const (
	ProofStatusModelPrepared = "model_prepared"
	ProofStatusPrepareFailed = "model_prepare_failed"
)

var ErrInvalidPrepareReceipt = errors.New("modelprepare: invalid prepare receipt")

type PrepareReceipt struct {
	JobID         string
	ResultHashHex string
	MeteringUnits uint64
	Metadata      map[string]any
}

type PrepareReceiptMetadata struct {
	PrepareID      string         `json:"prepare_id"`
	RequestID      string         `json:"request_id"`
	JobID          string         `json:"job_id"`
	ModelID        string         `json:"model_id"`
	Backend        string         `json:"backend"`
	Downloaded     bool           `json:"downloaded"`
	HashVerified   bool           `json:"hash_verified"`
	Installed      bool           `json:"installed"`
	ModelPath      string         `json:"model_path"`
	ModelSizeBytes int64          `json:"model_size_bytes"`
	Warm           bool           `json:"warm"`
	Benchmark      map[string]any `json:"benchmark,omitempty"`
	ProofStatus    string         `json:"proof_status"`
	ErrorCode      string         `json:"error_code,omitempty"`
}

func BuildPrepareReceipt(result PrepareExecutionResult) (PrepareReceipt, error) {
	result.Spec = NormalizePrepareSpec(result.Spec)
	if err := ValidatePrepareSpec(result.Spec); err != nil {
		return PrepareReceipt{}, err
	}
	metadata := prepareReceiptMetadataFromResult(result)
	if err := validatePrepareReceiptMetadata(metadata); err != nil {
		return PrepareReceipt{}, err
	}
	hashHex, err := HashPrepareReceiptMetadata(result.Spec.JobID, metadata)
	if err != nil {
		return PrepareReceipt{}, err
	}
	return PrepareReceipt{
		JobID:         result.Spec.JobID,
		ResultHashHex: hashHex,
		MeteringUnits: 1,
		Metadata: map[string]any{
			PrepareTask: metadata.Map(),
		},
	}, nil
}

func BuildPrepareRejectionReceipt(spec PrepareSpec, runErr error) PrepareReceipt {
	spec = NormalizePrepareSpec(spec)
	return buildPrepareRejectionReceipt(spec.PrepareID, spec.RequestID, spec.JobID, spec.ModelID, spec.Backend, runErr)
}

func BuildPrepareRejectionReceiptFromIdentity(identity PrepareAssignmentIdentity, runErr error) PrepareReceipt {
	return buildPrepareRejectionReceipt(identity.PrepareID, identity.RequestID, identity.JobID, identity.ModelID, backendLlamaCPP, runErr)
}

func buildPrepareRejectionReceipt(prepareID, requestID, jobID, modelID, backend string, runErr error) PrepareReceipt {
	jobID = cleanPrepareText(jobID, maxPrepareTextLen)
	if jobID == "" {
		jobID = "v7-prepare-model-rejected"
	}
	metadata := PrepareReceiptMetadata{
		PrepareID:      cleanPrepareText(prepareID, maxPrepareTextLen),
		RequestID:      cleanPrepareText(requestID, maxPrepareTextLen),
		JobID:          jobID,
		ModelID:        cleanPrepareText(modelID, maxPrepareTextLen),
		Backend:        normalizeBackend(backend),
		Downloaded:     false,
		HashVerified:   false,
		Installed:      false,
		ModelSizeBytes: 0,
		Warm:           false,
		ProofStatus:    ProofStatusPrepareFailed,
		ErrorCode:      ErrorCode(runErr),
	}
	if metadata.Backend == "" {
		metadata.Backend = backendLlamaCPP
	}
	hashHex, _ := HashPrepareReceiptMetadata(jobID, metadata)
	return PrepareReceipt{
		JobID:         jobID,
		ResultHashHex: hashHex,
		MeteringUnits: 0,
		Metadata: map[string]any{
			PrepareTask: metadata.Map(),
		},
	}
}

func HashPrepareReceiptMetadata(jobID string, metadata PrepareReceiptMetadata) (string, error) {
	jobID = cleanPrepareText(jobID, maxPrepareTextLen)
	if jobID == "" {
		return "", fmt.Errorf("%w: job_id required", ErrInvalidPrepareReceipt)
	}
	if err := validatePrepareReceiptMetadataForHash(metadata); err != nil {
		return "", err
	}
	envelope := struct {
		Task     string                 `json:"task"`
		JobID    string                 `json:"job_id"`
		Metadata PrepareReceiptMetadata `json:"v7_prepare_model"`
	}{
		Task:     PrepareTask,
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

func (m PrepareReceiptMetadata) Map() map[string]any {
	m = m.clone()
	out := map[string]any{
		"prepare_id":       m.PrepareID,
		"request_id":       m.RequestID,
		"job_id":           m.JobID,
		"model_id":         m.ModelID,
		"backend":          m.Backend,
		"downloaded":       m.Downloaded,
		"hash_verified":    m.HashVerified,
		"installed":        m.Installed,
		"model_path":       m.ModelPath,
		"model_size_bytes": m.ModelSizeBytes,
		"warm":             m.Warm,
		"proof_status":     m.ProofStatus,
	}
	if len(m.Benchmark) > 0 {
		out["benchmark"] = clonePrepareMap(m.Benchmark)
	}
	if m.ErrorCode != "" {
		out["error_code"] = m.ErrorCode
	}
	return out
}

func (m PrepareReceiptMetadata) clone() PrepareReceiptMetadata {
	m.PrepareID = cleanPrepareText(m.PrepareID, maxPrepareTextLen)
	m.RequestID = cleanPrepareText(m.RequestID, maxPrepareTextLen)
	m.JobID = cleanPrepareText(m.JobID, maxPrepareTextLen)
	m.ModelID = cleanPrepareText(m.ModelID, maxPrepareTextLen)
	m.Backend = normalizeBackend(m.Backend)
	if m.Backend == "" {
		m.Backend = backendLlamaCPP
	}
	m.ModelPath = safeModelPath(m.ModelPath)
	if m.ModelSizeBytes < 0 {
		m.ModelSizeBytes = 0
	}
	m.ProofStatus = cleanPrepareText(m.ProofStatus, maxPrepareTextLen)
	m.ErrorCode = cleanPrepareText(m.ErrorCode, maxPrepareTextLen)
	m.Benchmark = clonePrepareMap(m.Benchmark)
	return m
}

func prepareReceiptMetadataFromResult(result PrepareExecutionResult) PrepareReceiptMetadata {
	result.Spec = NormalizePrepareSpec(result.Spec)
	metadata := PrepareReceiptMetadata{
		PrepareID:      result.Spec.PrepareID,
		RequestID:      result.Spec.RequestID,
		JobID:          result.Spec.JobID,
		ModelID:        result.Spec.ModelID,
		Backend:        result.Spec.Backend,
		Downloaded:     result.Downloaded,
		HashVerified:   result.HashVerified,
		Installed:      result.Installed,
		ModelPath:      result.ModelPath,
		ModelSizeBytes: result.ModelSizeBytes,
		Warm:           result.Warm,
		ProofStatus:    ProofStatusModelPrepared,
	}
	if result.Benchmark != nil {
		metadata.Benchmark = benchmarkReceiptMap(*result.Benchmark)
	}
	return metadata.clone()
}

func validatePrepareReceiptMetadata(metadata PrepareReceiptMetadata) error {
	metadata = metadata.clone()
	var errs []error
	if metadata.ProofStatus == ProofStatusModelPrepared {
		if metadata.PrepareID == "" {
			errs = append(errs, fmt.Errorf("%w: prepare_id required", ErrInvalidPrepareReceipt))
		}
		if metadata.RequestID == "" {
			errs = append(errs, fmt.Errorf("%w: request_id required", ErrInvalidPrepareReceipt))
		}
		if metadata.ModelID == "" {
			errs = append(errs, fmt.Errorf("%w: model_id required", ErrInvalidPrepareReceipt))
		}
		if !metadata.Installed {
			errs = append(errs, fmt.Errorf("%w: installed receipt requires installed=true", ErrInvalidPrepareReceipt))
		}
		if metadata.ModelPath == "" {
			errs = append(errs, fmt.Errorf("%w: installed receipt requires model_path", ErrInvalidPrepareReceipt))
		}
	}
	if err := validatePrepareReceiptMetadataForHash(metadata); err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

func validatePrepareReceiptMetadataForHash(metadata PrepareReceiptMetadata) error {
	metadata = metadata.clone()
	var errs []error
	if metadata.JobID == "" {
		errs = append(errs, fmt.Errorf("%w: job_id required", ErrInvalidPrepareReceipt))
	}
	if metadata.Backend == "" {
		errs = append(errs, fmt.Errorf("%w: backend required", ErrInvalidPrepareReceipt))
	}
	if metadata.ModelSizeBytes < 0 {
		errs = append(errs, fmt.Errorf("%w: model_size_bytes must be non-negative", ErrInvalidPrepareReceipt))
	}
	switch metadata.ProofStatus {
	case ProofStatusModelPrepared:
	case ProofStatusPrepareFailed:
		if metadata.ErrorCode == "" {
			errs = append(errs, fmt.Errorf("%w: failed receipt requires error_code", ErrInvalidPrepareReceipt))
		}
	default:
		errs = append(errs, fmt.Errorf("%w: proof_status unknown %q", ErrInvalidPrepareReceipt, metadata.ProofStatus))
	}
	return errors.Join(errs...)
}

func benchmarkReceiptMap(snapshot llamacpp.BenchmarkStatusSnapshot) map[string]any {
	metrics := snapshot.Metrics
	return map[string]any{
		"status":             cleanPrepareText(snapshot.Status, maxPrepareTextLen),
		"proof_status":       cleanPrepareText(metrics.ProofStatus, maxPrepareTextLen),
		"sidecar_healthy":    metrics.SidecarHealthy,
		"model_loaded":       metrics.ModelLoaded,
		"warmup_runs":        metrics.WarmupRuns,
		"measured_runs":      metrics.MeasuredRuns,
		"p50_ttft_ms":        metrics.P50TTFTMs,
		"p95_ttft_ms":        metrics.P95TTFTMs,
		"p50_decode_tps":     metrics.P50DecodeTPS,
		"p50_end_to_end_tps": metrics.P50EndToEndTPS,
		"tokens_generated":   metrics.TokensGenerated,
	}
}

func clonePrepareMap(in map[string]any) map[string]any {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]any, len(in))
	for key, value := range in {
		key = cleanPrepareText(key, maxPrepareTextLen)
		if key == "" {
			continue
		}
		out[key] = value
	}
	return out
}

func PrepareReceiptJSONContainsNoUnsafeFields(receipt PrepareReceipt) bool {
	raw, err := json.Marshal(receipt.Metadata)
	if err != nil {
		return false
	}
	lower := bytes.ToLower(raw)
	for _, forbidden := range [][]byte{
		[]byte("artifact_uri"),
		[]byte("raw_prompt"),
		[]byte("prompt_text"),
		[]byte("raw_output"),
		[]byte("output_text"),
		[]byte("generated_text"),
		[]byte("model_bytes"),
		[]byte("raw_model"),
		[]byte("tensor_bytes"),
		[]byte("private_key"),
		[]byte("secret"),
		[]byte("token="),
	} {
		if bytes.Contains(lower, forbidden) {
			return false
		}
	}
	return true
}
