package modelwarm

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/Ryvion/ryvion-node/internal/v7/llamacpp"
)

const (
	ProofStatusModelWarmed     = "model_warmed"
	ProofStatusModelWarmFailed = "model_warm_failed"
)

var ErrInvalidWarmReceipt = errors.New("modelwarm: invalid warm receipt")

type WarmReceipt struct {
	JobID         string
	ResultHashHex string
	MeteringUnits uint64
	Metadata      map[string]any
}

type WarmReceiptMetadata struct {
	WarmID         string         `json:"warm_id"`
	RequestID      string         `json:"request_id"`
	JobID          string         `json:"job_id"`
	ModelID        string         `json:"model_id"`
	Backend        string         `json:"backend"`
	ModelPath      string         `json:"model_path"`
	ModelSizeBytes int64          `json:"model_size_bytes"`
	Warm           bool           `json:"warm"`
	Benchmark      map[string]any `json:"benchmark,omitempty"`
	ProofStatus    string         `json:"proof_status"`
	ErrorCode      string         `json:"error_code,omitempty"`
}

func BuildWarmReceipt(result WarmExecutionResult) (WarmReceipt, error) {
	result.Spec = NormalizeWarmSpec(result.Spec)
	if err := ValidateWarmSpec(result.Spec); err != nil {
		return WarmReceipt{}, err
	}
	metadata := warmReceiptMetadataFromResult(result)
	if err := validateWarmReceiptMetadata(metadata); err != nil {
		return WarmReceipt{}, err
	}
	hashHex, err := HashWarmReceiptMetadata(result.Spec.JobID, metadata)
	if err != nil {
		return WarmReceipt{}, err
	}
	return WarmReceipt{
		JobID:         result.Spec.JobID,
		ResultHashHex: hashHex,
		MeteringUnits: 1,
		Metadata: map[string]any{
			WarmTask: metadata.Map(),
		},
	}, nil
}

func BuildWarmRejectionReceipt(spec WarmSpec, runErr error) WarmReceipt {
	spec = NormalizeWarmSpec(spec)
	return buildWarmRejectionReceipt(spec.WarmID, spec.RequestID, spec.JobID, spec.ModelID, spec.Backend, "", 0, runErr)
}

func BuildWarmRejectionReceiptFromIdentity(identity WarmAssignmentIdentity, runErr error) WarmReceipt {
	return buildWarmRejectionReceipt(identity.WarmID, identity.RequestID, identity.JobID, identity.ModelID, backendLlamaCPP, "", 0, runErr)
}

func buildWarmRejectionReceipt(warmID, requestID, jobID, modelID, backend, modelPath string, modelSizeBytes int64, runErr error) WarmReceipt {
	jobID = cleanWarmText(jobID, maxWarmTextLen)
	if jobID == "" {
		jobID = "v7-warm-model-rejected"
	}
	metadata := WarmReceiptMetadata{
		WarmID:         cleanWarmText(warmID, maxWarmTextLen),
		RequestID:      cleanWarmText(requestID, maxWarmTextLen),
		JobID:          jobID,
		ModelID:        cleanWarmText(modelID, maxWarmTextLen),
		Backend:        normalizeBackend(backend),
		ModelPath:      safeModelPath(modelPath),
		ModelSizeBytes: modelSizeBytes,
		Warm:           false,
		ProofStatus:    ProofStatusModelWarmFailed,
		ErrorCode:      ErrorCode(runErr),
	}.clone()
	if metadata.Backend == "" {
		metadata.Backend = backendLlamaCPP
	}
	hashHex, _ := HashWarmReceiptMetadata(jobID, metadata)
	return WarmReceipt{
		JobID:         jobID,
		ResultHashHex: hashHex,
		MeteringUnits: 0,
		Metadata: map[string]any{
			WarmTask: metadata.Map(),
		},
	}
}

func HashWarmReceiptMetadata(jobID string, metadata WarmReceiptMetadata) (string, error) {
	jobID = cleanWarmText(jobID, maxWarmTextLen)
	if jobID == "" {
		return "", fmt.Errorf("%w: job_id required", ErrInvalidWarmReceipt)
	}
	if err := validateWarmReceiptMetadataForHash(metadata); err != nil {
		return "", err
	}
	envelope := struct {
		Task     string              `json:"task"`
		JobID    string              `json:"job_id"`
		Metadata WarmReceiptMetadata `json:"v7_warm_model"`
	}{
		Task:     WarmTask,
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

func (m WarmReceiptMetadata) Map() map[string]any {
	m = m.clone()
	out := map[string]any{
		"warm_id":          m.WarmID,
		"request_id":       m.RequestID,
		"job_id":           m.JobID,
		"model_id":         m.ModelID,
		"backend":          m.Backend,
		"model_path":       m.ModelPath,
		"model_size_bytes": m.ModelSizeBytes,
		"warm":             m.Warm,
		"proof_status":     m.ProofStatus,
	}
	if len(m.Benchmark) > 0 {
		out["benchmark"] = cloneWarmMap(m.Benchmark)
	}
	if m.ErrorCode != "" {
		out["error_code"] = m.ErrorCode
	}
	return out
}

func (m WarmReceiptMetadata) clone() WarmReceiptMetadata {
	m.WarmID = cleanWarmText(m.WarmID, maxWarmTextLen)
	m.RequestID = cleanWarmText(m.RequestID, maxWarmTextLen)
	m.JobID = cleanWarmText(m.JobID, maxWarmTextLen)
	m.ModelID = cleanWarmText(m.ModelID, maxWarmTextLen)
	m.Backend = normalizeBackend(m.Backend)
	if m.Backend == "" {
		m.Backend = backendLlamaCPP
	}
	m.ModelPath = safeModelPath(m.ModelPath)
	if m.ModelSizeBytes < 0 {
		m.ModelSizeBytes = 0
	}
	m.ProofStatus = cleanWarmText(m.ProofStatus, maxWarmTextLen)
	m.ErrorCode = cleanWarmText(m.ErrorCode, maxWarmTextLen)
	m.Benchmark = cloneWarmMap(m.Benchmark)
	return m
}

func warmReceiptMetadataFromResult(result WarmExecutionResult) WarmReceiptMetadata {
	result.Spec = NormalizeWarmSpec(result.Spec)
	metadata := WarmReceiptMetadata{
		WarmID:         result.Spec.WarmID,
		RequestID:      result.Spec.RequestID,
		JobID:          result.Spec.JobID,
		ModelID:        result.Spec.ModelID,
		Backend:        result.Spec.Backend,
		ModelPath:      result.ModelPath,
		ModelSizeBytes: result.ModelSizeBytes,
		Warm:           result.Warm,
		ProofStatus:    ProofStatusModelWarmed,
	}
	if result.Benchmark != nil {
		metadata.Benchmark = benchmarkReceiptMap(*result.Benchmark)
	}
	return metadata.clone()
}

func validateWarmReceiptMetadata(metadata WarmReceiptMetadata) error {
	metadata = metadata.clone()
	var errs []error
	if metadata.ProofStatus == ProofStatusModelWarmed {
		if metadata.WarmID == "" {
			errs = append(errs, fmt.Errorf("%w: warm_id required", ErrInvalidWarmReceipt))
		}
		if metadata.RequestID == "" {
			errs = append(errs, fmt.Errorf("%w: request_id required", ErrInvalidWarmReceipt))
		}
		if metadata.ModelID == "" {
			errs = append(errs, fmt.Errorf("%w: model_id required", ErrInvalidWarmReceipt))
		}
		if metadata.ModelPath == "" {
			errs = append(errs, fmt.Errorf("%w: warmed receipt requires model_path", ErrInvalidWarmReceipt))
		}
		if !metadata.Warm {
			errs = append(errs, fmt.Errorf("%w: warmed receipt requires warm=true", ErrInvalidWarmReceipt))
		}
	}
	if err := validateWarmReceiptMetadataForHash(metadata); err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

func validateWarmReceiptMetadataForHash(metadata WarmReceiptMetadata) error {
	metadata = metadata.clone()
	var errs []error
	if metadata.JobID == "" {
		errs = append(errs, fmt.Errorf("%w: job_id required", ErrInvalidWarmReceipt))
	}
	if metadata.Backend == "" {
		errs = append(errs, fmt.Errorf("%w: backend required", ErrInvalidWarmReceipt))
	}
	if metadata.ModelSizeBytes < 0 {
		errs = append(errs, fmt.Errorf("%w: model_size_bytes must be non-negative", ErrInvalidWarmReceipt))
	}
	switch metadata.ProofStatus {
	case ProofStatusModelWarmed:
	case ProofStatusModelWarmFailed:
		if metadata.ErrorCode == "" {
			errs = append(errs, fmt.Errorf("%w: failed receipt requires error_code", ErrInvalidWarmReceipt))
		}
	default:
		errs = append(errs, fmt.Errorf("%w: proof_status unknown %q", ErrInvalidWarmReceipt, metadata.ProofStatus))
	}
	return errors.Join(errs...)
}

func benchmarkReceiptMap(snapshot llamacpp.BenchmarkStatusSnapshot) map[string]any {
	metrics := snapshot.Metrics
	return map[string]any{
		"status":             cleanWarmText(snapshot.Status, maxWarmTextLen),
		"proof_status":       cleanWarmText(metrics.ProofStatus, maxWarmTextLen),
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

func cloneWarmMap(in map[string]any) map[string]any {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]any, len(in))
	for key, value := range in {
		key = cleanWarmText(key, maxWarmTextLen)
		if key == "" {
			continue
		}
		out[key] = value
	}
	return out
}

func WarmReceiptJSONContainsNoUnsafeFields(receipt WarmReceipt) bool {
	raw, err := json.Marshal(receipt.Metadata)
	if err != nil {
		return false
	}
	lower := bytes.ToLower(raw)
	for _, forbidden := range [][]byte{
		[]byte("raw_prompt"),
		[]byte("prompt_text"),
		[]byte("raw_output"),
		[]byte("output_text"),
		[]byte("generated_text"),
		[]byte("model_output"),
		[]byte("logprobs"),
		[]byte("tensor_bytes"),
		[]byte("raw_tensor"),
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

func safeModelPath(pathValue string) string {
	pathValue = strings.TrimSpace(pathValue)
	if pathValue == "" {
		return ""
	}
	return cleanWarmText(filepath.Clean(pathValue), maxWarmPathLen)
}
