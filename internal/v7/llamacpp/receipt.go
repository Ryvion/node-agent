package llamacpp

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
)

var ErrInvalidBackendBenchmarkReceipt = errors.New("llamacpp: invalid backend benchmark receipt")

type BackendBenchmarkReceipt struct {
	JobID         string
	ResultHashHex string
	MeteringUnits uint64
	Metadata      map[string]any
}

type BackendBenchmarkReceiptMetadata struct {
	RequestID      string  `json:"request_id"`
	JobID          string  `json:"job_id"`
	Backend        string  `json:"backend"`
	ModelID        string  `json:"model_id"`
	Available      bool    `json:"available"`
	SidecarHealthy bool    `json:"sidecar_healthy"`
	PromptHash     string  `json:"prompt_hash"`
	OutputHash     string  `json:"output_hash"`
	OutputBytes    int64   `json:"output_bytes"`
	WarmupRuns     int     `json:"warmup_runs"`
	MeasuredRuns   int     `json:"measured_runs"`
	P50TTFTMs      int64   `json:"p50_ttft_ms"`
	P95TTFTMs      int64   `json:"p95_ttft_ms"`
	P50TotalTimeMs int64   `json:"p50_total_time_ms"`
	P95TotalTimeMs int64   `json:"p95_total_time_ms"`
	P50DecodeTPS   float64 `json:"p50_decode_tps"`
	P95DecodeTPS   float64 `json:"p95_decode_tps"`
	P50EndToEndTPS float64 `json:"p50_end_to_end_tps"`
	P95EndToEndTPS float64 `json:"p95_end_to_end_tps"`
	ProofStatus    string  `json:"proof_status"`
}

func BuildBackendBenchmarkReceipt(spec BackendBenchmarkSpec, snapshot BenchmarkStatusSnapshot) (BackendBenchmarkReceipt, error) {
	spec = normalizeBackendBenchmarkSpec(spec)
	if err := ValidateBackendBenchmarkSpec(spec); err != nil {
		return BackendBenchmarkReceipt{}, err
	}
	snapshot = normalizeBenchmarkStatusSnapshot(snapshot)

	metadata := backendBenchmarkReceiptMetadataFromSnapshot(spec, snapshot)
	if err := validateBackendBenchmarkReceiptMetadata(spec, metadata); err != nil {
		return BackendBenchmarkReceipt{}, err
	}

	hashHex, err := HashBackendBenchmarkReceiptMetadata(spec.JobID, metadata)
	if err != nil {
		return BackendBenchmarkReceipt{}, err
	}

	meteringUnits := uint64(1)
	if metadata.ProofStatus != BenchmarkProofStatusMeasured {
		meteringUnits = 0
	}
	return BackendBenchmarkReceipt{
		JobID:         spec.JobID,
		ResultHashHex: hashHex,
		MeteringUnits: meteringUnits,
		Metadata: map[string]any{
			BackendBenchmarkTask: metadata.Map(),
		},
	}, nil
}

func BuildBackendBenchmarkRejectionReceipt(jobID string, runErr error) BackendBenchmarkReceipt {
	jobID = cleanStatusText(jobID, maxBackendBenchmarkIDLen)
	if jobID == "" {
		jobID = "v7-llamacpp-backend-benchmark-rejected"
	}
	reason := "llamacpp_backend_benchmark_rejected"
	if runErr != nil {
		reason = cleanStatusText(runErr.Error(), maxStatusReasonLen)
		if reason == "" {
			reason = "llamacpp_backend_benchmark_rejected"
		}
	}
	metadata := BackendBenchmarkReceiptMetadata{
		JobID:          jobID,
		Backend:        BackendName,
		Available:      false,
		SidecarHealthy: false,
		PromptHash:     HashBenchmarkPrompt(),
		ProofStatus:    BenchmarkProofStatusFailed,
	}
	payload := struct {
		Task     string                          `json:"task"`
		JobID    string                          `json:"job_id"`
		Error    string                          `json:"error"`
		Metadata BackendBenchmarkReceiptMetadata `json:"v7_llamacpp_backend_benchmark"`
	}{
		Task:     BackendBenchmarkTask,
		JobID:    jobID,
		Error:    reason,
		Metadata: metadata,
	}
	encoded, _ := json.Marshal(payload)
	sum := sha256.Sum256(encoded)
	return BackendBenchmarkReceipt{
		JobID:         jobID,
		ResultHashHex: hex.EncodeToString(sum[:]),
		MeteringUnits: 0,
		Metadata: map[string]any{
			BackendBenchmarkTask: metadata.Map(),
		},
	}
}

func HashBackendBenchmarkReceiptMetadata(jobID string, metadata BackendBenchmarkReceiptMetadata) (string, error) {
	jobID = cleanStatusText(jobID, maxBackendBenchmarkIDLen)
	if jobID == "" {
		return "", fmt.Errorf("%w: job_id required for result hash", ErrInvalidBackendBenchmarkReceipt)
	}
	if err := validateBackendBenchmarkReceiptMetadataForHash(metadata); err != nil {
		return "", err
	}
	envelope := struct {
		Task     string                          `json:"task"`
		JobID    string                          `json:"job_id"`
		Metadata BackendBenchmarkReceiptMetadata `json:"v7_llamacpp_backend_benchmark"`
	}{
		Task:     BackendBenchmarkTask,
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

func (m BackendBenchmarkReceiptMetadata) Map() map[string]any {
	m = m.clone()
	return map[string]any{
		"request_id":         m.RequestID,
		"job_id":             m.JobID,
		"backend":            m.Backend,
		"model_id":           m.ModelID,
		"available":          m.Available,
		"sidecar_healthy":    m.SidecarHealthy,
		"prompt_hash":        m.PromptHash,
		"output_hash":        m.OutputHash,
		"output_bytes":       m.OutputBytes,
		"warmup_runs":        m.WarmupRuns,
		"measured_runs":      m.MeasuredRuns,
		"p50_ttft_ms":        m.P50TTFTMs,
		"p95_ttft_ms":        m.P95TTFTMs,
		"p50_total_time_ms":  m.P50TotalTimeMs,
		"p95_total_time_ms":  m.P95TotalTimeMs,
		"p50_decode_tps":     m.P50DecodeTPS,
		"p95_decode_tps":     m.P95DecodeTPS,
		"p50_end_to_end_tps": m.P50EndToEndTPS,
		"p95_end_to_end_tps": m.P95EndToEndTPS,
		"proof_status":       m.ProofStatus,
	}
}

func (m BackendBenchmarkReceiptMetadata) clone() BackendBenchmarkReceiptMetadata {
	m.RequestID = cleanStatusText(m.RequestID, maxBackendBenchmarkIDLen)
	m.JobID = cleanStatusText(m.JobID, maxBackendBenchmarkIDLen)
	m.Backend = normalizeBackendBenchmarkName(m.Backend)
	m.ModelID = cleanStatusText(m.ModelID, maxBackendBenchmarkModelIDLen)
	m.PromptHash = cleanHash(m.PromptHash)
	m.OutputHash = cleanHash(m.OutputHash)
	m.ProofStatus = cleanStatusText(m.ProofStatus, maxStatusReasonLen)
	m.P50DecodeTPS = roundTPS(m.P50DecodeTPS)
	m.P95DecodeTPS = roundTPS(m.P95DecodeTPS)
	m.P50EndToEndTPS = roundTPS(m.P50EndToEndTPS)
	m.P95EndToEndTPS = roundTPS(m.P95EndToEndTPS)
	if m.OutputBytes < 0 {
		m.OutputBytes = 0
	}
	if m.WarmupRuns < 0 {
		m.WarmupRuns = 0
	}
	if m.MeasuredRuns < 0 {
		m.MeasuredRuns = 0
	}
	if m.P50TTFTMs < 0 {
		m.P50TTFTMs = 0
	}
	if m.P95TTFTMs < 0 {
		m.P95TTFTMs = 0
	}
	if m.P50TotalTimeMs < 0 {
		m.P50TotalTimeMs = 0
	}
	if m.P95TotalTimeMs < 0 {
		m.P95TotalTimeMs = 0
	}
	return m
}

func backendBenchmarkReceiptMetadataFromSnapshot(spec BackendBenchmarkSpec, snapshot BenchmarkStatusSnapshot) BackendBenchmarkReceiptMetadata {
	metrics := normalizeBenchmarkMetrics(snapshot.Metrics)
	proofStatus := metrics.ProofStatus
	if proofStatus == "" {
		proofStatus = BenchmarkProofStatusUnavailable
	}
	if snapshot.Status == BenchmarkStatusCompleted && proofStatus == BenchmarkProofStatusUnavailable && metrics.OutputHash != "" {
		proofStatus = BenchmarkProofStatusMeasured
	}
	return BackendBenchmarkReceiptMetadata{
		RequestID:      spec.RequestID,
		JobID:          spec.JobID,
		Backend:        firstNonEmpty(metrics.Backend, spec.Backend, BackendName),
		ModelID:        firstNonEmpty(metrics.ModelID, spec.ModelID),
		Available:      metrics.Available,
		SidecarHealthy: metrics.SidecarHealthy,
		PromptHash:     firstNonEmpty(metrics.PromptHash, HashBenchmarkPrompt()),
		OutputHash:     metrics.OutputHash,
		OutputBytes:    metrics.OutputBytes,
		WarmupRuns:     metrics.WarmupRuns,
		MeasuredRuns:   metrics.MeasuredRuns,
		P50TTFTMs:      metrics.P50TTFTMs,
		P95TTFTMs:      metrics.P95TTFTMs,
		P50TotalTimeMs: metrics.P50TotalTimeMs,
		P95TotalTimeMs: metrics.P95TotalTimeMs,
		P50DecodeTPS:   metrics.P50DecodeTPS,
		P95DecodeTPS:   metrics.P95DecodeTPS,
		P50EndToEndTPS: metrics.P50EndToEndTPS,
		P95EndToEndTPS: metrics.P95EndToEndTPS,
		ProofStatus:    proofStatus,
	}.clone()
}

func validateBackendBenchmarkReceiptMetadata(spec BackendBenchmarkSpec, metadata BackendBenchmarkReceiptMetadata) error {
	spec = normalizeBackendBenchmarkSpec(spec)
	metadata = metadata.clone()
	var errs []error
	if metadata.RequestID != spec.RequestID {
		errs = append(errs, fmt.Errorf("%w: receipt request_id must match spec", ErrInvalidBackendBenchmarkReceipt))
	}
	if metadata.JobID != spec.JobID {
		errs = append(errs, fmt.Errorf("%w: receipt job_id must match spec", ErrInvalidBackendBenchmarkReceipt))
	}
	if metadata.Backend != spec.Backend {
		errs = append(errs, fmt.Errorf("%w: receipt backend must match spec", ErrInvalidBackendBenchmarkReceipt))
	}
	if metadata.ModelID == "" {
		errs = append(errs, fmt.Errorf("%w: receipt model_id required", ErrInvalidBackendBenchmarkReceipt))
	}
	if err := validateBackendBenchmarkReceiptMetadataForHash(metadata); err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

func validateBackendBenchmarkReceiptMetadataForHash(metadata BackendBenchmarkReceiptMetadata) error {
	metadata = metadata.clone()
	var errs []error
	if metadata.JobID == "" {
		errs = append(errs, fmt.Errorf("%w: receipt job_id required", ErrInvalidBackendBenchmarkReceipt))
	}
	if metadata.Backend == "" {
		errs = append(errs, fmt.Errorf("%w: receipt backend required", ErrInvalidBackendBenchmarkReceipt))
	} else if metadata.Backend != BackendName {
		errs = append(errs, fmt.Errorf("%w: receipt backend must be %q", ErrInvalidBackendBenchmarkReceipt, BackendName))
	}
	if metadata.PromptHash == "" {
		errs = append(errs, fmt.Errorf("%w: receipt prompt_hash required", ErrInvalidBackendBenchmarkReceipt))
	}
	if metadata.OutputHash != "" && cleanHash(metadata.OutputHash) == "" {
		errs = append(errs, fmt.Errorf("%w: receipt output_hash must be sha256:<64 hex>", ErrInvalidBackendBenchmarkReceipt))
	}
	if metadata.WarmupRuns < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt warmup_runs must be non-negative", ErrInvalidBackendBenchmarkReceipt))
	}
	if metadata.MeasuredRuns < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt measured_runs must be non-negative", ErrInvalidBackendBenchmarkReceipt))
	}
	if metadata.OutputBytes < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt output_bytes must be non-negative", ErrInvalidBackendBenchmarkReceipt))
	}
	if metadata.P50TTFTMs < 0 || metadata.P95TTFTMs < 0 || metadata.P50TotalTimeMs < 0 || metadata.P95TotalTimeMs < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt timing percentiles must be non-negative", ErrInvalidBackendBenchmarkReceipt))
	}
	if !finiteBackendBenchmarkFloat(metadata.P50DecodeTPS) || !finiteBackendBenchmarkFloat(metadata.P95DecodeTPS) ||
		!finiteBackendBenchmarkFloat(metadata.P50EndToEndTPS) || !finiteBackendBenchmarkFloat(metadata.P95EndToEndTPS) {
		errs = append(errs, fmt.Errorf("%w: receipt tps percentiles must be finite", ErrInvalidBackendBenchmarkReceipt))
	}
	switch metadata.ProofStatus {
	case BenchmarkProofStatusMeasured:
		if !metadata.Available || !metadata.SidecarHealthy {
			errs = append(errs, fmt.Errorf("%w: measured receipt requires available healthy sidecar", ErrInvalidBackendBenchmarkReceipt))
		}
		if metadata.OutputHash == "" {
			errs = append(errs, fmt.Errorf("%w: measured receipt requires output_hash", ErrInvalidBackendBenchmarkReceipt))
		}
	case BenchmarkProofStatusUnavailable, BenchmarkProofStatusFailed:
	default:
		errs = append(errs, fmt.Errorf("%w: receipt proof_status unknown %q", ErrInvalidBackendBenchmarkReceipt, metadata.ProofStatus))
	}
	return errors.Join(errs...)
}

func finiteBackendBenchmarkFloat(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value >= 0
}

func BackendBenchmarkReceiptJSONContainsNoRawText(receipt BackendBenchmarkReceipt) bool {
	raw, err := json.Marshal(receipt.Metadata)
	if err != nil {
		return false
	}
	lower := bytes.ToLower(raw)
	for _, forbidden := range [][]byte{
		[]byte(strings.ToLower(internalBenchmarkPrompt)),
		[]byte("prompt_text"),
		[]byte("raw_prompt"),
		[]byte("output_text"),
		[]byte("generated_text"),
		[]byte("raw_output"),
		[]byte("logprobs"),
		[]byte("tensor"),
	} {
		if bytes.Contains(lower, forbidden) {
			return false
		}
	}
	return true
}
