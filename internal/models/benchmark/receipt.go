package modelbench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

type ModelBenchmarkReceipt struct {
	JobID         string
	ResultHashHex string
	MeteringUnits uint64
	Metadata      map[string]any
}

type ModelBenchmarkReceiptMetadata struct {
	RequestID   string                       `json:"request_id"`
	ModelID     string                       `json:"model_id"`
	PromptHash  string                       `json:"prompt_hash"`
	Runtime     ModelBenchmarkReceiptRuntime `json:"runtime"`
	Metrics     ModelBenchmarkReceiptMetrics `json:"metrics"`
	OutputHash  string                       `json:"output_hash,omitempty"`
	OutputBytes int64                        `json:"output_bytes"`
	ProofStatus string                       `json:"proof_status"`
}

type ModelBenchmarkReceiptRuntime struct {
	AgentVersion             string `json:"agent_version"`
	OS                       string `json:"os"`
	Arch                     string `json:"arch"`
	NativeInferenceSupported bool   `json:"native_inference_supported"`
	NativeInferenceReady     bool   `json:"native_inference_ready"`
	RuntimeKind              string `json:"runtime_kind"`
	ModelLoaded              bool   `json:"model_loaded"`
}

type ModelBenchmarkReceiptMetrics struct {
	WallTimeMs         int64   `json:"wall_time_ms"`
	TimeToFirstTokenMs int64   `json:"time_to_first_token_ms"`
	TokensGenerated    int64   `json:"tokens_generated"`
	TokensPerSecond    float64 `json:"tokens_per_second"`
	ModelLoadState     string  `json:"model_load_state"`
	ErrorCode          string  `json:"error_code,omitempty"`
}

func BuildModelBenchmarkReceipt(result ModelBenchmarkResult) (ModelBenchmarkReceipt, error) {
	result = normalizeModelBenchmarkResult(result)
	if err := ValidateModelBenchmarkResult(result); err != nil {
		return ModelBenchmarkReceipt{}, err
	}

	metadata := modelBenchmarkReceiptMetadataFromResult(result)
	if err := validateModelBenchmarkReceiptMetadata(metadata); err != nil {
		return ModelBenchmarkReceipt{}, err
	}

	hashHex, err := HashModelBenchmarkReceiptMetadata(result.JobID, metadata)
	if err != nil {
		return ModelBenchmarkReceipt{}, err
	}

	return ModelBenchmarkReceipt{
		JobID:         result.JobID,
		ResultHashHex: hashHex,
		MeteringUnits: 1,
		Metadata: map[string]any{
			ModelBenchmarkTask: metadata.Map(),
		},
	}, nil
}

func BuildModelBenchmarkRejectionReceipt(jobID string, runErr error) ModelBenchmarkReceipt {
	jobID = strings.TrimSpace(jobID)
	if jobID == "" {
		jobID = "v7-model-benchmark-rejected"
	}
	reason := "model benchmark rejected"
	if runErr != nil {
		reason = cleanLocalStatusError(runErr)
	}
	payload := struct {
		Task        string `json:"task"`
		JobID       string `json:"job_id"`
		ProofStatus string `json:"proof_status"`
		Error       string `json:"error"`
	}{
		Task:        ModelBenchmarkTask,
		JobID:       jobID,
		ProofStatus: string(ModelBenchmarkProofStatusFailed),
		Error:       reason,
	}
	encoded, _ := json.Marshal(payload)
	sum := sha256.Sum256(encoded)
	return ModelBenchmarkReceipt{
		JobID:         jobID,
		ResultHashHex: hex.EncodeToString(sum[:]),
		MeteringUnits: 0,
		Metadata: map[string]any{
			ModelBenchmarkTask: map[string]any{
				"proof_status": string(ModelBenchmarkProofStatusFailed),
				"error":        reason,
			},
		},
	}
}

func HashModelBenchmarkReceiptMetadata(jobID string, metadata ModelBenchmarkReceiptMetadata) (string, error) {
	jobID = strings.TrimSpace(jobID)
	if jobID == "" {
		return "", fmt.Errorf("%w: job_id required for result hash", ErrInvalidModelBenchmarkResult)
	}
	if err := validateModelBenchmarkReceiptMetadata(metadata); err != nil {
		return "", err
	}

	envelope := struct {
		Task     string                        `json:"task"`
		JobID    string                        `json:"job_id"`
		Metadata ModelBenchmarkReceiptMetadata `json:"v7_model_benchmark"`
	}{
		Task:     ModelBenchmarkTask,
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

func (m ModelBenchmarkReceiptMetadata) Map() map[string]any {
	out := map[string]any{
		"request_id":   m.RequestID,
		"model_id":     m.ModelID,
		"prompt_hash":  m.PromptHash,
		"runtime":      m.Runtime.Map(),
		"metrics":      m.Metrics.Map(),
		"output_bytes": m.OutputBytes,
		"proof_status": m.ProofStatus,
	}
	if strings.TrimSpace(m.OutputHash) != "" {
		out["output_hash"] = m.OutputHash
	}
	return out
}

func (m ModelBenchmarkReceiptRuntime) Map() map[string]any {
	return map[string]any{
		"agent_version":              m.AgentVersion,
		"os":                         m.OS,
		"arch":                       m.Arch,
		"native_inference_supported": m.NativeInferenceSupported,
		"native_inference_ready":     m.NativeInferenceReady,
		"runtime_kind":               m.RuntimeKind,
		"model_loaded":               m.ModelLoaded,
	}
}

func (m ModelBenchmarkReceiptMetrics) Map() map[string]any {
	out := map[string]any{
		"wall_time_ms":           m.WallTimeMs,
		"time_to_first_token_ms": m.TimeToFirstTokenMs,
		"tokens_generated":       m.TokensGenerated,
		"tokens_per_second":      m.TokensPerSecond,
		"model_load_state":       m.ModelLoadState,
	}
	if strings.TrimSpace(m.ErrorCode) != "" {
		out["error_code"] = m.ErrorCode
	}
	return out
}

func (m ModelBenchmarkReceiptMetadata) clone() ModelBenchmarkReceiptMetadata {
	return m
}

func modelBenchmarkReceiptMetadataFromResult(result ModelBenchmarkResult) ModelBenchmarkReceiptMetadata {
	metadata := ModelBenchmarkReceiptMetadata{
		RequestID:  result.RequestID,
		ModelID:    result.ModelID,
		PromptHash: result.PromptHash,
		Runtime: ModelBenchmarkReceiptRuntime{
			AgentVersion:             result.RuntimeInfo.AgentVersion,
			OS:                       result.RuntimeInfo.OS,
			Arch:                     result.RuntimeInfo.Arch,
			NativeInferenceSupported: result.RuntimeInfo.NativeInferenceSupported,
			NativeInferenceReady:     result.RuntimeInfo.NativeInferenceReady,
			RuntimeKind:              "native",
			ModelLoaded:              result.RuntimeInfo.ModelLoaded,
		},
		Metrics: ModelBenchmarkReceiptMetrics{
			WallTimeMs:         result.Metrics.WallTimeMs,
			TimeToFirstTokenMs: result.Metrics.TimeToFirstTokenMs,
			TokensGenerated:    result.Metrics.TokensGenerated,
			TokensPerSecond:    result.Metrics.TokensPerSecond,
			ModelLoadState:     receiptModelLoadState(result.Metrics.ModelLoadState),
			ErrorCode:          result.Metrics.ErrorCode,
		},
		OutputBytes: result.OutputBytes,
		ProofStatus: string(result.ProofStatus),
	}
	if result.ProofStatus == ModelBenchmarkProofStatusMeasured {
		metadata.OutputHash = result.OutputHash
	}
	return metadata
}

func validateModelBenchmarkReceiptMetadata(metadata ModelBenchmarkReceiptMetadata) error {
	metadata = metadata.clone()
	var errs []error
	if strings.TrimSpace(metadata.RequestID) == "" {
		errs = append(errs, fmt.Errorf("%w: receipt request_id required", ErrInvalidModelBenchmarkResult))
	}
	if strings.TrimSpace(metadata.ModelID) == "" {
		errs = append(errs, fmt.Errorf("%w: receipt model_id required", ErrInvalidModelBenchmarkResult))
	}
	if err := validateHashID(metadata.PromptHash, "receipt prompt_hash", ErrInvalidModelBenchmarkResult); err != nil {
		errs = append(errs, err)
	}
	if strings.TrimSpace(metadata.Runtime.AgentVersion) == "" {
		errs = append(errs, fmt.Errorf("%w: receipt runtime.agent_version required", ErrInvalidModelBenchmarkResult))
	}
	if strings.TrimSpace(metadata.Runtime.OS) == "" {
		errs = append(errs, fmt.Errorf("%w: receipt runtime.os required", ErrInvalidModelBenchmarkResult))
	}
	if strings.TrimSpace(metadata.Runtime.Arch) == "" {
		errs = append(errs, fmt.Errorf("%w: receipt runtime.arch required", ErrInvalidModelBenchmarkResult))
	}
	if strings.TrimSpace(metadata.Runtime.RuntimeKind) != "native" {
		errs = append(errs, fmt.Errorf("%w: receipt runtime.runtime_kind must be native", ErrInvalidModelBenchmarkResult))
	}
	if strings.TrimSpace(metadata.Metrics.ModelLoadState) == "" {
		errs = append(errs, fmt.Errorf("%w: receipt metrics.model_load_state required", ErrInvalidModelBenchmarkResult))
	}
	if metadata.Metrics.WallTimeMs < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt wall_time_ms must be non-negative", ErrInvalidModelBenchmarkResult))
	}
	if metadata.Metrics.TimeToFirstTokenMs < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt time_to_first_token_ms must be non-negative", ErrInvalidModelBenchmarkResult))
	}
	if metadata.Metrics.TokensGenerated < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt tokens_generated must be non-negative", ErrInvalidModelBenchmarkResult))
	}
	if !isFinite(metadata.Metrics.TokensPerSecond) || metadata.Metrics.TokensPerSecond < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt tokens_per_second must be finite and non-negative", ErrInvalidModelBenchmarkResult))
	}
	if metadata.OutputBytes < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt output_bytes must be non-negative", ErrInvalidModelBenchmarkResult))
	}
	switch ModelBenchmarkProofStatus(strings.TrimSpace(metadata.ProofStatus)) {
	case ModelBenchmarkProofStatusMeasured:
		if err := validateHashID(metadata.OutputHash, "receipt output_hash", ErrInvalidModelBenchmarkResult); err != nil {
			errs = append(errs, err)
		}
		if metadata.OutputBytes <= 0 {
			errs = append(errs, fmt.Errorf("%w: measured receipt output_bytes must be positive", ErrInvalidModelBenchmarkResult))
		}
	case ModelBenchmarkProofStatusUnavailable, ModelBenchmarkProofStatusFailed:
		if strings.TrimSpace(metadata.OutputHash) != "" {
			errs = append(errs, fmt.Errorf("%w: output_hash only allowed for native_model_measured", ErrInvalidModelBenchmarkResult))
		}
	default:
		errs = append(errs, fmt.Errorf("%w: receipt proof_status unknown %q", ErrInvalidModelBenchmarkResult, metadata.ProofStatus))
	}
	return errors.Join(errs...)
}

func receiptModelLoadState(state ModelBenchmarkModelLoadState) string {
	switch state {
	case ModelBenchmarkModelLoadStateLoaded:
		return "ready"
	default:
		return strings.TrimSpace(string(state))
	}
}
