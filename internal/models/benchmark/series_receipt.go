package modelbench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

const (
	ModelBenchmarkSeriesProofStatusMeasured    = "native_model_series_measured"
	ModelBenchmarkSeriesProofStatusUnavailable = "native_model_series_unavailable"
	ModelBenchmarkSeriesProofStatusFailed      = "native_model_series_failed"
)

var ErrInvalidModelBenchmarkSeriesReceipt = errors.New("modelbench: invalid model benchmark series receipt")

type ModelBenchmarkSeriesReceiptMetadata struct {
	RequestID       string                             `json:"request_id"`
	JobID           string                             `json:"job_id"`
	ModelID         string                             `json:"model_id"`
	PromptProfileID string                             `json:"prompt_profile_id"`
	PromptHash      string                             `json:"prompt_hash"`
	WarmupRuns      int                                `json:"warmup_runs"`
	MeasuredRuns    int                                `json:"measured_runs"`
	Runtime         ModelBenchmarkResultRuntime        `json:"runtime"`
	Trials          []ModelBenchmarkSeriesReceiptTrial `json:"trials"`
	Summary         ModelBenchmarkSeriesReceiptSummary `json:"summary"`
	ProofStatus     string                             `json:"proof_status"`
}

type ModelBenchmarkSeriesReceiptTrial struct {
	TrialIndex              int     `json:"trial_index"`
	Warmup                  bool    `json:"warmup"`
	ProofStatus             string  `json:"proof_status"`
	WallTimeMs              int64   `json:"wall_time_ms"`
	TimeToFirstTokenMs      int64   `json:"time_to_first_token_ms"`
	TokensGenerated         int64   `json:"tokens_generated"`
	EndToEndTokensPerSecond float64 `json:"end_to_end_tokens_per_second"`
	DecodeTokensPerSecond   float64 `json:"decode_tokens_per_second"`
	OutputHash              string  `json:"output_hash"`
	OutputBytes             int64   `json:"output_bytes"`
}

type ModelBenchmarkSeriesReceiptSummary struct {
	P50TTFTMs              int64   `json:"p50_ttft_ms"`
	P95TTFTMs              int64   `json:"p95_ttft_ms"`
	P50DecodeTPS           float64 `json:"p50_decode_tps"`
	P95DecodeTPS           float64 `json:"p95_decode_tps"`
	P50EndToEndTPS         float64 `json:"p50_end_to_end_tps"`
	SuccessfulMeasuredRuns int     `json:"successful_measured_runs"`
}

func BuildModelBenchmarkSeriesReceipt(result ModelBenchmarkSeriesResult) (ModelBenchmarkReceipt, error) {
	metadata := modelBenchmarkSeriesReceiptMetadataFromResult(result)
	if err := validateModelBenchmarkSeriesReceiptMetadata(metadata); err != nil {
		return ModelBenchmarkReceipt{}, err
	}

	hashHex, err := HashModelBenchmarkSeriesReceiptMetadata(result.JobID, metadata)
	if err != nil {
		return ModelBenchmarkReceipt{}, err
	}
	return ModelBenchmarkReceipt{
		JobID:         result.JobID,
		ResultHashHex: hashHex,
		MeteringUnits: 1,
		Metadata: map[string]any{
			ModelBenchmarkSeriesTask: metadata.Map(),
		},
	}, nil
}

func BuildModelBenchmarkSeriesRejectionReceipt(jobID string, runErr error) ModelBenchmarkReceipt {
	jobID = strings.TrimSpace(jobID)
	if jobID == "" {
		jobID = "v7-model-benchmark-series-rejected"
	}
	reason := "model benchmark series rejected"
	if runErr != nil {
		reason = cleanLocalStatusError(runErr)
	}
	payload := struct {
		Task        string `json:"task"`
		JobID       string `json:"job_id"`
		ProofStatus string `json:"proof_status"`
		Error       string `json:"error"`
	}{
		Task:        ModelBenchmarkSeriesTask,
		JobID:       jobID,
		ProofStatus: ModelBenchmarkSeriesProofStatusFailed,
		Error:       reason,
	}
	encoded, _ := json.Marshal(payload)
	sum := sha256.Sum256(encoded)
	return ModelBenchmarkReceipt{
		JobID:         jobID,
		ResultHashHex: hex.EncodeToString(sum[:]),
		MeteringUnits: 0,
		Metadata: map[string]any{
			ModelBenchmarkSeriesTask: map[string]any{
				"proof_status": ModelBenchmarkSeriesProofStatusFailed,
				"error":        reason,
			},
		},
	}
}

func HashModelBenchmarkSeriesReceiptMetadata(jobID string, metadata ModelBenchmarkSeriesReceiptMetadata) (string, error) {
	jobID = strings.TrimSpace(jobID)
	if jobID == "" {
		return "", fmt.Errorf("%w: job_id required for result hash", ErrInvalidModelBenchmarkSeriesReceipt)
	}
	if err := validateModelBenchmarkSeriesReceiptMetadata(metadata); err != nil {
		return "", err
	}

	envelope := struct {
		Task     string                              `json:"task"`
		JobID    string                              `json:"job_id"`
		Metadata ModelBenchmarkSeriesReceiptMetadata `json:"v7_model_benchmark_series"`
	}{
		Task:     ModelBenchmarkSeriesTask,
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

func (m ModelBenchmarkSeriesReceiptMetadata) Map() map[string]any {
	trials := make([]map[string]any, 0, len(m.Trials))
	for _, trial := range m.Trials {
		trials = append(trials, trial.Map())
	}
	return map[string]any{
		"request_id":        m.RequestID,
		"job_id":            m.JobID,
		"model_id":          m.ModelID,
		"prompt_profile_id": m.PromptProfileID,
		"prompt_hash":       m.PromptHash,
		"warmup_runs":       m.WarmupRuns,
		"measured_runs":     m.MeasuredRuns,
		"runtime":           m.Runtime.Map(),
		"trials":            trials,
		"summary":           m.Summary.Map(),
		"proof_status":      m.ProofStatus,
	}
}

func (m ModelBenchmarkSeriesReceiptTrial) Map() map[string]any {
	return map[string]any{
		"trial_index":                  m.TrialIndex,
		"warmup":                       m.Warmup,
		"proof_status":                 m.ProofStatus,
		"wall_time_ms":                 m.WallTimeMs,
		"time_to_first_token_ms":       m.TimeToFirstTokenMs,
		"tokens_generated":             m.TokensGenerated,
		"end_to_end_tokens_per_second": m.EndToEndTokensPerSecond,
		"decode_tokens_per_second":     m.DecodeTokensPerSecond,
		"output_hash":                  m.OutputHash,
		"output_bytes":                 m.OutputBytes,
	}
}

func (m ModelBenchmarkSeriesReceiptSummary) Map() map[string]any {
	return map[string]any{
		"p50_ttft_ms":              m.P50TTFTMs,
		"p95_ttft_ms":              m.P95TTFTMs,
		"p50_decode_tps":           m.P50DecodeTPS,
		"p95_decode_tps":           m.P95DecodeTPS,
		"p50_end_to_end_tps":       m.P50EndToEndTPS,
		"successful_measured_runs": m.SuccessfulMeasuredRuns,
	}
}

func (m ModelBenchmarkSeriesReceiptMetadata) clone() ModelBenchmarkSeriesReceiptMetadata {
	m.Trials = append([]ModelBenchmarkSeriesReceiptTrial(nil), m.Trials...)
	return m
}

func modelBenchmarkSeriesReceiptMetadataFromResult(result ModelBenchmarkSeriesResult) ModelBenchmarkSeriesReceiptMetadata {
	trials := make([]ModelBenchmarkSeriesReceiptTrial, 0, len(result.Trials))
	for _, trial := range result.Trials {
		trials = append(trials, ModelBenchmarkSeriesReceiptTrial{
			TrialIndex:              trial.TrialIndex,
			Warmup:                  trial.Warmup,
			ProofStatus:             string(trial.ProofStatus),
			WallTimeMs:              trial.WallTimeMs,
			TimeToFirstTokenMs:      trial.TimeToFirstTokenMs,
			TokensGenerated:         trial.TokensGenerated,
			EndToEndTokensPerSecond: trial.EndToEndTokensPerSecond,
			DecodeTokensPerSecond:   trial.DecodeTokensPerSecond,
			OutputHash:              strings.TrimSpace(trial.OutputHash),
			OutputBytes:             trial.OutputBytes,
		})
	}

	return ModelBenchmarkSeriesReceiptMetadata{
		RequestID:       result.RequestID,
		JobID:           result.JobID,
		ModelID:         result.ModelID,
		PromptProfileID: result.PromptProfileID,
		PromptHash:      result.PromptHash,
		WarmupRuns:      result.WarmupRuns,
		MeasuredRuns:    result.MeasuredRuns,
		Runtime:         result.Runtime,
		Trials:          trials,
		Summary: ModelBenchmarkSeriesReceiptSummary{
			P50TTFTMs:              result.Summary.P50TTFTMs,
			P95TTFTMs:              result.Summary.P95TTFTMs,
			P50DecodeTPS:           result.Summary.P50DecodeTPS,
			P95DecodeTPS:           result.Summary.P95DecodeTPS,
			P50EndToEndTPS:         result.Summary.P50EndToEndTPS,
			SuccessfulMeasuredRuns: result.Summary.SuccessfulMeasuredRuns,
		},
		ProofStatus: seriesReceiptProofStatus(result.Summary.ProofStatus),
	}
}

func validateModelBenchmarkSeriesReceiptMetadata(metadata ModelBenchmarkSeriesReceiptMetadata) error {
	metadata = metadata.clone()
	var errs []error
	if strings.TrimSpace(metadata.RequestID) == "" {
		errs = append(errs, fmt.Errorf("%w: receipt request_id required", ErrInvalidModelBenchmarkSeriesReceipt))
	}
	if strings.TrimSpace(metadata.JobID) == "" {
		errs = append(errs, fmt.Errorf("%w: receipt job_id required", ErrInvalidModelBenchmarkSeriesReceipt))
	}
	if strings.TrimSpace(metadata.ModelID) == "" {
		errs = append(errs, fmt.Errorf("%w: receipt model_id required", ErrInvalidModelBenchmarkSeriesReceipt))
	}
	if err := validateHashID(metadata.PromptHash, "receipt prompt_hash", ErrInvalidModelBenchmarkSeriesReceipt); err != nil {
		errs = append(errs, err)
	}
	if metadata.WarmupRuns < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt warmup_runs must be non-negative", ErrInvalidModelBenchmarkSeriesReceipt))
	}
	if metadata.MeasuredRuns <= 0 {
		errs = append(errs, fmt.Errorf("%w: receipt measured_runs must be greater than zero", ErrInvalidModelBenchmarkSeriesReceipt))
	}
	if len(metadata.Trials) != metadata.WarmupRuns+metadata.MeasuredRuns {
		errs = append(errs, fmt.Errorf("%w: receipt trials length must match warmup_runs + measured_runs", ErrInvalidModelBenchmarkSeriesReceipt))
	}
	if strings.TrimSpace(metadata.ProofStatus) == "" {
		errs = append(errs, fmt.Errorf("%w: receipt proof_status required", ErrInvalidModelBenchmarkSeriesReceipt))
	} else if !knownSeriesReceiptProofStatus(metadata.ProofStatus) {
		errs = append(errs, fmt.Errorf("%w: receipt proof_status unknown %q", ErrInvalidModelBenchmarkSeriesReceipt, metadata.ProofStatus))
	}
	if metadata.Summary.SuccessfulMeasuredRuns < 0 || metadata.Summary.SuccessfulMeasuredRuns > metadata.MeasuredRuns {
		errs = append(errs, fmt.Errorf("%w: receipt successful_measured_runs out of range", ErrInvalidModelBenchmarkSeriesReceipt))
	}
	if metadata.Summary.P50TTFTMs < 0 || metadata.Summary.P95TTFTMs < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt ttft percentiles must be non-negative", ErrInvalidModelBenchmarkSeriesReceipt))
	}
	if !isFinite(metadata.Summary.P50DecodeTPS) || metadata.Summary.P50DecodeTPS < 0 ||
		!isFinite(metadata.Summary.P95DecodeTPS) || metadata.Summary.P95DecodeTPS < 0 ||
		!isFinite(metadata.Summary.P50EndToEndTPS) || metadata.Summary.P50EndToEndTPS < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt tps percentiles must be finite and non-negative", ErrInvalidModelBenchmarkSeriesReceipt))
	}
	for i, trial := range metadata.Trials {
		if err := validateModelBenchmarkSeriesReceiptTrial(i, trial); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func validateModelBenchmarkSeriesReceiptTrial(index int, trial ModelBenchmarkSeriesReceiptTrial) error {
	var errs []error
	if trial.TrialIndex != index {
		errs = append(errs, fmt.Errorf("%w: receipt trial_index must be contiguous", ErrInvalidModelBenchmarkSeriesReceipt))
	}
	switch ModelBenchmarkProofStatus(strings.TrimSpace(trial.ProofStatus)) {
	case ModelBenchmarkProofStatusMeasured:
		if err := validateHashID(trial.OutputHash, "receipt trial output_hash", ErrInvalidModelBenchmarkSeriesReceipt); err != nil {
			errs = append(errs, err)
		}
		if trial.OutputBytes <= 0 {
			errs = append(errs, fmt.Errorf("%w: receipt measured trial output_bytes must be positive", ErrInvalidModelBenchmarkSeriesReceipt))
		}
	case ModelBenchmarkProofStatusUnavailable, ModelBenchmarkProofStatusFailed:
		if strings.TrimSpace(trial.OutputHash) != "" {
			errs = append(errs, fmt.Errorf("%w: receipt trial output_hash only allowed for measured trials", ErrInvalidModelBenchmarkSeriesReceipt))
		}
		if trial.OutputBytes != 0 {
			errs = append(errs, fmt.Errorf("%w: receipt non-measured trial output_bytes must be zero", ErrInvalidModelBenchmarkSeriesReceipt))
		}
	default:
		errs = append(errs, fmt.Errorf("%w: receipt trial proof_status unknown %q", ErrInvalidModelBenchmarkSeriesReceipt, trial.ProofStatus))
	}
	if trial.WallTimeMs < 0 || trial.TimeToFirstTokenMs < 0 || trial.TokensGenerated < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt trial timing and token counts must be non-negative", ErrInvalidModelBenchmarkSeriesReceipt))
	}
	if !isFinite(trial.EndToEndTokensPerSecond) || trial.EndToEndTokensPerSecond < 0 ||
		!isFinite(trial.DecodeTokensPerSecond) || trial.DecodeTokensPerSecond < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt trial tps must be finite and non-negative", ErrInvalidModelBenchmarkSeriesReceipt))
	}
	return errors.Join(errs...)
}

func seriesReceiptProofStatus(status ModelBenchmarkProofStatus) string {
	switch status {
	case ModelBenchmarkProofStatusMeasured:
		return ModelBenchmarkSeriesProofStatusMeasured
	case ModelBenchmarkProofStatusUnavailable:
		return ModelBenchmarkSeriesProofStatusUnavailable
	default:
		return ModelBenchmarkSeriesProofStatusFailed
	}
}

func knownSeriesReceiptProofStatus(status string) bool {
	switch strings.TrimSpace(status) {
	case ModelBenchmarkSeriesProofStatusMeasured,
		ModelBenchmarkSeriesProofStatusUnavailable,
		ModelBenchmarkSeriesProofStatusFailed:
		return true
	default:
		return false
	}
}
