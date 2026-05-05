package modelbench

import (
	"context"
	"errors"
	"fmt"
	"strings"
)

const (
	MaxModelBenchmarkSeriesWarmupRuns   = 5
	MaxModelBenchmarkSeriesMeasuredRuns = 20
)

var ErrInvalidModelBenchmarkSeriesSpec = errors.New("modelbench: invalid model benchmark series spec")

type ModelBenchmarkSeriesSpec struct {
	RequestID       string `json:"request_id"`
	JobID           string `json:"job_id"`
	ModelID         string `json:"model_id"`
	PromptProfileID string `json:"prompt_profile_id,omitempty"`
	PromptHash      string `json:"prompt_hash"`
	MaxTokens       int    `json:"max_tokens"`
	TimeoutMs       int64  `json:"timeout_ms"`
	WarmupRuns      int    `json:"warmup_runs"`
	MeasuredRuns    int    `json:"measured_runs"`
	CreatedAtUnixMs int64  `json:"created_at_unix_ms"`
}

type ModelBenchmarkTrialResult struct {
	TrialIndex              int                       `json:"trial_index"`
	Warmup                  bool                      `json:"warmup"`
	ProofStatus             ModelBenchmarkProofStatus `json:"proof_status"`
	WallTimeMs              int64                     `json:"wall_time_ms"`
	TimeToFirstTokenMs      int64                     `json:"time_to_first_token_ms"`
	TokensGenerated         int64                     `json:"tokens_generated"`
	EndToEndTokensPerSecond float64                   `json:"end_to_end_tokens_per_second"`
	DecodeTokensPerSecond   float64                   `json:"decode_tokens_per_second"`
	OutputHash              string                    `json:"output_hash,omitempty"`
	OutputBytes             int64                     `json:"output_bytes"`
	ErrorMessage            string                    `json:"error_message,omitempty"`
}

type ModelBenchmarkSeriesResult struct {
	Spec            ModelBenchmarkSeriesSpec    `json:"spec"`
	RequestID       string                      `json:"request_id"`
	JobID           string                      `json:"job_id"`
	ModelID         string                      `json:"model_id"`
	PromptProfileID string                      `json:"prompt_profile_id,omitempty"`
	PromptHash      string                      `json:"prompt_hash"`
	MaxTokens       int                         `json:"max_tokens"`
	TimeoutMs       int64                       `json:"timeout_ms"`
	WarmupRuns      int                         `json:"warmup_runs"`
	MeasuredRuns    int                         `json:"measured_runs"`
	CreatedAtUnixMs int64                       `json:"created_at_unix_ms"`
	Trials          []ModelBenchmarkTrialResult `json:"trials"`
	Summary         ModelBenchmarkSeriesSummary `json:"summary"`
}

func RunModelBenchmarkSeries(ctx context.Context, runner ModelBenchmarkRunner, spec ModelBenchmarkSeriesSpec) (ModelBenchmarkSeriesResult, error) {
	spec = normalizeModelBenchmarkSeriesSpec(spec)
	result := newModelBenchmarkSeriesResult(spec)

	if err := ValidateModelBenchmarkSeriesSpec(spec); err != nil {
		return result, err
	}
	if runner == nil {
		return result, ModelBenchmarkError{Code: "modelbench_runner_missing", Message: "model benchmark runner is not configured"}
	}

	totalRuns := spec.WarmupRuns + spec.MeasuredRuns
	result.Trials = make([]ModelBenchmarkTrialResult, 0, totalRuns)
	trialSpec := modelBenchmarkSpecFromSeries(spec)
	for i := 0; i < totalRuns; i++ {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		runResult, runErr := runner.RunModelBenchmark(ctx, trialSpec)
		trial := modelBenchmarkTrialResultFromRun(i, i < spec.WarmupRuns, runResult, runErr)
		result.Trials = append(result.Trials, trial)
	}

	summary, err := ComputeBenchmarkSeriesSummary(result.Trials)
	result.Summary = summary
	if err != nil {
		return result, err
	}
	return result, nil
}

func ValidateModelBenchmarkSeriesSpec(spec ModelBenchmarkSeriesSpec) error {
	spec = normalizeModelBenchmarkSeriesSpec(spec)

	var errs []error
	if spec.WarmupRuns < 0 {
		errs = append(errs, fmt.Errorf("%w: warmup_runs must be non-negative", ErrInvalidModelBenchmarkSeriesSpec))
	} else if spec.WarmupRuns > MaxModelBenchmarkSeriesWarmupRuns {
		errs = append(errs, fmt.Errorf("%w: warmup_runs exceeds maximum %d", ErrInvalidModelBenchmarkSeriesSpec, MaxModelBenchmarkSeriesWarmupRuns))
	}
	if spec.MeasuredRuns <= 0 {
		errs = append(errs, fmt.Errorf("%w: measured_runs must be greater than zero", ErrInvalidModelBenchmarkSeriesSpec))
	} else if spec.MeasuredRuns > MaxModelBenchmarkSeriesMeasuredRuns {
		errs = append(errs, fmt.Errorf("%w: measured_runs exceeds maximum %d", ErrInvalidModelBenchmarkSeriesSpec, MaxModelBenchmarkSeriesMeasuredRuns))
	}
	if err := ValidateModelBenchmarkSpec(modelBenchmarkSpecFromSeries(spec)); err != nil {
		errs = append(errs, fmt.Errorf("%w: %v", ErrInvalidModelBenchmarkSeriesSpec, err))
	}
	return errors.Join(errs...)
}

func normalizeModelBenchmarkSeriesSpec(spec ModelBenchmarkSeriesSpec) ModelBenchmarkSeriesSpec {
	spec.RequestID = strings.TrimSpace(spec.RequestID)
	spec.JobID = strings.TrimSpace(spec.JobID)
	spec.ModelID = strings.TrimSpace(spec.ModelID)
	spec.PromptProfileID = strings.TrimSpace(spec.PromptProfileID)
	spec.PromptHash = strings.TrimSpace(spec.PromptHash)
	return spec
}

func newModelBenchmarkSeriesResult(spec ModelBenchmarkSeriesSpec) ModelBenchmarkSeriesResult {
	return ModelBenchmarkSeriesResult{
		Spec:            spec,
		RequestID:       spec.RequestID,
		JobID:           spec.JobID,
		ModelID:         spec.ModelID,
		PromptProfileID: spec.PromptProfileID,
		PromptHash:      spec.PromptHash,
		MaxTokens:       spec.MaxTokens,
		TimeoutMs:       spec.TimeoutMs,
		WarmupRuns:      spec.WarmupRuns,
		MeasuredRuns:    spec.MeasuredRuns,
		CreatedAtUnixMs: spec.CreatedAtUnixMs,
	}
}

func modelBenchmarkSpecFromSeries(spec ModelBenchmarkSeriesSpec) ModelBenchmarkSpec {
	return ModelBenchmarkSpec{
		Task:            ModelBenchmarkTask,
		RequestID:       spec.RequestID,
		JobID:           spec.JobID,
		ModelID:         spec.ModelID,
		PromptProfileID: spec.PromptProfileID,
		PromptHash:      spec.PromptHash,
		MaxTokens:       spec.MaxTokens,
		TimeoutMs:       spec.TimeoutMs,
		CreatedAtUnixMs: spec.CreatedAtUnixMs,
	}
}

func modelBenchmarkTrialResultFromRun(index int, warmup bool, result ModelBenchmarkResult, runErr error) ModelBenchmarkTrialResult {
	status := result.ProofStatus
	if status == "" {
		status = ModelBenchmarkProofStatusFailed
	}
	if runErr != nil && status == ModelBenchmarkProofStatusMeasured {
		status = ModelBenchmarkProofStatusFailed
	}

	trial := ModelBenchmarkTrialResult{
		TrialIndex:              index,
		Warmup:                  warmup,
		ProofStatus:             status,
		WallTimeMs:              result.Metrics.WallTimeMs,
		TimeToFirstTokenMs:      result.Metrics.TimeToFirstTokenMs,
		TokensGenerated:         result.Metrics.TokensGenerated,
		EndToEndTokensPerSecond: result.Metrics.TokensPerSecond,
		DecodeTokensPerSecond:   computeDecodeTokensPerSecond(result.Metrics.TokensGenerated, result.Metrics.WallTimeMs, result.Metrics.TimeToFirstTokenMs),
		OutputHash:              safeSeriesOutputHash(result.OutputHash),
		OutputBytes:             result.OutputBytes,
		ErrorMessage:            seriesTrialErrorMessage(status, result.Metrics.ErrorCode, runErr),
	}
	if trial.ProofStatus != ModelBenchmarkProofStatusMeasured {
		trial.OutputHash = ""
		trial.OutputBytes = 0
	}
	return trial
}

func computeDecodeTokensPerSecond(tokensGenerated int64, wallTimeMs int64, timeToFirstTokenMs int64) float64 {
	if tokensGenerated > 1 && wallTimeMs > timeToFirstTokenMs {
		return float64(tokensGenerated-1) / (float64(wallTimeMs-timeToFirstTokenMs) / 1000)
	}
	return 0
}

func seriesTrialErrorMessage(status ModelBenchmarkProofStatus, errorCode string, runErr error) string {
	errorCode = safeSeriesCode(errorCode)
	if errorCode != "" {
		return errorCode
	}
	if runErr == nil {
		return ""
	}
	var benchErr ModelBenchmarkError
	if errors.As(runErr, &benchErr) {
		code := safeSeriesCode(benchErr.Code)
		if code != "" {
			return code
		}
	}
	switch status {
	case ModelBenchmarkProofStatusUnavailable:
		return "model benchmark trial unavailable"
	case ModelBenchmarkProofStatusFailed:
		return "model benchmark trial failed"
	default:
		return "model benchmark trial error"
	}
}

func safeSeriesOutputHash(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	if err := validateHashID(value, "output_hash", ErrInvalidModelBenchmarkSeriesSpec); err != nil {
		return ""
	}
	return value
}

func safeSeriesCode(value string) string {
	value = strings.TrimSpace(value)
	if value == "" || len(value) > 128 {
		return ""
	}
	for _, r := range value {
		switch {
		case r >= 'a' && r <= 'z':
		case r >= 'A' && r <= 'Z':
		case r >= '0' && r <= '9':
		case r == '_' || r == '-' || r == '.' || r == ':':
		default:
			return ""
		}
	}
	return value
}
