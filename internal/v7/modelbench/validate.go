package modelbench

import (
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/Ryvion/node-agent/internal/v7/localcas"
)

var (
	ErrInvalidModelBenchmarkSpec   = errors.New("modelbench: invalid model benchmark spec")
	ErrInvalidModelBenchmarkResult = errors.New("modelbench: invalid model benchmark result")
)

func ValidateModelBenchmarkSpec(spec ModelBenchmarkSpec) error {
	spec = normalizeModelBenchmarkSpec(spec)

	var errs []error
	if spec.Task == "" {
		errs = append(errs, fmt.Errorf("%w: task required", ErrInvalidModelBenchmarkSpec))
	} else if spec.Task != ModelBenchmarkTask {
		errs = append(errs, fmt.Errorf("%w: task must be %q", ErrInvalidModelBenchmarkSpec, ModelBenchmarkTask))
	}
	if spec.RequestID == "" {
		errs = append(errs, fmt.Errorf("%w: request_id required", ErrInvalidModelBenchmarkSpec))
	}
	if spec.JobID == "" {
		errs = append(errs, fmt.Errorf("%w: job_id required", ErrInvalidModelBenchmarkSpec))
	}
	if spec.ModelID == "" {
		errs = append(errs, fmt.Errorf("%w: model_id required", ErrInvalidModelBenchmarkSpec))
	}
	if err := validateHashID(spec.PromptHash, "prompt_hash", ErrInvalidModelBenchmarkSpec); err != nil {
		errs = append(errs, err)
	}
	if spec.MaxTokens <= 0 {
		errs = append(errs, fmt.Errorf("%w: max_tokens must be greater than zero", ErrInvalidModelBenchmarkSpec))
	} else if spec.MaxTokens > MaxModelBenchmarkTokens {
		errs = append(errs, fmt.Errorf("%w: max_tokens exceeds maximum %d", ErrInvalidModelBenchmarkSpec, MaxModelBenchmarkTokens))
	}
	if !isFinite(spec.Temperature) {
		errs = append(errs, fmt.Errorf("%w: temperature must be finite", ErrInvalidModelBenchmarkSpec))
	} else if spec.Temperature < 0 {
		errs = append(errs, fmt.Errorf("%w: temperature must be non-negative", ErrInvalidModelBenchmarkSpec))
	}
	if spec.TimeoutMs <= 0 {
		errs = append(errs, fmt.Errorf("%w: timeout_ms must be greater than zero", ErrInvalidModelBenchmarkSpec))
	} else if spec.TimeoutMs > MaxModelBenchmarkTimeoutMs {
		errs = append(errs, fmt.Errorf("%w: timeout_ms exceeds maximum %d", ErrInvalidModelBenchmarkSpec, MaxModelBenchmarkTimeoutMs))
	}
	if spec.CreatedAtUnixMs <= 0 {
		errs = append(errs, fmt.Errorf("%w: created_at_unix_ms must be greater than zero", ErrInvalidModelBenchmarkSpec))
	}
	return errors.Join(errs...)
}

func ValidateModelBenchmarkResult(result ModelBenchmarkResult) error {
	result = normalizeModelBenchmarkResult(result)

	var errs []error
	if result.RequestID == "" {
		errs = append(errs, fmt.Errorf("%w: request_id required", ErrInvalidModelBenchmarkResult))
	}
	if result.JobID == "" {
		errs = append(errs, fmt.Errorf("%w: job_id required", ErrInvalidModelBenchmarkResult))
	}
	if result.ModelID == "" {
		errs = append(errs, fmt.Errorf("%w: model_id required", ErrInvalidModelBenchmarkResult))
	}
	if err := validateHashID(result.PromptHash, "prompt_hash", ErrInvalidModelBenchmarkResult); err != nil {
		errs = append(errs, err)
	}
	if err := validateModelBenchmarkProofStatus(result.ProofStatus); err != nil {
		errs = append(errs, err)
	}
	if err := validateModelBenchmarkRuntimeInfo(result.RuntimeInfo, result.ModelID, result.ProofStatus); err != nil {
		errs = append(errs, err)
	}
	if err := validateModelBenchmarkMetrics(result.Metrics); err != nil {
		errs = append(errs, err)
	}
	if result.OutputBytes < 0 {
		errs = append(errs, fmt.Errorf("%w: output_bytes must be non-negative", ErrInvalidModelBenchmarkResult))
	}
	if result.OutputHash == "" {
		if result.ProofStatus == ModelBenchmarkProofStatusMeasured {
			errs = append(errs, fmt.Errorf("%w: output_hash required for native_model_measured", ErrInvalidModelBenchmarkResult))
		}
	} else if err := validateHashID(result.OutputHash, "output_hash", ErrInvalidModelBenchmarkResult); err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

func validateModelBenchmarkRuntimeInfo(info ModelBenchmarkRuntimeInfo, resultModelID string, status ModelBenchmarkProofStatus) error {
	var errs []error
	if info.AgentVersion == "" {
		errs = append(errs, fmt.Errorf("%w: runtime_info.agent_version required", ErrInvalidModelBenchmarkResult))
	}
	if info.OS == "" {
		errs = append(errs, fmt.Errorf("%w: runtime_info.os required", ErrInvalidModelBenchmarkResult))
	}
	if info.Arch == "" {
		errs = append(errs, fmt.Errorf("%w: runtime_info.arch required", ErrInvalidModelBenchmarkResult))
	}
	if info.RuntimeKind == "" {
		errs = append(errs, fmt.Errorf("%w: runtime_info.runtime_kind required", ErrInvalidModelBenchmarkResult))
	} else if disallowedRuntimeKind(info.RuntimeKind) {
		errs = append(errs, fmt.Errorf("%w: runtime_info.runtime_kind must be native/local, not container-backed", ErrInvalidModelBenchmarkResult))
	}
	if info.ModelID == "" {
		errs = append(errs, fmt.Errorf("%w: runtime_info.model_id required", ErrInvalidModelBenchmarkResult))
	} else if resultModelID != "" && info.ModelID != resultModelID {
		errs = append(errs, fmt.Errorf("%w: runtime_info.model_id must match result model_id", ErrInvalidModelBenchmarkResult))
	}
	if !info.GPUDetected && info.GPUModel != "" {
		errs = append(errs, fmt.Errorf("%w: runtime_info.gpu_model requires gpu_detected", ErrInvalidModelBenchmarkResult))
	}
	if status == ModelBenchmarkProofStatusMeasured {
		if !info.NativeInferenceSupported {
			errs = append(errs, fmt.Errorf("%w: measured result requires native inference support", ErrInvalidModelBenchmarkResult))
		}
		if !info.NativeInferenceReady {
			errs = append(errs, fmt.Errorf("%w: measured result requires native inference ready", ErrInvalidModelBenchmarkResult))
		}
		if !info.ModelLoaded {
			errs = append(errs, fmt.Errorf("%w: measured result requires loaded model", ErrInvalidModelBenchmarkResult))
		}
	}
	return errors.Join(errs...)
}

func validateModelBenchmarkMetrics(metrics ModelBenchmarkMetrics) error {
	var errs []error
	if metrics.StartedAtUnixMs < 0 {
		errs = append(errs, fmt.Errorf("%w: metrics.started_at_unix_ms must be non-negative", ErrInvalidModelBenchmarkResult))
	}
	if metrics.FinishedAtUnixMs < 0 {
		errs = append(errs, fmt.Errorf("%w: metrics.finished_at_unix_ms must be non-negative", ErrInvalidModelBenchmarkResult))
	}
	if metrics.StartedAtUnixMs > 0 && metrics.FinishedAtUnixMs > 0 && metrics.FinishedAtUnixMs < metrics.StartedAtUnixMs {
		errs = append(errs, fmt.Errorf("%w: metrics.finished_at_unix_ms must be greater than or equal to started_at_unix_ms", ErrInvalidModelBenchmarkResult))
	}
	if metrics.WallTimeMs < 0 {
		errs = append(errs, fmt.Errorf("%w: metrics.wall_time_ms must be non-negative", ErrInvalidModelBenchmarkResult))
	}
	if metrics.TimeToFirstTokenMs < 0 {
		errs = append(errs, fmt.Errorf("%w: metrics.time_to_first_token_ms must be non-negative", ErrInvalidModelBenchmarkResult))
	}
	if metrics.TokensGenerated < 0 {
		errs = append(errs, fmt.Errorf("%w: metrics.tokens_generated must be non-negative", ErrInvalidModelBenchmarkResult))
	}
	if !isFinite(metrics.TokensPerSecond) {
		errs = append(errs, fmt.Errorf("%w: metrics.tokens_per_second must be finite", ErrInvalidModelBenchmarkResult))
	} else if metrics.TokensPerSecond < 0 {
		errs = append(errs, fmt.Errorf("%w: metrics.tokens_per_second must be non-negative", ErrInvalidModelBenchmarkResult))
	}
	if metrics.PromptTokens != nil && *metrics.PromptTokens < 0 {
		errs = append(errs, fmt.Errorf("%w: metrics.prompt_tokens must be non-negative", ErrInvalidModelBenchmarkResult))
	}
	if metrics.CompletionTokens != nil && *metrics.CompletionTokens < 0 {
		errs = append(errs, fmt.Errorf("%w: metrics.completion_tokens must be non-negative", ErrInvalidModelBenchmarkResult))
	}
	if metrics.ModelLoadState == "" {
		errs = append(errs, fmt.Errorf("%w: metrics.model_load_state required", ErrInvalidModelBenchmarkResult))
	} else if !knownModelLoadState(metrics.ModelLoadState) {
		errs = append(errs, fmt.Errorf("%w: metrics.model_load_state unknown %q", ErrInvalidModelBenchmarkResult, metrics.ModelLoadState))
	}
	return errors.Join(errs...)
}

func validateModelBenchmarkProofStatus(status ModelBenchmarkProofStatus) error {
	switch status {
	case ModelBenchmarkProofStatusMeasured,
		ModelBenchmarkProofStatusUnavailable,
		ModelBenchmarkProofStatusFailed:
		return nil
	default:
		return fmt.Errorf("%w: proof_status unknown %q", ErrInvalidModelBenchmarkResult, status)
	}
}

func knownModelLoadState(state ModelBenchmarkModelLoadState) bool {
	switch state {
	case ModelBenchmarkModelLoadStateUnknown,
		ModelBenchmarkModelLoadStateNotLoaded,
		ModelBenchmarkModelLoadStateLoading,
		ModelBenchmarkModelLoadStateLoaded,
		ModelBenchmarkModelLoadStateUnavailable,
		ModelBenchmarkModelLoadStateFailed:
		return true
	default:
		return false
	}
}

func validateHashID(value string, field string, base error) error {
	value = strings.TrimSpace(value)
	if err := localcas.ValidateObjectID(localcas.ObjectID(value)); err != nil {
		return fmt.Errorf("%w: %s must be sha256:<64 lowercase hex>", base, field)
	}
	return nil
}

func disallowedRuntimeKind(kind string) bool {
	kind = strings.ToLower(strings.TrimSpace(kind))
	return strings.Contains(kind, "docker") ||
		strings.Contains(kind, "oci") ||
		strings.Contains(kind, "container")
}

func isFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
