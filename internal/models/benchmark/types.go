package modelbench

import (
	"context"
	"fmt"
	"strings"
)

const (
	ModelBenchmarkTask = "v7_model_benchmark"

	MaxModelBenchmarkTokens    = 512
	MaxModelBenchmarkTimeoutMs = 300_000

	ModelBenchmarkRuntimeKindNativeLocal = "native_local"
)

type ModelBenchmarkProofStatus string

const (
	ModelBenchmarkProofStatusMeasured    ModelBenchmarkProofStatus = "native_model_measured"
	ModelBenchmarkProofStatusUnavailable ModelBenchmarkProofStatus = "native_model_unavailable"
	ModelBenchmarkProofStatusFailed      ModelBenchmarkProofStatus = "native_model_failed"
)

type ModelBenchmarkModelLoadState string

const (
	ModelBenchmarkModelLoadStateUnknown     ModelBenchmarkModelLoadState = "unknown"
	ModelBenchmarkModelLoadStateNotLoaded   ModelBenchmarkModelLoadState = "not_loaded"
	ModelBenchmarkModelLoadStateLoading     ModelBenchmarkModelLoadState = "loading"
	ModelBenchmarkModelLoadStateLoaded      ModelBenchmarkModelLoadState = "loaded"
	ModelBenchmarkModelLoadStateUnavailable ModelBenchmarkModelLoadState = "unavailable"
	ModelBenchmarkModelLoadStateFailed      ModelBenchmarkModelLoadState = "failed"
)

type ModelBenchmarkRunner interface {
	RunModelBenchmark(context.Context, ModelBenchmarkSpec) (ModelBenchmarkResult, error)
}

type ModelBenchmarkPrompt struct {
	Label   string `json:"label"`
	Content []byte `json:"-"`
}

type ModelBenchmarkRuntimeInfo struct {
	AgentVersion             string `json:"agent_version"`
	OS                       string `json:"os"`
	Arch                     string `json:"arch"`
	NativeInferenceSupported bool   `json:"native_inference_supported"`
	NativeInferenceReady     bool   `json:"native_inference_ready"`
	RuntimeKind              string `json:"runtime_kind"`
	ModelID                  string `json:"model_id"`
	ModelLoaded              bool   `json:"model_loaded"`
	GPUDetected              bool   `json:"gpu_detected"`
	GPUModel                 string `json:"gpu_model,omitempty"`
}

type ModelBenchmarkMetrics struct {
	StartedAtUnixMs    int64                        `json:"started_at_unix_ms"`
	FinishedAtUnixMs   int64                        `json:"finished_at_unix_ms"`
	WallTimeMs         int64                        `json:"wall_time_ms"`
	TimeToFirstTokenMs int64                        `json:"time_to_first_token_ms"`
	TokensGenerated    int64                        `json:"tokens_generated"`
	TokensPerSecond    float64                      `json:"tokens_per_second"`
	PromptTokens       *int64                       `json:"prompt_tokens,omitempty"`
	CompletionTokens   *int64                       `json:"completion_tokens,omitempty"`
	ModelLoadState     ModelBenchmarkModelLoadState `json:"model_load_state"`
	ErrorCode          string                       `json:"error_code,omitempty"`
}

type ModelBenchmarkError struct {
	Code      string `json:"code"`
	Message   string `json:"message,omitempty"`
	Retryable bool   `json:"retryable,omitempty"`
}

func (e ModelBenchmarkError) Error() string {
	code := strings.TrimSpace(e.Code)
	message := strings.TrimSpace(e.Message)
	if code == "" {
		return message
	}
	if message == "" {
		return code
	}
	return fmt.Sprintf("%s: %s", code, message)
}
