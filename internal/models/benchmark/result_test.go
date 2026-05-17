package modelbench

import (
	"errors"
	"math"
	"strings"
	"testing"
)

func TestValidateModelBenchmarkResultValidMeasuredPasses(t *testing.T) {
	if err := ValidateModelBenchmarkResult(validMeasuredModelBenchmarkResult()); err != nil {
		t.Fatalf("ValidateModelBenchmarkResult() error = %v", err)
	}
}

func TestValidateModelBenchmarkResultMeasuredMissingOutputHashRejected(t *testing.T) {
	result := validMeasuredModelBenchmarkResult()
	result.OutputHash = ""

	err := ValidateModelBenchmarkResult(result)
	if !errors.Is(err, ErrInvalidModelBenchmarkResult) || !strings.Contains(err.Error(), "output_hash") {
		t.Fatalf("ValidateModelBenchmarkResult() error = %v, want output_hash required", err)
	}
}

func TestValidateModelBenchmarkResultUnavailableCanOmitOutputHash(t *testing.T) {
	result := validMeasuredModelBenchmarkResult()
	result.ProofStatus = ModelBenchmarkProofStatusUnavailable
	result.OutputHash = ""
	result.OutputBytes = 0
	result.RuntimeInfo.NativeInferenceReady = false
	result.RuntimeInfo.ModelLoaded = false
	result.Metrics.TokensGenerated = 0
	result.Metrics.TokensPerSecond = 0
	result.Metrics.TimeToFirstTokenMs = 0
	result.Metrics.ModelLoadState = ModelBenchmarkModelLoadStateUnavailable
	result.Metrics.ErrorCode = "native_inference_unavailable"

	if err := ValidateModelBenchmarkResult(result); err != nil {
		t.Fatalf("ValidateModelBenchmarkResult() error = %v", err)
	}
}

func TestValidateModelBenchmarkResultRejectsNonFiniteTokensPerSecond(t *testing.T) {
	for _, value := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		result := validMeasuredModelBenchmarkResult()
		result.Metrics.TokensPerSecond = value

		err := ValidateModelBenchmarkResult(result)
		if !errors.Is(err, ErrInvalidModelBenchmarkResult) || !strings.Contains(err.Error(), "tokens_per_second") {
			t.Fatalf("ValidateModelBenchmarkResult(%v) error = %v, want finite tokens_per_second", value, err)
		}
	}
}

func TestValidateModelBenchmarkResultRejectsNegativeCounters(t *testing.T) {
	tests := []struct {
		name      string
		mutate    func(*ModelBenchmarkResult)
		wantField string
	}{
		{
			name:      "wall time",
			mutate:    func(result *ModelBenchmarkResult) { result.Metrics.WallTimeMs = -1 },
			wantField: "wall_time_ms",
		},
		{
			name:      "tokens generated",
			mutate:    func(result *ModelBenchmarkResult) { result.Metrics.TokensGenerated = -1 },
			wantField: "tokens_generated",
		},
		{
			name:      "tokens per second",
			mutate:    func(result *ModelBenchmarkResult) { result.Metrics.TokensPerSecond = -0.01 },
			wantField: "tokens_per_second",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := validMeasuredModelBenchmarkResult()
			test.mutate(&result)

			err := ValidateModelBenchmarkResult(result)
			if !errors.Is(err, ErrInvalidModelBenchmarkResult) || !strings.Contains(err.Error(), test.wantField) {
				t.Fatalf("ValidateModelBenchmarkResult() error = %v, want %s invalid", err, test.wantField)
			}
		})
	}
}

func TestValidateModelBenchmarkResultRejectsContradictoryGPUModel(t *testing.T) {
	result := validMeasuredModelBenchmarkResult()
	result.RuntimeInfo.GPUDetected = false
	result.RuntimeInfo.GPUModel = "pretend-gpu"

	err := ValidateModelBenchmarkResult(result)
	if !errors.Is(err, ErrInvalidModelBenchmarkResult) || !strings.Contains(err.Error(), "gpu_model") {
		t.Fatalf("ValidateModelBenchmarkResult() error = %v, want gpu_model invalid", err)
	}
}

func validMeasuredModelBenchmarkResult() ModelBenchmarkResult {
	promptTokens := int64(8)
	completionTokens := int64(16)
	return ModelBenchmarkResult{
		RequestID:  "request-modelbench-1",
		JobID:      "job-modelbench-1",
		NodeID:     "node-local-1",
		ModelID:    "llama-local-7b-q4",
		PromptHash: modelBenchHash("Return one short readiness token."),
		RuntimeInfo: ModelBenchmarkRuntimeInfo{
			AgentVersion:             "dev",
			OS:                       "darwin",
			Arch:                     "arm64",
			NativeInferenceSupported: true,
			NativeInferenceReady:     true,
			RuntimeKind:              ModelBenchmarkRuntimeKindNativeLocal,
			ModelID:                  "llama-local-7b-q4",
			ModelLoaded:              true,
			GPUDetected:              true,
			GPUModel:                 "Apple M-series",
		},
		Metrics: ModelBenchmarkMetrics{
			StartedAtUnixMs:    1_800_000_000_100,
			FinishedAtUnixMs:   1_800_000_001_600,
			WallTimeMs:         1_500,
			TimeToFirstTokenMs: 250,
			TokensGenerated:    16,
			TokensPerSecond:    10.67,
			PromptTokens:       &promptTokens,
			CompletionTokens:   &completionTokens,
			ModelLoadState:     ModelBenchmarkModelLoadStateLoaded,
		},
		OutputHash:  modelBenchHash("model output"),
		OutputBytes: 48,
		ProofStatus: ModelBenchmarkProofStatusMeasured,
	}
}
