package modelbench

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestBuildModelBenchmarkSelfTestSpecValid(t *testing.T) {
	spec := BuildModelBenchmarkSelfTestSpec(DefaultModelBenchmarkSelfTestConfig(), time.UnixMilli(1_800_000_000_000))
	if err := ValidateModelBenchmarkSpec(spec); err != nil {
		t.Fatalf("ValidateModelBenchmarkSpec() error = %v", err)
	}
	if spec.Task != ModelBenchmarkTask {
		t.Fatalf("task = %q, want %q", spec.Task, ModelBenchmarkTask)
	}
	if spec.ModelID != defaultModelBenchmarkSelfTestModelID {
		t.Fatalf("model_id = %q, want default", spec.ModelID)
	}
	if spec.PromptHash != HashBenchmarkPrompt(DefaultModelBenchmarkPrompt()) {
		t.Fatalf("prompt_hash = %q, want default prompt hash", spec.PromptHash)
	}
}

func TestRunModelBenchmarkSelfTestSuccessfulFakeRunnerReturnsMeasuredResult(t *testing.T) {
	runner := fakeSelfTestRunner{
		run: func(_ context.Context, spec ModelBenchmarkSpec) (ModelBenchmarkResult, error) {
			return measuredResultForSpec(spec), nil
		},
	}

	result, err := RunModelBenchmarkSelfTest(context.Background(), &runner, ModelBenchmarkSelfTestConfig{
		ModelID:   "tinyllama",
		MaxTokens: 4,
		TimeoutMs: 5_000,
	})
	if err != nil {
		t.Fatalf("RunModelBenchmarkSelfTest() error = %v", err)
	}
	if result.ProofStatus != ModelBenchmarkProofStatusMeasured {
		t.Fatalf("proof_status = %q, want measured", result.ProofStatus)
	}
	if result.ModelID != "tinyllama" {
		t.Fatalf("model_id = %q, want tinyllama", result.ModelID)
	}
	if runner.seenSpec.ModelID != "tinyllama" || runner.seenSpec.MaxTokens != 4 {
		t.Fatalf("runner saw spec %+v", runner.seenSpec)
	}
}

func TestFormatModelBenchmarkSelfTestResultJSONDoesNotExposeRawOutput(t *testing.T) {
	spec := testModelBenchmarkSelfTestSpec()
	result := measuredResultForSpec(spec)
	formatted := FormatModelBenchmarkSelfTestResult(result, true)

	var decoded ModelBenchmarkResult
	if err := json.Unmarshal([]byte(formatted), &decoded); err != nil {
		t.Fatalf("JSON output is invalid: %v", err)
	}
	if decoded.OutputHash != result.OutputHash {
		t.Fatalf("output_hash = %q, want %q", decoded.OutputHash, result.OutputHash)
	}
	if strings.Contains(formatted, "raw model output") {
		t.Fatalf("formatted result leaked raw output: %s", formatted)
	}
}

func TestFormatModelBenchmarkSelfTestResultReadableContainsRequiredFields(t *testing.T) {
	formatted := FormatModelBenchmarkSelfTestResult(measuredResultForSpec(testModelBenchmarkSelfTestSpec()), false)
	for _, key := range []string{
		"ok:",
		"proof_status:",
		"model_id:",
		"runtime_kind:",
		"native_inference_ready:",
		"wall_time_ms:",
		"time_to_first_token_ms:",
		"tokens_generated:",
		"tokens_per_second:",
		"output_hash:",
		"error_code:",
	} {
		if !strings.Contains(formatted, key) {
			t.Fatalf("readable output missing %q: %s", key, formatted)
		}
	}
}

type fakeSelfTestRunner struct {
	seenSpec ModelBenchmarkSpec
	run      func(context.Context, ModelBenchmarkSpec) (ModelBenchmarkResult, error)
}

func (f *fakeSelfTestRunner) RunModelBenchmark(ctx context.Context, spec ModelBenchmarkSpec) (ModelBenchmarkResult, error) {
	f.seenSpec = spec
	return f.run(ctx, spec)
}

func measuredResultForSpec(spec ModelBenchmarkSpec) ModelBenchmarkResult {
	promptTokens := int64(8)
	completionTokens := int64(4)
	return ModelBenchmarkResult{
		RequestID:  spec.RequestID,
		JobID:      spec.JobID,
		ModelID:    spec.ModelID,
		PromptHash: spec.PromptHash,
		RuntimeInfo: ModelBenchmarkRuntimeInfo{
			AgentVersion:             "test",
			OS:                       "darwin",
			Arch:                     "arm64",
			NativeInferenceSupported: true,
			NativeInferenceReady:     true,
			RuntimeKind:              ModelBenchmarkRuntimeKindNativeLocal,
			ModelID:                  spec.ModelID,
			ModelLoaded:              true,
		},
		Metrics: ModelBenchmarkMetrics{
			StartedAtUnixMs:    spec.CreatedAtUnixMs,
			FinishedAtUnixMs:   spec.CreatedAtUnixMs + 250,
			WallTimeMs:         250,
			TimeToFirstTokenMs: 50,
			TokensGenerated:    4,
			TokensPerSecond:    16,
			PromptTokens:       &promptTokens,
			CompletionTokens:   &completionTokens,
			ModelLoadState:     ModelBenchmarkModelLoadStateLoaded,
		},
		OutputHash:  hashBenchmarkOutput([]byte("raw model output")),
		OutputBytes: int64(len("raw model output")),
		ProofStatus: ModelBenchmarkProofStatusMeasured,
	}
}
