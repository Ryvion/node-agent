package modelbench

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

const (
	defaultModelBenchmarkSelfTestModelID   = "ryvion-llama-3.2-3b"
	defaultModelBenchmarkSelfTestMaxTokens = 16
	defaultModelBenchmarkSelfTestTimeoutMs = 60_000
	defaultModelBenchmarkSelfTestTemp      = 0.1
	modelBenchmarkSelfTestPromptLabel      = "local-native-modelbench-selftest"
)

type ModelBenchmarkSelfTestConfig struct {
	ModelID   string
	MaxTokens int
	TimeoutMs int64
}

func DefaultModelBenchmarkSelfTestConfig() ModelBenchmarkSelfTestConfig {
	return ModelBenchmarkSelfTestConfig{
		ModelID:   defaultModelBenchmarkSelfTestModelID,
		MaxTokens: defaultModelBenchmarkSelfTestMaxTokens,
		TimeoutMs: defaultModelBenchmarkSelfTestTimeoutMs,
	}
}

func DefaultModelBenchmarkPrompt() ModelBenchmarkPrompt {
	return ModelBenchmarkPrompt{
		Label:   modelBenchmarkSelfTestPromptLabel,
		Content: []byte("Generate one short readiness sentence for a local Ryvion native model benchmark."),
	}
}

func RunModelBenchmarkSelfTest(ctx context.Context, runner ModelBenchmarkRunner, config ModelBenchmarkSelfTestConfig) (ModelBenchmarkResult, error) {
	if runner == nil {
		return ModelBenchmarkResult{}, ModelBenchmarkError{Code: "modelbench_runner_missing", Message: "model benchmark runner is not configured"}
	}
	config = normalizeModelBenchmarkSelfTestConfig(config)
	spec := BuildModelBenchmarkSelfTestSpec(config, time.Now())
	if err := ValidateModelBenchmarkSpec(spec); err != nil {
		return ModelBenchmarkResult{}, err
	}
	runCtx, cancel := context.WithTimeout(ctx, time.Duration(spec.TimeoutMs)*time.Millisecond)
	defer cancel()

	result, err := runner.RunModelBenchmark(runCtx, spec)
	if validationErr := ValidateModelBenchmarkResult(result); validationErr != nil {
		if err != nil {
			return result, fmt.Errorf("%w: %v", err, validationErr)
		}
		return result, validationErr
	}
	return result, err
}

func BuildModelBenchmarkSelfTestSpec(config ModelBenchmarkSelfTestConfig, createdAt time.Time) ModelBenchmarkSpec {
	config = normalizeModelBenchmarkSelfTestConfig(config)
	if createdAt.IsZero() {
		createdAt = time.Now()
	}
	createdMs := createdAt.UnixMilli()
	prompt := DefaultModelBenchmarkPrompt()
	return ModelBenchmarkSpec{
		Task:            ModelBenchmarkTask,
		RequestID:       fmt.Sprintf("modelbench-selftest-%d", createdMs),
		JobID:           fmt.Sprintf("modelbench-selftest-job-%d", createdMs),
		ModelID:         config.ModelID,
		PromptLabel:     prompt.Label,
		PromptHash:      HashBenchmarkPrompt(prompt),
		MaxTokens:       config.MaxTokens,
		Temperature:     defaultModelBenchmarkSelfTestTemp,
		TimeoutMs:       config.TimeoutMs,
		CreatedAtUnixMs: createdMs,
	}
}

func FormatModelBenchmarkSelfTestResult(result ModelBenchmarkResult, jsonOutput bool) string {
	if jsonOutput {
		encoded, err := json.MarshalIndent(result, "", "  ")
		if err != nil {
			return `{"proof_status":"native_model_failed","metrics":{"error_code":"format_modelbench_selftest_failed"}}`
		}
		return string(encoded)
	}

	ok := result.ProofStatus == ModelBenchmarkProofStatusMeasured
	errorCode := strings.TrimSpace(result.Metrics.ErrorCode)
	if errorCode == "" {
		errorCode = "none"
	}
	outputHash := strings.TrimSpace(result.OutputHash)
	if outputHash == "" {
		outputHash = "none"
	}
	return fmt.Sprintf(
		"ok: %t\nproof_status: %s\nmodel_id: %s\nruntime_kind: %s\nnative_inference_ready: %t\nwall_time_ms: %d\ntime_to_first_token_ms: %d\ntokens_generated: %d\ntokens_per_second: %.4f\noutput_hash: %s\nerror_code: %s",
		ok,
		result.ProofStatus,
		result.ModelID,
		result.RuntimeInfo.RuntimeKind,
		result.RuntimeInfo.NativeInferenceReady,
		result.Metrics.WallTimeMs,
		result.Metrics.TimeToFirstTokenMs,
		result.Metrics.TokensGenerated,
		result.Metrics.TokensPerSecond,
		outputHash,
		errorCode,
	)
}

func normalizeModelBenchmarkSelfTestConfig(config ModelBenchmarkSelfTestConfig) ModelBenchmarkSelfTestConfig {
	config.ModelID = strings.TrimSpace(config.ModelID)
	if config.ModelID == "" {
		config.ModelID = defaultModelBenchmarkSelfTestModelID
	}
	if config.MaxTokens <= 0 {
		config.MaxTokens = defaultModelBenchmarkSelfTestMaxTokens
	}
	if config.MaxTokens > MaxModelBenchmarkTokens {
		config.MaxTokens = MaxModelBenchmarkTokens
	}
	if config.TimeoutMs <= 0 {
		config.TimeoutMs = defaultModelBenchmarkSelfTestTimeoutMs
	}
	if config.TimeoutMs > MaxModelBenchmarkTimeoutMs {
		config.TimeoutMs = MaxModelBenchmarkTimeoutMs
	}
	return config
}
