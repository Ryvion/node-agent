package modelbench

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

const ModelBenchmarkSeriesTask = "v7_model_benchmark_series"

type modelBenchmarkSeriesAssignmentSpec struct {
	Task            string `json:"task"`
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

func IsModelBenchmarkSeriesSpecJSON(specJSON string) bool {
	var header struct {
		Task string `json:"task"`
	}
	if json.Unmarshal([]byte(strings.TrimSpace(specJSON)), &header) != nil {
		return false
	}
	return strings.TrimSpace(header.Task) == ModelBenchmarkSeriesTask
}

func ModelBenchmarkSeriesAssignmentIdentityFromJSON(specJSON string) (ModelBenchmarkAssignmentIdentity, bool) {
	var header struct {
		Task      string `json:"task"`
		JobID     string `json:"job_id"`
		RequestID string `json:"request_id"`
	}
	if json.Unmarshal([]byte(strings.TrimSpace(specJSON)), &header) != nil {
		return ModelBenchmarkAssignmentIdentity{}, false
	}
	if strings.TrimSpace(header.Task) != ModelBenchmarkSeriesTask {
		return ModelBenchmarkAssignmentIdentity{}, false
	}
	return ModelBenchmarkAssignmentIdentity{
		JobID:     cleanLocalStatusText(header.JobID, maxLocalStatusIDLen),
		RequestID: cleanLocalStatusText(header.RequestID, maxLocalStatusIDLen),
	}, true
}

func DecodeModelBenchmarkSeriesSpec(specJSON string) (ModelBenchmarkSeriesSpec, error) {
	raw := strings.TrimSpace(specJSON)
	if raw == "" {
		return ModelBenchmarkSeriesSpec{}, fmt.Errorf("%w: spec_json required", ErrInvalidModelBenchmarkSeriesSpec)
	}

	var assignment modelBenchmarkSeriesAssignmentSpec
	if err := json.Unmarshal([]byte(raw), &assignment); err != nil {
		return ModelBenchmarkSeriesSpec{}, fmt.Errorf("%w: decode spec_json: %v", ErrInvalidModelBenchmarkSeriesSpec, err)
	}
	if strings.TrimSpace(assignment.Task) != ModelBenchmarkSeriesTask {
		return ModelBenchmarkSeriesSpec{}, fmt.Errorf("%w: task must be %q", ErrInvalidModelBenchmarkSeriesSpec, ModelBenchmarkSeriesTask)
	}

	spec := normalizeModelBenchmarkSeriesSpec(ModelBenchmarkSeriesSpec{
		RequestID:       assignment.RequestID,
		JobID:           assignment.JobID,
		ModelID:         assignment.ModelID,
		PromptProfileID: assignment.PromptProfileID,
		PromptHash:      assignment.PromptHash,
		MaxTokens:       assignment.MaxTokens,
		TimeoutMs:       assignment.TimeoutMs,
		WarmupRuns:      assignment.WarmupRuns,
		MeasuredRuns:    assignment.MeasuredRuns,
		CreatedAtUnixMs: assignment.CreatedAtUnixMs,
	})
	if err := ValidateModelBenchmarkSeriesSpec(spec); err != nil {
		return ModelBenchmarkSeriesSpec{}, err
	}
	return spec, nil
}

func ExecuteModelBenchmarkSeriesAssignment(ctx context.Context, specJSON string, runner ModelBenchmarkRunner, env func(string) string) (ModelBenchmarkReceipt, bool, error) {
	if !IsModelBenchmarkSeriesSpecJSON(specJSON) {
		return ModelBenchmarkReceipt{}, false, nil
	}
	if !ModelBenchmarkEnabledFromEnv(env) {
		return ModelBenchmarkReceipt{}, false, nil
	}

	spec, err := DecodeModelBenchmarkSeriesSpec(specJSON)
	if err != nil {
		return ModelBenchmarkReceipt{}, true, err
	}
	receipt, err := ExecuteModelBenchmarkSeriesSpec(ctx, spec, runner)
	return receipt, true, err
}

func ExecuteModelBenchmarkSeriesSpec(ctx context.Context, spec ModelBenchmarkSeriesSpec, runner ModelBenchmarkRunner) (ModelBenchmarkReceipt, error) {
	result, runErr := RunModelBenchmarkSeries(ctx, runner, spec)
	receipt, receiptErr := BuildModelBenchmarkSeriesReceipt(result)
	if receiptErr != nil {
		if runErr != nil {
			return ModelBenchmarkReceipt{}, fmt.Errorf("%w: %v", runErr, receiptErr)
		}
		return ModelBenchmarkReceipt{}, receiptErr
	}
	return receipt, runErr
}
