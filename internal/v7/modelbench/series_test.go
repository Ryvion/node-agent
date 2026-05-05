package modelbench

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestRunModelBenchmarkSeriesRunsWarmupAndMeasuredTrials(t *testing.T) {
	spec := validModelBenchmarkSeriesSpec()
	spec.WarmupRuns = 2
	spec.MeasuredRuns = 3
	runner := &fakeSeriesRunner{
		results: []ModelBenchmarkResult{
			seriesRunResult(ModelBenchmarkProofStatusMeasured, 100, 1_100, 11, 10, ""),
			seriesRunResult(ModelBenchmarkProofStatusMeasured, 110, 1_210, 12, 10, ""),
			seriesRunResult(ModelBenchmarkProofStatusMeasured, 200, 1_200, 11, 9.17, ""),
			seriesRunResult(ModelBenchmarkProofStatusMeasured, 300, 1_400, 12, 8.57, ""),
			seriesRunResult(ModelBenchmarkProofStatusMeasured, 250, 1_450, 13, 8.97, ""),
		},
	}

	result, err := RunModelBenchmarkSeries(context.Background(), runner, spec)
	if err != nil {
		t.Fatalf("RunModelBenchmarkSeries() error = %v", err)
	}
	if runner.calls != 5 {
		t.Fatalf("runner calls = %d, want 5", runner.calls)
	}
	if len(result.Trials) != 5 {
		t.Fatalf("trials = %d, want 5", len(result.Trials))
	}
	for i, trial := range result.Trials {
		if trial.TrialIndex != i {
			t.Fatalf("trial[%d].TrialIndex = %d, want %d", i, trial.TrialIndex, i)
		}
		wantWarmup := i < spec.WarmupRuns
		if trial.Warmup != wantWarmup {
			t.Fatalf("trial[%d].Warmup = %v, want %v", i, trial.Warmup, wantWarmup)
		}
	}
	for i, got := range runner.specs {
		if got.Task != ModelBenchmarkTask || got.RequestID != spec.RequestID || got.JobID != spec.JobID || got.ModelID != spec.ModelID {
			t.Fatalf("runner spec[%d] = %+v, want series identity", i, got)
		}
		if got.PromptProfileID != spec.PromptProfileID || got.PromptHash != spec.PromptHash {
			t.Fatalf("runner spec[%d] prompt binding = %q/%q, want %q/%q", i, got.PromptProfileID, got.PromptHash, spec.PromptProfileID, spec.PromptHash)
		}
	}
	if result.Summary.WarmupRuns != 2 || result.Summary.MeasuredRuns != 3 || result.Summary.SuccessfulMeasuredRuns != 3 {
		t.Fatalf("summary counts = %+v, want 2 warmup, 3 measured successes", result.Summary)
	}
	if result.Summary.P50TTFTMs != 250 || result.Summary.P95TTFTMs != 300 {
		t.Fatalf("summary ttft p50/p95 = %d/%d, want 250/300", result.Summary.P50TTFTMs, result.Summary.P95TTFTMs)
	}
	if result.Trials[2].DecodeTokensPerSecond != 10 {
		t.Fatalf("decode TPS = %v, want 10", result.Trials[2].DecodeTokensPerSecond)
	}
}

func TestRunModelBenchmarkSeriesUnavailableRunnerProducesUnavailableSummary(t *testing.T) {
	spec := validModelBenchmarkSeriesSpec()
	spec.WarmupRuns = 1
	spec.MeasuredRuns = 2
	runner := &fakeSeriesRunner{
		results: []ModelBenchmarkResult{
			seriesRunResult(ModelBenchmarkProofStatusUnavailable, 0, 25, 0, 0, "native_runtime_unavailable"),
			seriesRunResult(ModelBenchmarkProofStatusUnavailable, 0, 30, 0, 0, "native_runtime_unavailable"),
			seriesRunResult(ModelBenchmarkProofStatusUnavailable, 0, 20, 0, 0, "native_runtime_unavailable"),
		},
	}

	result, err := RunModelBenchmarkSeries(context.Background(), runner, spec)
	if err != nil {
		t.Fatalf("RunModelBenchmarkSeries() error = %v", err)
	}
	if result.Summary.WarmupRuns != 1 || result.Summary.MeasuredRuns != 2 || result.Summary.SuccessfulMeasuredRuns != 0 || result.Summary.FailedMeasuredRuns != 2 {
		t.Fatalf("summary counts = %+v, want unavailable measured failures", result.Summary)
	}
	if result.Summary.ProofStatus != ModelBenchmarkProofStatusUnavailable {
		t.Fatalf("proof_status = %q, want unavailable", result.Summary.ProofStatus)
	}
	if result.Trials[1].ErrorMessage != "native_runtime_unavailable" {
		t.Fatalf("trial error_message = %q, want native_runtime_unavailable", result.Trials[1].ErrorMessage)
	}
}

func TestRunModelBenchmarkSeriesCountsRunnerErrorsAsFailedTrials(t *testing.T) {
	spec := validModelBenchmarkSeriesSpec()
	spec.WarmupRuns = 0
	spec.MeasuredRuns = 3
	runner := &fakeSeriesRunner{
		results: []ModelBenchmarkResult{
			seriesRunResult(ModelBenchmarkProofStatusMeasured, 100, 1_100, 11, 10, ""),
			seriesRunResult(ModelBenchmarkProofStatusFailed, 0, 10, 0, 0, "native_request_failed"),
			seriesRunResult(ModelBenchmarkProofStatusMeasured, 120, 1_120, 11, 9.82, ""),
		},
		errs: []error{
			nil,
			ModelBenchmarkError{Code: "native_request_failed", Message: "local inference request failed"},
			nil,
		},
	}

	result, err := RunModelBenchmarkSeries(context.Background(), runner, spec)
	if err != nil {
		t.Fatalf("RunModelBenchmarkSeries() error = %v", err)
	}
	if result.Summary.SuccessfulMeasuredRuns != 2 || result.Summary.FailedMeasuredRuns != 1 {
		t.Fatalf("summary counts = %+v, want 2 success and 1 failed", result.Summary)
	}
	if result.Trials[1].ProofStatus != ModelBenchmarkProofStatusFailed || result.Trials[1].ErrorMessage != "native_request_failed" {
		t.Fatalf("failed trial = %+v, want failed error code", result.Trials[1])
	}
}

func TestRunModelBenchmarkSeriesDoesNotExposeRawPromptOrOutput(t *testing.T) {
	spec := validModelBenchmarkSeriesSpec()
	spec.WarmupRuns = 0
	spec.MeasuredRuns = 1
	rawResult := seriesRunResult(ModelBenchmarkProofStatusMeasured, 100, 1_100, 11, 10, "")
	rawResult.OutputHash = "sensitive raw output"
	rawResult.Metrics.ErrorCode = "raw output failure"
	runner := &fakeSeriesRunner{
		results: []ModelBenchmarkResult{
			rawResult,
		},
	}

	result, err := RunModelBenchmarkSeries(context.Background(), runner, spec)
	if err != nil {
		t.Fatalf("RunModelBenchmarkSeries() error = %v", err)
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("json.Marshal(result) error = %v", err)
	}
	text := string(encoded)
	for _, forbidden := range []string{
		"Reply with exactly: ready.",
		"raw benchmark prompt content",
		"sensitive raw output",
		"prompt_text",
		"output_text",
		"raw_output",
	} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("series result leaked forbidden raw content %q: %s", forbidden, text)
		}
	}
}

func TestValidateModelBenchmarkSeriesSpecRejectsInvalidRunCounts(t *testing.T) {
	tests := []struct {
		name      string
		mutate    func(*ModelBenchmarkSeriesSpec)
		wantField string
	}{
		{
			name:      "measured zero",
			mutate:    func(spec *ModelBenchmarkSeriesSpec) { spec.MeasuredRuns = 0 },
			wantField: "measured_runs",
		},
		{
			name:      "measured cap",
			mutate:    func(spec *ModelBenchmarkSeriesSpec) { spec.MeasuredRuns = MaxModelBenchmarkSeriesMeasuredRuns + 1 },
			wantField: "measured_runs",
		},
		{
			name:      "warmup negative",
			mutate:    func(spec *ModelBenchmarkSeriesSpec) { spec.WarmupRuns = -1 },
			wantField: "warmup_runs",
		},
		{
			name:      "warmup cap",
			mutate:    func(spec *ModelBenchmarkSeriesSpec) { spec.WarmupRuns = MaxModelBenchmarkSeriesWarmupRuns + 1 },
			wantField: "warmup_runs",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			spec := validModelBenchmarkSeriesSpec()
			test.mutate(&spec)

			_, err := RunModelBenchmarkSeries(context.Background(), &fakeSeriesRunner{}, spec)
			if !errors.Is(err, ErrInvalidModelBenchmarkSeriesSpec) || !strings.Contains(err.Error(), test.wantField) {
				t.Fatalf("RunModelBenchmarkSeries() error = %v, want invalid %s", err, test.wantField)
			}
		})
	}
}

func validModelBenchmarkSeriesSpec() ModelBenchmarkSeriesSpec {
	profile, err := GetBenchmarkPromptProfile(string(BenchmarkPromptProfileShortTTFTProbe))
	if err != nil {
		panic(err)
	}
	return ModelBenchmarkSeriesSpec{
		RequestID:       "request-modelbench-series-1",
		JobID:           "job-modelbench-series-1",
		ModelID:         "llama-local-7b-q4",
		PromptProfileID: string(profile.ID),
		PromptHash:      BenchmarkPromptHash(profile),
		MaxTokens:       32,
		TimeoutMs:       30_000,
		WarmupRuns:      1,
		MeasuredRuns:    3,
		CreatedAtUnixMs: 1_800_000_000_000,
	}
}

func seriesRunResult(status ModelBenchmarkProofStatus, ttftMs int64, wallMs int64, tokens int64, endToEndTPS float64, errorCode string) ModelBenchmarkResult {
	result := validMeasuredModelBenchmarkResult()
	spec := validModelBenchmarkSeriesSpec()
	result.RequestID = spec.RequestID
	result.JobID = spec.JobID
	result.ModelID = spec.ModelID
	result.PromptHash = spec.PromptHash
	result.RuntimeInfo.ModelID = spec.ModelID
	result.Metrics.TimeToFirstTokenMs = ttftMs
	result.Metrics.WallTimeMs = wallMs
	result.Metrics.TokensGenerated = tokens
	result.Metrics.TokensPerSecond = endToEndTPS
	result.Metrics.ErrorCode = errorCode
	result.OutputHash = modelBenchHash("sensitive raw output")
	result.OutputBytes = 20
	result.ProofStatus = status
	if status != ModelBenchmarkProofStatusMeasured {
		result.RuntimeInfo.NativeInferenceReady = false
		result.RuntimeInfo.ModelLoaded = false
		result.Metrics.ModelLoadState = ModelBenchmarkModelLoadStateUnavailable
		if status == ModelBenchmarkProofStatusFailed {
			result.Metrics.ModelLoadState = ModelBenchmarkModelLoadStateFailed
			result.RuntimeInfo.NativeInferenceReady = true
		}
		result.OutputHash = ""
		result.OutputBytes = 0
	}
	return result
}

type fakeSeriesRunner struct {
	results []ModelBenchmarkResult
	errs    []error
	specs   []ModelBenchmarkSpec
	calls   int
}

func (f *fakeSeriesRunner) RunModelBenchmark(_ context.Context, spec ModelBenchmarkSpec) (ModelBenchmarkResult, error) {
	f.specs = append(f.specs, spec)
	index := f.calls
	f.calls++
	var result ModelBenchmarkResult
	if index < len(f.results) {
		result = f.results[index]
	}
	var err error
	if index < len(f.errs) {
		err = f.errs[index]
	}
	return result, err
}
