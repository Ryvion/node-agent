package modelbench

import (
	"errors"
	"math"
	"strings"
	"testing"
)

func TestComputeBenchmarkSeriesSummaryComputesPercentiles(t *testing.T) {
	trials := []ModelBenchmarkTrialResult{
		measuredSeriesTrial(0, true, 10_000, 100, 50, 50),
		measuredSeriesTrial(1, false, 500, 5, 1, 5),
		measuredSeriesTrial(2, false, 100, 10, 2, 10),
		measuredSeriesTrial(3, false, 300, 15, 3, 15),
		measuredSeriesTrial(4, false, 200, 20, 4, 20),
		measuredSeriesTrial(5, false, 400, 25, 5, 25),
	}

	summary, err := ComputeBenchmarkSeriesSummary(trials)
	if err != nil {
		t.Fatalf("ComputeBenchmarkSeriesSummary() error = %v", err)
	}
	if summary.WarmupRuns != 1 || summary.MeasuredRuns != 5 {
		t.Fatalf("run counts = warmup %d measured %d, want 1 and 5", summary.WarmupRuns, summary.MeasuredRuns)
	}
	if summary.SuccessfulMeasuredRuns != 5 || summary.FailedMeasuredRuns != 0 {
		t.Fatalf("measured success/fail = %d/%d, want 5/0", summary.SuccessfulMeasuredRuns, summary.FailedMeasuredRuns)
	}
	if summary.P50TTFTMs != 300 || summary.P95TTFTMs != 500 {
		t.Fatalf("ttft p50/p95 = %d/%d, want 300/500", summary.P50TTFTMs, summary.P95TTFTMs)
	}
	if summary.P50DecodeTPS != 15 || summary.P95DecodeTPS != 25 {
		t.Fatalf("decode p50/p95 = %v/%v, want 15/25", summary.P50DecodeTPS, summary.P95DecodeTPS)
	}
	if summary.P50EndToEndTPS != 3 || summary.P95EndToEndTPS != 5 {
		t.Fatalf("end-to-end p50/p95 = %v/%v, want 3/5", summary.P50EndToEndTPS, summary.P95EndToEndTPS)
	}
	if summary.MinDecodeTPS != 5 || summary.MaxDecodeTPS != 25 {
		t.Fatalf("decode min/max = %v/%v, want 5/25", summary.MinDecodeTPS, summary.MaxDecodeTPS)
	}
	if summary.ProofStatus != ModelBenchmarkProofStatusMeasured {
		t.Fatalf("proof_status = %q, want measured", summary.ProofStatus)
	}
}

func TestComputeBenchmarkSeriesSummaryWarmupExcluded(t *testing.T) {
	trials := []ModelBenchmarkTrialResult{
		measuredSeriesTrial(0, true, 9_999, 99, 99, 99),
		measuredSeriesTrial(1, false, 100, 1, 2, 3),
		measuredSeriesTrial(2, false, 200, 2, 3, 4),
		measuredSeriesTrial(3, false, 300, 3, 4, 5),
	}

	summary, err := ComputeBenchmarkSeriesSummary(trials)
	if err != nil {
		t.Fatalf("ComputeBenchmarkSeriesSummary() error = %v", err)
	}
	if summary.P95TTFTMs != 300 {
		t.Fatalf("p95 ttft = %d, want measured-only value 300", summary.P95TTFTMs)
	}
	if summary.P95DecodeTPS != 3 {
		t.Fatalf("p95 decode = %v, want measured-only value 3", summary.P95DecodeTPS)
	}
}

func TestComputeBenchmarkSeriesSummaryCountsFailedTrials(t *testing.T) {
	trials := []ModelBenchmarkTrialResult{
		measuredSeriesTrial(0, false, 100, 10, 10, 10),
		{
			TrialIndex:      1,
			ProofStatus:     ModelBenchmarkProofStatusFailed,
			WallTimeMs:      25,
			ErrorMessage:    "native_request_failed",
			TokensGenerated: 0,
		},
		{
			TrialIndex:      2,
			ProofStatus:     ModelBenchmarkProofStatusUnavailable,
			WallTimeMs:      5,
			ErrorMessage:    "native_runtime_unavailable",
			TokensGenerated: 0,
		},
	}

	summary, err := ComputeBenchmarkSeriesSummary(trials)
	if err != nil {
		t.Fatalf("ComputeBenchmarkSeriesSummary() error = %v", err)
	}
	if summary.MeasuredRuns != 3 || summary.SuccessfulMeasuredRuns != 1 || summary.FailedMeasuredRuns != 2 {
		t.Fatalf("summary counts = %+v, want measured=3 success=1 failed=2", summary)
	}
	if summary.ProofStatus != ModelBenchmarkProofStatusMeasured {
		t.Fatalf("proof_status = %q, want measured with partial success", summary.ProofStatus)
	}
}

func TestComputeBenchmarkSeriesSummaryAllUnavailableOrFailed(t *testing.T) {
	unavailable, err := ComputeBenchmarkSeriesSummary([]ModelBenchmarkTrialResult{
		{TrialIndex: 0, ProofStatus: ModelBenchmarkProofStatusUnavailable, ErrorMessage: "native_runtime_unavailable"},
		{TrialIndex: 1, ProofStatus: ModelBenchmarkProofStatusUnavailable, ErrorMessage: "native_runtime_unavailable"},
	})
	if err != nil {
		t.Fatalf("ComputeBenchmarkSeriesSummary(unavailable) error = %v", err)
	}
	if unavailable.ProofStatus != ModelBenchmarkProofStatusUnavailable || unavailable.FailedMeasuredRuns != 2 {
		t.Fatalf("unavailable summary = %+v, want unavailable failed=2", unavailable)
	}

	failed, err := ComputeBenchmarkSeriesSummary([]ModelBenchmarkTrialResult{
		{TrialIndex: 0, ProofStatus: ModelBenchmarkProofStatusFailed, ErrorMessage: "native_request_failed"},
		{TrialIndex: 1, ProofStatus: ModelBenchmarkProofStatusUnavailable, ErrorMessage: "native_runtime_unavailable"},
	})
	if err != nil {
		t.Fatalf("ComputeBenchmarkSeriesSummary(failed) error = %v", err)
	}
	if failed.ProofStatus != ModelBenchmarkProofStatusFailed || failed.FailedMeasuredRuns != 2 {
		t.Fatalf("failed summary = %+v, want failed failed=2", failed)
	}
}

func TestComputeBenchmarkSeriesSummaryRejectsInvalidTrials(t *testing.T) {
	_, err := ComputeBenchmarkSeriesSummary(nil)
	if !errors.Is(err, ErrInvalidModelBenchmarkSeriesSummary) || !strings.Contains(err.Error(), "measured") {
		t.Fatalf("ComputeBenchmarkSeriesSummary(nil) error = %v, want measured trials invalid", err)
	}

	_, err = ComputeBenchmarkSeriesSummary([]ModelBenchmarkTrialResult{{
		TrialIndex:              0,
		ProofStatus:             ModelBenchmarkProofStatusMeasured,
		EndToEndTokensPerSecond: math.Inf(1),
	}})
	if !errors.Is(err, ErrInvalidModelBenchmarkSeriesSummary) || !strings.Contains(err.Error(), "end_to_end_tokens_per_second") {
		t.Fatalf("ComputeBenchmarkSeriesSummary(nonfinite) error = %v, want TPS invalid", err)
	}
}

func TestComputeDecodeTokensPerSecondFormula(t *testing.T) {
	got := computeDecodeTokensPerSecond(11, 1_200, 200)
	if got != 10 {
		t.Fatalf("computeDecodeTokensPerSecond() = %v, want 10", got)
	}
	if got := computeDecodeTokensPerSecond(1, 1_200, 200); got != 0 {
		t.Fatalf("single-token decode TPS = %v, want 0", got)
	}
	if got := computeDecodeTokensPerSecond(11, 200, 200); got != 0 {
		t.Fatalf("zero decode window TPS = %v, want 0", got)
	}
}

func measuredSeriesTrial(index int, warmup bool, ttftMs int64, decodeTPS float64, endToEndTPS float64, tokens int64) ModelBenchmarkTrialResult {
	return ModelBenchmarkTrialResult{
		TrialIndex:              index,
		Warmup:                  warmup,
		ProofStatus:             ModelBenchmarkProofStatusMeasured,
		WallTimeMs:              ttftMs + 1_000,
		TimeToFirstTokenMs:      ttftMs,
		TokensGenerated:         tokens,
		EndToEndTokensPerSecond: endToEndTPS,
		DecodeTokensPerSecond:   decodeTPS,
		OutputHash:              modelBenchHash("series output"),
		OutputBytes:             16,
	}
}
