package modelbench

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
)

var ErrInvalidModelBenchmarkSeriesSummary = errors.New("modelbench: invalid model benchmark series summary")

type ModelBenchmarkSeriesSummary struct {
	WarmupRuns             int                       `json:"warmup_runs"`
	MeasuredRuns           int                       `json:"measured_runs"`
	SuccessfulMeasuredRuns int                       `json:"successful_measured_runs"`
	FailedMeasuredRuns     int                       `json:"failed_measured_runs"`
	P50TTFTMs              int64                     `json:"p50_ttft_ms"`
	P95TTFTMs              int64                     `json:"p95_ttft_ms"`
	P50DecodeTPS           float64                   `json:"p50_decode_tps"`
	P95DecodeTPS           float64                   `json:"p95_decode_tps"`
	P50EndToEndTPS         float64                   `json:"p50_end_to_end_tps"`
	P95EndToEndTPS         float64                   `json:"p95_end_to_end_tps"`
	MinDecodeTPS           float64                   `json:"min_decode_tps"`
	MaxDecodeTPS           float64                   `json:"max_decode_tps"`
	ProofStatus            ModelBenchmarkProofStatus `json:"proof_status"`
}

func ComputeBenchmarkSeriesSummary(trials []ModelBenchmarkTrialResult) (ModelBenchmarkSeriesSummary, error) {
	var summary ModelBenchmarkSeriesSummary
	var ttftValues []int64
	var decodeValues []float64
	var endToEndValues []float64
	var unavailableMeasuredRuns int

	for _, trial := range trials {
		if err := validateModelBenchmarkTrialResult(trial); err != nil {
			return summary, err
		}
		if trial.Warmup {
			summary.WarmupRuns++
			continue
		}

		summary.MeasuredRuns++
		if trial.ProofStatus == ModelBenchmarkProofStatusMeasured && strings.TrimSpace(trial.ErrorMessage) == "" {
			summary.SuccessfulMeasuredRuns++
			ttftValues = append(ttftValues, trial.TimeToFirstTokenMs)
			decodeValues = append(decodeValues, trial.DecodeTokensPerSecond)
			endToEndValues = append(endToEndValues, trial.EndToEndTokensPerSecond)
			continue
		}

		summary.FailedMeasuredRuns++
		if trial.ProofStatus == ModelBenchmarkProofStatusUnavailable {
			unavailableMeasuredRuns++
		}
	}

	if summary.MeasuredRuns == 0 {
		return summary, fmt.Errorf("%w: measured trials required", ErrInvalidModelBenchmarkSeriesSummary)
	}
	if summary.SuccessfulMeasuredRuns > 0 {
		summary.P50TTFTMs = percentileInt64(ttftValues, 50)
		summary.P95TTFTMs = percentileInt64(ttftValues, 95)
		summary.P50DecodeTPS = percentileFloat64(decodeValues, 50)
		summary.P95DecodeTPS = percentileFloat64(decodeValues, 95)
		summary.P50EndToEndTPS = percentileFloat64(endToEndValues, 50)
		summary.P95EndToEndTPS = percentileFloat64(endToEndValues, 95)
		summary.MinDecodeTPS, summary.MaxDecodeTPS = minMaxFloat64(decodeValues)
		summary.ProofStatus = ModelBenchmarkProofStatusMeasured
		return summary, nil
	}

	if unavailableMeasuredRuns == summary.MeasuredRuns {
		summary.ProofStatus = ModelBenchmarkProofStatusUnavailable
	} else {
		summary.ProofStatus = ModelBenchmarkProofStatusFailed
	}
	return summary, nil
}

func percentileInt64(values []int64, percentile float64) int64 {
	if len(values) == 0 {
		return 0
	}
	sorted := append([]int64(nil), values...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	return sorted[nearestRankIndex(len(sorted), percentile)]
}

func percentileFloat64(values []float64, percentile float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	return sorted[nearestRankIndex(len(sorted), percentile)]
}

func nearestRankIndex(length int, percentile float64) int {
	if length <= 0 {
		return 0
	}
	if percentile <= 0 {
		return 0
	}
	if percentile >= 100 {
		return length - 1
	}
	index := int(math.Ceil((percentile / 100) * float64(length)))
	if index < 1 {
		return 0
	}
	if index > length {
		return length - 1
	}
	return index - 1
}

func minMaxFloat64(values []float64) (float64, float64) {
	if len(values) == 0 {
		return 0, 0
	}
	min := values[0]
	max := values[0]
	for _, value := range values[1:] {
		if value < min {
			min = value
		}
		if value > max {
			max = value
		}
	}
	return min, max
}

func validateModelBenchmarkTrialResult(trial ModelBenchmarkTrialResult) error {
	var errs []error
	if trial.TrialIndex < 0 {
		errs = append(errs, fmt.Errorf("%w: trial_index must be non-negative", ErrInvalidModelBenchmarkSeriesSummary))
	}
	switch trial.ProofStatus {
	case ModelBenchmarkProofStatusMeasured,
		ModelBenchmarkProofStatusUnavailable,
		ModelBenchmarkProofStatusFailed:
	default:
		errs = append(errs, fmt.Errorf("%w: proof_status unknown %q", ErrInvalidModelBenchmarkSeriesSummary, trial.ProofStatus))
	}
	if trial.WallTimeMs < 0 {
		errs = append(errs, fmt.Errorf("%w: wall_time_ms must be non-negative", ErrInvalidModelBenchmarkSeriesSummary))
	}
	if trial.TimeToFirstTokenMs < 0 {
		errs = append(errs, fmt.Errorf("%w: time_to_first_token_ms must be non-negative", ErrInvalidModelBenchmarkSeriesSummary))
	}
	if trial.TokensGenerated < 0 {
		errs = append(errs, fmt.Errorf("%w: tokens_generated must be non-negative", ErrInvalidModelBenchmarkSeriesSummary))
	}
	if !isFinite(trial.EndToEndTokensPerSecond) || trial.EndToEndTokensPerSecond < 0 {
		errs = append(errs, fmt.Errorf("%w: end_to_end_tokens_per_second must be finite and non-negative", ErrInvalidModelBenchmarkSeriesSummary))
	}
	if !isFinite(trial.DecodeTokensPerSecond) || trial.DecodeTokensPerSecond < 0 {
		errs = append(errs, fmt.Errorf("%w: decode_tokens_per_second must be finite and non-negative", ErrInvalidModelBenchmarkSeriesSummary))
	}
	if trial.OutputHash != "" {
		if err := validateHashID(trial.OutputHash, "output_hash", ErrInvalidModelBenchmarkSeriesSummary); err != nil {
			errs = append(errs, err)
		}
	}
	if trial.OutputBytes < 0 {
		errs = append(errs, fmt.Errorf("%w: output_bytes must be non-negative", ErrInvalidModelBenchmarkSeriesSummary))
	}
	return errors.Join(errs...)
}
