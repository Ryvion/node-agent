package tensorplane

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

type TensorPlaneProbeConfig struct {
	Tokens           int
	HeadDim          int
	ValueDim         int
	DType            TensorDType
	Seed             int64
	WriteFixturePath string
	ReadFixturePath  string
}

type TensorPlaneProbeResult struct {
	OK                    bool        `json:"ok"`
	DType                 TensorDType `json:"dtype"`
	Tokens                int         `json:"tokens"`
	HeadDim               int         `json:"head_dim"`
	ValueDim              int         `json:"value_dim"`
	LocalMax              float64     `json:"local_max"`
	ExpSum                float64     `json:"exp_sum"`
	SummaryHash           string      `json:"summary_hash"`
	PageHash              string      `json:"page_hash"`
	ComputeTimeUs         int64       `json:"compute_time_us"`
	PayloadBytesEstimate  int64       `json:"payload_bytes_estimate"`
	MaxAbsDiffVsReference float64     `json:"max_abs_diff_vs_reference"`
}

type tensorPartialAttentionReference struct {
	LocalMax      float64
	ExpSum        float64
	WeightedValue []float64
}

func DefaultTensorPlaneProbeConfig() TensorPlaneProbeConfig {
	defaults := DefaultTensorPlaneFixtureConfig()
	return TensorPlaneProbeConfig{
		Tokens:   defaults.Tokens,
		HeadDim:  defaults.HeadDim,
		ValueDim: defaults.ValueDim,
		DType:    defaults.DType,
		Seed:     defaults.Seed,
	}
}

func RunTensorPlaneProbe(config TensorPlaneProbeConfig) (TensorPlaneProbeResult, error) {
	config.DType = NormalizeTensorDType(config.DType)
	config.WriteFixturePath = strings.TrimSpace(config.WriteFixturePath)
	config.ReadFixturePath = strings.TrimSpace(config.ReadFixturePath)

	fixture, err := tensorPlaneProbeFixture(config)
	if err != nil {
		return TensorPlaneProbeResult{}, err
	}
	page, query, err := TensorPlaneFixturePageAndQuery(fixture)
	if err != nil {
		return TensorPlaneProbeResult{}, err
	}
	summary, err := ComputeTensorPartialAttentionSummary(query, page)
	if err != nil {
		return TensorPlaneProbeResult{}, err
	}
	reference, err := computeNaiveTensorPartialAttentionReference(query, page)
	if err != nil {
		return TensorPlaneProbeResult{}, err
	}
	maxDiff := maxAbsDiffVsReference(summary, reference)
	if maxDiff > TensorPlaneProbeTolerance(summary.DType) {
		return TensorPlaneProbeResult{}, fmt.Errorf("%w: max_abs_diff_vs_reference %.17g exceeds tolerance %.17g", ErrInvalidAttentionQuery, maxDiff, TensorPlaneProbeTolerance(summary.DType))
	}

	return TensorPlaneProbeResult{
		OK:                    true,
		DType:                 summary.DType,
		Tokens:                page.Shape.Tokens,
		HeadDim:               page.Shape.HeadDim,
		ValueDim:              summary.ValueDim,
		LocalMax:              summary.LocalMax,
		ExpSum:                summary.ExpSum,
		SummaryHash:           summary.SummaryHash,
		PageHash:              summary.PageHash,
		ComputeTimeUs:         summary.ComputeTimeUs,
		PayloadBytesEstimate:  summary.PayloadBytesEstimate,
		MaxAbsDiffVsReference: maxDiff,
	}, nil
}

func FormatTensorPlaneProbeResult(result TensorPlaneProbeResult, jsonOutput bool) string {
	if jsonOutput {
		encoded, err := json.MarshalIndent(result, "", "  ")
		if err != nil {
			return `{"ok":false,"error":"format tensorplane self-test result"}`
		}
		return string(encoded)
	}
	return fmt.Sprintf(
		"ok: %t\ndtype: %s\ntokens: %d\nhead_dim: %d\nvalue_dim: %d\nlocal_max: %.17g\nexp_sum: %.17g\nsummary_hash: %s\npage_hash: %s\ncompute_time_us: %d\npayload_bytes_estimate: %d\nmax_abs_diff_vs_reference: %.17g",
		result.OK,
		result.DType,
		result.Tokens,
		result.HeadDim,
		result.ValueDim,
		result.LocalMax,
		result.ExpSum,
		result.SummaryHash,
		result.PageHash,
		result.ComputeTimeUs,
		result.PayloadBytesEstimate,
		result.MaxAbsDiffVsReference,
	)
}

func TensorPlaneProbeTolerance(dtype TensorDType) float64 {
	switch NormalizeTensorDType(dtype) {
	case TensorDTypeFloat16:
		return 1e-4
	default:
		return 1e-9
	}
}

func tensorPlaneProbeFixture(config TensorPlaneProbeConfig) (TensorPlaneFixture, error) {
	if config.WriteFixturePath != "" {
		generated, err := BuildTensorPlaneFixture(TensorPlaneFixtureConfig{
			Tokens:   config.Tokens,
			HeadDim:  config.HeadDim,
			ValueDim: config.ValueDim,
			DType:    config.DType,
			Seed:     config.Seed,
		})
		if err != nil {
			return TensorPlaneFixture{}, err
		}
		if err := WriteTensorPlaneFixtureFile(config.WriteFixturePath, generated); err != nil {
			return TensorPlaneFixture{}, err
		}
		if config.ReadFixturePath == "" {
			return LoadTensorPlaneFixtureFile(config.WriteFixturePath)
		}
	}
	if config.ReadFixturePath != "" {
		return LoadTensorPlaneFixtureFile(config.ReadFixturePath)
	}
	return BuildTensorPlaneFixture(TensorPlaneFixtureConfig{
		Tokens:   config.Tokens,
		HeadDim:  config.HeadDim,
		ValueDim: config.ValueDim,
		DType:    config.DType,
		Seed:     config.Seed,
	})
}

func computeNaiveTensorPartialAttentionReference(query AttentionQuery, page TensorPage) (tensorPartialAttentionReference, error) {
	page = normalizeTensorPage(page)
	query = normalizeAttentionQuery(query)
	if err := ValidateTensorPage(page); err != nil {
		return tensorPartialAttentionReference{}, err
	}
	if err := ValidateAttentionQuery(query, page); err != nil {
		return tensorPartialAttentionReference{}, err
	}
	localHead, err := resolveQueryLocalHead(query, page)
	if err != nil {
		return tensorPartialAttentionReference{}, err
	}

	localMax := math.Inf(-1)
	expSum := 0.0
	weightedValue := make([]float64, page.Shape.ValueDim)
	for token := 0; token < page.Shape.Tokens; token++ {
		logit := 0.0
		for dim, queryValue := range query.QueryVector {
			keyValue, err := decodeTensorFloat(page.DType, page.KeyData, keyElementIndex(page.Shape, localHead, token, dim))
			if err != nil {
				return tensorPartialAttentionReference{}, err
			}
			logit += float64(queryValue) * float64(keyValue) * query.Scale
			if !finiteFloat64(logit) {
				return tensorPartialAttentionReference{}, fmt.Errorf("%w: reference logit overflow at token %d", ErrInvalidAttentionQuery, token)
			}
		}
		if logit > localMax {
			scale := math.Exp(localMax - logit)
			expSum *= scale
			for dim := range weightedValue {
				weightedValue[dim] *= scale
			}
			localMax = logit
		}
		weight := math.Exp(logit - localMax)
		if !finiteFloat64(weight) {
			return tensorPartialAttentionReference{}, fmt.Errorf("%w: reference exponent overflow at token %d", ErrInvalidAttentionQuery, token)
		}
		expSum += weight
		for dim := 0; dim < page.Shape.ValueDim; dim++ {
			value, err := decodeTensorFloat(page.DType, page.ValueData, valueElementIndex(page.Shape, localHead, token, dim))
			if err != nil {
				return tensorPartialAttentionReference{}, err
			}
			weightedValue[dim] += weight * float64(value)
			if !finiteFloat64(weightedValue[dim]) {
				return tensorPartialAttentionReference{}, fmt.Errorf("%w: reference weighted value overflow at value_dim %d", ErrInvalidAttentionQuery, dim)
			}
		}
	}
	if !finiteFloat64(localMax) || expSum <= 0 || !finiteFloat64(expSum) {
		return tensorPartialAttentionReference{}, fmt.Errorf("%w: invalid reference summary", ErrInvalidAttentionQuery)
	}
	return tensorPartialAttentionReference{
		LocalMax:      localMax,
		ExpSum:        expSum,
		WeightedValue: weightedValue,
	}, nil
}

func maxAbsDiffVsReference(summary TensorPartialAttentionSummary, reference tensorPartialAttentionReference) float64 {
	maxDiff := math.Abs(summary.LocalMax - reference.LocalMax)
	if diff := math.Abs(summary.ExpSum - reference.ExpSum); diff > maxDiff {
		maxDiff = diff
	}
	for i := 0; i < len(summary.WeightedValue) && i < len(reference.WeightedValue); i++ {
		if diff := math.Abs(summary.WeightedValue[i] - reference.WeightedValue[i]); diff > maxDiff {
			maxDiff = diff
		}
	}
	if len(summary.WeightedValue) != len(reference.WeightedValue) {
		return math.Inf(1)
	}
	return maxDiff
}
