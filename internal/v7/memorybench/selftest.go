package memorybench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

const (
	defaultMemoryBenchSelfTestSeed       int64 = 42
	defaultMemoryBenchSelfTestTokenCount       = 256
	defaultMemoryBenchSelfTestValueDim         = 64
	memoryBenchSelfTestShardID                 = "local-selftest"
)

type MemoryBenchSelfTestConfig struct {
	Seed       int64
	TokenCount int
	ValueDim   int
}

type MemoryBenchSelfTestResult struct {
	OK                  bool    `json:"ok"`
	RequestID           string  `json:"request_id"`
	TokenCount          int     `json:"token_count"`
	ValueDim            int     `json:"value_dim"`
	ComputeTimeMs       int64   `json:"compute_time_ms"`
	OutputBytesEstimate int64   `json:"output_bytes_estimate"`
	LocalMax            float64 `json:"local_max"`
	ExpSum              float64 `json:"exp_sum"`
	SummaryHash         string  `json:"summary_hash"`
}

func DefaultMemoryBenchSelfTestConfig() MemoryBenchSelfTestConfig {
	return MemoryBenchSelfTestConfig{
		Seed:       defaultMemoryBenchSelfTestSeed,
		TokenCount: defaultMemoryBenchSelfTestTokenCount,
		ValueDim:   defaultMemoryBenchSelfTestValueDim,
	}
}

func RunMemoryBenchSelfTest(config MemoryBenchSelfTestConfig) (MemoryBenchSelfTestResult, error) {
	spec := memoryBenchSelfTestSpec(config)
	if err := ValidateBenchmarkSpec(spec); err != nil {
		return MemoryBenchSelfTestResult{}, err
	}

	request := GenerateSyntheticAttentionRequest(spec.Seed, spec.ShardID, spec.TokenCount, spec.ValueDim)
	request.RequestID = spec.RequestID
	request.JobID = spec.JobID
	request.ShardID = spec.ShardID
	request.CreatedAtUnixMs = spec.CreatedAtUnixMs

	response, err := ComputePartialAttentionSummary(request)
	if err != nil {
		return MemoryBenchSelfTestResult{}, err
	}
	if response.OutputBytesEstimate <= 0 {
		response.OutputBytesEstimate = estimatePartialAttentionSummaryBytes(response.Summary)
	}

	// Exercise the same local receipt builder used by hub-mediated benchmark assignments.
	if _, err := BuildBenchmarkReceipt(spec, response); err != nil {
		return MemoryBenchSelfTestResult{}, err
	}

	summaryHash, err := hashMemoryBenchSelfTestSummary(response)
	if err != nil {
		return MemoryBenchSelfTestResult{}, err
	}

	return MemoryBenchSelfTestResult{
		OK:                  true,
		RequestID:           response.RequestID,
		TokenCount:          response.Summary.TokenCount,
		ValueDim:            response.Summary.ValueDim,
		ComputeTimeMs:       response.ComputeTimeMs,
		OutputBytesEstimate: response.OutputBytesEstimate,
		LocalMax:            response.Summary.LocalMax,
		ExpSum:              response.Summary.ExpSum,
		SummaryHash:         summaryHash,
	}, nil
}

func FormatMemoryBenchSelfTestResult(result MemoryBenchSelfTestResult, jsonOutput bool) string {
	if jsonOutput {
		encoded, err := json.MarshalIndent(result, "", "  ")
		if err != nil {
			return `{"ok":false,"error":"format memorybench self-test result"}`
		}
		return string(encoded)
	}

	return fmt.Sprintf(
		"ok: %t\nrequest_id: %s\ntoken_count: %d\nvalue_dim: %d\ncompute_time_ms: %d\noutput_bytes_estimate: %d\nlocal_max: %.17g\nexp_sum: %.17g\nsummary_hash: %s",
		result.OK,
		result.RequestID,
		result.TokenCount,
		result.ValueDim,
		result.ComputeTimeMs,
		result.OutputBytesEstimate,
		result.LocalMax,
		result.ExpSum,
		result.SummaryHash,
	)
}

func hashMemoryBenchSelfTestSummary(response SyntheticAttentionResponse) (string, error) {
	payload := struct {
		RequestID           string    `json:"request_id"`
		TokenCount          int       `json:"token_count"`
		ValueDim            int       `json:"value_dim"`
		DType               string    `json:"dtype"`
		LocalMax            float64   `json:"local_max"`
		ExpSum              float64   `json:"exp_sum"`
		WeightedValue       []float64 `json:"weighted_value"`
		OutputBytesEstimate int64     `json:"output_bytes_estimate"`
	}{
		RequestID:           response.RequestID,
		TokenCount:          response.Summary.TokenCount,
		ValueDim:            response.Summary.ValueDim,
		DType:               response.Summary.DType,
		LocalMax:            response.Summary.LocalMax,
		ExpSum:              response.Summary.ExpSum,
		WeightedValue:       append([]float64(nil), response.Summary.WeightedValue...),
		OutputBytesEstimate: response.OutputBytesEstimate,
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:]), nil
}

func memoryBenchSelfTestSpec(config MemoryBenchSelfTestConfig) BenchmarkSpec {
	return BenchmarkSpec{
		Task:             BenchmarkTask,
		RequestID:        fmt.Sprintf("memorybench-selftest-%d-%d-%d", config.Seed, config.TokenCount, config.ValueDim),
		JobID:            fmt.Sprintf("memorybench-selftest-job-%d", config.Seed),
		ShardID:          memoryBenchSelfTestShardID,
		Seed:             config.Seed,
		TokenCount:       config.TokenCount,
		ValueDim:         config.ValueDim,
		SimulatedDelayMs: 0,
		CreatedAtUnixMs:  syntheticCreatedAtUnixMs(config.Seed),
	}
}
