package memorybench

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestRunMemoryBenchSelfTestDefaultSucceeds(t *testing.T) {
	result, err := RunMemoryBenchSelfTest(DefaultMemoryBenchSelfTestConfig())
	if err != nil {
		t.Fatalf("RunMemoryBenchSelfTest() error = %v", err)
	}
	if !result.OK {
		t.Fatal("ok = false, want true")
	}
	if result.RequestID != "memorybench-selftest-42-256-64" {
		t.Fatalf("request_id = %q, want deterministic default id", result.RequestID)
	}
	if result.TokenCount != 256 {
		t.Fatalf("token_count = %d, want 256", result.TokenCount)
	}
	if result.ValueDim != 64 {
		t.Fatalf("value_dim = %d, want 64", result.ValueDim)
	}
	if result.ComputeTimeMs < 0 {
		t.Fatalf("compute_time_ms = %d, want non-negative", result.ComputeTimeMs)
	}
	if result.OutputBytesEstimate <= 0 {
		t.Fatalf("output_bytes_estimate = %d, want positive", result.OutputBytesEstimate)
	}
	if result.ExpSum <= 0 {
		t.Fatalf("exp_sum = %v, want positive", result.ExpSum)
	}
	if len(result.SummaryHash) != 64 {
		t.Fatalf("summary_hash length = %d, want 64", len(result.SummaryHash))
	}
}

func TestRunMemoryBenchSelfTestDeterministicForSameSeed(t *testing.T) {
	config := MemoryBenchSelfTestConfig{
		Seed:       99,
		TokenCount: 32,
		ValueDim:   8,
	}

	first, err := RunMemoryBenchSelfTest(config)
	if err != nil {
		t.Fatalf("first RunMemoryBenchSelfTest() error = %v", err)
	}
	second, err := RunMemoryBenchSelfTest(config)
	if err != nil {
		t.Fatalf("second RunMemoryBenchSelfTest() error = %v", err)
	}
	if first.RequestID != second.RequestID {
		t.Fatalf("request ids differ: %q vs %q", first.RequestID, second.RequestID)
	}
	if first.LocalMax != second.LocalMax {
		t.Fatalf("local_max differs: %v vs %v", first.LocalMax, second.LocalMax)
	}
	if first.ExpSum != second.ExpSum {
		t.Fatalf("exp_sum differs: %v vs %v", first.ExpSum, second.ExpSum)
	}
	if first.SummaryHash != second.SummaryHash {
		t.Fatalf("summary_hash differs: %s vs %s", first.SummaryHash, second.SummaryHash)
	}
}

func TestRunMemoryBenchSelfTestRejectsInvalidTokenCount(t *testing.T) {
	_, err := RunMemoryBenchSelfTest(MemoryBenchSelfTestConfig{
		Seed:       42,
		TokenCount: 0,
		ValueDim:   64,
	})
	if !errors.Is(err, ErrInvalidBenchmarkSpec) || !strings.Contains(err.Error(), "token_count") {
		t.Fatalf("error = %v, want token_count invalid spec", err)
	}
}

func TestRunMemoryBenchSelfTestRejectsInvalidValueDim(t *testing.T) {
	_, err := RunMemoryBenchSelfTest(MemoryBenchSelfTestConfig{
		Seed:       42,
		TokenCount: 256,
		ValueDim:   0,
	})
	if !errors.Is(err, ErrInvalidBenchmarkSpec) || !strings.Contains(err.Error(), "value_dim") {
		t.Fatalf("error = %v, want value_dim invalid spec", err)
	}
}

func TestFormatMemoryBenchSelfTestResultJSONValid(t *testing.T) {
	result, err := RunMemoryBenchSelfTest(MemoryBenchSelfTestConfig{
		Seed:       7,
		TokenCount: 16,
		ValueDim:   4,
	})
	if err != nil {
		t.Fatalf("RunMemoryBenchSelfTest() error = %v", err)
	}

	var decoded MemoryBenchSelfTestResult
	if err := json.Unmarshal([]byte(FormatMemoryBenchSelfTestResult(result, true)), &decoded); err != nil {
		t.Fatalf("JSON output is invalid: %v", err)
	}
	if decoded.RequestID != result.RequestID {
		t.Fatalf("decoded request_id = %q, want %q", decoded.RequestID, result.RequestID)
	}
	if decoded.SummaryHash != result.SummaryHash {
		t.Fatalf("decoded summary_hash = %q, want %q", decoded.SummaryHash, result.SummaryHash)
	}
}

func TestFormatMemoryBenchSelfTestResultReadableContainsRequiredFields(t *testing.T) {
	result, err := RunMemoryBenchSelfTest(MemoryBenchSelfTestConfig{
		Seed:       7,
		TokenCount: 16,
		ValueDim:   4,
	})
	if err != nil {
		t.Fatalf("RunMemoryBenchSelfTest() error = %v", err)
	}

	formatted := FormatMemoryBenchSelfTestResult(result, false)
	for _, key := range []string{
		"ok:",
		"request_id:",
		"token_count:",
		"value_dim:",
		"compute_time_ms:",
		"output_bytes_estimate:",
		"local_max:",
		"exp_sum:",
		"summary_hash:",
	} {
		if !strings.Contains(formatted, key) {
			t.Fatalf("readable output missing %q: %s", key, formatted)
		}
	}
}
