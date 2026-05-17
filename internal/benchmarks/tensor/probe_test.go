package tensorplane

import (
	"encoding/json"
	"path/filepath"
	"testing"
)

func TestTensorPlaneProbeWriteThenReadFixtureSameSummaryHash(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tensorplane-fixture.json")
	config := testTensorPlaneProbeConfig(TensorDTypeFloat32)
	config.WriteFixturePath = path

	written, err := RunTensorPlaneProbe(config)
	if err != nil {
		t.Fatalf("write RunTensorPlaneProbe() error = %v", err)
	}

	readConfig := testTensorPlaneProbeConfig(TensorDTypeFloat32)
	readConfig.ReadFixturePath = path
	read, err := RunTensorPlaneProbe(readConfig)
	if err != nil {
		t.Fatalf("read RunTensorPlaneProbe() error = %v", err)
	}

	if written.SummaryHash != read.SummaryHash {
		t.Fatalf("summary_hash differs write/read: %s vs %s", written.SummaryHash, read.SummaryHash)
	}
	if written.PageHash != read.PageHash {
		t.Fatalf("page_hash differs write/read: %s vs %s", written.PageHash, read.PageHash)
	}
}

func TestTensorPlaneProbeFloat32SelfTestPasses(t *testing.T) {
	result, err := RunTensorPlaneProbe(testTensorPlaneProbeConfig(TensorDTypeFloat32))
	if err != nil {
		t.Fatalf("RunTensorPlaneProbe(float32) error = %v", err)
	}
	assertProbeResultValid(t, result, TensorDTypeFloat32)
	if result.MaxAbsDiffVsReference > TensorPlaneProbeTolerance(TensorDTypeFloat32) {
		t.Fatalf("max_abs_diff_vs_reference = %.17g", result.MaxAbsDiffVsReference)
	}
}

func TestTensorPlaneProbeFloat16SelfTestPassesWithinTolerance(t *testing.T) {
	result, err := RunTensorPlaneProbe(testTensorPlaneProbeConfig(TensorDTypeFloat16))
	if err != nil {
		t.Fatalf("RunTensorPlaneProbe(float16) error = %v", err)
	}
	assertProbeResultValid(t, result, TensorDTypeFloat16)
	if result.MaxAbsDiffVsReference > TensorPlaneProbeTolerance(TensorDTypeFloat16) {
		t.Fatalf("max_abs_diff_vs_reference = %.17g", result.MaxAbsDiffVsReference)
	}
}

func TestTensorPlaneProbeResultHasNoRawPromptOrOutputFields(t *testing.T) {
	result, err := RunTensorPlaneProbe(testTensorPlaneProbeConfig(TensorDTypeFloat32))
	if err != nil {
		t.Fatalf("RunTensorPlaneProbe() error = %v", err)
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	formatted := FormatTensorPlaneProbeResult(result, true)

	assertNoRawPromptOrOutputKeys(t, encoded)
	assertNoRawPromptOrOutputKeys(t, []byte(formatted))
}

func TestTensorPlaneProbeRejectsInvalidDType(t *testing.T) {
	config := testTensorPlaneProbeConfig(TensorDType("bfloat16"))
	if _, err := RunTensorPlaneProbe(config); err == nil {
		t.Fatal("RunTensorPlaneProbe() error = nil, want unsupported dtype rejection")
	}
}

func assertProbeResultValid(t *testing.T, result TensorPlaneProbeResult, dtype TensorDType) {
	t.Helper()
	if !result.OK {
		t.Fatal("result OK = false")
	}
	if result.DType != dtype {
		t.Fatalf("dtype = %s, want %s", result.DType, dtype)
	}
	if result.Tokens != 16 || result.HeadDim != 8 || result.ValueDim != 6 {
		t.Fatalf("dimensions = tokens %d head_dim %d value_dim %d, want 16/8/6", result.Tokens, result.HeadDim, result.ValueDim)
	}
	if result.SummaryHash == "" || result.PageHash == "" {
		t.Fatalf("hashes missing: summary=%q page=%q", result.SummaryHash, result.PageHash)
	}
	if result.PayloadBytesEstimate <= 0 {
		t.Fatalf("payload_bytes_estimate = %d, want > 0", result.PayloadBytesEstimate)
	}
}

func testTensorPlaneProbeConfig(dtype TensorDType) TensorPlaneProbeConfig {
	config := DefaultTensorPlaneProbeConfig()
	config.Tokens = 16
	config.HeadDim = 8
	config.ValueDim = 6
	config.DType = dtype
	config.Seed = 123
	return config
}
