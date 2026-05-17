package tensorplane

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"
)

func TestBuildBenchmarkReceiptMetadataContainsHashesAndSummary(t *testing.T) {
	result, err := RunBenchmarkSpec(context.Background(), validTensorPlaneBenchmarkSpec())
	if err != nil {
		t.Fatalf("RunBenchmarkSpec() error = %v", err)
	}
	receipt, err := BuildBenchmarkReceipt(result)
	if err != nil {
		t.Fatalf("BuildBenchmarkReceipt() error = %v", err)
	}
	metadata := receipt.Metadata[BenchmarkTask].(map[string]any)
	for _, key := range []string{
		"request_id",
		"job_id",
		"model_id",
		"layer_index",
		"dtype",
		"tokens",
		"head_dim",
		"value_dim",
		"page_hash",
		"query_hash",
		"summary_hash",
		"local_max",
		"exp_sum",
		"weighted_value",
		"weighted_value_length",
		"compute_time_us",
		"payload_bytes_estimate",
		"max_abs_diff_vs_reference",
		"correctness_status",
		"proof_status",
	} {
		if _, ok := metadata[key]; !ok {
			t.Fatalf("metadata missing key %q: %+v", key, metadata)
		}
	}
	for _, key := range []string{"page_hash", "query_hash", "summary_hash"} {
		hash, ok := metadata[key].(string)
		if !ok || !strings.HasPrefix(hash, "sha256:") || len(strings.TrimPrefix(hash, "sha256:")) != 64 {
			t.Fatalf("metadata[%q] = %#v, want sha256:<64 hex>", key, metadata[key])
		}
	}
	if metadata["weighted_value_length"] != result.Spec.ValueDim {
		t.Fatalf("weighted_value_length = %v, want %d", metadata["weighted_value_length"], result.Spec.ValueDim)
	}
	weighted, ok := metadata["weighted_value"].([]float64)
	if !ok || len(weighted) != result.Spec.ValueDim {
		t.Fatalf("weighted_value = %T len %d, want []float64 len %d", metadata["weighted_value"], len(weighted), result.Spec.ValueDim)
	}
	if metadata["correctness_status"] != CorrectnessStatusMatched {
		t.Fatalf("correctness_status = %v", metadata["correctness_status"])
	}
	if metadata["proof_status"] != ProofStatusTensorPlaneMeasured {
		t.Fatalf("proof_status = %v", metadata["proof_status"])
	}
}

func TestBenchmarkReceiptDoesNotExposeRawTensorBytesOrQueryVector(t *testing.T) {
	result, err := RunBenchmarkSpec(context.Background(), validTensorPlaneBenchmarkSpec())
	if err != nil {
		t.Fatalf("RunBenchmarkSpec() error = %v", err)
	}
	receipt, err := BuildBenchmarkReceipt(result)
	if err != nil {
		t.Fatalf("BuildBenchmarkReceipt() error = %v", err)
	}
	encoded, err := json.Marshal(receipt.Metadata)
	if err != nil {
		t.Fatalf("json.Marshal(metadata) error = %v", err)
	}
	text := string(encoded)
	for _, forbidden := range []string{
		"key_data",
		"value_data",
		"query_vector",
		"prompt",
		"generated_output",
		base64.StdEncoding.EncodeToString(result.Page.KeyData),
		base64.StdEncoding.EncodeToString(result.Page.ValueData),
	} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("receipt metadata leaked forbidden material %q: %s", forbidden, text)
		}
	}
}

func TestHashBenchmarkReceiptMetadataIgnoresMeasuredComputeTime(t *testing.T) {
	result, err := RunBenchmarkSpec(context.Background(), validTensorPlaneBenchmarkSpec())
	if err != nil {
		t.Fatalf("RunBenchmarkSpec() error = %v", err)
	}
	receipt, err := BuildBenchmarkReceipt(result)
	if err != nil {
		t.Fatalf("BuildBenchmarkReceipt() error = %v", err)
	}
	metadata := BenchmarkReceiptMetadata{
		RequestID:             result.Spec.RequestID,
		JobID:                 result.Spec.JobID,
		ModelID:               result.Spec.ModelID,
		LayerIndex:            result.Spec.LayerIndex,
		DType:                 result.Spec.DType,
		Tokens:                result.Spec.Tokens,
		HeadDim:               result.Spec.HeadDim,
		ValueDim:              result.Spec.ValueDim,
		PageHash:              receipt.Metadata[BenchmarkTask].(map[string]any)["page_hash"].(string),
		QueryHash:             receipt.Metadata[BenchmarkTask].(map[string]any)["query_hash"].(string),
		SummaryHash:           receipt.Metadata[BenchmarkTask].(map[string]any)["summary_hash"].(string),
		LocalMax:              result.Summary.LocalMax,
		ExpSum:                result.Summary.ExpSum,
		WeightedValue:         append([]float64(nil), result.Summary.WeightedValue...),
		WeightedValueLength:   len(result.Summary.WeightedValue),
		ComputeTimeUs:         result.Summary.ComputeTimeUs,
		PayloadBytesEstimate:  result.Summary.PayloadBytesEstimate,
		MaxAbsDiffVsReference: result.MaxAbsDiffVsReference,
		CorrectnessStatus:     result.CorrectnessStatus,
		ProofStatus:           ProofStatusTensorPlaneMeasured,
	}
	first, err := HashBenchmarkReceiptMetadata(result.Spec.JobID, metadata)
	if err != nil {
		t.Fatalf("first HashBenchmarkReceiptMetadata() error = %v", err)
	}
	metadata.ComputeTimeUs += 1_000_000
	second, err := HashBenchmarkReceiptMetadata(result.Spec.JobID, metadata)
	if err != nil {
		t.Fatalf("second HashBenchmarkReceiptMetadata() error = %v", err)
	}
	if first != second {
		t.Fatalf("result hash changed with compute_time_us: %s vs %s", first, second)
	}
}
