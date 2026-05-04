package modelbench

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestBuildModelBenchmarkReceiptMeasuredMetadataShape(t *testing.T) {
	receipt, err := BuildModelBenchmarkReceipt(validMeasuredModelBenchmarkResult())
	if err != nil {
		t.Fatalf("BuildModelBenchmarkReceipt() error = %v", err)
	}
	if receipt.JobID != "job-modelbench-1" {
		t.Fatalf("job_id = %q", receipt.JobID)
	}
	if len(receipt.ResultHashHex) != 64 {
		t.Fatalf("result_hash_hex = %q, want 64 hex chars", receipt.ResultHashHex)
	}

	metadata := receipt.Metadata[ModelBenchmarkTask].(map[string]any)
	for _, key := range []string{"request_id", "model_id", "prompt_hash", "runtime", "metrics", "output_hash", "output_bytes", "proof_status"} {
		if _, ok := metadata[key]; !ok {
			t.Fatalf("metadata missing %q: %+v", key, metadata)
		}
	}
	if metadata["proof_status"] != string(ModelBenchmarkProofStatusMeasured) {
		t.Fatalf("proof_status = %v", metadata["proof_status"])
	}
	runtimeMeta := metadata["runtime"].(map[string]any)
	if runtimeMeta["runtime_kind"] != "native" {
		t.Fatalf("runtime_kind = %v, want native", runtimeMeta["runtime_kind"])
	}
	if runtimeMeta["model_loaded"] != true {
		t.Fatalf("model_loaded = %v, want true", runtimeMeta["model_loaded"])
	}
	metrics := metadata["metrics"].(map[string]any)
	if metrics["model_load_state"] != "ready" {
		t.Fatalf("model_load_state = %v, want ready", metrics["model_load_state"])
	}
	if _, ok := runtimeMeta["model_id"]; ok {
		t.Fatalf("runtime metadata should not duplicate model_id: %+v", runtimeMeta)
	}
}

func TestBuildModelBenchmarkReceiptDeterministicFromSanitizedMetadata(t *testing.T) {
	result := validMeasuredModelBenchmarkResult()
	first, err := BuildModelBenchmarkReceipt(result)
	if err != nil {
		t.Fatalf("first BuildModelBenchmarkReceipt() error = %v", err)
	}
	second, err := BuildModelBenchmarkReceipt(result)
	if err != nil {
		t.Fatalf("second BuildModelBenchmarkReceipt() error = %v", err)
	}
	if first.ResultHashHex != second.ResultHashHex {
		t.Fatalf("hashes differ: %s vs %s", first.ResultHashHex, second.ResultHashHex)
	}
}

func TestBuildModelBenchmarkReceiptUnavailableOmitsOutputHash(t *testing.T) {
	receipt, err := BuildModelBenchmarkReceipt(validUnavailableModelBenchmarkResult())
	if err != nil {
		t.Fatalf("BuildModelBenchmarkReceipt() error = %v", err)
	}
	metadata := receipt.Metadata[ModelBenchmarkTask].(map[string]any)
	if _, ok := metadata["output_hash"]; ok {
		t.Fatalf("unavailable metadata contains output_hash: %+v", metadata)
	}
	if metadata["output_bytes"] != int64(0) {
		t.Fatalf("output_bytes = %v, want 0", metadata["output_bytes"])
	}
	if metadata["proof_status"] != string(ModelBenchmarkProofStatusUnavailable) {
		t.Fatalf("proof_status = %v, want unavailable", metadata["proof_status"])
	}
}

func TestModelBenchmarkReceiptMetadataContainsNoRawOutputText(t *testing.T) {
	receipt, err := BuildModelBenchmarkReceipt(validMeasuredModelBenchmarkResult())
	if err != nil {
		t.Fatalf("BuildModelBenchmarkReceipt() error = %v", err)
	}
	encoded, err := json.Marshal(receipt.Metadata)
	if err != nil {
		t.Fatalf("json.Marshal(metadata) error = %v", err)
	}
	text := string(encoded)
	for _, forbidden := range []string{
		"ready now",
		"Generate one short readiness sentence",
		"raw output",
		"prompt_text",
		"output_text",
	} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("receipt metadata contains forbidden raw text %q: %s", forbidden, text)
		}
	}
}
