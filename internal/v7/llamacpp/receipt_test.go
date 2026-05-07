package llamacpp

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestBuildBackendBenchmarkReceiptContainsHashesAndMetrics(t *testing.T) {
	spec, err := DecodeBackendBenchmarkSpec(testBackendBenchmarkSpecJSON(t))
	if err != nil {
		t.Fatalf("DecodeBackendBenchmarkSpec() error = %v", err)
	}
	receipt, err := BuildBackendBenchmarkReceipt(spec, testBackendBenchmarkSnapshot(BenchmarkProofStatusMeasured))
	if err != nil {
		t.Fatalf("BuildBackendBenchmarkReceipt() error = %v", err)
	}
	if receipt.JobID != spec.JobID || len(receipt.ResultHashHex) != 64 || receipt.MeteringUnits != 1 {
		t.Fatalf("receipt envelope = %+v", receipt)
	}
	metadata, ok := receipt.Metadata[BackendBenchmarkTask].(map[string]any)
	if !ok {
		t.Fatalf("metadata missing %q: %+v", BackendBenchmarkTask, receipt.Metadata)
	}
	for _, key := range []string{"prompt_hash", "output_hash"} {
		value, ok := metadata[key].(string)
		if !ok || !strings.HasPrefix(value, "sha256:") || len(value) != len("sha256:")+64 {
			t.Fatalf("%s = %#v, want sha256 hash", key, metadata[key])
		}
	}
	if metadata["p50_ttft_ms"] != int64(100) || metadata["p95_total_time_ms"] != int64(1100) {
		t.Fatalf("timing metadata = %+v", metadata)
	}
	if metadata["p50_decode_tps"] != 20.5 || metadata["p95_end_to_end_tps"] != 19.75 {
		t.Fatalf("tps metadata = %+v", metadata)
	}
	assertBackendBenchmarkReceiptJSONSafe(t, receipt, "secret llama output")
}

func TestBuildBackendBenchmarkReceiptRejectsMeasuredWithoutOutputHash(t *testing.T) {
	spec, err := DecodeBackendBenchmarkSpec(testBackendBenchmarkSpecJSON(t))
	if err != nil {
		t.Fatalf("DecodeBackendBenchmarkSpec() error = %v", err)
	}
	snapshot := testBackendBenchmarkSnapshot(BenchmarkProofStatusMeasured)
	snapshot.Metrics.OutputHash = ""
	_, err = BuildBackendBenchmarkReceipt(spec, snapshot)
	if err == nil {
		t.Fatal("BuildBackendBenchmarkReceipt() error = nil, want output_hash validation error")
	}
}

func TestBuildBackendBenchmarkRejectionReceiptUsesSafeMetadata(t *testing.T) {
	receipt := BuildBackendBenchmarkRejectionReceipt("job-rejected", errTestBackendBenchmarkFailure{})
	if receipt.JobID != "job-rejected" || receipt.MeteringUnits != 0 || len(receipt.ResultHashHex) != 64 {
		t.Fatalf("receipt = %+v", receipt)
	}
	metadata := receipt.Metadata[BackendBenchmarkTask].(map[string]any)
	if metadata["proof_status"] != BenchmarkProofStatusFailed {
		t.Fatalf("proof_status = %v", metadata["proof_status"])
	}
	assertBackendBenchmarkReceiptJSONSafe(t, receipt)
}

func assertBackendBenchmarkReceiptJSONSafe(t *testing.T, receipt BackendBenchmarkReceipt, extraForbidden ...string) {
	t.Helper()
	raw, err := json.Marshal(receipt.Metadata)
	if err != nil {
		t.Fatalf("json.Marshal(metadata) error = %v", err)
	}
	lower := strings.ToLower(string(raw))
	for _, forbidden := range append([]string{
		internalBenchmarkPrompt,
		"output_text",
		"generated_text",
		"raw_output",
		"logprobs",
		"token_logprobs",
		"tensor",
		"raw_tensor",
	}, extraForbidden...) {
		if strings.Contains(lower, strings.ToLower(forbidden)) {
			t.Fatalf("receipt metadata contains forbidden text %q: %s", forbidden, raw)
		}
	}
	if !BackendBenchmarkReceiptJSONContainsNoRawText(receipt) {
		t.Fatalf("BackendBenchmarkReceiptJSONContainsNoRawText() = false: %s", raw)
	}
}
