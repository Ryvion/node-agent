package inferencebench

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/Ryvion/node-agent/internal/v7/llamacpp"
)

func TestBuildBenchmarkReceiptContainsMetricsAndHashes(t *testing.T) {
	receipt, err := BuildBenchmarkReceipt(BenchmarkExecutionResult{
		Spec:            validInferenceBenchmarkSpec(),
		Backend:         llamacpp.BackendName,
		ModelID:         "tinyllama.Q4_K_M.gguf",
		PromptHash:      llamacpp.HashBenchmarkPrompt(),
		OutputHash:      "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		OutputBytes:     21,
		TokensGenerated: 7,
		TTFTMs:          50,
		TotalTimeMs:     400,
		DecodeTPS:       20,
		EndToEndTPS:     17.5,
		ProofStatus:     ProofStatusMeasured,
	})
	if err != nil {
		t.Fatalf("BuildBenchmarkReceipt() error = %v", err)
	}
	if receipt.JobID != "job-backend-inference-local" || len(receipt.ResultHashHex) != 64 || receipt.MeteringUnits != 1 {
		t.Fatalf("receipt = %+v, want measured receipt", receipt)
	}
	metadata, ok := receipt.Metadata[BenchmarkTask].(map[string]any)
	if !ok {
		t.Fatalf("metadata missing %q: %+v", BenchmarkTask, receipt.Metadata)
	}
	for _, key := range []string{"backend", "model_id", "prompt_hash", "output_hash", "tokens_generated", "ttft_ms", "total_time_ms", "decode_tps", "end_to_end_tps", "proof_status"} {
		if _, ok := metadata[key]; !ok {
			t.Fatalf("metadata missing %q: %+v", key, metadata)
		}
	}
	assertInferenceBenchmarkReceiptSafe(t, receipt)
}

func TestBuildBenchmarkReceiptRejectsMeasuredWithoutOutputHash(t *testing.T) {
	_, err := BuildBenchmarkReceipt(BenchmarkExecutionResult{
		Spec:            validInferenceBenchmarkSpec(),
		Backend:         llamacpp.BackendName,
		ModelID:         "tinyllama.Q4_K_M.gguf",
		PromptHash:      llamacpp.HashBenchmarkPrompt(),
		OutputBytes:     21,
		TokensGenerated: 7,
		TTFTMs:          50,
		TotalTimeMs:     400,
		DecodeTPS:       20,
		EndToEndTPS:     17.5,
		ProofStatus:     ProofStatusMeasured,
	})
	if err == nil {
		t.Fatal("BuildBenchmarkReceipt() error = nil, want output_hash validation error")
	}
}

func TestBuildBenchmarkRejectionReceiptIsSafe(t *testing.T) {
	receipt := BuildBenchmarkRejectionReceipt("job-rejected", errSensitiveInferenceBenchmark{})
	if receipt.MeteringUnits != 0 || len(receipt.ResultHashHex) != 64 {
		t.Fatalf("receipt = %+v, want rejection receipt", receipt)
	}
	metadata := receipt.Metadata[BenchmarkTask].(map[string]any)
	if metadata["proof_status"] != ProofStatusFailed {
		t.Fatalf("proof_status = %v, want failed", metadata["proof_status"])
	}
	if metadata["error_code"] != "backend_inference_error_redacted" {
		t.Fatalf("error_code = %v, want redacted", metadata["error_code"])
	}
	assertInferenceBenchmarkReceiptSafe(t, receipt, "secret generated_text payload")
}

func assertInferenceBenchmarkReceiptSafe(t *testing.T, receipt BenchmarkReceipt, extraForbidden ...string) {
	t.Helper()
	raw, err := json.Marshal(receipt.Metadata)
	if err != nil {
		t.Fatalf("json.Marshal(metadata) error = %v", err)
	}
	lower := strings.ToLower(string(raw))
	for _, forbidden := range append([]string{
		"write one short sentence",
		"distributed computing",
		"raw_prompt",
		"prompt_text",
		"output_text",
		"generated_text",
		"raw_output",
		"token_logprobs",
		"raw_tensor",
	}, extraForbidden...) {
		if strings.Contains(lower, strings.ToLower(forbidden)) {
			t.Fatalf("metadata leaked forbidden material %q: %s", forbidden, raw)
		}
	}
	if !BenchmarkReceiptJSONContainsNoRawText(receipt) {
		t.Fatalf("BenchmarkReceiptJSONContainsNoRawText() = false: %s", raw)
	}
}

type errSensitiveInferenceBenchmark struct{}

func (errSensitiveInferenceBenchmark) Error() string {
	return "secret generated_text payload"
}
