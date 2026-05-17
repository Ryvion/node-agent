package inferencebench

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/Ryvion/ryvion-node/internal/runtimes/llamacpp"
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
		P95TTFTMs:       55,
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
	for _, key := range []string{"backend", "model_id", "prompt_hash", "output_hash", "tokens_generated", "p50_ttft_ms", "p95_ttft_ms", "p50_decode_tps", "p50_end_to_end_tps", "proof_status"} {
		if _, ok := metadata[key]; !ok {
			t.Fatalf("metadata missing %q: %+v", key, metadata)
		}
	}
	if metadata["prompt_hash"] != validInferenceBenchmarkSpec().PromptHash {
		t.Fatalf("prompt_hash = %v, want spec prompt_hash", metadata["prompt_hash"])
	}
	assertHubInferenceBenchmarkSchemaFixture(t, validInferenceBenchmarkSpec(), metadata)
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
	if metadata["proof_status"] != ProofStatusRejected {
		t.Fatalf("proof_status = %v, want rejected", metadata["proof_status"])
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
		"messages",
		"input_text",
		"model_output",
		"completion",
		"token_logprobs",
		"key_data",
		"value_data",
		"query_vector",
		"tensor_bytes",
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

func assertHubInferenceBenchmarkSchemaFixture(t *testing.T, spec BenchmarkSpec, metadata map[string]any) {
	t.Helper()
	allowed := map[string]bool{
		"request_id":         true,
		"job_id":             true,
		"backend":            true,
		"model_id":           true,
		"prompt_hash":        true,
		"output_hash":        true,
		"tokens_generated":   true,
		"p50_ttft_ms":        true,
		"p95_ttft_ms":        true,
		"p50_decode_tps":     true,
		"p50_end_to_end_tps": true,
		"proof_status":       true,
	}
	for key := range metadata {
		if !allowed[key] {
			t.Fatalf("metadata key %q is not in expected hub schema: %+v", key, metadata)
		}
	}
	if metadata["backend"] != llamacpp.BackendName {
		t.Fatalf("backend = %v, want %q", metadata["backend"], llamacpp.BackendName)
	}
	if metadata["model_id"] != spec.ModelID {
		t.Fatalf("model_id = %v, want %q", metadata["model_id"], spec.ModelID)
	}
	if metadata["prompt_hash"] != spec.PromptHash {
		t.Fatalf("prompt_hash = %v, want spec prompt hash %q", metadata["prompt_hash"], spec.PromptHash)
	}
	if value, ok := metadata["output_hash"].(string); !ok || !strings.HasPrefix(value, "sha256:") || len(value) != len("sha256:")+64 {
		t.Fatalf("output_hash = %#v, want sha256 object id", metadata["output_hash"])
	}
	if tokens, ok := metadata["tokens_generated"].(int64); !ok || tokens <= 0 {
		t.Fatalf("tokens_generated = %#v, want positive int64", metadata["tokens_generated"])
	}
	if metadata["proof_status"] != ProofStatusMeasured {
		t.Fatalf("proof_status = %v, want %q", metadata["proof_status"], ProofStatusMeasured)
	}
	assertNoInferenceBenchmarkForbiddenKeys(t, metadata)
}

func assertNoInferenceBenchmarkForbiddenKeys(t *testing.T, value any) {
	t.Helper()
	forbidden := map[string]bool{
		"prompt":         true,
		"prompt_text":    true,
		"raw_prompt":     true,
		"messages":       true,
		"input_text":     true,
		"output_text":    true,
		"generated_text": true,
		"raw_output":     true,
		"model_output":   true,
		"completion":     true,
		"tokens":         true,
		"key_data":       true,
		"value_data":     true,
		"query_vector":   true,
		"tensor_bytes":   true,
	}
	switch typed := value.(type) {
	case map[string]any:
		for key, nested := range typed {
			if forbidden[key] {
				t.Fatalf("metadata includes forbidden key %q: %+v", key, typed)
			}
			assertNoInferenceBenchmarkForbiddenKeys(t, nested)
		}
	case []any:
		for _, nested := range typed {
			assertNoInferenceBenchmarkForbiddenKeys(t, nested)
		}
	}
}

type errSensitiveInferenceBenchmark struct{}

func (errSensitiveInferenceBenchmark) Error() string {
	return "secret generated_text payload"
}
