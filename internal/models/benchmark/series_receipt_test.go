package modelbench

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func TestBuildModelBenchmarkSeriesReceiptMetadataShape(t *testing.T) {
	result := measuredModelBenchmarkSeriesResult(t)

	receipt, err := BuildModelBenchmarkSeriesReceipt(result)
	if err != nil {
		t.Fatalf("BuildModelBenchmarkSeriesReceipt() error = %v", err)
	}
	if receipt.JobID != result.JobID {
		t.Fatalf("job_id = %q, want %q", receipt.JobID, result.JobID)
	}
	if len(receipt.ResultHashHex) != 64 {
		t.Fatalf("result_hash_hex = %q, want 64 hex chars", receipt.ResultHashHex)
	}

	metadata := receipt.Metadata[ModelBenchmarkSeriesTask].(map[string]any)
	for _, key := range []string{"request_id", "job_id", "model_id", "prompt_profile_id", "prompt_hash", "warmup_runs", "measured_runs", "runtime", "trials", "summary", "proof_status"} {
		if _, ok := metadata[key]; !ok {
			t.Fatalf("metadata missing %q: %+v", key, metadata)
		}
	}
	if metadata["proof_status"] != ModelBenchmarkSeriesProofStatusMeasured {
		t.Fatalf("proof_status = %v, want measured series", metadata["proof_status"])
	}
	trials := metadata["trials"].([]map[string]any)
	if len(trials) != result.WarmupRuns+result.MeasuredRuns {
		t.Fatalf("trial count = %d, want %d", len(trials), result.WarmupRuns+result.MeasuredRuns)
	}
	if _, ok := trials[0]["error_message"]; ok {
		t.Fatalf("trial metadata should not include error_message: %+v", trials[0])
	}
	runtimeMeta := metadata["runtime"].(map[string]any)
	if runtimeMeta["agent_version"] != result.Runtime.AgentVersion ||
		runtimeMeta["os"] != result.Runtime.OS ||
		runtimeMeta["arch"] != result.Runtime.Arch ||
		runtimeMeta["native_inference_supported"] != result.Runtime.NativeInferenceSupported ||
		runtimeMeta["native_inference_ready"] != result.Runtime.NativeInferenceReady ||
		runtimeMeta["runtime_kind"] != result.Runtime.RuntimeKind ||
		runtimeMeta["model_loaded"] != result.Runtime.ModelLoaded {
		t.Fatalf("runtime metadata = %+v, want %+v", runtimeMeta, result.Runtime)
	}
}

func TestBuildModelBenchmarkSeriesReceiptWarmupExcludedFromSummary(t *testing.T) {
	spec := validModelBenchmarkSeriesSpec()
	spec.WarmupRuns = 1
	spec.MeasuredRuns = 2
	runner := &fakeSeriesRunner{
		results: []ModelBenchmarkResult{
			seriesRunResult(ModelBenchmarkProofStatusMeasured, 9_999, 10_999, 99, 99, ""),
			seriesRunResult(ModelBenchmarkProofStatusMeasured, 100, 1_100, 11, 10, ""),
			seriesRunResult(ModelBenchmarkProofStatusMeasured, 200, 1_200, 11, 9.17, ""),
		},
	}
	result, err := RunModelBenchmarkSeries(context.Background(), runner, spec)
	if err != nil {
		t.Fatalf("RunModelBenchmarkSeries() error = %v", err)
	}

	receipt, err := BuildModelBenchmarkSeriesReceipt(result)
	if err != nil {
		t.Fatalf("BuildModelBenchmarkSeriesReceipt() error = %v", err)
	}
	summary := receipt.Metadata[ModelBenchmarkSeriesTask].(map[string]any)["summary"].(map[string]any)
	if summary["p50_ttft_ms"] != int64(100) || summary["p95_ttft_ms"] != int64(200) {
		t.Fatalf("summary includes warmup outlier: %+v", summary)
	}
	if summary["successful_measured_runs"] != 2 {
		t.Fatalf("successful_measured_runs = %v, want 2", summary["successful_measured_runs"])
	}
}

func TestBuildModelBenchmarkSeriesReceiptDeterministicFromSanitizedMetadata(t *testing.T) {
	result := measuredModelBenchmarkSeriesResult(t)
	first, err := BuildModelBenchmarkSeriesReceipt(result)
	if err != nil {
		t.Fatalf("first BuildModelBenchmarkSeriesReceipt() error = %v", err)
	}
	second, err := BuildModelBenchmarkSeriesReceipt(result)
	if err != nil {
		t.Fatalf("second BuildModelBenchmarkSeriesReceipt() error = %v", err)
	}
	if first.ResultHashHex != second.ResultHashHex {
		t.Fatalf("hashes differ: %s vs %s", first.ResultHashHex, second.ResultHashHex)
	}
}

func TestModelBenchmarkSeriesReceiptRuntimeSurvivesJSONRoundTrip(t *testing.T) {
	result := measuredModelBenchmarkSeriesResult(t)
	metadata := modelBenchmarkSeriesReceiptMetadataFromResult(result)

	encoded, err := json.Marshal(metadata)
	if err != nil {
		t.Fatalf("json.Marshal(metadata) error = %v", err)
	}
	var decoded ModelBenchmarkSeriesReceiptMetadata
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("json.Unmarshal(metadata) error = %v", err)
	}
	if decoded.Runtime != result.Runtime {
		t.Fatalf("decoded runtime = %+v, want %+v", decoded.Runtime, result.Runtime)
	}
}

func TestModelBenchmarkSeriesReceiptMetadataContainsNoRawOutputText(t *testing.T) {
	receipt, err := BuildModelBenchmarkSeriesReceipt(measuredModelBenchmarkSeriesResult(t))
	if err != nil {
		t.Fatalf("BuildModelBenchmarkSeriesReceipt() error = %v", err)
	}
	encoded, err := json.Marshal(receipt.Metadata)
	if err != nil {
		t.Fatalf("json.Marshal(metadata) error = %v", err)
	}
	text := string(encoded)
	for _, forbidden := range []string{
		"sensitive raw output",
		"Continue the numbered sequence",
		"Reply with exactly: ready.",
		"prompt_text",
		"output_text",
		"raw_output",
		"error_message",
	} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("receipt metadata contains forbidden raw text %q: %s", forbidden, text)
		}
	}
}

func measuredModelBenchmarkSeriesResult(t *testing.T) ModelBenchmarkSeriesResult {
	t.Helper()
	spec := validModelBenchmarkSeriesSpec()
	spec.WarmupRuns = 1
	spec.MeasuredRuns = 3
	runner := &fakeSeriesRunner{
		results: []ModelBenchmarkResult{
			seriesRunResult(ModelBenchmarkProofStatusMeasured, 50, 1_050, 11, 10.48, ""),
			seriesRunResult(ModelBenchmarkProofStatusMeasured, 100, 1_100, 11, 10, ""),
			seriesRunResult(ModelBenchmarkProofStatusMeasured, 200, 1_200, 11, 9.17, ""),
			seriesRunResult(ModelBenchmarkProofStatusMeasured, 300, 1_300, 11, 8.46, ""),
		},
	}
	result, err := RunModelBenchmarkSeries(context.Background(), runner, spec)
	if err != nil {
		t.Fatalf("RunModelBenchmarkSeries() error = %v", err)
	}
	return result
}
