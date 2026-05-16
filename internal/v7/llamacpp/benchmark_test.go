package llamacpp

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestBenchmarkComputesStreamingMetricsDeterministically(t *testing.T) {
	client := &queuedCompletionClient{steps: []completionStep{
		{result: completionResult("warmup text", 4, 90, 490, true)},
		{result: completionResult("alpha measured", 10, 100, 1100, true)},
		{result: completionResult("beta measured", 10, 200, 700, true)},
		{result: completionResult("gamma measured", 5, 50, 300, true)},
	}}
	runner := BenchmarkRunner{
		Sidecar: healthyBenchmarkSidecar(),
		Client:  client,
		Now:     fixedBenchmarkNow,
	}

	snapshot := runner.Run(context.Background(), BenchmarkConfig{
		NodeID:              "node-a",
		MaxTokens:           8,
		TimeoutMs:           10_000,
		Streaming:           true,
		WarmupRuns:          1,
		MeasuredRuns:        3,
		Acceleration:        "cuda",
		Warm:                true,
		ContextLengthTokens: 4096,
		StreamingSupported:  true,
	})
	if snapshot.Status != BenchmarkStatusCompleted {
		t.Fatalf("status = %q, want completed: %+v", snapshot.Status, snapshot)
	}
	metrics := snapshot.Metrics
	if !metrics.Available || !metrics.SidecarHealthy || !metrics.ModelLoaded {
		t.Fatalf("availability metrics = %+v, want available healthy loaded", metrics)
	}
	if metrics.PromptHash == "" || metrics.OutputHash == "" {
		t.Fatalf("hashes missing: %+v", metrics)
	}
	if metrics.OutputBytes != int64(len("alpha measured")+len("beta measured")+len("gamma measured")) {
		t.Fatalf("output_bytes = %d, want measured output bytes only", metrics.OutputBytes)
	}
	if metrics.TokensGenerated != 25 {
		t.Fatalf("tokens_generated = %d, want 25", metrics.TokensGenerated)
	}
	if metrics.P50TTFTMs != 100 || metrics.P95TTFTMs != 200 {
		t.Fatalf("ttft percentiles = %d/%d, want 100/200", metrics.P50TTFTMs, metrics.P95TTFTMs)
	}
	if metrics.P50TotalTimeMs != 700 || metrics.P95TotalTimeMs != 1100 {
		t.Fatalf("total percentiles = %d/%d, want 700/1100", metrics.P50TotalTimeMs, metrics.P95TotalTimeMs)
	}
	if metrics.P50DecodeTPS != 20 || metrics.P95DecodeTPS != 20 {
		t.Fatalf("decode tps percentiles = %.3f/%.3f, want 20/20", metrics.P50DecodeTPS, metrics.P95DecodeTPS)
	}
	if metrics.P50TPOTMs != 50 || metrics.P95TPOTMs != 50 {
		t.Fatalf("tpot percentiles = %.3f/%.3f, want 50/50", metrics.P50TPOTMs, metrics.P95TPOTMs)
	}
	if metrics.P50EndToEndTPS != 14.286 || metrics.P95EndToEndTPS != 16.667 {
		t.Fatalf("end-to-end tps percentiles = %.3f/%.3f, want 14.286/16.667", metrics.P50EndToEndTPS, metrics.P95EndToEndTPS)
	}
	if metrics.NodeID != "node-a" || metrics.Acceleration != "cuda" || !metrics.Warm ||
		metrics.ContextLengthBucket != "ctx_2k_4k" || metrics.OutputTokenBucket != "out_0_64" || !metrics.StreamingSupported {
		t.Fatalf("profile metrics = %+v", metrics)
	}
	if metrics.Backend != BackendName || metrics.RuntimeKind != BackendName || metrics.ProofStatus != BenchmarkProofStatusMeasured {
		t.Fatalf("backend/proof metrics = %+v", metrics)
	}
	assertBenchmarkJSONSafe(t, snapshot, "warmup text", "alpha measured", "beta measured", "gamma measured")
}

func TestBenchmarkFallsBackToNonStreamingWhenSSEUnavailable(t *testing.T) {
	client := &queuedCompletionClient{steps: []completionStep{
		{err: ClientError{Code: "llamacpp_stream_unavailable", StatusCode: 400}},
		{result: completionResult("fallback measured", 5, 500, 500, false)},
	}}
	runner := BenchmarkRunner{
		Sidecar: healthyBenchmarkSidecar(),
		Client:  client,
		Now:     fixedBenchmarkNow,
	}

	snapshot := runner.Run(context.Background(), BenchmarkConfig{
		MaxTokens:    8,
		TimeoutMs:    10_000,
		Streaming:    true,
		WarmupRuns:   0,
		MeasuredRuns: 1,
	})
	if snapshot.Status != BenchmarkStatusCompleted {
		t.Fatalf("status = %q, want completed: %+v", snapshot.Status, snapshot)
	}
	if len(client.requests) != 2 {
		t.Fatalf("requests = %d, want stream attempt plus fallback", len(client.requests))
	}
	if !client.requests[0].Stream || client.requests[1].Stream {
		t.Fatalf("request stream flags = %v/%v, want true/false", client.requests[0].Stream, client.requests[1].Stream)
	}
	if snapshot.Metrics.Streaming {
		t.Fatalf("metrics.streaming = true, want false after fallback")
	}
	if snapshot.Metrics.P50DecodeTPS != 10 {
		t.Fatalf("p50_decode_tps = %.3f, want 10", snapshot.Metrics.P50DecodeTPS)
	}
	assertBenchmarkJSONSafe(t, snapshot, "fallback measured")
}

func TestMeasurementPrefersBackendDecodeTiming(t *testing.T) {
	measurement := measurementFromCompletion(CompletionResult{
		Output:           []byte("measured output"),
		TokensGenerated:  25,
		TTFTMs:           29,
		TotalTimeMs:      795,
		BackendDecodeTPS: 45.05876564215049,
	})

	if measurement.decodeTPS != 45.05876564215049 {
		t.Fatalf("decodeTPS = %.12f, want backend decode timing", measurement.decodeTPS)
	}
	if measurement.endToEndTPS == measurement.decodeTPS {
		t.Fatalf("endToEndTPS should remain wall-clock based, got %.12f", measurement.endToEndTPS)
	}
}

func TestBenchmarkUsesNonStreamingDecodeProbeWhenStreamingDecodeIsSlower(t *testing.T) {
	stream := completionResult("stream measured", 25, 29, 795, true)
	stream.BackendDecodeMs = 758
	stream.BackendDecodeTPS = 33
	probe := completionResult("probe measured", 25, 795, 795, false)
	probe.BackendDecodeMs = 555
	probe.BackendDecodeTPS = 45
	client := &queuedCompletionClient{steps: []completionStep{
		{result: stream},
		{result: probe},
	}}
	runner := BenchmarkRunner{
		Sidecar: healthyBenchmarkSidecar(),
		Client:  client,
		Now:     fixedBenchmarkNow,
	}

	snapshot := runner.Run(context.Background(), BenchmarkConfig{
		MaxTokens:    32,
		TimeoutMs:    10_000,
		Streaming:    true,
		WarmupRuns:   0,
		MeasuredRuns: 1,
	})
	if snapshot.Status != BenchmarkStatusCompleted {
		t.Fatalf("status = %q, want completed: %+v", snapshot.Status, snapshot)
	}
	if len(client.requests) != 2 {
		t.Fatalf("requests = %d, want streaming measurement plus non-streaming probe", len(client.requests))
	}
	if !client.requests[0].Stream || client.requests[1].Stream {
		t.Fatalf("request stream flags = %v/%v, want true/false", client.requests[0].Stream, client.requests[1].Stream)
	}
	if snapshot.Metrics.P50DecodeTPS != 45 {
		t.Fatalf("p50_decode_tps = %.3f, want non-streaming backend probe", snapshot.Metrics.P50DecodeTPS)
	}
	if snapshot.Metrics.P50EndToEndTPS != 31.447 {
		t.Fatalf("p50_end_to_end_tps = %.3f, want streaming wall-clock metric", snapshot.Metrics.P50EndToEndTPS)
	}
	assertBenchmarkJSONSafe(t, snapshot, "stream measured", "probe measured")
}

func TestBenchmarkSidecarUnavailableReturnsSafeStatus(t *testing.T) {
	client := &queuedCompletionClient{}
	runner := BenchmarkRunner{
		Sidecar: &fakeBenchmarkSidecar{status: LlamaCppSidecarStatus{
			Enabled:   true,
			Available: false,
			Healthy:   false,
			BaseURL:   "http://127.0.0.1:45910",
			Backend:   BackendName,
			Reason:    "llama-server binary not detected",
		}},
		Client: client,
		Now:    fixedBenchmarkNow,
	}

	snapshot := runner.Run(context.Background(), BenchmarkConfig{
		MaxTokens:    8,
		TimeoutMs:    10_000,
		Streaming:    true,
		WarmupRuns:   0,
		MeasuredRuns: 1,
	})
	if snapshot.Status != BenchmarkStatusFailed {
		t.Fatalf("status = %q, want failed", snapshot.Status)
	}
	if snapshot.LastError != "llamacpp_sidecar_unavailable" {
		t.Fatalf("last_error = %q, want safe unavailable code", snapshot.LastError)
	}
	if snapshot.Metrics.Available || snapshot.Metrics.SidecarHealthy || snapshot.Metrics.ModelLoaded {
		t.Fatalf("metrics = %+v, want unavailable/unhealthy/unloaded", snapshot.Metrics)
	}
	if snapshot.Metrics.PromptHash == "" {
		t.Fatalf("prompt_hash empty in unavailable status: %+v", snapshot.Metrics)
	}
	if len(client.requests) != 0 {
		t.Fatalf("client calls = %d, want zero when sidecar unavailable", len(client.requests))
	}
	assertBenchmarkJSONSafe(t, snapshot)
}

func TestBenchmarkLocalStatusStoresLastRun(t *testing.T) {
	status := NewBenchmarkLocalStatus()
	snapshot := BenchmarkStatusSnapshot{
		LastRunAt: fixedBenchmarkNow(),
		Status:    BenchmarkStatusCompleted,
		Metrics: BenchmarkMetrics{
			PromptHash:  HashBenchmarkPrompt(),
			OutputHash:  "sha256:" + strings.Repeat("a", 64),
			Backend:     BackendName,
			RuntimeKind: BackendName,
			ProofStatus: BenchmarkProofStatusMeasured,
		},
	}
	status.Record(snapshot)
	got := status.Snapshot()
	if got.Status != BenchmarkStatusCompleted || got.Metrics.OutputHash == "" {
		t.Fatalf("snapshot = %+v, want completed with output hash", got)
	}
}

func TestCompleteInternalBenchmarkPromptUsesFixedPromptAndFallback(t *testing.T) {
	client := &queuedCompletionClient{steps: []completionStep{
		{err: ClientError{Code: "llamacpp_stream_unavailable", StatusCode: 400}},
		{result: completionResult("fallback measured", 5, 500, 500, false)},
	}}
	result, streamed, err := CompleteInternalBenchmarkPrompt(context.Background(), client, CompletionRequest{
		BaseURL:   "http://127.0.0.1:45910",
		ModelID:   "tinyllama.Q4_K_M.gguf",
		MaxTokens: 8,
		Stream:    true,
	})
	if err != nil {
		t.Fatalf("CompleteInternalBenchmarkPrompt() error = %v", err)
	}
	if streamed {
		t.Fatal("streamed = true, want false after fallback")
	}
	if result.TokensGenerated != 5 {
		t.Fatalf("tokens_generated = %d, want 5", result.TokensGenerated)
	}
	if len(client.requests) != 2 || client.requests[0].Prompt == "" || client.requests[1].Prompt == "" {
		t.Fatalf("requests = %+v, want fixed prompt on both attempts", client.requests)
	}
	if client.requests[0].Prompt != client.requests[1].Prompt {
		t.Fatalf("prompt changed across fallback")
	}
}

func TestKeepWarmEnabledFromEnv(t *testing.T) {
	if KeepWarmEnabledFromEnv(func(string) string { return "" }) {
		t.Fatal("KeepWarmEnabledFromEnv(empty) = true, want idle-safe false")
	}
	if !KeepWarmEnabledFromEnv(func(key string) string {
		if key == EnvKeepWarm {
			return "true"
		}
		return ""
	}) {
		t.Fatal("KeepWarmEnabledFromEnv(true) = false")
	}
	if KeepWarmEnabledFromEnv(func(key string) string {
		if key == EnvKeepWarm {
			return "false"
		}
		return ""
	}) {
		t.Fatal("KeepWarmEnabledFromEnv(false) = true")
	}
	if KeepWarmEnabledFromEnv(func(key string) string {
		if key == EnvDisableModelWarm {
			return "1"
		}
		return ""
	}) {
		t.Fatal("KeepWarmEnabledFromEnv(disable model warm) = true")
	}
}

type fakeBenchmarkSidecar struct {
	status LlamaCppSidecarStatus
}

func healthyBenchmarkSidecar() *fakeBenchmarkSidecar {
	return &fakeBenchmarkSidecar{status: LlamaCppSidecarStatus{
		Enabled:       true,
		Available:     true,
		Running:       true,
		Healthy:       true,
		BaseURL:       "http://127.0.0.1:45910",
		ModelPath:     "/models/tinyllama.Q4_K_M.gguf",
		ModelFilename: "tinyllama.Q4_K_M.gguf",
		Backend:       BackendName,
	}}
}

func (f *fakeBenchmarkSidecar) Start(context.Context) LlamaCppSidecarStatus {
	return f.status
}

func (f *fakeBenchmarkSidecar) Status(context.Context) LlamaCppSidecarStatus {
	return f.status
}

type completionStep struct {
	result CompletionResult
	err    error
}

type queuedCompletionClient struct {
	steps    []completionStep
	requests []CompletionRequest
}

func (c *queuedCompletionClient) Complete(_ context.Context, req CompletionRequest) (CompletionResult, error) {
	c.requests = append(c.requests, req)
	if len(c.steps) == 0 {
		return CompletionResult{}, ClientError{Code: "llamacpp_test_queue_empty"}
	}
	step := c.steps[0]
	c.steps = c.steps[1:]
	if step.err != nil {
		return CompletionResult{}, step.err
	}
	return step.result, nil
}

func completionResult(output string, tokens int64, ttftMs int64, totalMs int64, streamed bool) CompletionResult {
	return CompletionResult{
		Output:           []byte(output),
		OutputBytes:      int64(len(output)),
		PromptTokens:     7,
		CompletionTokens: tokens,
		TokensGenerated:  tokens,
		TTFTMs:           ttftMs,
		TotalTimeMs:      totalMs,
		Streamed:         streamed,
	}
}

func fixedBenchmarkNow() time.Time {
	return time.Unix(1_800_000_000, 0)
}

func assertBenchmarkJSONSafe(t *testing.T, snapshot BenchmarkStatusSnapshot, extraForbidden ...string) {
	t.Helper()
	raw, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatalf("json.Marshal(snapshot) error = %v", err)
	}
	lower := strings.ToLower(string(raw))
	for _, forbidden := range append([]string{
		internalBenchmarkPrompt,
		"raw_prompt",
		"prompt_text",
		"generated_text",
		"output_text",
		"model_output",
		"tensor_bytes",
		"raw_tensor",
	}, extraForbidden...) {
		if strings.Contains(lower, strings.ToLower(forbidden)) {
			t.Fatalf("benchmark JSON contains forbidden text %q: %s", forbidden, raw)
		}
	}
	if !BenchmarkJSONContainsNoRawText(snapshot) {
		t.Fatalf("BenchmarkJSONContainsNoRawText() = false: %s", raw)
	}
}
