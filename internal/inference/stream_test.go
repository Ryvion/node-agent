package inference

import (
	"context"
	"crypto/ed25519"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Ryvion/ryvion-node/internal/hub"
)

func TestWriteHubStreamErrorUsesOpenAIErrorShape(t *testing.T) {
	var out strings.Builder
	writeHubStreamError(&out, `insufficient VRAM: "1202" MB free`)

	line := strings.TrimSpace(out.String())
	if !strings.HasPrefix(line, "data: ") {
		t.Fatalf("stream error line = %q, want data prefix", line)
	}
	var payload struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &payload); err != nil {
		t.Fatalf("stream error JSON invalid: %v; line=%q", err, line)
	}
	if payload.Error.Message != `insufficient VRAM: "1202" MB free` || payload.Error.Type != "node_error" {
		t.Fatalf("payload = %+v", payload.Error)
	}
}

func TestApplyStreamingMetricsToMetadataOmitsZeroValues(t *testing.T) {
	meta := map[string]any{"executor": "llama-server"}
	applyStreamingMetricsToMetadata(meta, StreamingMetrics{})
	for _, key := range []string{
		"p50_ttft_ms", "ttft_ms",
		"p50_decode_tps", "decode_tps", "tps",
		"p50_end_to_end_tps", "end_to_end_tps",
		"completion_tokens",
	} {
		if _, ok := meta[key]; ok {
			t.Fatalf("zero StreamingMetrics should not write key %q", key)
		}
	}
}

func TestApplyStreamingMetricsToMetadataPopulatesHubKeys(t *testing.T) {
	meta := map[string]any{}
	applyStreamingMetricsToMetadata(meta, StreamingMetrics{
		TTFTMs:           120,
		DecodeTPS:        45.6,
		EndToEndTPS:      30.2,
		CompletionTokens: 32,
	})

	cases := map[string]any{
		"p50_ttft_ms":        int64(120),
		"ttft_ms":            int64(120),
		"p50_decode_tps":     45.6,
		"decode_tps":         45.6,
		"tps":                45.6,
		"p50_end_to_end_tps": 30.2,
		"end_to_end_tps":     30.2,
		"completion_tokens":  int64(32),
	}
	for key, want := range cases {
		got, ok := meta[key]
		if !ok {
			t.Fatalf("metadata missing key %q", key)
		}
		if got != want {
			t.Fatalf("metadata[%q] = %v (%T), want %v (%T)", key, got, got, want, want)
		}
	}
}

func TestStreamingSpeculativeLaunchEnablesNativeMTPForMTPModel(t *testing.T) {
	launch := streamingSpeculativeLaunchForModel(
		`C:\models\Qwen3.6-27B-MTP-Q5_K_M.gguf`,
		"",
		func(key string) string {
			if key == envStreamingNativeMTP {
				return "auto"
			}
			return ""
		},
	)
	if launch.Method != speculativeMethodNativeMTP {
		t.Fatalf("method = %q, want native MTP", launch.Method)
	}
	joined := strings.Join(launch.Args, " ")
	for _, want := range []string{"--spec-type draft-mtp", "--spec-draft-n-max 3"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("args = %q, missing %q", joined, want)
		}
	}
}

func TestStreamingSpeculativeLaunchFallsBackToNGramForPlainModel(t *testing.T) {
	launch := streamingSpeculativeLaunchForModel(
		`C:\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf`,
		"",
		func(key string) string {
			if key == envStreamingNativeMTP {
				return "1"
			}
			return ""
		},
	)
	if launch.Method != speculativeMethodNGramSimple {
		t.Fatalf("method = %q, want draftless ngram fallback", launch.Method)
	}
	joined := strings.Join(launch.Args, " ")
	if strings.Contains(joined, "draft-mtp") {
		t.Fatalf("plain model args = %q, should not force native MTP", joined)
	}
	for _, want := range []string{"--spec-type ngram-simple", "--spec-draft-n-max 16"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("args = %q, missing %q", joined, want)
		}
	}
}

func TestStreamingSpeculativeLaunchCanDisableDraftlessFallback(t *testing.T) {
	launch := streamingSpeculativeLaunchForModel(
		`C:\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf`,
		"",
		func(key string) string {
			if key == envStreamingSpecType {
				return "none"
			}
			return ""
		},
	)
	if launch.Method != "" || strings.Join(launch.Args, " ") != "" {
		t.Fatalf("launch = %+v, want no speculative flags when spec type is none", launch)
	}
}

func TestStreamingSpeculativeLaunchUsesExplicitNGramMode(t *testing.T) {
	launch := streamingSpeculativeLaunchForModel(
		`C:\models\nemotron-3-nano-omni-30b-a3b-Q4_K_M.gguf`,
		"",
		func(key string) string {
			switch key {
			case envStreamingSpecType:
				return speculativeMethodNGramMod
			case envStreamingDraftMaxTokens:
				return "24"
			default:
				return ""
			}
		},
	)
	if launch.Method != speculativeMethodNGramMod {
		t.Fatalf("method = %q, want explicit ngram-mod", launch.Method)
	}
	if joined := strings.Join(launch.Args, " "); !strings.Contains(joined, "--spec-type ngram-mod") || !strings.Contains(joined, "--spec-draft-n-max 24") {
		t.Fatalf("args = %q, want explicit ngram mode", joined)
	}
}

func TestApplyStreamingSpeculativeMetadata(t *testing.T) {
	meta := map[string]any{}
	launch := streamingSpeculativeLaunch{
		Method:         speculativeMethodNativeMTP,
		DraftMaxTokens: 3,
	}
	applyStreamingSpeculativeMetadata(meta, launch, StreamingMetrics{
		CompletionTokens:          10,
		SpeculativeTokensDrafted:  8,
		SpeculativeTokensAccepted: 5,
	})

	if meta["speculative_enabled"] != true || meta["speculative_method"] != speculativeMethodNativeMTP {
		t.Fatalf("flat speculative metadata = %#v", meta)
	}
	block, ok := meta["speculative"].(map[string]any)
	if !ok {
		t.Fatalf("missing speculative block: %#v", meta)
	}
	if block["tokens_drafted"] != int64(8) || block["tokens_accepted"] != int64(5) {
		t.Fatalf("speculative counts = %#v", block)
	}
	if block["estimated_speedup_ratio"] != 2.0 || meta["speculative_speedup"] != 2.0 {
		t.Fatalf("speculative speedup = block %#v flat %#v", block["estimated_speedup_ratio"], meta["speculative_speedup"])
	}
}

// TestRunStreamingJobCapturesTimingMetrics drives RunStreamingJob against a
// fake llama-server that emits a few SSE chunks with a measurable TTFT, then
// inspects the receipt the manager submits to a fake hub.
func TestRunStreamingJobCapturesTimingMetrics(t *testing.T) {
	const (
		ttftDelay  = 120 * time.Millisecond
		interToken = 25 * time.Millisecond
	)

	// Fake llama-server: sleeps to create observable TTFT, then streams
	// content chunks, then a usage chunk + [DONE].
	llamaSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		writeChunk := func(payload string) {
			fmt.Fprintf(w, "data: %s\n\n", payload)
			if flusher != nil {
				flusher.Flush()
			}
		}
		// Prefill stall — manager observes this as TTFT.
		time.Sleep(ttftDelay)
		writeChunk(`{"choices":[{"delta":{"content":"hello"}}]}`)
		time.Sleep(interToken)
		writeChunk(`{"choices":[{"delta":{"content":" world"}}]}`)
		time.Sleep(interToken)
		writeChunk(`{"choices":[{"delta":{"content":"!"}}]}`)
		// Terminal usage frame — llama-server emits this on recent builds.
		writeChunk(`{"choices":[{"delta":{}}],"usage":{"prompt_tokens":4,"completion_tokens":7,"total_tokens":11},"timings":{"n_drafted":11,"n_accepted":7}}`)
		writeChunk("[DONE]")
	}))
	defer llamaSrv.Close()

	llamaURL, err := url.Parse(llamaSrv.URL)
	if err != nil {
		t.Fatalf("parse llama server url: %v", err)
	}
	_, llamaPort, err := net.SplitHostPort(llamaURL.Host)
	if err != nil {
		t.Fatalf("split llama port: %v", err)
	}

	// Fake hub: captures the streamed body + the receipt POST so the test can
	// assert what landed in MetadataJSON.
	var (
		hubMu        sync.Mutex
		gotReceipt   map[string]any
		streamCalled bool
	)
	hubSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasPrefix(r.URL.Path, "/api/v1/node/inference/stream/"):
			hubMu.Lock()
			streamCalled = true
			hubMu.Unlock()
			_, _ = io.Copy(io.Discard, r.Body)
			w.WriteHeader(http.StatusOK)
		case r.URL.Path == "/api/v1/node/receipt":
			body, _ := io.ReadAll(r.Body)
			var raw map[string]any
			if err := json.Unmarshal(body, &raw); err != nil {
				t.Errorf("decode receipt body: %v", err)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			hubMu.Lock()
			gotReceipt = raw
			hubMu.Unlock()
			w.WriteHeader(http.StatusOK)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer hubSrv.Close()

	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("generate keypair: %v", err)
	}
	hubClient := hub.New(hubSrv.URL, pub, priv)

	// Build a Manager pointed at the fake llama-server. activeModelName is
	// pre-set so EnsureModel short-circuits without trying to download.
	mgr := &Manager{
		dataDir:         t.TempDir(),
		port:            llamaPort,
		activeModelName: "ryvion-llama-3.2-3b",
		speculative:     streamingSpeculativeLaunch{Method: speculativeMethodNativeMTP, DraftMaxTokens: 3},
		healthy:         true,
	}

	specJSON, err := json.Marshal(map[string]any{
		"messages":   []map[string]string{{"role": "user", "content": "hi"}},
		"max_tokens": 16,
	})
	if err != nil {
		t.Fatalf("marshal spec: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := mgr.RunStreamingJob(ctx, hubClient, "job-test-1", string(specJSON)); err != nil {
		t.Fatalf("RunStreamingJob returned error: %v", err)
	}

	hubMu.Lock()
	defer hubMu.Unlock()
	if !streamCalled {
		t.Fatalf("hub stream relay endpoint was not called")
	}
	if gotReceipt == nil {
		t.Fatalf("hub receipt endpoint was not called")
	}
	metaRaw, ok := gotReceipt["metadata"].(map[string]any)
	if !ok {
		t.Fatalf("receipt body has no metadata: %#v", gotReceipt)
	}

	// TTFT must be roughly the prefill stall — assert > 50ms with a
	// generous ceiling so flakiness on slow CI hosts doesn't trip us up.
	ttftRaw, ok := metaRaw["ttft_ms"]
	if !ok {
		t.Fatalf("metadata missing ttft_ms; got keys=%v", keys(metaRaw))
	}
	ttftMs := numericAsFloat(t, ttftRaw)
	if ttftMs <= 50 {
		t.Fatalf("ttft_ms = %v, want > 50 (prefill delay was %v)", ttftMs, ttftDelay)
	}

	// Decode TPS must be set and positive — three content chunks with usage
	// reporting 7 tokens divided by the post-TTFT window.
	decodeRaw, ok := metaRaw["decode_tps"]
	if !ok {
		t.Fatalf("metadata missing decode_tps; got keys=%v", keys(metaRaw))
	}
	if got := numericAsFloat(t, decodeRaw); got <= 0 {
		t.Fatalf("decode_tps = %v, want > 0", got)
	}

	// End-to-end TPS must be set and lower than decode TPS (it includes
	// prefill, decode does not).
	e2eRaw, ok := metaRaw["end_to_end_tps"]
	if !ok {
		t.Fatalf("metadata missing end_to_end_tps; got keys=%v", keys(metaRaw))
	}
	e2eTPS := numericAsFloat(t, e2eRaw)
	if e2eTPS <= 0 {
		t.Fatalf("end_to_end_tps = %v, want > 0", e2eTPS)
	}
	if e2eTPS >= numericAsFloat(t, decodeRaw) {
		t.Fatalf("end_to_end_tps (%v) should be < decode_tps (%v)", e2eTPS, decodeRaw)
	}

	// completion_tokens should reflect the usage frame value (7).
	tokRaw, ok := metaRaw["completion_tokens"]
	if !ok {
		t.Fatalf("metadata missing completion_tokens; got keys=%v", keys(metaRaw))
	}
	if got := numericAsFloat(t, tokRaw); int(got) != 7 {
		t.Fatalf("completion_tokens = %v, want 7", got)
	}

	// Hub also reads the p50_* aliases — verify they mirror their canonical
	// keys so the /v8 verifier columns light up.
	if got := numericAsFloat(t, metaRaw["p50_ttft_ms"]); got != ttftMs {
		t.Fatalf("p50_ttft_ms = %v, want %v", got, ttftMs)
	}
	if got := numericAsFloat(t, metaRaw["p50_decode_tps"]); got != numericAsFloat(t, decodeRaw) {
		t.Fatalf("p50_decode_tps = %v, want %v", got, decodeRaw)
	}
	if got := numericAsFloat(t, metaRaw["p50_end_to_end_tps"]); got != e2eTPS {
		t.Fatalf("p50_end_to_end_tps = %v, want %v", got, e2eTPS)
	}
	if metaRaw["speculative_enabled"] != true || metaRaw["speculative_method"] != speculativeMethodNativeMTP {
		t.Fatalf("flat speculative metadata = %#v", metaRaw)
	}
	specBlock, ok := metaRaw["speculative"].(map[string]any)
	if !ok {
		t.Fatalf("metadata missing speculative block: %#v", metaRaw)
	}
	if specBlock["tokens_drafted"] != float64(11) || specBlock["tokens_accepted"] != float64(7) {
		t.Fatalf("speculative block = %#v", specBlock)
	}
}

func keys(m map[string]any) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

func numericAsFloat(t *testing.T, v any) float64 {
	t.Helper()
	switch x := v.(type) {
	case float64:
		return x
	case float32:
		return float64(x)
	case int:
		return float64(x)
	case int64:
		return float64(x)
	case json.Number:
		f, err := x.Float64()
		if err != nil {
			t.Fatalf("json.Number.Float64: %v", err)
		}
		return f
	default:
		t.Fatalf("unexpected numeric type %T (%v)", v, v)
		return 0
	}
}
