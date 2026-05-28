package llamacpp

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestOpenAIClientStreamingChunksComputeTimings(t *testing.T) {
	base := time.Unix(1_800_000_000, 0)
	clock := &sequenceClock{times: []time.Time{
		base,
		base.Add(125 * time.Millisecond),
		base.Add(925 * time.Millisecond),
	}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("path = %q, want /v1/chat/completions", r.URL.Path)
		}
		var req openAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if !req.Stream {
			t.Fatalf("stream = false, want true")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"hello"}}]}`)
		fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":" world"}}],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}`)
		fmt.Fprintln(w, `data: {"choices":[{"finish_reason":"stop"}],"timings":{"predicted_n":2,"predicted_ms":50,"predicted_per_second":40}}`)
		fmt.Fprintln(w, `data: [DONE]`)
	}))
	defer server.Close()

	var deltas []string
	result, err := (OpenAIClient{HTTPClient: server.Client(), Now: clock.Now}).Complete(context.Background(), CompletionRequest{
		BaseURL:     server.URL,
		ModelID:     "tinyllama.Q4_K_M.gguf",
		Prompt:      internalBenchmarkPrompt,
		MaxTokens:   8,
		Temperature: 0,
		Stream:      true,
		OnDelta: func(delta CompletionDelta) error {
			deltas = append(deltas, delta.Text)
			return nil
		},
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if string(result.Output) != "hello world" {
		t.Fatalf("output = %q, want hello world", result.Output)
	}
	if strings.Join(deltas, "") != "hello world" || len(deltas) != 2 {
		t.Fatalf("deltas = %+v, want streamed text fragments", deltas)
	}
	if result.TTFTMs != 125 || result.TotalTimeMs != 925 {
		t.Fatalf("timings = ttft %d total %d, want 125/925", result.TTFTMs, result.TotalTimeMs)
	}
	if result.TokensGenerated != 2 || result.PromptTokens != 7 || result.CompletionTokens != 2 {
		t.Fatalf("tokens = %+v, want usage token counts", result)
	}
	if result.BackendDecodeMs != 50 || result.BackendDecodeTPS != 40 {
		t.Fatalf("backend decode stats = %+v, want 50ms/40tps", result)
	}
	if result.FinishReason != FinishReasonStop || result.BackendFinishReason != FinishReasonStop || result.MaxTokensReached {
		t.Fatalf("finish metadata = %+v, want natural stop", result)
	}
	raw, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("json.Marshal(result) error = %v", err)
	}
	if strings.Contains(string(raw), "hello world") {
		t.Fatalf("completion result JSON leaked output text: %s", raw)
	}
}

func TestOpenAIClientStreamingAcceptsReasoningOnlyDeltas(t *testing.T) {
	base := time.Unix(1_800_000_000, 0)
	clock := &sequenceClock{times: []time.Time{
		base,
		base.Add(200 * time.Millisecond),
		base.Add(900 * time.Millisecond),
	}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("path = %q, want /v1/chat/completions", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, `data: {"choices":[{"delta":{"reasoning_content":"thinking "}}]}`)
		fmt.Fprintln(w, `data: {"choices":[{"delta":{"reasoning":"through"}}],"usage":{"prompt_tokens":9,"completion_tokens":2,"total_tokens":11}}`)
		fmt.Fprintln(w, `data: {"choices":[{"finish_reason":"length"}],"timings":{"predicted_n":2}}`)
		fmt.Fprintln(w, `data: [DONE]`)
	}))
	defer server.Close()

	var deltas []string
	result, err := (OpenAIClient{HTTPClient: server.Client(), Now: clock.Now}).Complete(context.Background(), CompletionRequest{
		BaseURL:     server.URL,
		ModelID:     "Qwen3-8B-Q4_K_M.gguf",
		Prompt:      "Think briefly.",
		MaxTokens:   8,
		Temperature: 0,
		Stream:      true,
		OnDelta: func(delta CompletionDelta) error {
			deltas = append(deltas, delta.Text)
			return nil
		},
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if string(result.Output) != "thinking through" {
		t.Fatalf("output = %q, want reasoning-only stream output", result.Output)
	}
	if strings.Join(deltas, "") != "thinking through" {
		t.Fatalf("deltas = %+v, want reasoning deltas", deltas)
	}
	if result.TTFTMs != 200 || result.TotalTimeMs != 900 {
		t.Fatalf("timings = ttft %d total %d, want 200/900", result.TTFTMs, result.TotalTimeMs)
	}
	if result.TokensGenerated != 2 || result.PromptTokens != 9 {
		t.Fatalf("tokens = %+v, want usage token counts", result)
	}
}

func TestOpenAIClientNonStreamingCompletion(t *testing.T) {
	base := time.Unix(1_800_000_000, 0)
	clock := &sequenceClock{times: []time.Time{
		base,
		base.Add(600 * time.Millisecond),
	}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req openAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if req.Stream {
			t.Fatalf("stream = true, want false")
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintln(w, `{"choices":[{"message":{"content":"ready now"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`)
	}))
	defer server.Close()

	result, err := (OpenAIClient{HTTPClient: server.Client(), Now: clock.Now}).Complete(context.Background(), CompletionRequest{
		BaseURL:     server.URL,
		ModelID:     "tinyllama.Q4_K_M.gguf",
		Prompt:      internalBenchmarkPrompt,
		MaxTokens:   8,
		Temperature: 0,
		Stream:      false,
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if string(result.Output) != "ready now" {
		t.Fatalf("output = %q, want ready now", result.Output)
	}
	if result.TTFTMs != 600 || result.TotalTimeMs != 600 {
		t.Fatalf("timings = ttft %d total %d, want 600/600", result.TTFTMs, result.TotalTimeMs)
	}
	if result.Streamed {
		t.Fatalf("streamed = true, want false")
	}
	if result.FinishReason != FinishReasonStop || result.MaxTokensReached {
		t.Fatalf("finish metadata = %+v, want stop without max cap", result)
	}
	if result.TokensGenerated != 2 || result.RuntimeMeasurementStatus != RuntimeMeasurementStatusMeasured || result.MetadataParseStatus != MetadataParseStatusOK {
		t.Fatalf("normalized token metadata = %+v, want OpenAI usage token count", result)
	}
}

func TestOpenAIClientNonStreamingAcceptsReasoningOnlyMessage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("path = %q, want /v1/chat/completions", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintln(w, `{"choices":[{"message":{"reasoning_content":"reasoned but hit the cap"},"finish_reason":"length"}],"usage":{"prompt_tokens":5,"completion_tokens":6,"total_tokens":11}}`)
	}))
	defer server.Close()

	result, err := (OpenAIClient{HTTPClient: server.Client()}).Complete(context.Background(), CompletionRequest{
		BaseURL:     server.URL,
		ModelID:     "Qwen3-8B-Q4_K_M.gguf",
		Prompt:      "Think briefly.",
		MaxTokens:   6,
		Temperature: 0,
		Stream:      false,
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if string(result.Output) != "reasoned but hit the cap" {
		t.Fatalf("output = %q, want reasoning-only message output", result.Output)
	}
	if result.TokensGenerated != 6 || result.PromptTokens != 5 {
		t.Fatalf("tokens = %+v, want usage token counts", result)
	}
	if result.FinishReason != FinishReasonLength || !result.MaxTokensReached {
		t.Fatalf("finish metadata = %+v, want length cap", result)
	}
}

func TestOpenAIClientSendsResolvedSystemPromptAsChatMessage(t *testing.T) {
	const systemPrompt = "Answer as a concise Ryvion product demo."
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("path = %q, want /v1/chat/completions", r.URL.Path)
		}
		var req openAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if len(req.Messages) != 2 ||
			req.Messages[0].Role != "system" ||
			req.Messages[0].Content != systemPrompt ||
			req.Messages[1].Role != "user" ||
			req.Messages[1].Content != "Explain Ryvion." {
			t.Fatalf("messages = %+v, want system and user chat messages", req.Messages)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintln(w, `{"choices":[{"message":{"content":"Ryvion routes warm backends."},"finish_reason":"stop"}],"usage":{"completion_tokens":4}}`)
	}))
	defer server.Close()

	result, err := (OpenAIClient{HTTPClient: server.Client()}).Complete(context.Background(), CompletionRequest{
		BaseURL:      server.URL,
		ModelID:      "tinyllama.Q4_K_M.gguf",
		Prompt:       "Explain Ryvion.",
		SystemPrompt: systemPrompt,
		MaxTokens:    8,
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if result.PromptMode != PromptModeChatMessages || result.SystemPromptHash != HashSystemPrompt(systemPrompt) {
		t.Fatalf("prompt metadata = %+v, want chat mode and system prompt hash", result)
	}
	raw, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("json.Marshal(result) error = %v", err)
	}
	if strings.Contains(string(raw), systemPrompt) || strings.Contains(string(raw), "Explain Ryvion.") {
		t.Fatalf("completion result JSON leaked prompt text: %s", raw)
	}
}

func TestOpenAIClientSendsCachePromptSlotFields(t *testing.T) {
	slotID := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("path = %q, want /v1/chat/completions", r.URL.Path)
		}
		var req openAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if req.CachePrompt == nil || !*req.CachePrompt || req.NCacheReuse != 128 || req.IDSlot == nil || *req.IDSlot != 0 {
			t.Fatalf("cache request fields = %+v", req)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintln(w, `{"choices":[{"message":{"content":"cache hit path"},"finish_reason":"stop"}],"usage":{"completion_tokens":3}}`)
	}))
	defer server.Close()

	_, err := (OpenAIClient{HTTPClient: server.Client()}).Complete(context.Background(), CompletionRequest{
		BaseURL:          server.URL,
		ModelID:          "tinyllama.Q4_K_M.gguf",
		Prompt:           "Reuse the prefix.",
		MaxTokens:        8,
		CachePrompt:      true,
		CacheReuseTokens: 128,
		SlotID:           &slotID,
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
}

func TestOpenAIClientSendsProvidedMessagesWithLeadingSystemPrompt(t *testing.T) {
	const systemPrompt = "Use Ryvion runtime facts."
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req openAIChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		want := []openAIChatMessage{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: "What is ready?"},
			{Role: "assistant", Content: "The local model is warm."},
			{Role: "user", Content: "Summarize it."},
		}
		if len(req.Messages) != len(want) {
			t.Fatalf("messages = %+v, want %+v", req.Messages, want)
		}
		for i := range want {
			if req.Messages[i] != want[i] {
				t.Fatalf("message[%d] = %+v, want %+v", i, req.Messages[i], want[i])
			}
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintln(w, `{"choices":[{"message":{"content":"Warm and ready."},"finish_reason":"stop"}],"usage":{"completion_tokens":3}}`)
	}))
	defer server.Close()

	result, err := (OpenAIClient{HTTPClient: server.Client()}).Complete(context.Background(), CompletionRequest{
		BaseURL:      server.URL,
		ModelID:      "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		SystemPrompt: systemPrompt,
		Messages: []CompletionMessage{
			{Role: "system", Content: "Discard this stale system message."},
			{Role: "user", Content: "What is ready?"},
			{Role: "assistant", Content: "The local model is warm."},
			{Role: "user", Content: "Summarize it."},
		},
		MaxTokens: 8,
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if result.PromptMode != PromptModeChatMessages || result.SystemPromptHash != HashSystemPrompt(systemPrompt) {
		t.Fatalf("prompt metadata = %+v, want chat messages with system hash", result)
	}
}

func TestOpenAIClientSlotSaveRestoreUsesLocalSlotsEndpoint(t *testing.T) {
	var paths []string
	var filenames []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.String())
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s, want POST", r.Method)
		}
		var req slotActionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode slot request: %v", err)
		}
		filenames = append(filenames, req.Filename)
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Query().Get("action") {
		case "restore":
			fmt.Fprintln(w, `{"id_slot":0,"filename":"ryvion_slot_abc.bin","n_restored":12,"n_read":4096}`)
		case "save":
			fmt.Fprintln(w, `{"id_slot":0,"filename":"ryvion_slot_abc.bin","n_saved":13,"n_written":8192}`)
		default:
			t.Fatalf("unexpected action %q", r.URL.Query().Get("action"))
		}
	}))
	defer server.Close()

	client := OpenAIClient{HTTPClient: server.Client()}
	restore, err := client.RestoreSlot(context.Background(), SlotCacheRequest{BaseURL: server.URL, SlotID: 0, Filename: "ryvion_slot_abc.bin"})
	if err != nil {
		t.Fatalf("RestoreSlot() error = %v", err)
	}
	save, err := client.SaveSlot(context.Background(), SlotCacheRequest{BaseURL: server.URL, SlotID: 0, Filename: "ryvion_slot_abc.bin"})
	if err != nil {
		t.Fatalf("SaveSlot() error = %v", err)
	}
	if strings.Join(paths, ",") != "/slots/0?action=restore,/slots/0?action=save" {
		t.Fatalf("paths = %v", paths)
	}
	if strings.Join(filenames, ",") != "ryvion_slot_abc.bin,ryvion_slot_abc.bin" {
		t.Fatalf("filenames = %v", filenames)
	}
	if restore.RestoredTokens != 12 || save.SavedTokens != 13 {
		t.Fatalf("slot results restore=%+v save=%+v", restore, save)
	}
}

func TestOpenAIClientFallsBackToRawLlama3TemplateWhenChatUnavailable(t *testing.T) {
	const systemPrompt = "Ground the answer in Ryvion local runtime facts."
	var paths []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		switch r.URL.Path {
		case "/v1/chat/completions":
			http.NotFound(w, r)
		case "/completion":
			var req rawCompletionRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatalf("decode raw request: %v", err)
			}
			for _, want := range []string{
				"<|begin_of_text|><|start_header_id|>system<|end_header_id|>",
				systemPrompt,
				"<|start_header_id|>user<|end_header_id|>",
				"Explain Ryvion.",
				"<|start_header_id|>assistant<|end_header_id|>",
			} {
				if !strings.Contains(req.Prompt, want) {
					t.Fatalf("raw prompt missing %q: %q", want, req.Prompt)
				}
			}
			if req.Stream {
				t.Fatalf("raw stream = true, want false")
			}
			if req.NPredict != 8 {
				t.Fatalf("n_predict = %d, want 8", req.NPredict)
			}
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprintln(w, `{"content":"Ryvion routes warm local runtime requests.","tokens_evaluated":9,"timings":{"predicted_n":6},"stopped_eos":true}`)
		default:
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
	}))
	defer server.Close()

	result, err := (OpenAIClient{HTTPClient: server.Client()}).Complete(context.Background(), CompletionRequest{
		BaseURL:      server.URL,
		ModelID:      "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		Prompt:       "Explain Ryvion.",
		SystemPrompt: systemPrompt,
		MaxTokens:    8,
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if got := strings.Join(paths, ","); got != "/v1/chat/completions,/completion" {
		t.Fatalf("paths = %s, want chat then raw completion", got)
	}
	if result.PromptMode != PromptModeTemplate || result.SystemPromptHash != HashSystemPrompt(systemPrompt) {
		t.Fatalf("prompt metadata = %+v, want template/hash", result)
	}
	if string(result.Output) != "Ryvion routes warm local runtime requests." || result.TokensGenerated != 6 || result.PromptTokens != 9 {
		t.Fatalf("result = %+v, want raw completion output and metrics", result)
	}
}

func TestOpenAIClientFallsBackToStreamingRawCompletion(t *testing.T) {
	var rawPrompt string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/chat/completions":
			http.NotFound(w, r)
		case "/completion":
			var req rawCompletionRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatalf("decode raw request: %v", err)
			}
			rawPrompt = req.Prompt
			if !req.Stream {
				t.Fatal("raw stream = false, want true")
			}
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprintln(w, `data: {"content":"Ryvion"}`)
			fmt.Fprintln(w, `data: {"content":" streams","completion_tokens":2}`)
			fmt.Fprintln(w, `data: {"stop":true}`)
			fmt.Fprintln(w, `data: [DONE]`)
		default:
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
	}))
	defer server.Close()

	var deltas []string
	result, err := (OpenAIClient{HTTPClient: server.Client()}).Complete(context.Background(), CompletionRequest{
		BaseURL:      server.URL,
		ModelID:      "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		Prompt:       "Say two words.",
		SystemPrompt: "Use Ryvion facts.",
		MaxTokens:    8,
		Stream:       true,
		OnDelta: func(delta CompletionDelta) error {
			deltas = append(deltas, delta.Text)
			return nil
		},
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if result.PromptMode != PromptModeTemplate || !result.Streamed {
		t.Fatalf("result metadata = %+v, want streamed template result", result)
	}
	if string(result.Output) != "Ryvion streams" || strings.Join(deltas, "") != "Ryvion streams" {
		t.Fatalf("output = %q deltas = %+v, want streamed raw content", result.Output, deltas)
	}
	if !strings.Contains(rawPrompt, "<|start_header_id|>assistant<|end_header_id|>") {
		t.Fatalf("raw prompt missing assistant header: %q", rawPrompt)
	}
}

func TestOpenAIClientExtractsLlamaStoppedLimitFinishMetadata(t *testing.T) {
	base := time.Unix(1_800_000_000, 0)
	clock := &sequenceClock{times: []time.Time{base, base.Add(600 * time.Millisecond)}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintln(w, `{"content":"partial wor","stopped_limit":true,"tokens_evaluated":5,"timings":{"predicted_n":8}}`)
	}))
	defer server.Close()

	result, err := (OpenAIClient{HTTPClient: server.Client(), Now: clock.Now}).Complete(context.Background(), CompletionRequest{
		BaseURL:     server.URL,
		ModelID:     "tinyllama.Q4_K_M.gguf",
		Prompt:      internalBenchmarkPrompt,
		MaxTokens:   8,
		Temperature: 0,
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if result.TokensGenerated != 8 || result.PromptTokens != 5 || result.CompletionTokens != 8 {
		t.Fatalf("tokens = %+v, want llama.cpp backend token counts", result)
	}
	if result.RequestedMaxTokens != 8 || result.FinishReason != FinishReasonLength || result.BackendStopReason != "stopped_limit" || !result.MaxTokensReached {
		t.Fatalf("finish metadata = %+v, want stopped_limit length cap", result)
	}
	if result.RuntimeMeasurementStatus != RuntimeMeasurementStatusMeasured || result.MetadataParseStatus != MetadataParseStatusOK || result.TokenCountEstimated {
		t.Fatalf("measurement metadata = %+v, want precise llama.cpp timing count", result)
	}
}

func TestOpenAIClientInfersMaxTokensReachedWhenBackendReasonMissing(t *testing.T) {
	base := time.Unix(1_800_000_000, 0)
	clock := &sequenceClock{times: []time.Time{base, base.Add(600 * time.Millisecond)}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintln(w, `{"choices":[{"message":{"content":"partial wor"}}],"usage":{"completion_tokens":8}}`)
	}))
	defer server.Close()

	result, err := (OpenAIClient{HTTPClient: server.Client(), Now: clock.Now}).Complete(context.Background(), CompletionRequest{
		BaseURL:     server.URL,
		ModelID:     "tinyllama.Q4_K_M.gguf",
		Prompt:      internalBenchmarkPrompt,
		MaxTokens:   8,
		Temperature: 0,
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if result.FinishReason != FinishReasonLength || !result.MaxTokensReached {
		t.Fatalf("finish metadata = %+v, want length reason with inferred max cap", result)
	}
}

func TestOpenAIClientExtractsLlamaNPredictedTimingTokenCount(t *testing.T) {
	base := time.Unix(1_800_000_000, 0)
	clock := &sequenceClock{times: []time.Time{base, base.Add(400 * time.Millisecond)}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintln(w, `{"content":"timed raw shape","timings":{"n_predicted":6}}`)
	}))
	defer server.Close()

	result, err := (OpenAIClient{HTTPClient: server.Client(), Now: clock.Now}).Complete(context.Background(), CompletionRequest{
		BaseURL:     server.URL,
		ModelID:     "tinyllama.Q4_K_M.gguf",
		Prompt:      internalBenchmarkPrompt,
		MaxTokens:   16,
		Temperature: 0,
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if result.TokensGenerated != 6 || result.TokenCountEstimated {
		t.Fatalf("tokens = %+v, want n_predicted precise token count", result)
	}
	if result.RuntimeMeasurementStatus != RuntimeMeasurementStatusMeasured || result.MetadataParseStatus != MetadataParseStatusOK {
		t.Fatalf("measurement metadata = %+v", result)
	}
}

func TestOpenAIClientEstimatesMissingBackendTokenCountAsPartial(t *testing.T) {
	base := time.Unix(1_800_000_000, 0)
	clock := &sequenceClock{times: []time.Time{base, base.Add(300 * time.Millisecond)}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintln(w, `{"choices":[{"message":{"content":"fallback token estimate"}}],"finish_reason":"stop"}`)
	}))
	defer server.Close()

	result, err := (OpenAIClient{HTTPClient: server.Client(), Now: clock.Now}).Complete(context.Background(), CompletionRequest{
		BaseURL:     server.URL,
		ModelID:     "tinyllama.Q4_K_M.gguf",
		Prompt:      internalBenchmarkPrompt,
		MaxTokens:   16,
		Temperature: 0,
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if result.TokensGenerated != 3 || !result.TokenCountEstimated {
		t.Fatalf("tokens = %+v, want fallback estimate from generated text", result)
	}
	if result.RuntimeMeasurementStatus != RuntimeMeasurementStatusPartial || result.MetadataParseStatus != MetadataParseStatusOK {
		t.Fatalf("measurement metadata = %+v, want partial runtime with ok parse", result)
	}
}

func TestOpenAIClientStreamingUsesDeltasBeforeBackendTokenField(t *testing.T) {
	base := time.Unix(1_800_000_000, 0)
	clock := &sequenceClock{times: []time.Time{
		base,
		base.Add(50 * time.Millisecond),
		base.Add(500 * time.Millisecond),
	}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"one"}}]}`)
		fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":" two"}}],"tokens_generated":99}`)
		fmt.Fprintln(w, `data: {"choices":[{"finish_reason":"stop"}]}`)
		fmt.Fprintln(w, `data: [DONE]`)
	}))
	defer server.Close()

	result, err := (OpenAIClient{HTTPClient: server.Client(), Now: clock.Now}).Complete(context.Background(), CompletionRequest{
		BaseURL:   server.URL,
		ModelID:   "tinyllama.Q4_K_M.gguf",
		Prompt:    internalBenchmarkPrompt,
		MaxTokens: 16,
		Stream:    true,
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if result.TokensGenerated != 2 || result.TokenCountEstimated {
		t.Fatalf("tokens = %+v, want node-side streamed delta count before backend token field", result)
	}
	if result.FinishReason != FinishReasonStop {
		t.Fatalf("finish metadata = %+v, want final stop reason", result)
	}
}

func TestOpenAIClientNormalizesAlternateGeneratedTextShapes(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{
			name: "choice_content",
			body: `{"choices":[{"content":"direct choice"}],"usage":{"completion_tokens":2}}`,
			want: "direct choice",
		},
		{
			name: "top_level_content",
			body: `{"content":"top level content","usage":{"completion_tokens":3}}`,
			want: "top level content",
		},
		{
			name: "top_level_response",
			body: `{"response":"response text","usage":{"completion_tokens":2}}`,
			want: "response text",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			base := time.Unix(1_800_000_000, 0)
			clock := &sequenceClock{times: []time.Time{base, base.Add(100 * time.Millisecond)}}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprintln(w, tt.body)
			}))
			defer server.Close()

			result, err := (OpenAIClient{HTTPClient: server.Client(), Now: clock.Now}).Complete(context.Background(), CompletionRequest{
				BaseURL:     server.URL,
				ModelID:     "tinyllama.Q4_K_M.gguf",
				Prompt:      internalBenchmarkPrompt,
				MaxTokens:   8,
				Temperature: 0,
				Stream:      false,
			})
			if err != nil {
				t.Fatalf("Complete() error = %v", err)
			}
			if string(result.Output) != tt.want {
				t.Fatalf("output = %q, want %q", result.Output, tt.want)
			}
		})
	}
}

type sequenceClock struct {
	times []time.Time
	idx   int
}

func (c *sequenceClock) Now() time.Time {
	if len(c.times) == 0 {
		return time.Now()
	}
	if c.idx >= len(c.times) {
		return c.times[len(c.times)-1]
	}
	value := c.times[c.idx]
	c.idx++
	return value
}
