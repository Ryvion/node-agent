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
	raw, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("json.Marshal(result) error = %v", err)
	}
	if strings.Contains(string(raw), "hello world") {
		t.Fatalf("completion result JSON leaked output text: %s", raw)
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
		fmt.Fprintln(w, `{"choices":[{"message":{"content":"ready now"}}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`)
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
