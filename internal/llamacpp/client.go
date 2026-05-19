package llamacpp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type Client struct {
	HTTPClient *http.Client
	Now        func() time.Time
}

type Result struct {
	Text             string
	Model            string
	PromptHash       string
	OutputHash       string
	OutputBytes      int64
	PromptTokens     int64
	CompletionTokens int64
	TotalTokens      int64
	FinishReason     string
	DurationMs       int64
}

type chatRequest struct {
	Model       string    `json:"model,omitempty"`
	Messages    []Message `json:"messages"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Temperature float64   `json:"temperature,omitempty"`
	TopP        float64   `json:"top_p,omitempty"`
	Seed        int       `json:"seed,omitempty"`
	Stream      bool      `json:"stream"`
}

type chatResponse struct {
	Model   string `json:"model"`
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
		Text         string `json:"text"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
	Timings struct {
		PredictedN int64 `json:"predicted_n"`
		PromptN    int64 `json:"prompt_n"`
	} `json:"timings"`
}

func (c Client) Complete(ctx context.Context, cfg Config, spec Spec) (Result, error) {
	if err := cfg.Validate(); err != nil {
		return Result{}, err
	}
	spec = NormalizeSpec(spec)
	if err := ValidateSpec(spec); err != nil {
		return Result{}, err
	}
	model := cfg.ModelFor(spec)
	body, err := json.Marshal(chatRequest{
		Model:       model,
		Messages:    MessagesForRequest(spec),
		MaxTokens:   spec.MaxTokens,
		Temperature: spec.Temperature,
		TopP:        spec.TopP,
		Seed:        spec.Seed,
		Stream:      false,
	})
	if err != nil {
		return Result{}, fmt.Errorf("llama.cpp request marshal failed: %w", err)
	}
	started := c.now()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(cfg.ServerURL, "/")+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return Result{}, fmt.Errorf("llama.cpp request build failed: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	client := c.HTTPClient
	if client == nil {
		timeout := cfg.HTTPTimeout
		if timeout <= 0 {
			timeout = 10 * time.Minute
		}
		client = &http.Client{Timeout: timeout}
	}
	resp, err := client.Do(req)
	if err != nil {
		return Result{}, fmt.Errorf("llama.cpp request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
		return Result{}, fmt.Errorf("llama.cpp returned http %d", resp.StatusCode)
	}
	var decoded chatResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, 16*1024*1024)).Decode(&decoded); err != nil {
		return Result{}, fmt.Errorf("llama.cpp response decode failed: %w", err)
	}
	text, finish := firstChoice(decoded)
	if strings.TrimSpace(decoded.Model) != "" {
		model = strings.TrimSpace(decoded.Model)
	}
	result := Result{
		Text:             text,
		Model:            model,
		PromptHash:       PromptHash(spec),
		OutputHash:       OutputHash(text),
		OutputBytes:      int64(len([]byte(text))),
		PromptTokens:     decoded.Usage.PromptTokens,
		CompletionTokens: decoded.Usage.CompletionTokens,
		TotalTokens:      decoded.Usage.TotalTokens,
		FinishReason:     strings.TrimSpace(finish),
		DurationMs:       c.now().Sub(started).Milliseconds(),
	}
	if result.PromptTokens == 0 {
		result.PromptTokens = decoded.Timings.PromptN
	}
	if result.CompletionTokens == 0 {
		result.CompletionTokens = decoded.Timings.PredictedN
	}
	if result.TotalTokens == 0 {
		result.TotalTokens = result.PromptTokens + result.CompletionTokens
	}
	if result.FinishReason == "" {
		result.FinishReason = "unknown"
	}
	return result, nil
}

func (c Client) now() time.Time {
	if c.Now != nil {
		return c.Now()
	}
	return time.Now()
}

func firstChoice(resp chatResponse) (string, string) {
	if len(resp.Choices) == 0 {
		return "", "empty_choices"
	}
	choice := resp.Choices[0]
	text := choice.Message.Content
	if text == "" {
		text = choice.Text
	}
	return text, choice.FinishReason
}
