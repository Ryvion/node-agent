package llamacpp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const defaultClientErrorCode = "llamacpp_request_failed"

type CompletionRequest struct {
	BaseURL     string
	ModelID     string
	Prompt      string
	MaxTokens   int
	Temperature float64
	Stream      bool
	OnDelta     func(CompletionDelta) error
}

type CompletionDelta struct {
	Text string
}

type CompletionResult struct {
	Output              []byte `json:"-"`
	OutputBytes         int64  `json:"output_bytes"`
	PromptTokens        int64  `json:"prompt_tokens,omitempty"`
	CompletionTokens    int64  `json:"completion_tokens,omitempty"`
	RequestedMaxTokens  int    `json:"requested_max_tokens"`
	TokensGenerated     int64  `json:"tokens_generated"`
	FinishReason        string `json:"finish_reason"`
	BackendFinishReason string `json:"backend_finish_reason"`
	BackendStopReason   string `json:"backend_stop_reason"`
	MaxTokensReached    bool   `json:"max_tokens_reached"`
	TTFTMs              int64  `json:"ttft_ms"`
	TotalTimeMs         int64  `json:"total_time_ms"`
	Streamed            bool   `json:"streamed"`
}

type CompletionClient interface {
	Complete(context.Context, CompletionRequest) (CompletionResult, error)
}

type OpenAIClient struct {
	HTTPClient *http.Client
	Now        func() time.Time
}

type ClientError struct {
	Code       string
	StatusCode int
	Message    string
}

func (e ClientError) Error() string {
	code := strings.TrimSpace(e.Code)
	if code == "" {
		code = defaultClientErrorCode
	}
	if e.StatusCode > 0 {
		return fmt.Sprintf("%s: http %d", code, e.StatusCode)
	}
	return code
}

func IsStreamUnavailable(err error) bool {
	if err == nil {
		return false
	}
	var clientErr ClientError
	if errorAs(err, &clientErr) {
		return clientErr.Code == "llamacpp_stream_unavailable"
	}
	return strings.Contains(strings.ToLower(err.Error()), "llamacpp_stream_unavailable")
}

func (c OpenAIClient) Complete(ctx context.Context, req CompletionRequest) (CompletionResult, error) {
	req = normalizeCompletionRequest(req)
	if req.BaseURL == "" || !isLocalBaseURL(req.BaseURL) {
		return CompletionResult{}, ClientError{Code: "llamacpp_invalid_base_url"}
	}
	if strings.TrimSpace(req.Prompt) == "" {
		return CompletionResult{}, ClientError{Code: "llamacpp_prompt_required"}
	}

	started := c.now()
	body, err := json.Marshal(openAIChatRequest{
		Model: strings.TrimSpace(req.ModelID),
		Messages: []openAIChatMessage{
			{Role: "system", Content: "You are running a local llama.cpp readiness benchmark. Answer concisely."},
			{Role: "user", Content: req.Prompt},
		},
		Stream:      req.Stream,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
	})
	if err != nil {
		return CompletionResult{}, ClientError{Code: "llamacpp_request_marshal_failed"}
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(req.BaseURL, "/")+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return CompletionResult{}, ClientError{Code: "llamacpp_request_build_failed"}
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if req.Stream {
		httpReq.Header.Set("Accept", "text/event-stream")
	}

	client := c.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(httpReq)
	if err != nil {
		code := defaultClientErrorCode
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(ctx.Err(), context.DeadlineExceeded) {
			code = "llamacpp_timeout"
		}
		return CompletionResult{}, ClientError{Code: code}
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
		if req.Stream && streamUnsupportedStatus(resp.StatusCode) {
			return CompletionResult{}, ClientError{Code: "llamacpp_stream_unavailable", StatusCode: resp.StatusCode}
		}
		return CompletionResult{}, ClientError{Code: "llamacpp_response_status_failed", StatusCode: resp.StatusCode}
	}
	if req.Stream {
		contentType := strings.ToLower(strings.TrimSpace(resp.Header.Get("Content-Type")))
		if contentType != "" && !strings.Contains(contentType, "text/event-stream") {
			_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
			return CompletionResult{}, ClientError{Code: "llamacpp_stream_unavailable", StatusCode: resp.StatusCode}
		}
	}

	if req.Stream {
		return c.readStreamingCompletion(resp.Body, started, req.MaxTokens, req.OnDelta)
	}
	return c.readNonStreamingCompletion(resp.Body, started, req.MaxTokens)
}

func normalizeCompletionRequest(req CompletionRequest) CompletionRequest {
	req.BaseURL = strings.TrimRight(strings.TrimSpace(req.BaseURL), "/")
	req.ModelID = cleanStatusText(req.ModelID, maxStatusReasonLen)
	req.Prompt = strings.TrimSpace(req.Prompt)
	if req.MaxTokens <= 0 {
		req.MaxTokens = DefaultBenchmarkMaxTokens
	}
	if req.Temperature < 0 {
		req.Temperature = 0
	}
	return req
}

func streamUnsupportedStatus(status int) bool {
	switch status {
	case http.StatusBadRequest, http.StatusNotFound, http.StatusMethodNotAllowed, http.StatusNotAcceptable, http.StatusUnsupportedMediaType, http.StatusNotImplemented:
		return true
	default:
		return false
	}
}

func (c OpenAIClient) readStreamingCompletion(body io.Reader, started time.Time, requestedMaxTokens int, onDelta func(CompletionDelta) error) (CompletionResult, error) {
	var output bytes.Buffer
	var firstTokenAt time.Time
	var promptTokens int64
	var completionTokens int64
	var usageSeen bool
	var chunkTokens int64
	var finish completionFinishCapture

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			break
		}
		var chunk openAIChatStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return CompletionResult{}, ClientError{Code: "llamacpp_stream_decode_failed"}
		}
		if chunk.Error.Message != "" {
			return CompletionResult{}, ClientError{Code: "llamacpp_stream_error"}
		}
		finish.observeStreamChunk(chunk)
		if chunk.Usage != nil {
			usageSeen = true
			if chunk.Usage.PromptTokens >= 0 {
				promptTokens = chunk.Usage.PromptTokens
			}
			if chunk.Usage.CompletionTokens >= 0 {
				completionTokens = chunk.Usage.CompletionTokens
			}
		}
		for _, choice := range chunk.Choices {
			content := generatedTextFromStreamChoice(choice)
			if content == "" {
				continue
			}
			if firstTokenAt.IsZero() {
				firstTokenAt = c.now()
				if firstTokenAt.Before(started) {
					firstTokenAt = started
				}
			}
			if onDelta != nil {
				if err := onDelta(CompletionDelta{Text: content}); err != nil {
					return CompletionResult{}, ClientError{Code: "llamacpp_stream_delta_failed"}
				}
			}
			chunkTokens++
			output.WriteString(content)
		}
	}
	if err := scanner.Err(); err != nil {
		return CompletionResult{}, ClientError{Code: "llamacpp_stream_read_failed"}
	}
	if output.Len() == 0 {
		return CompletionResult{}, ClientError{Code: "llamacpp_empty_output"}
	}
	if completionTokens <= 0 && finish.tokensGenerated > 0 {
		completionTokens = finish.tokensGenerated
		usageSeen = true
	}
	if completionTokens <= 0 {
		completionTokens = chunkTokens
	}
	if completionTokens <= 0 {
		completionTokens = approximateGeneratedTokens(output.String())
	}
	finished := c.now()
	return buildCompletionResult(output.Bytes(), promptTokens, completionTokens, usageSeen, firstTokenAt, started, finished, true, finish.metadata(requestedMaxTokens, completionTokens)), nil
}

func (c OpenAIClient) readNonStreamingCompletion(body io.Reader, started time.Time, requestedMaxTokens int) (CompletionResult, error) {
	var payload openAIChatResponse
	if err := json.NewDecoder(io.LimitReader(body, 4*1024*1024)).Decode(&payload); err != nil {
		return CompletionResult{}, ClientError{Code: "llamacpp_response_decode_failed"}
	}
	if payload.Error.Message != "" {
		return CompletionResult{}, ClientError{Code: "llamacpp_response_error"}
	}
	var output bytes.Buffer
	output.WriteString(generatedTextFromChatResponse(payload))
	if output.Len() == 0 {
		return CompletionResult{}, ClientError{Code: "llamacpp_empty_output"}
	}
	var promptTokens int64
	var completionTokens int64
	var usageSeen bool
	if payload.Usage != nil {
		usageSeen = true
		if payload.Usage.PromptTokens >= 0 {
			promptTokens = payload.Usage.PromptTokens
		}
		if payload.Usage.CompletionTokens >= 0 {
			completionTokens = payload.Usage.CompletionTokens
		}
	}
	if promptTokens <= 0 && payload.TokensEvaluated > 0 {
		promptTokens = payload.TokensEvaluated
		usageSeen = true
	}
	if completionTokens <= 0 {
		if generated := backendGeneratedTokens(payload.TokensGenerated, payload.TokensPredicted, payload.Timings); generated > 0 {
			completionTokens = generated
			usageSeen = true
		}
	}
	if completionTokens <= 0 {
		completionTokens = approximateGeneratedTokens(output.String())
	}
	finished := c.now()
	return buildCompletionResult(output.Bytes(), promptTokens, completionTokens, usageSeen, finished, started, finished, false, finishMetadataFromChatResponse(payload, requestedMaxTokens, completionTokens)), nil
}

func buildCompletionResult(output []byte, promptTokens int64, completionTokens int64, usageSeen bool, firstTokenAt time.Time, started time.Time, finished time.Time, streamed bool, finish FinishMetadata) CompletionResult {
	if finished.Before(started) {
		finished = started
	}
	if firstTokenAt.IsZero() || firstTokenAt.Before(started) {
		firstTokenAt = started
	}
	if firstTokenAt.After(finished) {
		firstTokenAt = finished
	}
	result := CompletionResult{
		Output:              append([]byte(nil), output...),
		OutputBytes:         int64(len(output)),
		RequestedMaxTokens:  finish.RequestedMaxTokens,
		TokensGenerated:     completionTokens,
		FinishReason:        finish.FinishReason,
		BackendFinishReason: finish.BackendFinishReason,
		BackendStopReason:   finish.BackendStopReason,
		MaxTokensReached:    finish.MaxTokensReached,
		TTFTMs:              firstTokenAt.Sub(started).Milliseconds(),
		TotalTimeMs:         finished.Sub(started).Milliseconds(),
		Streamed:            streamed,
	}
	if usageSeen {
		result.PromptTokens = promptTokens
		result.CompletionTokens = completionTokens
	}
	return result
}

func approximateGeneratedTokens(output string) int64 {
	count := int64(len(strings.Fields(output)))
	if count == 0 && strings.TrimSpace(output) != "" {
		count = 1
	}
	return count
}

func (c OpenAIClient) now() time.Time {
	if c.Now != nil {
		return c.Now()
	}
	return time.Now()
}

type openAIChatRequest struct {
	Model       string              `json:"model,omitempty"`
	Messages    []openAIChatMessage `json:"messages"`
	Stream      bool                `json:"stream"`
	MaxTokens   int                 `json:"max_tokens"`
	Temperature float64             `json:"temperature"`
}

type openAIChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIUsage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

type openAIError struct {
	Message string `json:"message"`
}

type openAIChatStreamChunk struct {
	Choices         []openAIChatStreamChoice `json:"choices"`
	Usage           *openAIUsage             `json:"usage,omitempty"`
	Error           openAIError              `json:"error,omitempty"`
	FinishReason    string                   `json:"finish_reason,omitempty"`
	Stop            json.RawMessage          `json:"stop,omitempty"`
	StoppedEOS      bool                     `json:"stopped_eos,omitempty"`
	StoppedLimit    bool                     `json:"stopped_limit,omitempty"`
	TimedOut        bool                     `json:"timed_out,omitempty"`
	Timeout         bool                     `json:"timeout,omitempty"`
	TokensEvaluated int64                    `json:"tokens_evaluated,omitempty"`
	TokensGenerated int64                    `json:"tokens_generated,omitempty"`
	TokensPredicted int64                    `json:"tokens_predicted,omitempty"`
	Timings         *llamaTimings            `json:"timings,omitempty"`
}

type openAIChatStreamChoice struct {
	Delta struct {
		Content string `json:"content"`
	} `json:"delta"`
	Content      string `json:"content"`
	Text         string `json:"text"`
	FinishReason string `json:"finish_reason,omitempty"`
}

type openAIChatResponse struct {
	Choices         []openAIChatChoice `json:"choices"`
	Content         string             `json:"content"`
	Response        string             `json:"response"`
	Text            string             `json:"text"`
	Usage           *openAIUsage       `json:"usage,omitempty"`
	Error           openAIError        `json:"error,omitempty"`
	FinishReason    string             `json:"finish_reason,omitempty"`
	Stop            json.RawMessage    `json:"stop,omitempty"`
	StoppedEOS      bool               `json:"stopped_eos,omitempty"`
	StoppedLimit    bool               `json:"stopped_limit,omitempty"`
	TimedOut        bool               `json:"timed_out,omitempty"`
	Timeout         bool               `json:"timeout,omitempty"`
	TokensEvaluated int64              `json:"tokens_evaluated,omitempty"`
	TokensGenerated int64              `json:"tokens_generated,omitempty"`
	TokensPredicted int64              `json:"tokens_predicted,omitempty"`
	Timings         *llamaTimings      `json:"timings,omitempty"`
}

type openAIChatChoice struct {
	Message struct {
		Content string `json:"content"`
	} `json:"message"`
	Content      string `json:"content"`
	Text         string `json:"text"`
	FinishReason string `json:"finish_reason,omitempty"`
}

type llamaTimings struct {
	PredictedN int64 `json:"predicted_n,omitempty"`
}

func generatedTextFromStreamChoice(choice openAIChatStreamChoice) string {
	switch {
	case choice.Delta.Content != "":
		return choice.Delta.Content
	case choice.Content != "":
		return choice.Content
	default:
		return choice.Text
	}
}

func generatedTextFromChatResponse(payload openAIChatResponse) string {
	var output strings.Builder
	for _, choice := range payload.Choices {
		output.WriteString(generatedTextFromChatChoice(choice))
	}
	if output.Len() > 0 {
		return output.String()
	}
	switch {
	case payload.Content != "":
		return payload.Content
	case payload.Response != "":
		return payload.Response
	default:
		return payload.Text
	}
}

func generatedTextFromChatChoice(choice openAIChatChoice) string {
	switch {
	case choice.Message.Content != "":
		return choice.Message.Content
	case choice.Content != "":
		return choice.Content
	default:
		return choice.Text
	}
}
