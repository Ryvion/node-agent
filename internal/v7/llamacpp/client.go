package llamacpp

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const defaultClientErrorCode = "llamacpp_request_failed"

const (
	RuntimeMeasurementStatusMeasured = "measured"
	RuntimeMeasurementStatusPartial  = "partial"
	RuntimeMeasurementStatusUnknown  = "unknown"

	MetadataParseStatusOK      = "ok"
	MetadataParseStatusPartial = "partial"

	PromptModeChatMessages  = "chat_messages"
	PromptModeTemplate      = "template"
	PromptModeRawCompletion = "raw_completion"
)

const defaultCompletionSystemPrompt = "You are running a local llama.cpp readiness benchmark. Answer concisely."

type CompletionRequest struct {
	BaseURL      string
	ModelID      string
	Prompt       string
	SystemPrompt string
	Messages     []CompletionMessage
	MaxTokens    int
	Temperature  float64
	Stream       bool
	OnDelta      func(CompletionDelta) error
}

type CompletionMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type CompletionDelta struct {
	Text string
}

type CompletionResult struct {
	Output                   []byte  `json:"-"`
	OutputBytes              int64   `json:"output_bytes"`
	PromptTokens             int64   `json:"prompt_tokens,omitempty"`
	CompletionTokens         int64   `json:"completion_tokens,omitempty"`
	RequestedMaxTokens       int     `json:"requested_max_tokens"`
	TokensGenerated          int64   `json:"tokens_generated"`
	FinishReason             string  `json:"finish_reason"`
	BackendFinishReason      string  `json:"backend_finish_reason"`
	BackendStopReason        string  `json:"backend_stop_reason"`
	MaxTokensReached         bool    `json:"max_tokens_reached"`
	RuntimeMeasurementStatus string  `json:"runtime_measurement_status"`
	MetadataParseStatus      string  `json:"metadata_parse_status"`
	TokenCountEstimated      bool    `json:"token_count_estimated,omitempty"`
	PromptMode               string  `json:"prompt_mode,omitempty"`
	SystemPromptHash         string  `json:"system_prompt_hash,omitempty"`
	TTFTMs                   int64   `json:"ttft_ms"`
	TotalTimeMs              int64   `json:"total_time_ms"`
	BackendDecodeMs          float64 `json:"backend_decode_ms,omitempty"`
	BackendDecodeTPS         float64 `json:"backend_decode_tps,omitempty"`
	Streamed                 bool    `json:"streamed"`

	// V8 Phase 1.2: speculative-decoding runtime counts.
	// Populated when llama-server reports them via the timings block;
	// zero on llama-server builds that don't surface the values.
	SpeculativeTokensDrafted  int64 `json:"speculative_tokens_drafted,omitempty"`
	SpeculativeTokensAccepted int64 `json:"speculative_tokens_accepted,omitempty"`
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
	if strings.TrimSpace(req.Prompt) == "" && len(req.Messages) == 0 {
		return CompletionResult{}, ClientError{Code: "llamacpp_prompt_required"}
	}

	result, err := c.completeChat(ctx, req)
	if err == nil {
		return result, nil
	}
	if !isChatCompletionUnavailable(err) {
		return CompletionResult{}, err
	}
	return c.completeRaw(ctx, req)
}

func (c OpenAIClient) completeChat(ctx context.Context, req CompletionRequest) (CompletionResult, error) {
	started := c.now()
	body, err := json.Marshal(openAIChatRequest{
		Model:       strings.TrimSpace(req.ModelID),
		Messages:    openAIChatMessages(req.Messages),
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
		if endpointUnsupportedStatus(resp.StatusCode) {
			return CompletionResult{}, ClientError{Code: "llamacpp_chat_completion_unavailable", StatusCode: resp.StatusCode}
		}
		return CompletionResult{}, ClientError{Code: "llamacpp_response_status_failed", StatusCode: resp.StatusCode}
	}
	if req.Stream {
		contentType := strings.ToLower(strings.TrimSpace(resp.Header.Get("Content-Type")))
		if contentType != "" && !strings.Contains(contentType, "text/event-stream") {
			_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
			return CompletionResult{}, ClientError{Code: "llamacpp_chat_completion_unavailable", StatusCode: resp.StatusCode}
		}
	}

	if req.Stream {
		result, err := c.readStreamingCompletion(resp.Body, started, req.MaxTokens, req.OnDelta)
		if err != nil {
			return CompletionResult{}, err
		}
		result.PromptMode = PromptModeChatMessages
		result.SystemPromptHash = effectiveSystemPromptHash(req)
		return result, nil
	}
	result, err := c.readNonStreamingCompletion(resp.Body, started, req.MaxTokens)
	if err != nil {
		return CompletionResult{}, err
	}
	result.PromptMode = PromptModeChatMessages
	result.SystemPromptHash = effectiveSystemPromptHash(req)
	return result, nil
}

func (c OpenAIClient) completeRaw(ctx context.Context, req CompletionRequest) (CompletionResult, error) {
	started := c.now()
	prompt, promptMode := rawCompletionPrompt(req)
	body, err := json.Marshal(rawCompletionRequest{
		Prompt:      prompt,
		NPredict:    req.MaxTokens,
		Temperature: req.Temperature,
		Stream:      req.Stream,
	})
	if err != nil {
		return CompletionResult{}, ClientError{Code: "llamacpp_request_marshal_failed"}
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(req.BaseURL, "/")+"/completion", bytes.NewReader(body))
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
		if req.Stream && endpointUnsupportedStatus(resp.StatusCode) {
			return CompletionResult{}, ClientError{Code: "llamacpp_stream_unavailable", StatusCode: resp.StatusCode}
		}
		return CompletionResult{}, ClientError{Code: "llamacpp_completion_response_status_failed", StatusCode: resp.StatusCode}
	}
	if req.Stream {
		contentType := strings.ToLower(strings.TrimSpace(resp.Header.Get("Content-Type")))
		if contentType != "" && !strings.Contains(contentType, "text/event-stream") {
			_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
			return CompletionResult{}, ClientError{Code: "llamacpp_stream_unavailable", StatusCode: resp.StatusCode}
		}
	}

	if req.Stream {
		result, err := c.readStreamingCompletion(resp.Body, started, req.MaxTokens, req.OnDelta)
		if err != nil {
			return CompletionResult{}, err
		}
		result.PromptMode = promptMode
		result.SystemPromptHash = effectiveSystemPromptHash(req)
		return result, nil
	}
	result, err := c.readNonStreamingCompletion(resp.Body, started, req.MaxTokens)
	if err != nil {
		return CompletionResult{}, err
	}
	result.PromptMode = promptMode
	result.SystemPromptHash = effectiveSystemPromptHash(req)
	return result, nil
}

func normalizeCompletionRequest(req CompletionRequest) CompletionRequest {
	req.BaseURL = strings.TrimRight(strings.TrimSpace(req.BaseURL), "/")
	req.ModelID = cleanStatusText(req.ModelID, maxStatusReasonLen)
	req.Prompt = strings.TrimSpace(req.Prompt)
	req.SystemPrompt = strings.TrimSpace(req.SystemPrompt)
	req.Messages = normalizeCompletionMessages(req.Messages)
	if len(req.Messages) == 0 && req.SystemPrompt == "" {
		req.SystemPrompt = defaultCompletionSystemPrompt
	}
	if len(req.Messages) == 0 {
		req.Messages = buildCompletionMessages(req.SystemPrompt, req.Prompt)
	} else if req.SystemPrompt != "" {
		req.Messages = withLeadingSystemPrompt(req.SystemPrompt, req.Messages)
	}
	if req.MaxTokens <= 0 {
		req.MaxTokens = DefaultBenchmarkMaxTokens
	}
	if req.Temperature < 0 {
		req.Temperature = 0
	}
	return req
}

func normalizeCompletionMessages(messages []CompletionMessage) []CompletionMessage {
	if len(messages) == 0 {
		return nil
	}
	normalized := make([]CompletionMessage, 0, len(messages))
	for _, message := range messages {
		role := normalizeChatRole(message.Role)
		content := strings.TrimSpace(message.Content)
		if role == "" || content == "" {
			continue
		}
		normalized = append(normalized, CompletionMessage{Role: role, Content: content})
	}
	return normalized
}

func normalizeChatRole(role string) string {
	switch strings.ToLower(strings.TrimSpace(role)) {
	case "system", "user", "assistant":
		return strings.ToLower(strings.TrimSpace(role))
	default:
		return ""
	}
}

func buildCompletionMessages(systemPrompt, prompt string) []CompletionMessage {
	messages := make([]CompletionMessage, 0, 2)
	if systemPrompt = strings.TrimSpace(systemPrompt); systemPrompt != "" {
		messages = append(messages, CompletionMessage{Role: "system", Content: systemPrompt})
	}
	if prompt = strings.TrimSpace(prompt); prompt != "" {
		messages = append(messages, CompletionMessage{Role: "user", Content: prompt})
	}
	return messages
}

func withLeadingSystemPrompt(systemPrompt string, messages []CompletionMessage) []CompletionMessage {
	systemPrompt = strings.TrimSpace(systemPrompt)
	if systemPrompt == "" {
		return messages
	}
	out := make([]CompletionMessage, 0, len(messages)+1)
	out = append(out, CompletionMessage{Role: "system", Content: systemPrompt})
	for _, message := range messages {
		if message.Role == "system" {
			continue
		}
		out = append(out, message)
	}
	return out
}

func openAIChatMessages(messages []CompletionMessage) []openAIChatMessage {
	out := make([]openAIChatMessage, 0, len(messages))
	for _, message := range messages {
		out = append(out, openAIChatMessage{Role: message.Role, Content: message.Content})
	}
	return out
}

func effectiveSystemPromptHash(req CompletionRequest) string {
	if hash := HashSystemPrompt(req.SystemPrompt); hash != "" {
		return hash
	}
	for _, message := range req.Messages {
		if message.Role == "system" {
			return HashSystemPrompt(message.Content)
		}
	}
	return ""
}

func HashSystemPrompt(systemPrompt string) string {
	systemPrompt = strings.TrimSpace(systemPrompt)
	if systemPrompt == "" {
		return ""
	}
	sum := sha256.Sum256([]byte("ryvion:v7:llamacpp_system_prompt:v1\n" + systemPrompt))
	return "sha256:" + hex.EncodeToString(sum[:])
}

func endpointUnsupportedStatus(status int) bool {
	switch status {
	case http.StatusBadRequest, http.StatusNotFound, http.StatusMethodNotAllowed, http.StatusNotAcceptable, http.StatusUnsupportedMediaType, http.StatusNotImplemented:
		return true
	default:
		return false
	}
}

func isChatCompletionUnavailable(err error) bool {
	if err == nil {
		return false
	}
	var clientErr ClientError
	if errorAs(err, &clientErr) {
		return clientErr.Code == "llamacpp_chat_completion_unavailable"
	}
	return strings.Contains(strings.ToLower(err.Error()), "llamacpp_chat_completion_unavailable")
}

func (c OpenAIClient) readStreamingCompletion(body io.Reader, started time.Time, requestedMaxTokens int, onDelta func(CompletionDelta) error) (CompletionResult, error) {
	var output bytes.Buffer
	var firstTokenAt time.Time
	var promptTokens int64
	var usageCompletionTokens int64
	var chunkTokens int64
	var finish completionFinishCapture
	// V8 Phase 1.2: capture speculative-decoding counts from any chunk
	// that carries a Timings block (typically the terminal one).
	var specDrafted, specAccepted int64
	var backendDecodeMs float64
	var backendDecodeTPS float64

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
			if chunk.Usage.PromptTokens >= 0 {
				promptTokens = chunk.Usage.PromptTokens
			}
			if chunk.Usage.CompletionTokens >= 0 {
				usageCompletionTokens = chunk.Usage.CompletionTokens
			}
		}
		if d, a := chunk.Timings.speculativeCounts(); d > 0 || a > 0 {
			specDrafted, specAccepted = d, a
		}
		if ms, tps := chunk.Timings.decodeStats(); ms > 0 || tps > 0 {
			backendDecodeMs, backendDecodeTPS = ms, tps
		}
		content := generatedTextFromStreamChunk(chunk)
		if content != "" {
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
	measurement := normalizeCompletionTokenMeasurement(completionTokenSources{
		UsageCompletionTokens: usageCompletionTokens,
		TimingTokens:          finish.timingTokens,
		StreamedDeltaTokens:   chunkTokens,
		BackendTokenCount:     finish.backendTokenCount,
		GeneratedText:         output.String(),
	})
	finished := c.now()
	result := buildCompletionResult(output.Bytes(), promptTokens, measurement, firstTokenAt, started, finished, true, finish.metadata(requestedMaxTokens, measurement.TokensGenerated))
	result.applyBackendDecodeStats(backendDecodeMs, backendDecodeTPS)
	result.SpeculativeTokensDrafted = specDrafted
	result.SpeculativeTokensAccepted = specAccepted
	return result, nil
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
	var usageCompletionTokens int64
	if payload.Usage != nil {
		if payload.Usage.PromptTokens >= 0 {
			promptTokens = payload.Usage.PromptTokens
		}
		if payload.Usage.CompletionTokens >= 0 {
			usageCompletionTokens = payload.Usage.CompletionTokens
		}
	}
	if promptTokens <= 0 && payload.TokensEvaluated > 0 {
		promptTokens = payload.TokensEvaluated
	}
	measurement := normalizeCompletionTokenMeasurement(completionTokenSources{
		UsageCompletionTokens: usageCompletionTokens,
		TimingTokens:          timingGeneratedTokens(payload.Timings, payload.PredictedN, payload.NPredicted),
		BackendTokenCount:     backendReturnedTokenCount(payload.TokensGenerated, payload.TokensPredicted, payload.CompletionTokens),
		GeneratedText:         output.String(),
	})
	finished := c.now()
	result := buildCompletionResult(output.Bytes(), promptTokens, measurement, finished, started, finished, false, finishMetadataFromChatResponse(payload, requestedMaxTokens, measurement.TokensGenerated))
	result.applyBackendDecodeStats(payload.Timings.decodeStats())
	result.SpeculativeTokensDrafted, result.SpeculativeTokensAccepted = payload.Timings.speculativeCounts()
	return result, nil
}

func buildCompletionResult(output []byte, promptTokens int64, measurement completionTokenMeasurement, firstTokenAt time.Time, started time.Time, finished time.Time, streamed bool, finish FinishMetadata) CompletionResult {
	if finished.Before(started) {
		finished = started
	}
	if firstTokenAt.IsZero() || firstTokenAt.Before(started) {
		firstTokenAt = started
	}
	if firstTokenAt.After(finished) {
		firstTokenAt = finished
	}
	completionTokens := measurement.TokensGenerated
	result := CompletionResult{
		Output:                   append([]byte(nil), output...),
		OutputBytes:              int64(len(output)),
		RequestedMaxTokens:       finish.RequestedMaxTokens,
		TokensGenerated:          completionTokens,
		FinishReason:             finish.FinishReason,
		BackendFinishReason:      finish.BackendFinishReason,
		BackendStopReason:        finish.BackendStopReason,
		MaxTokensReached:         finish.MaxTokensReached,
		RuntimeMeasurementStatus: measurement.RuntimeMeasurementStatus,
		MetadataParseStatus:      measurement.MetadataParseStatus,
		TokenCountEstimated:      measurement.TokenCountEstimated,
		TTFTMs:                   firstTokenAt.Sub(started).Milliseconds(),
		TotalTimeMs:              finished.Sub(started).Milliseconds(),
		Streamed:                 streamed,
	}
	if promptTokens > 0 {
		result.PromptTokens = promptTokens
	}
	if measurement.ReportCompletionTokens {
		result.CompletionTokens = completionTokens
	}
	return result
}

func (result *CompletionResult) applyBackendDecodeStats(decodeMs float64, decodeTPS float64) {
	if result == nil {
		return
	}
	if decodeMs > 0 {
		result.BackendDecodeMs = decodeMs
	}
	if decodeTPS > 0 {
		result.BackendDecodeTPS = decodeTPS
		return
	}
	if decodeMs > 0 && result.TokensGenerated > 0 {
		result.BackendDecodeTPS = float64(result.TokensGenerated) / (decodeMs / 1000)
	}
}

type completionTokenSources struct {
	UsageCompletionTokens int64
	TimingTokens          int64
	StreamedDeltaTokens   int64
	BackendTokenCount     int64
	GeneratedText         string
}

type completionTokenMeasurement struct {
	TokensGenerated          int64
	RuntimeMeasurementStatus string
	MetadataParseStatus      string
	TokenCountEstimated      bool
	ReportCompletionTokens   bool
}

func normalizeCompletionTokenMeasurement(src completionTokenSources) completionTokenMeasurement {
	switch {
	case src.UsageCompletionTokens > 0:
		return measuredCompletionTokens(src.UsageCompletionTokens, true)
	case src.TimingTokens > 0:
		return measuredCompletionTokens(src.TimingTokens, true)
	case src.StreamedDeltaTokens > 0:
		return measuredCompletionTokens(src.StreamedDeltaTokens, true)
	case src.BackendTokenCount > 0:
		return measuredCompletionTokens(src.BackendTokenCount, true)
	}
	if estimated := approximateGeneratedTokens(src.GeneratedText); estimated > 0 {
		return completionTokenMeasurement{
			TokensGenerated:          estimated,
			RuntimeMeasurementStatus: RuntimeMeasurementStatusPartial,
			MetadataParseStatus:      MetadataParseStatusOK,
			TokenCountEstimated:      true,
		}
	}
	return completionTokenMeasurement{
		RuntimeMeasurementStatus: RuntimeMeasurementStatusUnknown,
		MetadataParseStatus:      MetadataParseStatusPartial,
	}
}

func measuredCompletionTokens(tokens int64, report bool) completionTokenMeasurement {
	return completionTokenMeasurement{
		TokensGenerated:          tokens,
		RuntimeMeasurementStatus: RuntimeMeasurementStatusMeasured,
		MetadataParseStatus:      MetadataParseStatusOK,
		ReportCompletionTokens:   report,
	}
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

type rawCompletionRequest struct {
	Prompt      string  `json:"prompt"`
	NPredict    int     `json:"n_predict"`
	Temperature float64 `json:"temperature"`
	Stream      bool    `json:"stream"`
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
	Choices          []openAIChatStreamChoice `json:"choices"`
	Content          string                   `json:"content"`
	Response         string                   `json:"response"`
	Text             string                   `json:"text"`
	Usage            *openAIUsage             `json:"usage,omitempty"`
	Error            openAIError              `json:"error,omitempty"`
	FinishReason     string                   `json:"finish_reason,omitempty"`
	Stop             json.RawMessage          `json:"stop,omitempty"`
	StoppedEOS       bool                     `json:"stopped_eos,omitempty"`
	StoppedLimit     bool                     `json:"stopped_limit,omitempty"`
	TimedOut         bool                     `json:"timed_out,omitempty"`
	Timeout          bool                     `json:"timeout,omitempty"`
	TokensEvaluated  int64                    `json:"tokens_evaluated,omitempty"`
	CompletionTokens int64                    `json:"completion_tokens,omitempty"`
	TokensGenerated  int64                    `json:"tokens_generated,omitempty"`
	TokensPredicted  int64                    `json:"tokens_predicted,omitempty"`
	PredictedN       int64                    `json:"predicted_n,omitempty"`
	NPredicted       int64                    `json:"n_predicted,omitempty"`
	Timings          *llamaTimings            `json:"timings,omitempty"`
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
	Choices          []openAIChatChoice `json:"choices"`
	Content          string             `json:"content"`
	Response         string             `json:"response"`
	Text             string             `json:"text"`
	Usage            *openAIUsage       `json:"usage,omitempty"`
	Error            openAIError        `json:"error,omitempty"`
	FinishReason     string             `json:"finish_reason,omitempty"`
	Stop             json.RawMessage    `json:"stop,omitempty"`
	StoppedEOS       bool               `json:"stopped_eos,omitempty"`
	StoppedLimit     bool               `json:"stopped_limit,omitempty"`
	TimedOut         bool               `json:"timed_out,omitempty"`
	Timeout          bool               `json:"timeout,omitempty"`
	TokensEvaluated  int64              `json:"tokens_evaluated,omitempty"`
	CompletionTokens int64              `json:"completion_tokens,omitempty"`
	TokensGenerated  int64              `json:"tokens_generated,omitempty"`
	TokensPredicted  int64              `json:"tokens_predicted,omitempty"`
	PredictedN       int64              `json:"predicted_n,omitempty"`
	NPredicted       int64              `json:"n_predicted,omitempty"`
	Timings          *llamaTimings      `json:"timings,omitempty"`
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
	PredictedN         int64   `json:"predicted_n,omitempty"`
	NPredicted         int64   `json:"n_predicted,omitempty"`
	PredictedMs        float64 `json:"predicted_ms,omitempty"`
	PredictedPerSecond float64 `json:"predicted_per_second,omitempty"`

	// V8 Phase 1.2: speculative-decoding counts emitted by llama-server
	// when running with --model-draft. Older builds may emit only one
	// of these; we read whichever is present.
	NDrafted       int64 `json:"n_drafted,omitempty"`
	NAccepted      int64 `json:"n_accepted,omitempty"`
	DraftN         int64 `json:"draft_n,omitempty"`
	DraftNAccepted int64 `json:"draft_n_accepted,omitempty"`
}

func (t *llamaTimings) decodeStats() (float64, float64) {
	if t == nil {
		return 0, 0
	}
	ms := t.PredictedMs
	if ms < 0 {
		ms = 0
	}
	tps := t.PredictedPerSecond
	if tps < 0 {
		tps = 0
	}
	return ms, tps
}

// speculativeCounts returns (drafted, accepted) from any name variant
// llama-server may use across versions.
func (t *llamaTimings) speculativeCounts() (int64, int64) {
	if t == nil {
		return 0, 0
	}
	drafted := t.NDrafted
	if drafted == 0 {
		drafted = t.DraftN
	}
	accepted := t.NAccepted
	if accepted == 0 {
		accepted = t.DraftNAccepted
	}
	if drafted < 0 {
		drafted = 0
	}
	if accepted < 0 {
		accepted = 0
	}
	return drafted, accepted
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

func generatedTextFromStreamChunk(chunk openAIChatStreamChunk) string {
	var output strings.Builder
	for _, choice := range chunk.Choices {
		output.WriteString(generatedTextFromStreamChoice(choice))
	}
	if output.Len() > 0 {
		return output.String()
	}
	switch {
	case chunk.Content != "":
		return chunk.Content
	case chunk.Response != "":
		return chunk.Response
	default:
		return chunk.Text
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

func rawCompletionPrompt(req CompletionRequest) (string, string) {
	if isLlama3InstructModel(req.ModelID) {
		return llama3InstructPrompt(req.Messages), PromptModeTemplate
	}
	return genericRawCompletionPrompt(req.Messages), PromptModeRawCompletion
}

func isLlama3InstructModel(modelID string) bool {
	value := strings.ToLower(strings.TrimSpace(modelID))
	value = strings.NewReplacer("_", "-", " ", "-").Replace(value)
	if !strings.Contains(value, "instruct") {
		return false
	}
	return strings.Contains(value, "llama-3") || strings.Contains(value, "llama3")
}

func llama3InstructPrompt(messages []CompletionMessage) string {
	var out strings.Builder
	out.WriteString("<|begin_of_text|>")
	for _, message := range messages {
		role := normalizeChatRole(message.Role)
		content := strings.TrimSpace(message.Content)
		if role == "" || content == "" {
			continue
		}
		out.WriteString("<|start_header_id|>")
		out.WriteString(role)
		out.WriteString("<|end_header_id|>\n\n")
		out.WriteString(content)
		out.WriteString("<|eot_id|>")
	}
	out.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
	return out.String()
}

func genericRawCompletionPrompt(messages []CompletionMessage) string {
	var out strings.Builder
	for _, message := range messages {
		role := normalizeChatRole(message.Role)
		content := strings.TrimSpace(message.Content)
		if role == "" || content == "" {
			continue
		}
		out.WriteString(strings.ToUpper(role[:1]))
		out.WriteString(role[1:])
		out.WriteString(": ")
		out.WriteString(content)
		out.WriteString("\n\n")
	}
	out.WriteString("Assistant:")
	return out.String()
}
