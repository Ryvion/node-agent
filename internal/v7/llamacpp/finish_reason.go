package llamacpp

import (
	"encoding/json"
	"strings"
)

const (
	FinishReasonStop      = "stop"
	FinishReasonLength    = "length"
	FinishReasonMaxTokens = "max_tokens"
	FinishReasonTimeout   = "timeout"
	FinishReasonError     = "error"
	FinishReasonUnknown   = "unknown"
)

type FinishMetadata struct {
	RequestedMaxTokens  int
	TokensGenerated     int64
	FinishReason        string
	BackendFinishReason string
	BackendStopReason   string
	MaxTokensReached    bool
}

type finishMetadataInput struct {
	RequestedMaxTokens  int
	TokensGenerated     int64
	BackendFinishReason string
	BackendStopReason   string
	StoppedEOS          bool
	StoppedLimit        bool
	TimedOut            bool
	StopSeen            bool
}

type completionFinishCapture struct {
	backendFinishReason string
	backendStopReason   string
	stoppedEOS          bool
	stoppedLimit        bool
	timedOut            bool
	stopSeen            bool
	tokensGenerated     int64
}

func NormalizeCompletionFinishMetadata(result CompletionResult, requestedMaxTokens int, tokensGenerated int64) FinishMetadata {
	if result.RequestedMaxTokens > 0 {
		requestedMaxTokens = result.RequestedMaxTokens
	}
	if result.TokensGenerated > 0 {
		tokensGenerated = result.TokensGenerated
	}
	backendFinish := firstNonEmptyFinishText(result.BackendFinishReason, result.FinishReason)
	metadata := normalizeFinishMetadata(finishMetadataInput{
		RequestedMaxTokens:  requestedMaxTokens,
		TokensGenerated:     tokensGenerated,
		BackendFinishReason: backendFinish,
		BackendStopReason:   result.BackendStopReason,
	})
	if normalized := normalizeFinishReason(result.FinishReason); normalized != "" {
		metadata.FinishReason = normalized
		if normalized == FinishReasonLength || normalized == FinishReasonMaxTokens {
			metadata.MaxTokensReached = true
		}
	}
	if result.MaxTokensReached {
		metadata.MaxTokensReached = true
	}
	return metadata
}

func finishMetadataFromChatResponse(payload openAIChatResponse, requestedMaxTokens int, tokensGenerated int64) FinishMetadata {
	backendFinish := payload.FinishReason
	for _, choice := range payload.Choices {
		backendFinish = firstNonEmptyFinishText(backendFinish, choice.FinishReason)
	}
	backendStop, stopSeen := backendStopReasonFromRaw(payload.Stop)
	return normalizeFinishMetadata(finishMetadataInput{
		RequestedMaxTokens:  requestedMaxTokens,
		TokensGenerated:     tokensGenerated,
		BackendFinishReason: backendFinish,
		BackendStopReason:   backendStop,
		StoppedEOS:          payload.StoppedEOS,
		StoppedLimit:        payload.StoppedLimit,
		TimedOut:            payload.TimedOut || payload.Timeout,
		StopSeen:            stopSeen,
	})
}

func (c *completionFinishCapture) observeStreamChunk(chunk openAIChatStreamChunk) {
	if c == nil {
		return
	}
	c.backendFinishReason = firstNonEmptyFinishText(c.backendFinishReason, chunk.FinishReason)
	for _, choice := range chunk.Choices {
		c.backendFinishReason = firstNonEmptyFinishText(c.backendFinishReason, choice.FinishReason)
	}
	if stopReason, stopSeen := backendStopReasonFromRaw(chunk.Stop); stopSeen {
		c.backendStopReason = firstNonEmptyFinishText(c.backendStopReason, stopReason)
		c.stopSeen = true
	}
	c.stoppedEOS = c.stoppedEOS || chunk.StoppedEOS
	c.stoppedLimit = c.stoppedLimit || chunk.StoppedLimit
	c.timedOut = c.timedOut || chunk.TimedOut || chunk.Timeout
	if generated := backendGeneratedTokens(chunk.TokensGenerated, chunk.TokensPredicted, chunk.Timings); generated > 0 {
		c.tokensGenerated = generated
	}
}

func (c completionFinishCapture) metadata(requestedMaxTokens int, tokensGenerated int64) FinishMetadata {
	if tokensGenerated <= 0 && c.tokensGenerated > 0 {
		tokensGenerated = c.tokensGenerated
	}
	return normalizeFinishMetadata(finishMetadataInput{
		RequestedMaxTokens:  requestedMaxTokens,
		TokensGenerated:     tokensGenerated,
		BackendFinishReason: c.backendFinishReason,
		BackendStopReason:   c.backendStopReason,
		StoppedEOS:          c.stoppedEOS,
		StoppedLimit:        c.stoppedLimit,
		TimedOut:            c.timedOut,
		StopSeen:            c.stopSeen,
	})
}

func normalizeFinishMetadata(input finishMetadataInput) FinishMetadata {
	requested := input.RequestedMaxTokens
	if requested < 0 {
		requested = 0
	}
	tokens := input.TokensGenerated
	if tokens < 0 {
		tokens = 0
	}
	backendFinish := cleanFinishText(input.BackendFinishReason)
	backendStop := cleanFinishText(input.BackendStopReason)
	finish := normalizeFinishReason(backendFinish)
	hasBackendSignal := (backendFinish != "" && backendFinish != FinishReasonUnknown) ||
		(backendStop != "" && backendStop != FinishReasonUnknown) ||
		input.StoppedEOS || input.StoppedLimit || input.TimedOut || input.StopSeen

	switch {
	case input.TimedOut:
		finish = FinishReasonTimeout
		backendStop = firstNonEmptyFinishText(backendStop, "timeout")
	case input.StoppedLimit:
		finish = FinishReasonLength
		backendStop = firstNonEmptyFinishText(backendStop, "stopped_limit")
	case input.StoppedEOS:
		finish = FinishReasonStop
		backendStop = firstNonEmptyFinishText(backendStop, "stopped_eos")
	case finish == "":
		finish = normalizeFinishReason(backendStop)
	}
	if finish == "" {
		finish = FinishReasonUnknown
	}
	maxTokensReached := finish == FinishReasonLength || finish == FinishReasonMaxTokens || input.StoppedLimit
	if !hasBackendSignal && requested > 0 && tokens >= int64(requested) {
		maxTokensReached = true
	}
	if backendFinish == "" {
		backendFinish = FinishReasonUnknown
	}
	if backendStop == "" {
		backendStop = FinishReasonUnknown
	}
	return FinishMetadata{
		RequestedMaxTokens:  requested,
		TokensGenerated:     tokens,
		FinishReason:        finish,
		BackendFinishReason: backendFinish,
		BackendStopReason:   backendStop,
		MaxTokensReached:    maxTokensReached,
	}
}

func normalizeFinishReason(value string) string {
	value = cleanFinishText(value)
	switch value {
	case "":
		return ""
	case FinishReasonStop, "stopped", "eos", "stopped_eos", "end_of_sequence", "end-of-sequence", "end_of_text", "eos_token":
		return FinishReasonStop
	case FinishReasonLength, "limit", "stopped_limit", "token_limit", "context_length", "max_new_tokens":
		return FinishReasonLength
	case FinishReasonMaxTokens, "max_token", "max_tokens_reached":
		return FinishReasonMaxTokens
	case FinishReasonTimeout, "timed_out", "deadline", "deadline_exceeded":
		return FinishReasonTimeout
	case FinishReasonError:
		return FinishReasonError
	case FinishReasonUnknown:
		return FinishReasonUnknown
	default:
		return ""
	}
}

func backendStopReasonFromRaw(raw json.RawMessage) (string, bool) {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return "", false
	}
	lower := strings.ToLower(trimmed)
	switch lower {
	case "false", "0":
		return "", false
	case "true", "1":
		return FinishReasonStop, true
	}
	if strings.HasPrefix(trimmed, `"`) {
		var text string
		if json.Unmarshal(raw, &text) == nil {
			reason := backendStopReasonFromString(text)
			return reason, reason != ""
		}
	}
	return FinishReasonStop, true
}

func backendStopReasonFromString(value string) string {
	value = cleanFinishText(value)
	switch normalizeFinishReason(value) {
	case FinishReasonLength:
		return "stopped_limit"
	case FinishReasonMaxTokens:
		return FinishReasonMaxTokens
	case FinishReasonTimeout:
		return FinishReasonTimeout
	case FinishReasonStop:
		return FinishReasonStop
	default:
		if value == "" || value == "false" || value == "0" || value == "null" {
			return ""
		}
		return FinishReasonStop
	}
}

func firstNonEmptyFinishText(values ...string) string {
	for _, value := range values {
		if cleaned := cleanFinishText(value); cleaned != "" {
			return cleaned
		}
	}
	return ""
}

func cleanFinishText(value string) string {
	value = strings.ToLower(cleanStatusText(value, maxStatusReasonLen))
	value = strings.TrimSpace(value)
	if value == "" || strings.Contains(value, "redacted") {
		return ""
	}
	value = strings.NewReplacer(" ", "_", "-", "_").Replace(value)
	for _, r := range value {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '_' || r == '.' || r == ':' {
			continue
		}
		return ""
	}
	return value
}

func backendGeneratedTokens(tokensGenerated int64, tokensPredicted int64, timings *llamaTimings) int64 {
	if tokensGenerated > 0 {
		return tokensGenerated
	}
	if tokensPredicted > 0 {
		return tokensPredicted
	}
	if timings != nil && timings.PredictedN > 0 {
		return timings.PredictedN
	}
	return 0
}
