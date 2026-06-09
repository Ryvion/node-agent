package inference

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"path/filepath"
	"strings"
	"time"

	"github.com/Ryvion/ryvion-node/internal/hub"
)

// chatRequest is the OpenAI-compatible request to local llama-server.
type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Stream      bool          `json:"stream"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature *float64      `json:"temperature,omitempty"`
	// TopP/TopK/MinP override llama-server's defaults for reasoning models.
	// They are pointers + omitempty so non-reasoning requests send nothing and
	// the server keeps its own defaults. See reasoningSamplingForModel.
	TopP *float64 `json:"top_p,omitempty"`
	TopK *int     `json:"top_k,omitempty"`
	MinP *float64 `json:"min_p,omitempty"`
	// ReasoningEffort: "low" | "medium" | "high". Set for reasoning models
	// (GPT-OSS, Qwen3-reasoning, DeepSeek R1). llama-server routes this
	// into the chat template's reasoning_effort kwarg via --jinja, which
	// changes how much the model "thinks" before answering. Empty string
	// is omitted from the JSON and falls back to runtime default.
	ReasoningEffort string `json:"reasoning_effort,omitempty"`
}

// chatMessage carries either a plain string OR an OpenAI multimodal
// content array, e.g.:
//
//	{"role":"user","content":[
//	  {"type":"text","text":"What's in this image?"},
//	  {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
//	]}
//
// Content is `json.RawMessage` so the multimodal payload survives the
// hub → spec_json → node-agent → llama-server hop verbatim. llama-server
// (with `--mmproj` loaded for vision-capable models like Gemma 4 26B
// or Nemotron 3 Nano Omni) consumes the OpenAI multimodal format
// directly — re-decoding the bytes as a Go `string` would corrupt
// the array form and crash the unmarshal step.
type chatMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// MessageText returns the text content of a message, treating
// multimodal arrays as the concatenation of their text parts. Used
// only for diagnostics + non-content code paths (logging, metrics);
// the actual chat payload is forwarded to llama-server as raw JSON.
func (m chatMessage) MessageText() string {
	if len(m.Content) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(m.Content, &s); err == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(m.Content, &parts); err != nil {
		return ""
	}
	var b strings.Builder
	for _, p := range parts {
		if p.Type == "text" && p.Text != "" {
			if b.Len() > 0 {
				b.WriteByte('\n')
			}
			b.WriteString(p.Text)
		}
	}
	return b.String()
}

// specPayload is what the hub sends as spec_json for inference jobs.
type specPayload struct {
	Messages    []chatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature *float64      `json:"temperature,omitempty"`
	Model       string        `json:"model,omitempty"`
	ModelURL    string        `json:"model_url,omitempty"`    // Presigned download URL for custom models
	ModelFormat string        `json:"model_format,omitempty"` // "gguf", "onnx", etc.
	ModelName   string        `json:"model_name,omitempty"`   // Human-readable name
	Task        string        `json:"task,omitempty"`         // "custom_inference", "embedding"
	Input       string        `json:"input,omitempty"`        // Text input for embedding tasks
	// V8ReasoningEffort is the hub's scheduler-metadata key carrying the
	// OpenAI-compatible reasoning_effort param ("low"|"medium"|"high").
	// Underscore prefix marks it as out-of-band — it does not enter the
	// model's transcript hash, so reruns at a different effort produce
	// distinct receipts without polluting the message stream.
	V8ReasoningEffort string `json:"_v8_reasoning_effort,omitempty"`
}

// normalizeStreamReasoningEffort lower-cases the inbound effort token and
// drops anything that isn't a valid OpenAI-compatible reasoning_effort.
// Returns "" on invalid input — llama-server then falls back to whatever
// default the chat template / --chat-template-kwargs encoded.
func normalizeStreamReasoningEffort(raw string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "low", "medium", "high":
		return strings.ToLower(strings.TrimSpace(raw))
	default:
		return ""
	}
}

// reasoningSamplingForModel returns llama.cpp sampling parameters tuned to a
// reasoning model family. ok=false for non-reasoning models, which keep the
// buyer-supplied temperature and llama-server's own defaults.
//
// Reasoning models degrade badly under llama-server's default sampling
// (top_p 0.9, top_k 40, min_p 0.1) combined with the low temperature the
// playground sends (0.4): GPT-OSS in particular falls into a repetitive
// "analysis" loop that never emits its final channel. Each family is pinned to
// its vendor-recommended profile instead:
//
//   - GPT-OSS: temp 1.0, top_p 1.0, no repetition penalty (OpenAI / llama.cpp
//     guide discussions/15396 — "Do not use repetition penalties"). We keep a
//     modest top_k 40 rather than OpenAI's top_k 0: with no tail truncation the
//     model occasionally samples a sub-1% token mid-word (the guide warns about
//     exactly this), producing stray typos. At temp 1.0, 40 candidates is far
//     more diversity than the ~handful that drove the original low-temp
//     repetition loop, so it clips the garbage tail without regressing.
//   - Qwen3 / DeepSeek-R1 thinking: temp 0.6, top_p 0.95, top_k 20 (Qwen team's
//     published thinking-mode recommendation).
//
// min_p is pinned to 0 in both cases to override llama-server's 0.1 default,
// which otherwise prunes the low-probability tokens these models rely on.
func reasoningSamplingForModel(modelName string) (temp, topP, minP float64, topK int, ok bool) {
	switch streamingFamilyHintForModel(modelName) {
	case "gpt-oss":
		return 1.0, 1.0, 0.0, 40, true
	case "qwen", "deepseek":
		return 0.6, 0.95, 0.0, 20, true
	default:
		return 0, 0, 0, 0, false
	}
}

// IsEmbeddingJob returns true when the hub-provided spec_json asks the node
// to produce an embedding vector rather than a chat completion.
func IsEmbeddingJob(specJSON string) bool {
	var s specPayload
	if err := json.Unmarshal([]byte(specJSON), &s); err != nil {
		return false
	}
	return strings.EqualFold(strings.TrimSpace(s.Task), "embedding")
}

// RequestedNativeModelForSpec returns the built-in native model a spec will
// ask llama-server to load. Custom model specs intentionally return false
// because their model path is resolved by EnsureCustomModel at execution time.
func RequestedNativeModelForSpec(specJSON string) (string, bool) {
	var s specPayload
	if err := json.Unmarshal([]byte(specJSON), &s); err != nil {
		return "", false
	}
	if strings.EqualFold(strings.TrimSpace(s.Task), "custom_inference") && strings.TrimSpace(s.ModelURL) != "" {
		return "", false
	}
	modelName := strings.TrimSpace(s.Model)
	if modelName == "" {
		if strings.EqualFold(strings.TrimSpace(s.Task), "embedding") {
			modelName = "nomic-embed-text-v1.5"
		} else {
			modelName = "ryvion-llama-3.2-3b"
		}
	}
	if _, ok := NativeModels[modelName]; !ok {
		return "", false
	}
	return modelName, true
}

// StreamingMetrics summarises the latency / throughput numbers a single
// streaming inference job observed locally. The /v8 verifier (and hub
// dashboards) read these from the receipt's MetadataJSON via the keys
// p50_ttft_ms, p50_decode_tps, p50_end_to_end_tps, etc. Zero values mean the
// metric was not measurable for this job (e.g. the request errored before the
// first token arrived); callers must treat them as "unknown" rather than 0.
type StreamingMetrics struct {
	// TTFTMs is wall-clock milliseconds between dispatching the request to
	// llama-server and observing the first content delta on the SSE stream.
	TTFTMs int64
	// DecodeTPS is generated tokens per second measured between the first
	// content delta and the end of the stream — i.e. excludes prefill / TTFT.
	DecodeTPS float64
	// EndToEndTPS is generated tokens per second measured against the full
	// wall-clock duration of the job (includes prefill / TTFT).
	EndToEndTPS float64
	// CompletionTokens is the count llama-server reported in its terminal
	// `usage` chunk; falls back to a per-chunk content count when absent.
	CompletionTokens int64
	// SpeculativeTokensDrafted/Accepted are emitted by recent llama-server
	// builds when the server runs with draft-model or native-MTP speculation.
	SpeculativeTokensDrafted  int64
	SpeculativeTokensAccepted int64
}

// RunStreamingJob handles an inference job by calling the local llama-server
// with streaming, and relaying chunks to the hub.
func (m *Manager) RunStreamingJob(ctx context.Context, hubClient *hub.Client, jobID, specJSON string, metadataExtras ...map[string]any) error {
	start := time.Now()
	var (
		firstTokenAt              time.Time
		usageTokens               int64 // from llama-server's terminal usage chunk
		chunkTokens               int64 // fallback: count of chunks with non-empty delta
		hasFirstToken             bool
		speculativeTokensDrafted  int64
		speculativeTokensAccepted int64
	)

	var spec specPayload
	if err := json.Unmarshal([]byte(specJSON), &spec); err != nil {
		return fmt.Errorf("parse spec_json: %w", err)
	}
	if len(spec.Messages) == 0 {
		return fmt.Errorf("spec_json has no messages")
	}

	maxTokens := spec.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 2048
	}

	modelName := strings.TrimSpace(spec.Model)

	// Custom model: download from URL and load it
	if spec.Task == "custom_inference" && spec.ModelURL != "" {
		customName := strings.TrimSpace(spec.ModelName)
		if customName == "" {
			customName = "custom-model"
		}
		slog.Info("custom model inference requested", "model_name", customName, "format", spec.ModelFormat)
		if err := m.EnsureCustomModel(ctx, customName, spec.ModelURL); err != nil {
			return fmt.Errorf("ensure custom model %s: %w", customName, err)
		}
		modelName = customName
	} else {
		if modelName == "" {
			modelName = "ryvion-llama-3.2-3b"
		}
		if err := m.EnsureModel(ctx, modelName); err != nil {
			return fmt.Errorf("ensure model %s: %w", modelName, err)
		}
	}
	servedModelIDs, err := m.verifyLoadedModel(ctx, modelName)
	if err != nil {
		return err
	}

	reqBody := chatRequest{
		Model:           modelName,
		Messages:        spec.Messages,
		ReasoningEffort: normalizeStreamReasoningEffort(spec.V8ReasoningEffort),
		Stream:          true,
		MaxTokens:       maxTokens,
		Temperature:     spec.Temperature,
	}
	// Reasoning models need their vendor-recommended sampling profile. The
	// buyer-supplied temperature (the playground sends 0.4) plus llama-server's
	// default min_p 0.1 / top_k 40 / top_p 0.9 sends GPT-OSS into a repetitive
	// "analysis" loop that burns the whole token budget without ever emitting
	// its final answer. Pin the full profile for reasoning families, overriding
	// the request temperature; non-reasoning models are left untouched.
	if temp, topP, minP, topK, ok := reasoningSamplingForModel(modelName); ok {
		reqBody.Temperature = &temp
		reqBody.TopP = &topP
		reqBody.MinP = &minP
		reqBody.TopK = &topK
	}
	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	url := m.ServerURL() + "/v1/chat/completions"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("llama-server request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("llama-server returned %d: %s", resp.StatusCode, string(body))
	}

	// Set up pipe: read SSE from llama-server, relay to hub as chunked POST
	pr, pw := io.Pipe()

	// Start hub stream upload in background
	streamErr := make(chan error, 1)
	go func() {
		streamErr <- hubClient.StreamInference(ctx, jobID, pr)
	}()

	// Read SSE lines from llama-server, relay as-is to hub
	var fullContent strings.Builder
	hash := sha256.New()
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64*1024), 64*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			pw.Write([]byte("data: [DONE]\n\n"))
			break
		}

		// Check if llama-server emitted an internal error stream chunk
		var errChunk struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal([]byte(data), &errChunk); err == nil && errChunk.Error.Message != "" {
			writeHubStreamError(pw, "llama-server stream error: "+errChunk.Error.Message)
			pw.Close()
			<-streamErr
			return fmt.Errorf("llama-server stream error: %s", errChunk.Error.Message)
		}

		// Extract content for hash/receipt + capture llama-server's terminal
		// `usage` object when emitted (some llama.cpp builds attach it to the
		// final non-[DONE] chunk; others emit a standalone usage frame).
		//
		// Also extract `reasoning_content` (and the alternate `reasoning`
		// spelling). Reasoning-capable models like Qwen3 and GPT-OSS emit
		// their thinking text in this field when llama-server is launched
		// with `--jinja --reasoning-format deepseek|auto`. We count it
		// toward fullContent / hasFirstToken so:
		//   1. The empty-output guard below doesn't fire for reasoning-only
		//      responses (Qwen3 at high effort can spend its entire
		//      max_tokens budget on thinking with no final answer).
		//   2. TTFT measures time-to-first-thinking-token, not
		//      time-to-first-content-token — the buyer-facing cold-start
		//      indicator stops as soon as ANY token arrives.
		//   3. The receipt's content hash covers reasoning too, so reruns
		//      at the same effort are content-addressable end-to-end.
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content          string `json:"content"`
					ReasoningContent string `json:"reasoning_content"`
					Reasoning        string `json:"reasoning"`
				} `json:"delta"`
			} `json:"choices"`
			Usage *struct {
				PromptTokens     int64 `json:"prompt_tokens"`
				CompletionTokens int64 `json:"completion_tokens"`
				TotalTokens      int64 `json:"total_tokens"`
			} `json:"usage,omitempty"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err == nil {
			if chunk.Usage != nil && chunk.Usage.CompletionTokens > 0 {
				usageTokens = chunk.Usage.CompletionTokens
			}
			if drafted, accepted := streamingSpeculativeCountsFromPayload([]byte(data)); drafted > 0 || accepted > 0 {
				if drafted > speculativeTokensDrafted {
					speculativeTokensDrafted = drafted
				}
				if accepted > speculativeTokensAccepted {
					speculativeTokensAccepted = accepted
				}
			}
			if len(chunk.Choices) > 0 {
				delta := chunk.Choices[0].Delta
				content := delta.Content
				reasoning := delta.ReasoningContent
				if reasoning == "" {
					reasoning = delta.Reasoning
				}
				if content != "" || reasoning != "" {
					if !hasFirstToken {
						firstTokenAt = time.Now()
						hasFirstToken = true
					}
				}
				if content != "" {
					chunkTokens++
					fullContent.WriteString(content)
					hash.Write([]byte(content))
				}
				if reasoning != "" {
					chunkTokens++
					fullContent.WriteString(reasoning)
					hash.Write([]byte(reasoning))
				}
			}
		}

		// Relay SSE line to hub
		pw.Write([]byte(line + "\n\n"))
	}

	if err := scanner.Err(); err != nil {
		writeHubStreamError(pw, fmt.Sprintf("reading llama-server stream failed: %v", err))
		pw.Close()
		<-streamErr
		return fmt.Errorf("reading llama-server stream failed: %w", err)
	}

	if err := ctx.Err(); err != nil {
		writeHubStreamError(pw, fmt.Sprintf("job context cancelled (timeout limit reached): %v", err))
		pw.Close()
		<-streamErr
		return fmt.Errorf("job context cancelled: %w", err)
	}

	if fullContent.Len() == 0 {
		writeHubStreamError(pw, "llama-server returned empty output (context window or memory exceeded)")
		pw.Close()
		<-streamErr
		return fmt.Errorf("llama-server returned empty inference generation")
	}

	pw.Close()

	// Wait for hub stream to finish
	if err := <-streamErr; err != nil {
		slog.Warn("hub stream relay error", "job_id", jobID, "error", err)
	}

	finishedAt := time.Now()
	duration := finishedAt.Sub(start)
	resultHash := hex.EncodeToString(hash.Sum(nil))

	// Compute streaming-timing metrics. Prefer llama-server's usage count for
	// the token total; fall back to per-chunk counting if absent. Decode TPS
	// excludes prefill (TTFT); end-to-end TPS includes it. The /v8 verifier
	// reads p50_* and *_tps keys to populate its TTFT/DECODE TPS columns;
	// emitting only the keys we measured keeps zero from being mistaken for a
	// real value downstream.
	tokensGenerated := usageTokens
	if tokensGenerated == 0 {
		tokensGenerated = chunkTokens
	}
	metrics := StreamingMetrics{
		CompletionTokens:          tokensGenerated,
		SpeculativeTokensDrafted:  speculativeTokensDrafted,
		SpeculativeTokensAccepted: speculativeTokensAccepted,
	}
	if hasFirstToken {
		metrics.TTFTMs = firstTokenAt.Sub(start).Milliseconds()
		decodeWindow := finishedAt.Sub(firstTokenAt).Seconds()
		if tokensGenerated > 0 && decodeWindow > 0 {
			metrics.DecodeTPS = float64(tokensGenerated) / decodeWindow
		}
	}
	endToEndWindow := duration.Seconds()
	if tokensGenerated > 0 && endToEndWindow > 0 {
		metrics.EndToEndTPS = float64(tokensGenerated) / endToEndWindow
	}

	// Submit receipt — truncate response tail to avoid bloating metadata.
	tail := fullContent.String()
	if len(tail) > 4096 {
		tail = tail[len(tail)-4096:]
	}
	meta := mergeReceiptMetadata(metadataExtras...)
	meta["executor"] = "llama-server"
	meta["model"] = m.ModelName()
	meta["model_id"] = modelName
	meta["requested_model_id"] = modelName
	if len(servedModelIDs) > 0 {
		meta["served_model_ids"] = servedModelIDs
		meta["served_model_id"] = servedModelIDs[0]
	}
	meta["duration_ms"] = duration.Milliseconds()
	meta["exit_code"] = 0
	meta["response_length"] = fullContent.Len()
	meta["stderr_tail"] = tail
	applyStreamingMetricsToMetadata(meta, metrics)
	applyStreamingSpeculativeMetadata(meta, m.streamingSpeculativeLaunch(), metrics)
	// Retry on transient failure: this receipt proves completed streaming work,
	// and a single dropped submit means the job never reaches "completed" and
	// the operator isn't paid.
	if err := hubClient.SubmitReceiptWithRetry(ctx, hub.Receipt{
		JobID:         jobID,
		ResultHashHex: resultHash,
		MeteringUnits: 1,
		Metadata:      meta,
	}); err != nil {
		slog.Warn("submit receipt failed", "job_id", jobID, "error", err)
		return fmt.Errorf("submit receipt: %w", err)
	}

	slog.Info("streaming inference complete",
		"job_id", jobID,
		"duration", duration,
		"tokens", tokensGenerated,
		"ttft_ms", metrics.TTFTMs,
		"decode_tps", metrics.DecodeTPS,
		"end_to_end_tps", metrics.EndToEndTPS,
		"speculative_method", m.streamingSpeculativeLaunch().Method,
	)
	return nil
}

func (m *Manager) verifyLoadedModel(ctx context.Context, modelName string) ([]string, error) {
	cfg, ok := NativeModels[strings.TrimSpace(modelName)]
	if !ok {
		return nil, nil
	}
	verifyCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	ok, ids, err := m.loadedModelMatches(verifyCtx, modelName)
	if err != nil {
		return ids, fmt.Errorf("verify loaded model %s: %w", modelName, err)
	}
	if len(ids) == 0 {
		return ids, fmt.Errorf("verify loaded model %s: llama-server returned no model ids", modelName)
	}
	if ok {
		return ids, nil
	}
	return ids, fmt.Errorf("loaded model mismatch: requested %s (%s), llama-server reports %s", modelName, cfg.FileName, strings.Join(ids, ","))
}

func (m *Manager) loadedModelMatches(ctx context.Context, modelName string) (bool, []string, error) {
	cfg, ok := NativeModels[strings.TrimSpace(modelName)]
	if !ok {
		return true, nil, nil
	}
	ids, err := m.loadedModelIDs(ctx)
	if err != nil {
		return false, ids, err
	}
	return loadedModelIDsMatch(modelName, cfg, ids), ids, nil
}

func (m *Manager) loadedModelIDs(ctx context.Context) ([]string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, m.ServerURL()+"/v1/models", nil)
	if err != nil {
		return nil, err
	}
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, fmt.Errorf("llama-server /v1/models returned %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	var payload struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 64*1024)).Decode(&payload); err != nil {
		return nil, fmt.Errorf("decode llama-server /v1/models: %w", err)
	}
	ids := make([]string, 0, len(payload.Data))
	for _, row := range payload.Data {
		if id := strings.TrimSpace(row.ID); id != "" {
			ids = append(ids, id)
		}
	}
	return ids, nil
}

func loadedModelIDsMatch(modelName string, cfg ModelConfig, ids []string) bool {
	aliases := modelMatchAliases(modelName, cfg)
	for _, id := range ids {
		for _, candidate := range modelIDMatchCandidates(id) {
			if aliases[candidate] {
				return true
			}
		}
	}
	return false
}

func modelMatchAliases(modelName string, cfg ModelConfig) map[string]bool {
	aliases := map[string]bool{}
	for _, value := range []string{modelName, cfg.FileName} {
		for _, candidate := range modelIDMatchCandidates(value) {
			if candidate != "" {
				aliases[candidate] = true
			}
		}
	}
	return aliases
}

func modelIDMatchCandidates(value string) []string {
	normalized := normalizeModelIDForMatch(value)
	if normalized == "" {
		return nil
	}
	base := normalizeModelIDForMatch(filepath.Base(strings.ReplaceAll(normalized, "\\", "/")))
	out := []string{normalized}
	if base != "" && base != normalized {
		out = append(out, base)
	}
	for _, candidate := range []string{normalized, base} {
		if strings.HasSuffix(candidate, ".gguf") {
			out = append(out, strings.TrimSuffix(candidate, ".gguf"))
		}
	}
	return dedupeStrings(out)
}

func normalizeModelIDForMatch(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	value = strings.Trim(value, `"'`)
	value = strings.TrimPrefix(value, "local/")
	return value
}

func dedupeStrings(values []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(values))
	for _, value := range values {
		if value == "" || seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	return out
}

// applyStreamingMetricsToMetadata writes the streaming-timing keys the hub
// verifier expects into the receipt metadata map. Only non-zero metrics are
// written so the hub can keep treating absence as "not measured" rather than
// surfacing 0 ms / 0 tps in the dashboard. Speculative-decoding keys are
// emitted separately by applyStreamingSpeculativeMetadata because they depend
// on the selected launch mode as well as optional runtime timing counters.
func applyStreamingMetricsToMetadata(meta map[string]any, metrics StreamingMetrics) {
	if meta == nil {
		return
	}
	if metrics.TTFTMs > 0 {
		meta["p50_ttft_ms"] = metrics.TTFTMs
		meta["ttft_ms"] = metrics.TTFTMs
	}
	if metrics.DecodeTPS > 0 {
		meta["p50_decode_tps"] = metrics.DecodeTPS
		meta["decode_tps"] = metrics.DecodeTPS
		meta["tps"] = metrics.DecodeTPS
	}
	if metrics.EndToEndTPS > 0 {
		meta["p50_end_to_end_tps"] = metrics.EndToEndTPS
		meta["end_to_end_tps"] = metrics.EndToEndTPS
	}
	if metrics.CompletionTokens > 0 {
		meta["completion_tokens"] = metrics.CompletionTokens
	}
}

func writeHubStreamError(w io.Writer, message string) {
	payload, err := json.Marshal(map[string]any{
		"error": map[string]string{
			"message": strings.TrimSpace(message),
			"type":    "node_error",
		},
	})
	if err != nil {
		payload = []byte(`{"error":{"message":"streaming inference failed","type":"node_error"}}`)
	}
	_, _ = w.Write([]byte("data: " + string(payload) + "\n\n"))
}

// embedRequest and embedResponse are the OpenAI-compatible shapes that
// llama-server speaks at its /v1/embeddings endpoint.
type embedRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

type embedResponse struct {
	Object string `json:"object"`
	Data   []struct {
		Object    string    `json:"object"`
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// RunEmbeddingJob handles a native embedding job. The manager hot-swaps to
// the requested embedding model (if not already loaded), posts to the local
// llama-server /v1/embeddings endpoint, and submits a receipt with the
// vector inline in metadata. No SSE relay — embeddings are one-shot.
func (m *Manager) RunEmbeddingJob(ctx context.Context, hubClient *hub.Client, jobID, specJSON string, metadataExtras ...map[string]any) error {
	start := time.Now()

	var spec specPayload
	if err := json.Unmarshal([]byte(specJSON), &spec); err != nil {
		return fmt.Errorf("parse spec_json: %w", err)
	}
	input := strings.TrimSpace(spec.Input)
	if input == "" {
		return fmt.Errorf("embedding spec missing input")
	}
	modelName := strings.TrimSpace(spec.Model)
	if modelName == "" {
		modelName = "nomic-embed-text-v1.5"
	}
	if cfg, ok := NativeModels[modelName]; !ok || cfg.Mode != ModeEmbedding {
		return fmt.Errorf("model %q is not a registered native embedding model", modelName)
	}
	if err := m.EnsureModel(ctx, modelName); err != nil {
		return fmt.Errorf("ensure embedding model %s: %w", modelName, err)
	}

	reqBody, err := json.Marshal(embedRequest{Model: modelName, Input: input})
	if err != nil {
		return fmt.Errorf("marshal embed request: %w", err)
	}
	url := m.ServerURL() + "/v1/embeddings"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := (&http.Client{}).Do(req)
	if err != nil {
		return fmt.Errorf("llama-server embed request: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("llama-server embed returned %d: %s", resp.StatusCode, string(body))
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return fmt.Errorf("read embed response: %w", err)
	}
	var embResp embedResponse
	if err := json.Unmarshal(body, &embResp); err != nil {
		return fmt.Errorf("decode embed response: %w", err)
	}
	if len(embResp.Data) == 0 || len(embResp.Data[0].Embedding) == 0 {
		return fmt.Errorf("llama-server returned empty embedding")
	}
	vector := embResp.Data[0].Embedding

	// Hash the raw vector bytes for the receipt — buyer can reverify by
	// recomputing sha256(float32 little-endian of returned vector).
	hasher := sha256.New()
	for _, v := range vector {
		var buf [4]byte
		binaryLittleEndianPutFloat32(buf[:], v)
		hasher.Write(buf[:])
	}
	resultHash := hex.EncodeToString(hasher.Sum(nil))
	duration := time.Since(start)

	meta := mergeReceiptMetadata(metadataExtras...)
	meta["executor"] = "llama-server"
	meta["task"] = "embedding"
	meta["model"] = modelName
	meta["duration_ms"] = duration.Milliseconds()
	meta["exit_code"] = 0
	meta["dimensions"] = len(vector)
	meta["prompt_tokens"] = embResp.Usage.PromptTokens
	meta["embedding"] = vector

	if err := hubClient.SubmitReceiptWithRetry(ctx, hub.Receipt{
		JobID:         jobID,
		ResultHashHex: resultHash,
		MeteringUnits: 1,
		Metadata:      meta,
	}); err != nil {
		return fmt.Errorf("submit embed receipt: %w", err)
	}

	slog.Info("native embedding complete", "job_id", jobID, "model", modelName, "dims", len(vector), "duration", duration)
	return nil
}

func mergeReceiptMetadata(extras ...map[string]any) map[string]any {
	out := map[string]any{}
	for _, extra := range extras {
		for key, value := range extra {
			if strings.TrimSpace(key) != "" {
				out[key] = value
			}
		}
	}
	return out
}

// binaryLittleEndianPutFloat32 writes a float32 in little-endian bytes.
// Used for deterministic receipt hashing of the output vector.
func binaryLittleEndianPutFloat32(dst []byte, v float32) {
	bits := math.Float32bits(v)
	dst[0] = byte(bits)
	dst[1] = byte(bits >> 8)
	dst[2] = byte(bits >> 16)
	dst[3] = byte(bits >> 24)
}
