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
	"strings"
	"time"

	"github.com/Ryvion/node-agent/internal/hub"
)

// chatRequest is the OpenAI-compatible request to local llama-server.
type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Stream      bool          `json:"stream"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature *float64      `json:"temperature,omitempty"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
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

// RunStreamingJob handles an inference job by calling the local llama-server
// with streaming, and relaying chunks to the hub.
func (m *Manager) RunStreamingJob(ctx context.Context, hubClient *hub.Client, jobID, specJSON string) error {
	start := time.Now()

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

	reqBody := chatRequest{
		Model:       modelName,
		Messages:    spec.Messages,
		Stream:      true,
		MaxTokens:   maxTokens,
		Temperature: spec.Temperature,
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
			msg := fmt.Sprintf("data: {\"error\": \"llama-server stream error: %s\"}\n\n", errChunk.Error.Message)
			pw.Write([]byte(msg))
			pw.Close()
			<-streamErr
			return fmt.Errorf("llama-server stream error: %s", errChunk.Error.Message)
		}

		// Extract content for hash/receipt
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err == nil && len(chunk.Choices) > 0 {
			content := chunk.Choices[0].Delta.Content
			fullContent.WriteString(content)
			hash.Write([]byte(content))
		}

		// Relay SSE line to hub
		pw.Write([]byte(line + "\n\n"))
	}

	if err := scanner.Err(); err != nil {
		msg := fmt.Sprintf("data: {\"error\": \"reading llama-server stream failed: %v\"}\n\n", err)
		pw.Write([]byte(msg))
		pw.Close()
		<-streamErr
		return fmt.Errorf("reading llama-server stream failed: %w", err)
	}

	if err := ctx.Err(); err != nil {
		msg := fmt.Sprintf("data: {\"error\": \"job context cancelled (timeout limit reached): %v\"}\n\n", err)
		pw.Write([]byte(msg))
		pw.Close()
		<-streamErr
		return fmt.Errorf("job context cancelled: %w", err)
	}

	if fullContent.Len() == 0 {
		msg := "data: {\"error\": \"llama-server returned empty output (context window or memory exceeded)\"}\n\n"
		pw.Write([]byte(msg))
		pw.Close()
		<-streamErr
		return fmt.Errorf("llama-server returned empty inference generation")
	}

	pw.Close()

	// Wait for hub stream to finish
	if err := <-streamErr; err != nil {
		slog.Warn("hub stream relay error", "job_id", jobID, "error", err)
	}

	duration := time.Since(start)
	resultHash := hex.EncodeToString(hash.Sum(nil))

	// Submit receipt — truncate response tail to avoid bloating metadata.
	tail := fullContent.String()
	if len(tail) > 4096 {
		tail = tail[len(tail)-4096:]
	}
	if err := hubClient.SubmitReceipt(ctx, hub.Receipt{
		JobID:         jobID,
		ResultHashHex: resultHash,
		MeteringUnits: 1,
		Metadata: map[string]any{
			"executor":        "llama-server",
			"model":           m.ModelName(),
			"duration_ms":     duration.Milliseconds(),
			"exit_code":       0,
			"response_length": fullContent.Len(),
			"stderr_tail":     tail,
		},
	}); err != nil {
		slog.Warn("submit receipt failed", "job_id", jobID, "error", err)
		return fmt.Errorf("submit receipt: %w", err)
	}

	slog.Info("streaming inference complete", "job_id", jobID, "duration", duration, "tokens_approx", fullContent.Len())
	return nil
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
func (m *Manager) RunEmbeddingJob(ctx context.Context, hubClient *hub.Client, jobID, specJSON string) error {
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

	if err := hubClient.SubmitReceipt(ctx, hub.Receipt{
		JobID:         jobID,
		ResultHashHex: resultHash,
		MeteringUnits: 1,
		Metadata: map[string]any{
			"executor":      "llama-server",
			"task":          "embedding",
			"model":         modelName,
			"duration_ms":   duration.Milliseconds(),
			"exit_code":     0,
			"dimensions":    len(vector),
			"prompt_tokens": embResp.Usage.PromptTokens,
			"embedding":     vector,
		},
	}); err != nil {
		return fmt.Errorf("submit embed receipt: %w", err)
	}

	slog.Info("native embedding complete", "job_id", jobID, "model", modelName, "dims", len(vector), "duration", duration)
	return nil
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
