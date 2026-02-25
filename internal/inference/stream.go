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
		maxTokens = 1024
	}

	modelName := strings.TrimSpace(spec.Model)
	if modelName == "" {
		modelName = "ryvion-llama-3.2-3b"
	}

	if err := m.EnsureModel(ctx, modelName); err != nil {
		return fmt.Errorf("ensure model %s: %w", modelName, err)
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
