package llamacpp

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestClientCompletesChatAndHashesSafely(t *testing.T) {
	var gotPath string
	var gotBody chatRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "local-model",
			"choices": []map[string]any{{
				"message":       map[string]any{"content": "verified answer"},
				"finish_reason": "stop",
			}},
			"usage": map[string]any{
				"prompt_tokens":     4,
				"completion_tokens": 2,
				"total_tokens":      6,
			},
		})
	}))
	defer server.Close()

	now := time.Unix(100, 0)
	client := Client{
		HTTPClient: server.Client(),
		Now: func() time.Time {
			now = now.Add(25 * time.Millisecond)
			return now
		},
	}
	result, err := client.Complete(context.Background(), Config{ServerURL: server.URL, Model: "configured-model"}, Spec{
		Task:      Task,
		Prompt:    "private prompt",
		MaxTokens: 8,
	})
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if gotPath != "/v1/chat/completions" {
		t.Fatalf("path = %q", gotPath)
	}
	if gotBody.Model != "configured-model" || len(gotBody.Messages) != 1 || gotBody.Messages[0].Content != "private prompt" {
		t.Fatalf("unexpected request body: %+v", gotBody)
	}
	if result.Text != "verified answer" || result.OutputHash == "" || result.PromptHash == "" || result.CompletionTokens != 2 {
		t.Fatalf("unexpected result: %+v", result)
	}
	metadata := BuildReceiptMetadata(result, map[string]any{
		"prompt": "private",
		"safe":   "ok",
		"events": []any{
			map[string]any{"raw_prompt": "nested-private", "status": "ok"},
			map[string]any{"output_text": "nested-output", "duration_ms": 25},
		},
	})
	raw, _ := json.Marshal(metadata)
	if strings.Contains(string(raw), "private") || strings.Contains(string(raw), "nested-output") || strings.Contains(string(raw), "verified answer") {
		t.Fatalf("metadata leaked raw text: %s", string(raw))
	}
	if metadata["safe"] != "ok" {
		t.Fatalf("metadata lost safe field: %+v", metadata)
	}
}

func TestConfigRejectsRemoteServerURL(t *testing.T) {
	if IsLocalHTTPBaseURL("http://example.com:8080") {
		t.Fatal("remote llama.cpp URL should be rejected")
	}
	if !IsLocalHTTPBaseURL("http://127.0.0.1:8080") {
		t.Fatal("local llama.cpp URL should be allowed")
	}
}

func TestDecodeSpecValidatesTaskAndInput(t *testing.T) {
	if _, err := DecodeSpec(`{"task":"llama_cpp_inference","messages":[{"role":"user","content":"hi"}]}`); err != nil {
		t.Fatalf("DecodeSpec() error = %v", err)
	}
	if _, err := DecodeSpec(`{"task":"scene_render","prompt":"hi"}`); err == nil {
		t.Fatal("DecodeSpec() accepted old render task")
	}
	if _, err := DecodeSpec(`{"task":"llama_cpp_inference"}`); err == nil {
		t.Fatal("DecodeSpec() accepted missing prompt/messages")
	}
}

func TestDecodeSpecOutputArtifactDefaultAndOverride(t *testing.T) {
	defaultSpec, err := DecodeSpec(`{"task":"llama_cpp_inference","prompt":"hi"}`)
	if err != nil {
		t.Fatalf("DecodeSpec(default) error = %v", err)
	}
	if !ShouldWriteOutputArtifact(defaultSpec) {
		t.Fatal("ShouldWriteOutputArtifact(default) = false, want true")
	}

	disabledSpec, err := DecodeSpec(`{"task":"llama_cpp_inference","prompt":"hi","output_artifact":false}`)
	if err != nil {
		t.Fatalf("DecodeSpec(disabled) error = %v", err)
	}
	if ShouldWriteOutputArtifact(disabledSpec) {
		t.Fatal("ShouldWriteOutputArtifact(disabled) = true, want false")
	}
}

func TestExecuteRespectsOutputArtifactFalse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "local-model",
			"choices": []map[string]any{{
				"message":       map[string]any{"content": "artifact should stay local only when requested"},
				"finish_reason": "stop",
			}},
		})
	}))
	defer server.Close()

	execution, err := Execute(
		context.Background(),
		`{"task":"llama_cpp_inference","prompt":"hi","output_artifact":false}`,
		Config{ServerURL: server.URL, Model: "local-model"},
		Client{HTTPClient: server.Client()},
		map[string]any{},
	)
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if execution.OutputPath != "" {
		t.Fatalf("Execute() OutputPath = %q, want empty when output_artifact=false", execution.OutputPath)
	}
	if execution.ResultHashHex == "" {
		t.Fatal("Execute() ResultHashHex is empty")
	}
}
