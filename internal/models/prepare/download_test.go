package modelprepare

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func TestDownloadArtifactAddsHuggingFaceBearerToken(t *testing.T) {
	t.Setenv("HF_TOKEN", "hf_test_token")
	var gotAuth string
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		gotAuth = req.Header.Get("Authorization")
		return &http.Response{
			StatusCode:    http.StatusOK,
			Body:          io.NopCloser(strings.NewReader("gguf")),
			ContentLength: 4,
			Header:        make(http.Header),
			Request:       req,
		}, nil
	})}

	destination := filepath.Join(t.TempDir(), "gemma-3-27b-it-q4_0.gguf")
	result, err := DownloadArtifact(
		context.Background(),
		"https://huggingface.co/google/gemma-3-27b-it-qat-q4_0-gguf/resolve/main/gemma-3-27b-it-q4_0.gguf",
		destination,
		DownloadOptions{HTTPClient: client, MaxBytes: 4, ExpectedSizeBytes: 4},
	)
	if err != nil {
		t.Fatalf("DownloadArtifact() error = %v, want nil", err)
	}
	if gotAuth != "Bearer hf_test_token" {
		t.Fatalf("Authorization = %q, want HF bearer token", gotAuth)
	}
	if result.Bytes != 4 {
		t.Fatalf("bytes = %d, want 4", result.Bytes)
	}
	body, err := os.ReadFile(destination)
	if err != nil {
		t.Fatalf("read destination: %v", err)
	}
	if string(body) != "gguf" {
		t.Fatalf("body = %q, want gguf", body)
	}
}

func TestDownloadArtifactDoesNotAttachTokenToNonHuggingFaceURL(t *testing.T) {
	t.Setenv("HF_TOKEN", "hf_test_token")
	var gotAuth string
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		gotAuth = req.Header.Get("Authorization")
		return &http.Response{
			StatusCode:    http.StatusOK,
			Body:          io.NopCloser(strings.NewReader("gguf")),
			ContentLength: 4,
			Header:        make(http.Header),
			Request:       req,
		}, nil
	})}

	_, err := DownloadArtifact(
		context.Background(),
		"https://models.example.invalid/model.gguf",
		filepath.Join(t.TempDir(), "model.gguf"),
		DownloadOptions{HTTPClient: client, MaxBytes: 4, ExpectedSizeBytes: 4},
	)
	if err != nil {
		t.Fatalf("DownloadArtifact() error = %v, want nil", err)
	}
	if gotAuth != "" {
		t.Fatalf("Authorization = %q, want empty for non-Hugging Face artifact", gotAuth)
	}
}

func TestDownloadArtifactUsesNodeAuthButStripsItOnCDNRedirect(t *testing.T) {
	body := []byte("gguf-cdn-body")
	var hubToken string
	var cdnToken string

	cdn := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cdnToken = r.Header.Get("X-Node-Token")
		w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		_, _ = w.Write(body)
	}))
	defer cdn.Close()

	hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hubToken = r.Header.Get("X-Node-Token")
		http.Redirect(w, r, cdn.URL+"/model.gguf", http.StatusTemporaryRedirect)
	}))
	defer hub.Close()

	destination := filepath.Join(t.TempDir(), "model.gguf")
	result, err := DownloadArtifact(
		context.Background(),
		hub.URL+"/api/v1/node/models/qwen3-8b-reasoning/download",
		destination,
		DownloadOptions{
			MaxBytes:          uint64(len(body)),
			ExpectedSizeBytes: int64(len(body)),
			AllowInsecureHTTP: true,
			AttachAuth: func(req *http.Request, _ string) {
				req.Header.Set("X-Node-Token", "node-secret")
			},
		},
	)
	if err != nil {
		t.Fatalf("DownloadArtifact() error = %v, want nil", err)
	}
	if result.Bytes != int64(len(body)) {
		t.Fatalf("bytes = %d, want %d", result.Bytes, len(body))
	}
	if hubToken != "node-secret" {
		t.Fatalf("hub token = %q, want node auth token", hubToken)
	}
	if cdnToken != "" {
		t.Fatalf("cdn received node token %q; node auth must not follow CDN redirects", cdnToken)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}
