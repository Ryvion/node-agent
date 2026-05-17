package modelprepare

import (
	"context"
	"io"
	"net/http"
	"os"
	"path/filepath"
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

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}
