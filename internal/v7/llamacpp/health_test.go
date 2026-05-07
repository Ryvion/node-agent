package llamacpp

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestCheckHealthSuccessOnHealthEndpoint(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			t.Fatalf("unexpected path = %q", r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(server.Close)

	result := CheckHealth(context.Background(), server.URL, server.Client(), time.Second)
	if !result.Healthy || result.Endpoint != "/health" || result.Error != "" {
		t.Fatalf("health result = %+v, want healthy /health", result)
	}
	if result.CheckedAt.IsZero() {
		t.Fatalf("checked_at is zero: %+v", result)
	}
}

func TestCheckHealthFallsBackToModelsEndpoint(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			http.NotFound(w, r)
		case "/v1/models":
			w.WriteHeader(http.StatusOK)
		default:
			t.Fatalf("unexpected path = %q", r.URL.Path)
		}
	}))
	t.Cleanup(server.Close)

	result := CheckHealth(context.Background(), server.URL, server.Client(), time.Second)
	if !result.Healthy || result.Endpoint != "/v1/models" {
		t.Fatalf("health result = %+v, want healthy /v1/models fallback", result)
	}
}

func TestCheckHealthFailureIsSafeAndLocalOnly(t *testing.T) {
	t.Parallel()

	result := CheckHealth(context.Background(), "https://example.com", nil, time.Second)
	if result.Healthy || result.Error == "" {
		t.Fatalf("external health result = %+v, want safe failure", result)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "unavailable", http.StatusServiceUnavailable)
	}))
	t.Cleanup(server.Close)

	result = CheckHealth(context.Background(), server.URL, server.Client(), time.Second)
	if result.Healthy || result.Error == "" {
		t.Fatalf("failing health result = %+v, want unhealthy error", result)
	}
}
