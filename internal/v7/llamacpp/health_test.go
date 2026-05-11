package llamacpp

import (
	"context"
	"fmt"
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

func TestFetchServerPropertiesParsesSafeBuildHints(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/props" {
			t.Fatalf("unexpected path = %q", r.URL.Path)
		}
		_, _ = fmt.Fprint(w, `{
			"build_info": "llama.cpp b999 CUDA",
			"system_info": "AVX2 CUDA = 1",
			"default_generation_settings": {
				"n_gpu_layers": 999
			}
		}`)
	}))
	t.Cleanup(server.Close)

	props, ok := FetchServerProperties(context.Background(), server.URL, server.Client(), time.Second)
	if !ok {
		t.Fatal("FetchServerProperties() ok = false, want true")
	}
	if props.BuildInfo != "llama.cpp b999 CUDA" ||
		props.SystemInfo != "AVX2 CUDA = 1" ||
		props.ReportedGPULayers != 999 ||
		len(props.ReportedAcceleration) != 1 ||
		props.ReportedAcceleration[0] != "cuda" {
		t.Fatalf("props = %+v, want safe CUDA build hints", props)
	}
}

func TestFetchServerPropertiesIsLocalOnly(t *testing.T) {
	t.Parallel()

	if props, ok := FetchServerProperties(context.Background(), "https://example.com", nil, time.Second); ok ||
		props.BuildInfo != "" ||
		props.SystemInfo != "" ||
		props.ReportedGPULayers != 0 ||
		len(props.ReportedAcceleration) != 0 {
		t.Fatalf("external props = %+v/%v, want safe empty failure", props, ok)
	}
}
