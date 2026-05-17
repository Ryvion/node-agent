package netprofile

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestProbeHTTPHandlesMockServer(t *testing.T) {
	var heads atomic.Int32
	var gets atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("User-Agent"); got != "ryvion-test" {
			t.Fatalf("User-Agent = %q, want ryvion-test", got)
		}
		switch r.Method {
		case http.MethodHead:
			heads.Add(1)
			w.WriteHeader(http.StatusNoContent)
		case http.MethodGet:
			gets.Add(1)
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(strings.Repeat("x", 128)))
		default:
			t.Fatalf("unexpected method %s", r.Method)
		}
	}))
	defer server.Close()

	profile, err := ProbeHTTP(context.Background(), ProbeConfig{
		TargetURL:           server.URL,
		Samples:             3,
		TimeoutMs:           1000,
		MaxDownloadBytes:    64,
		EnableDownloadProbe: true,
		UserAgent:           "ryvion-test",
	})
	if err != nil {
		t.Fatalf("ProbeHTTP() error = %v", err)
	}
	if profile.SampleCount != 3 {
		t.Fatalf("SampleCount = %d, want 3", profile.SampleCount)
	}
	if profile.ProbeTarget != server.URL {
		t.Fatalf("ProbeTarget = %q, want %q", profile.ProbeTarget, server.URL)
	}
	if profile.RTTMsP50 <= 0 {
		t.Fatalf("RTTMsP50 = %v, want > 0", profile.RTTMsP50)
	}
	if profile.DownloadMbpsP50 <= 0 {
		t.Fatalf("DownloadMbpsP50 = %v, want > 0", profile.DownloadMbpsP50)
	}
	if profile.LossRateP95 != 0 {
		t.Fatalf("LossRateP95 = %v, want 0", profile.LossRateP95)
	}
	if got := heads.Load(); got != 3 {
		t.Fatalf("HEAD requests = %d, want 3", got)
	}
	if got := gets.Load(); got != 3 {
		t.Fatalf("GET requests = %d, want 3", got)
	}
	if profile.UploadMbpsP50 != 0 || profile.UploadMbpsP95 != 0 {
		t.Fatalf("upload metrics = %v/%v, want explicit unknown zeros", profile.UploadMbpsP50, profile.UploadMbpsP95)
	}
}

func TestProbeHTTPFailedSamplesIncreaseLoss(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodHead {
			t.Fatalf("unexpected method %s", r.Method)
		}
		call := calls.Add(1)
		if call%2 == 1 {
			http.Error(w, "temporary failure", http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	profile, err := ProbeHTTP(context.Background(), ProbeConfig{
		TargetURL: server.URL,
		Samples:   4,
		TimeoutMs: 1000,
	})
	if err == nil {
		t.Fatal("ProbeHTTP() error = nil, want failed sample error")
	}
	if profile.SampleCount != 4 {
		t.Fatalf("SampleCount = %d, want 4", profile.SampleCount)
	}
	if !nearlyEqual(profile.LossRateP95, 0.5) {
		t.Fatalf("LossRateP95 = %v, want 0.5", profile.LossRateP95)
	}
	if profile.RTTMsP50 <= 0 {
		t.Fatalf("RTTMsP50 = %v, want successful samples to contribute", profile.RTTMsP50)
	}
}

func TestProbeHTTPContextCancellation(t *testing.T) {
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	profile, err := ProbeHTTP(ctx, ProbeConfig{
		TargetURL: server.URL,
		Samples:   3,
		TimeoutMs: 1000,
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("ProbeHTTP() error = %v, want context.Canceled", err)
	}
	if profile.SampleCount != 0 {
		t.Fatalf("SampleCount = %d, want 0", profile.SampleCount)
	}
	if got := requests.Load(); got != 0 {
		t.Fatalf("requests = %d, want 0", got)
	}
}

func TestProbeHTTPDownloadCapRespected(t *testing.T) {
	var rangeHeader atomic.Value
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			w.WriteHeader(http.StatusNoContent)
		case http.MethodGet:
			rangeHeader.Store(r.Header.Get("Range"))
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(strings.Repeat("x", 1024)))
		default:
			t.Fatalf("unexpected method %s", r.Method)
		}
	}))
	defer server.Close()

	profile, err := ProbeHTTP(context.Background(), ProbeConfig{
		TargetURL:           server.URL,
		Samples:             1,
		TimeoutMs:           1000,
		MaxDownloadBytes:    8,
		EnableDownloadProbe: true,
	})
	if err != nil {
		t.Fatalf("ProbeHTTP() error = %v", err)
	}
	if got, _ := rangeHeader.Load().(string); got != "bytes=0-7" {
		t.Fatalf("Range = %q, want bytes=0-7", got)
	}
	if profile.DownloadMbpsP50 <= 0 {
		t.Fatalf("DownloadMbpsP50 = %v, want > 0", profile.DownloadMbpsP50)
	}
}

func TestProbeHTTPDefaultConfigDoesNotPerformUploadProbe(t *testing.T) {
	var uploadRequests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodHead:
			w.WriteHeader(http.StatusNoContent)
		case http.MethodPost, http.MethodPut, http.MethodPatch:
			uploadRequests.Add(1)
			w.WriteHeader(http.StatusNoContent)
		default:
			t.Fatalf("unexpected method %s", r.Method)
		}
	}))
	defer server.Close()

	profile, err := ProbeHTTP(context.Background(), ProbeConfig{
		TargetURL: server.URL,
		Samples:   1,
		TimeoutMs: 1000,
	})
	if err != nil {
		t.Fatalf("ProbeHTTP() error = %v", err)
	}
	if got := uploadRequests.Load(); got != 0 {
		t.Fatalf("upload requests = %d, want 0", got)
	}
	if profile.UploadMbpsP50 != 0 || profile.UploadMbpsP95 != 0 {
		t.Fatalf("upload metrics = %v/%v, want explicit unknown zeros", profile.UploadMbpsP50, profile.UploadMbpsP95)
	}
}
