package hub

import (
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// 1. TestStress_ConcurrentHeartbeats
// ---------------------------------------------------------------------------

func TestStress_ConcurrentHeartbeats(t *testing.T) {
	t.Parallel()

	pub, priv := testKeyPair()
	pubHex := hex.EncodeToString(pub)

	var received atomic.Int64
	var sigErrors atomic.Int64

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/node/heartbeat" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		var req struct {
			PublicKeyHex string  `json:"public_key_hex"`
			TimestampMs  int64   `json:"timestamp_ms"`
			CPUUtil      float64 `json:"cpu_util"`
			MemUtil      float64 `json:"mem_util"`
			GPUUtil      float64 `json:"gpu_util"`
			PowerWatts   float64 `json:"power_watts"`
			Signature    []byte  `json:"signature"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}

		// Verify signature
		msg := signPayload(
			"heartbeat",
			pubHex,
			fmt.Sprintf("%d", req.TimestampMs),
			formatFloat(req.CPUUtil),
			formatFloat(req.MemUtil),
			formatFloat(req.GPUUtil),
			formatFloat(req.PowerWatts),
		)
		if !ed25519.Verify(pub, msg, req.Signature) {
			sigErrors.Add(1)
			w.WriteHeader(http.StatusUnauthorized)
			return
		}

		received.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"latest_version":"1.0.0"}`))
	}))
	defer ts.Close()

	c := New(ts.URL, pub, priv)

	const goroutines = 50
	var wg sync.WaitGroup
	wg.Add(goroutines)

	errors := make([]error, goroutines)
	for i := 0; i < goroutines; i++ {
		go func(idx int) {
			defer wg.Done()
			_, err := c.Heartbeat(context.Background(), Metrics{
				TimestampMs: time.Now().UnixMilli() + int64(idx),
				CPUUtil:     float64(idx),
				MemUtil:     float64(idx) * 1.5,
				GPUUtil:     float64(idx) * 0.5,
				PowerWatts:  float64(idx) * 10,
			})
			errors[idx] = err
		}(i)
	}

	wg.Wait()

	for i, err := range errors {
		if err != nil {
			t.Errorf("goroutine %d failed: %v", i, err)
		}
	}
	if sigErrors.Load() > 0 {
		t.Errorf("signature verification failed %d times", sigErrors.Load())
	}
	if got := received.Load(); got != goroutines {
		t.Errorf("server received %d heartbeats, want %d", got, goroutines)
	}
}

// ---------------------------------------------------------------------------
// 2. TestStress_ConcurrentReceiptSubmission
// ---------------------------------------------------------------------------

func TestStress_ConcurrentReceiptSubmission(t *testing.T) {
	t.Parallel()

	pub, priv := testKeyPair()
	pubHex := hex.EncodeToString(pub)

	var received atomic.Int64
	var sigErrors atomic.Int64

	seenJobs := struct {
		mu sync.Mutex
		m  map[string]bool
	}{m: make(map[string]bool)}

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/node/receipt" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		var req struct {
			JobID         string         `json:"job_id"`
			PublicKeyHex  string         `json:"public_key_hex"`
			ResultHashHex string         `json:"result_hash_hex"`
			MeteringUnits uint64         `json:"metering_units"`
			Signature     []byte         `json:"signature"`
			Metadata      map[string]any `json:"metadata,omitempty"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}

		// Verify signature
		msg := signPayload("receipt", req.JobID, pubHex, req.ResultHashHex, fmt.Sprintf("%d", req.MeteringUnits))
		if !ed25519.Verify(pub, msg, req.Signature) {
			sigErrors.Add(1)
			w.WriteHeader(http.StatusUnauthorized)
			return
		}

		seenJobs.mu.Lock()
		seenJobs.m[req.JobID] = true
		seenJobs.mu.Unlock()

		received.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	c := New(ts.URL, pub, priv)

	const goroutines = 100
	var wg sync.WaitGroup
	wg.Add(goroutines)

	errors := make([]error, goroutines)
	for i := 0; i < goroutines; i++ {
		go func(idx int) {
			defer wg.Done()
			jobID := fmt.Sprintf("job_%04d", idx)
			hash := sha256.Sum256([]byte(jobID))
			hashHex := hex.EncodeToString(hash[:])
			err := c.SubmitReceipt(context.Background(), Receipt{
				JobID:         jobID,
				ResultHashHex: hashHex,
				MeteringUnits: uint64(idx + 1),
			})
			errors[idx] = err
		}(i)
	}

	wg.Wait()

	for i, err := range errors {
		if err != nil {
			t.Errorf("goroutine %d failed: %v", i, err)
		}
	}
	if sigErrors.Load() > 0 {
		t.Errorf("signature verification failed %d times", sigErrors.Load())
	}
	if got := received.Load(); got != goroutines {
		t.Errorf("server received %d receipts, want %d", got, goroutines)
	}

	seenJobs.mu.Lock()
	defer seenJobs.mu.Unlock()
	if len(seenJobs.m) != goroutines {
		t.Errorf("unique job IDs received: %d, want %d", len(seenJobs.m), goroutines)
	}
}

// ---------------------------------------------------------------------------
// 3. TestStress_FetchWorkTimeout
// ---------------------------------------------------------------------------

func TestStress_FetchWorkTimeout(t *testing.T) {
	t.Parallel()

	pub, priv := testKeyPair()

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Simulate a slow server
		time.Sleep(5 * time.Second)
		_, _ = w.Write([]byte(`{"has_work":false}`))
	}))
	defer ts.Close()

	c := New(ts.URL, pub, priv)

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	_, err := c.FetchWork(ctx)
	if err == nil {
		t.Fatal("expected error from FetchWork with short deadline, got nil")
	}
	// The error should be related to context deadline or timeout
	errStr := err.Error()
	if !strings.Contains(errStr, "deadline exceeded") &&
		!strings.Contains(errStr, "context deadline") &&
		!strings.Contains(errStr, "Timeout") &&
		!strings.Contains(errStr, "timeout") &&
		!strings.Contains(errStr, "canceled") {
		t.Errorf("expected timeout/deadline error, got: %v", err)
	}
}

// ---------------------------------------------------------------------------
// 4. TestStress_SignatureConsistency
// ---------------------------------------------------------------------------

func TestStress_SignatureConsistency(t *testing.T) {
	t.Parallel()

	pub, priv := testKeyPair()
	c := New("http://localhost", pub, priv)

	// Ed25519 is deterministic — same message must produce the same signature
	const iterations = 1000
	message := []string{"receipt", "job_123", c.pubHex(), "abcdef", "42"}

	firstSig := c.sign(message...)
	firstHex := hex.EncodeToString(firstSig)

	for i := 1; i < iterations; i++ {
		sig := c.sign(message...)
		sigHex := hex.EncodeToString(sig)
		if sigHex != firstHex {
			t.Fatalf("signature mismatch at iteration %d: got %s, want %s", i, sigHex, firstHex)
		}
		// Also verify the signature is valid
		payload := "RYV1|" + strings.Join(message, "|")
		sum := sha256.Sum256([]byte(payload))
		if !ed25519.Verify(pub, sum[:], sig) {
			t.Fatalf("signature verification failed at iteration %d", i)
		}
	}
}

// ---------------------------------------------------------------------------
// 5. TestStress_ServerErrorHandling
// ---------------------------------------------------------------------------

func TestStress_ServerErrorHandling(t *testing.T) {
	t.Parallel()

	pub, priv := testKeyPair()

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"internal server error"}`))
	}))
	defer ts.Close()

	c := New(ts.URL, pub, priv)
	ctx := context.Background()

	// Register should return error, not panic
	t.Run("register_500", func(t *testing.T) {
		err := c.Register(ctx, Capabilities{
			GPUModel: "RTX 4090",
			CPUCores: 16,
			RAMBytes: 64,
			VRAMBytes: 24,
		}, "gpu", "", "US")
		if err == nil {
			t.Fatal("expected error from Register with 500 server, got nil")
		}
		if !strings.Contains(err.Error(), "500") {
			t.Errorf("expected error to contain 500, got: %v", err)
		}
	})

	// SubmitReceipt should return error, not panic
	t.Run("receipt_500", func(t *testing.T) {
		hash := sha256.Sum256([]byte("test"))
		err := c.SubmitReceipt(ctx, Receipt{
			JobID:         "job_err",
			ResultHashHex: hex.EncodeToString(hash[:]),
			MeteringUnits: 1,
		})
		if err == nil {
			t.Fatal("expected error from SubmitReceipt with 500 server, got nil")
		}
		if !strings.Contains(err.Error(), "500") {
			t.Errorf("expected error to contain 500, got: %v", err)
		}
	})

	// Heartbeat should return error, not panic
	t.Run("heartbeat_500", func(t *testing.T) {
		_, err := c.Heartbeat(ctx, Metrics{
			TimestampMs: time.Now().UnixMilli(),
			CPUUtil:     50.0,
			MemUtil:     60.0,
		})
		if err == nil {
			t.Fatal("expected error from Heartbeat with 500 server, got nil")
		}
		if !strings.Contains(err.Error(), "500") {
			t.Errorf("expected error to contain 500, got: %v", err)
		}
	})

	// SendHealthReport should return error, not panic
	t.Run("health_500", func(t *testing.T) {
		err := c.SendHealthReport(ctx, HealthReport{
			TimestampMs: time.Now().UnixMilli(),
			GPUReady:    true,
			DockerGPU:   false,
			Message:     "test",
		})
		if err == nil {
			t.Fatal("expected error from SendHealthReport with 500 server, got nil")
		}
		if !strings.Contains(err.Error(), "500") {
			t.Errorf("expected error to contain 500, got: %v", err)
		}
	})

	// SolveChallenge should return error, not panic
	t.Run("challenge_500", func(t *testing.T) {
		err := c.SolveChallenge(ctx)
		if err == nil {
			t.Fatal("expected error from SolveChallenge with 500 server, got nil")
		}
		if !strings.Contains(err.Error(), "500") {
			t.Errorf("expected error to contain 500, got: %v", err)
		}
	})

	// SavePayout should return error, not panic
	t.Run("payout_500", func(t *testing.T) {
		err := c.SavePayout(ctx, "acct_test123", "USD")
		if err == nil {
			t.Fatal("expected error from SavePayout with 500 server, got nil")
		}
		if !strings.Contains(err.Error(), "500") {
			t.Errorf("expected error to contain 500, got: %v", err)
		}
	})
}

// ---------------------------------------------------------------------------
// 6. TestStress_LargePayloadSubmission
// ---------------------------------------------------------------------------

func TestStress_LargePayloadSubmission(t *testing.T) {
	t.Parallel()

	pub, priv := testKeyPair()

	var receivedSize atomic.Int64

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/node/receipt" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		var req struct {
			JobID         string         `json:"job_id"`
			PublicKeyHex  string         `json:"public_key_hex"`
			ResultHashHex string         `json:"result_hash_hex"`
			MeteringUnits uint64         `json:"metering_units"`
			Signature     []byte         `json:"signature"`
			Metadata      map[string]any `json:"metadata,omitempty"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		// Check that the large metadata arrived intact
		if req.Metadata != nil {
			if v, ok := req.Metadata["large_value"].(string); ok {
				receivedSize.Store(int64(len(v)))
			}
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	c := New(ts.URL, pub, priv)

	// Build 100KB metadata value
	const targetSize = 100 * 1024
	largeValue := strings.Repeat("A", targetSize)

	hash := sha256.Sum256([]byte("large-job"))
	err := c.SubmitReceipt(context.Background(), Receipt{
		JobID:         "job_large",
		ResultHashHex: hex.EncodeToString(hash[:]),
		MeteringUnits: 1,
		Metadata:      map[string]any{"large_value": largeValue},
	})
	if err != nil {
		t.Fatalf("SubmitReceipt with large metadata failed: %v", err)
	}

	got := receivedSize.Load()
	if got != targetSize {
		t.Errorf("server received metadata of size %d, want %d", got, targetSize)
	}
}

// formatFloat mirrors the formatFloatJSON helper in client.go for signature verification
func formatFloat(v float64) string {
	b, _ := json.Marshal(v)
	return string(b)
}
