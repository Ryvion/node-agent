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
	"strconv"
	"sync"
	"testing"
)

func TestRegisterSignsExpectedMessage(t *testing.T) {
	pub, priv := testKeyPair()
	pubHex := hex.EncodeToString(pub)
	var (
		mu         sync.Mutex
		handlerErr error
	)
	setHandlerErr := func(err error) {
		mu.Lock()
		defer mu.Unlock()
		if handlerErr == nil {
			handlerErr = err
		}
	}

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/node/register" {
			setHandlerErr(fmt.Errorf("unexpected path: %s", r.URL.Path))
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if got := r.Header.Get("X-Bind-Token"); got != "bind-123" {
			setHandlerErr(fmt.Errorf("bind token header mismatch: %q", got))
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if got := r.Header.Get("X-Wallet"); got != "wallet-abc" {
			setHandlerErr(fmt.Errorf("wallet header mismatch: %q", got))
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		var req struct {
			PublicKeyHex      string `json:"public_key_hex"`
			DeviceType        string `json:"device_type"`
			DeclaredCountry   string `json:"declared_country"`
			GPUModel          string `json:"gpu_model"`
			CPUCores          uint32 `json:"cpu_cores"`
			RAMBytes          uint64 `json:"ram_bytes"`
			VRAMBytes         uint64 `json:"vram_bytes"`
			Sensors           string `json:"sensors"`
			BandwidthMbps     uint64 `json:"bandwidth_mbps"`
			GeohashBucket     uint64 `json:"geohash_bucket"`
			AttestationMethod uint32 `json:"attestation_method"`
			ReferralCode      string `json:"referral_code"`
			Signature         []byte `json:"signature"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			setHandlerErr(fmt.Errorf("decode request: %w", err))
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if req.DeclaredCountry != "CA" {
			setHandlerErr(fmt.Errorf("declared country mismatch: %q", req.DeclaredCountry))
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		parts := []string{"register", pubHex, req.DeviceType}
		if req.DeclaredCountry != "" {
			parts = append(parts, req.DeclaredCountry)
		}
		parts = append(parts,
			req.GPUModel,
			strconv.FormatUint(uint64(req.CPUCores), 10),
			strconv.FormatUint(req.RAMBytes, 10),
			strconv.FormatUint(req.VRAMBytes, 10),
			req.Sensors,
			strconv.FormatUint(req.BandwidthMbps, 10),
			strconv.FormatUint(req.GeohashBucket, 10),
			strconv.FormatUint(uint64(req.AttestationMethod), 10),
		)
		msg := signPayload(parts...)
		if !ed25519.Verify(pub, msg, req.Signature) {
			setHandlerErr(fmt.Errorf("invalid signature"))
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	c := New(ts.URL, pub, priv, WithBindToken("bind-123"), WithWallet("wallet-abc"))
	err := c.Register(context.Background(), Capabilities{
		GPUModel:          "RTX 4090",
		CPUCores:          16,
		RAMBytes:          64,
		VRAMBytes:         24,
		Sensors:           "nvidia",
		BandwidthMbps:     1000,
		GeohashBucket:     0,
		AttestationMethod: 0,
	}, "gpu", "ref-xyz", "ca")
	if err != nil {
		t.Fatalf("register failed: %v", err)
	}
	if handlerErr != nil {
		t.Fatalf("handler failed: %v", handlerErr)
	}
}

func TestFetchWorkNoWork(t *testing.T) {
	pub, priv := testKeyPair()
	pubHex := hex.EncodeToString(pub)
	var (
		mu         sync.Mutex
		handlerErr error
	)
	setHandlerErr := func(err error) {
		mu.Lock()
		defer mu.Unlock()
		if handlerErr == nil {
			handlerErr = err
		}
	}

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/node/work" {
			setHandlerErr(fmt.Errorf("unexpected path: %s", r.URL.Path))
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if got := r.URL.Query().Get("pubkey"); got != pubHex {
			setHandlerErr(fmt.Errorf("pubkey query mismatch: %q", got))
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		tsStr := r.Header.Get("X-Node-Timestamp")
		sigHex := r.Header.Get("X-Node-Signature")
		sig, err := hex.DecodeString(sigHex)
		if err != nil {
			setHandlerErr(fmt.Errorf("decode signature: %w", err))
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		msg := signPayload("work", pubHex, tsStr)
		if !ed25519.Verify(pub, msg, sig) {
			setHandlerErr(fmt.Errorf("invalid work signature"))
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		_, _ = w.Write([]byte(`{"has_work":false}`))
	}))
	defer ts.Close()

	c := New(ts.URL, pub, priv)
	work, err := c.FetchWork(context.Background())
	if err != nil {
		t.Fatalf("fetch work failed: %v", err)
	}
	if work != nil {
		t.Fatalf("expected nil work, got %+v", work)
	}
	if handlerErr != nil {
		t.Fatalf("handler failed: %v", handlerErr)
	}
}

func TestHeartbeatParsesVerifiedLocation(t *testing.T) {
	pub, priv := testKeyPair()
	pubHex := hex.EncodeToString(pub)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/node/heartbeat" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		var req struct {
			PublicKeyHex string `json:"public_key_hex"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if req.PublicKeyHex != pubHex {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"latest_version":      "v1.2.48",
			"country_code":        "CA",
			"location_approved":   true,
			"sovereign_verified":  true,
			"verification_source": "geoip_country_fallback",
			"trust_reason":        "declared country matches observed network country",
		})
	}))
	defer ts.Close()

	c := New(ts.URL, pub, priv)
	resp, err := c.Heartbeat(context.Background(), Metrics{TimestampMs: 123})
	if err != nil {
		t.Fatalf("heartbeat failed: %v", err)
	}
	if resp.LatestVersion != "v1.2.48" {
		t.Fatalf("latest version = %q, want %q", resp.LatestVersion, "v1.2.48")
	}
	if resp.CountryCode != "CA" || !resp.LocationApproved || !resp.SovereignVerified {
		t.Fatalf("unexpected heartbeat response: %+v", resp)
	}
}

func TestSubmitReceiptSignsExpectedMessage(t *testing.T) {
	pub, priv := testKeyPair()
	pubHex := hex.EncodeToString(pub)
	var (
		mu         sync.Mutex
		handlerErr error
	)
	setHandlerErr := func(err error) {
		mu.Lock()
		defer mu.Unlock()
		if handlerErr == nil {
			handlerErr = err
		}
	}

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/node/receipt" {
			setHandlerErr(fmt.Errorf("unexpected path: %s", r.URL.Path))
			w.WriteHeader(http.StatusNotFound)
			return
		}
		var req struct {
			JobID         string `json:"job_id"`
			PublicKeyHex  string `json:"public_key_hex"`
			ResultHashHex string `json:"result_hash_hex"`
			Units         uint64 `json:"metering_units"`
			Signature     []byte `json:"signature"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			setHandlerErr(fmt.Errorf("decode request: %w", err))
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if req.PublicKeyHex != pubHex {
			setHandlerErr(fmt.Errorf("public key mismatch: %s", req.PublicKeyHex))
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		msg := signPayload("receipt", req.JobID, pubHex, req.ResultHashHex, strconv.FormatUint(req.Units, 10))
		if !ed25519.Verify(pub, msg, req.Signature) {
			setHandlerErr(fmt.Errorf("invalid receipt signature"))
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	c := New(ts.URL, pub, priv)
	err := c.SubmitReceipt(context.Background(), Receipt{JobID: "job_1", ResultHashHex: "abcd", MeteringUnits: 3})
	if err != nil {
		t.Fatalf("submit receipt failed: %v", err)
	}
	if handlerErr != nil {
		t.Fatalf("handler failed: %v", handlerErr)
	}
}

func TestReportAgentHealthSignsExpectedMessageAndReturnsStop(t *testing.T) {
	pub, priv := testKeyPair()
	pubHex := hex.EncodeToString(pub)
	deploymentID := "agd_test"
	var (
		mu         sync.Mutex
		handlerErr error
	)
	setHandlerErr := func(err error) {
		mu.Lock()
		defer mu.Unlock()
		if handlerErr == nil {
			handlerErr = err
		}
	}

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/node/agent-health/"+deploymentID {
			setHandlerErr(fmt.Errorf("unexpected path: %s", r.URL.Path))
			w.WriteHeader(http.StatusNotFound)
			return
		}
		var req struct {
			PublicKeyHex  string `json:"public_key_hex"`
			TimestampMs   int64  `json:"timestamp_ms"`
			Status        string `json:"status"`
			UptimeSeconds int    `json:"uptime_seconds"`
			Signature     []byte `json:"signature"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			setHandlerErr(fmt.Errorf("decode request: %w", err))
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if req.PublicKeyHex != pubHex {
			setHandlerErr(fmt.Errorf("public key mismatch: %s", req.PublicKeyHex))
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		msg := signPayload("agent_health", pubHex, deploymentID, strconv.FormatInt(req.TimestampMs, 10), strconv.Itoa(req.UptimeSeconds), req.Status)
		if !ed25519.Verify(pub, msg, req.Signature) {
			setHandlerErr(fmt.Errorf("invalid agent health signature"))
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		_, _ = w.Write([]byte(`{"ok":true,"should_stop":true,"status":"stopped","job_status":"failed"}`))
	}))
	defer ts.Close()

	c := New(ts.URL, pub, priv)
	resp, err := c.ReportAgentHealth(context.Background(), deploymentID, 15)
	if err != nil {
		t.Fatalf("report agent health failed: %v", err)
	}
	if !resp.ShouldStop || resp.Status != "stopped" || resp.JobStatus != "failed" {
		t.Fatalf("unexpected health response: %+v", resp)
	}
	if handlerErr != nil {
		t.Fatalf("handler failed: %v", handlerErr)
	}
}

func testKeyPair() (ed25519.PublicKey, ed25519.PrivateKey) {
	seed := make([]byte, ed25519.SeedSize)
	for i := range seed {
		seed[i] = byte(i + 1)
	}
	priv := ed25519.NewKeyFromSeed(seed)
	pub := priv.Public().(ed25519.PublicKey)
	return pub, priv
}

func signPayload(parts ...string) []byte {
	joined := "RYV1|"
	for i, p := range parts {
		if i > 0 {
			joined += "|"
		}
		joined += p
	}
	sum := sha256.Sum256([]byte(joined))
	return sum[:]
}
