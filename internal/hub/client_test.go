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
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Ryvion/ryvion-node/internal/capabilities/passport"
	heartbeat "github.com/Ryvion/ryvion-node/internal/hub/heartbeat"
	"github.com/Ryvion/ryvion-node/internal/hw"
	routedinference "github.com/Ryvion/ryvion-node/internal/inference/routed"
	netprofile "github.com/Ryvion/ryvion-node/internal/network/profile"
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
			CPUModel          string `json:"cpu_model"`
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
		if req.CPUModel != "AMD Ryzen 9 7900X" {
			setHandlerErr(fmt.Errorf("cpu model mismatch: %q", req.CPUModel))
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
		CPUModel:          "AMD Ryzen 9 7900X",
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

func TestSendHealthReportUsesManagedOCIField(t *testing.T) {
	pub, priv := testKeyPair()
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
		if r.URL.Path != "/api/v1/node/health" {
			setHandlerErr(fmt.Errorf("unexpected path: %s", r.URL.Path))
			w.WriteHeader(http.StatusNotFound)
			return
		}
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			setHandlerErr(fmt.Errorf("decode request: %w", err))
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if _, ok := req["runtime_gpu"]; ok {
			setHandlerErr(fmt.Errorf("legacy runtime_gpu field should not be sent"))
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if got, ok := req["managed_oci_gpu_ready"].(bool); !ok || !got {
			setHandlerErr(fmt.Errorf("managed_oci_gpu_ready = %v, want true", req["managed_oci_gpu_ready"]))
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	c := New(ts.URL, pub, priv)
	err := c.SendHealthReport(context.Background(), HealthReport{
		TimestampMs: 123,
		GPUReady:    true,
		RuntimeGPU:  true,
		Message:     "runtime-ready:1",
	})
	if err != nil {
		t.Fatalf("send health report failed: %v", err)
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

func TestFetchWorkGraphAbortSignsCurrentGraphRequest(t *testing.T) {
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
		if r.URL.Path != "/api/v1/node/workgraph-aborts" {
			setHandlerErr(fmt.Errorf("unexpected path: %s", r.URL.Path))
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if got := r.URL.Query().Get("pubkey"); got != pubHex {
			setHandlerErr(fmt.Errorf("pubkey query mismatch: %q", got))
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if got := r.URL.Query().Get("workgraph_id"); got != "wg-current" {
			setHandlerErr(fmt.Errorf("workgraph_id query mismatch: %q", got))
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
		msg := signPayload("workgraph_abort", pubHex, "wg-current", tsStr)
		if !ed25519.Verify(pub, msg, sig) {
			setHandlerErr(fmt.Errorf("invalid workgraph abort signature"))
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		_, _ = w.Write([]byte(`{"schema_version":"ryvion.node_workgraph_aborts.v1","status":"ok","aborted":true,"abort":{"workgraph_hash":"sha256:test","abort_epoch":9,"reason":"disconnect","no_credit_after_ms":1700000000000}}`))
	}))
	defer ts.Close()

	c := New(ts.URL, pub, priv)
	abort, err := c.FetchWorkGraphAbort(context.Background(), "wg-current")
	if err != nil {
		t.Fatalf("fetch abort failed: %v", err)
	}
	if abort == nil || abort.AbortEpoch != 9 || abort.Reason != "disconnect" {
		t.Fatalf("abort = %#v, want epoch 9 disconnect", abort)
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
			"latest_version":         "v1.2.48",
			"node_id":                "node-123",
			"country_code":           "CA",
			"location_approved":      true,
			"sovereign_verified":     true,
			"verification_source":    "geoip_country_fallback",
			"trust_reason":           "declared country matches observed network country",
			"v7_snapshot_upserted":   true,
			"snapshot_model_count":   2,
			"snapshot_backend_count": 1,
			"has_capability_profile": true,
			"hub_instance_id":        "hub-test",
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
	if resp.NodeID != "node-123" ||
		resp.V7SnapshotUpserted == nil ||
		!*resp.V7SnapshotUpserted ||
		resp.SnapshotModelCount != 2 ||
		resp.SnapshotBackendCount != 1 ||
		!resp.HasCapabilityProfile ||
		resp.HubInstanceID != "hub-test" {
		t.Fatalf("unexpected V7 heartbeat response summary: %+v", resp)
	}
}

func TestHeartbeatProbeTargetUsesLightweightPingEndpoint(t *testing.T) {
	pub, priv := testKeyPair()
	c := New("https://api.ryvion.ai", pub, priv)
	if got := c.HeartbeatProbeTarget(); got != "https://api.ryvion.ai/api/v1/ping" {
		t.Fatalf("HeartbeatProbeTarget() = %q", got)
	}
}

func TestProbeHubRTTUsesHeadPingEndpoint(t *testing.T) {
	pub, priv := testKeyPair()
	var gotMethod string
	var gotPath string
	var calls int
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		gotMethod = r.Method
		gotPath = r.URL.Path
		if r.Method != http.MethodHead || r.URL.Path != "/api/v1/ping" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer ts.Close()

	c := New(ts.URL, pub, priv)
	rtt, err := c.ProbeHubRTT(context.Background())
	if err != nil {
		t.Fatalf("ProbeHubRTT() error = %v", err)
	}
	if rtt <= 0 || rtt > time.Second {
		t.Fatalf("rtt = %s, want small positive duration", rtt)
	}
	if gotMethod != http.MethodHead || gotPath != "/api/v1/ping" {
		t.Fatalf("probe request = %s %s, want HEAD /api/v1/ping", gotMethod, gotPath)
	}
	if calls != 3 {
		t.Fatalf("probe calls = %d, want best-of-3 HEAD probes", calls)
	}
}

func TestProbeHubRTTUsesBestHeadSample(t *testing.T) {
	pub, priv := testKeyPair()
	delays := []time.Duration{80 * time.Millisecond, 10 * time.Millisecond, 60 * time.Millisecond}
	var calls int
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodHead || r.URL.Path != "/api/v1/ping" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if calls < len(delays) {
			time.Sleep(delays[calls])
		}
		calls++
		w.WriteHeader(http.StatusNoContent)
	}))
	defer ts.Close()

	c := New(ts.URL, pub, priv)
	rtt, err := c.ProbeHubRTT(context.Background())
	if err != nil {
		t.Fatalf("ProbeHubRTT() error = %v", err)
	}
	if calls != 3 {
		t.Fatalf("probe calls = %d, want 3", calls)
	}
	if rtt >= 50*time.Millisecond {
		t.Fatalf("rtt = %s, want best low sample rather than slow first sample", rtt)
	}
}

func TestHeartbeatOmitsV7PayloadWhenUnset(t *testing.T) {
	pub, priv := testKeyPair()
	var handlerErr error

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/node/heartbeat" {
			handlerErr = fmt.Errorf("unexpected path: %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		var req struct {
			PublicKeyHex string          `json:"public_key_hex"`
			TimestampMs  int64           `json:"timestamp_ms"`
			V7           json.RawMessage `json:"v7"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			handlerErr = fmt.Errorf("decode request: %w", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if len(req.V7) != 0 {
			handlerErr = fmt.Errorf("v7 payload should be omitted when unset: %s", string(req.V7))
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	c := New(ts.URL, pub, priv)
	if _, err := c.Heartbeat(context.Background(), Metrics{TimestampMs: 123}); err != nil {
		t.Fatalf("heartbeat failed: %v", err)
	}
	if handlerErr != nil {
		t.Fatalf("handler failed: %v", handlerErr)
	}
}

func TestHeartbeatIncludesV7PayloadAndPreservesOldFields(t *testing.T) {
	pub, priv := testKeyPair()
	pubHex := hex.EncodeToString(pub)
	v7Payload := testV7HeartbeatPayload(t)
	var handlerErr error

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/node/heartbeat" {
			handlerErr = fmt.Errorf("unexpected path: %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		var req struct {
			PublicKeyHex string          `json:"public_key_hex"`
			TimestampMs  int64           `json:"timestamp_ms"`
			CPUUtil      float64         `json:"cpu_util"`
			MemUtil      float64         `json:"mem_util"`
			GPUUtil      float64         `json:"gpu_util"`
			PowerWatts   float64         `json:"power_watts"`
			GPUThrottled bool            `json:"gpu_throttled"`
			Network      json.RawMessage `json:"network_profile"`
			V7           json.RawMessage `json:"v7"`
			Signature    []byte          `json:"signature"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			handlerErr = fmt.Errorf("decode request: %w", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if req.PublicKeyHex != pubHex || req.TimestampMs != 456 || req.CPUUtil != 1.25 || req.MemUtil != 2.5 ||
			req.GPUUtil != 3.75 || req.PowerWatts != 4.5 || !req.GPUThrottled {
			handlerErr = fmt.Errorf("old heartbeat fields not preserved: %+v", req)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if len(req.V7) == 0 {
			handlerErr = fmt.Errorf("v7 payload missing")
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		var network struct {
			RTTMsP95    float64 `json:"rtt_ms_p95"`
			JitterMsP95 float64 `json:"jitter_ms_p95"`
			SampleCount int     `json:"sample_count"`
		}
		if err := json.Unmarshal(req.Network, &network); err != nil {
			handlerErr = fmt.Errorf("decode root network profile: %w", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if network.RTTMsP95 != 44 || network.JitterMsP95 != 3 || network.SampleCount != 2 {
			handlerErr = fmt.Errorf("root network profile = %#v, want heartbeat RTT profile", network)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		var v7 struct {
			CapabilityPassport struct {
				SchemaVersion string `json:"schema_version"`
			} `json:"capability_passport"`
		}
		if err := json.Unmarshal(req.V7, &v7); err != nil {
			handlerErr = fmt.Errorf("decode v7 payload: %w", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if v7.CapabilityPassport.SchemaVersion != capability.SchemaVersionV1 {
			handlerErr = fmt.Errorf("passport schema = %q, want %q", v7.CapabilityPassport.SchemaVersion, capability.SchemaVersionV1)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		msg := signPayload(
			"heartbeat",
			pubHex,
			strconv.FormatInt(req.TimestampMs, 10),
			formatFloatJSON(req.CPUUtil),
			formatFloatJSON(req.MemUtil),
			formatFloatJSON(req.GPUUtil),
			formatFloatJSON(req.PowerWatts),
		)
		if !ed25519.Verify(pub, msg, req.Signature) {
			handlerErr = fmt.Errorf("invalid signature")
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	c := New(ts.URL, pub, priv)
	if _, err := c.Heartbeat(context.Background(), Metrics{
		TimestampMs:  456,
		CPUUtil:      1.25,
		MemUtil:      2.5,
		GPUUtil:      3.75,
		PowerWatts:   4.5,
		GPUThrottled: true,
		NetworkProfile: &netprofile.NetworkProfile{
			RTTMsP95:    44,
			JitterMsP95: 3,
			SampleCount: 2,
		},
		V7Heartbeat: v7Payload,
	}); err != nil {
		t.Fatalf("heartbeat failed: %v", err)
	}
	if handlerErr != nil {
		t.Fatalf("handler failed: %v", handlerErr)
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

func TestSubmitSpeculativeDraftPacketUsesWindowEndpointAndAdminKey(t *testing.T) {
	var gotPath string
	var gotAdmin string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAdmin = r.Header.Get("X-Admin-Key")
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s, want POST", r.Method)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		if body["packet_id"] != "pkt-client" {
			t.Fatalf("body = %#v", body)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"schema_version":"ryvion.speculative.draft_packet_decision.v1","accepted":true,"reason":"accepted","packet_id":"pkt-client"}`))
	}))
	defer ts.Close()

	pub, priv := testKeyPair()
	c := New(ts.URL, pub, priv, WithAdminKey("admin-secret"))
	decision, err := c.SubmitSpeculativeDraftPacket(context.Background(), "win-client", map[string]any{"packet_id": "pkt-client"})
	if err != nil {
		t.Fatalf("SubmitSpeculativeDraftPacket() error = %v", err)
	}
	if gotPath != "/api/v1/speculative/windows/win-client/draft-packets" {
		t.Fatalf("path = %q", gotPath)
	}
	if gotAdmin != "admin-secret" {
		t.Fatalf("admin header = %q", gotAdmin)
	}
	if !decision.Accepted || decision.Reason != "accepted" {
		t.Fatalf("decision = %+v", decision)
	}
}

func TestSubmitSpeculativeDraftPacketBatchUsesBatchEndpoint(t *testing.T) {
	var gotPath string
	var gotAdmin string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAdmin = r.Header.Get("X-Admin-Key")
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s, want POST", r.Method)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		packets, ok := body["packets"].([]any)
		if !ok || len(packets) != 2 {
			t.Fatalf("body packets = %#v", body["packets"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"schema_version":"ryvion.speculative.draft_packet_batch_decision.v1","window_id":"win-client","attempted":2,"accepted":2,"rejected":0,"decisions":[{"accepted":true,"reason":"accepted","packet_id":"pkt-a"},{"accepted":true,"reason":"accepted","packet_id":"pkt-b"}]}`))
	}))
	defer ts.Close()

	pub, priv := testKeyPair()
	c := New(ts.URL, pub, priv, WithAdminKey("admin-secret"))
	decision, err := c.SubmitSpeculativeDraftPacketBatch(context.Background(), "win-client", []map[string]any{
		{"packet_id": "pkt-a"},
		{"packet_id": "pkt-b"},
	})
	if err != nil {
		t.Fatalf("SubmitSpeculativeDraftPacketBatch() error = %v", err)
	}
	if gotPath != "/api/v1/speculative/windows/win-client/draft-packets/batch" {
		t.Fatalf("path = %q", gotPath)
	}
	if gotAdmin != "admin-secret" {
		t.Fatalf("admin header = %q", gotAdmin)
	}
	if decision.Attempted != 2 || decision.Accepted != 2 || len(decision.Decisions) != 2 {
		t.Fatalf("decision = %+v", decision)
	}
}

func TestSubmitSpeculativeLiveLabVerifierResultRedactsAcceptedTextForHTTPFallback(t *testing.T) {
	var body map[string]any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/api/v1/node/speculative/live-lab/runs/flab-client/verifier-results" {
			t.Fatalf("path = %q", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	pub, priv := testKeyPair()
	client := New(ts.URL, pub, priv)
	err := client.SubmitSpeculativeLiveLabVerifierResult(context.Background(), "flab-client", SpeculativeLiveLabVerifierResult{
		JobID:              "job-verify",
		WindowID:           "win-client",
		WaveIndex:          1,
		AcceptedLen:        2,
		AcceptedText:       "private accepted text",
		AcceptedTextPublic: true,
	})
	if err != nil {
		t.Fatalf("SubmitSpeculativeLiveLabVerifierResult() error = %v", err)
	}
	if body["accepted_text"] != nil {
		t.Fatalf("HTTP fallback leaked accepted_text: %#v", body)
	}
	sum := sha256.Sum256([]byte("private accepted text"))
	if body["accepted_text_hash"] != "sha256:"+hex.EncodeToString(sum[:]) {
		t.Fatalf("accepted_text_hash = %#v, want hash", body["accepted_text_hash"])
	}
}

func TestSendDashboardInferenceProgressPostsChunkBatch(t *testing.T) {
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
		if r.URL.Path != "/api/v1/node/inference/chunks" {
			setHandlerErr(fmt.Errorf("unexpected path: %s", r.URL.Path))
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if got := r.Header.Get("X-Node-Pubkey"); got != pubHex {
			setHandlerErr(fmt.Errorf("node pubkey header = %q, want %q", got, pubHex))
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		var req struct {
			RunID        string                          `json:"run_id"`
			JobID        string                          `json:"job_id"`
			NodeID       string                          `json:"node_id"`
			PublicKeyHex string                          `json:"public_key_hex"`
			SeqStart     int64                           `json:"seq_start"`
			Chunks       []routedinference.ProgressChunk `json:"chunks"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			setHandlerErr(fmt.Errorf("decode request: %w", err))
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if req.RunID != "run_1" || req.JobID != "job_1" || req.NodeID != pubHex || req.PublicKeyHex != pubHex || req.SeqStart != 1 {
			setHandlerErr(fmt.Errorf("progress identity = %+v", req))
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if len(req.Chunks) != 2 || req.Chunks[0].Seq != 1 || req.Chunks[0].Text != "Ryvion" || req.Chunks[1].Seq != 2 || req.Chunks[1].Text != " streams" {
			setHandlerErr(fmt.Errorf("chunks = %+v", req.Chunks))
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"accepted":true}`))
	}))
	defer ts.Close()

	c := New(ts.URL, pub, priv)
	err := c.SendDashboardInferenceProgress(context.Background(), routedinference.ProgressBatch{
		RunID:    "run_1",
		JobID:    "job_1",
		SeqStart: 1,
		Chunks: []routedinference.ProgressChunk{
			{Seq: 1, Type: "delta", Text: "Ryvion"},
			{Seq: 2, Type: "delta", Text: " streams"},
		},
	})
	if err != nil {
		t.Fatalf("SendDashboardInferenceProgress() error = %v", err)
	}
	if handlerErr != nil {
		t.Fatalf("handler failed: %v", handlerErr)
	}
}

func testV7HeartbeatPayload(t *testing.T) *heartbeat.V7HeartbeatPayload {
	t.Helper()

	payload, err := heartbeat.BuildV7HeartbeatPayload(heartbeat.BuildV7HeartbeatPayloadInput{
		AgentVersion:  "test",
		NodePublicKey: strings.Repeat("a", 64),
		OS:            "linux",
		Arch:          "amd64",
		HardwareCapabilities: hw.CapSet{
			CPUCores: 4,
			RAMBytes: 8 * 1024 * 1024 * 1024,
		},
		RuntimeProfile: capability.RuntimeProfile{
			NativeInferenceSupported: true,
			SupportedRunnerKinds:     []string{"native_streaming"},
		},
		SandboxCapabilitySummary: capability.SandboxCapabilitySummary{
			RejectsUnsafePickle: true,
		},
		CreatedAtUnixMs: 123,
	})
	if err != nil {
		t.Fatalf("BuildV7HeartbeatPayload() error = %v", err)
	}
	return &payload
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
