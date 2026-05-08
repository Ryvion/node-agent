package main

import (
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Ryvion/node-agent/internal/diagnostics"
	"github.com/Ryvion/node-agent/internal/hub"
	"github.com/Ryvion/node-agent/internal/hw"
	"github.com/Ryvion/node-agent/internal/inference"
	"github.com/Ryvion/node-agent/internal/runtimeexec"
	v7dashboardinference "github.com/Ryvion/node-agent/internal/v7/dashboardinference"
	v7inferencebench "github.com/Ryvion/node-agent/internal/v7/inferencebench"
	v7kvprobe "github.com/Ryvion/node-agent/internal/v7/kvprobe"
	v7llamacpp "github.com/Ryvion/node-agent/internal/v7/llamacpp"
	v7memorybench "github.com/Ryvion/node-agent/internal/v7/memorybench"
	v7modelbench "github.com/Ryvion/node-agent/internal/v7/modelbench"
	v7tensorplane "github.com/Ryvion/node-agent/internal/v7/tensorplane"
)

// fakeClient implements a minimal receipt submitter that fails N times then succeeds.
type fakeClient struct {
	failCount   int
	calls       atomic.Int32
	lastReceipt hub.Receipt
}

func (f *fakeClient) SubmitReceipt(_ context.Context, receipt hub.Receipt) error {
	f.lastReceipt = receipt
	n := int(f.calls.Add(1))
	if n <= f.failCount {
		return fmt.Errorf("simulated failure %d", n)
	}
	return nil
}

func TestSubmitReceiptWithRetry_SucceedsAfterFailures(t *testing.T) {
	fc := &fakeClient{failCount: 3}
	receipt := hub.Receipt{JobID: "test-job-1", ResultHashHex: "abc123", MeteringUnits: 1}

	err := submitReceiptWithRetryTestable(context.Background(), fc, receipt)
	if err != nil {
		t.Fatalf("expected success after transient failures, got: %v", err)
	}
	if got := int(fc.calls.Load()); got != 4 {
		t.Fatalf("expected 4 calls (3 fail + 1 success), got %d", got)
	}
}

func TestPrepareReceiptForSubmissionFlagOffDoesNotAttachV7Proof(t *testing.T) {
	t.Setenv(v7ProofFlagEnv, "")
	receipt := hub.Receipt{
		JobID:         "job-no-proof",
		ResultHashHex: strings.Repeat("a", 64),
		MeteringUnits: 2,
		Metadata: map[string]any{
			"executor": "oci",
		},
	}

	got := prepareReceiptForSubmission(receipt)
	if _, ok := got.Metadata[v7ProofMetadataKey]; ok {
		t.Fatalf("metadata contains %q with flag off: %+v", v7ProofMetadataKey, got.Metadata[v7ProofMetadataKey])
	}
	if got.Metadata["executor"] != "oci" {
		t.Fatalf("metadata changed with flag off: %+v", got.Metadata)
	}
}

func TestPrepareReceiptForSubmissionFlagOnWithOutputBytesAttachesV7Proof(t *testing.T) {
	t.Setenv(v7ProofFlagEnv, "1")
	rawOutput := []byte("raw runner output should not be serialized")
	receipt := hub.Receipt{
		JobID:         "job-proof-full",
		ResultHashHex: strings.Repeat("b", 64),
		MeteringUnits: 7,
		Metadata: map[string]any{
			v7ProofOutputBytesMetadataKey: rawOutput,
			"executor":                    "oci",
			"assignment_id":               "assignment-proof-full",
			"node_id":                     "node-proof-full",
			"model_id":                    "llama-3.1-8b.gguf",
			"model_revision":              "rev-1",
			"quantization_id":             "q4_k_m",
			"artifact_kind":               "result_json",
			"started_at_unix_ms":          int64(1_800_000_000_000),
			"finished_at_unix_ms":         int64(1_800_000_001_000),
		},
	}

	got := prepareReceiptForSubmission(receipt)
	if _, ok := got.Metadata[v7ProofOutputBytesMetadataKey]; ok {
		t.Fatalf("metadata still contains raw output source key %q", v7ProofOutputBytesMetadataKey)
	}
	proof, ok := got.Metadata[v7ProofMetadataKey].(map[string]any)
	if !ok {
		t.Fatalf("%q metadata type = %T, want map[string]any", v7ProofMetadataKey, got.Metadata[v7ProofMetadataKey])
	}
	if proof["proof_status"] != "complete" {
		t.Fatalf("proof_status = %v, want complete", proof["proof_status"])
	}
	if gotHash, ok := proof["output_hash"].(string); !ok || !strings.HasPrefix(gotHash, "sha256:") {
		t.Fatalf("output_hash = %v, want sha256 object id", proof["output_hash"])
	}
	if gotBytes, ok := proof["output_bytes"].(int64); !ok || gotBytes != int64(len(rawOutput)) {
		t.Fatalf("output_bytes = %v, want %d", proof["output_bytes"], len(rawOutput))
	}
	if proof["artifact_manifest"] == nil {
		t.Fatal("artifact_manifest is nil")
	}
	if proof["evidence_payload"] == nil {
		t.Fatal("evidence_payload is nil")
	}

	encoded, err := json.Marshal(got.Metadata)
	if err != nil {
		t.Fatalf("json.Marshal(metadata) error = %v", err)
	}
	if strings.Contains(string(encoded), string(rawOutput)) {
		t.Fatalf("V7 proof metadata contains raw output bytes: %s", encoded)
	}
}

func TestPrepareReceiptForSubmissionFlagOnNoOutputBytesAttachesPartialProof(t *testing.T) {
	t.Setenv(v7ProofFlagEnv, "1")
	resultHash := strings.Repeat("c", 64)
	receipt := hub.Receipt{
		JobID:         "job-proof-partial",
		ResultHashHex: resultHash,
		MeteringUnits: 3,
		Metadata: map[string]any{
			"executor": "oci",
		},
	}

	got := prepareReceiptForSubmission(receipt)
	proof, ok := got.Metadata[v7ProofMetadataKey].(map[string]any)
	if !ok {
		t.Fatalf("%q metadata type = %T, want map[string]any", v7ProofMetadataKey, got.Metadata[v7ProofMetadataKey])
	}
	if proof["proof_status"] != "unavailable_no_output_bytes" {
		t.Fatalf("proof_status = %v, want unavailable_no_output_bytes", proof["proof_status"])
	}
	if proof["result_hash"] != resultHash {
		t.Fatalf("result_hash = %v, want %s", proof["result_hash"], resultHash)
	}
	if proof["output_hash"] != "sha256:"+resultHash {
		t.Fatalf("output_hash = %v, want sha256 result hash", proof["output_hash"])
	}
	if proof["artifact_manifest"] != nil {
		t.Fatalf("artifact_manifest = %v, want nil for partial proof", proof["artifact_manifest"])
	}
	if proof["evidence_payload"] != nil {
		t.Fatalf("evidence_payload = %v, want nil for partial proof", proof["evidence_payload"])
	}
}

func TestSubmitReceiptWithRetryTestableV7ProofBuildFailureStillSubmits(t *testing.T) {
	t.Setenv(v7ProofFlagEnv, "1")
	rawOutput := []byte("raw output from failed proof build")
	fc := &fakeClient{}
	receipt := hub.Receipt{
		JobID:         "job-proof-build-failure",
		ResultHashHex: strings.Repeat("d", 64),
		MeteringUnits: 1,
		Metadata: map[string]any{
			v7ProofOutputBytesMetadataKey: rawOutput,
			"executor":                    "oci",
			"artifact_kind":               "invalid_kind",
		},
	}

	err := submitReceiptWithRetryTestable(context.Background(), fc, receipt)
	if err != nil {
		t.Fatalf("submitReceiptWithRetryTestable() error = %v", err)
	}
	if got := int(fc.calls.Load()); got != 1 {
		t.Fatalf("receipt submissions = %d, want 1", got)
	}
	proof, ok := fc.lastReceipt.Metadata[v7ProofMetadataKey].(map[string]any)
	if !ok {
		t.Fatalf("%q metadata type = %T, want map[string]any", v7ProofMetadataKey, fc.lastReceipt.Metadata[v7ProofMetadataKey])
	}
	if proof["proof_status"] != "build_failed" {
		t.Fatalf("proof_status = %v, want build_failed", proof["proof_status"])
	}
	if _, ok := fc.lastReceipt.Metadata[v7ProofOutputBytesMetadataKey]; ok {
		t.Fatalf("submitted metadata still contains raw output source key %q", v7ProofOutputBytesMetadataKey)
	}

	encoded, err := json.Marshal(fc.lastReceipt.Metadata)
	if err != nil {
		t.Fatalf("json.Marshal(metadata) error = %v", err)
	}
	if strings.Contains(string(encoded), string(rawOutput)) {
		t.Fatalf("submitted metadata contains raw output bytes: %s", encoded)
	}
}

func TestBuildOptionalV7HeartbeatPayloadDefaultsOnAndHonorsExplicitOff(t *testing.T) {
	t.Setenv("RYV_DISABLE_OCI", "1")
	t.Setenv("RYV_LLAMA_CPP_PROBE_MODEL", "")
	caps := hw.CapSet{
		CPUCores: 4,
		RAMBytes: 8 * 1024 * 1024 * 1024,
	}
	runtimeMgr := newRuntimeManager("test", runtimeContractMetadata{})

	t.Setenv("RYV_NODE_V7_CAPS", "0")
	if payload := buildOptionalV7HeartbeatPayload("pubkey", caps, "cpu", "ca", nil, runtimeMgr); payload != nil {
		t.Fatalf("payload = %+v, want nil when RYV_NODE_V7_CAPS is explicitly off", payload)
	}

	t.Setenv("RYV_NODE_V7_CAPS", "")
	payload := buildOptionalV7HeartbeatPayload("pubkey", caps, "cpu", "ca", nil, runtimeMgr)
	if payload == nil {
		t.Fatal("payload = nil, want V7 payload by default")
	}
	if payload.CapabilityPassport.NodePublicKey != "pubkey" {
		t.Fatalf("node public key = %q, want pubkey", payload.CapabilityPassport.NodePublicKey)
	}
	if payload.KVCapability == nil || payload.KVCapability.RuntimeKind != v7kvprobe.RuntimeKindNative {
		t.Fatalf("kv capability = %+v, want native runtime capability probe", payload.KVCapability)
	}
	expectedTensorAccess := buildRuntimeTensorAccessStatus(nil)
	if payload.TensorAccess != expectedTensorAccess {
		t.Fatalf("tensor_access = %+v, want local status builder value %+v", payload.TensorAccess, expectedTensorAccess)
	}
	if payload.TensorAccess.KVAccessSupported ||
		payload.TensorAccess.KVSnapshotSupported ||
		payload.TensorAccess.HiddenStateAccessSupported ||
		payload.TensorAccess.LogitsAccessSupported ||
		payload.TensorAccess.AttentionHookSupported {
		t.Fatalf("tensor_access should be reporting-only unsupported capability: %+v", payload.TensorAccess)
	}
	expectedRuntimeInventory := buildRuntimeInventoryStatus(operatorRuntimeInfo{}, expectedTensorAccess, nil)
	if !reflect.DeepEqual(payload.RuntimeInventory, expectedRuntimeInventory) {
		t.Fatalf("runtime_inventory = %+v, want local status builder value %+v", payload.RuntimeInventory, expectedRuntimeInventory)
	}
	if payload.RuntimeInventory.Provider != "noop" {
		t.Fatalf("runtime_inventory provider = %q, want noop", payload.RuntimeInventory.Provider)
	}
	if payload.HardwareCapacity.OS == "" ||
		payload.HardwareCapacity.Arch == "" ||
		payload.HardwareCapacity.GPUName == "" ||
		payload.HardwareCapacity.PowerProfile == "" ||
		payload.HardwareCapacity.ThermalRisk == "" {
		t.Fatalf("hardware_capacity missing safe status fields: %+v", payload.HardwareCapacity)
	}
	expectedBackendRuntimes := v7llamacpp.EnrichBackendRuntimes(v7llamacpp.NormalizeBackendRuntimes(v7llamacpp.BackendRuntimes{}), expectedRuntimeInventory, payload.HardwareCapacity)
	if !reflect.DeepEqual(payload.BackendRuntimes, expectedBackendRuntimes) {
		t.Fatalf("backend_runtimes = %+v, want default local builder value %+v", payload.BackendRuntimes, expectedBackendRuntimes)
	}
	expectedBackendProbes := buildBackendProbeStatus()
	if !reflect.DeepEqual(payload.BackendProbes, expectedBackendProbes) {
		t.Fatalf("backend_probes = %+v, want local status builder value %+v", payload.BackendProbes, expectedBackendProbes)
	}
	if payload.CapabilityProfile.SchemaVersion == "" || payload.CapabilityProfile.Reason == "" {
		t.Fatalf("capability_profile missing compact status: %+v", payload.CapabilityProfile)
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("json.Marshal(payload) error = %v", err)
	}
	if !strings.Contains(string(raw), `"tensor_access"`) {
		t.Fatalf("V7 heartbeat payload missing tensor_access: %s", raw)
	}
	for _, want := range []string{`"runtime_inventory"`, `"loaded_models"`, `"candidate_backends"`, `"backend_candidates"`, `"gguf_models"`, `"hardware_capacity"`, `"cpu_logical_cores"`, `"gpu_vram_bytes"`, `"model_policy"`, `"runtime_policy"`, `"model_cache"`, `"backend_probes"`, `"backend_runtimes"`, `"capability_profile"`, `"llama_cpp"`} {
		if !strings.Contains(string(raw), want) {
			t.Fatalf("V7 heartbeat payload missing %s: %s", want, raw)
		}
	}
	lower := strings.ToLower(string(raw))
	for _, forbidden := range []string{"raw_prompt", "prompt_text", "generated_text", "model_output", "output_text", "key_data", "value_data", "query_vector", "tensor_bytes", "raw_tensor", "weighted_value"} {
		if strings.Contains(lower, forbidden) {
			t.Fatalf("V7 heartbeat payload contains forbidden marker %q: %s", forbidden, raw)
		}
	}
}

func TestBuildV7HeartbeatPayloadIncludesActiveBackendRuntime(t *testing.T) {
	t.Setenv("RYV_NODE_V7_CAPS", "1")
	t.Setenv("RYV_LLAMA_CPP_PROBE_MODEL", "")
	caps := hw.CapSet{
		CPUCores: 4,
		RAMBytes: 8 * 1024 * 1024 * 1024,
	}
	backendRuntimes := v7llamacpp.BuildBackendRuntimes(v7llamacpp.LlamaCppSidecarStatus{
		Enabled:                true,
		Available:              true,
		Running:                true,
		Healthy:                true,
		BaseURL:                "http://127.0.0.1:45910",
		ModelPath:              "/tmp/ryvion-models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		ModelFilename:          "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		ModelSizeBytes:         2019377696,
		ModelFamilyHint:        "llama",
		QuantizationHint:       "Q4_K_M",
		Backend:                v7llamacpp.BackendName,
		OpenAICompatible:       true,
		SupportsTextGeneration: true,
		SupportsStreaming:      true,
	})

	payload := buildOptionalV7HeartbeatPayloadWithBackendRuntimes("pubkey", caps, "cpu", "ca", nil, nil, backendRuntimes)
	if payload == nil {
		t.Fatal("payload = nil, want V7 payload")
	}
	expectedBackendRuntimes := v7llamacpp.EnrichBackendRuntimes(backendRuntimes, payload.RuntimeInventory, payload.HardwareCapacity)
	if !reflect.DeepEqual(payload.BackendRuntimes, expectedBackendRuntimes) {
		t.Fatalf("backend_runtimes = %+v, want local builder value %+v", payload.BackendRuntimes, expectedBackendRuntimes)
	}
	runtime := payload.BackendRuntimes.LlamaCPP
	if !runtime.Loaded || !runtime.Warm || runtime.ModelID != "Llama-3.2-3B-Instruct-Q4_K_M.gguf" {
		t.Fatalf("backend runtime residency = %+v, want loaded warm llama.cpp model", runtime)
	}
	if runtime.SupportsKVAccess || runtime.SupportsTensorHooks {
		t.Fatalf("backend runtime should remain reporting-only without KV/tensor hooks: %+v", runtime)
	}
}

func TestBuildV7HeartbeatPayloadRuntimeInventoryMatchesOperatorStatusBuilder(t *testing.T) {
	t.Setenv("RYV_NODE_V7_CAPS", "1")
	t.Setenv("RYV_LLAMA_CPP_PROBE_MODEL", "")
	caps := hw.CapSet{
		CPUCores: 4,
		RAMBytes: 8 * 1024 * 1024 * 1024,
	}
	infMgr := inference.New(t.TempDir())
	runtimeMgr := newRuntimeManager("test", runtimeContractMetadata{})

	payload, err := buildV7HeartbeatPayloadForNode("pubkey", caps, "cpu", "ca", infMgr, runtimeMgr)
	if err != nil {
		t.Fatalf("buildV7HeartbeatPayloadForNode() error = %v", err)
	}
	expectedTensorAccess := buildRuntimeTensorAccessStatus(infMgr)
	expectedRuntimeInventory := buildRuntimeInventoryStatus(operatorRuntimeInfo{
		NativeInferenceReady: inference.NativeRuntimeAvailable() && infMgr.Healthy(),
		NativeModel:          infMgr.ModelName(),
	}, expectedTensorAccess, infMgr)
	if !reflect.DeepEqual(payload.RuntimeInventory, expectedRuntimeInventory) {
		t.Fatalf("runtime_inventory = %+v, want local status builder value %+v", payload.RuntimeInventory, expectedRuntimeInventory)
	}
	expectedBackendRuntimes := v7llamacpp.EnrichBackendRuntimes(v7llamacpp.NormalizeBackendRuntimes(v7llamacpp.BackendRuntimes{}), expectedRuntimeInventory, payload.HardwareCapacity)
	if !reflect.DeepEqual(payload.BackendRuntimes, expectedBackendRuntimes) {
		t.Fatalf("backend_runtimes = %+v, want local status builder value %+v", payload.BackendRuntimes, expectedBackendRuntimes)
	}
	if len(payload.RuntimeInventory.LoadedModels) != 1 {
		t.Fatalf("runtime_inventory loaded_models = %+v, want one local native model entry", payload.RuntimeInventory.LoadedModels)
	}
	if payload.RuntimeInventory.LoadedModels[0].ModelID != infMgr.ModelName() {
		t.Fatalf("loaded model id = %q, want %q", payload.RuntimeInventory.LoadedModels[0].ModelID, infMgr.ModelName())
	}
	expectedBackendProbes := buildBackendProbeStatus()
	if !reflect.DeepEqual(payload.BackendProbes, expectedBackendProbes) {
		t.Fatalf("backend_probes = %+v, want local status builder value %+v", payload.BackendProbes, expectedBackendProbes)
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("json.Marshal(payload) error = %v", err)
	}
	if !strings.Contains(string(raw), `"runtime_inventory"`) ||
		!strings.Contains(string(raw), `"candidate_backends"`) ||
		!strings.Contains(string(raw), `"backend_candidates"`) ||
		!strings.Contains(string(raw), `"gguf_models"`) ||
		!strings.Contains(string(raw), `"hardware_capacity"`) ||
		!strings.Contains(string(raw), `"model_policy"`) ||
		!strings.Contains(string(raw), `"runtime_policy"`) ||
		!strings.Contains(string(raw), `"model_cache"`) ||
		!strings.Contains(string(raw), `"backend_probes"`) ||
		!strings.Contains(string(raw), `"llama_cpp"`) {
		t.Fatalf("V7 heartbeat payload missing runtime inventory fields: %s", raw)
	}
}

func TestBuildOptionalV7HeartbeatPayloadDoesNotProbeRuntime(t *testing.T) {
	t.Setenv("RYV_NODE_V7_CAPS", "1")
	caps := hw.CapSet{
		CPUCores: 4,
		RAMBytes: 8 * 1024 * 1024 * 1024,
	}
	runtimeMgr := newRuntimeManager("test", runtimeContractMetadata{
		Binary:       "/opt/ryvion/runtime/ryvion-runtime",
		Backend:      "/opt/ryvion/runtime/backend/ryvion-oci",
		ManifestHash: "manifest-hash",
	})

	called := false
	prevProbe := probeManagedRuntimeStatus
	probeManagedRuntimeStatus = func(_ context.Context, _ string, _ func(string) string, _ string) (runtimeexec.Status, bool) {
		called = true
		return runtimeexec.Status{CLIInstalled: true, Ready: true, GPUReady: true, Health: "ready"}, true
	}
	defer func() { probeManagedRuntimeStatus = prevProbe }()

	payload := buildOptionalV7HeartbeatPayload("pubkey", caps, "cpu", "ca", nil, runtimeMgr)
	if payload == nil {
		t.Fatal("payload = nil, want V7 payload")
	}
	if called {
		t.Fatal("V7 heartbeat payload construction should not probe managed runtime status")
	}
	if !payload.EvidenceCapabilitySummary.SupportsRuntimeHash {
		t.Fatal("expected runtime hash support when runtime manifest hash is already available")
	}
}

func TestSendHeartbeatOmitsV7WhenFlagOff(t *testing.T) {
	t.Setenv("RYV_NODE_V7_CAPS", "0")
	client, gotV7, calls := heartbeatTestClient(t)

	ok := sendHeartbeat(context.Background(), client, validHeartbeatCaps(), "cpu", "ca", nil, nil)
	if !ok {
		t.Fatal("sendHeartbeat() = false, want true")
	}
	if calls.Load() != 1 {
		t.Fatalf("heartbeat calls = %d, want 1", calls.Load())
	}
	if gotV7.Load() {
		t.Fatal("heartbeat sent V7 payload with RYV_NODE_V7_CAPS explicitly off")
	}
}

func TestSendHeartbeatIncludesV7ByDefault(t *testing.T) {
	t.Setenv("RYV_NODE_V7_CAPS", "")
	client, gotV7, calls := heartbeatTestClient(t)

	ok := sendHeartbeat(context.Background(), client, validHeartbeatCaps(), "cpu", "ca", nil, nil)
	if !ok {
		t.Fatal("sendHeartbeat() = false, want true")
	}
	if calls.Load() != 1 {
		t.Fatalf("heartbeat calls = %d, want 1", calls.Load())
	}
	if !gotV7.Load() {
		t.Fatal("heartbeat did not send V7 payload by default")
	}
}

func TestSendHeartbeatFallsBackToLegacyWhenV7BuildFails(t *testing.T) {
	t.Setenv("RYV_NODE_V7_CAPS", "1")
	client, gotV7, calls := heartbeatTestClient(t)

	ok := sendHeartbeat(context.Background(), client, hw.CapSet{}, "cpu", "ca", nil, nil)
	if !ok {
		t.Fatal("sendHeartbeat() = false, want true")
	}
	if calls.Load() != 1 {
		t.Fatalf("heartbeat calls = %d, want 1", calls.Load())
	}
	if gotV7.Load() {
		t.Fatal("heartbeat should fall back to legacy metrics when V7 payload build fails")
	}
}

func TestSubmitReceiptWithRetry_ExhaustsRetries(t *testing.T) {
	fc := &fakeClient{failCount: 10} // always fails
	receipt := hub.Receipt{JobID: "test-job-2", ResultHashHex: "def456", MeteringUnits: 1}

	err := submitReceiptWithRetryTestable(context.Background(), fc, receipt)
	if err == nil {
		t.Fatal("expected error after exhausting retries")
	}
	if got := int(fc.calls.Load()); got != 5 {
		t.Fatalf("expected exactly 5 attempts, got %d", got)
	}
}

func heartbeatTestClient(t *testing.T) (*hub.Client, *atomic.Bool, *atomic.Int32) {
	t.Helper()
	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	gotV7 := &atomic.Bool{}
	calls := &atomic.Int32{}
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/node/heartbeat" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		calls.Add(1)
		var req struct {
			V7 json.RawMessage `json:"v7"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		gotV7.Store(len(req.V7) > 0)
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(ts.Close)
	return hub.New(ts.URL, pub, priv), gotV7, calls
}

func validHeartbeatCaps() hw.CapSet {
	return hw.CapSet{
		CPUCores: 4,
		RAMBytes: 8 * 1024 * 1024 * 1024,
	}
}

func TestSubmitReceiptWithRetry_RespectsContextCancel(t *testing.T) {
	fc := &fakeClient{failCount: 10}
	receipt := hub.Receipt{JobID: "test-job-3", ResultHashHex: "ghi789", MeteringUnits: 1}

	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()

	start := time.Now()
	err := submitReceiptWithRetryTestable(ctx, fc, receipt)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected error on cancelled context")
	}
	// Should bail out within ~1 second, not wait for all 5 retries
	if elapsed > 3*time.Second {
		t.Fatalf("took too long (%v), context cancellation not respected", elapsed)
	}
}

func TestJobActiveFlag_PreventsUpdate(t *testing.T) {
	// Verify the atomic flag mechanism works
	jobActive.Store(0)
	if jobActive.Load() != 0 {
		t.Fatal("expected jobActive=0 initially")
	}
	jobActive.Store(1)
	if jobActive.Load() != 1 {
		t.Fatal("expected jobActive=1 after store")
	}
	jobActive.Store(0)
	if jobActive.Load() != 0 {
		t.Fatal("expected jobActive=0 after reset")
	}
}

func TestParseModelBenchSelfTestFlags(t *testing.T) {
	config, jsonOutput, err := parseModelBenchSelfTestFlags([]string{
		"--model", "tinyllama",
		"--tokens", "8",
		"--timeout", "1500ms",
		"--json",
	})
	if err != nil {
		t.Fatalf("parseModelBenchSelfTestFlags() error = %v", err)
	}
	if !jsonOutput {
		t.Fatal("json output = false, want true")
	}
	if config.ModelID != "tinyllama" {
		t.Fatalf("model_id = %q, want tinyllama", config.ModelID)
	}
	if config.MaxTokens != 8 {
		t.Fatalf("max_tokens = %d, want 8", config.MaxTokens)
	}
	if config.TimeoutMs != 1500 {
		t.Fatalf("timeout_ms = %d, want 1500", config.TimeoutMs)
	}
}

func TestParseModelBenchTimeoutAcceptsRawMilliseconds(t *testing.T) {
	got, err := parseModelBenchTimeoutMs("60000")
	if err != nil {
		t.Fatalf("parseModelBenchTimeoutMs() error = %v", err)
	}
	if got != 60_000 {
		t.Fatalf("timeout_ms = %d, want 60000", got)
	}
}

func TestParseLlamaCppBenchFlags(t *testing.T) {
	config, jsonOutput, err := parseLlamaCppBenchFlags([]string{
		"--json",
		"--model", "tinyllama.Q4_K_M.gguf",
		"--max-tokens", "16",
		"--runs", "5",
		"--warmup-runs", "2",
		"--timeout", "1500ms",
	})
	if err != nil {
		t.Fatalf("parseLlamaCppBenchFlags() error = %v", err)
	}
	if !jsonOutput {
		t.Fatal("json output = false, want true")
	}
	if config.ModelID != "tinyllama.Q4_K_M.gguf" {
		t.Fatalf("model_id = %q, want tinyllama.Q4_K_M.gguf", config.ModelID)
	}
	if config.MaxTokens != 16 || config.MeasuredRuns != 5 || config.WarmupRuns != 2 || config.TimeoutMs != 1500 {
		t.Fatalf("config = %+v, want max_tokens/runs/warmups/timeout overrides", config)
	}
	if !config.Streaming {
		t.Fatalf("streaming = false, want default true")
	}
}

func TestParseLlamaCppBenchTimeoutAcceptsRawMilliseconds(t *testing.T) {
	got, err := parseLlamaCppBenchTimeoutMs("60000")
	if err != nil {
		t.Fatalf("parseLlamaCppBenchTimeoutMs() error = %v", err)
	}
	if got != 60_000 {
		t.Fatalf("timeout_ms = %d, want 60000", got)
	}
}

func TestDetectManagedOCIBackendWithProbesWithoutDaemonRejectsCPUContainerWork(t *testing.T) {
	cli, ready, gpu := detectManagedOCIBackendWithProbes(false,
		func() string { return "/usr/bin/docker" },
		func(string) bool { return false },
		func(string) bool {
			t.Fatal("gpu probe should not run when daemon is unavailable")
			return false
		},
	)

	if !cli {
		t.Fatal("expected OCI backend CLI to be detected")
	}
	if ready {
		t.Fatal("expected managed OCI backend to be unavailable")
	}
	if gpu {
		t.Fatal("expected managed OCI GPU check to be false")
	}
}

func TestDetectManagedOCIBackendWithProbesReportsGPUReadiness(t *testing.T) {
	cli, ready, gpu := detectManagedOCIBackendWithProbes(true,
		func() string { return "/usr/bin/docker" },
		func(string) bool { return true },
		func(bin string) bool {
			if bin != "/usr/bin/docker" {
				t.Fatalf("unexpected OCI backend bin %q", bin)
			}
			return true
		},
	)

	if !cli || !ready || !gpu {
		t.Fatalf("expected managed OCI backend and GPU to be ready, got cli=%v ready=%v gpu=%v", cli, ready, gpu)
	}
}

func TestManagedOCIRuntimeUnavailableErrorMatchesNamedPipeFailure(t *testing.T) {
	err := fmt.Errorf("docker run failed")
	logs := "failed to connect to the docker API at npipe:////./pipe/oci_linux_adapter; check if the daemon is running"
	if !managedOCIRuntimeUnavailableError(err, logs) {
		t.Fatal("expected named pipe OCI backend failure to be detected")
	}
}

func TestManagedOCIRuntimeUnavailableErrorIgnoresRegularContainerFailures(t *testing.T) {
	err := fmt.Errorf("exit status 1")
	logs := "Traceback: model weights missing"
	if managedOCIRuntimeUnavailableError(err, logs) {
		t.Fatal("expected ordinary container error to not be treated as OCI backend failure")
	}
}

func TestPublicAIOptInEnabled(t *testing.T) {
	prevResolver := operatorConfigPathResolver
	configPath := filepath.Join(t.TempDir(), "config.json")
	operatorConfigPathResolver = func() (string, error) {
		return configPath, nil
	}
	defer func() {
		operatorConfigPathResolver = prevResolver
	}()

	prevState := operatorRuntimeState
	operatorRuntimeState = nil
	defer func() {
		operatorRuntimeState = prevState
	}()

	t.Setenv("RYV_PUBLIC_AI", "")
	if !publicAIOptInEnabled() {
		t.Fatal("expected legacy missing public AI preference to preserve default-on participation")
	}

	t.Setenv("RYV_PUBLIC_AI", "1")
	if !publicAIOptInEnabled() {
		t.Fatal("expected public AI opt-in to enable on 1")
	}

	t.Setenv("RYV_PUBLIC_AI", "true")
	if !publicAIOptInEnabled() {
		t.Fatal("expected public AI opt-in to enable on true")
	}

	t.Setenv("RYV_PUBLIC_AI", "no")
	if publicAIOptInEnabled() {
		t.Fatal("expected public AI opt-in to disable on no")
	}
}

func TestResolveInitialPublicAIOptInUsesSavedPreferences(t *testing.T) {
	prevResolver := operatorConfigPathResolver
	configPath := filepath.Join(t.TempDir(), "config.json")
	operatorConfigPathResolver = func() (string, error) {
		return configPath, nil
	}
	defer func() {
		operatorConfigPathResolver = prevResolver
	}()

	t.Setenv("RYV_PUBLIC_AI", "")
	if err := saveOperatorPreferences(operatorPreferences{PublicAIOptIn: true}); err != nil {
		t.Fatalf("saveOperatorPreferences() error = %v", err)
	}

	got, err := resolveInitialPublicAIOptIn()
	if err != nil {
		t.Fatalf("resolveInitialPublicAIOptIn() error = %v", err)
	}
	if !got {
		t.Fatal("expected saved operator preference to enable public AI opt-in")
	}
}

func TestResolveInitialPublicAIOptInTreatsLegacySavedFalseAsDefaultOn(t *testing.T) {
	prevResolver := operatorConfigPathResolver
	configPath := filepath.Join(t.TempDir(), "config.json")
	operatorConfigPathResolver = func() (string, error) {
		return configPath, nil
	}
	defer func() {
		operatorConfigPathResolver = prevResolver
	}()

	t.Setenv("RYV_PUBLIC_AI", "")
	if err := saveOperatorPreferences(operatorPreferences{PublicAIOptIn: false}); err != nil {
		t.Fatalf("saveOperatorPreferences() error = %v", err)
	}

	got, err := resolveInitialPublicAIOptIn()
	if err != nil {
		t.Fatalf("resolveInitialPublicAIOptIn() error = %v", err)
	}
	if !got {
		t.Fatal("expected legacy saved false without opt-out marker to default public AI on")
	}
}

func TestResolveInitialPublicAIOptInHonorsExplicitOptOut(t *testing.T) {
	prevResolver := operatorConfigPathResolver
	configPath := filepath.Join(t.TempDir(), "config.json")
	operatorConfigPathResolver = func() (string, error) {
		return configPath, nil
	}
	defer func() {
		operatorConfigPathResolver = prevResolver
	}()

	t.Setenv("RYV_PUBLIC_AI", "")
	if err := saveOperatorPreferences(operatorPreferences{PublicAIOptIn: false, PublicAIOptOut: true}); err != nil {
		t.Fatalf("saveOperatorPreferences() error = %v", err)
	}

	got, err := resolveInitialPublicAIOptIn()
	if err != nil {
		t.Fatalf("resolveInitialPublicAIOptIn() error = %v", err)
	}
	if got {
		t.Fatal("expected explicit public AI opt-out marker to keep public AI disabled")
	}
}

func TestResolveInitialPublicAIOptInPrefersEnvOverride(t *testing.T) {
	prevResolver := operatorConfigPathResolver
	configPath := filepath.Join(t.TempDir(), "config.json")
	operatorConfigPathResolver = func() (string, error) {
		return configPath, nil
	}
	defer func() {
		operatorConfigPathResolver = prevResolver
	}()

	if err := saveOperatorPreferences(operatorPreferences{PublicAIOptIn: false}); err != nil {
		t.Fatalf("saveOperatorPreferences() error = %v", err)
	}

	t.Setenv("RYV_PUBLIC_AI", "1")
	got, err := resolveInitialPublicAIOptIn()
	if err != nil {
		t.Fatalf("resolveInitialPublicAIOptIn() error = %v", err)
	}
	if !got {
		t.Fatal("expected env override to take precedence over saved preferences")
	}
}

func TestResolveInitialDeclaredCountryUsesSavedPreferences(t *testing.T) {
	prevResolver := operatorConfigPathResolver
	configPath := filepath.Join(t.TempDir(), "config.json")
	operatorConfigPathResolver = func() (string, error) {
		return configPath, nil
	}
	defer func() {
		operatorConfigPathResolver = prevResolver
	}()

	t.Setenv("RYV_DECLARED_COUNTRY", "")
	if err := saveOperatorPreferences(operatorPreferences{DeclaredCountry: "ca"}); err != nil {
		t.Fatalf("saveOperatorPreferences() error = %v", err)
	}

	got, err := resolveInitialDeclaredCountry("")
	if err != nil {
		t.Fatalf("resolveInitialDeclaredCountry() error = %v", err)
	}
	if got != "CA" {
		t.Fatalf("declared country = %q, want %q", got, "CA")
	}
}

func TestResolveInitialDeclaredCountryPrefersEnvOverride(t *testing.T) {
	prevResolver := operatorConfigPathResolver
	configPath := filepath.Join(t.TempDir(), "config.json")
	operatorConfigPathResolver = func() (string, error) {
		return configPath, nil
	}
	defer func() {
		operatorConfigPathResolver = prevResolver
	}()

	if err := saveOperatorPreferences(operatorPreferences{DeclaredCountry: "CA"}); err != nil {
		t.Fatalf("saveOperatorPreferences() error = %v", err)
	}

	t.Setenv("RYV_DECLARED_COUNTRY", "de")
	got, err := resolveInitialDeclaredCountry("")
	if err != nil {
		t.Fatalf("resolveInitialDeclaredCountry() error = %v", err)
	}
	if got != "DE" {
		t.Fatalf("declared country = %q, want %q", got, "DE")
	}
}

func TestResolveRuntimeContractMetadataUsesSavedPreferences(t *testing.T) {
	prevResolver := operatorConfigPathResolver
	configPath := filepath.Join(t.TempDir(), "config.json")
	operatorConfigPathResolver = func() (string, error) {
		return configPath, nil
	}
	defer func() {
		operatorConfigPathResolver = prevResolver
	}()

	t.Setenv("RYV_RUNTIME_CHANNEL", "")
	t.Setenv("RYV_RUNTIME_CHANNEL_VERSION", "")
	t.Setenv("RYV_RUNTIME_PROVIDER", "")
	t.Setenv("RYV_RUNTIME_MODE", "")
	t.Setenv("RYV_RUNTIME_SOURCE", "")
	t.Setenv("RYV_RUNTIME_ARTIFACT", "")
	t.Setenv("RYV_RUNTIME_BACKEND_BINARY", "")
	t.Setenv("RYV_RUNTIME_ENGINE_BINARY", "")
	t.Setenv("RYV_RUNTIME_ENGINE_KIND", "")
	t.Setenv("RYV_RUNTIME_MANIFEST_HASH", "")

	want := operatorPreferences{
		PublicAIOptIn:         true,
		RuntimeChannel:        "managed_oci_v1",
		RuntimeChannelVersion: "2026.04.14",
		RuntimeProvider:       "oci_linux_adapter",
		RuntimeMode:           "host_package",
		RuntimeSource:         "ryvion_runtime_kit",
		RuntimeArtifact:       "ryvion-runtime-kit-linux-amd64-2026.04.14.tar.gz",
		RuntimeBackendBinary:  "/opt/ryvion/runtime/backend/ryvion-oci",
		RuntimeEngineBinary:   "/usr/bin/podman",
		RuntimeEngineKind:     "podman",
		RuntimeManifestHash:   "abc123",
	}
	if err := saveOperatorPreferences(want); err != nil {
		t.Fatalf("saveOperatorPreferences() error = %v", err)
	}

	got, err := resolveRuntimeContractMetadata("dev")
	if err != nil {
		t.Fatalf("resolveRuntimeContractMetadata() error = %v", err)
	}
	if got.Channel != want.RuntimeChannel || got.Version != want.RuntimeChannelVersion || got.Provider != want.RuntimeProvider || got.Mode != want.RuntimeMode || got.Source != want.RuntimeSource || got.Artifact != want.RuntimeArtifact || got.Backend != want.RuntimeBackendBinary || got.Engine != want.RuntimeEngineBinary || got.EngineKind != want.RuntimeEngineKind || got.ManifestHash != want.RuntimeManifestHash {
		t.Fatalf("unexpected runtime metadata: %+v", got)
	}
}

func TestResolveRuntimeContractMetadataPrefersEnvOverride(t *testing.T) {
	prevResolver := operatorConfigPathResolver
	configPath := filepath.Join(t.TempDir(), "config.json")
	operatorConfigPathResolver = func() (string, error) {
		return configPath, nil
	}
	defer func() {
		operatorConfigPathResolver = prevResolver
	}()

	if err := saveOperatorPreferences(operatorPreferences{
		RuntimeChannel:        "managed_oci_v1",
		RuntimeChannelVersion: "2026.04.14",
		RuntimeProvider:       "oci_linux_adapter",
		RuntimeMode:           "host_package",
		RuntimeSource:         "ryvion_runtime_kit",
		RuntimeArtifact:       "ryvion-runtime-kit-linux-amd64-2026.04.14.tar.gz",
		RuntimeBackendBinary:  "/opt/ryvion/runtime/backend/ryvion-oci",
		RuntimeEngineBinary:   "/usr/bin/podman",
		RuntimeEngineKind:     "podman",
		RuntimeManifestHash:   "abc123",
	}); err != nil {
		t.Fatalf("saveOperatorPreferences() error = %v", err)
	}

	t.Setenv("RYV_RUNTIME_PROVIDER", "oci_desktop_adapter")
	t.Setenv("RYV_RUNTIME_MODE", "desktop")
	t.Setenv("RYV_RUNTIME_SOURCE", "signed_release_channel")
	t.Setenv("RYV_RUNTIME_ARTIFACT", "ryvion-runtime-kit-windows-amd64-2026.04.14.zip")
	t.Setenv("RYV_RUNTIME_BACKEND_BINARY", `C:\Program Files\Ryvion\runtime\backend\ryvion-oci.cmd`)
	t.Setenv("RYV_RUNTIME_ENGINE_BINARY", `C:\Program Files\RedHat\Podman\podman.exe`)
	t.Setenv("RYV_RUNTIME_ENGINE_KIND", "podman")
	got, err := resolveRuntimeContractMetadata("dev")
	if err != nil {
		t.Fatalf("resolveRuntimeContractMetadata() error = %v", err)
	}
	if got.Provider != "oci_desktop_adapter" || got.Mode != "desktop" || got.Source != "signed_release_channel" || got.Artifact != "ryvion-runtime-kit-windows-amd64-2026.04.14.zip" {
		t.Fatalf("expected env override to win, got %+v", got)
	}
	if got.Backend != `C:\Program Files\Ryvion\runtime\backend\ryvion-oci.cmd` || got.Engine != `C:\Program Files\RedHat\Podman\podman.exe` || got.EngineKind != "podman" {
		t.Fatalf("expected env override to win, got %+v", got)
	}
}

func TestRuntimeManagerPrefersManagedRuntimeWrapperStatus(t *testing.T) {
	prevProbe := probeManagedRuntimeStatus
	probeManagedRuntimeStatus = func(_ context.Context, _ string, _ func(string) string, _ string) (runtimeexec.Status, bool) {
		return runtimeexec.Status{
			BinaryPath:   "/opt/ryvion/runtime/ryvion-runtime",
			BackendPath:  "/opt/ryvion/runtime/backend/ryvion-oci",
			EnginePath:   "/usr/bin/podman",
			EngineKind:   "podman",
			CLIInstalled: true,
			Ready:        true,
			GPUReady:     true,
			Health:       "ready",
		}, true
	}
	defer func() {
		probeManagedRuntimeStatus = prevProbe
	}()

	runtimeMgr := newRuntimeManager("dev", runtimeContractMetadata{
		Channel:      "managed_oci_v1",
		Version:      "2026.04.14.1",
		Provider:     "oci_linux_adapter",
		Mode:         "host_package",
		Source:       "ryvion_runtime_kit",
		Artifact:     "ryvion-runtime-kit-linux-amd64-2026.04.14.1.tar.gz",
		Binary:       "/opt/ryvion/runtime/ryvion-runtime",
		Backend:      "/opt/ryvion/runtime/backend/ryvion-oci",
		Engine:       "/usr/bin/podman",
		EngineKind:   "podman",
		ManifestHash: "abc123",
	})

	snap := runtimeMgr.Snapshot(true)
	if !snap.Ready || !snap.GPUReady || !snap.CLIInstalled {
		t.Fatalf("expected wrapper snapshot to be ready, got %+v", snap)
	}
	if snap.Binary != "/opt/ryvion/runtime/ryvion-runtime" || snap.Backend != "/opt/ryvion/runtime/backend/ryvion-oci" || snap.Engine != "/usr/bin/podman" || snap.EngineKind != "podman" {
		t.Fatalf("unexpected wrapper paths: %+v", snap)
	}

	tokens := runtimeMgr.StatusTokens(true)
	if !containsToken(tokens, "runtime-binary:/opt/ryvion/runtime/ryvion-runtime") {
		t.Fatalf("expected runtime binary token, got %v", tokens)
	}
	if !containsToken(tokens, "runtime-backend:/opt/ryvion/runtime/backend/ryvion-oci") {
		t.Fatalf("expected runtime backend token, got %v", tokens)
	}
	if !containsToken(tokens, "runtime-engine:/usr/bin/podman") {
		t.Fatalf("expected runtime engine token, got %v", tokens)
	}
	if !containsToken(tokens, "runtime-engine-kind:podman") {
		t.Fatalf("expected runtime engine kind token, got %v", tokens)
	}
}

func TestResolveInitialPublicAIOptInAutoEnablesWhenOCIDisabled(t *testing.T) {
	prevResolver := operatorConfigPathResolver
	configPath := filepath.Join(t.TempDir(), "config.json")
	operatorConfigPathResolver = func() (string, error) { return configPath, nil }
	defer func() { operatorConfigPathResolver = prevResolver }()

	t.Setenv("RYV_PUBLIC_AI", "")
	t.Setenv("RYV_DISABLE_OCI", "1")

	got, err := resolveInitialPublicAIOptIn()
	if err != nil {
		t.Fatalf("resolveInitialPublicAIOptIn() error = %v", err)
	}
	if !got {
		t.Fatal("expected disabled OCI lane to auto-opt the node into public AI streaming work")
	}
}

func TestOCILaneDisabledSkipsManagedRuntimeProbe(t *testing.T) {
	t.Setenv("RYV_DISABLE_OCI", "1")

	called := false
	prevProbe := probeManagedRuntimeStatus
	probeManagedRuntimeStatus = func(_ context.Context, _ string, _ func(string) string, _ string) (runtimeexec.Status, bool) {
		called = true
		return runtimeexec.Status{CLIInstalled: true, Ready: true, GPUReady: true, Health: "ready"}, true
	}
	defer func() { probeManagedRuntimeStatus = prevProbe }()

	runtimeMgr := newRuntimeManager("dev", runtimeContractMetadata{
		Channel: "managed_oci_v1",
		Version: "2026.04.16.10",
	})
	snap := runtimeMgr.Snapshot(true)
	if called {
		t.Fatal("managed runtime probe should be skipped when RYV_DISABLE_OCI=1")
	}
	if snap.Ready || snap.GPUReady || snap.CLIInstalled {
		t.Fatalf("expected disabled snapshot to report no managed OCI capability, got %+v", snap)
	}
	if snap.Health != "disabled" || snap.Mode != "native_only" || snap.Source != "operator_opt_out" {
		t.Fatalf("expected disabled/native-only snapshot, got %+v", snap)
	}

	tokens := runtimeMgr.StatusTokens(true)
	if !containsToken(tokens, "oci-lane:disabled") {
		t.Fatalf("expected oci-lane:disabled token, got %v", tokens)
	}
	if !containsToken(tokens, "runtime-health:disabled") {
		t.Fatalf("expected runtime-health:disabled token, got %v", tokens)
	}
	if !containsToken(tokens, "cap:managed_oci_cpu:0") {
		t.Fatalf("expected cap:managed_oci_cpu:0 token, got %v", tokens)
	}
}

func TestRuntimeWarmingHeuristicWindowsPodman(t *testing.T) {
	t.Parallel()

	if !runtimeWarmingHeuristic("windows", true, false, "warming", `C:\Program Files\Ryvion\runtime\backend\ryvion-oci.cmd`, "podman") {
		t.Fatal("expected explicit warming health to be preserved")
	}
	if runtimeWarmingHeuristic("windows", true, false, "degraded", `C:\Program Files\Ryvion\runtime\backend\ryvion-oci.cmd`, "podman") {
		t.Fatal("did not expect degraded Windows runtime to be treated as warming")
	}
	if runtimeWarmingHeuristic("linux", true, false, "degraded", "/opt/ryvion/runtime/backend/ryvion-oci", "podman") {
		t.Fatal("did not expect non-Windows runtime to be treated as warming")
	}
}

func TestProcessOptionalV7MemoryBenchmarkRecordsLocalLifecycle(t *testing.T) {
	oldState := operatorRuntimeState
	oldDiagnostics := workLoopDiagnostics
	status := v7memorybench.NewLocalStatus()
	operatorRuntimeState = &operatorRuntime{v7MemoryBenchmark: status}
	workLoopDiagnostics = diagnostics.NewWorkLoopDiagnostics()
	t.Cleanup(func() {
		operatorRuntimeState = oldState
		workLoopDiagnostics = oldDiagnostics
	})
	t.Setenv(v7memorybench.BenchmarkFlagEnv, "1")

	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	var receiptCalls atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/node/receipt" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		receiptCalls.Add(1)
		var req struct {
			JobID    string         `json:"job_id"`
			Metadata map[string]any `json:"metadata"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode receipt: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if req.JobID != "job-bench-local" {
			t.Errorf("receipt job_id = %q", req.JobID)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if _, ok := req.Metadata[v7memorybench.BenchmarkTask]; !ok {
			t.Errorf("receipt metadata missing %q: %+v", v7memorybench.BenchmarkTask, req.Metadata)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	spec := map[string]any{
		"task":               v7memorybench.BenchmarkTask,
		"request_id":         "request-bench-local",
		"job_id":             "job-bench-local",
		"shard_id":           "shard-a",
		"seed":               int64(7),
		"token_count":        4,
		"value_dim":          2,
		"created_at_unix_ms": int64(1_800_000_000_123),
		"prompt":             "raw prompt must not leak",
		"output":             "raw output must not leak",
		"weighted_value":     []float64{1, 2, 3},
	}
	specJSON, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	handled, result, err := processOptionalV7MemoryBenchmark(context.Background(), hub.New(ts.URL, pub, priv), &hub.WorkAssignment{
		JobID:    "job-bench-local",
		Kind:     "benchmark",
		SpecJSON: string(specJSON),
	}, nil, false)
	if err != nil {
		t.Fatalf("processOptionalV7MemoryBenchmark() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if result == nil || result.ResultHashHex == "" {
		t.Fatalf("result = %+v, want hash", result)
	}
	if got := receiptCalls.Load(); got != 1 {
		t.Fatalf("receipt calls = %d, want 1", got)
	}

	snapshot := status.Snapshot()
	if snapshot.LastSeenBenchmarkJobID != "job-bench-local" || snapshot.LastSeenRequestID != "request-bench-local" {
		t.Fatalf("seen status = %+v", snapshot)
	}
	if snapshot.LastExecutedJobID != "job-bench-local" {
		t.Fatalf("last executed job id = %q", snapshot.LastExecutedJobID)
	}
	if snapshot.LastReceiptSubmittedJobID != "job-bench-local" {
		t.Fatalf("last receipt submitted job id = %q", snapshot.LastReceiptSubmittedJobID)
	}
	if snapshot.Counters.Seen != 1 || snapshot.Counters.Executed != 1 || snapshot.Counters.ReceiptSubmitted != 1 {
		t.Fatalf("unexpected counters: %+v", snapshot.Counters)
	}
	if snapshot.Counters.ReceiptFailed != 0 || snapshot.Counters.Rejected != 0 {
		t.Fatalf("unexpected failure counters: %+v", snapshot.Counters)
	}

	workSnapshot := workLoopDiagnostics.Snapshot()
	if got := countMainWorkLoopEvents(workSnapshot.RecentEvents, "receipt_build_start"); got != 1 {
		t.Fatalf("receipt_build_start events = %d, want 1: %+v", got, workSnapshot.RecentEvents)
	}
	if got := countMainWorkLoopEvents(workSnapshot.RecentEvents, "receipt_build_end"); got != 1 {
		t.Fatalf("receipt_build_end events = %d, want 1: %+v", got, workSnapshot.RecentEvents)
	}
	if got := countMainWorkLoopEvents(workSnapshot.RecentEvents, "receipt_metadata_start"); got != 1 {
		t.Fatalf("receipt_metadata_start events = %d, want 1: %+v", got, workSnapshot.RecentEvents)
	}
	if workSnapshot.LastReceiptBuildMs != workSnapshot.LastReceiptTotalBuildMs || workSnapshot.LastReceiptTotalBuildUs <= 0 {
		t.Fatalf("receipt build status did not use actual build timings: %+v", workSnapshot)
	}
	if workSnapshot.LastReceiptMetadataGapUs < 0 {
		t.Fatalf("metadata gap = %d us, want non-negative", workSnapshot.LastReceiptMetadataGapUs)
	}
	if workSnapshot.LastReceiptReadyAt == "" || workSnapshot.LastReceiptReadyToSubmitUs < 0 || workSnapshot.LastReceiptSubmitQueueGapUs != workSnapshot.LastReceiptReadyToSubmitUs {
		t.Fatalf("ready-to-submit gap not recorded separately: %+v", workSnapshot)
	}
	if workSnapshot.LastReceiptReadyToSubmitUs > 100_000 {
		t.Fatalf("ready-to-submit gap = %d us, want near-zero fast-path submit", workSnapshot.LastReceiptReadyToSubmitUs)
	}
	for _, name := range []string{"receipt_ready", "receipt_ready_to_submit_gap", "receipt_submit_start", "receipt_submit_end"} {
		if got := countMainWorkLoopEvents(workSnapshot.RecentEvents, name); got != 1 {
			t.Fatalf("%s events = %d, want 1: %+v", name, got, workSnapshot.RecentEvents)
		}
	}
	for _, name := range []string{"v7_fast_path_start", "v7_fast_path_receipt_ready", "v7_fast_path_submit_start", "v7_fast_path_submit_end", "pre_submit_block_start", "pre_submit_block_end"} {
		if got := countMainWorkLoopEvents(workSnapshot.RecentEvents, name); got != 1 {
			t.Fatalf("%s events = %d, want 1: %+v", name, got, workSnapshot.RecentEvents)
		}
	}
	if !mainWorkLoopEventOrder(workSnapshot.RecentEvents, "receipt_build_end", "receipt_ready", "receipt_ready_to_submit_gap", "receipt_submit_start", "receipt_submit_end") {
		t.Fatalf("receipt event timeline is out of order: %+v", workSnapshot.RecentEvents)
	}
	if !mainWorkLoopEventOrder(workSnapshot.RecentEvents, "v7_fast_path_receipt_ready", "receipt_ready", "v7_fast_path_submit_start", "receipt_ready_to_submit_gap", "receipt_submit_start", "receipt_submit_end", "v7_fast_path_submit_end") {
		t.Fatalf("fast-path receipt event timeline is out of order: %+v", workSnapshot.RecentEvents)
	}
	assertMainWorkLoopEventsChronological(t, workSnapshot.RecentEvents)
	gapEvent := findMainWorkLoopEvent(workSnapshot.RecentEvents, "receipt_ready_to_submit_gap")
	if gapEvent.SafeContext["gap_us"] == "" || gapEvent.SafeContext["job_id"] != "job-bench-local" || gapEvent.SafeContext["kind"] != "benchmark" || gapEvent.SafeContext["spec_task"] != v7memorybench.BenchmarkTask {
		t.Fatalf("gap event safe context missing expected fields: %+v", gapEvent.SafeContext)
	}
	encodedWorkSnapshot, err := json.Marshal(workSnapshot)
	if err != nil {
		t.Fatalf("json.Marshal(workSnapshot) error = %v", err)
	}
	for _, forbidden := range []string{"raw prompt must not leak", "raw output must not leak", `"weighted_value":[`} {
		if strings.Contains(string(encodedWorkSnapshot), forbidden) {
			t.Fatalf("work loop diagnostics leaked forbidden material %q: %s", forbidden, encodedWorkSnapshot)
		}
	}
}

func TestProcessOptionalV7MemoryBenchmarkFastPathSkipsRuntimeProbe(t *testing.T) {
	oldState := operatorRuntimeState
	oldDiagnostics := workLoopDiagnostics
	oldProbe := probeManagedRuntimeStatus
	status := v7memorybench.NewLocalStatus()
	operatorRuntimeState = &operatorRuntime{v7MemoryBenchmark: status}
	workLoopDiagnostics = diagnostics.NewWorkLoopDiagnostics()
	var probeCalls atomic.Int32
	probeManagedRuntimeStatus = func(_ context.Context, _ string, _ func(string) string, _ string) (runtimeexec.Status, bool) {
		probeCalls.Add(1)
		time.Sleep(200 * time.Millisecond)
		return runtimeexec.Status{
			BinaryPath:   "slow-wrapper",
			BackendPath:  "slow-backend",
			EnginePath:   "slow-engine",
			EngineKind:   "podman",
			CLIInstalled: true,
			Ready:        true,
			GPUReady:     true,
			Health:       "ready",
		}, true
	}
	t.Cleanup(func() {
		operatorRuntimeState = oldState
		workLoopDiagnostics = oldDiagnostics
		probeManagedRuntimeStatus = oldProbe
	})
	t.Setenv(v7memorybench.BenchmarkFlagEnv, "1")

	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	var receiptCalls atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receiptCalls.Add(1)
		var req struct {
			JobID    string         `json:"job_id"`
			Metadata map[string]any `json:"metadata"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode receipt: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if req.JobID != "job-bench-fast" {
			t.Errorf("receipt job_id = %q", req.JobID)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if _, ok := req.Metadata[v7memorybench.BenchmarkTask]; !ok {
			t.Errorf("receipt metadata missing %q: %+v", v7memorybench.BenchmarkTask, req.Metadata)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if req.Metadata["runtime_version"] != "2026.05.test" || req.Metadata["runtime_binary"] != "slow-wrapper" || req.Metadata["runtime_engine_kind"] != "podman" {
			t.Errorf("runtime metadata = %+v", req.Metadata)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	runtimeMgr := newRuntimeManager("dev", runtimeContractMetadata{
		Version:      "2026.05.test",
		Binary:       "slow-wrapper",
		Backend:      "slow-backend",
		Engine:       "slow-engine",
		EngineKind:   "podman",
		ManifestHash: "manifest-fast",
	})
	handled, result, err := processOptionalV7MemoryBenchmark(context.Background(), hub.New(ts.URL, pub, priv), &hub.WorkAssignment{
		JobID:    "job-bench-fast",
		Kind:     "benchmark",
		SpecJSON: testV7MemoryBenchmarkSpecJSON(t, "job-bench-fast", "request-bench-fast"),
	}, runtimeMgr, true)
	if err != nil {
		t.Fatalf("processOptionalV7MemoryBenchmark() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if result == nil || result.ResultHashHex == "" {
		t.Fatalf("result = %+v, want hash", result)
	}
	if got := receiptCalls.Load(); got != 1 {
		t.Fatalf("receipt calls = %d, want 1", got)
	}
	if got := probeCalls.Load(); got != 0 {
		t.Fatalf("managed runtime probe calls = %d, want 0 on V7 benchmark fast path", got)
	}
	workSnapshot := workLoopDiagnostics.Snapshot()
	if workSnapshot.LastReceiptReadyToSubmitUs < 0 || workSnapshot.LastReceiptReadyToSubmitUs > 100_000 {
		t.Fatalf("ready-to-submit gap = %d us, want near-zero fast-path submit", workSnapshot.LastReceiptReadyToSubmitUs)
	}
	if workSnapshot.ReceiptSubmittedCount != 1 || workSnapshot.ReceiptFailedCount != 0 {
		t.Fatalf("unexpected receipt counters: %+v", workSnapshot)
	}
	for _, name := range []string{"v7_fast_path_start", "v7_fast_path_receipt_ready", "v7_fast_path_submit_start", "receipt_submit_start", "receipt_submit_end", "v7_fast_path_submit_end"} {
		if got := countMainWorkLoopEvents(workSnapshot.RecentEvents, name); got != 1 {
			t.Fatalf("%s events = %d, want 1: %+v", name, got, workSnapshot.RecentEvents)
		}
	}
	if !mainWorkLoopEventOrder(workSnapshot.RecentEvents, "v7_fast_path_receipt_ready", "receipt_ready", "v7_fast_path_submit_start", "receipt_ready_to_submit_gap", "receipt_submit_start", "receipt_submit_end", "v7_fast_path_submit_end") {
		t.Fatalf("fast-path receipt event timeline is out of order: %+v", workSnapshot.RecentEvents)
	}
}

func TestProcessOptionalV7MemoryBenchmarkSubmitFailureRecordsDiagnostics(t *testing.T) {
	oldState := operatorRuntimeState
	oldDiagnostics := workLoopDiagnostics
	status := v7memorybench.NewLocalStatus()
	operatorRuntimeState = &operatorRuntime{v7MemoryBenchmark: status}
	workLoopDiagnostics = diagnostics.NewWorkLoopDiagnostics()
	t.Cleanup(func() {
		operatorRuntimeState = oldState
		workLoopDiagnostics = oldDiagnostics
	})
	t.Setenv(v7memorybench.BenchmarkFlagEnv, "1")

	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	var receiptCalls atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receiptCalls.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer ts.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	handled, result, err := processOptionalV7MemoryBenchmark(ctx, hub.New(ts.URL, pub, priv), &hub.WorkAssignment{
		JobID:    "job-bench-submit-fails",
		Kind:     "benchmark",
		SpecJSON: testV7MemoryBenchmarkSpecJSON(t, "job-bench-submit-fails", "request-bench-submit-fails"),
	}, nil, false)
	if err == nil {
		t.Fatal("processOptionalV7MemoryBenchmark() error = nil, want submit error")
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if result == nil || result.ResultHashHex == "" {
		t.Fatalf("result = %+v, want built receipt snapshot despite submit error", result)
	}
	if got := receiptCalls.Load(); got != 1 {
		t.Fatalf("receipt calls = %d, want 1 before context stops retry backoff", got)
	}

	statusSnapshot := status.Snapshot()
	if statusSnapshot.Counters.Seen != 1 || statusSnapshot.Counters.Executed != 1 || statusSnapshot.Counters.ReceiptSubmitted != 0 || statusSnapshot.Counters.ReceiptFailed != 1 {
		t.Fatalf("unexpected local status counters: %+v", statusSnapshot.Counters)
	}
	workSnapshot := workLoopDiagnostics.Snapshot()
	if workSnapshot.ReceiptSubmittedCount != 0 || workSnapshot.ReceiptFailedCount != 1 {
		t.Fatalf("unexpected work-loop receipt counters: %+v", workSnapshot)
	}
	if workSnapshot.LastReceiptSubmitError == "" {
		t.Fatalf("last receipt submit error not recorded: %+v", workSnapshot)
	}
	if workSnapshot.LastReceiptReadyToSubmitUs < 0 || workSnapshot.LastReceiptReadyToSubmitUs > 100_000 {
		t.Fatalf("ready-to-submit gap = %d us, want near-zero first submit attempt", workSnapshot.LastReceiptReadyToSubmitUs)
	}
	if got := countMainWorkLoopEvents(workSnapshot.RecentEvents, "receipt_submit_start"); got != 1 {
		t.Fatalf("receipt_submit_start events = %d, want 1 before retry cancellation: %+v", got, workSnapshot.RecentEvents)
	}
}

func TestProcessOptionalV7MemoryBenchmarkNormalJobDoesNotRecordStatus(t *testing.T) {
	oldState := operatorRuntimeState
	status := v7memorybench.NewLocalStatus()
	operatorRuntimeState = &operatorRuntime{v7MemoryBenchmark: status}
	t.Cleanup(func() {
		operatorRuntimeState = oldState
	})

	handled, result, err := processOptionalV7MemoryBenchmark(context.Background(), nil, &hub.WorkAssignment{
		JobID:    "job-normal",
		Kind:     "native_report",
		SpecJSON: `{"task":"native_report","job_id":"job-normal"}`,
	}, nil, false)
	if err != nil {
		t.Fatalf("processOptionalV7MemoryBenchmark() error = %v", err)
	}
	if handled {
		t.Fatal("handled = true, want false")
	}
	if result != nil {
		t.Fatalf("result = %+v, want nil", result)
	}
	if snapshot := status.Snapshot(); snapshot.Counters != (v7memorybench.LocalStatusCounters{}) {
		t.Fatalf("status counters changed for normal job: %+v", snapshot.Counters)
	}
}

func TestProcessOptionalV7TensorPlaneBenchmarkFlagOffDoesNotHandle(t *testing.T) {
	t.Setenv(v7tensorplane.BenchmarkFlagEnv, "")
	oldStatus := v7TensorPlaneBenchmarkStatus
	status := v7tensorplane.NewLocalStatus()
	v7TensorPlaneBenchmarkStatus = status
	t.Cleanup(func() {
		v7TensorPlaneBenchmarkStatus = oldStatus
	})

	handled, result, err := processOptionalV7TensorPlaneBenchmark(context.Background(), nil, &hub.WorkAssignment{
		JobID:    "job-tensorplane-local",
		Kind:     "benchmark",
		SpecJSON: testV7TensorPlaneBenchmarkSpecJSON(t, "job-tensorplane-local", "request-tensorplane-local"),
	}, nil, false)
	if err != nil {
		t.Fatalf("processOptionalV7TensorPlaneBenchmark() error = %v", err)
	}
	if handled {
		t.Fatal("handled = true, want false when flag is off")
	}
	if result != nil {
		t.Fatalf("result = %+v, want nil", result)
	}
	if snapshot := status.Snapshot(); snapshot.Counters != (v7tensorplane.LocalStatusCounters{}) {
		t.Fatalf("status counters changed with flag off: %+v", snapshot.Counters)
	}
}

func TestProcessOptionalV7TensorPlaneBenchmarkSubmitsMeasuredReceipt(t *testing.T) {
	t.Setenv(v7tensorplane.BenchmarkFlagEnv, "1")
	oldStatus := v7TensorPlaneBenchmarkStatus
	oldDiagnostics := workLoopDiagnostics
	status := v7tensorplane.NewLocalStatus()
	v7TensorPlaneBenchmarkStatus = status
	workLoopDiagnostics = diagnostics.NewWorkLoopDiagnostics()
	t.Cleanup(func() {
		v7TensorPlaneBenchmarkStatus = oldStatus
		workLoopDiagnostics = oldDiagnostics
	})

	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	var receiptCalls atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/node/receipt" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		receiptCalls.Add(1)
		var req struct {
			JobID         string         `json:"job_id"`
			ResultHashHex string         `json:"result_hash_hex"`
			Metadata      map[string]any `json:"metadata"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode receipt: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if req.JobID != "job-tensorplane-local" {
			t.Errorf("receipt job_id = %q", req.JobID)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if len(req.ResultHashHex) != 64 {
			t.Errorf("result_hash_hex = %q, want 64 hex chars", req.ResultHashHex)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		metadata, ok := req.Metadata[v7tensorplane.BenchmarkTask].(map[string]any)
		if !ok {
			t.Errorf("receipt metadata missing %q: %+v", v7tensorplane.BenchmarkTask, req.Metadata)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if metadata["proof_status"] != v7tensorplane.ProofStatusTensorPlaneMeasured {
			t.Errorf("proof_status = %v", metadata["proof_status"])
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if metadata["correctness_status"] != v7tensorplane.CorrectnessStatusMatched {
			t.Errorf("correctness_status = %v", metadata["correctness_status"])
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		for _, key := range []string{"page_hash", "query_hash", "summary_hash"} {
			value, ok := metadata[key].(string)
			if !ok || !strings.HasPrefix(value, "sha256:") {
				t.Errorf("%s = %#v, want sha256 hash", key, metadata[key])
				w.WriteHeader(http.StatusBadRequest)
				return
			}
		}
		encoded, _ := json.Marshal(req.Metadata)
		for _, forbidden := range []string{"key_data", "value_data", "query_vector", "prompt text", "generated output"} {
			if strings.Contains(string(encoded), forbidden) {
				t.Errorf("metadata leaked forbidden material %q: %s", forbidden, encoded)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	handled, result, err := processOptionalV7TensorPlaneBenchmark(context.Background(), hub.New(ts.URL, pub, priv), &hub.WorkAssignment{
		JobID:    "job-tensorplane-local",
		Kind:     "benchmark",
		SpecJSON: testV7TensorPlaneBenchmarkSpecJSON(t, "job-tensorplane-local", "request-tensorplane-local"),
	}, nil, true)
	if err != nil {
		t.Fatalf("processOptionalV7TensorPlaneBenchmark() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if result == nil || result.ResultHashHex == "" || result.ExitCode != 0 {
		t.Fatalf("result = %+v, want successful tensorplane receipt", result)
	}
	if got := receiptCalls.Load(); got != 1 {
		t.Fatalf("receipt calls = %d, want 1", got)
	}

	snapshot := status.Snapshot()
	if snapshot.LastJobID != "job-tensorplane-local" {
		t.Fatalf("last_job_id = %q", snapshot.LastJobID)
	}
	if snapshot.Counters.Seen != 1 || snapshot.Counters.Executed != 1 || snapshot.Counters.ReceiptSubmitted != 1 || snapshot.Counters.ReceiptFailed != 0 {
		t.Fatalf("unexpected counters: %+v", snapshot.Counters)
	}
	workSnapshot := workLoopDiagnostics.Snapshot()
	if workSnapshot.LastWorkSpecTask != v7tensorplane.BenchmarkTask {
		t.Fatalf("last work spec task = %q", workSnapshot.LastWorkSpecTask)
	}
	for _, name := range []string{"v7_fast_path_start", "v7_fast_path_receipt_ready", "v7_fast_path_submit_start", "receipt_submit_start", "receipt_submit_end", "v7_fast_path_submit_end"} {
		if got := countMainWorkLoopEvents(workSnapshot.RecentEvents, name); got != 1 {
			t.Fatalf("%s events = %d, want 1: %+v", name, got, workSnapshot.RecentEvents)
		}
	}
}

func TestProcessOptionalV7TensorPlaneBenchmarkSubmitFailureRecordsDiagnostics(t *testing.T) {
	t.Setenv(v7tensorplane.BenchmarkFlagEnv, "1")
	oldStatus := v7TensorPlaneBenchmarkStatus
	oldDiagnostics := workLoopDiagnostics
	status := v7tensorplane.NewLocalStatus()
	v7TensorPlaneBenchmarkStatus = status
	workLoopDiagnostics = diagnostics.NewWorkLoopDiagnostics()
	t.Cleanup(func() {
		v7TensorPlaneBenchmarkStatus = oldStatus
		workLoopDiagnostics = oldDiagnostics
	})

	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer ts.Close()

	handled, result, err := processOptionalV7TensorPlaneBenchmark(context.Background(), hub.New(ts.URL, pub, priv), &hub.WorkAssignment{
		JobID:    "job-tensorplane-fail",
		Kind:     "benchmark",
		SpecJSON: testV7TensorPlaneBenchmarkSpecJSON(t, "job-tensorplane-fail", "request-tensorplane-fail"),
	}, nil, false)
	if err == nil {
		t.Fatal("processOptionalV7TensorPlaneBenchmark() error = nil, want submit error")
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if result == nil || result.ResultHashHex == "" {
		t.Fatalf("result = %+v, want built receipt snapshot despite submit error", result)
	}
	snapshot := status.Snapshot()
	if snapshot.Counters.Seen != 1 || snapshot.Counters.Executed != 1 || snapshot.Counters.ReceiptSubmitted != 0 || snapshot.Counters.ReceiptFailed != 1 {
		t.Fatalf("unexpected counters: %+v", snapshot.Counters)
	}
	if snapshot.LastError == "" {
		t.Fatalf("last_error not recorded: %+v", snapshot)
	}
	workSnapshot := workLoopDiagnostics.Snapshot()
	if workSnapshot.ReceiptSubmittedCount != 0 || workSnapshot.ReceiptFailedCount != 1 {
		t.Fatalf("unexpected work-loop receipt counters: %+v", workSnapshot)
	}
}

func TestProcessOptionalV7ModelBenchmarkFlagOffDoesNotHandle(t *testing.T) {
	t.Setenv(v7modelbench.ModelBenchmarkFlagEnv, "")
	oldStatus := v7ModelBenchmarkStatus
	oldFactory := newV7ModelBenchmarkRunner
	status := v7modelbench.NewLocalStatus()
	fakeRunner := &fakeModelBenchmarkRunner{result: testMeasuredModelBenchmarkResult()}
	v7ModelBenchmarkStatus = status
	newV7ModelBenchmarkRunner = func(_ *inference.Manager, _ bool) v7modelbench.ModelBenchmarkRunner {
		return fakeRunner
	}
	t.Cleanup(func() {
		v7ModelBenchmarkStatus = oldStatus
		newV7ModelBenchmarkRunner = oldFactory
	})

	handled, result, err := processOptionalV7ModelBenchmark(context.Background(), nil, &hub.WorkAssignment{
		JobID:    "job-modelbench-local",
		Kind:     "benchmark",
		SpecJSON: testModelBenchmarkSpecJSON(t),
	}, nil, nil, false)
	if err != nil {
		t.Fatalf("processOptionalV7ModelBenchmark() error = %v", err)
	}
	if handled {
		t.Fatal("handled = true, want false when flag is off")
	}
	if result != nil {
		t.Fatalf("result = %+v, want nil", result)
	}
	if fakeRunner.calls != 0 {
		t.Fatalf("runner calls = %d, want 0", fakeRunner.calls)
	}
	if snapshot := status.Snapshot(); snapshot.Counters != (v7modelbench.LocalStatusCounters{}) {
		t.Fatalf("status counters changed with flag off: %+v", snapshot.Counters)
	}
}

func TestProcessOptionalV7LlamaCppBackendBenchmarkFlagOffDoesNotHandle(t *testing.T) {
	t.Setenv(v7llamacpp.BackendBenchmarkFlagEnv, "")
	oldStatus := v7BackendBenchmarkStatus
	oldFactory := newV7LlamaCppBackendBenchmarkRunner
	oldState := operatorRuntimeState
	status := v7llamacpp.NewBackendBenchmarkLocalStatus()
	fakeRunner := &fakeLlamaCppBackendBenchmarkRunner{snapshot: testLlamaCppBackendBenchmarkSnapshot(v7llamacpp.BenchmarkProofStatusMeasured)}
	v7BackendBenchmarkStatus = status
	operatorRuntimeState = nil
	newV7LlamaCppBackendBenchmarkRunner = func() v7llamacpp.BackendBenchmarkRunner {
		return fakeRunner
	}
	t.Cleanup(func() {
		v7BackendBenchmarkStatus = oldStatus
		newV7LlamaCppBackendBenchmarkRunner = oldFactory
		operatorRuntimeState = oldState
	})

	handled, result, err := processOptionalV7LlamaCppBackendBenchmark(context.Background(), nil, &hub.WorkAssignment{
		JobID:    "job-llamacpp-backend-local",
		Kind:     "benchmark",
		SpecJSON: testLlamaCppBackendBenchmarkSpecJSON(t),
	}, nil, false)
	if err != nil {
		t.Fatalf("processOptionalV7LlamaCppBackendBenchmark() error = %v", err)
	}
	if handled {
		t.Fatal("handled = true, want false when flag is off")
	}
	if result != nil {
		t.Fatalf("result = %+v, want nil", result)
	}
	if fakeRunner.calls != 0 {
		t.Fatalf("runner calls = %d, want 0", fakeRunner.calls)
	}
	if snapshot := status.Snapshot(); snapshot.Counters != (v7llamacpp.BackendBenchmarkLocalStatusCounters{}) {
		t.Fatalf("status counters changed with flag off: %+v", snapshot.Counters)
	}
}

func TestProcessOptionalV7BackendInferenceBenchmarkFlagOffDoesNotHandle(t *testing.T) {
	t.Setenv(v7inferencebench.BenchmarkFlagEnv, "")
	oldFactory := newV7BackendInferenceBenchmarkRunner
	oldStatus := v7InferenceBenchmarkStatus
	status := v7inferencebench.NewLocalStatus()
	v7InferenceBenchmarkStatus = status
	fakeRunner := &fakeBackendInferenceBenchmarkRunner{result: testBackendInferenceBenchmarkResult(v7inferencebench.ProofStatusMeasured)}
	newV7BackendInferenceBenchmarkRunner = func() v7inferencebench.BenchmarkRunner {
		return fakeRunner
	}
	t.Cleanup(func() {
		newV7BackendInferenceBenchmarkRunner = oldFactory
		v7InferenceBenchmarkStatus = oldStatus
	})

	handled, result, err := processOptionalV7BackendInferenceBenchmark(context.Background(), nil, &hub.WorkAssignment{
		JobID:    "job-backend-inference-local",
		Kind:     "benchmark",
		SpecJSON: testBackendInferenceBenchmarkSpecJSON(t),
	}, nil, false)
	if err != nil {
		t.Fatalf("processOptionalV7BackendInferenceBenchmark() error = %v", err)
	}
	if handled {
		t.Fatal("handled = true, want false when flag is off")
	}
	if result != nil {
		t.Fatalf("result = %+v, want nil", result)
	}
	if fakeRunner.calls != 0 {
		t.Fatalf("runner calls = %d, want 0", fakeRunner.calls)
	}
	if snapshot := status.Snapshot(); snapshot.Counters != (v7inferencebench.LocalStatusCounters{}) {
		t.Fatalf("status counters changed with flag off: %+v", snapshot.Counters)
	}
}

func TestProcessOptionalV7BackendInferenceBenchmarkSubmitsMeasuredReceipt(t *testing.T) {
	t.Setenv(v7inferencebench.BenchmarkFlagEnv, "1")
	oldFactory := newV7BackendInferenceBenchmarkRunner
	oldStatus := v7InferenceBenchmarkStatus
	oldDiagnostics := workLoopDiagnostics
	status := v7inferencebench.NewLocalStatus()
	fakeRunner := &fakeBackendInferenceBenchmarkRunner{result: testBackendInferenceBenchmarkResult(v7inferencebench.ProofStatusMeasured)}
	v7InferenceBenchmarkStatus = status
	workLoopDiagnostics = diagnostics.NewWorkLoopDiagnostics()
	newV7BackendInferenceBenchmarkRunner = func() v7inferencebench.BenchmarkRunner {
		return fakeRunner
	}
	t.Cleanup(func() {
		newV7BackendInferenceBenchmarkRunner = oldFactory
		v7InferenceBenchmarkStatus = oldStatus
		workLoopDiagnostics = oldDiagnostics
	})

	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	var receiptCalls atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/node/receipt" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		receiptCalls.Add(1)
		var req struct {
			JobID         string         `json:"job_id"`
			ResultHashHex string         `json:"result_hash_hex"`
			Metadata      map[string]any `json:"metadata"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode receipt: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if req.JobID != "job-backend-inference-local" {
			t.Errorf("receipt job_id = %q", req.JobID)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if len(req.ResultHashHex) != 64 {
			t.Errorf("result_hash_hex = %q, want 64 hex chars", req.ResultHashHex)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		metadata, ok := req.Metadata[v7inferencebench.BenchmarkTask].(map[string]any)
		if !ok {
			t.Errorf("receipt metadata missing %q: %+v", v7inferencebench.BenchmarkTask, req.Metadata)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if metadata["proof_status"] != v7inferencebench.ProofStatusMeasured {
			t.Errorf("proof_status = %v", metadata["proof_status"])
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		for _, key := range []string{"prompt_hash", "output_hash"} {
			value, ok := metadata[key].(string)
			if !ok || !strings.HasPrefix(value, "sha256:") {
				t.Errorf("%s = %#v, want sha256 hash", key, metadata[key])
				w.WriteHeader(http.StatusBadRequest)
				return
			}
		}
		if metadata["tokens_generated"] != float64(12) || metadata["p50_ttft_ms"] != float64(100) || metadata["p95_ttft_ms"] != float64(100) {
			t.Errorf("metrics metadata = %+v", metadata)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if metadata["p50_decode_tps"] != float64(20) || metadata["p50_end_to_end_tps"] != float64(17.143) {
			t.Errorf("tps metadata = %+v", metadata)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		for _, forbiddenKey := range []string{"output_bytes", "total_time_ms", "ttft_ms", "decode_tps", "end_to_end_tps", "prompt", "messages", "output_text", "generated_text", "raw_output", "tokens", "key_data", "value_data", "query_vector", "tensor_bytes"} {
			if _, ok := metadata[forbiddenKey]; ok {
				t.Errorf("metadata includes forbidden key %q: %+v", forbiddenKey, metadata)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
		}
		encoded, _ := json.Marshal(req.Metadata)
		for _, forbidden := range []string{"secret measured output", "distributed computing", "output_text", "generated_text", "raw_output", "token_logprobs", "raw_tensor"} {
			if strings.Contains(strings.ToLower(string(encoded)), strings.ToLower(forbidden)) {
				t.Errorf("metadata leaked forbidden material %q: %s", forbidden, encoded)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	handled, result, err := processOptionalV7BackendInferenceBenchmark(context.Background(), hub.New(ts.URL, pub, priv), &hub.WorkAssignment{
		JobID:    "job-backend-inference-local",
		Kind:     "benchmark",
		SpecJSON: testBackendInferenceBenchmarkSpecJSON(t),
	}, nil, true)
	if err != nil {
		t.Fatalf("processOptionalV7BackendInferenceBenchmark() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if result == nil || result.ResultHashHex == "" || result.ExitCode != 0 {
		t.Fatalf("result = %+v, want successful backend inference benchmark receipt", result)
	}
	if got := receiptCalls.Load(); got != 1 {
		t.Fatalf("receipt calls = %d, want 1", got)
	}
	if fakeRunner.calls != 1 {
		t.Fatalf("runner calls = %d, want 1", fakeRunner.calls)
	}
	statusSnapshot := status.Snapshot()
	if statusSnapshot.LastJobID != "job-backend-inference-local" || statusSnapshot.LastError != "" {
		t.Fatalf("unexpected local inference benchmark status: %+v", statusSnapshot)
	}
	if statusSnapshot.Counters.Seen != 1 || statusSnapshot.Counters.Executed != 1 || statusSnapshot.Counters.ReceiptSubmitted != 1 || statusSnapshot.Counters.ReceiptFailed != 0 {
		t.Fatalf("unexpected local inference benchmark counters: %+v", statusSnapshot.Counters)
	}
	workSnapshot := workLoopDiagnostics.Snapshot()
	if workSnapshot.ReceiptSubmittedCount != 1 || workSnapshot.ReceiptFailedCount != 0 {
		t.Fatalf("unexpected work-loop receipt counters: %+v", workSnapshot)
	}
}

func TestProcessOptionalV7DashboardInferenceSubmitsMeasuredReceipt(t *testing.T) {
	t.Setenv(v7dashboardinference.FlagEnv, "1")
	oldFactory := newV7DashboardInferenceRunner
	oldStatus := v7DashboardInferenceStatus
	oldDiagnostics := workLoopDiagnostics
	status := v7dashboardinference.NewLocalStatus()
	fakeRunner := &fakeDashboardInferenceRunner{result: testDashboardInferenceResult(v7dashboardinference.ProofStatusMeasured)}
	v7DashboardInferenceStatus = status
	workLoopDiagnostics = diagnostics.NewWorkLoopDiagnostics()
	newV7DashboardInferenceRunner = func() v7dashboardinference.Runner {
		return fakeRunner
	}
	t.Cleanup(func() {
		newV7DashboardInferenceRunner = oldFactory
		v7DashboardInferenceStatus = oldStatus
		workLoopDiagnostics = oldDiagnostics
	})

	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	var receiptCalls atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/node/receipt" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		receiptCalls.Add(1)
		var req struct {
			JobID         string         `json:"job_id"`
			ResultHashHex string         `json:"result_hash_hex"`
			Metadata      map[string]any `json:"metadata"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode receipt: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if req.JobID != "v7dashboardinfer_job" {
			t.Errorf("receipt job_id = %q", req.JobID)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if len(req.ResultHashHex) != 64 {
			t.Errorf("result_hash_hex = %q, want 64 hex chars", req.ResultHashHex)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		metadata, ok := req.Metadata[v7dashboardinference.Task].(map[string]any)
		if !ok {
			t.Errorf("receipt metadata missing %q: %+v", v7dashboardinference.Task, req.Metadata)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if metadata["proof_status"] != v7dashboardinference.ProofStatusMeasured {
			t.Errorf("proof_status = %v", metadata["proof_status"])
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if metadata["request_id"] != "dashboardinfer_request" || metadata["run_id"] != "dashboardinfer_run" {
			t.Errorf("request/run metadata = %+v", metadata)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if metadata["backend"] != v7llamacpp.BackendName || metadata["model_id"] != "Llama-3.2-3B-Instruct-Q4_K_M.gguf" {
			t.Errorf("backend/model metadata = %+v", metadata)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		outputHash, ok := metadata["output_hash"].(string)
		if !ok || !strings.HasPrefix(outputHash, "sha256:") {
			t.Errorf("output_hash = %#v, want sha256 hash", metadata["output_hash"])
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if metadata["output_bytes"] != float64(len("dashboard measured output")) || metadata["tokens_generated"] != float64(9) {
			t.Errorf("output metric metadata = %+v", metadata)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if metadata["ttft_ms"] != float64(120) || metadata["total_time_ms"] != float64(720) {
			t.Errorf("timing metadata = %+v", metadata)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if metadata["decode_tps"] != float64(15) || metadata["end_to_end_tps"] != float64(12.5) {
			t.Errorf("tps metadata = %+v", metadata)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		encoded, _ := json.Marshal(req.Metadata)
		for _, forbidden := range []string{"dashboard measured output", "raw_prompt", "prompt_text", "messages", "input_text", "output_text", "generated_text", "raw_output", "completion", "token_logprobs", "key_data", "value_data", "query_vector", "tensor_bytes", "secret"} {
			if strings.Contains(strings.ToLower(string(encoded)), strings.ToLower(forbidden)) {
				t.Errorf("metadata leaked forbidden material %q: %s", forbidden, encoded)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	handled, result, err := processOptionalV7DashboardInference(context.Background(), hub.New(ts.URL, pub, priv), &hub.WorkAssignment{
		JobID:    "v7dashboardinfer_job",
		Kind:     "inference",
		SpecJSON: testDashboardInferenceSpecJSON(t),
	}, nil, true)
	if err != nil {
		t.Fatalf("processOptionalV7DashboardInference() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if result == nil || result.ResultHashHex == "" || result.ExitCode != 0 {
		t.Fatalf("result = %+v, want successful dashboard inference receipt", result)
	}
	if got := receiptCalls.Load(); got != 1 {
		t.Fatalf("receipt calls = %d, want 1", got)
	}
	if fakeRunner.calls != 1 {
		t.Fatalf("runner calls = %d, want 1", fakeRunner.calls)
	}
	snapshot := status.Snapshot()
	if snapshot.LastRunID != "dashboardinfer_run" || snapshot.LastJobID != "v7dashboardinfer_job" || snapshot.LastError != "" {
		t.Fatalf("unexpected dashboard inference status: %+v", snapshot)
	}
	if snapshot.Counters.Seen != 1 || snapshot.Counters.Executed != 1 || snapshot.Counters.ReceiptSubmitted != 1 || snapshot.Counters.ReceiptFailed != 0 || snapshot.Counters.Rejected != 0 {
		t.Fatalf("unexpected dashboard inference counters: %+v", snapshot.Counters)
	}
}

func TestProcessOptionalV7DashboardInferenceSubmitsOptInGeneratedText(t *testing.T) {
	t.Setenv(v7dashboardinference.FlagEnv, "1")
	t.Setenv(v7dashboardinference.TextOutputFlagEnv, "1")
	oldFactory := newV7DashboardInferenceRunner
	oldStatus := v7DashboardInferenceStatus
	oldDiagnostics := workLoopDiagnostics
	status := v7dashboardinference.NewLocalStatus()
	result := testDashboardInferenceResult(v7dashboardinference.ProofStatusMeasured)
	result.Spec = v7dashboardinference.Spec{}
	result.GeneratedText = "Ryvion routes AI work to warm, ready nodes."
	fakeRunner := &fakeDashboardInferenceRunner{result: result}
	v7DashboardInferenceStatus = status
	workLoopDiagnostics = diagnostics.NewWorkLoopDiagnostics()
	newV7DashboardInferenceRunner = func() v7dashboardinference.Runner {
		return fakeRunner
	}
	t.Cleanup(func() {
		newV7DashboardInferenceRunner = oldFactory
		v7DashboardInferenceStatus = oldStatus
		workLoopDiagnostics = oldDiagnostics
	})

	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	var receiptCalls atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receiptCalls.Add(1)
		var req struct {
			Metadata map[string]any `json:"metadata"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode receipt: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		metadata, ok := req.Metadata[v7dashboardinference.Task].(map[string]any)
		if !ok {
			t.Errorf("receipt metadata missing %q: %+v", v7dashboardinference.Task, req.Metadata)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if metadata["generated_text"] != "Ryvion routes AI work to warm, ready nodes." || metadata["generated_text_truncated"] != false {
			t.Errorf("generated text metadata = %+v", metadata)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if metadata["output_hash"] == "" || metadata["tokens_generated"] != float64(9) || metadata["ttft_ms"] != float64(120) {
			t.Errorf("metadata missing hash/timing: %+v", metadata)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		encoded, _ := json.Marshal(req.Metadata)
		for _, forbidden := range []string{"private dashboard prompt", "raw_prompt", "prompt_text", "messages", "input_text", "output_text", "raw_output", "completion", "token_logprobs", "key_data", "value_data", "query_vector", "tensor_bytes", "secret"} {
			if strings.Contains(strings.ToLower(string(encoded)), strings.ToLower(forbidden)) {
				t.Errorf("metadata leaked forbidden material %q: %s", forbidden, encoded)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	handled, processResult, err := processOptionalV7DashboardInference(context.Background(), hub.New(ts.URL, pub, priv), &hub.WorkAssignment{
		JobID:    "v7dashboardinfer_job",
		Kind:     "inference",
		SpecJSON: testDashboardInferenceTextSpecJSON(t),
	}, nil, true)
	if err != nil {
		t.Fatalf("processOptionalV7DashboardInference() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if processResult == nil || processResult.ResultHashHex == "" || processResult.ExitCode != 0 {
		t.Fatalf("result = %+v, want successful dashboard inference receipt", processResult)
	}
	if got := receiptCalls.Load(); got != 1 {
		t.Fatalf("receipt calls = %d, want 1", got)
	}
}

func TestProcessOptionalV7DashboardInferenceSendsProgressBeforeReceipt(t *testing.T) {
	t.Setenv(v7dashboardinference.FlagEnv, "1")
	t.Setenv(v7dashboardinference.TextOutputFlagEnv, "1")
	t.Setenv(v7dashboardinference.StreamingFlagEnv, "1")
	oldFactory := newV7DashboardInferenceRunner
	oldStatus := v7DashboardInferenceStatus
	oldDiagnostics := workLoopDiagnostics
	status := v7dashboardinference.NewLocalStatus()
	generated := "Ryvion streams"
	fakeRunner := &fakeDashboardInferenceProgressRunner{generated: generated}
	v7DashboardInferenceStatus = status
	workLoopDiagnostics = diagnostics.NewWorkLoopDiagnostics()
	newV7DashboardInferenceRunner = func() v7dashboardinference.Runner {
		return fakeRunner
	}
	t.Cleanup(func() {
		newV7DashboardInferenceRunner = oldFactory
		v7DashboardInferenceStatus = oldStatus
		workLoopDiagnostics = oldDiagnostics
	})

	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	var (
		mu     sync.Mutex
		events []string
	)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		events = append(events, r.URL.Path)
		mu.Unlock()
		switch r.URL.Path {
		case "/api/v1/node/inference/chunks":
			var req struct {
				RunID    string `json:"run_id"`
				JobID    string `json:"job_id"`
				NodeID   string `json:"node_id"`
				SeqStart int64  `json:"seq_start"`
				Chunks   []struct {
					Seq  int64  `json:"seq"`
					Type string `json:"type"`
					Text string `json:"text"`
				} `json:"chunks"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Errorf("decode chunks: %v", err)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			if req.RunID != "dashboardinfer_run" || req.JobID != "v7dashboardinfer_job" || req.NodeID != "node-dashboard-local" || req.SeqStart != 1 {
				t.Errorf("chunk identity = %+v", req)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			if len(req.Chunks) != 2 || req.Chunks[0].Seq != 1 || req.Chunks[0].Text != "Ryvion" || req.Chunks[1].Seq != 2 || req.Chunks[1].Text != " streams" {
				t.Errorf("chunks = %+v", req.Chunks)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"accepted":true}`))
		case "/api/v1/node/receipt":
			var req struct {
				Metadata map[string]any `json:"metadata"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Errorf("decode receipt: %v", err)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			metadata := req.Metadata[v7dashboardinference.Task].(map[string]any)
			if metadata["generated_text"] != generated {
				t.Errorf("receipt generated_text = %#v, want %q", metadata["generated_text"], generated)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			w.WriteHeader(http.StatusOK)
		default:
			t.Errorf("unexpected path: %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer ts.Close()

	handled, processResult, err := processOptionalV7DashboardInference(context.Background(), hub.New(ts.URL, pub, priv), &hub.WorkAssignment{
		JobID:    "v7dashboardinfer_job",
		Kind:     "inference",
		SpecJSON: testDashboardInferenceTextSpecJSON(t),
	}, nil, true)
	if err != nil {
		t.Fatalf("processOptionalV7DashboardInference() error = %v", err)
	}
	if !handled || processResult == nil || processResult.ExitCode != 0 {
		t.Fatalf("handled=%v result=%+v, want successful process result", handled, processResult)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(events) != 2 || events[0] != "/api/v1/node/inference/chunks" || events[1] != "/api/v1/node/receipt" {
		t.Fatalf("events = %+v, want chunks before receipt", events)
	}
}

func TestProcessOptionalV7DashboardInferenceRecordsRejectedCounter(t *testing.T) {
	t.Setenv(v7dashboardinference.FlagEnv, "1")
	oldFactory := newV7DashboardInferenceRunner
	oldStatus := v7DashboardInferenceStatus
	oldDiagnostics := workLoopDiagnostics
	status := v7dashboardinference.NewLocalStatus()
	fakeRunner := &fakeDashboardInferenceRunner{result: testDashboardInferenceResult(v7dashboardinference.ProofStatusRejected)}
	v7DashboardInferenceStatus = status
	workLoopDiagnostics = diagnostics.NewWorkLoopDiagnostics()
	newV7DashboardInferenceRunner = func() v7dashboardinference.Runner {
		return fakeRunner
	}
	t.Cleanup(func() {
		newV7DashboardInferenceRunner = oldFactory
		v7DashboardInferenceStatus = oldStatus
		workLoopDiagnostics = oldDiagnostics
	})

	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	var receiptCalls atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receiptCalls.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	handled, result, err := processOptionalV7DashboardInference(context.Background(), hub.New(ts.URL, pub, priv), &hub.WorkAssignment{
		JobID:    "v7dashboardinfer_job",
		Kind:     "inference",
		SpecJSON: testDashboardInferenceSpecJSON(t),
	}, nil, true)
	if err == nil {
		t.Fatal("processOptionalV7DashboardInference() error = nil, want safe rejection error")
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if result == nil || result.ExitCode != 1 {
		t.Fatalf("result = %+v, want rejected dashboard inference receipt snapshot", result)
	}
	if got := receiptCalls.Load(); got != 1 {
		t.Fatalf("receipt calls = %d, want 1", got)
	}
	if fakeRunner.calls != 1 {
		t.Fatalf("runner calls = %d, want 1", fakeRunner.calls)
	}
	snapshot := status.Snapshot()
	if snapshot.LastRunID != "dashboardinfer_run" || snapshot.LastJobID != "v7dashboardinfer_job" {
		t.Fatalf("unexpected dashboard inference status: %+v", snapshot)
	}
	if snapshot.Counters.Seen != 1 || snapshot.Counters.Executed != 0 || snapshot.Counters.ReceiptSubmitted != 1 || snapshot.Counters.ReceiptFailed != 0 || snapshot.Counters.Rejected != 1 {
		t.Fatalf("unexpected dashboard inference counters: %+v", snapshot.Counters)
	}
}

func TestProcessOptionalV7DashboardInferenceRecordsReceiptFailureCounter(t *testing.T) {
	t.Setenv(v7dashboardinference.FlagEnv, "1")
	oldFactory := newV7DashboardInferenceRunner
	oldStatus := v7DashboardInferenceStatus
	oldDiagnostics := workLoopDiagnostics
	status := v7dashboardinference.NewLocalStatus()
	fakeRunner := &fakeDashboardInferenceRunner{result: testDashboardInferenceResult(v7dashboardinference.ProofStatusMeasured)}
	v7DashboardInferenceStatus = status
	workLoopDiagnostics = diagnostics.NewWorkLoopDiagnostics()
	newV7DashboardInferenceRunner = func() v7dashboardinference.Runner {
		return fakeRunner
	}
	t.Cleanup(func() {
		newV7DashboardInferenceRunner = oldFactory
		v7DashboardInferenceStatus = oldStatus
		workLoopDiagnostics = oldDiagnostics
	})

	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	var receiptCalls atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receiptCalls.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer ts.Close()

	handled, result, err := processOptionalV7DashboardInference(context.Background(), hub.New(ts.URL, pub, priv), &hub.WorkAssignment{
		JobID:    "v7dashboardinfer_job",
		Kind:     "inference",
		SpecJSON: testDashboardInferenceSpecJSON(t),
	}, nil, true)
	if err == nil {
		t.Fatal("processOptionalV7DashboardInference() error = nil, want submit error")
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if result == nil || result.ResultHashHex == "" || result.ExitCode != 0 {
		t.Fatalf("result = %+v, want measured dashboard inference receipt snapshot despite submit error", result)
	}
	if got := receiptCalls.Load(); got != 1 {
		t.Fatalf("receipt calls = %d, want 1", got)
	}
	if fakeRunner.calls != 1 {
		t.Fatalf("runner calls = %d, want 1", fakeRunner.calls)
	}
	snapshot := status.Snapshot()
	if snapshot.LastRunID != "dashboardinfer_run" || snapshot.LastJobID != "v7dashboardinfer_job" {
		t.Fatalf("unexpected dashboard inference status: %+v", snapshot)
	}
	if snapshot.LastError == "" {
		t.Fatalf("last_error not recorded: %+v", snapshot)
	}
	if snapshot.Counters.Seen != 1 || snapshot.Counters.Executed != 1 || snapshot.Counters.ReceiptSubmitted != 0 || snapshot.Counters.ReceiptFailed != 1 || snapshot.Counters.Rejected != 0 {
		t.Fatalf("unexpected dashboard inference counters: %+v", snapshot.Counters)
	}
	workSnapshot := workLoopDiagnostics.Snapshot()
	if workSnapshot.ReceiptSubmittedCount != 0 || workSnapshot.ReceiptFailedCount != 1 {
		t.Fatalf("unexpected work-loop receipt counters: %+v", workSnapshot)
	}
}

func TestProcessOptionalV7LlamaCppBackendBenchmarkSubmitsMeasuredReceipt(t *testing.T) {
	t.Setenv(v7llamacpp.BackendBenchmarkFlagEnv, "1")
	oldStatus := v7BackendBenchmarkStatus
	oldFactory := newV7LlamaCppBackendBenchmarkRunner
	oldState := operatorRuntimeState
	oldDiagnostics := workLoopDiagnostics
	status := v7llamacpp.NewBackendBenchmarkLocalStatus()
	fakeRunner := &fakeLlamaCppBackendBenchmarkRunner{snapshot: testLlamaCppBackendBenchmarkSnapshot(v7llamacpp.BenchmarkProofStatusMeasured)}
	v7BackendBenchmarkStatus = status
	operatorRuntimeState = nil
	workLoopDiagnostics = diagnostics.NewWorkLoopDiagnostics()
	newV7LlamaCppBackendBenchmarkRunner = func() v7llamacpp.BackendBenchmarkRunner {
		return fakeRunner
	}
	t.Cleanup(func() {
		v7BackendBenchmarkStatus = oldStatus
		newV7LlamaCppBackendBenchmarkRunner = oldFactory
		operatorRuntimeState = oldState
		workLoopDiagnostics = oldDiagnostics
	})

	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	var receiptCalls atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/node/receipt" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		receiptCalls.Add(1)
		var req struct {
			JobID         string         `json:"job_id"`
			ResultHashHex string         `json:"result_hash_hex"`
			Metadata      map[string]any `json:"metadata"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode receipt: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if req.JobID != "job-llamacpp-backend-local" {
			t.Errorf("receipt job_id = %q", req.JobID)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if len(req.ResultHashHex) != 64 {
			t.Errorf("result_hash_hex = %q, want 64 hex chars", req.ResultHashHex)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		metadata, ok := req.Metadata[v7llamacpp.BackendBenchmarkTask].(map[string]any)
		if !ok {
			t.Errorf("receipt metadata missing %q: %+v", v7llamacpp.BackendBenchmarkTask, req.Metadata)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if metadata["proof_status"] != v7llamacpp.BenchmarkProofStatusMeasured {
			t.Errorf("proof_status = %v", metadata["proof_status"])
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		for _, key := range []string{"prompt_hash", "output_hash"} {
			value, ok := metadata[key].(string)
			if !ok || !strings.HasPrefix(value, "sha256:") {
				t.Errorf("%s = %#v, want sha256 hash", key, metadata[key])
				w.WriteHeader(http.StatusBadRequest)
				return
			}
		}
		if metadata["p50_ttft_ms"] != float64(100) || metadata["p95_total_time_ms"] != float64(1100) {
			t.Errorf("timing metadata = %+v", metadata)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		encoded, _ := json.Marshal(req.Metadata)
		for _, forbidden := range []string{"secret llama output", "distributed computing", "output_text", "generated_text", "raw_output", "token_logprobs", "raw_tensor"} {
			if strings.Contains(strings.ToLower(string(encoded)), strings.ToLower(forbidden)) {
				t.Errorf("metadata leaked forbidden material %q: %s", forbidden, encoded)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	handled, result, err := processOptionalV7LlamaCppBackendBenchmark(context.Background(), hub.New(ts.URL, pub, priv), &hub.WorkAssignment{
		JobID:    "job-llamacpp-backend-local",
		Kind:     "benchmark",
		SpecJSON: testLlamaCppBackendBenchmarkSpecJSON(t),
	}, nil, true)
	if err != nil {
		t.Fatalf("processOptionalV7LlamaCppBackendBenchmark() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if result == nil || result.ResultHashHex == "" || result.ExitCode != 0 {
		t.Fatalf("result = %+v, want successful backend benchmark receipt", result)
	}
	if got := receiptCalls.Load(); got != 1 {
		t.Fatalf("receipt calls = %d, want 1", got)
	}
	if fakeRunner.calls != 1 {
		t.Fatalf("runner calls = %d, want 1", fakeRunner.calls)
	}
	snapshot := status.Snapshot()
	if snapshot.LastJobID != "job-llamacpp-backend-local" {
		t.Fatalf("last_job_id = %q", snapshot.LastJobID)
	}
	if snapshot.Counters.Seen != 1 || snapshot.Counters.Executed != 1 || snapshot.Counters.ReceiptSubmitted != 1 || snapshot.Counters.ReceiptFailed != 0 {
		t.Fatalf("unexpected counters: %+v", snapshot.Counters)
	}
}

func TestProcessOptionalV7ModelBenchmarkSubmitsMeasuredReceipt(t *testing.T) {
	t.Setenv(v7modelbench.ModelBenchmarkFlagEnv, "1")
	oldStatus := v7ModelBenchmarkStatus
	oldFactory := newV7ModelBenchmarkRunner
	status := v7modelbench.NewLocalStatus()
	fakeRunner := &fakeModelBenchmarkRunner{result: testMeasuredModelBenchmarkResult()}
	v7ModelBenchmarkStatus = status
	newV7ModelBenchmarkRunner = func(_ *inference.Manager, gpuDetected bool) v7modelbench.ModelBenchmarkRunner {
		if !gpuDetected {
			t.Fatal("gpuDetected = false, want true")
		}
		return fakeRunner
	}
	t.Cleanup(func() {
		v7ModelBenchmarkStatus = oldStatus
		newV7ModelBenchmarkRunner = oldFactory
	})

	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	var receiptCalls atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/node/receipt" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		receiptCalls.Add(1)
		var req struct {
			JobID         string         `json:"job_id"`
			ResultHashHex string         `json:"result_hash_hex"`
			Metadata      map[string]any `json:"metadata"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode receipt: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if req.JobID != "job-modelbench-local" {
			t.Errorf("receipt job_id = %q", req.JobID)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if len(req.ResultHashHex) != 64 {
			t.Errorf("result_hash_hex = %q, want 64 hex chars", req.ResultHashHex)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		metadata, ok := req.Metadata[v7modelbench.ModelBenchmarkTask].(map[string]any)
		if !ok {
			t.Errorf("receipt metadata missing %q: %+v", v7modelbench.ModelBenchmarkTask, req.Metadata)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if metadata["proof_status"] != string(v7modelbench.ModelBenchmarkProofStatusMeasured) {
			t.Errorf("proof_status = %v", metadata["proof_status"])
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if _, ok := metadata["output_hash"]; !ok {
			t.Errorf("measured metadata missing output_hash: %+v", metadata)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		encoded, _ := json.Marshal(req.Metadata)
		if strings.Contains(string(encoded), "secret completion text") {
			t.Errorf("metadata leaked raw output: %s", encoded)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	handled, result, err := processOptionalV7ModelBenchmark(context.Background(), hub.New(ts.URL, pub, priv), &hub.WorkAssignment{
		JobID:    "job-modelbench-local",
		Kind:     "benchmark",
		SpecJSON: testModelBenchmarkSpecJSON(t),
	}, nil, nil, true)
	if err != nil {
		t.Fatalf("processOptionalV7ModelBenchmark() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if result == nil || result.ResultHashHex == "" {
		t.Fatalf("result = %+v, want hash", result)
	}
	if got := receiptCalls.Load(); got != 1 {
		t.Fatalf("receipt calls = %d, want 1", got)
	}
	if fakeRunner.calls != 1 {
		t.Fatalf("runner calls = %d, want 1", fakeRunner.calls)
	}

	snapshot := status.Snapshot()
	if snapshot.LastSeenBenchmarkJobID != "job-modelbench-local" || snapshot.LastSeenRequestID != "request-modelbench-local" {
		t.Fatalf("seen status = %+v", snapshot)
	}
	if snapshot.LastExecutedJobID != "job-modelbench-local" {
		t.Fatalf("last executed job id = %q", snapshot.LastExecutedJobID)
	}
	if snapshot.LastReceiptSubmittedJobID != "job-modelbench-local" {
		t.Fatalf("last receipt submitted job id = %q", snapshot.LastReceiptSubmittedJobID)
	}
	if snapshot.Counters.Seen != 1 || snapshot.Counters.Executed != 1 || snapshot.Counters.ReceiptSubmitted != 1 || snapshot.Counters.ReceiptFailed != 0 {
		t.Fatalf("unexpected counters: %+v", snapshot.Counters)
	}
}

func TestProcessOptionalV7ModelBenchmarkSubmitsUnavailableReceipt(t *testing.T) {
	t.Setenv(v7modelbench.ModelBenchmarkFlagEnv, "1")
	oldStatus := v7ModelBenchmarkStatus
	oldFactory := newV7ModelBenchmarkRunner
	status := v7modelbench.NewLocalStatus()
	fakeRunner := &fakeModelBenchmarkRunner{result: testUnavailableModelBenchmarkResult()}
	v7ModelBenchmarkStatus = status
	newV7ModelBenchmarkRunner = func(_ *inference.Manager, _ bool) v7modelbench.ModelBenchmarkRunner {
		return fakeRunner
	}
	t.Cleanup(func() {
		v7ModelBenchmarkStatus = oldStatus
		newV7ModelBenchmarkRunner = oldFactory
	})

	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Metadata map[string]any `json:"metadata"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode receipt: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		metadata := req.Metadata[v7modelbench.ModelBenchmarkTask].(map[string]any)
		if metadata["proof_status"] != string(v7modelbench.ModelBenchmarkProofStatusUnavailable) {
			t.Errorf("proof_status = %v", metadata["proof_status"])
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if _, ok := metadata["output_hash"]; ok {
			t.Errorf("unavailable metadata contains output_hash: %+v", metadata)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		runtimeMeta := metadata["runtime"].(map[string]any)
		if runtimeMeta["native_inference_ready"] != false || runtimeMeta["model_loaded"] != false {
			t.Errorf("runtime metadata = %+v, want native/model not ready", runtimeMeta)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	handled, result, err := processOptionalV7ModelBenchmark(context.Background(), hub.New(ts.URL, pub, priv), &hub.WorkAssignment{
		JobID:    "job-modelbench-local",
		Kind:     "benchmark",
		SpecJSON: testModelBenchmarkSpecJSON(t),
	}, nil, nil, false)
	if err != nil {
		t.Fatalf("processOptionalV7ModelBenchmark() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if result == nil || result.ExitCode != 0 {
		t.Fatalf("result = %+v, want successful unavailable receipt", result)
	}
	snapshot := status.Snapshot()
	if snapshot.Counters.Executed != 1 || snapshot.Counters.ReceiptSubmitted != 1 || snapshot.Counters.ReceiptFailed != 0 {
		t.Fatalf("unexpected counters: %+v", snapshot.Counters)
	}
}

func TestProcessOptionalV7ModelBenchmarkSubmitsSeriesReceipt(t *testing.T) {
	t.Setenv(v7modelbench.ModelBenchmarkFlagEnv, "1")
	oldStatus := v7ModelBenchmarkStatus
	oldFactory := newV7ModelBenchmarkRunner
	status := v7modelbench.NewLocalStatus()
	fakeRunner := &fakeModelBenchmarkRunner{
		results: []v7modelbench.ModelBenchmarkResult{
			testMeasuredModelBenchmarkSeriesResult(9_999, 10_999, 99, 99),
			testMeasuredModelBenchmarkSeriesResult(100, 1_100, 11, 10),
			testMeasuredModelBenchmarkSeriesResult(200, 1_200, 11, 9.17),
		},
	}
	v7ModelBenchmarkStatus = status
	newV7ModelBenchmarkRunner = func(_ *inference.Manager, _ bool) v7modelbench.ModelBenchmarkRunner {
		return fakeRunner
	}
	t.Cleanup(func() {
		v7ModelBenchmarkStatus = oldStatus
		newV7ModelBenchmarkRunner = oldFactory
	})

	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	var receiptCalls atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receiptCalls.Add(1)
		var req struct {
			JobID         string         `json:"job_id"`
			ResultHashHex string         `json:"result_hash_hex"`
			Metadata      map[string]any `json:"metadata"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode receipt: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if req.JobID != "job-modelbench-series-local" {
			t.Errorf("receipt job_id = %q", req.JobID)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if len(req.ResultHashHex) != 64 {
			t.Errorf("result_hash_hex = %q, want 64 hex chars", req.ResultHashHex)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		metadata, ok := req.Metadata[v7modelbench.ModelBenchmarkSeriesTask].(map[string]any)
		if !ok {
			t.Errorf("receipt metadata missing %q: %+v", v7modelbench.ModelBenchmarkSeriesTask, req.Metadata)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if metadata["proof_status"] != v7modelbench.ModelBenchmarkSeriesProofStatusMeasured {
			t.Errorf("proof_status = %v", metadata["proof_status"])
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		summary := metadata["summary"].(map[string]any)
		if summary["successful_measured_runs"] != float64(2) || summary["p50_ttft_ms"] != float64(100) || summary["p95_ttft_ms"] != float64(200) {
			t.Errorf("summary = %+v, want measured-only p50/p95 and 2 successes", summary)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		trials := metadata["trials"].([]any)
		if len(trials) != 3 {
			t.Errorf("trials len = %d, want 3", len(trials))
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		encoded, _ := json.Marshal(req.Metadata)
		for _, forbidden := range []string{"secret series completion text", "Continue the numbered sequence", "prompt_text", "output_text", "error_message"} {
			if strings.Contains(string(encoded), forbidden) {
				t.Errorf("metadata leaked raw series content %q: %s", forbidden, encoded)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	handled, result, err := processOptionalV7ModelBenchmark(context.Background(), hub.New(ts.URL, pub, priv), &hub.WorkAssignment{
		JobID:    "job-modelbench-series-local",
		Kind:     "benchmark",
		SpecJSON: testModelBenchmarkSeriesSpecJSON(t, 1, 2),
	}, nil, nil, true)
	if err != nil {
		t.Fatalf("processOptionalV7ModelBenchmark() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if result == nil || result.ResultHashHex == "" || result.ExitCode != 0 {
		t.Fatalf("result = %+v, want successful series hash", result)
	}
	if got := receiptCalls.Load(); got != 1 {
		t.Fatalf("receipt calls = %d, want 1", got)
	}
	if fakeRunner.calls != 3 {
		t.Fatalf("runner calls = %d, want 3", fakeRunner.calls)
	}

	snapshot := status.Snapshot()
	if snapshot.LastSeenBenchmarkJobID != "job-modelbench-series-local" || snapshot.LastSeenRequestID != "request-modelbench-series-local" {
		t.Fatalf("seen status = %+v", snapshot)
	}
	if snapshot.Counters.Seen != 1 || snapshot.Counters.Executed != 1 || snapshot.Counters.ReceiptSubmitted != 1 || snapshot.Counters.ReceiptFailed != 0 {
		t.Fatalf("unexpected counters: %+v", snapshot.Counters)
	}
}

func TestProcessOptionalV7ModelBenchmarkSubmitsUnavailableSeriesReceipt(t *testing.T) {
	t.Setenv(v7modelbench.ModelBenchmarkFlagEnv, "1")
	oldStatus := v7ModelBenchmarkStatus
	oldFactory := newV7ModelBenchmarkRunner
	status := v7modelbench.NewLocalStatus()
	fakeRunner := &fakeModelBenchmarkRunner{
		results: []v7modelbench.ModelBenchmarkResult{
			testUnavailableModelBenchmarkSeriesResult(),
			testUnavailableModelBenchmarkSeriesResult(),
		},
	}
	v7ModelBenchmarkStatus = status
	newV7ModelBenchmarkRunner = func(_ *inference.Manager, _ bool) v7modelbench.ModelBenchmarkRunner {
		return fakeRunner
	}
	t.Cleanup(func() {
		v7ModelBenchmarkStatus = oldStatus
		newV7ModelBenchmarkRunner = oldFactory
	})

	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Metadata map[string]any `json:"metadata"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode receipt: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		metadata := req.Metadata[v7modelbench.ModelBenchmarkSeriesTask].(map[string]any)
		if metadata["proof_status"] != v7modelbench.ModelBenchmarkSeriesProofStatusUnavailable {
			t.Errorf("proof_status = %v", metadata["proof_status"])
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		summary := metadata["summary"].(map[string]any)
		if summary["successful_measured_runs"] != float64(0) {
			t.Errorf("summary = %+v, want 0 successful measured runs", summary)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		trials := metadata["trials"].([]any)
		measured := trials[1].(map[string]any)
		if measured["output_hash"] != "" || measured["output_bytes"] != float64(0) {
			t.Errorf("unavailable measured trial exposed output fields: %+v", measured)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer ts.Close()

	handled, result, err := processOptionalV7ModelBenchmark(context.Background(), hub.New(ts.URL, pub, priv), &hub.WorkAssignment{
		JobID:    "job-modelbench-series-local",
		Kind:     "benchmark",
		SpecJSON: testModelBenchmarkSeriesSpecJSON(t, 1, 1),
	}, nil, nil, false)
	if err != nil {
		t.Fatalf("processOptionalV7ModelBenchmark() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if result == nil || result.ExitCode != 0 {
		t.Fatalf("result = %+v, want successful unavailable series receipt", result)
	}
	if fakeRunner.calls != 2 {
		t.Fatalf("runner calls = %d, want 2", fakeRunner.calls)
	}
	snapshot := status.Snapshot()
	if snapshot.Counters.Executed != 1 || snapshot.Counters.ReceiptSubmitted != 1 || snapshot.Counters.ReceiptFailed != 0 {
		t.Fatalf("unexpected counters: %+v", snapshot.Counters)
	}
}

func TestProcessOptionalV7ModelBenchmarkNormalJobDoesNotRecordStatus(t *testing.T) {
	oldStatus := v7ModelBenchmarkStatus
	status := v7modelbench.NewLocalStatus()
	v7ModelBenchmarkStatus = status
	t.Cleanup(func() {
		v7ModelBenchmarkStatus = oldStatus
	})
	t.Setenv(v7modelbench.ModelBenchmarkFlagEnv, "1")

	handled, result, err := processOptionalV7ModelBenchmark(context.Background(), nil, &hub.WorkAssignment{
		JobID:    "job-normal",
		Kind:     "native_report",
		SpecJSON: `{"task":"native_report","job_id":"job-normal"}`,
	}, nil, nil, false)
	if err != nil {
		t.Fatalf("processOptionalV7ModelBenchmark() error = %v", err)
	}
	if handled {
		t.Fatal("handled = true, want false")
	}
	if result != nil {
		t.Fatalf("result = %+v, want nil", result)
	}
	if snapshot := status.Snapshot(); snapshot.Counters != (v7modelbench.LocalStatusCounters{}) {
		t.Fatalf("status counters changed for normal job: %+v", snapshot.Counters)
	}
}

type fakeLlamaCppBackendBenchmarkRunner struct {
	snapshot v7llamacpp.BenchmarkStatusSnapshot
	configs  []v7llamacpp.BenchmarkConfig
	calls    int
}

func (f *fakeLlamaCppBackendBenchmarkRunner) Run(_ context.Context, config v7llamacpp.BenchmarkConfig) v7llamacpp.BenchmarkStatusSnapshot {
	f.calls++
	f.configs = append(f.configs, config)
	return f.snapshot
}

func testLlamaCppBackendBenchmarkSpecJSON(t *testing.T) string {
	t.Helper()
	spec := v7llamacpp.BackendBenchmarkSpec{
		Task:            v7llamacpp.BackendBenchmarkTask,
		RequestID:       "request-llamacpp-backend-local",
		JobID:           "job-llamacpp-backend-local",
		Backend:         v7llamacpp.BackendName,
		ModelID:         "tinyllama.Q4_K_M.gguf",
		MaxTokens:       16,
		WarmupRuns:      1,
		MeasuredRuns:    3,
		TimeoutMs:       30_000,
		CreatedAtUnixMs: 1_800_000_000_123,
	}
	encoded, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	return string(encoded)
}

func testLlamaCppBackendBenchmarkSnapshot(proofStatus string) v7llamacpp.BenchmarkStatusSnapshot {
	available := proofStatus == v7llamacpp.BenchmarkProofStatusMeasured
	outputHash := testSHA256ObjectID("secret llama output")
	outputBytes := int64(len("secret llama output"))
	status := v7llamacpp.BenchmarkStatusCompleted
	if !available {
		outputHash = ""
		outputBytes = 0
		status = v7llamacpp.BenchmarkStatusFailed
	}
	return v7llamacpp.BenchmarkStatusSnapshot{
		LastRunAt: time.Unix(1_800_000_001, 0),
		Status:    status,
		Metrics: v7llamacpp.BenchmarkMetrics{
			Available:      available,
			SidecarHealthy: available,
			ModelLoaded:    available,
			ModelID:        "tinyllama.Q4_K_M.gguf",
			PromptHash:     v7llamacpp.HashBenchmarkPrompt(),
			OutputHash:     outputHash,
			OutputBytes:    outputBytes,
			WarmupRuns:     1,
			MeasuredRuns:   3,
			P50TTFTMs:      100,
			P95TTFTMs:      200,
			P50TotalTimeMs: 800,
			P95TotalTimeMs: 1100,
			P50DecodeTPS:   20.5,
			P95DecodeTPS:   21.5,
			P50EndToEndTPS: 18.25,
			P95EndToEndTPS: 19.75,
			Backend:        v7llamacpp.BackendName,
			RuntimeKind:    v7llamacpp.BackendName,
			ProofStatus:    proofStatus,
			Streaming:      true,
		},
	}
}

type fakeBackendInferenceBenchmarkRunner struct {
	result v7inferencebench.BenchmarkExecutionResult
	err    error
	specs  []v7inferencebench.BenchmarkSpec
	calls  int
}

func (f *fakeBackendInferenceBenchmarkRunner) RunBackendInferenceBenchmark(_ context.Context, spec v7inferencebench.BenchmarkSpec) (v7inferencebench.BenchmarkExecutionResult, error) {
	f.calls++
	f.specs = append(f.specs, spec)
	if f.err != nil {
		return v7inferencebench.BenchmarkExecutionResult{}, f.err
	}
	return f.result, nil
}

func testBackendInferenceBenchmarkSpecJSON(t *testing.T) string {
	t.Helper()
	spec := v7inferencebench.BenchmarkSpec{
		Task:            v7inferencebench.BenchmarkTask,
		RequestID:       "request-backend-inference-local",
		BenchmarkID:     "benchmark-backend-inference-local",
		JobID:           "job-backend-inference-local",
		Backend:         v7llamacpp.BackendName,
		ModelID:         "tinyllama.Q4_K_M.gguf",
		TargetNodeID:    "node-backend-inference-local",
		PromptHash:      v7llamacpp.HashBenchmarkPrompt(),
		PromptProfileID: v7inferencebench.BenchmarkPromptProfileID,
		MaxTokens:       16,
		TimeoutMs:       30_000,
		CreatedAtUnixMs: 1_800_000_000_123,
	}
	encoded, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	return string(encoded)
}

func testBackendInferenceBenchmarkResult(proofStatus string) v7inferencebench.BenchmarkExecutionResult {
	spec := v7inferencebench.BenchmarkSpec{
		Task:            v7inferencebench.BenchmarkTask,
		RequestID:       "request-backend-inference-local",
		BenchmarkID:     "benchmark-backend-inference-local",
		JobID:           "job-backend-inference-local",
		Backend:         v7llamacpp.BackendName,
		ModelID:         "tinyllama.Q4_K_M.gguf",
		TargetNodeID:    "node-backend-inference-local",
		PromptHash:      v7llamacpp.HashBenchmarkPrompt(),
		PromptProfileID: v7inferencebench.BenchmarkPromptProfileID,
		MaxTokens:       16,
		TimeoutMs:       30_000,
		CreatedAtUnixMs: 1_800_000_000_123,
	}
	result := v7inferencebench.BenchmarkExecutionResult{
		Spec:            spec,
		Backend:         v7llamacpp.BackendName,
		ModelID:         "tinyllama.Q4_K_M.gguf",
		PromptHash:      v7llamacpp.HashBenchmarkPrompt(),
		OutputHash:      testSHA256ObjectID("secret measured output"),
		OutputBytes:     int64(len("secret measured output")),
		TokensGenerated: 12,
		TTFTMs:          100,
		P95TTFTMs:       100,
		TotalTimeMs:     700,
		DecodeTPS:       20,
		EndToEndTPS:     17.143,
		ProofStatus:     proofStatus,
	}
	if proofStatus != v7inferencebench.ProofStatusMeasured {
		result.OutputHash = ""
		result.OutputBytes = 0
		result.TokensGenerated = 0
		result.TTFTMs = 0
		result.TotalTimeMs = 0
		result.DecodeTPS = 0
		result.EndToEndTPS = 0
		result.ErrorCode = "backend_inference_failed"
	}
	return result
}

type fakeDashboardInferenceRunner struct {
	result v7dashboardinference.ExecutionResult
	err    error
	specs  []v7dashboardinference.Spec
	calls  int
}

func (f *fakeDashboardInferenceRunner) RunDashboardInference(_ context.Context, spec v7dashboardinference.Spec) (v7dashboardinference.ExecutionResult, error) {
	f.calls++
	f.specs = append(f.specs, spec)
	if f.err != nil {
		return v7dashboardinference.ExecutionResult{}, f.err
	}
	result := f.result
	if result.Spec.Task == "" {
		result.Spec = spec
	}
	return result, nil
}

type fakeDashboardInferenceProgressRunner struct {
	generated string
	calls     int
}

func (f *fakeDashboardInferenceProgressRunner) RunDashboardInference(ctx context.Context, spec v7dashboardinference.Spec) (v7dashboardinference.ExecutionResult, error) {
	return f.RunDashboardInferenceWithProgress(ctx, spec, nil)
}

func (f *fakeDashboardInferenceProgressRunner) RunDashboardInferenceWithProgress(ctx context.Context, spec v7dashboardinference.Spec, progress v7dashboardinference.ProgressSender) (v7dashboardinference.ExecutionResult, error) {
	f.calls++
	if progress != nil {
		if err := progress.SendDashboardInferenceProgress(ctx, v7dashboardinference.ProgressBatch{
			RunID:    spec.RunID,
			JobID:    spec.JobID,
			NodeID:   spec.TargetNodeID,
			SeqStart: 1,
			Chunks: []v7dashboardinference.ProgressChunk{
				{Seq: 1, Type: "delta", Text: "Ryvion"},
				{Seq: 2, Type: "delta", Text: " streams"},
			},
		}); err != nil {
			return v7dashboardinference.ExecutionResult{}, err
		}
	}
	output := []byte(f.generated)
	return v7dashboardinference.ExecutionResult{
		Spec:            spec,
		Backend:         spec.Backend,
		ModelID:         spec.ModelID,
		OutputHash:      v7dashboardinference.HashOutput(spec.JobID, output),
		OutputBytes:     int64(len(output)),
		TokensGenerated: 2,
		TTFTMs:          100,
		TotalTimeMs:     300,
		DecodeTPS:       10,
		EndToEndTPS:     6.667,
		ProofStatus:     v7dashboardinference.ProofStatusMeasured,
		GeneratedText:   f.generated,
	}, nil
}

func testDashboardInferenceSpecJSON(t *testing.T) string {
	t.Helper()
	spec := v7dashboardinference.Spec{
		Task:            v7dashboardinference.Task,
		RequestID:       "dashboardinfer_request",
		RunID:           "dashboardinfer_run",
		JobID:           "v7dashboardinfer_job",
		Backend:         v7llamacpp.BackendName,
		ModelID:         "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		TargetNodeID:    "node-dashboard-local",
		MaxTokens:       32,
		Stream:          true,
		CreatedAtUnixMs: 1_800_000_000_123,
	}
	encoded, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	return string(encoded)
}

func testDashboardInferenceTextSpecJSON(t *testing.T) string {
	t.Helper()
	spec := v7dashboardinference.Spec{
		Task:            v7dashboardinference.Task,
		RequestID:       "dashboardinfer_request",
		RunID:           "dashboardinfer_run",
		JobID:           "v7dashboardinfer_job",
		Backend:         v7llamacpp.BackendName,
		ModelID:         "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		TargetNodeID:    "node-dashboard-local",
		Prompt:          "private dashboard prompt",
		ReturnText:      true,
		MaxReturnChars:  8192,
		MaxTokens:       32,
		Stream:          true,
		CreatedAtUnixMs: 1_800_000_000_123,
	}
	encoded, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	return string(encoded)
}

func testDashboardInferenceResult(proofStatus string) v7dashboardinference.ExecutionResult {
	spec := v7dashboardinference.Spec{
		Task:            v7dashboardinference.Task,
		RequestID:       "dashboardinfer_request",
		RunID:           "dashboardinfer_run",
		JobID:           "v7dashboardinfer_job",
		Backend:         v7llamacpp.BackendName,
		ModelID:         "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		TargetNodeID:    "node-dashboard-local",
		MaxTokens:       32,
		Stream:          true,
		CreatedAtUnixMs: 1_800_000_000_123,
	}
	result := v7dashboardinference.ExecutionResult{
		Spec:            spec,
		Backend:         v7llamacpp.BackendName,
		ModelID:         "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		OutputHash:      testSHA256ObjectID("dashboard measured output"),
		OutputBytes:     int64(len("dashboard measured output")),
		TokensGenerated: 9,
		TTFTMs:          120,
		TotalTimeMs:     720,
		DecodeTPS:       15,
		EndToEndTPS:     12.5,
		ProofStatus:     proofStatus,
	}
	if proofStatus != v7dashboardinference.ProofStatusMeasured {
		result.OutputHash = v7dashboardinference.HashOutput(spec.JobID, nil)
		result.OutputBytes = 0
		result.TokensGenerated = 0
		result.TTFTMs = 0
		result.TotalTimeMs = 0
		result.DecodeTPS = 0
		result.EndToEndTPS = 0
		result.ErrorCode = "dashboard_inference_failed"
	}
	return result
}

type fakeModelBenchmarkRunner struct {
	result  v7modelbench.ModelBenchmarkResult
	results []v7modelbench.ModelBenchmarkResult
	err     error
	errs    []error
	calls   int
}

func (f *fakeModelBenchmarkRunner) RunModelBenchmark(_ context.Context, _ v7modelbench.ModelBenchmarkSpec) (v7modelbench.ModelBenchmarkResult, error) {
	index := f.calls
	f.calls++
	result := f.result
	if index < len(f.results) {
		result = f.results[index]
	}
	err := f.err
	if index < len(f.errs) {
		err = f.errs[index]
	}
	return result, err
}

func testModelBenchmarkSpecJSON(t *testing.T) string {
	t.Helper()
	spec := v7modelbench.ModelBenchmarkSpec{
		Task:            v7modelbench.ModelBenchmarkTask,
		RequestID:       "request-modelbench-local",
		JobID:           "job-modelbench-local",
		ModelID:         "ryvion-llama-3.2-3b",
		PromptLabel:     "fixed-readiness-smoke",
		PromptHash:      testSHA256ObjectID("prompt"),
		MaxTokens:       16,
		Temperature:     0.1,
		TimeoutMs:       30_000,
		CreatedAtUnixMs: 1_800_000_000_123,
	}
	encoded, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	return string(encoded)
}

func testMeasuredModelBenchmarkResult() v7modelbench.ModelBenchmarkResult {
	return v7modelbench.ModelBenchmarkResult{
		RequestID:  "request-modelbench-local",
		JobID:      "job-modelbench-local",
		ModelID:    "ryvion-llama-3.2-3b",
		PromptHash: testSHA256ObjectID("prompt"),
		RuntimeInfo: v7modelbench.ModelBenchmarkRuntimeInfo{
			AgentVersion:             "test",
			OS:                       "darwin",
			Arch:                     "arm64",
			NativeInferenceSupported: true,
			NativeInferenceReady:     true,
			RuntimeKind:              v7modelbench.ModelBenchmarkRuntimeKindNativeLocal,
			ModelID:                  "ryvion-llama-3.2-3b",
			ModelLoaded:              true,
			GPUDetected:              true,
			GPUModel:                 "test-gpu",
		},
		Metrics: v7modelbench.ModelBenchmarkMetrics{
			StartedAtUnixMs:    1_800_000_000_123,
			FinishedAtUnixMs:   1_800_000_001_357,
			WallTimeMs:         1234,
			TimeToFirstTokenMs: 200,
			TokensGenerated:    16,
			TokensPerSecond:    12.3,
			ModelLoadState:     v7modelbench.ModelBenchmarkModelLoadStateLoaded,
		},
		OutputHash:  testSHA256ObjectID("secret completion text"),
		OutputBytes: int64(len("secret completion text")),
		ProofStatus: v7modelbench.ModelBenchmarkProofStatusMeasured,
	}
}

func testUnavailableModelBenchmarkResult() v7modelbench.ModelBenchmarkResult {
	result := testMeasuredModelBenchmarkResult()
	result.RuntimeInfo.NativeInferenceReady = false
	result.RuntimeInfo.ModelLoaded = false
	result.Metrics.TimeToFirstTokenMs = 0
	result.Metrics.TokensGenerated = 0
	result.Metrics.TokensPerSecond = 0
	result.Metrics.ModelLoadState = v7modelbench.ModelBenchmarkModelLoadStateUnavailable
	result.Metrics.ErrorCode = "native_model_unavailable"
	result.OutputHash = ""
	result.OutputBytes = 0
	result.ProofStatus = v7modelbench.ModelBenchmarkProofStatusUnavailable
	return result
}

func testModelBenchmarkSeriesSpecJSON(t *testing.T, warmupRuns int, measuredRuns int) string {
	t.Helper()
	profile, err := v7modelbench.GetBenchmarkPromptProfile(string(v7modelbench.BenchmarkPromptProfileLongDecodeProbe))
	if err != nil {
		t.Fatalf("GetBenchmarkPromptProfile() error = %v", err)
	}
	spec := map[string]any{
		"task":               v7modelbench.ModelBenchmarkSeriesTask,
		"request_id":         "request-modelbench-series-local",
		"job_id":             "job-modelbench-series-local",
		"model_id":           "ryvion-llama-3.2-3b",
		"prompt_profile_id":  string(profile.ID),
		"prompt_hash":        v7modelbench.BenchmarkPromptHash(profile),
		"max_tokens":         32,
		"timeout_ms":         30_000,
		"warmup_runs":        warmupRuns,
		"measured_runs":      measuredRuns,
		"created_at_unix_ms": int64(1_800_000_000_123),
	}
	encoded, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	return string(encoded)
}

func testMeasuredModelBenchmarkSeriesResult(ttftMs int64, wallMs int64, tokens int64, tps float64) v7modelbench.ModelBenchmarkResult {
	profile, _ := v7modelbench.GetBenchmarkPromptProfile(string(v7modelbench.BenchmarkPromptProfileLongDecodeProbe))
	result := testMeasuredModelBenchmarkResult()
	result.RequestID = "request-modelbench-series-local"
	result.JobID = "job-modelbench-series-local"
	result.PromptHash = v7modelbench.BenchmarkPromptHash(profile)
	result.Metrics.TimeToFirstTokenMs = ttftMs
	result.Metrics.WallTimeMs = wallMs
	result.Metrics.TokensGenerated = tokens
	result.Metrics.TokensPerSecond = tps
	result.OutputHash = testSHA256ObjectID(fmt.Sprintf("secret series completion text %d", ttftMs))
	result.OutputBytes = int64(len("secret series completion text"))
	return result
}

func testUnavailableModelBenchmarkSeriesResult() v7modelbench.ModelBenchmarkResult {
	result := testMeasuredModelBenchmarkSeriesResult(0, 25, 0, 0)
	result.RuntimeInfo.NativeInferenceReady = false
	result.RuntimeInfo.ModelLoaded = false
	result.Metrics.TimeToFirstTokenMs = 0
	result.Metrics.TokensGenerated = 0
	result.Metrics.TokensPerSecond = 0
	result.Metrics.ModelLoadState = v7modelbench.ModelBenchmarkModelLoadStateUnavailable
	result.Metrics.ErrorCode = "native_runtime_unavailable"
	result.OutputHash = ""
	result.OutputBytes = 0
	result.ProofStatus = v7modelbench.ModelBenchmarkProofStatusUnavailable
	return result
}

func testSHA256ObjectID(value string) string {
	sum := sha256.Sum256([]byte(value))
	return "sha256:" + hex.EncodeToString(sum[:])
}

func containsToken(tokens []string, want string) bool {
	for _, token := range tokens {
		if token == want {
			return true
		}
	}
	return false
}

func countMainWorkLoopEvents(events []diagnostics.WorkLoopEvent, name string) int {
	count := 0
	for _, event := range events {
		if event.Name == name {
			count++
		}
	}
	return count
}

func findMainWorkLoopEvent(events []diagnostics.WorkLoopEvent, name string) diagnostics.WorkLoopEvent {
	for _, event := range events {
		if event.Name == name {
			return event
		}
	}
	return diagnostics.WorkLoopEvent{}
}

func mainWorkLoopEventOrder(events []diagnostics.WorkLoopEvent, names ...string) bool {
	next := 0
	for _, event := range events {
		if next < len(names) && event.Name == names[next] {
			next++
		}
	}
	return next == len(names)
}

func assertMainWorkLoopEventsChronological(t *testing.T, events []diagnostics.WorkLoopEvent) {
	t.Helper()
	var previous time.Time
	for i, event := range events {
		parsed, err := time.Parse(time.RFC3339Nano, event.At)
		if err != nil {
			t.Fatalf("event[%d] time parse error: %v; event=%+v", i, err, event)
		}
		if !previous.IsZero() && parsed.Before(previous) {
			t.Fatalf("event[%d] time %s before previous %s; events=%+v", i, event.At, previous.Format(time.RFC3339Nano), events)
		}
		previous = parsed
	}
}

func testV7TensorPlaneBenchmarkSpecJSON(t *testing.T, jobID, requestID string) string {
	t.Helper()
	spec := map[string]any{
		"task":               v7tensorplane.BenchmarkTask,
		"request_id":         requestID,
		"job_id":             jobID,
		"model_id":           "model-fixture",
		"layer_index":        2,
		"dtype":              string(v7tensorplane.TensorDTypeFloat32),
		"tokens":             8,
		"head_dim":           4,
		"value_dim":          3,
		"seed":               int64(99),
		"timeout_ms":         int64(5_000),
		"created_at_unix_ms": int64(1_800_000_000_123),
		"prompt_text":        "prompt text must not leak",
		"generated_output":   "generated output must not leak",
	}
	specJSON, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	return string(specJSON)
}

func testV7MemoryBenchmarkSpecJSON(t *testing.T, jobID, requestID string) string {
	t.Helper()
	spec := map[string]any{
		"task":               v7memorybench.BenchmarkTask,
		"request_id":         requestID,
		"job_id":             jobID,
		"shard_id":           "shard-a",
		"seed":               int64(7),
		"token_count":        4,
		"value_dim":          2,
		"created_at_unix_ms": int64(1_800_000_000_123),
		"weighted_value":     []float64{1, 2, 3},
	}
	specJSON, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	return string(specJSON)
}

// submitReceiptWithRetryTestable is the same logic as submitReceiptWithRetry
// but accepts an interface so we can inject a fake client.
type receiptSubmitter interface {
	SubmitReceipt(ctx context.Context, receipt hub.Receipt) error
}

func submitReceiptWithRetryTestable(ctx context.Context, client receiptSubmitter, receipt hub.Receipt) error {
	receipt = prepareReceiptForSubmission(receipt)
	const maxAttempts = 5
	delay := 2 * time.Millisecond // fast for tests
	var lastErr error
	for i := 0; i < maxAttempts; i++ {
		if err := client.SubmitReceipt(ctx, receipt); err != nil {
			lastErr = err
			select {
			case <-ctx.Done():
				return fmt.Errorf("context cancelled during receipt retry: %w", lastErr)
			case <-time.After(delay):
			}
			delay = time.Duration(float64(delay) * 2)
			if delay > 30*time.Millisecond {
				delay = 30 * time.Millisecond
			}
			continue
		}
		return nil
	}
	return fmt.Errorf("receipt submission failed after %d attempts: %w", maxAttempts, lastErr)
}
