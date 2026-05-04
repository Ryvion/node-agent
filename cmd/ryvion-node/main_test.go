package main

import (
	"context"
	"crypto/ed25519"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Ryvion/node-agent/internal/hub"
	"github.com/Ryvion/node-agent/internal/hw"
	"github.com/Ryvion/node-agent/internal/runtimeexec"
	v7memorybench "github.com/Ryvion/node-agent/internal/v7/memorybench"
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

func TestBuildOptionalV7HeartbeatPayloadHonorsEnvFlag(t *testing.T) {
	t.Setenv("RYV_DISABLE_OCI", "1")
	caps := hw.CapSet{
		CPUCores: 4,
		RAMBytes: 8 * 1024 * 1024 * 1024,
	}
	runtimeMgr := newRuntimeManager("test", runtimeContractMetadata{})

	t.Setenv("RYV_NODE_V7_CAPS", "")
	if payload := buildOptionalV7HeartbeatPayload("pubkey", caps, "cpu", "ca", nil, runtimeMgr); payload != nil {
		t.Fatalf("payload = %+v, want nil when RYV_NODE_V7_CAPS is off", payload)
	}

	t.Setenv("RYV_NODE_V7_CAPS", "1")
	payload := buildOptionalV7HeartbeatPayload("pubkey", caps, "cpu", "ca", nil, runtimeMgr)
	if payload == nil {
		t.Fatal("payload = nil, want V7 payload when RYV_NODE_V7_CAPS=1")
	}
	if payload.CapabilityPassport.NodePublicKey != "pubkey" {
		t.Fatalf("node public key = %q, want pubkey", payload.CapabilityPassport.NodePublicKey)
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
	t.Setenv("RYV_NODE_V7_CAPS", "")
	client, gotV7, calls := heartbeatTestClient(t)

	ok := sendHeartbeat(context.Background(), client, validHeartbeatCaps(), "cpu", "ca", nil, nil)
	if !ok {
		t.Fatal("sendHeartbeat() = false, want true")
	}
	if calls.Load() != 1 {
		t.Fatalf("heartbeat calls = %d, want 1", calls.Load())
	}
	if gotV7.Load() {
		t.Fatal("heartbeat sent V7 payload with RYV_NODE_V7_CAPS off")
	}
}

func TestSendHeartbeatIncludesV7WhenFlagOn(t *testing.T) {
	t.Setenv("RYV_NODE_V7_CAPS", "1")
	client, gotV7, calls := heartbeatTestClient(t)

	ok := sendHeartbeat(context.Background(), client, validHeartbeatCaps(), "cpu", "ca", nil, nil)
	if !ok {
		t.Fatal("sendHeartbeat() = false, want true")
	}
	if calls.Load() != 1 {
		t.Fatalf("heartbeat calls = %d, want 1", calls.Load())
	}
	if !gotV7.Load() {
		t.Fatal("heartbeat did not send V7 payload with RYV_NODE_V7_CAPS=1")
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
	status := v7memorybench.NewLocalStatus()
	operatorRuntimeState = &operatorRuntime{v7MemoryBenchmark: status}
	t.Cleanup(func() {
		operatorRuntimeState = oldState
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

func containsToken(tokens []string, want string) bool {
	for _, token := range tokens {
		if token == want {
			return true
		}
	}
	return false
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
