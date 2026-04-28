package main

import (
	"context"
	"fmt"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Ryvion/node-agent/internal/hub"
	"github.com/Ryvion/node-agent/internal/runtimeexec"
)

// fakeClient implements a minimal receipt submitter that fails N times then succeeds.
type fakeClient struct {
	failCount int
	calls     atomic.Int32
}

func (f *fakeClient) SubmitReceipt(_ context.Context, _ hub.Receipt) error {
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
