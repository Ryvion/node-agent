package modelwarm

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/Ryvion/node-agent/internal/v7/llamacpp"
	"github.com/Ryvion/node-agent/internal/v7/modelcache"
	"github.com/Ryvion/node-agent/internal/v7/modelpolicy"
)

func TestExecuteWarmAssignmentSwitchesLlamaCppModelAndBuildsReceipt(t *testing.T) {
	t.Parallel()

	cacheDir := t.TempDir()
	phiPath := filepath.Join(cacheDir, "phi-4-Q4_K_M.gguf")
	if err := os.WriteFile(phiPath, []byte("gguf"), 0o644); err != nil {
		t.Fatalf("write cached model: %v", err)
	}
	manager := &fakeWarmManager{
		status: llamacpp.LlamaCppSidecarStatus{
			Enabled:       true,
			Available:     true,
			Running:       true,
			Healthy:       true,
			ModelPath:     filepath.Join(cacheDir, "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
			ModelFilename: "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
			Backend:       llamacpp.BackendName,
		},
	}
	runner := &fakeWarmBenchmarkRunner{
		snapshot: testWarmBenchmarkSnapshot(),
	}

	receipt, handled, err := ExecuteWarmAssignment(context.Background(), testWarmSpecJSON(t, testWarmSpec()), ExecuteOptions{
		Getenv:          testWarmGetenv,
		Policy:          testWarmPolicy(cacheDir),
		LlamaCppManager: manager,
		BenchmarkRunner: runner,
	})
	if err != nil {
		t.Fatalf("ExecuteWarmAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if manager.restartCalls != 1 || manager.modelPath != phiPath {
		t.Fatalf("manager restart calls/path = %d/%q, want one restart to %q", manager.restartCalls, manager.modelPath, phiPath)
	}
	if !manager.enabled {
		t.Fatal("manager enabled = false, want warm execution to enable sidecar")
	}
	if runner.calls != 1 {
		t.Fatalf("benchmark calls = %d, want 1", runner.calls)
	}
	metadata := warmMetadata(t, receipt)
	if metadata["warm_id"] != "warm-local" ||
		metadata["model_id"] != "phi-4-Q4_K_M.gguf" ||
		metadata["backend"] != "llama.cpp" ||
		metadata["model_path"] != phiPath ||
		metadata["warm"] != true ||
		metadata["proof_status"] != ProofStatusModelWarmed {
		t.Fatalf("receipt metadata = %+v", metadata)
	}
	if _, ok := metadata["benchmark"].(map[string]any); !ok {
		t.Fatalf("benchmark metadata missing: %+v", metadata)
	}
	if !WarmReceiptJSONContainsNoUnsafeFields(receipt) {
		raw, _ := json.Marshal(receipt.Metadata)
		t.Fatalf("receipt leaked unsafe fields: %s", raw)
	}
}

func TestExecuteWarmAssignmentWaitsForAsyncWarmSidecar(t *testing.T) {
	t.Parallel()

	cacheDir := t.TempDir()
	phiPath := filepath.Join(cacheDir, "phi-4-Q4_K_M.gguf")
	if err := os.WriteFile(phiPath, []byte("gguf"), 0o644); err != nil {
		t.Fatalf("write cached model: %v", err)
	}
	manager := &asyncWarmManager{
		status: llamacpp.LlamaCppSidecarStatus{
			Enabled:       true,
			Available:     true,
			Running:       true,
			Healthy:       true,
			ModelPath:     filepath.Join(cacheDir, "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
			ModelFilename: "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
			Backend:       llamacpp.BackendName,
		},
	}

	receipt, handled, err := ExecuteWarmAssignment(context.Background(), testWarmSpecJSON(t, testWarmSpec()), ExecuteOptions{
		Getenv:          testWarmGetenv,
		Policy:          testWarmPolicy(cacheDir),
		LlamaCppManager: manager,
		BenchmarkRunner: &fakeWarmBenchmarkRunner{snapshot: testWarmBenchmarkSnapshot()},
	})
	if err != nil {
		t.Fatalf("ExecuteWarmAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if manager.restartCalls != 1 || manager.statusCalls != 1 {
		t.Fatalf("manager calls restart/status-after-restart = %d/%d, want 1/1", manager.restartCalls, manager.statusCalls)
	}
	metadata := warmMetadata(t, receipt)
	if metadata["warm"] != true || metadata["proof_status"] != ProofStatusModelWarmed {
		t.Fatalf("receipt metadata = %+v, want warmed receipt", metadata)
	}
}

func TestExecuteWarmAssignmentWaitsForAlreadyLoadingRequestedModel(t *testing.T) {
	t.Parallel()

	cacheDir := t.TempDir()
	phiPath := filepath.Join(cacheDir, "phi-4-Q4_K_M.gguf")
	if err := os.WriteFile(phiPath, []byte("gguf"), 0o644); err != nil {
		t.Fatalf("write cached model: %v", err)
	}
	manager := &pollingWarmManager{
		statuses: []llamacpp.LlamaCppSidecarStatus{{
			Enabled:       true,
			Available:     true,
			Running:       true,
			Healthy:       false,
			ModelPath:     phiPath,
			ModelFilename: filepath.Base(phiPath),
			Backend:       llamacpp.BackendName,
		}, {
			Enabled:       true,
			Available:     true,
			Running:       true,
			Healthy:       true,
			ModelPath:     phiPath,
			ModelFilename: filepath.Base(phiPath),
			Backend:       llamacpp.BackendName,
		}},
	}

	receipt, handled, err := ExecuteWarmAssignment(context.Background(), testWarmSpecJSON(t, testWarmSpec()), ExecuteOptions{
		Getenv:          testWarmGetenv,
		Policy:          testWarmPolicy(cacheDir),
		LlamaCppManager: manager,
		BenchmarkRunner: &fakeWarmBenchmarkRunner{snapshot: testWarmBenchmarkSnapshot()},
	})
	if err != nil {
		t.Fatalf("ExecuteWarmAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if manager.restartCalls != 0 || manager.statusCalls != 2 {
		t.Fatalf("manager calls restart/status = %d/%d, want no restart and status polling", manager.restartCalls, manager.statusCalls)
	}
	metadata := warmMetadata(t, receipt)
	if metadata["warm"] != true || metadata["proof_status"] != ProofStatusModelWarmed {
		t.Fatalf("receipt metadata = %+v, want warmed receipt", metadata)
	}
}

func TestExecuteWarmAssignmentRestartsAttachedWarmServer(t *testing.T) {
	t.Parallel()

	cacheDir := t.TempDir()
	phiPath := filepath.Join(cacheDir, "phi-4-Q4_K_M.gguf")
	if err := os.WriteFile(phiPath, []byte("gguf"), 0o644); err != nil {
		t.Fatalf("write cached model: %v", err)
	}
	manager := &fakeWarmManager{
		status: llamacpp.LlamaCppSidecarStatus{
			Enabled:       true,
			Available:     true,
			Running:       true,
			Healthy:       true,
			Attached:      true,
			ModelPath:     phiPath,
			ModelFilename: filepath.Base(phiPath),
			Backend:       llamacpp.BackendName,
		},
	}

	receipt, handled, err := ExecuteWarmAssignment(context.Background(), testWarmSpecJSON(t, testWarmSpec()), ExecuteOptions{
		Getenv:          testWarmGetenv,
		Policy:          testWarmPolicy(cacheDir),
		LlamaCppManager: manager,
		BenchmarkRunner: &fakeWarmBenchmarkRunner{snapshot: testWarmBenchmarkSnapshot()},
	})
	if err != nil {
		t.Fatalf("ExecuteWarmAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if manager.restartCalls != 1 || manager.modelPath != phiPath {
		t.Fatalf("manager restart calls/path = %d/%q, want restart of attached warm server to %q", manager.restartCalls, manager.modelPath, phiPath)
	}
	metadata := warmMetadata(t, receipt)
	if metadata["warm"] != true || metadata["proof_status"] != ProofStatusModelWarmed {
		t.Fatalf("receipt metadata = %+v, want warmed receipt", metadata)
	}
}

func TestExecuteWarmAssignmentMissingModelReturnsSafeReceipt(t *testing.T) {
	t.Parallel()

	receipt, handled, err := ExecuteWarmAssignment(context.Background(), testWarmSpecJSON(t, testWarmSpec()), ExecuteOptions{
		Getenv: testWarmGetenv,
		Policy: testWarmPolicy(t.TempDir()),
	})
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if err == nil {
		t.Fatal("error = nil, want model_not_found")
	}
	metadata := warmMetadata(t, receipt)
	if metadata["proof_status"] != ProofStatusModelWarmFailed || metadata["error_code"] != "model_not_found" || metadata["warm"] != false {
		t.Fatalf("receipt metadata = %+v", metadata)
	}
	if receipt.MeteringUnits != 0 || receipt.ResultHashHex == "" {
		t.Fatalf("receipt envelope = %+v, want rejection hash and zero units", receipt)
	}
}

func TestExecuteWarmAssignmentUsesSpecModelPathWhenCacheScanMisses(t *testing.T) {
	t.Parallel()

	modelDir := t.TempDir()
	phiPath := filepath.Join(modelDir, "phi-4-Q4_K_M.gguf")
	if err := os.WriteFile(phiPath, []byte("gguf-path"), 0o644); err != nil {
		t.Fatalf("write phi fixture: %v", err)
	}
	spec := testWarmSpec()
	spec.ModelPath = phiPath
	manager := &fakeWarmManager{
		status: llamacpp.LlamaCppSidecarStatus{
			Enabled:       true,
			Available:     true,
			Running:       true,
			Healthy:       true,
			ModelPath:     filepath.Join(t.TempDir(), "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
			ModelFilename: "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
			Backend:       llamacpp.BackendName,
		},
	}

	receipt, handled, err := ExecuteWarmAssignment(context.Background(), testWarmSpecJSON(t, spec), ExecuteOptions{
		Getenv:          testWarmGetenv,
		Policy:          testWarmPolicy(t.TempDir()),
		LlamaCppManager: manager,
		BenchmarkRunner: &fakeWarmBenchmarkRunner{snapshot: testWarmBenchmarkSnapshot()},
	})
	if err != nil || !handled {
		t.Fatalf("ExecuteWarmAssignment() handled=%v error=%v, want spec-path success", handled, err)
	}
	if manager.restartCalls != 1 || manager.modelPath != phiPath {
		t.Fatalf("manager restart calls/path = %d/%q, want one restart to %q", manager.restartCalls, manager.modelPath, phiPath)
	}
	metadata := warmMetadata(t, receipt)
	if metadata["model_path"] != phiPath || metadata["warm"] != true {
		t.Fatalf("receipt metadata = %+v, want warmed spec model path", metadata)
	}
}

func TestFindCachedModelMatchesGemmaQATForLegacyCatalogName(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "gemma-3-27b-it-q4_0.gguf")
	status := modelcache.Status{
		CacheDir: filepath.Dir(path),
		Models: []modelcache.Model{{
			ModelID:   "gemma-3-27b-it-q4_0.gguf",
			Filename:  "gemma-3-27b-it-q4_0.gguf",
			Path:      path,
			SizeBytes: 18 * 1024 * 1024 * 1024,
			Format:    "gguf",
			Installed: true,
		}},
	}

	model, ok := findCachedModel(status, "gemma-3-27b-it-Q4_K_M.gguf")
	if !ok || model.Path != path {
		t.Fatalf("findCachedModel() = %+v/%v, want local Gemma QAT artifact", model, ok)
	}
}

func TestFindCachedModelMatchesGemma4QATForCatalogName(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "gemma-4-27b-it-q4_0.gguf")
	status := modelcache.Status{
		CacheDir: filepath.Dir(path),
		Models: []modelcache.Model{{
			ModelID:   "gemma-4-27b-it-q4_0.gguf",
			Filename:  "gemma-4-27b-it-q4_0.gguf",
			Path:      path,
			SizeBytes: 18 * 1024 * 1024 * 1024,
			Format:    "gguf",
			Installed: true,
		}},
	}

	model, ok := findCachedModel(status, "gemma-4-27b-it-Q4_K_M.gguf")
	if !ok || model.Path != path {
		t.Fatalf("findCachedModel() = %+v/%v, want local Gemma 4 QAT artifact", model, ok)
	}
}

func TestExecuteWarmAssignmentFlagOffBuildsDisabledReceipt(t *testing.T) {
	t.Parallel()

	receipt, handled, err := ExecuteWarmAssignment(context.Background(), testWarmSpecJSON(t, testWarmSpec()), ExecuteOptions{
		Getenv: func(key string) string {
			if key == WarmDisableEnv {
				return "1"
			}
			return ""
		},
	})
	if !handled {
		t.Fatal("handled = false, want true for explicit warm task")
	}
	if err == nil {
		t.Fatal("error = nil, want disabled")
	}
	metadata := warmMetadata(t, receipt)
	if metadata["error_code"] != "model_warm_disabled" || metadata["proof_status"] != ProofStatusModelWarmFailed {
		t.Fatalf("receipt metadata = %+v", metadata)
	}
}

func TestWarmReceiptRejectsUnsafeBenchmarkFields(t *testing.T) {
	t.Parallel()

	receipt, err := BuildWarmReceipt(WarmExecutionResult{
		Spec:           testWarmSpec(),
		ModelPath:      filepath.Join(t.TempDir(), "phi-4-Q4_K_M.gguf"),
		ModelSizeBytes: 4,
		Warm:           true,
		Benchmark: &llamacpp.BenchmarkStatusSnapshot{
			Status: llamacpp.BenchmarkStatusCompleted,
			Metrics: llamacpp.BenchmarkMetrics{
				SidecarHealthy:   true,
				ModelLoaded:      true,
				WarmupRuns:       1,
				MeasuredRuns:     1,
				TokensGenerated:  5,
				P50TTFTMs:        100,
				P50DecodeTPS:     10.5,
				P50EndToEndTPS:   8.5,
				PromptHash:       llamacpp.HashBenchmarkPrompt(),
				OutputHash:       hashWarmBytes([]byte("raw output should stay hashed only")),
				OutputBytes:      int64(len("raw output should stay hashed only")),
				ProofStatus:      llamacpp.BenchmarkProofStatusMeasured,
				Backend:          llamacpp.BackendName,
				RuntimeKind:      llamacpp.BackendName,
				CompletionTokens: 5,
			},
		},
	})
	if err != nil {
		t.Fatalf("BuildWarmReceipt() error = %v", err)
	}
	if !WarmReceiptJSONContainsNoUnsafeFields(receipt) {
		raw, _ := json.Marshal(receipt.Metadata)
		t.Fatalf("receipt leaked unsafe fields: %s", raw)
	}
}

func testWarmSpec() WarmSpec {
	return WarmSpec{
		Task:                  WarmTask,
		RequestID:             "request-local",
		WarmID:                "warm-local",
		JobID:                 "job-local",
		ModelID:               "phi-4-Q4_K_M.gguf",
		Backend:               "llama.cpp",
		RunBenchmarkAfterWarm: true,
		TimeoutMs:             30_000,
	}
}

func testWarmSpecJSON(t *testing.T, spec WarmSpec) string {
	t.Helper()
	encoded, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	return string(encoded)
}

func testWarmGetenv(name string) string {
	if name == WarmFlagEnv {
		return "1"
	}
	return ""
}

func testWarmPolicy(cacheDir string) modelpolicy.Policy {
	return modelpolicy.Policy{
		AutoDownload:        false,
		MaxSingleModelBytes: 16 << 30,
		MaxCacheBytes:       64 << 30,
		CacheDir:            cacheDir,
		AllowedFamilies:     []string{"llama", "phi"},
		AllowedFormats:      []string{"gguf"},
	}
}

func warmMetadata(t *testing.T, receipt WarmReceipt) map[string]any {
	t.Helper()
	metadata, ok := receipt.Metadata[WarmTask].(map[string]any)
	if !ok {
		t.Fatalf("receipt metadata missing %q: %+v", WarmTask, receipt.Metadata)
	}
	return metadata
}

type fakeWarmManager struct {
	status       llamacpp.LlamaCppSidecarStatus
	enabled      bool
	restartCalls int
	modelPath    string
}

func (f *fakeWarmManager) SetEnabled(enabled bool) llamacpp.LlamaCppSidecarConfig {
	f.enabled = enabled
	f.status.Enabled = enabled
	return llamacpp.LlamaCppSidecarConfig{Enabled: enabled, ModelPath: f.status.ModelPath}
}

func (f *fakeWarmManager) Status(context.Context) llamacpp.LlamaCppSidecarStatus {
	return f.status
}

func (f *fakeWarmManager) RestartWithModel(_ context.Context, modelPath string) llamacpp.LlamaCppSidecarStatus {
	f.restartCalls++
	f.modelPath = modelPath
	f.status = llamacpp.LlamaCppSidecarStatus{
		Enabled:       true,
		Available:     true,
		Running:       true,
		Healthy:       true,
		ModelPath:     modelPath,
		ModelFilename: filepath.Base(modelPath),
		Backend:       llamacpp.BackendName,
	}
	return f.status
}

type asyncWarmManager struct {
	status       llamacpp.LlamaCppSidecarStatus
	enabled      bool
	restartCalls int
	statusCalls  int
	pendingPath  string
}

func (f *asyncWarmManager) SetEnabled(enabled bool) llamacpp.LlamaCppSidecarConfig {
	f.enabled = enabled
	f.status.Enabled = enabled
	return llamacpp.LlamaCppSidecarConfig{Enabled: enabled, ModelPath: f.status.ModelPath}
}

func (f *asyncWarmManager) Status(context.Context) llamacpp.LlamaCppSidecarStatus {
	if f.pendingPath != "" {
		f.statusCalls++
		f.status = llamacpp.LlamaCppSidecarStatus{
			Enabled:       true,
			Available:     true,
			Running:       true,
			Healthy:       true,
			ModelPath:     f.pendingPath,
			ModelFilename: filepath.Base(f.pendingPath),
			Backend:       llamacpp.BackendName,
		}
		f.pendingPath = ""
	}
	return f.status
}

func (f *asyncWarmManager) RestartWithModel(_ context.Context, modelPath string) llamacpp.LlamaCppSidecarStatus {
	f.restartCalls++
	f.pendingPath = modelPath
	f.status = llamacpp.LlamaCppSidecarStatus{
		Enabled:       true,
		Available:     true,
		Running:       true,
		Healthy:       false,
		ModelPath:     modelPath,
		ModelFilename: filepath.Base(modelPath),
		Backend:       llamacpp.BackendName,
	}
	return f.status
}

type pollingWarmManager struct {
	statuses     []llamacpp.LlamaCppSidecarStatus
	enabled      bool
	restartCalls int
	statusCalls  int
}

func (f *pollingWarmManager) SetEnabled(enabled bool) llamacpp.LlamaCppSidecarConfig {
	f.enabled = enabled
	for i := range f.statuses {
		f.statuses[i].Enabled = enabled
	}
	return llamacpp.LlamaCppSidecarConfig{Enabled: enabled}
}

func (f *pollingWarmManager) Status(context.Context) llamacpp.LlamaCppSidecarStatus {
	f.statusCalls++
	if len(f.statuses) == 0 {
		return llamacpp.LlamaCppSidecarStatus{}
	}
	if f.statusCalls <= len(f.statuses) {
		return f.statuses[f.statusCalls-1]
	}
	return f.statuses[len(f.statuses)-1]
}

func (f *pollingWarmManager) RestartWithModel(_ context.Context, modelPath string) llamacpp.LlamaCppSidecarStatus {
	f.restartCalls++
	return llamacpp.LlamaCppSidecarStatus{
		Enabled:       true,
		Available:     true,
		Running:       true,
		Healthy:       true,
		ModelPath:     modelPath,
		ModelFilename: filepath.Base(modelPath),
		Backend:       llamacpp.BackendName,
	}
}

type fakeWarmBenchmarkRunner struct {
	calls    int
	snapshot llamacpp.BenchmarkStatusSnapshot
}

func (f *fakeWarmBenchmarkRunner) Run(_ context.Context, _ llamacpp.BenchmarkConfig) llamacpp.BenchmarkStatusSnapshot {
	f.calls++
	return f.snapshot
}

func testWarmBenchmarkSnapshot() llamacpp.BenchmarkStatusSnapshot {
	return llamacpp.BenchmarkStatusSnapshot{
		Status: llamacpp.BenchmarkStatusCompleted,
		Metrics: llamacpp.BenchmarkMetrics{
			Available:       true,
			SidecarHealthy:  true,
			ModelLoaded:     true,
			ModelID:         "phi-4-Q4_K_M.gguf",
			PromptHash:      llamacpp.HashBenchmarkPrompt(),
			OutputHash:      hashWarmBytes([]byte("benchmark output")),
			OutputBytes:     int64(len("benchmark output")),
			WarmupRuns:      1,
			MeasuredRuns:    1,
			P50TTFTMs:       100,
			P95TTFTMs:       110,
			P50TotalTimeMs:  500,
			P95TotalTimeMs:  550,
			P50DecodeTPS:    20,
			P50EndToEndTPS:  18,
			TokensGenerated: 8,
			Backend:         llamacpp.BackendName,
			RuntimeKind:     llamacpp.BackendName,
			ProofStatus:     llamacpp.BenchmarkProofStatusMeasured,
		},
	}
}

func hashWarmBytes(body []byte) string {
	sum := sha256.Sum256(body)
	return "sha256:" + hex.EncodeToString(sum[:])
}
