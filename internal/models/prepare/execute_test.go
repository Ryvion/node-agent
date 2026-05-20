package modelprepare

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/Ryvion/ryvion-node/internal/models/cache"
	"github.com/Ryvion/ryvion-node/internal/models/policy"
	"github.com/Ryvion/ryvion-node/internal/runtimes/llamacpp"
)

func TestExecutePrepareAssignmentPolicyBlocksDownload(t *testing.T) {
	t.Parallel()

	var calls atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("gguf"))
	}))
	defer ts.Close()

	specJSON := testPrepareSpecJSON(t, testPrepareSpec(ts.URL, []byte("gguf"), t.TempDir()))
	receipt, handled, err := ExecutePrepareAssignment(context.Background(), specJSON, ExecuteOptions{
		Getenv: func(name string) string {
			if name == PrepareFlagEnv {
				return "1"
			}
			return ""
		},
		Policy: modelpolicy.Policy{
			AutoDownload:        false,
			MaxSingleModelBytes: 1024,
			MaxCacheBytes:       4096,
			CacheDir:            t.TempDir(),
			AllowedFamilies:     []string{"llama"},
			AllowedFormats:      []string{"gguf"},
		},
		AllowInsecureHTTP: true,
	})
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if err == nil {
		t.Fatal("error = nil, want policy block")
	}
	if got := calls.Load(); got != 0 {
		t.Fatalf("download calls = %d, want 0", got)
	}
	metadata := prepareMetadata(t, receipt)
	if metadata["proof_status"] != ProofStatusPrepareFailed || metadata["error_code"] != modelpolicy.PrepareDecisionAutoDownloadDisabled {
		t.Fatalf("receipt metadata = %+v", metadata)
	}
}

func TestExecutePrepareAssignmentDownloadsVerifiesAndInstallsFromHTTP(t *testing.T) {
	t.Parallel()

	body := []byte("gguf-model-body")
	cacheDir := t.TempDir()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/octet-stream")
		_, _ = w.Write(body)
	}))
	defer ts.Close()

	spec := testPrepareSpec(ts.URL+"/TinyLlama.Q4_K_M.gguf", body, cacheDir)
	receipt, handled, err := ExecutePrepareAssignment(context.Background(), testPrepareSpecJSON(t, spec), testPrepareOptions(cacheDir))
	if err != nil {
		t.Fatalf("ExecutePrepareAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	metadata := prepareMetadata(t, receipt)
	if metadata["downloaded"] != true || metadata["hash_verified"] != true || metadata["installed"] != true {
		t.Fatalf("receipt install flags = %+v", metadata)
	}
	if metadata["proof_status"] != ProofStatusModelPrepared {
		t.Fatalf("proof_status = %v", metadata["proof_status"])
	}
	modelPath, ok := metadata["model_path"].(string)
	if !ok || strings.TrimSpace(modelPath) == "" {
		t.Fatalf("model_path = %#v", metadata["model_path"])
	}
	installed, err := os.ReadFile(modelPath)
	if err != nil {
		t.Fatalf("read installed model: %v", err)
	}
	if string(installed) != string(body) {
		t.Fatalf("installed model body = %q", installed)
	}
}

func TestExecutePrepareAssignmentSHAMismatchFailsSafely(t *testing.T) {
	t.Parallel()

	body := []byte("correct-body")
	cacheDir := t.TempDir()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(body)
	}))
	defer ts.Close()

	spec := testPrepareSpec(ts.URL+"/TinyLlama.Q4_K_M.gguf", body, cacheDir)
	spec.ArtifactSHA256 = hashBytes([]byte("different-body"))
	receipt, handled, err := ExecutePrepareAssignment(context.Background(), testPrepareSpecJSON(t, spec), testPrepareOptions(cacheDir))
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if err == nil {
		t.Fatal("error = nil, want hash mismatch")
	}
	metadata := prepareMetadata(t, receipt)
	if metadata["error_code"] != "hash_mismatch" || metadata["installed"] != false {
		t.Fatalf("receipt metadata = %+v", metadata)
	}
	destination, err := modelcache.ModelPath(cacheDir, spec.ModelID, spec.ArtifactURI)
	if err != nil {
		t.Fatalf("ModelPath() error = %v", err)
	}
	if _, err := os.Stat(destination); !os.IsNotExist(err) {
		t.Fatalf("destination stat err = %v, want not exist", err)
	}
}

func TestExecutePrepareAssignmentExistingModelSkipsDownloadWhenHashMatches(t *testing.T) {
	t.Parallel()

	body := []byte("cached-body")
	cacheDir := t.TempDir()
	spec := testPrepareSpec("https://models.example/TinyLlama.Q4_K_M.gguf", body, cacheDir)
	destination, err := modelcache.ModelPath(cacheDir, spec.ModelID, spec.ArtifactURI)
	if err != nil {
		t.Fatalf("ModelPath() error = %v", err)
	}
	if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
		t.Fatalf("create cache dir: %v", err)
	}
	if err := os.WriteFile(destination, body, 0o644); err != nil {
		t.Fatalf("write existing model: %v", err)
	}

	var calls atomic.Int32
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		_, _ = w.Write([]byte("new-body"))
	}))
	defer ts.Close()
	spec.ArtifactURI = ts.URL + "/TinyLlama.Q4_K_M.gguf"

	receipt, handled, err := ExecutePrepareAssignment(context.Background(), testPrepareSpecJSON(t, spec), testPrepareOptions(cacheDir))
	if err != nil {
		t.Fatalf("ExecutePrepareAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if calls.Load() != 0 {
		t.Fatalf("download calls = %d, want 0", calls.Load())
	}
	metadata := prepareMetadata(t, receipt)
	if metadata["downloaded"] != false || metadata["hash_verified"] != true || metadata["installed"] != true {
		t.Fatalf("receipt metadata = %+v", metadata)
	}
}

func TestExecutePrepareAssignmentKeepWarmStartsLlamaCppManager(t *testing.T) {
	t.Parallel()

	body := []byte("warm-body")
	cacheDir := t.TempDir()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(body)
	}))
	defer ts.Close()

	spec := testPrepareSpec(ts.URL+"/TinyLlama.Q4_K_M.gguf", body, cacheDir)
	spec.KeepWarm = true
	manager := &fakePrepareManager{}
	receipt, handled, err := ExecutePrepareAssignment(context.Background(), testPrepareSpecJSON(t, spec), ExecuteOptions{
		Getenv:            testPrepareGetenv,
		Policy:            testPreparePolicy(cacheDir),
		AllowInsecureHTTP: true,
		LlamaCppManager:   manager,
	})
	if err != nil {
		t.Fatalf("ExecutePrepareAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if manager.restartCalls != 1 {
		t.Fatalf("restart calls = %d, want 1", manager.restartCalls)
	}
	metadata := prepareMetadata(t, receipt)
	if metadata["warm"] != true {
		t.Fatalf("warm metadata = %+v", metadata)
	}
	if manager.modelPath == "" || manager.modelPath != metadata["model_path"] {
		t.Fatalf("manager model path = %q metadata=%+v", manager.modelPath, metadata)
	}
}

func TestPrepareReceiptExcludesUnsafeFields(t *testing.T) {
	t.Parallel()

	spec := testPrepareSpec("https://models.example/TinyLlama.Q4_K_M.gguf?token=secret", []byte("safe-body"), t.TempDir())
	receipt, err := BuildPrepareReceipt(PrepareExecutionResult{
		Spec:           spec,
		Downloaded:     true,
		HashVerified:   true,
		Installed:      true,
		ModelPath:      filepath.Join(t.TempDir(), "TinyLlama.Q4_K_M.gguf"),
		ModelSizeBytes: int64(len("safe-body")),
		Warm:           true,
		Benchmark: &llamacpp.BenchmarkStatusSnapshot{
			Status: llamacpp.BenchmarkStatusCompleted,
			Metrics: llamacpp.BenchmarkMetrics{
				SidecarHealthy:  true,
				ModelLoaded:     true,
				WarmupRuns:      1,
				MeasuredRuns:    1,
				TokensGenerated: 4,
				ProofStatus:     llamacpp.BenchmarkProofStatusMeasured,
			},
		},
	})
	if err != nil {
		t.Fatalf("BuildPrepareReceipt() error = %v", err)
	}
	if !PrepareReceiptJSONContainsNoUnsafeFields(receipt) {
		raw, _ := json.Marshal(receipt.Metadata)
		t.Fatalf("receipt leaked unsafe fields: %s", raw)
	}
}

func testPrepareSpec(artifactURI string, body []byte, cacheDir string) PrepareSpec {
	return PrepareSpec{
		Task:                     PrepareTask,
		PrepareID:                "prepare-local",
		RequestID:                "request-local",
		JobID:                    "job-local",
		ModelID:                  "TinyLlama.Q4_K_M.gguf",
		ArtifactURI:              artifactURI,
		ArtifactSHA256:           hashBytes(body),
		ArtifactSizeBytes:        int64(len(body)),
		Backend:                  "llama.cpp",
		KeepWarm:                 false,
		RunBenchmarkAfterPrepare: false,
		TimeoutMs:                30_000,
		ModelFamily:              "llama",
		ArtifactFormat:           "gguf",
	}
}

func testPrepareSpecJSON(t *testing.T, spec PrepareSpec) string {
	t.Helper()
	encoded, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	return string(encoded)
}

func testPrepareOptions(cacheDir string) ExecuteOptions {
	return ExecuteOptions{
		Getenv:            testPrepareGetenv,
		Policy:            testPreparePolicy(cacheDir),
		AllowInsecureHTTP: true,
	}
}

func testPrepareGetenv(name string) string {
	if name == PrepareFlagEnv {
		return "1"
	}
	return ""
}

func testPreparePolicy(cacheDir string) modelpolicy.Policy {
	return modelpolicy.Policy{
		AutoDownload:        true,
		MaxSingleModelBytes: 1024 * 1024,
		MaxCacheBytes:       4 * 1024 * 1024,
		CacheDir:            cacheDir,
		AllowedFamilies:     []string{"llama"},
		AllowedFormats:      []string{"gguf"},
	}
}

func prepareMetadata(t *testing.T, receipt PrepareReceipt) map[string]any {
	t.Helper()
	metadata, ok := receipt.Metadata[PrepareTask].(map[string]any)
	if !ok {
		t.Fatalf("receipt metadata missing %q: %+v", PrepareTask, receipt.Metadata)
	}
	return metadata
}

func hashBytes(body []byte) string {
	sum := sha256.Sum256(body)
	return "sha256:" + hex.EncodeToString(sum[:])
}

type fakePrepareManager struct {
	restartCalls int
	modelPath    string
	status       llamacpp.LlamaCppSidecarStatus
}

func (f *fakePrepareManager) Start(context.Context) llamacpp.LlamaCppSidecarStatus {
	return f.status
}

func (f *fakePrepareManager) Status(context.Context) llamacpp.LlamaCppSidecarStatus {
	return f.status
}

func (f *fakePrepareManager) RestartWithModel(_ context.Context, modelPath string) llamacpp.LlamaCppSidecarStatus {
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
