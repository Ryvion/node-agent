package inference

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestSupportedNativeChatModelsGatesGemmaByVRAM(t *testing.T) {
	t.Setenv("HF_TOKEN", "test-token")

	lowVRAM := SupportedNativeChatModels(12 * 1024 * 1024 * 1024)
	for _, model := range lowVRAM {
		if model == "gemma-4-26b-a4b-it" {
			t.Fatal("expected Gemma 4 to be hidden below 16GB VRAM")
		}
	}

	enoughVRAM := SupportedNativeChatModels(16 * 1024 * 1024 * 1024)
	found := false
	for _, model := range enoughVRAM {
		if model == "gemma-4-26b-a4b-it" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected Gemma 4 to be advertised on 16GB VRAM nodes")
	}
}

func TestSupportedNativeChatModelsIncludesQwenReasoning(t *testing.T) {
	models := SupportedNativeChatModels(8 * 1024 * 1024 * 1024)
	for _, model := range models {
		if model == "qwen3-8b-reasoning" {
			return
		}
	}
	t.Fatal("expected Qwen3 reasoning to be supported by the native model registry")
}

func TestSupportedNativeChatModelsGatesGPTOSSByVRAM(t *testing.T) {
	lowVRAM := SupportedNativeChatModels(12 * 1024 * 1024 * 1024)
	for _, model := range lowVRAM {
		if model == "gpt-oss-20b" {
			t.Fatal("expected GPT-OSS 20B to be hidden below 16GB VRAM")
		}
	}

	models := SupportedNativeChatModels(16 * 1024 * 1024 * 1024)
	for _, model := range models {
		if model == "gpt-oss-20b" {
			return
		}
	}
	t.Fatal("expected GPT-OSS 20B to advertise on 16GB VRAM nodes")
}

func TestSupportedNativeChatModelsAllowsDriverReservedVRAMOn16GBCards(t *testing.T) {
	t.Setenv("HF_TOKEN", "test-token")

	reportedVRAM := uint64(17171480576) // RTX 4070 Ti SUPER can report just under exact 16 GiB.
	models := SupportedNativeChatModels(reportedVRAM)
	foundGemma := false
	foundGPTOSS := false
	for _, model := range models {
		foundGemma = foundGemma || model == "gemma-4-26b-a4b-it"
		foundGPTOSS = foundGPTOSS || model == "gpt-oss-20b"
	}
	if !foundGemma {
		t.Fatalf("expected Gemma 4 to be advertised for 16GB-class GPU with %d reported bytes", reportedVRAM)
	}
	if !foundGPTOSS {
		t.Fatalf("expected GPT-OSS 20B to be advertised for 16GB-class GPU with %d reported bytes", reportedVRAM)
	}
}

func TestSupportedNativeChatModelsAdvertisesGemmaWithoutHFToken(t *testing.T) {
	// The ggml-org Gemma 4 26B-A4B-it GGUF mirror is publicly downloadable
	// without an HF token (verified 2026-05: HEAD → 302 to CDN, no auth).
	// This regression guard prevents anyone from re-flipping
	// RequiresHuggingFaceAuth=true and silently hiding the model from every
	// operator who hasn't set HF_TOKEN. Without this, the user's 24GB VRAM
	// node could not advertise gemma even though the weights were free.
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HUGGINGFACE_TOKEN", "")

	models := SupportedNativeChatModels(16 * 1024 * 1024 * 1024)
	found := false
	for _, model := range models {
		if model == "gemma-4-26b-a4b-it" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected Gemma model to advertise without HF_TOKEN — the ggml-org GGUF mirror is publicly accessible")
	}
}

func TestSupportedNativeChatModelsAdvertisesGemmaEvenWithPlatformDownloadsDisabled(t *testing.T) {
	// Even with RYV_DISABLE_PLATFORM_MODEL_DOWNLOADS=1 (operator opted out
	// of the Ryvion-managed proxy), Gemma must still advertise because the
	// upstream URL doesn't actually require auth. Previously this combo
	// hid gemma since RequiresHuggingFaceAuth=true && no token && platform
	// disabled => skip. Public mirror makes that gating obsolete.
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HUGGINGFACE_TOKEN", "")
	t.Setenv("RYV_DISABLE_PLATFORM_MODEL_DOWNLOADS", "1")

	models := SupportedNativeChatModels(16 * 1024 * 1024 * 1024)
	for _, model := range models {
		if model == "gemma-4-26b-a4b-it" {
			return
		}
	}
	t.Fatal("expected Gemma model to advertise without HF_TOKEN even when platform-managed downloads are disabled")
}

func TestEnsureModelRestartsWhenServedModelMismatchesActiveName(t *testing.T) {
	var restartCount atomic.Int32
	llamaSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if restartCount.Load() == 0 {
			_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"Llama-3.2-3B-Instruct-Q4_K_M.gguf","object":"model"}]}`))
			return
		}
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"gpt-oss-20b-mxfp4.gguf","object":"model"}]}`))
	}))
	defer llamaSrv.Close()

	mgr := &Manager{
		dataDir:         t.TempDir(),
		port:            testServerPort(t, llamaSrv.URL),
		activeModelName: "gpt-oss-20b",
		healthy:         true,
	}
	mgr.cancel = func() {
		restartCount.Add(1)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := mgr.EnsureModel(ctx, "gpt-oss-20b"); err != nil {
		t.Fatalf("EnsureModel returned error: %v", err)
	}
	if restartCount.Load() == 0 {
		t.Fatal("expected EnsureModel to restart stale llama-server after loaded model mismatch")
	}
}

func TestEnsureModelClearsCustomModelPathWhenSwitchingToRegistryModel(t *testing.T) {
	llamaSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"Llama-3.2-3B-Instruct-Q4_K_M.gguf","object":"model"}]}`))
	}))
	defer llamaSrv.Close()

	var restarted atomic.Bool
	mgr := &Manager{
		dataDir:         t.TempDir(),
		port:            testServerPort(t, llamaSrv.URL),
		activeModelName: "custom-model-abc.gguf",
		activeModelPath: filepath.Join(t.TempDir(), "custom-model-abc.gguf"),
		healthy:         true,
	}
	mgr.cancel = func() {
		restarted.Store(true)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := mgr.EnsureModel(ctx, "ryvion-llama-3.2-3b"); err != nil {
		t.Fatalf("EnsureModel returned error: %v", err)
	}
	if !restarted.Load() {
		t.Fatal("expected registry switch to restart llama-server")
	}
	if mgr.activeModelPath != "" {
		t.Fatalf("activeModelPath = %q, want cleared for registry model", mgr.activeModelPath)
	}
}

func TestShouldInstallServerRefreshesUnmarkedWindowsCUDABundle(t *testing.T) {
	serverPath := filepath.Join(t.TempDir(), "llama-server.exe")
	if err := os.WriteFile(serverPath, []byte("old cpu server"), 0o755); err != nil {
		t.Fatalf("write server: %v", err)
	}

	install, reason := shouldInstallServer(
		serverPath,
		serverSourceMarkerPath(serverPath),
		"https://github.com/ggml-org/llama.cpp/releases/download/b9180/llama-b9180-bin-win-cuda-12.4-x64.zip",
		false,
	)
	if !install || reason != "missing_source_marker_for_windows_gpu_runtime" {
		t.Fatalf("shouldInstallServer() = %t/%q, want refresh for unmarked CUDA bundle", install, reason)
	}
}

func TestShouldInstallServerKeepsExplicitOperatorServer(t *testing.T) {
	serverPath := filepath.Join(t.TempDir(), "llama-server.exe")
	if err := os.WriteFile(serverPath, []byte("operator server"), 0o755); err != nil {
		t.Fatalf("write server: %v", err)
	}

	install, reason := shouldInstallServer(
		serverPath,
		serverSourceMarkerPath(serverPath),
		"https://github.com/ggml-org/llama.cpp/releases/download/b9180/llama-b9180-bin-win-cuda-12.4-x64.zip",
		true,
	)
	if install || reason != "" {
		t.Fatalf("shouldInstallServer() = %t/%q, want explicit operator server preserved", install, reason)
	}
}

func TestShouldInstallServerUsesSourceMarker(t *testing.T) {
	dir := t.TempDir()
	serverPath := filepath.Join(dir, "llama-server.exe")
	markerPath := serverSourceMarkerPath(serverPath)
	sourceURL := "https://github.com/ggml-org/llama.cpp/releases/download/b9180/llama-b9180-bin-win-cuda-12.4-x64.zip"
	if err := os.WriteFile(serverPath, []byte("current server"), 0o755); err != nil {
		t.Fatalf("write server: %v", err)
	}
	if err := writeServerSourceMarker(markerPath, sourceURL); err != nil {
		t.Fatalf("write marker: %v", err)
	}

	install, reason := shouldInstallServer(serverPath, markerPath, sourceURL, false)
	if install || reason != "" {
		t.Fatalf("shouldInstallServer() = %t/%q, want current marker accepted", install, reason)
	}

	install, reason = shouldInstallServer(serverPath, markerPath, sourceURL+"?new=1", false)
	if !install || reason != "source_url_changed" {
		t.Fatalf("shouldInstallServer() = %t/%q, want changed source refresh", install, reason)
	}
}

func TestExpectedServerSourceMarkerIncludesWindowsCUDARuntime(t *testing.T) {
	sourceURL := "https://github.com/ggml-org/llama.cpp/releases/download/b9180/llama-b9180-bin-win-cuda-12.4-x64.zip"
	marker := expectedServerSourceMarker(sourceURL)
	if !containsAll(marker, sourceURL, windowsCUDARuntimeURL, "installer=ryvion-managed-llama-v3") {
		t.Fatalf("expected Windows CUDA marker to include server/runtime/v3 marker, got %q", marker)
	}
}

func TestExpectedServerSourceMarkerIncludesWindowsVulkanMarker(t *testing.T) {
	sourceURL := "https://github.com/ggml-org/llama.cpp/releases/download/b9180/llama-b9180-bin-win-vulkan-x64.zip"
	marker := expectedServerSourceMarker(sourceURL)
	if !containsAll(marker, sourceURL, "windows_accelerator=vulkan", "installer=ryvion-managed-llama-v4") {
		t.Fatalf("expected Windows Vulkan marker to include source/vulkan/v4 marker, got %q", marker)
	}
}

func TestShouldInstallServerRefreshesLegacySourceOnlyMarker(t *testing.T) {
	dir := t.TempDir()
	serverPath := filepath.Join(dir, "llama-server.exe")
	markerPath := serverSourceMarkerPath(serverPath)
	sourceURL := "https://github.com/ggml-org/llama.cpp/releases/download/b9180/llama-b9180-bin-win-cuda-12.4-x64.zip"
	if err := os.WriteFile(serverPath, []byte("server"), 0o755); err != nil {
		t.Fatalf("write server: %v", err)
	}
	if err := os.WriteFile(markerPath, []byte(sourceURL+"\n"), 0o644); err != nil {
		t.Fatalf("write legacy marker: %v", err)
	}

	install, reason := shouldInstallServer(serverPath, markerPath, sourceURL, false)
	if !install || reason != "source_url_changed" {
		t.Fatalf("shouldInstallServer() = %t/%q, want legacy marker refresh", install, reason)
	}
}

func containsAll(text string, wants ...string) bool {
	for _, want := range wants {
		if !strings.Contains(text, want) {
			return false
		}
	}
	return true
}

func TestNewReportsBlockerReasonBeforeStart(t *testing.T) {
	t.Setenv("RYV_SERVER_URL", "")
	mgr := New(t.TempDir())
	if mgr == nil {
		t.Fatal("New returned nil")
	}
	if mgr.Healthy() {
		t.Fatal("expected new manager to report Healthy() == false before Start")
	}
	got := mgr.BlockerReason()
	// On platforms with a bundled llama-server URL we report not-started;
	// otherwise platform-unsupported. Either is non-empty and dashboard-safe.
	if got == BlockerNone {
		t.Fatalf("expected non-empty blocker reason on a fresh manager, got %q", got)
	}
	if got != BlockerNotStarted && got != BlockerPlatformUnsupported {
		t.Fatalf("unexpected initial blocker %q", got)
	}
}

func TestSetHealthyClearsBlocker(t *testing.T) {
	mgr := New(t.TempDir())
	mgr.setBlockerReason(BlockerStartupTimeout)
	if got := mgr.BlockerReason(); got != BlockerStartupTimeout {
		t.Fatalf("setBlockerReason did not stick: %q", got)
	}
	mgr.setHealthy(true)
	if !mgr.Healthy() {
		t.Fatal("expected Healthy() true after setHealthy(true)")
	}
	if got := mgr.BlockerReason(); got != BlockerNone {
		t.Fatalf("expected blocker cleared after setHealthy(true), got %q", got)
	}
}

func TestSetBlockerReasonForcesUnhealthy(t *testing.T) {
	mgr := New(t.TempDir())
	mgr.setHealthy(true)
	mgr.setBlockerReason(BlockerProcessFailed)
	if mgr.Healthy() {
		t.Fatal("expected Healthy() false after setBlockerReason of non-empty token")
	}
	if got := mgr.BlockerReason(); got != BlockerProcessFailed {
		t.Fatalf("expected BlockerProcessFailed, got %q", got)
	}
}

func TestNilManagerBlockerReasonIsNotStarted(t *testing.T) {
	var mgr *Manager
	if got := mgr.BlockerReason(); got != BlockerNotStarted {
		t.Fatalf("nil manager should report BlockerNotStarted, got %q", got)
	}
}

func TestResolvedStartupTimeoutDefaultAndOverride(t *testing.T) {
	t.Setenv("RYV_INFERENCE_STARTUP_TIMEOUT_SECONDS", "")
	if got := resolvedStartupTimeout(); got != defaultStartupTimeout {
		t.Fatalf("expected default startup timeout %v, got %v", defaultStartupTimeout, got)
	}
	t.Setenv("RYV_INFERENCE_STARTUP_TIMEOUT_SECONDS", "300")
	if got := resolvedStartupTimeout(); got.Seconds() != 300 {
		t.Fatalf("expected 300s timeout, got %v", got)
	}
	t.Setenv("RYV_INFERENCE_STARTUP_TIMEOUT_SECONDS", "garbage")
	if got := resolvedStartupTimeout(); got != defaultStartupTimeout {
		t.Fatalf("garbage env value must fall back to default, got %v", got)
	}
	t.Setenv("RYV_INFERENCE_STARTUP_TIMEOUT_SECONDS", "0")
	if got := resolvedStartupTimeout(); got != defaultStartupTimeout {
		t.Fatalf("non-positive override must fall back to default, got %v", got)
	}
}

func testServerPort(t *testing.T, rawURL string) string {
	t.Helper()
	parsed, err := url.Parse(rawURL)
	if err != nil {
		t.Fatalf("parse server URL: %v", err)
	}
	_, port, err := net.SplitHostPort(parsed.Host)
	if err != nil {
		t.Fatalf("split server port: %v", err)
	}
	return port
}

func TestDownloadModelFileSerializesConcurrentCallers(t *testing.T) {
	t.Parallel()

	// Slow handler — sleeps mid-stream so concurrent callers overlap.
	// If the per-model lock works, only ONE GET reaches the server even
	// when many goroutines race. Without the lock, every goroutine would
	// hit the server and the .tmp file would get truncated repeatedly.
	var hits int32
	body := strings.Repeat("x", 4<<20) // 4 MiB > the >1MiB "downloaded" floor
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		// Honest Content-Length so net/http actually reads the body. An
		// earlier draft set this to "0" and discovered the bug below:
		// dst ended up at size 0, the file-exists skip on subsequent
		// callers failed, and all 5 goroutines re-downloaded — which
		// looked like a lock failure but was a test-setup artifact.
		w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		// Stream in small chunks with a tiny sleep so concurrent callers
		// definitely overlap during the lock window.
		flusher, _ := w.(http.Flusher)
		const chunk = 64 << 10
		for i := 0; i < len(body); i += chunk {
			end := i + chunk
			if end > len(body) {
				end = len(body)
			}
			_, _ = w.Write([]byte(body[i:end]))
			if flusher != nil {
				flusher.Flush()
			}
			time.Sleep(2 * time.Millisecond)
		}
	}))
	defer srv.Close()

	mgr := New(t.TempDir())
	if err := os.MkdirAll(filepath.Join(mgr.dataDir, "models"), 0o755); err != nil {
		t.Fatalf("mkdir models: %v", err)
	}
	dst := filepath.Join(mgr.dataDir, "models", "Qwen3-8B-Q4_K_M.gguf")

	// Fire 5 concurrent download requests for the same model.
	const callers = 5
	errs := make(chan error, callers)
	for i := 0; i < callers; i++ {
		go func() {
			errs <- mgr.downloadModelFile(context.Background(), srv.URL, dst)
		}()
	}
	for i := 0; i < callers; i++ {
		if err := <-errs; err != nil {
			t.Fatalf("downloadModelFile call %d: %v", i, err)
		}
	}

	// Only one caller actually hit the server — the other 4 skipped after
	// acquiring the lock and seeing the file already on disk.
	if got := atomic.LoadInt32(&hits); got != 1 {
		t.Fatalf("expected exactly 1 HTTP GET to the upstream, got %d", got)
	}
	if info, err := os.Stat(dst); err != nil {
		t.Fatalf("stat dst after download: %v", err)
	} else if info.Size() != int64(len(body)) {
		t.Fatalf("dst size = %d, want %d (truncation bug?)", info.Size(), len(body))
	}
}
