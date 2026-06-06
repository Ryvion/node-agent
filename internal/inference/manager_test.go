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

func TestModelDownloadURLPublicModelsBypassHubProxy(t *testing.T) {
	// Regression guard for the "reasoning models stuck loading / timeout" bug.
	//
	// qwen3-8b-reasoning and gpt-oss-20b are PUBLIC (RequiresHuggingFaceAuth
	// =false) but carry a PlatformPath fallback. The node must download them
	// directly from HuggingFace's CDN — exactly like phi-4/tinyllama, which
	// have no PlatformPath and work fine. Routing them through the hub's
	// single-machine artifact proxy streamed the 5GB/12GB GGUF inline and
	// pushed first-request cold starts past the 300s EnsureModel deadline,
	// surfacing to buyers as "timeout waiting for <model> to start".
	//
	// This asserts the production configuration: hubURL + nodeToken set and
	// platform-managed downloads at their default (enabled). Even so, public
	// models must resolve to their direct upstream URL.
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HUGGINGFACE_TOKEN", "")

	m := &Manager{hubURL: "https://api.ryvion.ai", nodeToken: func(int64) string { return "node-token" }}

	// nemotron-3-nano-omni-30b-a3b is public (ggml-org/NVIDIA-Nemotron-3-Nano-Omni),
	// so it downloads directly from HF like the other public models.
	for _, id := range []string{"qwen3-8b-reasoning", "gpt-oss-20b", "gemma-4-26b-a4b-it", "nemotron-3-nano-omni-30b-a3b"} {
		cfg, ok := NativeModels[id]
		if !ok {
			t.Fatalf("model %s missing from native registry", id)
		}
		got := m.modelDownloadURL(cfg)
		if got != cfg.URL {
			t.Errorf("model %s: expected direct upstream URL %q, got %q", id, cfg.URL, got)
		}
		if strings.Contains(got, "/api/v1/node/models/") {
			t.Errorf("model %s: routed through hub proxy %q — public models must download from HuggingFace directly", id, got)
		}
	}
}

func TestModelDownloadURLGatedModelWithoutTokenUsesHubProxy(t *testing.T) {
	// A genuinely gated repo with no local HF token CANNOT be fetched by the
	// node itself, so it must fall back to the hub proxy (the hub attaches
	// its own upstream token). This is the only case the proxy exists for.
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HUGGINGFACE_TOKEN", "")

	m := &Manager{hubURL: "https://api.ryvion.ai", nodeToken: func(int64) string { return "node-token" }}
	cfg := ModelConfig{
		FileName:                "gated.gguf",
		URL:                     "https://huggingface.co/gated/repo/resolve/main/gated.gguf",
		PlatformPath:            "/api/v1/node/models/gated/download",
		RequiresHuggingFaceAuth: true,
	}
	got := m.modelDownloadURL(cfg)
	want := "https://api.ryvion.ai/api/v1/node/models/gated/download"
	if got != want {
		t.Errorf("gated model without token: expected hub proxy %q, got %q", want, got)
	}
}

func TestModelDownloadURLGatedModelWithTokenUsesDirectURL(t *testing.T) {
	// With a local HF token the node can pull the gated repo directly, so it
	// should NOT detour through the hub proxy.
	t.Setenv("HF_TOKEN", "hf-secret")
	t.Setenv("HUGGINGFACE_TOKEN", "")

	m := &Manager{hubURL: "https://api.ryvion.ai", nodeToken: func(int64) string { return "node-token" }}
	cfg := ModelConfig{
		FileName:                "gated.gguf",
		URL:                     "https://huggingface.co/gated/repo/resolve/main/gated.gguf",
		PlatformPath:            "/api/v1/node/models/gated/download",
		RequiresHuggingFaceAuth: true,
	}
	if got := m.modelDownloadURL(cfg); got != cfg.URL {
		t.Errorf("gated model with token: expected direct URL %q, got %q", cfg.URL, got)
	}
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

func TestEnsureModelWaitsWhileDownloadProgresses(t *testing.T) {
	// Regression guard for "reasoning model stuck at first-time load then
	// times out". llama-server reports a different model (the GGUF is still
	// downloading), so EnsureModel never sees a match. A *continuously
	// progressing* download must keep resetting the stall clock so a large
	// first-time load on a slow link isn't killed mid-transfer — EnsureModel
	// should ultimately exit via the job context, NOT via the
	// "timeout waiting ... to start" stall path.
	llamaSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf","object":"model"}]}`))
	}))
	defer llamaSrv.Close()

	mgr := &Manager{
		dataDir:             t.TempDir(),
		port:                testServerPort(t, llamaSrv.URL),
		activeModelName:     "qwen3-8b-reasoning",
		healthy:             true,
		startupStallTimeout: 2 * time.Second,
		activeDownloads:     map[string]*ActiveDownload{},
	}
	mgr.cancel = func() {}

	mgr.registerDownload("qwen3-8b-reasoning")
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		var done int64
		for {
			select {
			case <-stop:
				return
			default:
			}
			done += 50 * 1024 * 1024 // ~50 MB per tick
			mgr.updateDownloadProgress("qwen3-8b-reasoning", done, 5*1024*1024*1024)
			time.Sleep(300 * time.Millisecond)
		}
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Second)
	defer cancel()
	start := time.Now()
	err := mgr.EnsureModel(ctx, "qwen3-8b-reasoning")
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected EnsureModel to return an error (model never loads in this test)")
	}
	if strings.Contains(err.Error(), "timeout waiting for") {
		t.Fatalf("EnsureModel hit the stall timeout after %v despite a continuously-progressing download: %v", elapsed, err)
	}
	if elapsed < 3*time.Second {
		t.Fatalf("EnsureModel returned after %v — expected it to wait past the %v stall window while the download progressed", elapsed, mgr.startupStallTimeout)
	}
}

func TestEnsureModelTimesOutWhenDownloadStalls(t *testing.T) {
	// With no download progress (never started or stalled) and no model loaded,
	// EnsureModel must still surface its "timeout waiting ... to start" error so
	// a genuinely stuck node fails fast instead of hanging forever.
	llamaSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf","object":"model"}]}`))
	}))
	defer llamaSrv.Close()

	mgr := &Manager{
		dataDir:             t.TempDir(),
		port:                testServerPort(t, llamaSrv.URL),
		activeModelName:     "qwen3-8b-reasoning",
		healthy:             true,
		startupStallTimeout: 1 * time.Second,
		activeDownloads:     map[string]*ActiveDownload{},
	}
	mgr.cancel = func() {}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	err := mgr.EnsureModel(ctx, "qwen3-8b-reasoning")
	if err == nil || !strings.Contains(err.Error(), "timeout waiting for") {
		t.Fatalf("expected stall timeout error, got %v", err)
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
	// Body must start with the GGUF magic header — downloadModelFile now
	// validates after fetch and removes/errors on a missing magic. 4 MiB
	// of payload is still well above the >1MiB sanity floor.
	body := "GGUF" + strings.Repeat("x", 4<<20)
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

func TestDownloadModelFileRemovesCorruptOnDiskGGUF(t *testing.T) {
	t.Parallel()

	// HTTP server serves a valid (magic="GGUF") body so the redownload
	// succeeds — the test asserts that downloadModelFile rejects the
	// pre-existing corrupt file on disk + actually goes out to the
	// network instead of silently returning early.
	var hits int32
	body := "GGUF" + strings.Repeat("y", 2<<20) // 2 MiB
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		_, _ = w.Write([]byte(body))
	}))
	defer srv.Close()

	mgr := New(t.TempDir())
	if err := os.MkdirAll(filepath.Join(mgr.dataDir, "models"), 0o755); err != nil {
		t.Fatalf("mkdir models: %v", err)
	}
	dst := filepath.Join(mgr.dataDir, "models", "Qwen3-8B-Q4_K_M.gguf")

	// Seed a "corrupt" file > 1 MiB so the size check passes but the
	// magic-byte check fails — simulates a previously-interrupted
	// download that left garbage on disk.
	corrupt := strings.Repeat("0", 2<<20)
	if err := os.WriteFile(dst, []byte(corrupt), 0o644); err != nil {
		t.Fatalf("seed corrupt file: %v", err)
	}

	if err := mgr.downloadModelFile(context.Background(), srv.URL, dst); err != nil {
		t.Fatalf("downloadModelFile: %v", err)
	}
	if got := atomic.LoadInt32(&hits); got != 1 {
		t.Fatalf("expected 1 HTTP GET after corrupt-file recovery, got %d", got)
	}
	// The bad file must have been replaced with a valid GGUF.
	if err := validateGGUF(dst); err != nil {
		t.Fatalf("post-recovery file still invalid: %v", err)
	}
	if info, err := os.Stat(dst); err != nil {
		t.Fatalf("stat dst: %v", err)
	} else if info.Size() != int64(len(body)) {
		t.Fatalf("dst size = %d, want %d", info.Size(), len(body))
	}
}

func TestDownloadModelFileStripsNodeTokenOnRedirect(t *testing.T) {
	t.Parallel()

	body := "GGUF" + strings.Repeat("z", 2<<20)
	var cdnToken string
	cdn := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cdnToken = r.Header.Get(nodeTokenHeader)
		w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		_, _ = w.Write([]byte(body))
	}))
	defer cdn.Close()

	var hubToken string
	hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hubToken = r.Header.Get(nodeTokenHeader)
		http.Redirect(w, r, cdn.URL+"/Qwen3-8B-Q4_K_M.gguf", http.StatusTemporaryRedirect)
	}))
	defer hub.Close()

	mgr := New(t.TempDir())
	mgr.nodeToken = func(int64) string { return "node-secret" }
	if err := os.MkdirAll(filepath.Join(mgr.dataDir, "models"), 0o755); err != nil {
		t.Fatalf("mkdir models: %v", err)
	}
	dst := filepath.Join(mgr.dataDir, "models", "Qwen3-8B-Q4_K_M.gguf")

	if err := mgr.downloadModelFile(context.Background(), hub.URL+"/api/v1/node/models/qwen3-8b-reasoning/download", dst); err != nil {
		t.Fatalf("downloadModelFile: %v", err)
	}
	if hubToken != "node-secret" {
		t.Fatalf("hub token = %q, want node auth token", hubToken)
	}
	if cdnToken != "" {
		t.Fatalf("cdn received node token %q; node auth must not follow CDN redirects", cdnToken)
	}
}

func TestStreamingFamilyHintForModel(t *testing.T) {
	cases := []struct{ path, want string }{
		{"/models/Qwen3-8B-Q4_K_M.gguf", "qwen"},
		{"/models/gpt-oss-20b-mxfp4.gguf", "gpt-oss"},
		{"/models/gptoss-20b.gguf", "gpt-oss"},
		{"/models/DeepSeek-R1-Q4_K_M.gguf", "deepseek"},
		{"/models/phi-4-Q4_K_M.gguf", "phi"},
		{"/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf", "llama"},
		{"/models/gemma-4-26B-A4B-it-Q4_K_M.gguf", "gemma"},
		{"/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "llama"},
		{"/models/unknown-model.gguf", ""},
		{"", ""},
	}
	for _, tc := range cases {
		got := streamingFamilyHintForModel(tc.path)
		if got != tc.want {
			t.Errorf("streamingFamilyHintForModel(%q) = %q, want %q", tc.path, got, tc.want)
		}
	}
}

func TestStreamingJinjaRequiredForModel(t *testing.T) {
	// Models that NEED --jinja for chat template / reasoning_effort.
	for _, path := range []string{
		"/models/Qwen3-8B-Q4_K_M.gguf",
		"/models/gpt-oss-20b-mxfp4.gguf",
		"/models/DeepSeek-R1.gguf",
	} {
		if !streamingJinjaRequiredForModel(path) {
			t.Errorf("expected --jinja for %q (reasoning-capable model)", path)
		}
	}
	// Models that do NOT need --jinja.
	for _, path := range []string{
		"/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		"/models/phi-4-Q4_K_M.gguf",
		"/models/tinyllama.gguf",
	} {
		if streamingJinjaRequiredForModel(path) {
			t.Errorf("did not expect --jinja for %q", path)
		}
	}
}

func TestStreamingReasoningFormatForModel(t *testing.T) {
	cases := []struct{ path, want string }{
		{"/models/Qwen3-8B-Q4_K_M.gguf", "deepseek"},
		{"/models/DeepSeek-R1-Q4_K_M.gguf", "deepseek"},
		{"/models/gpt-oss-20b-mxfp4.gguf", "auto"},
		{"/models/Llama-3.2-3B.gguf", ""},
		{"/models/phi-4.gguf", ""},
		{"", ""},
	}
	for _, tc := range cases {
		got := streamingReasoningFormatForModel(tc.path)
		if got != tc.want {
			t.Errorf("streamingReasoningFormatForModel(%q) = %q, want %q", tc.path, got, tc.want)
		}
	}
}

func TestSortPrewarmByPriority(t *testing.T) {
	// Input is sort.Strings() output for a 24GB Mac — the alphabetical
	// order that we're reordering AWAY from.
	in := []string{
		"gemma-4-26b-a4b-it",
		"gpt-oss-20b",
		"nemotron-3-nano-omni-30b-a3b",
		"phi-4",
		"qwen3-8b-reasoning",
		"ryvion-llama-3.2-3b",
		"tinyllama",
	}
	got := sortPrewarmByPriority(in)
	want := []string{
		"tinyllama",
		"ryvion-llama-3.2-3b",
		"phi-4",
		"qwen3-8b-reasoning",
		"gpt-oss-20b",
		"gemma-4-26b-a4b-it",
		"nemotron-3-nano-omni-30b-a3b",
	}
	for i, w := range want {
		if got[i] != w {
			t.Fatalf("position %d: got %q, want %q (full got=%v)", i, got[i], w, got)
		}
	}
}

func TestSelectPrewarmModelsDefaultLeanSkipsHugeModels(t *testing.T) {
	supported := []string{
		"gemma-4-26b-a4b-it",
		"gpt-oss-20b",
		"nemotron-3-nano-omni-30b-a3b",
		"phi-4",
		"qwen3-8b-reasoning",
		"ryvion-llama-3.2-3b",
		"tinyllama",
	}
	got := selectPrewarmModels(supported, prewarmTestEnv(nil))
	want := []string{"tinyllama", "ryvion-llama-3.2-3b", "qwen3-8b-reasoning"}
	requireStringSlice(t, got, want)
}

func TestSelectPrewarmModelsAllKeepsHugeModels(t *testing.T) {
	supported := []string{
		"gemma-4-26b-a4b-it",
		"gpt-oss-20b",
		"nemotron-3-nano-omni-30b-a3b",
		"phi-4",
		"qwen3-8b-reasoning",
		"ryvion-llama-3.2-3b",
		"tinyllama",
	}
	got := selectPrewarmModels(supported, prewarmTestEnv(map[string]string{
		"RYV_MODEL_PREWARM_MODE": "all",
	}))
	want := []string{
		"tinyllama",
		"ryvion-llama-3.2-3b",
		"phi-4",
		"qwen3-8b-reasoning",
		"gpt-oss-20b",
		"gemma-4-26b-a4b-it",
		"nemotron-3-nano-omni-30b-a3b",
	}
	requireStringSlice(t, got, want)
}

func TestSelectPrewarmModelsExplicitListPreservesOperatorOrder(t *testing.T) {
	supported := []string{"qwen3-8b-reasoning", "ryvion-llama-3.2-3b", "nemotron-3-nano-omni-30b-a3b"}
	got := selectPrewarmModels(supported, prewarmTestEnv(map[string]string{
		"RYV_PREWARM_MODELS": "nemotron-3-nano-omni-30b-a3b,missing-model,qwen3-8b-reasoning,nemotron-3-nano-omni-30b-a3b",
	}))
	want := []string{"nemotron-3-nano-omni-30b-a3b", "qwen3-8b-reasoning"}
	requireStringSlice(t, got, want)
}

func TestSelectPrewarmModelsOff(t *testing.T) {
	got := selectPrewarmModels([]string{"tinyllama", "ryvion-llama-3.2-3b"}, prewarmTestEnv(map[string]string{
		"RYV_MODEL_PREWARM_MODE": "off",
	}))
	if len(got) != 0 {
		t.Fatalf("expected no prewarm models, got %v", got)
	}
}

func TestSortPrewarmByPriorityHandlesUnknownModels(t *testing.T) {
	// Future model not yet in prewarmPriority should land AFTER all
	// known ones but still be present (not dropped).
	in := []string{"tinyllama", "some-future-13b", "qwen3-8b-reasoning"}
	got := sortPrewarmByPriority(in)
	if len(got) != 3 {
		t.Fatalf("dropped a model: got %v", got)
	}
	if got[0] != "tinyllama" || got[1] != "qwen3-8b-reasoning" {
		t.Fatalf("known-prewarm ordering wrong: %v", got)
	}
	if got[2] != "some-future-13b" {
		t.Fatalf("unknown model should land last, got %v", got)
	}
}

func prewarmTestEnv(values map[string]string) func(string) string {
	return func(key string) string {
		return values[key]
	}
}

func requireStringSlice(t *testing.T, got, want []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}
