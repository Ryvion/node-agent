package inference

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
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
