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
		if model == "gemma-3-27b-it" {
			t.Fatal("expected Gemma 27B to be hidden below 16GB VRAM")
		}
	}

	enoughVRAM := SupportedNativeChatModels(16 * 1024 * 1024 * 1024)
	found := false
	for _, model := range enoughVRAM {
		if model == "gemma-3-27b-it" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected Gemma 27B to be advertised on 16GB VRAM nodes")
	}
}

func TestSupportedNativeChatModelsAllowsDriverReservedVRAMOn16GBCards(t *testing.T) {
	t.Setenv("HF_TOKEN", "test-token")

	reportedVRAM := uint64(17171480576) // RTX 4070 Ti SUPER can report just under exact 16 GiB.
	models := SupportedNativeChatModels(reportedVRAM)
	for _, model := range models {
		if model == "gemma-3-27b-it" {
			return
		}
	}
	t.Fatalf("expected Gemma 27B to be advertised for 16GB-class GPU with %d reported bytes", reportedVRAM)
}

func TestSupportedNativeChatModelsRequiresTokenForGatedGemma(t *testing.T) {
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HUGGINGFACE_TOKEN", "")

	models := SupportedNativeChatModels(16 * 1024 * 1024 * 1024)
	found := false
	for _, model := range models {
		if model == "gemma-3-27b-it" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected gated Gemma model to be advertised through Ryvion platform-managed model downloads")
	}
}

func TestSupportedNativeChatModelsCanDisablePlatformGatedModels(t *testing.T) {
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HUGGINGFACE_TOKEN", "")
	t.Setenv("RYV_DISABLE_PLATFORM_MODEL_DOWNLOADS", "1")

	models := SupportedNativeChatModels(16 * 1024 * 1024 * 1024)
	for _, model := range models {
		if model == "gemma-3-27b-it" {
			t.Fatal("expected gated Gemma model to stay hidden when platform downloads are disabled and no local token is configured")
		}
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
		"https://github.com/ggml-org/llama.cpp/releases/download/b8106/llama-b8106-bin-win-cuda-12.4-x64.zip",
		false,
	)
	if !install || reason != "missing_source_marker_for_windows_cuda_runtime" {
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
		"https://github.com/ggml-org/llama.cpp/releases/download/b8106/llama-b8106-bin-win-cuda-12.4-x64.zip",
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
	sourceURL := "https://github.com/ggml-org/llama.cpp/releases/download/b8106/llama-b8106-bin-win-cuda-12.4-x64.zip"
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
	sourceURL := "https://github.com/ggml-org/llama.cpp/releases/download/b8106/llama-b8106-bin-win-cuda-12.4-x64.zip"
	marker := expectedServerSourceMarker(sourceURL)
	if !containsAll(marker, sourceURL, windowsCUDARuntimeURL, "installer=ryvion-managed-llama-v3") {
		t.Fatalf("expected Windows CUDA marker to include server/runtime/v3 marker, got %q", marker)
	}
}

func TestShouldInstallServerRefreshesLegacySourceOnlyMarker(t *testing.T) {
	dir := t.TempDir()
	serverPath := filepath.Join(dir, "llama-server.exe")
	markerPath := serverSourceMarkerPath(serverPath)
	sourceURL := "https://github.com/ggml-org/llama.cpp/releases/download/b8106/llama-b8106-bin-win-cuda-12.4-x64.zip"
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
