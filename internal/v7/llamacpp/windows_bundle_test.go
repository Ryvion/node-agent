package llamacpp

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestShouldInstallWindowsGPUServerBundleRefreshesMissingMarker(t *testing.T) {
	serverPath := filepath.Join(t.TempDir(), "llama-server.exe")
	if err := os.WriteFile(serverPath, []byte("server"), 0o755); err != nil {
		t.Fatalf("write server: %v", err)
	}

	install, reason := shouldInstallWindowsGPUServerBundle(serverPath, managedServerSourceMarkerPath(serverPath))
	if !install || reason != "missing_source_marker_for_windows_cuda_runtime" {
		t.Fatalf("shouldInstallWindowsGPUServerBundle() = %t/%q, want missing marker refresh", install, reason)
	}
}

func TestShouldInstallWindowsGPUServerBundleAcceptsCurrentMarker(t *testing.T) {
	dir := t.TempDir()
	serverPath := filepath.Join(dir, "llama-server.exe")
	markerPath := managedServerSourceMarkerPath(serverPath)
	if err := os.WriteFile(serverPath, []byte("server"), 0o755); err != nil {
		t.Fatalf("write server: %v", err)
	}
	if err := os.WriteFile(markerPath, []byte(expectedManagedServerSourceMarker()+"\n"), 0o644); err != nil {
		t.Fatalf("write marker: %v", err)
	}

	install, reason := shouldInstallWindowsGPUServerBundle(serverPath, markerPath)
	if install || reason != "" {
		t.Fatalf("shouldInstallWindowsGPUServerBundle() = %t/%q, want current marker accepted", install, reason)
	}
}

func TestShouldInstallWindowsGPUServerBundleRefreshesV2Marker(t *testing.T) {
	dir := t.TempDir()
	serverPath := filepath.Join(dir, "llama-server.exe")
	markerPath := managedServerSourceMarkerPath(serverPath)
	if err := os.WriteFile(serverPath, []byte("server"), 0o755); err != nil {
		t.Fatalf("write server: %v", err)
	}
	v2Marker := "https://github.com/ggml-org/llama.cpp/releases/download/b8106/llama-b8106-bin-win-cuda-12.4-x64.zip\ninstaller=ryvion-managed-llama-v2\n"
	if err := os.WriteFile(markerPath, []byte(v2Marker), 0o644); err != nil {
		t.Fatalf("write marker: %v", err)
	}

	install, reason := shouldInstallWindowsGPUServerBundle(serverPath, markerPath)
	if !install || reason != "source_url_changed" {
		t.Fatalf("shouldInstallWindowsGPUServerBundle() = %t/%q, want v2 marker refresh", install, reason)
	}
}

func TestExpectedManagedServerSourceMarkerIncludesVulkanBundle(t *testing.T) {
	marker := expectedManagedServerSourceMarker()
	for _, want := range []string{managedWindowsGPUServerURL, "windows_accelerator=vulkan", "installer=ryvion-managed-llama-v4"} {
		if !strings.Contains(marker, want) {
			t.Fatalf("expected marker to contain %q, got %q", want, marker)
		}
	}
}
