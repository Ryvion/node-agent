package llamacpp

import (
	"os"
	"path/filepath"
	"testing"
)

func TestShouldInstallWindowsCUDAServerBundleRefreshesMissingMarker(t *testing.T) {
	serverPath := filepath.Join(t.TempDir(), "llama-server.exe")
	if err := os.WriteFile(serverPath, []byte("server"), 0o755); err != nil {
		t.Fatalf("write server: %v", err)
	}

	install, reason := shouldInstallWindowsCUDAServerBundle(serverPath, managedServerSourceMarkerPath(serverPath))
	if !install || reason != "missing_source_marker_for_windows_cuda_runtime" {
		t.Fatalf("shouldInstallWindowsCUDAServerBundle() = %t/%q, want missing marker refresh", install, reason)
	}
}

func TestShouldInstallWindowsCUDAServerBundleAcceptsCurrentMarker(t *testing.T) {
	dir := t.TempDir()
	serverPath := filepath.Join(dir, "llama-server.exe")
	markerPath := managedServerSourceMarkerPath(serverPath)
	if err := os.WriteFile(serverPath, []byte("server"), 0o755); err != nil {
		t.Fatalf("write server: %v", err)
	}
	if err := os.WriteFile(markerPath, []byte(expectedManagedServerSourceMarker()+"\n"), 0o644); err != nil {
		t.Fatalf("write marker: %v", err)
	}

	install, reason := shouldInstallWindowsCUDAServerBundle(serverPath, markerPath)
	if install || reason != "" {
		t.Fatalf("shouldInstallWindowsCUDAServerBundle() = %t/%q, want current marker accepted", install, reason)
	}
}
