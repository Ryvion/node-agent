package llamacpp

import (
	"os"
	"path/filepath"
	"strings"
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

func TestShouldInstallWindowsCUDAServerBundleRefreshesV2Marker(t *testing.T) {
	dir := t.TempDir()
	serverPath := filepath.Join(dir, "llama-server.exe")
	markerPath := managedServerSourceMarkerPath(serverPath)
	if err := os.WriteFile(serverPath, []byte("server"), 0o755); err != nil {
		t.Fatalf("write server: %v", err)
	}
	v2Marker := managedWindowsCUDAServerURL + "\ninstaller=ryvion-managed-llama-v2\n"
	if err := os.WriteFile(markerPath, []byte(v2Marker), 0o644); err != nil {
		t.Fatalf("write marker: %v", err)
	}

	install, reason := shouldInstallWindowsCUDAServerBundle(serverPath, markerPath)
	if !install || reason != "source_url_changed" {
		t.Fatalf("shouldInstallWindowsCUDAServerBundle() = %t/%q, want v2 marker refresh", install, reason)
	}
}

func TestExpectedManagedServerSourceMarkerIncludesRuntimeBundle(t *testing.T) {
	marker := expectedManagedServerSourceMarker()
	for _, want := range []string{managedWindowsCUDAServerURL, managedWindowsCUDARuntimeURL, "installer=ryvion-managed-llama-v3"} {
		if !strings.Contains(marker, want) {
			t.Fatalf("expected marker to contain %q, got %q", want, marker)
		}
	}
}
