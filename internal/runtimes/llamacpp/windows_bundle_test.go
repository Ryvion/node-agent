package llamacpp

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestShouldInstallWindowsGPUServerBundleRefreshesMissingMarker(t *testing.T) {
	serverPath := filepath.Join(t.TempDir(), "llama-server.exe")
	if err := os.WriteFile(serverPath, []byte("server"), 0o755); err != nil {
		t.Fatalf("write server: %v", err)
	}

	install, reason := shouldInstallWindowsGPUServerBundle(serverPath, managedServerSourceMarkerPath(serverPath), managedWindowsCUDABundle())
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
	if err := os.WriteFile(markerPath, []byte(expectedManagedServerSourceMarker(managedWindowsCUDABundle())+"\n"), 0o644); err != nil {
		t.Fatalf("write marker: %v", err)
	}

	install, reason := shouldInstallWindowsGPUServerBundle(serverPath, markerPath, managedWindowsCUDABundle())
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

	install, reason := shouldInstallWindowsGPUServerBundle(serverPath, markerPath, managedWindowsCUDABundle())
	if !install || reason != "source_url_changed" {
		t.Fatalf("shouldInstallWindowsGPUServerBundle() = %t/%q, want v2 marker refresh", install, reason)
	}
}

func TestExpectedManagedServerSourceMarkerIncludesCUDABundle(t *testing.T) {
	marker := expectedManagedServerSourceMarker(managedWindowsCUDABundle())
	for _, want := range []string{
		managedWindowsCUDAServerURL,
		managedWindowsCUDARuntimeURL,
		"windows_accelerator=cuda",
		"installer=ryvion-managed-llama-v5",
	} {
		if !strings.Contains(marker, want) {
			t.Fatalf("expected marker to contain %q, got %q", want, marker)
		}
	}
}

func TestSelectManagedWindowsGPUBundlePrefersCUDAForNVIDIA(t *testing.T) {
	bundle := selectManagedWindowsGPUBundle(
		func(string) string { return "" },
		func(name string) (string, error) {
			if name == "nvidia-smi" {
				return `C:\Windows\System32\nvidia-smi.exe`, nil
			}
			return "", os.ErrNotExist
		},
		func(string) (os.FileInfo, error) { return nil, os.ErrNotExist },
	)
	if bundle.Accelerator != "cuda" || bundle.ServerURL != managedWindowsCUDAServerURL {
		t.Fatalf("bundle = %+v, want cuda bundle", bundle)
	}
}

func TestSelectManagedWindowsGPUBundlePrefersCUDAForKnownNVIDIASMIPath(t *testing.T) {
	bundle := selectManagedWindowsGPUBundle(
		func(key string) string {
			switch key {
			case "SystemRoot":
				return `C:\Windows`
			default:
				return ""
			}
		},
		func(string) (string, error) { return "", os.ErrNotExist },
		func(path string) (os.FileInfo, error) {
			if strings.EqualFold(path, `C:\Windows\System32\nvidia-smi.exe`) {
				return fakeWindowsBundleFileInfo{name: "nvidia-smi.exe"}, nil
			}
			return nil, os.ErrNotExist
		},
	)
	if bundle.Accelerator != "cuda" || bundle.ServerURL != managedWindowsCUDAServerURL {
		t.Fatalf("bundle = %+v, want cuda bundle from known nvidia-smi path", bundle)
	}
}

func TestSelectManagedWindowsGPUBundleAllowsVulkanOverride(t *testing.T) {
	bundle := selectManagedWindowsGPUBundle(
		func(key string) string {
			if key == EnvWindowsAccelerator {
				return "vulkan"
			}
			return ""
		},
		func(string) (string, error) { return `C:\Windows\System32\nvidia-smi.exe`, nil },
		func(string) (os.FileInfo, error) { return fakeWindowsBundleFileInfo{name: "nvidia-smi.exe"}, nil },
	)
	if bundle.Accelerator != "vulkan" || bundle.ServerURL != managedWindowsVulkanServerURL {
		t.Fatalf("bundle = %+v, want explicit Vulkan bundle", bundle)
	}
}

type fakeWindowsBundleFileInfo struct {
	name string
	dir  bool
}

func (f fakeWindowsBundleFileInfo) Name() string       { return f.name }
func (f fakeWindowsBundleFileInfo) Size() int64        { return 1 }
func (f fakeWindowsBundleFileInfo) Mode() os.FileMode  { return 0o755 }
func (f fakeWindowsBundleFileInfo) ModTime() time.Time { return time.Unix(100, 0) }
func (f fakeWindowsBundleFileInfo) IsDir() bool        { return f.dir }
func (f fakeWindowsBundleFileInfo) Sys() any           { return nil }
