package llamacpp

import (
	"archive/zip"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

const managedWindowsCUDAServerURL = "https://github.com/ggml-org/llama.cpp/releases/download/b8106/llama-b8106-bin-win-cuda-12.4-x64.zip"
const managedWindowsCUDARuntimeURL = "https://github.com/ggml-org/llama.cpp/releases/download/b8106/cudart-llama-bin-win-cuda-12.4-x64.zip"
const managedServerSourceMarkerName = ".llama-server-source"

func ensureWindowsCUDAServerBundle(ctx context.Context, serverPath string) (bool, string, error) {
	serverPath = strings.TrimSpace(serverPath)
	if runtime.GOOS != "windows" || runtime.GOARCH != "amd64" || serverPath == "" {
		return false, "", nil
	}
	install, reason := shouldInstallWindowsCUDAServerBundle(serverPath, managedServerSourceMarkerPath(serverPath))
	if !install {
		return false, "", nil
	}
	if err := os.MkdirAll(filepath.Dir(serverPath), 0o755); err != nil {
		return false, reason, fmt.Errorf("create llama.cpp bin dir: %w", err)
	}
	if err := downloadAndExtractWindowsCUDAArchive(ctx, managedWindowsCUDAServerURL, serverPath, true); err != nil {
		return false, reason, err
	}
	if err := downloadAndExtractWindowsCUDAArchive(ctx, managedWindowsCUDARuntimeURL, serverPath, false); err != nil {
		return false, reason, err
	}
	if err := os.WriteFile(managedServerSourceMarkerPath(serverPath), []byte(expectedManagedServerSourceMarker()+"\n"), 0o644); err != nil {
		slog.Warn("failed to write llama.cpp source marker", "path", managedServerSourceMarkerPath(serverPath), "error", err)
	}
	return true, reason, nil
}

func shouldInstallWindowsCUDAServerBundle(serverPath string, markerPath string) (bool, string) {
	if _, err := os.Stat(serverPath); err != nil {
		if os.IsNotExist(err) {
			return true, "missing_binary"
		}
		return false, ""
	}
	marker, err := os.ReadFile(markerPath)
	if err != nil {
		return true, "missing_source_marker_for_windows_cuda_runtime"
	}
	if strings.TrimSpace(string(marker)) != expectedManagedServerSourceMarker() {
		return true, "source_url_changed"
	}
	return false, ""
}

func managedServerSourceMarkerPath(serverPath string) string {
	return filepath.Join(filepath.Dir(strings.TrimSpace(serverPath)), managedServerSourceMarkerName)
}

func expectedManagedServerSourceMarker() string {
	return managedWindowsCUDAServerURL + "\ncuda_runtime_url=" + managedWindowsCUDARuntimeURL + "\ninstaller=ryvion-managed-llama-v3"
}

func downloadAndExtractWindowsCUDAArchive(ctx context.Context, sourceURL string, dst string, requireServer bool) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, sourceURL, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: HTTP %d", sourceURL, resp.StatusCode)
	}

	tmpZip, err := os.CreateTemp("", "ryv-v7-llama-cuda-*.zip")
	if err != nil {
		return err
	}
	tmpPath := tmpZip.Name()
	defer os.Remove(tmpPath)
	if _, err := io.Copy(tmpZip, resp.Body); err != nil {
		tmpZip.Close()
		return err
	}
	if err := tmpZip.Close(); err != nil {
		return err
	}

	zr, err := zip.OpenReader(tmpPath)
	if err != nil {
		return fmt.Errorf("zip: %w", err)
	}
	defer zr.Close()

	binDir := filepath.Dir(dst)
	foundServer := false
	for _, f := range zr.File {
		if f.FileInfo().IsDir() {
			continue
		}
		name := filepath.Base(f.Name)
		lower := strings.ToLower(name)
		isServer := lower == "llama-server.exe"
		isLib := strings.HasSuffix(lower, ".dll")
		if !isServer && !isLib {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return err
		}
		outPath := filepath.Join(binDir, name)
		if isServer {
			outPath = dst
		}
		out, err := os.OpenFile(outPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o755)
		if err != nil {
			rc.Close()
			return err
		}
		if _, err := io.Copy(out, rc); err != nil {
			out.Close()
			rc.Close()
			return err
		}
		out.Close()
		rc.Close()
		if isServer {
			foundServer = true
		}
	}
	if requireServer && !foundServer {
		return fmt.Errorf("llama-server.exe not found in Windows CUDA bundle")
	}
	return nil
}
