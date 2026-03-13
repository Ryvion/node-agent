package inference

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

const (
	defaultPort       = "8081"
	defaultThreads    = "4"
	defaultGPULayers  = "99"
	defaultCtxSize    = "16384"
	healthCheckPeriod = 5 * time.Second
	startupTimeout    = 120 * time.Second
)

type ModelConfig struct {
	FileName string
	URL      string
}

// NativeModels maps UI model names to GGUF downloads
var NativeModels = map[string]ModelConfig{
	"ryvion-llama-3.2-3b": {"Llama-3.2-3B-Instruct-Q4_K_M.gguf", "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"},
	"phi-4":               {"phi-4-Q4_K_M.gguf", "https://huggingface.co/bartowski/phi-4-GGUF/resolve/main/phi-4-Q4_K_M.gguf"},
	"tinyllama":           {"tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"},
}

// platformServerURL returns the correct llama.cpp release URL for the current OS/arch.
func platformServerURL() string {
	const base = "https://github.com/ggml-org/llama.cpp/releases/download/b8106/"
	switch runtime.GOOS + "/" + runtime.GOARCH {
	case "darwin/arm64":
		return base + "llama-b8106-bin-macos-arm64.tar.gz"
	case "darwin/amd64":
		return base + "llama-b8106-bin-macos-x64.tar.gz"
	case "linux/amd64":
		// CUDA 12 build for Linux GPU nodes
		return base + "llama-b8106-bin-ubuntu-x64.tar.gz"
	case "linux/arm64":
		return base + "llama-b8106-bin-ubuntu-arm64.tar.gz"
	case "windows/amd64":
		return base + "llama-b8106-bin-win-cuda-12.4-x64.zip"
	case "windows/arm64":
		return base + "llama-b8106-bin-win-cpu-arm64.zip"
	default:
		// Windows or unsupported — caller should check
		return ""
	}
}

// NativeRuntimeAvailable reports whether this host can run the bundled native
// inference server with the current defaults.
func NativeRuntimeAvailable() bool {
	if strings.TrimSpace(os.Getenv("RYV_SERVER_URL")) != "" {
		return true
	}
	return platformServerURL() != ""
}

type Manager struct {
	dataDir    string
	port       string
	threads    string
	gpuLayers  string
	ctxSize    string
	serverURL  string
	serverPath string

	mu              sync.RWMutex
	healthy         bool
	cmd             *exec.Cmd
	cancel          context.CancelFunc
	activeModelName string
	activeModelPath string
}

func New(dataDir string) *Manager {
	if dataDir == "" {
		home, _ := os.UserHomeDir()
		dataDir = filepath.Join(home, ".ryvion")
	}
	port := envOr("RYV_INFERENCE_PORT", defaultPort)
	return &Manager{
		dataDir:         dataDir,
		port:            port,
		threads:         envOr("RYV_INFERENCE_THREADS", defaultThreads),
		gpuLayers:       envOr("RYV_GPU_LAYERS", defaultGPULayers),
		ctxSize:         envOr("RYV_CTX_SIZE", defaultCtxSize),
		serverURL:       envOr("RYV_SERVER_URL", platformServerURL()),
		activeModelName: "ryvion-llama-3.2-3b",
	}
}

func (m *Manager) Start(ctx context.Context) error {
	if m.serverURL == "" {
		slog.Info("inference manager: no llama-server binary available for this platform, skipping",
			"os", runtime.GOOS, "arch", runtime.GOARCH)
		<-ctx.Done()
		return ctx.Err()
	}

	binDir := filepath.Join(m.dataDir, "bin")
	modelDir := filepath.Join(m.dataDir, "models")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		return fmt.Errorf("create bin dir: %w", err)
	}
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		return fmt.Errorf("create model dir: %w", err)
	}

	serverPath := filepath.Join(binDir, serverBinaryName())
	if _, err := os.Stat(serverPath); os.IsNotExist(err) {
		slog.Info("downloading llama-server", "url", m.serverURL)
		if err := downloadAndExtractServer(ctx, m.serverURL, serverPath); err != nil {
			return fmt.Errorf("download llama-server: %w", err)
		}
		slog.Info("llama-server downloaded", "path", serverPath)
	}
	m.serverPath = serverPath

	// Start server with auto-restart
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		m.mu.RLock()
		currentModel := m.activeModelName
		m.mu.RUnlock()

		cfg, ok := NativeModels[currentModel]
		if !ok {
			currentModel = "ryvion-llama-3.2-3b"
			cfg = NativeModels[currentModel]
		}

		modelPath := filepath.Join(modelDir, cfg.FileName)
		if _, err := os.Stat(modelPath); os.IsNotExist(err) {
			slog.Info("downloading model", "model", currentModel, "url", cfg.URL)
			if err := downloadFile(ctx, cfg.URL, modelPath); err != nil {
				slog.Error("failed to download model", "error", err)
				time.Sleep(5 * time.Second)
				continue
			}
			slog.Info("model downloaded", "path", modelPath)
		}

		m.mu.Lock()
		m.activeModelPath = modelPath
		m.mu.Unlock()

		slog.Info("starting llama-server", "port", m.port, "model", modelPath)
		if err := m.runServer(ctx); err != nil {
			slog.Warn("llama-server exited", "error", err)
		}
		m.setHealthy(false)
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(3 * time.Second):
			slog.Info("restarting llama-server")
		}
	}
}

func (m *Manager) Stop() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.cancel != nil {
		m.cancel()
	}
	m.healthy = false
}

func (m *Manager) Healthy() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.healthy
}

func (m *Manager) ServerURL() string {
	return "http://localhost:" + m.port
}

func (m *Manager) ModelName() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.activeModelName
}

func (m *Manager) EnsureModel(ctx context.Context, modelName string) error {
	m.mu.RLock()
	current := m.activeModelName
	m.mu.RUnlock()

	if current == modelName {
		return nil
	}

	_, ok := NativeModels[modelName]
	if !ok {
		return fmt.Errorf("model %s not supported in native registry", modelName)
	}

	slog.Info("switching native model", "from", current, "to", modelName)

	m.mu.Lock()
	m.activeModelName = modelName
	if m.cancel != nil {
		m.cancel() // Stop the server, it will restart with the new model
	}
	m.mu.Unlock()

	// Wait for the new model's server to become healthy
	for i := 0; i < 300; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(1 * time.Second):
			m.mu.RLock()
			hc := m.healthy
			am := m.activeModelName
			m.mu.RUnlock()
			if hc && am == modelName {
				return nil
			}
		}
	}
	return fmt.Errorf("timeout waiting for %s to start", modelName)
}

func (m *Manager) setHealthy(v bool) {
	m.mu.Lock()
	m.healthy = v
	m.mu.Unlock()
}

func (m *Manager) runServer(ctx context.Context) error {
	serverCtx, cancel := context.WithCancel(ctx)
	m.mu.Lock()
	m.cancel = cancel
	m.mu.Unlock()
	defer cancel()

	args := []string{
		"--model", m.activeModelPath,
		"--port", m.port,
		"--host", "127.0.0.1",
		"--threads", m.threads,
		"--ctx-size", m.ctxSize,
		"--log-disable",
	}

	// Metal acceleration on macOS ARM64
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		args = append(args, "--n-gpu-layers", m.gpuLayers)
	}

	cmd := exec.CommandContext(serverCtx, m.serverPath, args...)
	// Send llama-server output to a log file to avoid mixing with JSON slog.
	logPath := filepath.Join(m.dataDir, "llama-server.log")
	if logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644); err == nil {
		cmd.Stdout = logFile
		cmd.Stderr = logFile
		defer logFile.Close()
	} else {
		cmd.Stdout = io.Discard
		cmd.Stderr = io.Discard
	}
	// Set library path so llama-server finds its shared libs
	binDir := filepath.Dir(m.serverPath)
	env := os.Environ()
	if runtime.GOOS == "windows" {
		// On Windows, colocated CUDA/llama DLLs are resolved through PATH.
		env = append(env, "PATH="+binDir+";"+os.Getenv("PATH"))
	} else {
		env = append(env,
			"DYLD_LIBRARY_PATH="+binDir,
			"LD_LIBRARY_PATH="+binDir,
		)
	}
	cmd.Env = env

	m.mu.Lock()
	m.cmd = cmd
	m.mu.Unlock()

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start llama-server: %w", err)
	}

	// Lower priority so operator workloads (games, etc.) take precedence.
	if cmd.Process != nil {
		setLowPriority(cmd.Process.Pid)
	}

	// Wait for health check to pass
	go m.healthLoop(serverCtx)

	return cmd.Wait()
}

func (m *Manager) healthLoop(ctx context.Context) {
	url := m.ServerURL() + "/health"
	client := &http.Client{Timeout: 3 * time.Second}

	// Initial startup: wait up to startupTimeout for first healthy response
	deadline := time.After(startupTimeout)
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-deadline:
			slog.Warn("llama-server failed to become healthy within timeout")
			return
		case <-ticker.C:
			if checkHealth(ctx, client, url) {
				slog.Info("llama-server is healthy", "url", m.ServerURL())
				m.setHealthy(true)
				goto monitoring
			}
		}
	}

monitoring:
	ticker.Reset(healthCheckPeriod)
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if !checkHealth(ctx, client, url) {
				slog.Warn("llama-server health check failed")
				m.setHealthy(false)
			} else if !m.Healthy() {
				slog.Info("llama-server recovered")
				m.setHealthy(true)
			}
		}
	}
}

func checkHealth(ctx context.Context, client *http.Client, url string) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return false
	}
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// downloadFile downloads a URL to a local file with progress logging.
func downloadFile(ctx context.Context, url, dst string) error {
	tmp := dst + ".tmp"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: HTTP %d", url, resp.StatusCode)
	}

	f, err := os.Create(tmp)
	if err != nil {
		return err
	}

	total := resp.ContentLength
	pw := &progressWriter{dst: f, total: total, label: filepath.Base(dst)}
	if _, err := io.Copy(pw, resp.Body); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	f.Close()
	return os.Rename(tmp, dst)
}

type progressWriter struct {
	dst     io.Writer
	total   int64
	written int64
	label   string
	lastLog time.Time
}

func (pw *progressWriter) Write(p []byte) (int, error) {
	n, err := pw.dst.Write(p)
	pw.written += int64(n)
	if time.Since(pw.lastLog) > 5*time.Second {
		pw.lastLog = time.Now()
		if pw.total > 0 {
			pct := float64(pw.written) / float64(pw.total) * 100
			slog.Info("downloading", "file", pw.label, "progress", fmt.Sprintf("%.1f%%", pct),
				"downloaded_mb", pw.written/(1024*1024), "total_mb", pw.total/(1024*1024))
		} else {
			slog.Info("downloading", "file", pw.label, "downloaded_mb", pw.written/(1024*1024))
		}
	}
	return n, err
}

func serverBinaryName() string {
	if runtime.GOOS == "windows" {
		return "llama-server.exe"
	}
	return "llama-server"
}

// downloadAndExtractServer downloads a llama.cpp release and extracts
// llama-server plus required shared libraries.
func downloadAndExtractServer(ctx context.Context, url, dst string) error {
	if strings.HasSuffix(strings.ToLower(url), ".zip") {
		return downloadAndExtractServerZip(ctx, url, dst)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: HTTP %d", url, resp.StatusCode)
	}

	gz, err := gzip.NewReader(resp.Body)
	if err != nil {
		return fmt.Errorf("gzip: %w", err)
	}
	defer gz.Close()

	binDir := filepath.Dir(dst)
	foundServer := false

	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("tar: %w", err)
		}

		name := filepath.Base(hdr.Name)
		isServer := name == serverBinaryName()
		isLib := strings.HasSuffix(name, ".dylib") || strings.HasSuffix(name, ".so") ||
			strings.Contains(name, ".so.")

		if !isServer && !isLib {
			continue
		}

		outPath := filepath.Join(binDir, name)

		switch hdr.Typeflag {
		case tar.TypeSymlink:
			// Recreate symlink (e.g. libmtmd.0.dylib → libmtmd.0.0.8106.dylib)
			os.Remove(outPath)
			target := filepath.Base(hdr.Linkname)
			if err := os.Symlink(target, outPath); err != nil {
				slog.Warn("failed to create symlink", "name", name, "target", target, "error", err)
			}
		case tar.TypeReg:
			perm := os.FileMode(0o644)
			if isServer {
				perm = 0o755
			}
			f, err := os.OpenFile(outPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, perm)
			if err != nil {
				return err
			}
			if _, err := io.Copy(f, tr); err != nil {
				f.Close()
				return err
			}
			f.Close()
			if isServer {
				foundServer = true
				slog.Info("extracted llama-server", "path", outPath, "size", hdr.Size)
			}
		}
	}
	if !foundServer {
		return fmt.Errorf("llama-server binary not found in archive")
	}
	return nil
}

func downloadAndExtractServerZip(ctx context.Context, url, dst string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: HTTP %d", url, resp.StatusCode)
	}

	tmpZip, err := os.CreateTemp("", "ryv-llama-*.zip")
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

	stat, err := os.Stat(tmpPath)
	if err != nil {
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
		lc := strings.ToLower(name)
		isServer := lc == strings.ToLower(serverBinaryName())
		isLib := strings.HasSuffix(lc, ".dll") || strings.HasSuffix(lc, ".so") || strings.HasSuffix(lc, ".dylib") ||
			strings.Contains(lc, ".so.")
		if !isServer && !isLib {
			continue
		}

		rc, err := f.Open()
		if err != nil {
			return err
		}

		outPath := filepath.Join(binDir, name)
		perm := os.FileMode(0o644)
		if isServer {
			perm = 0o755
		}
		out, err := os.OpenFile(outPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, perm)
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
			slog.Info("extracted llama-server", "path", outPath, "size", f.UncompressedSize64, "zip_size", stat.Size())
		}
	}

	if !foundServer {
		return fmt.Errorf("llama-server binary not found in zip archive")
	}
	return nil
}

func envOr(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}
