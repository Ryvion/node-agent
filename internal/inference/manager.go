package inference

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
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

	"github.com/Ryvion/node-agent/internal/runtimeexec"
)

const (
	defaultPort       = "8081"
	defaultThreads    = "4"
	defaultGPULayers  = "99"
	defaultCtxSize    = "16384"
	healthCheckPeriod = 5 * time.Second
	startupTimeout    = 120 * time.Second
)

// ModelMode distinguishes chat-style decoder models from encoder models that
// llama-server serves via its /v1/embeddings endpoint. An empty Mode defaults
// to chat — preserving the prior ModelConfig shape.
type ModelMode string

const (
	ModeChat      ModelMode = ""
	ModeEmbedding ModelMode = "embedding"
)

type ModelConfig struct {
	FileName string
	URL      string
	// Mode switches llama-server between chat/completion serving (default)
	// and embedding serving (--embedding flag, /v1/embeddings endpoint).
	Mode ModelMode
}

// NativeModels maps UI model names to GGUF downloads
var NativeModels = map[string]ModelConfig{
	"ryvion-llama-3.2-3b": {FileName: "Llama-3.2-3B-Instruct-Q4_K_M.gguf", URL: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"},
	"phi-4":               {FileName: "phi-4-Q4_K_M.gguf", URL: "https://huggingface.co/bartowski/phi-4-GGUF/resolve/main/phi-4-Q4_K_M.gguf"},
	"tinyllama":           {FileName: "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", URL: "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"},
	// Phase 1c: native embeddings. nomic-embed-text-v1.5 is 137M params,
	// 768-dim, matches OpenAI text-embedding-3-small quality on MTEB, and
	// the Q4_K_M GGUF is ~90MB. llama-server serves it via /v1/embeddings
	// when launched with --embedding.
	"nomic-embed-text-v1.5": {
		FileName: "nomic-embed-text-v1.5.Q4_K_M.gguf",
		URL:      "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf",
		Mode:     ModeEmbedding,
	},
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
	activeModelMode ModelMode
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

// amdSmokeTestPassed tracks whether the AMD GPU dry-run succeeded.
var amdSmokeTestPassed bool

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
		if err := checkDiskSpace(m.dataDir); err != nil {
			return fmt.Errorf("disk space check: %w", err)
		}
		slog.Info("downloading llama-server", "url", m.serverURL)
		if err := downloadAndExtractServer(ctx, m.serverURL, serverPath); err != nil {
			return fmt.Errorf("download llama-server: %w", err)
		}
		slog.Info("llama-server downloaded", "path", serverPath)
	}
	m.serverPath = serverPath

	// AMD GPU smoke test — verify ROCm compatibility before accepting work
	if _, err := os.Stat("/dev/kfd"); err == nil {
		slog.Info("AMD GPU detected, running compatibility smoke test")
		if err := m.runAMDSmokeTest(ctx, modelDir); err != nil {
			slog.Error("AMD GPU smoke test FAILED — GPU inference may not work",
				"error", err,
				"hint", "check ROCm version and gfx architecture compatibility")
			// Don't return error — still allow CPU inference and native mode
		} else {
			amdSmokeTestPassed = true
			slog.Info("AMD GPU smoke test PASSED — ROCm inference is operational")
		}
	}

	// Start server with auto-restart
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		m.mu.RLock()
		currentModel := m.activeModelName
		customPath := m.activeModelPath
		m.mu.RUnlock()

		var modelPath string
		if customPath != "" && strings.HasSuffix(customPath, ".gguf") {
			// Custom model — path was already set by EnsureCustomModel
			if _, err := os.Stat(customPath); err == nil {
				modelPath = customPath
			}
		}
		var activeMode ModelMode
		if modelPath == "" {
			// Native registry model
			cfg, ok := NativeModels[currentModel]
			if !ok {
				currentModel = "ryvion-llama-3.2-3b"
				cfg = NativeModels[currentModel]
			}
			activeMode = cfg.Mode
			modelPath = filepath.Join(modelDir, cfg.FileName)
			if _, err := os.Stat(modelPath); os.IsNotExist(err) {
				if err := checkDiskSpace(m.dataDir); err != nil {
					slog.Error("disk space check failed before model download", "error", err)
					time.Sleep(5 * time.Second)
					continue
				}
				slog.Info("downloading model", "model", currentModel, "url", cfg.URL)
				if err := downloadFile(ctx, cfg.URL, modelPath); err != nil {
					slog.Error("failed to download model", "error", err)
					time.Sleep(5 * time.Second)
					continue
				}
				slog.Info("model downloaded", "path", modelPath)
			}
		}

		m.mu.Lock()
		m.activeModelPath = modelPath
		m.activeModelMode = activeMode
		m.mu.Unlock()

		slog.Info("starting llama-server", "port", m.port, "model", modelPath, "mode", string(activeMode))
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

// EnsureCustomModel downloads a custom GGUF model from a URL and hot-swaps the server to use it.
func (m *Manager) EnsureCustomModel(ctx context.Context, modelName, modelURL string) error {
	m.mu.RLock()
	current := m.activeModelName
	m.mu.RUnlock()

	// Download model if not already cached
	modelsDir := filepath.Join(m.dataDir, "models")
	os.MkdirAll(modelsDir, 0755)
	// Use a hash of the URL as the filename to cache per-URL
	h := sha256.Sum256([]byte(modelURL))
	fileName := modelName + "-" + hex.EncodeToString(h[:8]) + ".gguf"
	modelPath := filepath.Join(modelsDir, fileName)

	if _, err := os.Stat(modelPath); err != nil {
		if err := checkDiskSpace(m.dataDir); err != nil {
			return fmt.Errorf("disk space check: %w", err)
		}
		slog.Info("downloading custom model", "name", modelName, "path", modelPath)
		if err := downloadFile(ctx, modelURL, modelPath); err != nil {
			return fmt.Errorf("download custom model: %w", err)
		}
		if err := validateGGUF(modelPath); err != nil {
			os.Remove(modelPath)
			return fmt.Errorf("invalid custom model file: %w", err)
		}
		slog.Info("custom model downloaded", "name", modelName, "path", modelPath)
	}

	// If already loaded, skip restart
	if current == fileName {
		return nil
	}

	slog.Info("switching to custom model", "from", current, "to", modelName)

	m.mu.Lock()
	m.activeModelName = fileName
	m.activeModelPath = modelPath
	if m.cancel != nil {
		m.cancel()
	}
	m.mu.Unlock()

	// Wait for server to become healthy with the new model
	for i := 0; i < 300; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(1 * time.Second):
			m.mu.RLock()
			hc := m.healthy
			m.mu.RUnlock()
			if hc {
				return nil
			}
		}
	}
	return fmt.Errorf("timeout waiting for custom model %s to start", modelName)
}

func (m *Manager) setHealthy(v bool) {
	m.mu.Lock()
	m.healthy = v
	m.mu.Unlock()
}

func (m *Manager) runServer(ctx context.Context) error {
	m.setHealthy(false)

	m.mu.RLock()
	modelPath := m.activeModelPath
	port := m.port
	m.mu.RUnlock()

	// Try containerized mode first (more secure — isolates GGUF parsing)
	if m.useContainerizedInference() {
		return m.runServerContainerized(ctx, modelPath, port)
	}

	// Fallback: native mode
	return m.runServerNative(ctx, modelPath, port)
}

// useContainerizedInference reports whether a managed OCI backend is available for sandboxed inference.
// On Windows, always prefer native mode — native inference is more reliable than GPU passthrough through the container backend
// containers is unreliable (OOM kills, exit 137). The native llama-server.exe with CUDA works better.
func (m *Manager) useContainerizedInference() bool {
	if os.Getenv("RYV_NATIVE_INFERENCE_ONLY") == "1" {
		return false
	}
	if runtime.GOOS == "windows" {
		return false
	}
	backend, err := runtimeexec.ResolveBackendCommand(runtime.GOOS, os.Getenv)
	if err != nil {
		return false
	}
	cmd := exec.Command(backend, "info")
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard
	return cmd.Run() == nil
}

// runServerContainerized runs llama-server inside the container backend with GPU passthrough.
func (m *Manager) runServerContainerized(ctx context.Context, modelPath, port string) error {
	serverCtx, cancel := context.WithCancel(ctx)
	m.mu.Lock()
	m.cancel = cancel
	m.mu.Unlock()
	defer cancel()

	modelDir := filepath.Dir(modelPath)
	modelFile := filepath.Base(modelPath)

	image := os.Getenv("RYV_INFERENCE_IMAGE")
	if image == "" {
		image = "ghcr.io/ggml-org/llama.cpp:server"
	}

	args := []string{
		"run", "--rm",
		"--name", "ryvion-inference",
		// Security constraints
		"--security-opt=no-new-privileges:true",
		"--cap-drop=ALL",
		"--memory=8g",
		"--pids-limit=256",
	}

	// Detect GPU type and add passthrough
	if _, err := exec.Command("nvidia-smi").Output(); err == nil {
		args = append(args, "--gpus", "all")
	} else if _, err := os.Stat("/dev/kfd"); err == nil {
		// ROCm (AMD) GPU
		args = append(args, "--device=/dev/kfd", "--device=/dev/dri", "--group-add=video")
	}

	// Mount model directory read-only, expose the port
	args = append(args,
		"-v", modelDir+":/models:ro",
		"-p", port+":"+port,
		image,
		"--model", "/models/"+modelFile,
		"--port", port,
		"--host", "0.0.0.0",
		"--threads", m.threads,
		"--ctx-size", m.ctxSize,
	)

	// GPU layers (skip on macOS where Metal isn't available inside containers)
	if runtime.GOOS != "darwin" {
		args = append(args, "--n-gpu-layers", m.gpuLayers)
	}

	slog.Info("starting containerized llama-server",
		"image", image,
		"model", modelFile,
		"port", port,
	)

	ociExec, err := runtimeexec.ResolveExecutor(runtime.GOOS, os.Getenv)
	if err != nil {
		slog.Warn("containerized inference executor unavailable, falling back to native", "error", err)
		return m.runServerNative(ctx, modelPath, port)
	}
	execArgs := append(append([]string{}, ociExec.PrefixArgs...), args...)
	cmd := exec.CommandContext(serverCtx, ociExec.Command, execArgs...)
	// Send container output to a log file
	logPath := filepath.Join(m.dataDir, "llama-server.log")
	if logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644); err == nil {
		cmd.Stdout = logFile
		cmd.Stderr = logFile
		defer logFile.Close()
	} else {
		cmd.Stdout = io.Discard
		cmd.Stderr = io.Discard
	}

	m.mu.Lock()
	m.cmd = cmd
	m.mu.Unlock()

	if err := cmd.Start(); err != nil {
		slog.Warn("containerized inference failed to start, falling back to native", "error", err)
		return m.runServerNative(ctx, modelPath, port)
	}

	go m.healthLoop(serverCtx)

	waitErr := cmd.Wait()

	// Clean up container on exit (may already be removed by --rm, but be safe)
	cleanupArgs := append(append([]string{}, ociExec.PrefixArgs...), "rm", "-f", "ryvion-inference")
	cleanup := exec.Command(ociExec.Command, cleanupArgs...)
	cleanup.Stdout = io.Discard
	cleanup.Stderr = io.Discard
	cleanup.Run()

	if waitErr != nil {
		slog.Error("containerized llama-server exited with error",
			"error", waitErr,
			"model", modelPath,
			"log_file", logPath,
		)
	}
	return waitErr
}

// runServerNative runs llama-server directly on the host.
func (m *Manager) runServerNative(ctx context.Context, modelPath, port string) error {
	serverCtx, cancel := context.WithCancel(ctx)
	m.mu.Lock()
	m.cancel = cancel
	m.mu.Unlock()
	defer cancel()

	args := []string{
		"--model", modelPath,
		"--port", port,
		"--host", "127.0.0.1",
		"--threads", m.threads,
		"--ctx-size", m.ctxSize,
		"--log-disable",
	}

	m.mu.RLock()
	mode := m.activeModelMode
	m.mu.RUnlock()
	if mode == ModeEmbedding {
		// --embedding routes the server's /v1/embeddings endpoint at this
		// model. Pooling=mean is the sensible default for encoder-style
		// models like nomic-embed-text / bge.
		args = append(args, "--embedding", "--pooling", "mean")
	}

	// GPU offloading: Metal on macOS, CUDA on Linux/Windows
	// --n-gpu-layers=99 offloads all layers to GPU when available.
	// llama.cpp gracefully falls back to CPU if no GPU is detected.
	switch runtime.GOOS {
	case "darwin":
		// Metal acceleration (ARM64 and AMD64 with Metal)
		args = append(args, "--n-gpu-layers", m.gpuLayers)
	case "linux", "windows":
		// CUDA acceleration if available; llama.cpp ignores flag if no GPU
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

	slog.Info("launching llama-server (native)",
		"binary", m.serverPath,
		"model", modelPath,
		"port", port,
		"threads", m.threads,
		"gpu_layers", m.gpuLayers,
		"ctx_size", m.ctxSize,
		"os", runtime.GOOS,
		"arch", runtime.GOARCH,
	)

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start llama-server: %w", err)
	}

	// Lower priority so operator workloads (games, etc.) take precedence.
	if cmd.Process != nil {
		setLowPriority(cmd.Process.Pid)
	}

	// Wait for health check to pass
	go m.healthLoop(serverCtx)

	waitErr := cmd.Wait()
	if waitErr != nil {
		slog.Error("llama-server process exited with error",
			"error", waitErr,
			"model", modelPath,
			"log_file", logPath,
		)
	}
	return waitErr
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
	client := &http.Client{Timeout: 30 * time.Minute}
	resp, err := client.Do(req)
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

// runAMDSmokeTest downloads a tiny model and runs a quick inference to verify ROCm works.
// If the gfx architecture is incompatible, llama-server will segfault — we catch that here.
func (m *Manager) runAMDSmokeTest(ctx context.Context, modelDir string) error {
	// Use tinyllama as the smoke test model — small enough to download quickly
	testModel := filepath.Join(modelDir, "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
	if _, err := os.Stat(testModel); os.IsNotExist(err) {
		cfg, ok := NativeModels["tinyllama"]
		if !ok {
			return fmt.Errorf("tinyllama not in native registry for smoke test")
		}
		if err := checkDiskSpace(m.dataDir); err != nil {
			return fmt.Errorf("disk space check: %w", err)
		}
		slog.Info("downloading smoke test model", "model", cfg.FileName)
		if err := downloadFile(ctx, cfg.URL, testModel); err != nil {
			return fmt.Errorf("download smoke test model: %w", err)
		}
	}

	// Try running llama-server with the test model for a quick health check
	testCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()

	port := "18081" // Use a different port to not conflict with main server
	args := []string{
		"--model", testModel,
		"--port", port,
		"--host", "127.0.0.1",
		"--threads", "2",
		"--ctx-size", "512",
		"--n-gpu-layers", "99",
	}

	// For older RDNA2 cards, try injecting HSA_OVERRIDE_GFX_VERSION
	gfxVersion := os.Getenv("HSA_OVERRIDE_GFX_VERSION")

	cmd := exec.CommandContext(testCtx, m.serverPath, args...)
	cmd.Env = append(os.Environ(), "CUDA_VISIBLE_DEVICES=0")
	if gfxVersion != "" {
		cmd.Env = append(cmd.Env, "HSA_OVERRIDE_GFX_VERSION="+gfxVersion)
	}
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start smoke test server: %w", err)
	}

	// Wait for health check
	healthURL := "http://127.0.0.1:" + port + "/health"
	passed := false
	for i := 0; i < 30; i++ {
		time.Sleep(2 * time.Second)
		resp, err := http.Get(healthURL)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				passed = true
				break
			}
		}
		// Check if process died (segfault)
		if cmd.ProcessState != nil && cmd.ProcessState.Exited() {
			break
		}
	}

	// Kill the test server
	if cmd.Process != nil {
		cmd.Process.Kill()
	}
	cmd.Wait()

	if !passed {
		return fmt.Errorf("smoke test failed — llama-server did not become healthy (possible gfx incompatibility)")
	}
	return nil
}
