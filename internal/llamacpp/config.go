package llamacpp

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	EnvServerURL  = "RYV_LLAMA_CPP_SERVER_URL"
	EnvServerURL2 = "RYV_LLAMA_CPP_URL"
	EnvModel      = "RYV_LLAMA_CPP_MODEL"
	EnvModelAlias = "RYV_LLAMA_CPP_MODEL_ALIAS"

	EnvEnabled    = "RYV_LLAMA_CPP_ENABLED"
	EnvServerPath = "RYV_LLAMA_CPP_SERVER_PATH"
	EnvModelPath  = "RYV_LLAMA_CPP_MODEL_PATH"
	EnvHost       = "RYV_LLAMA_CPP_HOST"
	EnvPort       = "RYV_LLAMA_CPP_PORT"
	EnvCtxSize    = "RYV_LLAMA_CPP_CTX_SIZE"
	EnvKeepWarm   = "RYV_LLAMA_CPP_KEEP_WARM"

	DefaultHost    = "127.0.0.1"
	DefaultPort    = 45910
	DefaultCtxSize = 4096
)

type Config struct {
	ServerURL    string
	Model        string
	ServerPath   string
	ModelPath    string
	Host         string
	Port         int
	CtxSize      int
	Enabled      bool
	KeepWarm     bool
	ProbeTimeout time.Duration
	HTTPTimeout  time.Duration
}

type Health struct {
	Available bool
	Endpoint  string
	Model     string
	Error     string
}

func ResolveConfig(getenv func(string) string) Config {
	if getenv == nil {
		getenv = os.Getenv
	}
	serverPath := cleanText(getenv(EnvServerPath), MaxPromptBytes)
	modelPath := cleanText(getenv(EnvModelPath), MaxPromptBytes)
	enabled := envBool(getenv(EnvEnabled))
	keepWarm := envBool(getenv(EnvKeepWarm))
	rawHost := strings.TrimSpace(getenv(EnvHost))
	host := normalizeHost(rawHost)
	port := envInt(getenv(EnvPort), 0)
	hostConfigured := rawHost != ""
	legacyEndpointConfigured := hostConfigured || port > 0 || enabled || keepWarm || serverPath != "" || modelPath != ""
	if legacyEndpointConfigured {
		if host == "" && !hostConfigured {
			host = DefaultHost
		}
		if port == 0 {
			port = DefaultPort
		}
	}
	serverURL := firstNonEmpty(getenv(EnvServerURL), getenv(EnvServerURL2))
	if serverURL == "" && host != "" && port > 0 {
		serverURL = fmt.Sprintf("http://%s:%d", host, port)
	}
	model := firstNonEmpty(getenv(EnvModel), getenv(EnvModelAlias))
	if model == "" {
		model = modelIDFromPath(modelPath)
	}
	return Config{
		ServerURL:    strings.TrimRight(serverURL, "/"),
		Model:        cleanText(model, MaxModelIDLen),
		ServerPath:   serverPath,
		ModelPath:    modelPath,
		Host:         host,
		Port:         port,
		CtxSize:      envInt(getenv(EnvCtxSize), DefaultCtxSize),
		Enabled:      enabled,
		KeepWarm:     keepWarm,
		ProbeTimeout: envDuration(getenv, "RYV_LLAMA_CPP_PROBE_TIMEOUT", 2*time.Second),
		HTTPTimeout:  envDuration(getenv, "RYV_LLAMA_CPP_HTTP_TIMEOUT", 10*time.Minute),
	}
}

func (c Config) Validate() error {
	if strings.TrimSpace(c.ServerURL) == "" {
		return fmt.Errorf("llama.cpp server URL required")
	}
	if !IsLocalHTTPBaseURL(c.ServerURL) {
		return fmt.Errorf("llama.cpp server URL must be local HTTP")
	}
	return nil
}

func (c Config) ModelFor(spec Spec) string {
	if strings.TrimSpace(spec.Model) != "" {
		return strings.TrimSpace(spec.Model)
	}
	return strings.TrimSpace(c.Model)
}

func (c Config) ManagedServerEnabled() bool {
	if !(c.Enabled || c.KeepWarm) {
		return false
	}
	return strings.TrimSpace(c.ServerPath) != "" || strings.TrimSpace(c.ModelPath) != ""
}

func BuildManagedServerArgs(cfg Config) []string {
	host := normalizeHost(cfg.Host)
	if host == "" {
		host = DefaultHost
	}
	port := cfg.Port
	if port <= 0 || port > 65535 {
		port = DefaultPort
	}
	args := []string{
		"--model", strings.TrimSpace(cfg.ModelPath),
		"--host", host,
		"--port", strconv.Itoa(port),
	}
	if cfg.CtxSize > 0 {
		args = append(args, "--ctx-size", strconv.Itoa(cfg.CtxSize))
	}
	return args
}

func StartManagedServer(ctx context.Context, cfg Config, output ioWriter) (*exec.Cmd, error) {
	if !cfg.ManagedServerEnabled() {
		return nil, nil
	}
	cfg.ServerPath = strings.TrimSpace(cfg.ServerPath)
	cfg.ModelPath = strings.TrimSpace(cfg.ModelPath)
	if cfg.ServerPath == "" {
		return nil, fmt.Errorf("llama.cpp server path required")
	}
	if cfg.ModelPath == "" {
		return nil, fmt.Errorf("llama.cpp model path required")
	}
	if _, err := os.Stat(cfg.ServerPath); err != nil {
		return nil, fmt.Errorf("llama.cpp server not available: %w", err)
	}
	if _, err := os.Stat(cfg.ModelPath); err != nil {
		return nil, fmt.Errorf("llama.cpp model not available: %w", err)
	}
	if err := (Config{ServerURL: cfg.ServerURL}).Validate(); err != nil {
		return nil, err
	}
	cmd := exec.CommandContext(ctx, cfg.ServerPath, BuildManagedServerArgs(cfg)...)
	cmd.Env = os.Environ()
	if output != nil {
		cmd.Stdout = output
		cmd.Stderr = output
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("llama.cpp server start failed: %w", err)
	}
	return cmd, nil
}

func Probe(ctx context.Context, cfg Config, client *http.Client) Health {
	cfg.ServerURL = strings.TrimRight(strings.TrimSpace(cfg.ServerURL), "/")
	if err := cfg.Validate(); err != nil {
		return Health{Error: err.Error(), Model: strings.TrimSpace(cfg.Model)}
	}
	timeout := cfg.ProbeTimeout
	if timeout <= 0 {
		timeout = 2 * time.Second
	}
	if client == nil {
		client = &http.Client{Timeout: timeout}
	}
	probeCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	for _, endpoint := range []string{"/health", "/v1/models"} {
		req, err := http.NewRequestWithContext(probeCtx, http.MethodGet, cfg.ServerURL+endpoint, nil)
		if err != nil {
			continue
		}
		resp, err := client.Do(req)
		if err != nil {
			continue
		}
		_ = resp.Body.Close()
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			return Health{Available: true, Endpoint: endpoint, Model: strings.TrimSpace(cfg.Model)}
		}
	}
	return Health{Model: strings.TrimSpace(cfg.Model), Error: "llama.cpp local server not reachable"}
}

func IsLocalHTTPBaseURL(raw string) bool {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return false
	}
	if parsed.Scheme != "http" {
		return false
	}
	host := strings.ToLower(strings.Trim(parsed.Hostname(), "[]"))
	switch host {
	case "localhost", "127.0.0.1", "::1":
		return true
	default:
		return false
	}
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func modelIDFromPath(path string) string {
	base := strings.TrimSpace(filepath.Base(path))
	if base == "." || base == string(filepath.Separator) {
		return ""
	}
	for _, suffix := range []string{".gguf", ".bin"} {
		if strings.HasSuffix(strings.ToLower(base), suffix) {
			return strings.TrimSuffix(base, base[len(base)-len(suffix):])
		}
	}
	return base
}

func normalizeHost(value string) string {
	host := strings.Trim(strings.ToLower(strings.TrimSpace(value)), "[]")
	switch host {
	case "", "localhost", "127.0.0.1", "::1":
		return firstNonEmpty(host, DefaultHost)
	default:
		return ""
	}
}

func envBool(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "yes", "y", "on", "enabled":
		return true
	default:
		return false
	}
}

func envInt(value string, fallback int) int {
	raw := strings.TrimSpace(value)
	if raw == "" {
		return fallback
	}
	out, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	return out
}

func envDuration(getenv func(string) string, name string, fallback time.Duration) time.Duration {
	raw := strings.TrimSpace(getenv(name))
	if raw == "" {
		return fallback
	}
	if parsed, err := time.ParseDuration(raw); err == nil && parsed > 0 {
		return parsed
	}
	if seconds, err := strconv.Atoi(raw); err == nil && seconds > 0 {
		return time.Duration(seconds) * time.Second
	}
	return fallback
}

type ioWriter interface {
	Write([]byte) (int, error)
}
