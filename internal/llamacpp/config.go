package llamacpp

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	EnvServerURL  = "RYV_LLAMA_CPP_SERVER_URL"
	EnvServerURL2 = "RYV_LLAMA_CPP_URL"
	EnvModel      = "RYV_LLAMA_CPP_MODEL"
	EnvModelAlias = "RYV_LLAMA_CPP_MODEL_ALIAS"
)

type Config struct {
	ServerURL    string
	Model        string
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
	return Config{
		ServerURL:    strings.TrimRight(firstNonEmpty(getenv(EnvServerURL), getenv(EnvServerURL2)), "/"),
		Model:        cleanText(firstNonEmpty(getenv(EnvModel), getenv(EnvModelAlias)), MaxModelIDLen),
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
