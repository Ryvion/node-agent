package llamacpp

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

type healthHTTPClient interface {
	Do(*http.Request) (*http.Response, error)
}

func CheckHealth(ctx context.Context, rawBaseURL string, client healthHTTPClient, timeout time.Duration) HealthResult {
	checkedAt := time.Now()
	rawBaseURL = strings.TrimRight(strings.TrimSpace(rawBaseURL), "/")
	if rawBaseURL == "" || !isLocalBaseURL(rawBaseURL) {
		return HealthResult{CheckedAt: checkedAt, Error: "llama.cpp health endpoint must be local http"}
	}
	if client == nil {
		client = defaultHealthHTTPClient(timeout)
	}
	if timeout <= 0 {
		timeout = defaultHealthTimeout
	}
	checkCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	errors := make([]string, 0, 2)
	for _, endpoint := range []string{"/health", "/v1/models"} {
		req, err := http.NewRequestWithContext(checkCtx, http.MethodGet, rawBaseURL+endpoint, nil)
		if err != nil {
			errors = append(errors, endpoint+": invalid request")
			continue
		}
		resp, err := client.Do(req)
		if err != nil {
			errors = append(errors, endpoint+": "+cleanStatusText(err.Error(), 120))
			continue
		}
		_, _ = io.CopyN(io.Discard, resp.Body, 1024)
		_ = resp.Body.Close()
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			return HealthResult{Healthy: true, Endpoint: endpoint, CheckedAt: checkedAt}
		}
		errors = append(errors, endpoint+": http "+resp.Status)
	}
	return HealthResult{CheckedAt: checkedAt, Error: cleanStatusText(strings.Join(errors, "; "), maxStatusReasonLen)}
}

func FetchServerProperties(ctx context.Context, rawBaseURL string, client healthHTTPClient, timeout time.Duration) (LlamaCppServerProperties, bool) {
	rawBaseURL = strings.TrimRight(strings.TrimSpace(rawBaseURL), "/")
	if rawBaseURL == "" || !isLocalBaseURL(rawBaseURL) {
		return LlamaCppServerProperties{}, false
	}
	if client == nil {
		client = defaultHealthHTTPClient(timeout)
	}
	if timeout <= 0 {
		timeout = defaultHealthTimeout
	}
	checkCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(checkCtx, http.MethodGet, rawBaseURL+"/props", nil)
	if err != nil {
		return LlamaCppServerProperties{}, false
	}
	resp, err := client.Do(req)
	if err != nil {
		return LlamaCppServerProperties{}, false
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		_, _ = io.CopyN(io.Discard, resp.Body, 1024)
		return LlamaCppServerProperties{}, false
	}
	var raw map[string]any
	decoder := json.NewDecoder(io.LimitReader(resp.Body, 64*1024))
	if err := decoder.Decode(&raw); err != nil {
		return LlamaCppServerProperties{}, false
	}
	props := LlamaCppServerProperties{
		BuildInfo:            firstStringProperty(raw, "build_info", "build"),
		SystemInfo:           firstStringProperty(raw, "system_info", "system"),
		ReportedGPULayers:    firstIntProperty(raw, "n_gpu_layers", "gpu_layers", "n-gpu-layers"),
		ReportedAcceleration: accelerationFromProperties(raw),
	}
	normalized := normalizeServerProperties(&props)
	if normalized == nil {
		return LlamaCppServerProperties{}, false
	}
	return *normalized, true
}

func defaultHealthHTTPClient(timeout time.Duration) *http.Client {
	if timeout <= 0 {
		timeout = defaultHealthTimeout
	}
	return &http.Client{Timeout: timeout}
}

func firstStringProperty(raw map[string]any, keys ...string) string {
	for _, key := range keys {
		if value, ok := findProperty(raw, key); ok {
			if text, ok := value.(string); ok {
				return text
			}
		}
	}
	return ""
}

func firstIntProperty(raw map[string]any, keys ...string) int {
	for _, key := range keys {
		if value, ok := findProperty(raw, key); ok {
			switch typed := value.(type) {
			case float64:
				if typed > 0 {
					return int(typed)
				}
			case string:
				n, err := strconv.Atoi(strings.TrimSpace(typed))
				if err == nil && n > 0 {
					return n
				}
			}
		}
	}
	return 0
}

func findProperty(value any, want string) (any, bool) {
	want = strings.ToLower(strings.TrimSpace(want))
	if want == "" {
		return nil, false
	}
	switch typed := value.(type) {
	case map[string]any:
		for key, child := range typed {
			if strings.ToLower(strings.TrimSpace(key)) == want {
				return child, true
			}
			if found, ok := findProperty(child, want); ok {
				return found, true
			}
		}
	case []any:
		for _, child := range typed {
			if found, ok := findProperty(child, want); ok {
				return found, true
			}
		}
	}
	return nil, false
}

func accelerationFromProperties(raw map[string]any) []string {
	parts := make([]string, 0, 4)
	for _, key := range []string{"build_info", "system_info", "build", "system"} {
		if value := firstStringProperty(raw, key); value != "" {
			parts = append(parts, value)
		}
	}
	combined := strings.ToLower(strings.Join(parts, " "))
	acceleration := []string{}
	if strings.Contains(combined, "cuda") || strings.Contains(combined, "cublas") {
		acceleration = append(acceleration, "cuda")
	}
	if strings.Contains(combined, "metal") {
		acceleration = append(acceleration, "metal")
	}
	if strings.Contains(combined, "vulkan") {
		acceleration = append(acceleration, "vulkan")
	}
	if len(acceleration) == 0 && strings.TrimSpace(combined) != "" {
		acceleration = append(acceleration, "cpu")
	}
	return normalizeAcceleration(acceleration)
}
