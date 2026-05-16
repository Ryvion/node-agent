package sglang

import (
	"context"
	"io"
	"net/http"
	"strings"
	"time"
)

const defaultHealthTimeout = 750 * time.Millisecond

type healthHTTPClient interface {
	Do(*http.Request) (*http.Response, error)
}

func CheckHealth(ctx context.Context, rawBaseURL string, client healthHTTPClient, timeout time.Duration) HealthResult {
	checkedAt := time.Now()
	rawBaseURL = strings.TrimRight(strings.TrimSpace(rawBaseURL), "/")
	if rawBaseURL == "" || !isLocalBaseURL(rawBaseURL) {
		return HealthResult{CheckedAt: checkedAt, Error: "SGLang health endpoint must be local http"}
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

func defaultHealthHTTPClient(timeout time.Duration) *http.Client {
	if timeout <= 0 {
		timeout = defaultHealthTimeout
	}
	return &http.Client{Timeout: timeout}
}
