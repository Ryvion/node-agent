package netprofile

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

func ProbeHTTP(ctx context.Context, config ProbeConfig) (NetworkProfile, error) {
	config, target, err := normalizeProbeConfig(config)
	profile := NetworkProfile{
		ProbeTarget:      target,
		MeasuredAtUnixMs: time.Now().UnixMilli(),
	}
	if err != nil {
		return profile, err
	}
	if config.EnableUploadProbe {
		return profile, fmt.Errorf("upload probe is not implemented; leave EnableUploadProbe false to advertise upload as unknown")
	}

	client := &http.Client{
		Timeout: time.Duration(config.TimeoutMs) * time.Millisecond,
	}
	results := make([]ProbeResult, 0, config.Samples)
	var sampleErrs []error
	for i := 0; i < config.Samples; i++ {
		if err := ctx.Err(); err != nil {
			profile.SampleCount = len(results)
			profile.MeasuredAtUnixMs = time.Now().UnixMilli()
			fillProfileFromResults(&profile, results)
			return profile, errors.Join(append(sampleErrs, err)...)
		}

		result, err := probeHTTPSample(ctx, client, target, config)
		results = append(results, result)
		if err != nil {
			sampleErrs = append(sampleErrs, fmt.Errorf("sample %d failed: %w", i+1, err))
			if ctx.Err() != nil {
				break
			}
		}
	}

	profile.SampleCount = len(results)
	profile.MeasuredAtUnixMs = time.Now().UnixMilli()
	fillProfileFromResults(&profile, results)
	if err := ValidateNetworkProfile(profile); err != nil {
		sampleErrs = append(sampleErrs, err)
	}
	return profile, errors.Join(sampleErrs...)
}

func normalizeProbeConfig(config ProbeConfig) (ProbeConfig, string, error) {
	target := strings.TrimSpace(config.TargetURL)
	if target == "" {
		return config, "", fmt.Errorf("target URL required")
	}
	parsed, err := url.Parse(target)
	if err != nil {
		return config, "", fmt.Errorf("parse target URL: %w", err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return config, "", fmt.Errorf("target URL must use http or https")
	}
	if parsed.Host == "" {
		return config, "", fmt.Errorf("target URL host required")
	}
	if parsed.User != nil {
		return config, "", fmt.Errorf("target URL must not contain user info")
	}
	parsed.Fragment = ""

	if config.TimeoutMs <= 0 {
		config.TimeoutMs = DefaultTimeoutMs
	}
	if config.Samples <= 0 {
		config.Samples = DefaultSamples
	}
	if config.MaxDownloadBytes <= 0 {
		config.MaxDownloadBytes = DefaultMaxDownloadBytes
	}
	if strings.TrimSpace(config.UserAgent) == "" {
		config.UserAgent = DefaultUserAgent
	}
	return config, parsed.String(), nil
}

func probeHTTPSample(ctx context.Context, client *http.Client, target string, config ProbeConfig) (ProbeResult, error) {
	result := ProbeResult{}
	rttMs, err := probeRTT(ctx, client, target, config.UserAgent)
	if err != nil {
		result.Error = err.Error()
		return result, err
	}
	result.RTTMs = rttMs

	if config.EnableDownloadProbe {
		downloadMbps, downloadBytes, err := probeDownload(ctx, client, target, config)
		result.DownloadMbps = downloadMbps
		result.DownloadBytes = downloadBytes
		if err != nil {
			result.Error = err.Error()
			return result, err
		}
	}

	result.Success = true
	return result, nil
}

func probeRTT(ctx context.Context, client *http.Client, target string, userAgent string) (float64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodHead, target, nil)
	if err != nil {
		return 0, err
	}
	req.Header.Set("User-Agent", userAgent)

	start := time.Now()
	resp, err := client.Do(req)
	elapsed := time.Since(start)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, io.LimitReader(resp.Body, 1024))

	if resp.StatusCode == http.StatusMethodNotAllowed || resp.StatusCode == http.StatusNotImplemented {
		return probeSmallGETRTT(ctx, client, target, userAgent)
	}
	if resp.StatusCode >= 500 {
		return 0, fmt.Errorf("RTT probe returned HTTP %d", resp.StatusCode)
	}
	return durationMillis(elapsed), nil
}

func probeSmallGETRTT(ctx context.Context, client *http.Client, target string, userAgent string) (float64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		return 0, err
	}
	req.Header.Set("User-Agent", userAgent)
	req.Header.Set("Range", "bytes=0-0")

	start := time.Now()
	resp, err := client.Do(req)
	elapsed := time.Since(start)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, io.LimitReader(resp.Body, 1))

	if resp.StatusCode >= 500 {
		return 0, fmt.Errorf("RTT GET fallback returned HTTP %d", resp.StatusCode)
	}
	return durationMillis(elapsed), nil
}

func probeDownload(ctx context.Context, client *http.Client, target string, config ProbeConfig) (float64, int64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if err != nil {
		return 0, 0, err
	}
	req.Header.Set("User-Agent", config.UserAgent)
	req.Header.Set("Range", fmt.Sprintf("bytes=0-%d", config.MaxDownloadBytes-1))

	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return 0, 0, err
	}
	defer resp.Body.Close()

	downloadBytes, readErr := io.Copy(io.Discard, io.LimitReader(resp.Body, config.MaxDownloadBytes))
	elapsed := time.Since(start)
	if readErr != nil {
		return 0, downloadBytes, readErr
	}
	if resp.StatusCode >= 500 {
		return 0, downloadBytes, fmt.Errorf("download probe returned HTTP %d", resp.StatusCode)
	}
	if downloadBytes <= 0 || elapsed <= 0 {
		return 0, downloadBytes, nil
	}
	return (float64(downloadBytes) * 8) / elapsed.Seconds() / 1_000_000, downloadBytes, nil
}

func fillProfileFromResults(profile *NetworkProfile, results []ProbeResult) {
	if len(results) == 0 {
		return
	}
	rtts := make([]float64, 0, len(results))
	downloads := make([]float64, 0, len(results))
	uploads := make([]float64, 0, len(results))
	failed := 0
	for _, result := range results {
		if !result.Success {
			failed++
		}
		if result.RTTMs > 0 {
			rtts = append(rtts, result.RTTMs)
		}
		if result.DownloadMbps > 0 {
			downloads = append(downloads, result.DownloadMbps)
		}
		if result.UploadMbps > 0 {
			uploads = append(uploads, result.UploadMbps)
		}
	}
	profile.RTTMsP50, profile.RTTMsP95 = EstimateStats(rtts)
	profile.DownloadMbpsP50, profile.DownloadMbpsP95 = EstimateStats(downloads)
	profile.UploadMbpsP50, profile.UploadMbpsP95 = EstimateStats(uploads)
	_, profile.JitterMsP95 = EstimateStats(jitterSamples(rtts))
	profile.LossRateP95 = float64(failed) / float64(len(results))
}

func jitterSamples(rtts []float64) []float64 {
	if len(rtts) < 2 {
		return nil
	}
	jitters := make([]float64, 0, len(rtts)-1)
	for i := 1; i < len(rtts); i++ {
		delta := rtts[i] - rtts[i-1]
		if delta < 0 {
			delta = -delta
		}
		jitters = append(jitters, delta)
	}
	return jitters
}

func durationMillis(duration time.Duration) float64 {
	return float64(duration) / float64(time.Millisecond)
}
