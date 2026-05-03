package netprofile

import (
	"errors"
	"fmt"
	"math"
	"net/url"
	"strconv"
	"strings"
)

type PathProbeReport struct {
	NodeID           string          `json:"node_id"`
	Target           PathProbeTarget `json:"target"`
	RTTMsP50         float64         `json:"rtt_ms_p50"`
	RTTMsP95         float64         `json:"rtt_ms_p95"`
	JitterMsP95      float64         `json:"jitter_ms_p95"`
	LossRateP95      float64         `json:"loss_rate_p95"`
	UploadMbpsP50    float64         `json:"upload_mbps_p50"`
	UploadMbpsP95    float64         `json:"upload_mbps_p95"`
	DownloadMbpsP50  float64         `json:"download_mbps_p50"`
	DownloadMbpsP95  float64         `json:"download_mbps_p95"`
	MeasuredAtUnixMs int64           `json:"measured_at_unix_ms"`
}

type PathProbeTarget struct {
	Scheme   string `json:"scheme,omitempty"`
	Hostname string `json:"hostname"`
	Port     string `json:"port,omitempty"`
}

func BuildPathProbeReport(nodeID string, profile NetworkProfile) (PathProbeReport, error) {
	if err := ValidateNetworkProfile(profile); err != nil {
		return PathProbeReport{}, err
	}

	target, err := pathProbeTargetFromProfileTarget(profile.ProbeTarget)
	if err != nil {
		return PathProbeReport{}, err
	}

	report := PathProbeReport{
		NodeID:           strings.TrimSpace(nodeID),
		Target:           target,
		RTTMsP50:         profile.RTTMsP50,
		RTTMsP95:         profile.RTTMsP95,
		JitterMsP95:      profile.JitterMsP95,
		LossRateP95:      profile.LossRateP95,
		UploadMbpsP50:    profile.UploadMbpsP50,
		UploadMbpsP95:    profile.UploadMbpsP95,
		DownloadMbpsP50:  profile.DownloadMbpsP50,
		DownloadMbpsP95:  profile.DownloadMbpsP95,
		MeasuredAtUnixMs: profile.MeasuredAtUnixMs,
	}
	if err := ValidatePathProbeReport(report); err != nil {
		return PathProbeReport{}, err
	}
	return report, nil
}

func ValidatePathProbeReport(report PathProbeReport) error {
	var errs []error
	if strings.TrimSpace(report.NodeID) == "" {
		errs = append(errs, fmt.Errorf("node id required"))
	}
	if report.NodeID != strings.TrimSpace(report.NodeID) {
		errs = append(errs, fmt.Errorf("node id must not have leading or trailing whitespace"))
	}
	if containsPathReportControl(report.NodeID) {
		errs = append(errs, fmt.Errorf("node id must not contain control characters"))
	}
	if err := validatePathProbeTarget(report.Target); err != nil {
		errs = append(errs, err)
	}
	if err := validatePathProbeReportMetrics(report); err != nil {
		errs = append(errs, err)
	}
	if report.MeasuredAtUnixMs < 0 {
		errs = append(errs, fmt.Errorf("measured time must not be negative"))
	}
	if err := validateNoSecretLikePathProbeReportValues(report); err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

func pathProbeTargetFromProfileTarget(rawTarget string) (PathProbeTarget, error) {
	rawTarget = strings.TrimSpace(rawTarget)
	if rawTarget == "" {
		return PathProbeTarget{}, fmt.Errorf("probe target required")
	}
	if containsPathReportControl(rawTarget) {
		return PathProbeTarget{}, fmt.Errorf("probe target must not contain control characters")
	}

	parsed, err := url.Parse(rawTarget)
	if err != nil {
		return PathProbeTarget{}, fmt.Errorf("parse probe target: %w", err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return PathProbeTarget{}, fmt.Errorf("probe target must use http or https")
	}
	if parsed.User != nil {
		return PathProbeTarget{}, fmt.Errorf("probe target must not contain user info")
	}

	target := PathProbeTarget{
		Scheme:   strings.ToLower(parsed.Scheme),
		Hostname: strings.ToLower(strings.TrimSpace(parsed.Hostname())),
		Port:     parsed.Port(),
	}
	if err := validatePathProbeTarget(target); err != nil {
		return PathProbeTarget{}, err
	}
	return target, nil
}

func validatePathProbeTarget(target PathProbeTarget) error {
	var errs []error
	if target.Scheme != "" {
		scheme := strings.ToLower(strings.TrimSpace(target.Scheme))
		if scheme != target.Scheme {
			errs = append(errs, fmt.Errorf("target scheme must be lowercase and trimmed"))
		}
		if scheme != "http" && scheme != "https" {
			errs = append(errs, fmt.Errorf("target scheme must be http or https"))
		}
	}
	if strings.TrimSpace(target.Hostname) == "" {
		errs = append(errs, fmt.Errorf("target hostname required"))
	}
	if target.Hostname != strings.TrimSpace(target.Hostname) {
		errs = append(errs, fmt.Errorf("target hostname must not have leading or trailing whitespace"))
	}
	if target.Hostname != strings.ToLower(target.Hostname) {
		errs = append(errs, fmt.Errorf("target hostname must be lowercase"))
	}
	if containsPathReportControl(target.Scheme) || containsPathReportControl(target.Hostname) || containsPathReportControl(target.Port) {
		errs = append(errs, fmt.Errorf("target must not contain control characters"))
	}
	if target.Port != "" {
		port, err := strconv.Atoi(target.Port)
		if err != nil || port < 1 || port > 65535 {
			errs = append(errs, fmt.Errorf("target port must be a valid TCP port"))
		}
	}
	return errors.Join(errs...)
}

func validatePathProbeReportMetrics(report PathProbeReport) error {
	metrics := []struct {
		name  string
		value float64
	}{
		{name: "RTT ms p50", value: report.RTTMsP50},
		{name: "RTT ms p95", value: report.RTTMsP95},
		{name: "jitter ms p95", value: report.JitterMsP95},
		{name: "loss rate p95", value: report.LossRateP95},
		{name: "upload Mbps p50", value: report.UploadMbpsP50},
		{name: "upload Mbps p95", value: report.UploadMbpsP95},
		{name: "download Mbps p50", value: report.DownloadMbpsP50},
		{name: "download Mbps p95", value: report.DownloadMbpsP95},
	}

	var errs []error
	for _, metric := range metrics {
		if math.IsNaN(metric.value) || math.IsInf(metric.value, 0) {
			errs = append(errs, fmt.Errorf("%s must be finite", metric.name))
			continue
		}
		if metric.value < 0 {
			errs = append(errs, fmt.Errorf("%s must not be negative", metric.name))
		}
	}
	if report.LossRateP95 > 1 {
		errs = append(errs, fmt.Errorf("loss rate p95 must not exceed 1"))
	}
	return errors.Join(errs...)
}

func validateNoSecretLikePathProbeReportValues(report PathProbeReport) error {
	fields := []struct {
		name  string
		value string
	}{
		{name: "node_id", value: report.NodeID},
		{name: "target.scheme", value: report.Target.Scheme},
		{name: "target.hostname", value: report.Target.Hostname},
		{name: "target.port", value: report.Target.Port},
	}

	var found []string
	for _, field := range fields {
		if looksLikePathReportSecret(field.value) {
			found = append(found, field.name)
		}
	}
	if len(found) > 0 {
		return fmt.Errorf("path probe report contains secret-like values at %s", strings.Join(found, ", "))
	}
	return nil
}

func containsPathReportControl(value string) bool {
	return strings.ContainsAny(value, "\r\n\t")
}

func looksLikePathReportSecret(value string) bool {
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "" {
		return false
	}
	needles := []string{
		"-----begin private key",
		"authorization:",
		"bearer ",
		"api_key",
		"apikey",
		"access_token",
		"refresh_token",
		"private_key",
		"client_secret",
		"password=",
		"passwd=",
		"secret=",
		"token=",
	}
	for _, needle := range needles {
		if strings.Contains(value, needle) {
			return true
		}
	}
	return strings.HasPrefix(value, "sk-")
}
