package netprofile

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

const (
	DefaultTimeoutMs        = 5000
	DefaultSamples          = 3
	DefaultMaxDownloadBytes = 64 * 1024
	DefaultUserAgent        = "ryvion-node-netprofile/1"
)

type NetworkProfile struct {
	UploadMbpsP50    float64 `json:"upload_mbps_p50"`
	UploadMbpsP95    float64 `json:"upload_mbps_p95"`
	DownloadMbpsP50  float64 `json:"download_mbps_p50"`
	DownloadMbpsP95  float64 `json:"download_mbps_p95"`
	RTTMsP50         float64 `json:"rtt_ms_p50"`
	RTTMsP95         float64 `json:"rtt_ms_p95"`
	JitterMsP95      float64 `json:"jitter_ms_p95"`
	LossRateP95      float64 `json:"loss_rate_p95"`
	ProbeTarget      string  `json:"probe_target,omitempty"`
	SampleCount      int     `json:"sample_count"`
	MeasuredAtUnixMs int64   `json:"measured_at_unix_ms"`
}

type ProbeConfig struct {
	TargetURL           string
	TimeoutMs           int
	Samples             int
	MaxDownloadBytes    int64
	EnableDownloadProbe bool
	EnableUploadProbe   bool
	UserAgent           string
}

type ProbeResult struct {
	RTTMs         float64 `json:"rtt_ms"`
	DownloadMbps  float64 `json:"download_mbps"`
	UploadMbps    float64 `json:"upload_mbps"`
	DownloadBytes int64   `json:"download_bytes"`
	Success       bool    `json:"success"`
	Error         string  `json:"error,omitempty"`
}

func ValidateNetworkProfile(profile NetworkProfile) error {
	metrics := map[string]float64{
		"upload Mbps p50":   profile.UploadMbpsP50,
		"upload Mbps p95":   profile.UploadMbpsP95,
		"download Mbps p50": profile.DownloadMbpsP50,
		"download Mbps p95": profile.DownloadMbpsP95,
		"RTT ms p50":        profile.RTTMsP50,
		"RTT ms p95":        profile.RTTMsP95,
		"jitter ms p95":     profile.JitterMsP95,
		"loss rate p95":     profile.LossRateP95,
	}
	var errs []error
	for name, value := range metrics {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			errs = append(errs, fmt.Errorf("%s must be finite", name))
			continue
		}
		if value < 0 {
			errs = append(errs, fmt.Errorf("%s must not be negative", name))
		}
	}
	if profile.LossRateP95 > 1 {
		errs = append(errs, fmt.Errorf("loss rate p95 must not exceed 1"))
	}
	if profile.SampleCount < 0 {
		errs = append(errs, fmt.Errorf("sample count must not be negative"))
	}
	if profile.MeasuredAtUnixMs < 0 {
		errs = append(errs, fmt.Errorf("measured time must not be negative"))
	}
	if strings.ContainsAny(profile.ProbeTarget, "\r\n\t") {
		errs = append(errs, fmt.Errorf("probe target must not contain control characters"))
	}
	return errors.Join(errs...)
}
