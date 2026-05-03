package netprofile

import (
	"math"
	"strings"
	"testing"
)

func TestEstimateStatsCalculatesP50AndP95(t *testing.T) {
	p50, p95 := EstimateStats([]float64{40, 10, 30, 20})
	if !nearlyEqual(p50, 25) {
		t.Fatalf("p50 = %v, want 25", p50)
	}
	if !nearlyEqual(p95, 38.5) {
		t.Fatalf("p95 = %v, want 38.5", p95)
	}
}

func TestEstimateStatsIgnoresInvalidSamples(t *testing.T) {
	p50, p95 := EstimateStats([]float64{math.NaN(), -1, math.Inf(1), 10, 30})
	if !nearlyEqual(p50, 20) {
		t.Fatalf("p50 = %v, want 20", p50)
	}
	if !nearlyEqual(p95, 29) {
		t.Fatalf("p95 = %v, want 29", p95)
	}
}

func TestValidateNetworkProfileRejectsInvalidMetrics(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(*NetworkProfile)
		want string
	}{
		{
			name: "negative",
			edit: func(profile *NetworkProfile) {
				profile.UploadMbpsP50 = -1
			},
			want: "must not be negative",
		},
		{
			name: "nan",
			edit: func(profile *NetworkProfile) {
				profile.RTTMsP50 = math.NaN()
			},
			want: "must be finite",
		},
		{
			name: "inf",
			edit: func(profile *NetworkProfile) {
				profile.DownloadMbpsP95 = math.Inf(1)
			},
			want: "must be finite",
		},
		{
			name: "loss greater than one",
			edit: func(profile *NetworkProfile) {
				profile.LossRateP95 = 1.1
			},
			want: "must not exceed 1",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			profile := validNetworkProfile()
			tc.edit(&profile)
			err := ValidateNetworkProfile(profile)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("ValidateNetworkProfile() error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestValidateNetworkProfileAllowsExplicitUnknownZeros(t *testing.T) {
	profile := NetworkProfile{
		ProbeTarget:      "https://hub.example/probe",
		SampleCount:      0,
		MeasuredAtUnixMs: 123,
	}
	if err := ValidateNetworkProfile(profile); err != nil {
		t.Fatalf("ValidateNetworkProfile() error = %v", err)
	}
}

func validNetworkProfile() NetworkProfile {
	return NetworkProfile{
		UploadMbpsP50:    1,
		UploadMbpsP95:    2,
		DownloadMbpsP50:  3,
		DownloadMbpsP95:  4,
		RTTMsP50:         5,
		RTTMsP95:         6,
		JitterMsP95:      1,
		LossRateP95:      0.1,
		ProbeTarget:      "https://hub.example/probe",
		SampleCount:      3,
		MeasuredAtUnixMs: 123,
	}
}

func nearlyEqual(got, want float64) bool {
	const epsilon = 1e-9
	diff := got - want
	if diff < 0 {
		diff = -diff
	}
	return diff <= epsilon
}
