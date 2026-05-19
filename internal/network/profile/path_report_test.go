package netprofile

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func TestBuildPathProbeReportValid(t *testing.T) {
	profile := validNetworkProfile()
	profile.ProbeTarget = "https://Hub.Example:8443/probe"

	report, err := BuildPathProbeReport(" node-1 ", profile)
	if err != nil {
		t.Fatalf("BuildPathProbeReport() error = %v", err)
	}
	if report.NodeID != "node-1" {
		t.Fatalf("NodeID = %q, want node-1", report.NodeID)
	}
	if report.Target.Scheme != "https" {
		t.Fatalf("target scheme = %q, want https", report.Target.Scheme)
	}
	if report.Target.Hostname != "hub.example" {
		t.Fatalf("target hostname = %q, want hub.example", report.Target.Hostname)
	}
	if report.Target.Port != "8443" {
		t.Fatalf("target port = %q, want 8443", report.Target.Port)
	}
	if report.RTTMsP50 != profile.RTTMsP50 ||
		report.RTTMsP95 != profile.RTTMsP95 ||
		report.JitterMsP95 != profile.JitterMsP95 ||
		report.LossRateP95 != profile.LossRateP95 ||
		report.UploadMbpsP50 != profile.UploadMbpsP50 ||
		report.UploadMbpsP95 != profile.UploadMbpsP95 ||
		report.DownloadMbpsP50 != profile.DownloadMbpsP50 ||
		report.DownloadMbpsP95 != profile.DownloadMbpsP95 ||
		report.MeasuredAtUnixMs != profile.MeasuredAtUnixMs {
		t.Fatalf("report metrics = %+v, want profile metrics copied from %+v", report, profile)
	}
	if err := ValidatePathProbeReport(report); err != nil {
		t.Fatalf("ValidatePathProbeReport() error = %v", err)
	}
}

func TestBuildPathProbeReportRejectsMissingNode(t *testing.T) {
	_, err := BuildPathProbeReport(" \t ", validNetworkProfile())
	if err == nil || !strings.Contains(err.Error(), "node id required") {
		t.Fatalf("BuildPathProbeReport() error = %v, want missing node rejection", err)
	}
}

func TestValidatePathProbeReportRejectsNegativeAndNaNMetrics(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(*PathProbeReport)
		want string
	}{
		{
			name: "negative",
			edit: func(report *PathProbeReport) {
				report.DownloadMbpsP95 = -1
			},
			want: "must not be negative",
		},
		{
			name: "nan",
			edit: func(report *PathProbeReport) {
				report.RTTMsP95 = math.NaN()
			},
			want: "must be finite",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			report := validPathProbeReport(t)
			tc.edit(&report)
			err := ValidatePathProbeReport(report)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("ValidatePathProbeReport() error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestBuildPathProbeReportUsesProfileTarget(t *testing.T) {
	profile := validNetworkProfile()
	profile.ProbeTarget = "http://path-target.example:8080/api/v1/network/probe"

	report, err := BuildPathProbeReport("node-1", profile)
	if err != nil {
		t.Fatalf("BuildPathProbeReport() error = %v", err)
	}
	if report.Target.Scheme != "http" {
		t.Fatalf("target scheme = %q, want http", report.Target.Scheme)
	}
	if report.Target.Hostname != "path-target.example" {
		t.Fatalf("target hostname = %q, want path-target.example", report.Target.Hostname)
	}
	if report.Target.Port != "8080" {
		t.Fatalf("target port = %q, want 8080", report.Target.Port)
	}
}

func TestBuildPathProbeReportContainsNoSecretFields(t *testing.T) {
	t.Setenv("RYV_DEMO_KEY", "demo-key-secret")
	t.Setenv("RYV_ADMIN_KEY", "admin-key-secret")
	t.Setenv("RYV_BIND_TOKEN", "bind-token-secret")

	profile := validNetworkProfile()
	profile.ProbeTarget = "https://hub.example/private/probe?access_token=target-secret&password=another-secret"

	report, err := BuildPathProbeReport("node-1", profile)
	if err != nil {
		t.Fatalf("BuildPathProbeReport() error = %v", err)
	}
	raw, err := json.Marshal(report)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	body := string(raw)
	for _, forbidden := range []string{
		"demo-key-secret",
		"admin-key-secret",
		"bind-token-secret",
		"target-secret",
		"another-secret",
		"access_token",
		"password=",
		"/private/probe",
	} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("report JSON leaked %q: %s", forbidden, body)
		}
	}
}

func validPathProbeReport(t *testing.T) PathProbeReport {
	t.Helper()
	report, err := BuildPathProbeReport("node-1", validNetworkProfile())
	if err != nil {
		t.Fatalf("BuildPathProbeReport() error = %v", err)
	}
	return report
}
