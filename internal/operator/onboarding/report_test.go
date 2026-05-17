package onboarding

import (
	"strings"
	"testing"
	"time"
)

func TestFormatOnboardingReportStable(t *testing.T) {
	report := OnboardingReport{
		SchemaVersion: reportSchemaVersion,
		OS:            "linux",
		Arch:          "amd64",
		GeneratedAt:   time.Unix(123, 0),
		Checks: []OnboardingCheck{
			{ID: "platform", Title: "Platform", Status: CheckStatusOK, Message: "linux/amd64 is supported"},
			{ID: "oci_runtime", Title: "Optional OCI runtime", Status: CheckStatusWarning, Message: "Docker/Podman runtime was not detected; native onboarding can continue"},
			{ID: "hub_url", Title: "Hub URL", Status: CheckStatusHardError, Message: "invalid hub URL; set RYV_HUB_URL to an http(s) URL"},
		},
	}

	got := FormatOnboardingReport(report)
	want := strings.Join([]string{
		"Ryvion node onboarding report",
		"Platform: linux/amd64",
		"Result: hard_error (1 hard errors, 1 warnings)",
		"- [ok] Platform: linux/amd64 is supported",
		"- [warning] Optional OCI runtime: Docker/Podman runtime was not detected; native onboarding can continue",
		"- [hard_error] Hub URL: invalid hub URL; set RYV_HUB_URL to an http(s) URL",
	}, "\n")
	if got != want {
		t.Fatalf("formatted report mismatch\n--- got ---\n%s\n--- want ---\n%s", got, want)
	}
}

func TestFormatOnboardingReportRedactsSecretQueryValues(t *testing.T) {
	report := OnboardingReport{
		OS:   "linux",
		Arch: "amd64",
		Checks: []OnboardingCheck{
			{ID: "example", Title: "Example", Status: CheckStatusWarning, Message: "failed with token=super-secret password=another-secret"},
		},
	}

	got := FormatOnboardingReport(report)
	for _, forbidden := range []string{"super-secret", "another-secret"} {
		if strings.Contains(got, forbidden) {
			t.Fatalf("formatted report leaked %q:\n%s", forbidden, got)
		}
	}
	if !strings.Contains(got, "token=<redacted>") || !strings.Contains(got, "password=<redacted>") {
		t.Fatalf("formatted report did not include redaction markers:\n%s", got)
	}
}
