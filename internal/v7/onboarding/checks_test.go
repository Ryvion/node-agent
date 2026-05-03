package onboarding

import (
	"strings"
	"testing"
	"time"

	"github.com/Ryvion/node-agent/internal/hw"
)

func TestRunBasicOnboardingChecksIncludesWarningsAndHardErrors(t *testing.T) {
	report := RunBasicOnboardingChecksWithOptions(testOptions(CheckOptions{
		HubURL:                 "not a url",
		NativeRuntimeAvailable: func() bool { return false },
		ResolveOCIEnginePath: func(string, func(string) string) string {
			return ""
		},
	}))

	if report.WarningCount() == 0 {
		t.Fatalf("expected warnings in report: %+v", report.Checks)
	}
	if report.HardErrorCount() == 0 {
		t.Fatalf("expected hard errors in report: %+v", report.Checks)
	}
	if !hasCheckStatus(report, "native_inference", CheckStatusWarning) {
		t.Fatalf("native inference warning missing: %+v", report.Checks)
	}
	if !hasCheckStatus(report, "hub_url", CheckStatusHardError) {
		t.Fatalf("hub URL hard error missing: %+v", report.Checks)
	}
}

func TestDockerMissingIsWarningNotHardError(t *testing.T) {
	report := RunBasicOnboardingChecksWithOptions(testOptions(CheckOptions{
		ResolveOCIEnginePath: func(string, func(string) string) string {
			return ""
		},
	}))

	if !hasCheckStatus(report, "oci_runtime", CheckStatusWarning) {
		t.Fatalf("OCI/Docker missing should be warning: %+v", report.Checks)
	}
	check := findCheck(report, "oci_runtime")
	if check.Status == CheckStatusHardError {
		t.Fatalf("Docker missing must not be a hard error: %+v", check)
	}
}

func TestInvalidHubURLIsHardError(t *testing.T) {
	report := RunBasicOnboardingChecksWithOptions(testOptions(CheckOptions{
		HubURL: "://bad",
	}))

	if !hasCheckStatus(report, "hub_url", CheckStatusHardError) {
		t.Fatalf("invalid hub URL should be hard error: %+v", report.Checks)
	}
}

func TestRunBasicOnboardingChecksDoesNotLeakSecrets(t *testing.T) {
	secret := "super-secret-token"
	report := RunBasicOnboardingChecksWithOptions(testOptions(CheckOptions{
		HubURL: "https://user:" + secret + "@hub.example/path?token=" + secret,
		Getenv: func(key string) string {
			switch key {
			case "RYV_BIND_TOKEN", "RYV_ADMIN_KEY":
				return secret
			default:
				return ""
			}
		},
	}))

	formatted := FormatOnboardingReport(report)
	if strings.Contains(formatted, secret) {
		t.Fatalf("report leaked secret %q:\n%s", secret, formatted)
	}
	if strings.Contains(formatted, "user:") {
		t.Fatalf("report leaked URL user info:\n%s", formatted)
	}
}

func TestCPUFallbackIsOKWithoutGPU(t *testing.T) {
	report := RunBasicOnboardingChecksWithOptions(testOptions(CheckOptions{
		DetectCaps: func(string) hw.CapSet {
			return hw.CapSet{
				CPUCores: 4,
				RAMBytes: 8 * 1024 * 1024 * 1024,
			}
		},
	}))

	if !hasCheckStatus(report, "compute_fallback", CheckStatusOK) {
		t.Fatalf("CPU fallback should be OK: %+v", report.Checks)
	}
	if strings.Contains(strings.ToLower(findCheck(report, "compute_fallback").Message), "required") {
		t.Fatalf("CPU fallback message should not imply GPU is required: %+v", findCheck(report, "compute_fallback"))
	}
}

func testOptions(overrides CheckOptions) CheckOptions {
	options := CheckOptions{
		AgentVersion: "test",
		HubURL:       "https://api.ryvion.example",
		DataDir:      "",
		OS:           "linux",
		Arch:         "amd64",
		Getenv: func(string) string {
			return ""
		},
		DetectCaps: func(string) hw.CapSet {
			return hw.CapSet{
				GPUModel:  "NVIDIA Test GPU",
				CPUCores:  8,
				RAMBytes:  16 * 1024 * 1024 * 1024,
				VRAMBytes: 8 * 1024 * 1024 * 1024,
			}
		},
		NativeRuntimeAvailable: func() bool {
			return true
		},
		ResolveOCIEnginePath: func(string, func(string) string) string {
			return "/usr/bin/podman"
		},
		Now: func() time.Time {
			return time.Unix(123, 0)
		},
	}

	if overrides.AgentVersion != "" {
		options.AgentVersion = overrides.AgentVersion
	}
	if overrides.HubURL != "" {
		options.HubURL = overrides.HubURL
	}
	if overrides.DataDir != "" {
		options.DataDir = overrides.DataDir
	}
	if overrides.DeviceType != "" {
		options.DeviceType = overrides.DeviceType
	}
	if overrides.OS != "" {
		options.OS = overrides.OS
	}
	if overrides.Arch != "" {
		options.Arch = overrides.Arch
	}
	if overrides.Getenv != nil {
		options.Getenv = overrides.Getenv
	}
	if overrides.DetectCaps != nil {
		options.DetectCaps = overrides.DetectCaps
	}
	if overrides.NativeRuntimeAvailable != nil {
		options.NativeRuntimeAvailable = overrides.NativeRuntimeAvailable
	}
	if overrides.ResolveOCIEnginePath != nil {
		options.ResolveOCIEnginePath = overrides.ResolveOCIEnginePath
	}
	if overrides.Now != nil {
		options.Now = overrides.Now
	}
	return options
}

func hasCheckStatus(report OnboardingReport, id string, status CheckStatus) bool {
	return findCheck(report, id).Status == status
}

func findCheck(report OnboardingReport, id string) OnboardingCheck {
	for _, check := range report.Checks {
		if check.ID == id {
			return check
		}
	}
	return OnboardingCheck{}
}
