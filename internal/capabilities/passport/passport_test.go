package capability

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestValidCPUOnlyPassportPasses(t *testing.T) {
	passport := validCPUPassport()
	if err := ValidateCapabilityPassport(passport); err != nil {
		t.Fatalf("ValidateCapabilityPassport() error = %v", err)
	}
}

func TestValidGPUPassportPasses(t *testing.T) {
	passport := validCPUPassport()
	passport.HardwareProfile.GPUModel = "NVIDIA GeForce RTX 4090"
	passport.HardwareProfile.GPUVendor = "nvidia"
	passport.HardwareProfile.VRAMBytes = 24 * 1024 * 1024 * 1024
	if err := ValidateCapabilityPassport(passport); err != nil {
		t.Fatalf("ValidateCapabilityPassport() error = %v", err)
	}
}

func TestMissingSchemaRejected(t *testing.T) {
	passport := validCPUPassport()
	passport.SchemaVersion = ""
	if err := ValidateCapabilityPassport(passport); err == nil || !strings.Contains(err.Error(), "schema version") {
		t.Fatalf("ValidateCapabilityPassport() error = %v, want schema version error", err)
	}
}

func TestMissingOSOrArchRejected(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(*CapabilityPassport)
		want string
	}{
		{name: "os", edit: func(passport *CapabilityPassport) { passport.OS = "" }, want: "os required"},
		{name: "arch", edit: func(passport *CapabilityPassport) { passport.Arch = "" }, want: "arch required"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			passport := validCPUPassport()
			tc.edit(&passport)
			if err := ValidateCapabilityPassport(passport); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("ValidateCapabilityPassport() error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestZeroCPURejected(t *testing.T) {
	passport := validCPUPassport()
	passport.HardwareProfile.CPUCores = 0
	if err := ValidateCapabilityPassport(passport); err == nil || !strings.Contains(err.Error(), "cpu cores") {
		t.Fatalf("ValidateCapabilityPassport() error = %v, want cpu cores error", err)
	}
}

func TestManagedOCIWithoutRuntimeRejected(t *testing.T) {
	passport := validCPUPassport()
	passport.RuntimeProfile.OCIAvailable = false
	passport.RenderCapabilitySummary.SupportsManagedOCI = true
	if err := ValidateCapabilityPassport(passport); err == nil || !strings.Contains(err.Error(), "managed OCI support") {
		t.Fatalf("ValidateCapabilityPassport() error = %v, want managed OCI runtime error", err)
	}
}

func TestPassportContainsNoObviousSecretFields(t *testing.T) {
	passport := validCPUPassport()
	data, err := json.Marshal(passport)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	lower := strings.ToLower(string(data))
	for _, needle := range []string{"secret", "token", "private", "password", "authorization"} {
		if strings.Contains(lower, needle) {
			t.Fatalf("passport JSON contains %q: %s", needle, string(data))
		}
	}
}

func validCPUPassport() CapabilityPassport {
	return CapabilityPassport{
		SchemaVersion: SchemaVersionV1,
		AgentVersion:  "dev",
		OS:            "linux",
		Arch:          "amd64",
		DeviceType:    "operator",
		HardwareProfile: HardwareProfile{
			CPUCores: 8,
			RAMBytes: 32 * 1024 * 1024 * 1024,
		},
		RuntimeProfile: RuntimeProfile{
			OCIAvailable:         true,
			SupportedRunnerKinds: []string{"managed_oci"},
		},
		RenderCapabilitySummary: RenderCapabilitySummary{
			SupportsManagedOCI:     true,
			SupportsArtifactUpload: true,
		},
		SandboxCapabilitySummary: SandboxCapabilitySummary{
			RejectsUntrustedCustomRunners: true,
		},
		CreatedAtUnixMs: 1,
	}
}
