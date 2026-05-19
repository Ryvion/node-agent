package capability

import (
	"testing"

	"github.com/Ryvion/ryvion-node/internal/hw"
)

func TestBuildCapabilityPassportFromFacts(t *testing.T) {
	passport, err := BuildCapabilityPassport(BuildPassportInput{
		AgentVersion:    "v1.2.3",
		OS:              "windows",
		Arch:            "amd64",
		DeviceType:      "operator",
		DeclaredCountry: "ca",
		HardwareCapabilities: hw.CapSet{
			CPUCores:     16,
			RAMBytes:     64 * 1024 * 1024 * 1024,
			GPUModel:     "NVIDIA GeForce RTX 4090",
			VRAMBytes:    24 * 1024 * 1024 * 1024,
			Sensors:      "nvidia-driver:555.85 model:NVIDIA GeForce RTX 4090",
			TEESupported: true,
			TEEType:      "tdx",
		},
		RuntimeProfile: RuntimeProfile{
			OCIAvailable:         true,
			SupportedRunnerKinds: []string{"managed_oci"},
		},
		WorkCapabilitySummary: WorkCapabilitySummary{
			SupportsManagedOCI:     true,
			SupportsArtifactUpload: true,
		},
		CreatedAtUnixMs: 123,
	})
	if err != nil {
		t.Fatalf("BuildCapabilityPassport() error = %v", err)
	}
	if passport.SchemaVersion != SchemaVersionV1 {
		t.Fatalf("schema version = %q, want %q", passport.SchemaVersion, SchemaVersionV1)
	}
	if passport.DeclaredCountry != "CA" {
		t.Fatalf("declared country = %q, want CA", passport.DeclaredCountry)
	}
	if passport.HardwareProfile.CPUCores != 16 || passport.HardwareProfile.RAMBytes == 0 {
		t.Fatalf("hardware profile not mapped from caps: %+v", passport.HardwareProfile)
	}
	if passport.HardwareProfile.GPUVendor != "nvidia" {
		t.Fatalf("gpu vendor = %q, want nvidia", passport.HardwareProfile.GPUVendor)
	}
	if passport.HardwareProfile.DriverVersion != "555.85" {
		t.Fatalf("driver version = %q, want 555.85", passport.HardwareProfile.DriverVersion)
	}
	if !passport.SandboxCapabilitySummary.RejectsUntrustedCustomRunners {
		t.Fatal("builder should default to rejecting untrusted custom runners")
	}
	if !passport.SandboxCapabilitySummary.NetworkIsolationSupported {
		t.Fatal("OCI availability should advertise network isolation support")
	}
	if !containsString(passport.WorkCapabilitySummary.SupportedWorkKinds, "custom_runtime") {
		t.Fatalf("default work kinds = %v, want generic runtime kind", passport.WorkCapabilitySummary.SupportedWorkKinds)
	}
}

func TestBuildCapabilityPassportClonesSlices(t *testing.T) {
	runnerKinds := []string{"managed_oci"}
	workKinds := []string{"custom_runtime"}

	passport, err := BuildCapabilityPassport(BuildPassportInput{
		AgentVersion: "dev",
		OS:           "linux",
		Arch:         "amd64",
		HardwareProfile: HardwareProfile{
			CPUCores: 4,
			RAMBytes: 8 * 1024 * 1024 * 1024,
		},
		RuntimeProfile: RuntimeProfile{
			SupportedRunnerKinds: runnerKinds,
		},
		WorkCapabilitySummary: WorkCapabilitySummary{
			SupportedWorkKinds: workKinds,
		},
		CreatedAtUnixMs: 123,
	})
	if err != nil {
		t.Fatalf("BuildCapabilityPassport() error = %v", err)
	}

	runnerKinds[0] = "custom"
	workKinds[0] = "native_report"

	if got := passport.RuntimeProfile.SupportedRunnerKinds[0]; got != "managed_oci" {
		t.Fatalf("runner kind mutated through input slice: %q", got)
	}
	if got := passport.WorkCapabilitySummary.SupportedWorkKinds[0]; got != "custom_runtime" {
		t.Fatalf("work kind mutated through input slice: %q", got)
	}
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
