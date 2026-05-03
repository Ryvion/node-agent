package capability

import (
	"testing"

	"github.com/Ryvion/node-agent/internal/hw"
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
			NativeInferenceSupported: true,
			OCIAvailable:             true,
			LlamaServerAvailable:     true,
			ImageRuntimeAvailable:    true,
			SupportedRunnerKinds:     []string{"native_streaming", "managed_oci"},
		},
		ModelCapabilitySummary: ModelCapabilitySummary{
			ResidentModelIDs:      []string{"ryvion-llama-3.2-3b"},
			MaxResidentModelBytes: 8 * 1024 * 1024 * 1024,
			SupportsModelLease:    true,
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
	if !passport.SandboxCapabilitySummary.RejectsUnsafePickle {
		t.Fatal("builder should default to rejecting unsafe pickle")
	}
	if !passport.SandboxCapabilitySummary.NetworkIsolationSupported {
		t.Fatal("OCI availability should advertise network isolation support")
	}
	if !containsString(passport.ModelCapabilitySummary.SupportedModelFormats, "gguf") ||
		!containsString(passport.ModelCapabilitySummary.SupportedModelFormats, "safetensors") {
		t.Fatalf("default formats = %v, want gguf and safetensors", passport.ModelCapabilitySummary.SupportedModelFormats)
	}
}

func TestBuildCapabilityPassportClonesSlices(t *testing.T) {
	runnerKinds := []string{"native_streaming"}
	formats := []string{"GGUF"}
	residentModels := []string{"model-a"}

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
		ModelCapabilitySummary: ModelCapabilitySummary{
			SupportedModelFormats: formats,
			ResidentModelIDs:      residentModels,
		},
		CreatedAtUnixMs: 123,
	})
	if err != nil {
		t.Fatalf("BuildCapabilityPassport() error = %v", err)
	}

	runnerKinds[0] = "managed_oci"
	formats[0] = "pickle"
	residentModels[0] = "model-b"

	if got := passport.RuntimeProfile.SupportedRunnerKinds[0]; got != "native_streaming" {
		t.Fatalf("runner kind mutated through input slice: %q", got)
	}
	if got := passport.ModelCapabilitySummary.SupportedModelFormats[0]; got != "gguf" {
		t.Fatalf("format mutated through input slice: %q", got)
	}
	if got := passport.ModelCapabilitySummary.ResidentModelIDs[0]; got != "model-a" {
		t.Fatalf("resident model mutated through input slice: %q", got)
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
