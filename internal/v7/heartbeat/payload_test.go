package heartbeat

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/Ryvion/node-agent/internal/hw"
	"github.com/Ryvion/node-agent/internal/v7/capability"
	"github.com/Ryvion/node-agent/internal/v7/kvprobe"
	"github.com/Ryvion/node-agent/internal/v7/modellease"
	"github.com/Ryvion/node-agent/internal/v7/netprofile"
	"github.com/Ryvion/node-agent/internal/v7/sandbox"
)

func TestV7HeartbeatEnabledFromEnv(t *testing.T) {
	t.Setenv(EnvV7Caps, "")
	if V7HeartbeatEnabledFromEnv() {
		t.Fatal("flag should be disabled when env is empty")
	}

	t.Setenv(EnvV7Caps, "1")
	if !V7HeartbeatEnabledFromEnv() {
		t.Fatal("flag should be enabled for 1")
	}
}

func TestBuildV7HeartbeatPayloadIncludesCapabilityPassport(t *testing.T) {
	payload, err := BuildV7HeartbeatPayload(validInput())
	if err != nil {
		t.Fatalf("BuildV7HeartbeatPayload() error = %v", err)
	}

	if payload.SchemaVersion != SchemaVersionV1 {
		t.Fatalf("schema version = %q, want %q", payload.SchemaVersion, SchemaVersionV1)
	}
	if payload.CapabilityPassport.SchemaVersion != capability.SchemaVersionV1 {
		t.Fatalf("passport schema = %q, want %q", payload.CapabilityPassport.SchemaVersion, capability.SchemaVersionV1)
	}
	if payload.CapabilityPassport.AgentVersion != "v1.2.3" {
		t.Fatalf("agent version = %q, want v1.2.3", payload.CapabilityPassport.AgentVersion)
	}
	if payload.CapabilityPassport.DeclaredCountry != "CA" {
		t.Fatalf("declared country = %q, want CA", payload.CapabilityPassport.DeclaredCountry)
	}
	if payload.NetworkProfile == nil {
		t.Fatal("network profile should be included when provided")
	}
	if payload.ModelLeaseSummary == nil || payload.ModelLeaseSummary.ResidentLeases != 1 {
		t.Fatalf("model lease summary = %+v, want one resident lease", payload.ModelLeaseSummary)
	}
	if payload.KVCapability == nil || payload.KVCapability.Reason != kvprobe.ReasonTextGenerationOnly {
		t.Fatalf("kv capability = %+v, want compact unsupported capability", payload.KVCapability)
	}
	if payload.CASSummary == nil || !payload.CASSummary.Enabled {
		t.Fatalf("CAS summary = %+v, want enabled", payload.CASSummary)
	}
	if payload.SandboxPolicySummary == nil || !payload.SandboxPolicySummary.RejectsUnsafePickle {
		t.Fatalf("sandbox summary = %+v, want unsafe pickle rejection", payload.SandboxPolicySummary)
	}
	if payload.EvidenceCapabilitySummary == nil || !payload.EvidenceCapabilitySummary.SupportsRYV3EvidencePayload {
		t.Fatalf("evidence summary = %+v, want RYV3 support", payload.EvidenceCapabilitySummary)
	}
}

func TestBuildV7HeartbeatPayloadContainsNoKnownSecrets(t *testing.T) {
	t.Setenv("RYV_DEMO_KEY", "demo-key-secret")
	t.Setenv("RYV_ADMIN_KEY", "admin-key-secret")
	t.Setenv("RYV_BIND_TOKEN", "bind-token-secret")
	t.Setenv("RYV_WALLET", "wallet-secret")

	payload, err := BuildV7HeartbeatPayload(validInput())
	if err != nil {
		t.Fatalf("BuildV7HeartbeatPayload() error = %v", err)
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	body := string(raw)
	for _, secret := range []string{"demo-key-secret", "admin-key-secret", "bind-token-secret", "wallet-secret"} {
		if strings.Contains(body, secret) {
			t.Fatalf("payload JSON leaked secret %q: %s", secret, body)
		}
	}
}

func TestBuildV7HeartbeatPayloadJSONMarshalWorks(t *testing.T) {
	payload, err := BuildV7HeartbeatPayload(validInput())
	if err != nil {
		t.Fatalf("BuildV7HeartbeatPayload() error = %v", err)
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	if !strings.Contains(string(raw), `"capability_passport"`) {
		t.Fatalf("payload JSON missing capability_passport: %s", string(raw))
	}
}

func TestBuildV7HeartbeatPayloadRejectsInvalidNetworkProfile(t *testing.T) {
	input := validInput()
	input.NetworkProfile.UploadMbpsP50 = -1
	if _, err := BuildV7HeartbeatPayload(input); err == nil {
		t.Fatal("BuildV7HeartbeatPayload() error = nil, want invalid network profile error")
	}
}

func TestBuildV7HeartbeatPayloadOmitsUnavailableLeaseAndCASSummaries(t *testing.T) {
	input := validInput()
	input.ModelLeases = nil
	input.CASCapabilitySummary = capability.CASCapabilitySummary{}

	payload, err := BuildV7HeartbeatPayload(input)
	if err != nil {
		t.Fatalf("BuildV7HeartbeatPayload() error = %v", err)
	}
	if payload.ModelLeaseSummary != nil {
		t.Fatalf("model lease summary = %+v, want nil when no lease state is available", payload.ModelLeaseSummary)
	}
	if payload.CASSummary != nil {
		t.Fatalf("CAS summary = %+v, want nil when no CAS state is available", payload.CASSummary)
	}
	if !payload.CapabilityPassport.ModelCapabilitySummary.SupportsModelLease {
		t.Fatal("model lease support capability should remain advertised")
	}
}

func validInput() BuildV7HeartbeatPayloadInput {
	return BuildV7HeartbeatPayloadInput{
		AgentVersion:    "v1.2.3",
		NodePublicKey:   strings.Repeat("a", 64),
		OS:              "linux",
		Arch:            "amd64",
		DeviceType:      "gpu",
		DeclaredCountry: "ca",
		HardwareCapabilities: hw.CapSet{
			GPUModel:     "NVIDIA GeForce RTX 4090",
			CPUCores:     16,
			RAMBytes:     64 * 1024 * 1024 * 1024,
			VRAMBytes:    24 * 1024 * 1024 * 1024,
			Sensors:      "nvidia-driver:555.85",
			TEESupported: true,
			TEEType:      "tdx",
		},
		RuntimeProfile: capability.RuntimeProfile{
			NativeInferenceSupported: true,
			OCIAvailable:             true,
			LlamaServerAvailable:     true,
			ImageRuntimeAvailable:    true,
			SupportedRunnerKinds:     []string{"native_streaming", "managed_oci"},
		},
		NetworkProfile: &netprofile.NetworkProfile{
			UploadMbpsP50:    0,
			UploadMbpsP95:    0,
			DownloadMbpsP50:  100,
			DownloadMbpsP95:  120,
			RTTMsP50:         10,
			RTTMsP95:         20,
			JitterMsP95:      2,
			LossRateP95:      0,
			ProbeTarget:      "https://hub.example/probe",
			SampleCount:      3,
			MeasuredAtUnixMs: 123,
		},
		ModelCapabilitySummary: capability.ModelCapabilitySummary{
			ResidentModelIDs:      []string{"ryvion-llama-3.2-3b"},
			MaxResidentModelBytes: 8 * 1024 * 1024 * 1024,
			SupportsModelLease:    true,
		},
		ModelLeases: []modellease.ModelLease{
			{
				LeaseID:              "lease-1",
				NodeID:               "node-1",
				ModelID:              "ryvion-llama-3.2-3b",
				State:                modellease.ModelLeaseStateResident,
				VRAMReservedBytes:    8 * 1024 * 1024 * 1024,
				LeaseExpiresAtUnixMs: 456,
				UpdatedAtUnixMs:      123,
			},
		},
		KVCapability: &kvprobe.Capability{
			KVAccessSupported:          false,
			KVSnapshotSupported:        false,
			HiddenStateAccessSupported: false,
			LogitsAccessSupported:      false,
			AttentionHookSupported:     false,
			Backend:                    kvprobe.BackendLlamaCPP,
			ModelID:                    "ryvion-llama-3.2-3b",
			ModelLoaded:                true,
			RuntimeKind:                kvprobe.RuntimeKindNative,
			Reason:                     kvprobe.ReasonTextGenerationOnly,
		},
		CASCapabilitySummary: capability.CASCapabilitySummary{
			Enabled:  true,
			MaxBytes: 100 * 1024 * 1024,
		},
		SandboxCapabilitySummary: capability.SandboxCapabilitySummary{
			RejectsUnsafePickle:        true,
			RunnerAllowlistEnabled:     true,
			FilesystemIsolationPlanned: true,
			NetworkIsolationSupported:  true,
		},
		SandboxPolicy: ptr(sandbox.DefaultSandboxPolicy()),
		EvidenceCapabilitySummary: capability.EvidenceCapabilitySummary{
			SupportsArtifactManifest:    true,
			SupportsRYV3EvidencePayload: true,
			SupportsRuntimeHash:         true,
		},
		CreatedAtUnixMs: 123,
	}
}

func ptr[T any](value T) *T {
	return &value
}
