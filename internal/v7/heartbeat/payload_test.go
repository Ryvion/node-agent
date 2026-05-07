package heartbeat

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/Ryvion/node-agent/internal/hw"
	"github.com/Ryvion/node-agent/internal/v7/capability"
	"github.com/Ryvion/node-agent/internal/v7/kvprobe"
	"github.com/Ryvion/node-agent/internal/v7/modellease"
	"github.com/Ryvion/node-agent/internal/v7/netprofile"
	"github.com/Ryvion/node-agent/internal/v7/runtimeinventory"
	"github.com/Ryvion/node-agent/internal/v7/sandbox"
	"github.com/Ryvion/node-agent/internal/v7/tensoraccess"
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
	if payload.TensorAccess.Provider != tensoraccess.ProviderNoop {
		t.Fatalf("tensor_access provider = %q, want %q", payload.TensorAccess.Provider, tensoraccess.ProviderNoop)
	}
	if !payload.TensorAccess.TensorPlaneDemoSupported || !payload.TensorAccess.ModelLoaded {
		t.Fatalf("tensor_access = %+v, want demo support and loaded model from input", payload.TensorAccess)
	}
	if payload.RuntimeInventory.RuntimeKind != runtimeinventory.RuntimeKindNative ||
		payload.RuntimeInventory.Backend != runtimeinventory.BackendNative ||
		payload.RuntimeInventory.Provider != runtimeinventory.ProviderNoop {
		t.Fatalf("runtime_inventory identity = %+v, want native/noop inventory", payload.RuntimeInventory)
	}
	if len(payload.RuntimeInventory.LoadedModels) != 1 || payload.RuntimeInventory.LoadedModels[0].ModelID != "ryvion-llama-3.2-3b" {
		t.Fatalf("runtime_inventory loaded_models = %+v, want local native model", payload.RuntimeInventory.LoadedModels)
	}
	if !payload.RuntimeInventory.CandidateBackends.LlamaCPPDetected ||
		!payload.RuntimeInventory.CandidateBackends.PythonTransformersDetected ||
		!payload.RuntimeInventory.CandidateBackends.GGUFModelsDetected {
		t.Fatalf("runtime_inventory candidate_backends = %+v, want mocked candidates", payload.RuntimeInventory.CandidateBackends)
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
	if !strings.Contains(string(raw), `"tensor_access"`) {
		t.Fatalf("payload JSON missing tensor_access: %s", string(raw))
	}
	if !strings.Contains(string(raw), `"runtime_inventory"`) {
		t.Fatalf("payload JSON missing runtime_inventory: %s", string(raw))
	}
	if !strings.Contains(string(raw), `"loaded_models"`) || !strings.Contains(string(raw), `"candidate_backends"`) {
		t.Fatalf("payload JSON missing runtime inventory details: %s", string(raw))
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

func TestBuildV7HeartbeatPayloadDefaultsUnsupportedNoopTensorAccess(t *testing.T) {
	input := validInput()
	input.TensorAccess = nil

	payload, err := BuildV7HeartbeatPayload(input)
	if err != nil {
		t.Fatalf("BuildV7HeartbeatPayload() error = %v", err)
	}
	if payload.TensorAccess.Provider != tensoraccess.ProviderNoop {
		t.Fatalf("tensor_access provider = %q, want noop", payload.TensorAccess.Provider)
	}
	if payload.TensorAccess.KVAccessSupported ||
		payload.TensorAccess.KVSnapshotSupported ||
		payload.TensorAccess.HiddenStateAccessSupported ||
		payload.TensorAccess.LogitsAccessSupported ||
		payload.TensorAccess.AttentionHookSupported {
		t.Fatalf("default tensor_access should not advertise hooks: %+v", payload.TensorAccess)
	}
	if payload.TensorAccess.ModelLoaded {
		t.Fatalf("default tensor_access model_loaded = true: %+v", payload.TensorAccess)
	}
	if payload.TensorAccess.Reason == "" {
		t.Fatalf("default tensor_access reason is empty: %+v", payload.TensorAccess)
	}
}

func TestBuildV7HeartbeatPayloadDefaultsUnsupportedNoopRuntimeInventory(t *testing.T) {
	input := validInput()
	input.RuntimeInventory = nil

	payload, err := BuildV7HeartbeatPayload(input)
	if err != nil {
		t.Fatalf("BuildV7HeartbeatPayload() error = %v", err)
	}
	if payload.RuntimeInventory.RuntimeKind != runtimeinventory.RuntimeKindUnknown {
		t.Fatalf("runtime_kind = %q, want unknown", payload.RuntimeInventory.RuntimeKind)
	}
	if payload.RuntimeInventory.Backend != runtimeinventory.BackendUnknown {
		t.Fatalf("backend = %q, want unknown", payload.RuntimeInventory.Backend)
	}
	if payload.RuntimeInventory.Provider != runtimeinventory.ProviderNoop {
		t.Fatalf("provider = %q, want noop", payload.RuntimeInventory.Provider)
	}
	if payload.RuntimeInventory.ProcessMode != runtimeinventory.ProcessModeUnknown {
		t.Fatalf("process_mode = %q, want unknown", payload.RuntimeInventory.ProcessMode)
	}
	if payload.RuntimeInventory.NativeInferenceReady {
		t.Fatalf("native_inference_ready = true, want false: %+v", payload.RuntimeInventory)
	}
	if len(payload.RuntimeInventory.LoadedModels) != 0 {
		t.Fatalf("loaded_models = %+v, want empty unsupported state", payload.RuntimeInventory.LoadedModels)
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("json.Marshal(payload) error = %v", err)
	}
	if !strings.Contains(string(raw), `"runtime_inventory"`) {
		t.Fatalf("payload JSON missing runtime_inventory: %s", raw)
	}
}

func TestBuildV7HeartbeatPayloadNormalizesTensorAccess(t *testing.T) {
	input := validInput()
	longReason := "  " + strings.Repeat("r", 300) + "\n"
	longModelID := "  " + strings.Repeat("m", 160) + "\t"
	input.TensorAccess = &tensoraccess.TensorAccessCapability{
		Provider:    "  NOOP  ",
		Backend:     "  NATIVE ",
		RuntimeKind: "  NATIVE ",
		ModelID:     longModelID,
		ModelLoaded: true,
		Reason:      longReason,
	}

	payload, err := BuildV7HeartbeatPayload(input)
	if err != nil {
		t.Fatalf("BuildV7HeartbeatPayload() error = %v", err)
	}
	if payload.TensorAccess.Provider != tensoraccess.ProviderNoop ||
		payload.TensorAccess.Backend != tensoraccess.BackendNative ||
		payload.TensorAccess.RuntimeKind != tensoraccess.RuntimeKindNative {
		t.Fatalf("tensor_access was not normalized: %+v", payload.TensorAccess)
	}
	if len(payload.TensorAccess.ModelID) != 128 {
		t.Fatalf("model_id length = %d, want 128", len(payload.TensorAccess.ModelID))
	}
	if len(payload.TensorAccess.Reason) != 256 {
		t.Fatalf("reason length = %d, want 256", len(payload.TensorAccess.Reason))
	}
	if strings.ContainsAny(payload.TensorAccess.ModelID, " \t\n") || strings.ContainsAny(payload.TensorAccess.Reason, "\t\n") {
		t.Fatalf("tensor_access text was not sanitized: %+v", payload.TensorAccess)
	}
}

func TestBuildV7HeartbeatPayloadContainsNoRawTensorPromptOrOutputFields(t *testing.T) {
	payload, err := BuildV7HeartbeatPayload(validInput())
	if err != nil {
		t.Fatalf("BuildV7HeartbeatPayload() error = %v", err)
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	body := strings.ToLower(string(raw))
	for _, forbidden := range []string{
		"raw_tensor",
		"tensor_bytes",
		"key_data",
		"value_data",
		"query_vector",
		"raw_prompt",
		"prompt_text",
		"generated_text",
		"model_output",
		"output_text",
	} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("payload JSON contains forbidden field marker %q: %s", forbidden, body)
		}
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
		TensorAccess: &tensoraccess.TensorAccessCapability{
			Provider:                   tensoraccess.ProviderNoop,
			Backend:                    tensoraccess.BackendNative,
			KVAccessSupported:          false,
			KVSnapshotSupported:        false,
			HiddenStateAccessSupported: false,
			LogitsAccessSupported:      false,
			AttentionHookSupported:     false,
			TensorPlaneDemoSupported:   true,
			ModelLoaded:                true,
			RuntimeKind:                tensoraccess.RuntimeKindNative,
			ModelID:                    "ryvion-llama-3.2-3b",
			Reason:                     tensoraccess.ReasonTextGenerationOnly,
		},
		RuntimeInventory: ptr(validRuntimeInventory()),
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

func validRuntimeInventory() runtimeinventory.Inventory {
	return runtimeinventory.BuildInventory(runtimeinventory.RuntimeStatus{
		RuntimeKind:             runtimeinventory.RuntimeKindNative,
		Backend:                 runtimeinventory.BackendNative,
		Provider:                runtimeinventory.ProviderNoop,
		ProcessMode:             runtimeinventory.ProcessModeSidecar,
		NativeInferenceReady:    true,
		NativeModel:             "ryvion-llama-3.2-3b",
		ModelLoaded:             true,
		SupportsTextGeneration:  true,
		SupportsStreaming:       true,
		SupportsTensorPlaneDemo: true,
		Reason:                  tensoraccess.ReasonTextGenerationOnly,
	}, runtimeinventory.CandidateBackendDetector{
		LookPath: func(name string) (string, error) {
			switch name {
			case "llama-server", "python3":
				return "/usr/local/bin/" + name, nil
			default:
				return "", errors.New("not found")
			}
		},
		ReadDirNames: func(dir string, limit int) ([]string, error) {
			return []string{"model.Q4_K_M.gguf"}, nil
		},
		ConfiguredModelDirs: []string{"/tmp/ryvion-models"},
		UserHomeDir: func() (string, error) {
			return "", errors.New("not configured")
		},
	})
}

func ptr[T any](value T) *T {
	return &value
}
