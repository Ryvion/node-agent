package heartbeat

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/Ryvion/ryvion-node/internal/hw"
	"github.com/Ryvion/ryvion-node/internal/models/cache"
	"github.com/Ryvion/ryvion-node/internal/models/lease"
	"github.com/Ryvion/ryvion-node/internal/models/policy"
	"github.com/Ryvion/ryvion-node/internal/runtimes/llamacpp"
	"github.com/Ryvion/ryvion-node/internal/v7/backendprobe"
	"github.com/Ryvion/ryvion-node/internal/v7/capability"
	v7hardware "github.com/Ryvion/ryvion-node/internal/v7/hardware"
	"github.com/Ryvion/ryvion-node/internal/v7/kvprobe"
	"github.com/Ryvion/ryvion-node/internal/v7/netprofile"
	"github.com/Ryvion/ryvion-node/internal/v7/runtimeinventory"
	"github.com/Ryvion/ryvion-node/internal/v7/sandbox"
	"github.com/Ryvion/ryvion-node/internal/v7/tensoraccess"
)

func TestV7HeartbeatEnabledFromEnv(t *testing.T) {
	t.Setenv(EnvV7Caps, "")
	if !V7HeartbeatEnabledFromEnv() {
		t.Fatal("V7 heartbeat should be enabled by default when env is empty")
	}

	t.Setenv(EnvV7Caps, "0")
	if V7HeartbeatEnabledFromEnv() {
		t.Fatal("V7 heartbeat should be disabled for explicit 0")
	}
}

func TestExperimentalSpeculativeHeartbeatEnabledFromEnv(t *testing.T) {
	t.Setenv(EnvExperimentalSpeculativeFields, "")
	if ExperimentalSpeculativeHeartbeatEnabledFromEnv() {
		t.Fatal("experimental speculative heartbeat should be disabled by default")
	}

	t.Setenv(EnvExperimentalSpeculativeFields, "1")
	if !ExperimentalSpeculativeHeartbeatEnabledFromEnv() {
		t.Fatal("experimental speculative heartbeat should be enabled for explicit 1")
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
	if payload.NodeID != strings.Repeat("a", 64) {
		t.Fatalf("node_id = %q, want node public key fallback", payload.NodeID)
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
	if len(payload.RuntimeInventory.BackendCandidates) == 0 {
		t.Fatalf("runtime_inventory backend_candidates = %+v, want detailed backend rows", payload.RuntimeInventory.BackendCandidates)
	}
	if len(payload.RuntimeInventory.GGUFModels) != 1 {
		t.Fatalf("runtime_inventory gguf_models = %+v, want one mocked GGUF model", payload.RuntimeInventory.GGUFModels)
	}
	if payload.HardwareCapacity.OS != "linux" ||
		payload.HardwareCapacity.Arch != "amd64" ||
		payload.HardwareCapacity.GPUVendor != v7hardware.GPUVendorNVIDIA ||
		payload.HardwareCapacity.GPUName != "NVIDIA GeForce RTX 4090" ||
		payload.HardwareCapacity.GPUVRAMBytes != 24*1024*1024*1024 {
		t.Fatalf("hardware_capacity = %+v, want mocked RTX 4090 capacity", payload.HardwareCapacity)
	}
	if payload.ModelPolicy.CacheDir != "/tmp/ryvion-models" || payload.ModelPolicy.MaxSingleModelBytes != 8*1024*1024*1024 {
		t.Fatalf("model_policy = %+v, want mocked cache policy", payload.ModelPolicy)
	}
	if !payload.ModelPolicy.RuntimePolicy.AllowRuntimeExecution ||
		payload.ModelPolicy.RuntimePolicy.MaxRuntimeModelBytes != 4*1024*1024*1024 ||
		payload.ModelPolicy.RuntimePolicy.MaxRuntimeParameterCountBillions != 8 ||
		payload.ModelPolicy.RuntimePolicy.AllowLargeModels {
		t.Fatalf("model_policy.runtime_policy = %+v, want mocked runtime policy", payload.ModelPolicy.RuntimePolicy)
	}
	if len(payload.ModelPolicy.RuntimePolicy.DenyModelIDs) != 1 ||
		payload.ModelPolicy.RuntimePolicy.DenyModelIDs[0] != "phi-4-Q4_K_M.gguf" {
		t.Fatalf("runtime_policy.deny_model_ids = %+v", payload.ModelPolicy.RuntimePolicy.DenyModelIDs)
	}
	if payload.ModelCache.CacheDir != "/tmp/ryvion-models" || len(payload.ModelCache.Models) != 1 {
		t.Fatalf("model_cache = %+v, want mocked cache status", payload.ModelCache)
	}
	if !payload.RuntimeInventory.CandidateBackends.LlamaCPPDetected ||
		!payload.RuntimeInventory.CandidateBackends.PythonTransformersDetected ||
		!payload.RuntimeInventory.CandidateBackends.GGUFModelsDetected {
		t.Fatalf("runtime_inventory candidate_backends = %+v, want mocked candidates", payload.RuntimeInventory.CandidateBackends)
	}
	if !payload.BackendProbes.LlamaCPP.Available ||
		payload.BackendProbes.LlamaCPP.BinaryPath != "/opt/ryvion/bin/llama-cli" ||
		payload.BackendProbes.LlamaCPP.ServerBinaryPath != "/opt/ryvion/bin/llama-server" ||
		payload.BackendProbes.LlamaCPP.BenchBinaryPath != "/opt/ryvion/bin/llama-bench" ||
		payload.BackendProbes.LlamaCPP.Version != "llama.cpp build 456" ||
		!payload.BackendProbes.LlamaCPP.GGUFModelsDetected ||
		!payload.BackendProbes.LlamaCPP.CandidateForFastTextRuntime ||
		!payload.BackendProbes.LlamaCPP.CandidateForRealTensorAccess {
		t.Fatalf("backend_probes.llama_cpp = %+v, want mocked llama.cpp probe", payload.BackendProbes.LlamaCPP)
	}
	if payload.BackendProbes.LlamaCPP.SupportsKVAccess || payload.BackendProbes.LlamaCPP.SupportsTensorHooks {
		t.Fatalf("backend_probes should not claim KV/tensor hooks: %+v", payload.BackendProbes.LlamaCPP)
	}
	if !payload.BackendRuntimes.LlamaCPP.Enabled ||
		!payload.BackendRuntimes.LlamaCPP.Running ||
		!payload.BackendRuntimes.LlamaCPP.Healthy ||
		!payload.BackendRuntimes.LlamaCPP.Loaded ||
		!payload.BackendRuntimes.LlamaCPP.Warm ||
		payload.BackendRuntimes.LlamaCPP.ModelID != "Llama-3.2-3B-Instruct-Q4_K_M.gguf" {
		t.Fatalf("backend_runtimes.llama_cpp = %+v, want active loaded sidecar", payload.BackendRuntimes.LlamaCPP)
	}
	if !payload.CapabilityProfile.V7DashboardInference ||
		!payload.CapabilityProfile.TextOutput ||
		!payload.CapabilityProfile.Streaming ||
		!payload.CapabilityProfile.HashMetricsReceipts ||
		!payload.CapabilityProfile.BackendTextGeneration ||
		!payload.CapabilityProfile.BackendWarm ||
		!payload.CapabilityProfile.Ready {
		t.Fatalf("capability_profile = %+v, want default inference capabilities from mocked backend/model", payload.CapabilityProfile)
	}
	if len(payload.CapabilityProfile.Models) != 1 ||
		payload.CapabilityProfile.Models[0].ModelID != "Llama-3.2-3B-Instruct-Q4_K_M.gguf" ||
		!payload.CapabilityProfile.Models[0].Resident ||
		!payload.CapabilityProfile.Models[0].Runnable {
		t.Fatalf("capability_profile models = %+v, want resident runnable llama", payload.CapabilityProfile.Models)
	}
	if payload.CapabilityProfile.SpeculativeDecoding != nil {
		t.Fatalf("capability_profile.speculative_decoding = %+v, want omitted by default", payload.CapabilityProfile.SpeculativeDecoding)
	}
	if len(payload.SpeculativeProfiles) != 0 {
		t.Fatalf("speculative_profiles = %+v, want omitted from heartbeat payload", payload.SpeculativeProfiles)
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
	if !strings.Contains(string(raw), `"hardware_capacity"`) {
		t.Fatalf("payload JSON missing hardware_capacity: %s", string(raw))
	}
	if !strings.Contains(string(raw), `"model_policy"`) ||
		!strings.Contains(string(raw), `"runtime_policy"`) ||
		!strings.Contains(string(raw), `"deny_model_ids"`) ||
		!strings.Contains(string(raw), `"model_cache"`) {
		t.Fatalf("payload JSON missing model policy/cache: %s", string(raw))
	}
	if !strings.Contains(string(raw), `"loaded_models"`) ||
		!strings.Contains(string(raw), `"candidate_backends"`) ||
		!strings.Contains(string(raw), `"backend_candidates"`) ||
		!strings.Contains(string(raw), `"gguf_models"`) {
		t.Fatalf("payload JSON missing runtime inventory details: %s", string(raw))
	}
	if !strings.Contains(string(raw), `"backend_probes"`) ||
		!strings.Contains(string(raw), `"backend_runtimes"`) ||
		!strings.Contains(string(raw), `"llama_cpp"`) {
		t.Fatalf("payload JSON missing backend details: %s", string(raw))
	}
	if !strings.Contains(string(raw), `"capability_profile"`) ||
		!strings.Contains(string(raw), `"v7_dashboard_inference"`) ||
		!strings.Contains(string(raw), `"hash_metrics_receipts"`) {
		t.Fatalf("payload JSON missing capability profile: %s", string(raw))
	}
	if strings.Contains(string(raw), `"speculative_decoding"`) || strings.Contains(string(raw), `"speculative_profiles"`) {
		t.Fatalf("payload JSON should omit speculative fields by default: %s", string(raw))
	}
}

func TestBuildV7HeartbeatPayloadIncludesExperimentalSpeculativeCapabilityWhenEnabled(t *testing.T) {
	t.Setenv(EnvExperimentalSpeculativeFields, "1")

	payload, err := BuildV7HeartbeatPayload(validInput())
	if err != nil {
		t.Fatalf("BuildV7HeartbeatPayload() error = %v", err)
	}
	if payload.CapabilityProfile.SpeculativeDecoding == nil ||
		!payload.CapabilityProfile.SpeculativeDecoding.Supported ||
		!payload.CapabilityProfile.SpeculativeDecoding.Enabled ||
		payload.CapabilityProfile.SpeculativeDecoding.DefaultMethod != "ngram" {
		t.Fatalf("capability_profile.speculative_decoding = %+v, want experimental ngram capability", payload.CapabilityProfile.SpeculativeDecoding)
	}
	if len(payload.SpeculativeProfiles) != 0 {
		t.Fatalf("speculative_profiles = %+v, want no top-level heartbeat profiles", payload.SpeculativeProfiles)
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	if !strings.Contains(string(raw), `"speculative_decoding"`) {
		t.Fatalf("experimental payload JSON missing capability_profile.speculative_decoding: %s", raw)
	}
	if strings.Contains(string(raw), `"speculative_profiles"`) {
		t.Fatalf("experimental payload JSON should not include top-level speculative_profiles: %s", raw)
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

func TestBuildV7HeartbeatPayloadDefaultsSafeHardwareCapacity(t *testing.T) {
	input := validInput()
	input.HardwareCapacity = nil

	payload, err := BuildV7HeartbeatPayload(input)
	if err != nil {
		t.Fatalf("BuildV7HeartbeatPayload() error = %v", err)
	}
	if payload.HardwareCapacity.OS != "linux" || payload.HardwareCapacity.Arch != "amd64" {
		t.Fatalf("hardware_capacity identity = %+v, want input OS/arch fallback", payload.HardwareCapacity)
	}
	if payload.HardwareCapacity.GPUDetected ||
		payload.HardwareCapacity.GPUVendor != v7hardware.GPUVendorUnknown ||
		payload.HardwareCapacity.GPUName != "unknown" ||
		payload.HardwareCapacity.PowerProfile != v7hardware.PowerProfileUnknown ||
		payload.HardwareCapacity.ThermalRisk != v7hardware.ThermalRiskUnknown {
		t.Fatalf("default hardware_capacity should be safe unknown capacity: %+v", payload.HardwareCapacity)
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("json.Marshal(payload) error = %v", err)
	}
	if !strings.Contains(string(raw), `"hardware_capacity"`) {
		t.Fatalf("payload JSON missing hardware_capacity: %s", raw)
	}
}

func TestBuildV7HeartbeatPayloadDefaultsSafeBackendProbes(t *testing.T) {
	input := validInput()
	input.BackendProbes = nil

	payload, err := BuildV7HeartbeatPayload(input)
	if err != nil {
		t.Fatalf("BuildV7HeartbeatPayload() error = %v", err)
	}
	if payload.BackendProbes.LlamaCPP.Available ||
		payload.BackendProbes.LlamaCPP.SupportsTextGeneration ||
		payload.BackendProbes.LlamaCPP.SupportsStreaming ||
		payload.BackendProbes.LlamaCPP.SupportsOpenAICompatibleServer ||
		payload.BackendProbes.LlamaCPP.SupportsKVAccess ||
		payload.BackendProbes.LlamaCPP.SupportsTensorHooks ||
		payload.BackendProbes.LlamaCPP.CandidateForFastTextRuntime ||
		payload.BackendProbes.LlamaCPP.CandidateForRealTensorAccess {
		t.Fatalf("default backend_probes should be safe false values: %+v", payload.BackendProbes.LlamaCPP)
	}
	if payload.BackendProbes.LlamaCPP.Version != "unknown" ||
		payload.BackendProbes.LlamaCPP.Reason != "llama.cpp binary not detected" {
		t.Fatalf("default backend_probes version/reason = %q/%q", payload.BackendProbes.LlamaCPP.Version, payload.BackendProbes.LlamaCPP.Reason)
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("json.Marshal(payload) error = %v", err)
	}
	if !strings.Contains(string(raw), `"backend_probes"`) || !strings.Contains(string(raw), `"llama_cpp"`) {
		t.Fatalf("payload JSON missing backend_probes: %s", raw)
	}
}

func TestBuildV7HeartbeatPayloadDefaultsSafeBackendRuntimes(t *testing.T) {
	input := validInput()
	input.BackendRuntimes = nil

	payload, err := BuildV7HeartbeatPayload(input)
	if err != nil {
		t.Fatalf("BuildV7HeartbeatPayload() error = %v", err)
	}
	runtime := payload.BackendRuntimes.LlamaCPP
	if runtime.Enabled || runtime.Available || runtime.Running || runtime.Healthy || runtime.Loaded || runtime.Warm {
		t.Fatalf("default backend_runtimes.llama_cpp should be safe false values: %+v", runtime)
	}
	if runtime.SupportsKVAccess || runtime.SupportsTensorHooks {
		t.Fatalf("default backend_runtimes should not claim KV/tensor hooks: %+v", runtime)
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("json.Marshal(payload) error = %v", err)
	}
	if !strings.Contains(string(raw), `"backend_runtimes"`) || !strings.Contains(string(raw), `"llama_cpp"`) {
		t.Fatalf("payload JSON missing backend_runtimes: %s", raw)
	}
}

func TestBuildV7HeartbeatPayloadNormalizesBackendRuntimes(t *testing.T) {
	input := validInput()
	longPath := "/tmp/" + strings.Repeat("p", 600)
	input.BackendRuntimes = &llamacpp.BackendRuntimes{
		LlamaCPP: llamacpp.BackendRuntimeStatus{
			Enabled:                true,
			Available:              true,
			Running:                true,
			Healthy:                true,
			Backend:                llamacpp.BackendName,
			BaseURL:                "http://127.0.0.1:45910",
			ModelID:                "model.Q4_K_M.gguf",
			ModelPath:              longPath + "\n",
			ModelFilename:          "model.Q4_K_M.gguf",
			Loaded:                 true,
			Warm:                   true,
			OpenAICompatible:       true,
			SupportsTextGeneration: true,
			SupportsStreaming:      true,
			SupportsKVAccess:       true,
			SupportsTensorHooks:    true,
			LastHealthAtUnixMs:     123,
			LastError:              strings.Repeat("r", 300) + "\n",
		},
	}

	payload, err := BuildV7HeartbeatPayload(input)
	if err != nil {
		t.Fatalf("BuildV7HeartbeatPayload() error = %v", err)
	}
	runtime := payload.BackendRuntimes.LlamaCPP
	if len(runtime.ModelPath) != 512 {
		t.Fatalf("model_path length = %d, want 512", len(runtime.ModelPath))
	}
	if len(runtime.LastError) != 256 {
		t.Fatalf("last_error length = %d, want 256", len(runtime.LastError))
	}
	if runtime.SupportsKVAccess || runtime.SupportsTensorHooks {
		t.Fatalf("backend_runtimes should not advertise KV/tensor hooks: %+v", runtime)
	}
	if strings.ContainsAny(runtime.ModelPath+runtime.LastError, "\t\n\r") {
		t.Fatalf("backend runtime text still contains control whitespace: %+v", runtime)
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

func TestBuildV7HeartbeatPayloadNormalizesBackendProbes(t *testing.T) {
	input := validInput()
	longPath := "/tmp/" + strings.Repeat("p", 600)
	input.BackendProbes = &backendprobe.Probes{
		LlamaCPP: backendprobe.LlamaCPPProbe{
			Available:                      true,
			BinaryPath:                     longPath + "\n",
			ServerBinaryPath:               longPath + "\t",
			BenchBinaryPath:                longPath + "\r",
			Version:                        "ggml_metal_device_init: tensor API disabled\nllama.cpp build 789\n",
			SupportsTextGeneration:         true,
			SupportsStreaming:              true,
			SupportsOpenAICompatibleServer: true,
			SupportsKVAccess:               true,
			SupportsTensorHooks:            true,
			CandidateForFastTextRuntime:    true,
			CandidateForRealTensorAccess:   true,
			Reason:                         strings.Repeat("r", 300) + "\n",
		},
	}

	payload, err := BuildV7HeartbeatPayload(input)
	if err != nil {
		t.Fatalf("BuildV7HeartbeatPayload() error = %v", err)
	}
	probe := payload.BackendProbes.LlamaCPP
	if len(probe.BinaryPath) != 512 ||
		len(probe.ServerBinaryPath) != 512 ||
		len(probe.BenchBinaryPath) != 512 {
		t.Fatalf("probe paths were not capped: %+v", probe)
	}
	if probe.Version != "llama.cpp build 789" {
		t.Fatalf("version = %q, want clean llama.cpp version line", probe.Version)
	}
	if len(probe.Reason) != 256 {
		t.Fatalf("reason length = %d, want 256", len(probe.Reason))
	}
	if strings.ContainsAny(probe.BinaryPath+probe.ServerBinaryPath+probe.BenchBinaryPath+probe.Reason, "\t\n\r") {
		t.Fatalf("probe text still contains control whitespace: %+v", probe)
	}
	if probe.SupportsKVAccess || probe.SupportsTensorHooks {
		t.Fatalf("normalization must not advertise KV/tensor hooks: %+v", probe)
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
		HardwareCapacity: ptr(validHardwareCapacity()),
		ModelPolicy: ptr(modelpolicy.Status{
			AutoDownload:           false,
			MaxSingleModelBytes:    8 * 1024 * 1024 * 1024,
			MaxCacheBytes:          50 * 1024 * 1024 * 1024,
			CacheDir:               "/tmp/ryvion-models",
			AllowedFamilies:        []string{"llama", "phi", "qwen", "gemma"},
			AllowedFormats:         []string{"gguf"},
			KeepWarmModelIDs:       []string{"Llama-3.2-3B-Instruct-Q4_K_M.gguf"},
			EvictionPolicy:         "lru",
			AllowLicenseRestricted: false,
			RuntimePolicy: modelpolicy.RuntimePolicy{
				AllowRuntimeExecution:              true,
				MaxRuntimeModelBytes:               4 * 1024 * 1024 * 1024,
				MaxRuntimeParameterCountBillions:   8,
				AllowCPUOffload:                    true,
				AllowLargeModels:                   false,
				DenyModelIDs:                       []string{"phi-4-Q4_K_M.gguf"},
				AllowModelIDs:                      []string{},
				DenyFamilies:                       []string{},
				AllowFamilies:                      []string{"llama"},
				RequireExplicitAllowForLargeModels: true,
			},
		}),
		ModelCache: ptr(modelcache.Status{
			CacheDir:   "/tmp/ryvion-models",
			TotalBytes: 2048,
			Models: []modelcache.Model{{
				ModelID:          "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
				Filename:         "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
				Path:             "/tmp/ryvion-models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
				SizeBytes:        2048,
				FamilyHint:       "llama",
				QuantizationHint: "Q4_K_M",
				Format:           "gguf",
				Installed:        true,
				HashVerified:     false,
				LastSeenAt:       time.Unix(100, 0),
			}},
		}),
		BackendProbes:   ptr(validBackendProbes()),
		BackendRuntimes: ptr(validBackendRuntimes()),
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

func validHardwareCapacity() v7hardware.CapacityInventory {
	return v7hardware.NormalizeInventory(v7hardware.CapacityInventory{
		OS:                      "linux",
		Arch:                    "amd64",
		CPULogicalCores:         16,
		SystemRAMBytes:          64 * 1024 * 1024 * 1024,
		AvailableRAMBytes:       48 * 1024 * 1024 * 1024,
		GPUDetected:             true,
		GPUVendor:               v7hardware.GPUVendorNVIDIA,
		GPUName:                 "NVIDIA GeForce RTX 4090",
		GPUVRAMBytes:            24 * 1024 * 1024 * 1024,
		CUDAAvailable:           true,
		DiskFreeBytesModelCache: 90 * 1024 * 1024 * 1024,
		PowerProfile:            v7hardware.PowerProfileDesktop,
		ThermalRisk:             v7hardware.ThermalRiskLow,
	})
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
		VersionCommand: func(path string, timeout time.Duration) (string, error) {
			return "llama.cpp build 456", nil
		},
		ConfiguredModelDirs: []string{"/tmp/ryvion-models"},
		UserHomeDir: func() (string, error) {
			return "", errors.New("not configured")
		},
	})
}

func validBackendProbes() backendprobe.Probes {
	return backendprobe.NormalizeProbes(backendprobe.Probes{
		LlamaCPP: backendprobe.LlamaCPPProbe{
			Available:                      true,
			BinaryPath:                     "/opt/ryvion/bin/llama-cli",
			ServerBinaryPath:               "/opt/ryvion/bin/llama-server",
			BenchBinaryPath:                "/opt/ryvion/bin/llama-bench",
			Version:                        "llama.cpp build 456",
			GGUFModelsDetected:             true,
			ProbeModelConfigured:           false,
			SupportsTextGeneration:         true,
			SupportsStreaming:              true,
			SupportsOpenAICompatibleServer: true,
			SupportsKVAccess:               false,
			SupportsTensorHooks:            false,
			CandidateForFastTextRuntime:    true,
			CandidateForRealTensorAccess:   true,
			Reason:                         "llama.cpp detected; real KV/tensor hooks require adapter implementation",
		},
	})
}

func validBackendRuntimes() llamacpp.BackendRuntimes {
	return llamacpp.BuildBackendRuntimes(llamacpp.LlamaCppSidecarStatus{
		Enabled:                true,
		Available:              true,
		Running:                true,
		Healthy:                true,
		BaseURL:                "http://127.0.0.1:45910",
		ModelPath:              "/tmp/ryvion-models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		ModelFilename:          "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		ModelSizeBytes:         2019377696,
		ModelFamilyHint:        "llama",
		QuantizationHint:       "Q4_K_M",
		Backend:                llamacpp.BackendName,
		OpenAICompatible:       true,
		SupportsTextGeneration: true,
		SupportsStreaming:      true,
		SupportsKVAccess:       false,
		SupportsTensorHooks:    false,
	})
}

func ptr[T any](value T) *T {
	return &value
}
