package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/Ryvion/node-agent/internal/hub"
	"github.com/Ryvion/node-agent/internal/hw"
	"github.com/Ryvion/node-agent/internal/runtimeexec"
	v7dashboardinference "github.com/Ryvion/node-agent/internal/v7/dashboardinference"
	v7inferencebench "github.com/Ryvion/node-agent/internal/v7/inferencebench"
	v7kvprobe "github.com/Ryvion/node-agent/internal/v7/kvprobe"
	v7llamacpp "github.com/Ryvion/node-agent/internal/v7/llamacpp"
	v7memorybench "github.com/Ryvion/node-agent/internal/v7/memorybench"
	v7modelbench "github.com/Ryvion/node-agent/internal/v7/modelbench"
)

func TestAllowLocalOrigin(t *testing.T) {
	t.Parallel()

	allowed := []string{
		"http://localhost:1420",
		"http://127.0.0.1:45890",
		"tauri://localhost",
		"https://tauri.localhost",
	}
	blocked := []string{
		"",
		"https://ryvion.ai",
		"http://example.com",
		"file://local",
	}

	for _, origin := range allowed {
		if !allowLocalOrigin(origin) {
			t.Fatalf("expected origin %q to be allowed", origin)
		}
	}
	for _, origin := range blocked {
		if allowLocalOrigin(origin) {
			t.Fatalf("expected origin %q to be blocked", origin)
		}
	}
}

func TestOperatorAPIPort(t *testing.T) {
	t.Parallel()

	old, hadOld := os.LookupEnv("RYV_UI_PORT")
	defer func() {
		if hadOld {
			_ = os.Setenv("RYV_UI_PORT", old)
			return
		}
		_ = os.Unsetenv("RYV_UI_PORT")
	}()

	_ = os.Unsetenv("RYV_UI_PORT")
	if got := operatorAPIPort(""); got != defaultOperatorAPIPort {
		t.Fatalf("expected default port %q, got %q", defaultOperatorAPIPort, got)
	}
	if got := operatorAPIPort("5000"); got != "5000" {
		t.Fatalf("expected flag port 5000, got %q", got)
	}

	_ = os.Setenv("RYV_UI_PORT", "61234")
	if got := operatorAPIPort("5000"); got != "61234" {
		t.Fatalf("expected env override port 61234, got %q", got)
	}
}

func TestLogRingWriteTail(t *testing.T) {
	t.Parallel()

	ring := newLogRing(3)
	if _, err := ring.Write([]byte("line one\nline two")); err != nil {
		t.Fatalf("write 1: %v", err)
	}
	if _, err := ring.Write([]byte("\nline three\nline four\n")); err != nil {
		t.Fatalf("write 2: %v", err)
	}

	got := ring.tail(10)
	want := []string{"line two", "line three", "line four"}
	if len(got) != len(want) {
		t.Fatalf("expected %d lines, got %d: %#v", len(want), len(got), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("line %d: expected %q, got %q", i, want[i], got[i])
		}
	}
}

func TestOperatorAPIModelBenchmarkEndpointReturnsUnavailableJSON(t *testing.T) {
	t.Parallel()

	port := freeOperatorAPITestPort(t)
	state := &operatorRuntime{
		version:      "test",
		hubURL:       "https://api.ryvion.ai",
		deviceType:   "gpu",
		publicKeyHex: "abc123",
		caps: hw.CapSet{
			CPUCores:  8,
			RAMBytes:  16 << 30,
			GPUModel:  "test-gpu",
			VRAMBytes: 8 << 30,
		},
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	startOperatorAPIServer(ctx, state, port)

	payload := []byte(`{"model_id":"ryvion-llama-3.2-3b","max_tokens":16,"timeout_ms":60000}`)
	respBody := postOperatorAPITestJSON(t, port, "/api/v1/operator/v7/model-benchmark/run", payload)
	if strings.Contains(string(respBody), "Generate one short readiness") {
		t.Fatalf("operator benchmark response leaked raw prompt: %s", respBody)
	}

	var result v7modelbench.ModelBenchmarkResult
	if err := json.Unmarshal(respBody, &result); err != nil {
		t.Fatalf("decode response: %v\nbody: %s", err, respBody)
	}
	if result.ProofStatus != v7modelbench.ModelBenchmarkProofStatusUnavailable {
		t.Fatalf("proof_status = %q, want unavailable; body=%s", result.ProofStatus, respBody)
	}
	if result.Metrics.ErrorCode == "" {
		t.Fatalf("error_code empty; body=%s", respBody)
	}
	if err := v7modelbench.ValidateModelBenchmarkResult(result); err != nil {
		t.Fatalf("ValidateModelBenchmarkResult() error = %v", err)
	}
}

func TestOperatorAPIStatusEndpointIncludesWorkLoop(t *testing.T) {
	port := freeOperatorAPITestPort(t)
	state := &operatorRuntime{
		version:      "test",
		hubURL:       "https://api.ryvion.ai",
		deviceType:   "gpu",
		publicKeyHex: "abc123",
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	startOperatorAPIServer(ctx, state, port)

	respBody := getOperatorAPITestJSON(t, port, "/api/v1/operator/status")
	if !strings.Contains(string(respBody), `"work_loop"`) {
		t.Fatalf("operator status response missing work_loop: %s", respBody)
	}

	var status operatorStatusResponse
	if err := json.Unmarshal(respBody, &status); err != nil {
		t.Fatalf("decode status response: %v\nbody: %s", err, respBody)
	}
	if _, err := json.Marshal(status.WorkLoop); err != nil {
		t.Fatalf("json.Marshal(status.WorkLoop) error = %v", err)
	}
}

func TestOperatorAPIStatusEndpointIncludesHardwareCapacity(t *testing.T) {
	t.Setenv("RYV_MODEL_CACHE_DIR", t.TempDir())
	port := freeOperatorAPITestPort(t)
	state := &operatorRuntime{
		version:      "test",
		hubURL:       "https://api.ryvion.ai",
		deviceType:   "gpu",
		publicKeyHex: "abc123",
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	startOperatorAPIServer(ctx, state, port)

	respBody := getOperatorAPITestJSON(t, port, "/api/v1/operator/status")
	var status operatorStatusResponse
	if err := json.Unmarshal(respBody, &status); err != nil {
		t.Fatalf("decode status response: %v\nbody: %s", err, respBody)
	}
	if status.HardwareCapacity.OS != runtime.GOOS || status.HardwareCapacity.Arch != runtime.GOARCH {
		t.Fatalf("hardware_capacity identity = %+v, want %s/%s", status.HardwareCapacity, runtime.GOOS, runtime.GOARCH)
	}
	if status.NodeID != "abc123" || status.AgentVersion != "test" || status.OS != runtime.GOOS || status.Arch != runtime.GOARCH {
		t.Fatalf("top-level identity = node_id:%q agent_version:%q os:%q arch:%q", status.NodeID, status.AgentVersion, status.OS, status.Arch)
	}
	if status.HardwareCapacity.GPUVendor == "" ||
		status.HardwareCapacity.GPUName == "" ||
		status.HardwareCapacity.PowerProfile == "" ||
		status.HardwareCapacity.ThermalRisk == "" {
		t.Fatalf("hardware_capacity missing safe fields: %+v", status.HardwareCapacity)
	}
	text := strings.ToLower(string(respBody))
	for _, want := range []string{
		`"hardware_capacity"`,
		`"node_id"`,
		`"agent_version"`,
		`"cpu_logical_cores"`,
		`"cpu_name"`,
		`"system_ram_bytes"`,
		`"available_ram_bytes"`,
		`"gpu_detected"`,
		`"gpu_vendor"`,
		`"gpu_name"`,
		`"gpu_vram_bytes"`,
		`"unified_memory"`,
		`"metal_available"`,
		`"cuda_available"`,
		`"vulkan_available"`,
		`"directml_available"`,
		`"acceleration_hints"`,
		`"disk_free_bytes_model_cache"`,
		`"power_profile"`,
		`"thermal_risk"`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("operator status missing %s: %s", want, respBody)
		}
	}
	for _, forbidden := range []string{"raw_prompt", "prompt_text", "model_output", "output_text", "generated_text", "key_data", "value_data", "query_vector", "tensor_bytes", "raw_tensor", "secret"} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("status JSON contains forbidden marker %q: %s", forbidden, respBody)
		}
	}
}

func TestOperatorAPIStatusEndpointIncludesTensorAccessCapability(t *testing.T) {
	port := freeOperatorAPITestPort(t)
	state := &operatorRuntime{
		version:      "test",
		hubURL:       "https://api.ryvion.ai",
		deviceType:   "gpu",
		publicKeyHex: "abc123",
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	startOperatorAPIServer(ctx, state, port)

	respBody := getOperatorAPITestJSON(t, port, "/api/v1/operator/status")
	var status operatorStatusResponse
	if err := json.Unmarshal(respBody, &status); err != nil {
		t.Fatalf("decode status response: %v\nbody: %s", err, respBody)
	}
	if status.TensorAccess.RuntimeKind != v7kvprobe.RuntimeKindNative {
		t.Fatalf("top-level tensor_access = %+v, want native runtime kind", status.TensorAccess)
	}
	if status.TensorAccess.Reason == "" {
		t.Fatalf("top-level tensor_access reason is empty: %+v", status.TensorAccess)
	}
	if !status.TensorAccess.TensorPlaneDemoSupported {
		t.Fatalf("top-level tensor_access tensorplane_demo_supported = false: %+v", status.TensorAccess)
	}
	if status.TensorAccess.KVAccessSupported {
		t.Fatalf("top-level tensor_access kv_access_supported = true by default: %+v", status.TensorAccess)
	}
	if status.Runtime.TensorAccess.RuntimeKind != v7kvprobe.RuntimeKindNative {
		t.Fatalf("runtime tensor_access = %+v, want native runtime kind", status.Runtime.TensorAccess)
	}
	if status.Runtime.TensorAccess.KVAccessSupported ||
		status.Runtime.TensorAccess.KVSnapshotSupported ||
		status.Runtime.TensorAccess.HiddenStateAccessSupported ||
		status.Runtime.TensorAccess.LogitsAccessSupported ||
		status.Runtime.TensorAccess.AttentionHookSupported {
		t.Fatalf("operator status should not advertise tensor hooks yet: %+v", status.Runtime.TensorAccess)
	}
	text := string(respBody)
	for _, want := range []string{
		`"tensor_access"`,
		`"kv_access_supported"`,
		`"kv_snapshot_supported"`,
		`"hidden_state_access_supported"`,
		`"logits_access_supported"`,
		`"attention_hook_supported"`,
		`"tensorplane_demo_supported"`,
		`"runtime_kind"`,
		`"reason"`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("status JSON missing %s: %s", want, text)
		}
	}
	for _, forbidden := range []string{"raw_prompt", "prompt_text", "model_output", "output_text", "generated_text", "key_data", "value_data", "query_vector", "tensor_bytes"} {
		if strings.Contains(strings.ToLower(text), forbidden) {
			t.Fatalf("status JSON contains forbidden text marker %q: %s", forbidden, text)
		}
	}
}

func TestOperatorAPIStatusEndpointIncludesModelPolicyAndCache(t *testing.T) {
	cacheDir := t.TempDir()
	modelPath := filepath.Join(cacheDir, "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
	if err := os.WriteFile(modelPath, []byte("gguf"), 0o644); err != nil {
		t.Fatalf("write model fixture: %v", err)
	}
	t.Setenv("RYV_MODEL_AUTO_DOWNLOAD", "1")
	t.Setenv("RYV_MODEL_MAX_SINGLE_GB", "4")
	t.Setenv("RYV_MODEL_MAX_CACHE_GB", "12")
	t.Setenv("RYV_MODEL_CACHE_DIR", cacheDir)
	t.Setenv("RYV_MODEL_ALLOWED_FAMILIES", "llama,qwen")
	t.Setenv("RYV_MODEL_ALLOWED_FORMATS", "gguf")
	t.Setenv("RYV_MODEL_KEEP_WARM_IDS", "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
	t.Setenv("RYV_MODEL_EVICTION_POLICY", "lru")
	t.Setenv("RYV_MODEL_ALLOW_LICENSE_RESTRICTED", "0")
	t.Setenv("RYV_MODEL_RUNTIME_MAX_SINGLE_GB", "4")
	t.Setenv("RYV_MODEL_RUNTIME_MAX_PARAMS_B", "8")
	t.Setenv("RYV_MODEL_DENY_IDS", "phi-4-Q4_K_M.gguf")
	t.Setenv("RYV_MODEL_ALLOW_IDS", "")
	t.Setenv("RYV_MODEL_RUNTIME_ALLOW_LARGE", "0")
	t.Setenv("RYV_MODEL_REQUIRE_EXPLICIT_ALLOW_LARGE", "1")

	port := freeOperatorAPITestPort(t)
	state := &operatorRuntime{
		version:      "test",
		hubURL:       "https://api.ryvion.ai",
		deviceType:   "gpu",
		publicKeyHex: "abc123",
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	startOperatorAPIServer(ctx, state, port)

	respBody := getOperatorAPITestJSON(t, port, "/api/v1/operator/status")
	var status operatorStatusResponse
	if err := json.Unmarshal(respBody, &status); err != nil {
		t.Fatalf("decode status response: %v\nbody: %s", err, respBody)
	}
	if !status.ModelPolicy.AutoDownload ||
		status.ModelPolicy.MaxSingleModelBytes != 4*1024*1024*1024 ||
		status.ModelPolicy.MaxCacheBytes != 12*1024*1024*1024 ||
		status.ModelPolicy.CacheDir != cacheDir ||
		status.ModelPolicy.AllowLicenseRestricted {
		t.Fatalf("model_policy = %+v", status.ModelPolicy)
	}
	if len(status.ModelPolicy.KeepWarmModelIDs) != 1 || status.ModelPolicy.KeepWarmModelIDs[0] != "Llama-3.2-3B-Instruct-Q4_K_M.gguf" {
		t.Fatalf("keep_warm_model_ids = %+v", status.ModelPolicy.KeepWarmModelIDs)
	}
	if !status.ModelPolicy.RuntimePolicy.AllowRuntimeExecution ||
		status.ModelPolicy.RuntimePolicy.MaxRuntimeModelBytes != 4*1024*1024*1024 ||
		status.ModelPolicy.RuntimePolicy.MaxRuntimeParameterCountBillions != 8 ||
		!status.ModelPolicy.RuntimePolicy.AllowCPUOffload ||
		status.ModelPolicy.RuntimePolicy.AllowLargeModels ||
		!status.ModelPolicy.RuntimePolicy.RequireExplicitAllowForLargeModels {
		t.Fatalf("runtime_policy = %+v", status.ModelPolicy.RuntimePolicy)
	}
	if len(status.ModelPolicy.RuntimePolicy.DenyModelIDs) != 1 || status.ModelPolicy.RuntimePolicy.DenyModelIDs[0] != "phi-4-Q4_K_M.gguf" {
		t.Fatalf("deny_model_ids = %+v", status.ModelPolicy.RuntimePolicy.DenyModelIDs)
	}
	if got, want := strings.Join(status.ModelPolicy.RuntimePolicy.AllowFamilies, ","), "llama"; got != want {
		t.Fatalf("runtime allow_families = %q, want %q", got, want)
	}
	if status.ModelCache.CacheDir != cacheDir || len(status.ModelCache.Models) != 1 {
		t.Fatalf("model_cache = %+v, want one model in configured cache dir", status.ModelCache)
	}
	model := status.ModelCache.Models[0]
	if model.ModelID != "Llama-3.2-3B-Instruct-Q4_K_M.gguf" ||
		model.FamilyHint != "llama" ||
		model.QuantizationHint != "Q4_K_M" ||
		model.Format != "gguf" ||
		!model.Installed ||
		model.HashVerified {
		t.Fatalf("model cache row = %+v", model)
	}
	text := strings.ToLower(string(respBody))
	for _, want := range []string{`"model_policy"`, `"runtime_policy"`, `"max_runtime_model_bytes"`, `"deny_model_ids"`, `"model_cache"`, `"auto_download"`, `"max_cache_bytes"`, `"keep_warm_model_ids"`, `"hash_verified"`} {
		if !strings.Contains(text, want) {
			t.Fatalf("operator status missing %s: %s", want, respBody)
		}
	}
	for _, forbidden := range []string{"raw_prompt", "prompt_text", "model_output", "output_text", "generated_text", "key_data", "value_data", "query_vector", "tensor_bytes", "raw_tensor", "secret"} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("operator status contains forbidden marker %q: %s", forbidden, respBody)
		}
	}
}

func TestOperatorStatusIncludesCapabilityProfile(t *testing.T) {
	cacheDir := t.TempDir()
	modelPath := filepath.Join(cacheDir, "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
	if err := os.WriteFile(modelPath, []byte("gguf"), 0o644); err != nil {
		t.Fatalf("write model fixture: %v", err)
	}
	t.Setenv("RYV_MODEL_CACHE_DIR", cacheDir)

	state := &operatorRuntime{
		version:      "test",
		hubURL:       "https://api.ryvion.ai",
		deviceType:   "gpu",
		publicKeyHex: "abc123",
		caps: hw.CapSet{
			CPUCores:  8,
			RAMBytes:  32 << 30,
			GPUModel:  "Apple M4 Pro",
			VRAMBytes: 0,
		},
	}

	status := state.statusSnapshot(defaultOperatorAPIPort)
	if status.CapabilityProfile.SchemaVersion == "" {
		t.Fatalf("capability_profile missing: %+v", status.CapabilityProfile)
	}
	if status.CapabilityProfile.Hardware.OS == "" ||
		status.CapabilityProfile.Policy.MaxRuntimeModelBytes == 0 ||
		status.CapabilityProfile.BackendRuntime.Backend == "" {
		t.Fatalf("capability_profile compact summaries missing: %+v", status.CapabilityProfile)
	}
	if status.CapabilityProfile.SpeculativeDecoding == nil || status.CapabilityProfile.SpeculativeDecoding.Methods == nil {
		t.Fatalf("capability_profile.speculative_decoding missing method list: %+v", status.CapabilityProfile.SpeculativeDecoding)
	}
	raw, err := json.Marshal(status)
	if err != nil {
		t.Fatalf("json.Marshal(status) error = %v", err)
	}
	text := strings.ToLower(string(raw))
	for _, want := range []string{`"capability_profile"`, `"speculative_decoding"`, `"speculative_profiles"`, `"v7_dashboard_inference"`, `"text_output"`, `"streaming"`, `"hash_metrics_receipts"`, `"backend_text_generation"`, `"backend_warm"`, `"models"`} {
		if !strings.Contains(text, want) {
			t.Fatalf("status JSON missing %s: %s", want, raw)
		}
	}
	for _, forbidden := range []string{"raw_prompt", "prompt_text", "model_output", "output_text", "generated_text", "key_data", "value_data", "query_vector", "tensor_bytes", "raw_tensor", "secret"} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("status JSON contains forbidden marker %q: %s", forbidden, raw)
		}
	}
}

func TestOperatorAPIStatusEndpointIncludesLlamaCppSidecar(t *testing.T) {
	t.Setenv(v7llamacpp.EnvKeepWarm, "0")

	port := freeOperatorAPITestPort(t)
	state := &operatorRuntime{
		version:      "test",
		hubURL:       "https://api.ryvion.ai",
		deviceType:   "gpu",
		publicKeyHex: "abc123",
		llamaCppSidecar: v7llamacpp.NewManager(v7llamacpp.LlamaCppSidecarConfig{
			Enabled: false,
			Host:    v7llamacpp.DefaultHost,
			Port:    v7llamacpp.DefaultPort,
		}),
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	startOperatorAPIServer(ctx, state, port)

	respBody := getOperatorAPITestJSON(t, port, "/api/v1/operator/status")
	var status operatorStatusResponse
	if err := json.Unmarshal(respBody, &status); err != nil {
		t.Fatalf("decode status response: %v\nbody: %s", err, respBody)
	}
	if status.LlamaCPPSidecar.Enabled || status.LlamaCPPSidecar.Running || status.LlamaCPPSidecar.Healthy {
		t.Fatalf("llama_cpp_sidecar = %+v, want disabled stopped", status.LlamaCPPSidecar)
	}
	if status.LlamaCPPSidecar.KeepWarmEnabled ||
		status.LlamaCPPSidecar.RestartCount != 0 ||
		status.LlamaCPPSidecar.LastKeepWarmError != "" ||
		status.LlamaCPPSidecar.RestartBackoffSeconds != int(v7llamacpp.DefaultRestartBackoff/time.Second) {
		t.Fatalf("llama_cpp_sidecar keepwarm fields = %+v, want disabled defaults", status.LlamaCPPSidecar)
	}
	if status.BackendRuntimes.LlamaCPP.Running ||
		status.BackendRuntimes.LlamaCPP.Healthy ||
		status.BackendRuntimes.LlamaCPP.Loaded ||
		status.BackendRuntimes.LlamaCPP.Warm {
		t.Fatalf("backend_runtimes.llama_cpp = %+v, want no active loaded sidecar", status.BackendRuntimes.LlamaCPP)
	}
	if status.BackendRuntimes.LlamaCPP.Backend != v7llamacpp.BackendName || status.BackendRuntimes.LlamaCPP.Health == "" {
		t.Fatalf("backend_runtimes.llama_cpp missing backend health metadata: %+v", status.BackendRuntimes.LlamaCPP)
	}
	if status.LlamaCPPSidecar.Backend != v7llamacpp.BackendName ||
		!status.LlamaCPPSidecar.OpenAICompatible ||
		!status.LlamaCPPSidecar.SupportsTextGeneration ||
		!status.LlamaCPPSidecar.SupportsStreaming ||
		status.LlamaCPPSidecar.SupportsKVAccess ||
		status.LlamaCPPSidecar.SupportsTensorHooks {
		t.Fatalf("llama_cpp_sidecar capability flags = %+v", status.LlamaCPPSidecar)
	}
	text := strings.ToLower(string(respBody))
	if !strings.Contains(text, `"llama_cpp_sidecar"`) ||
		!strings.Contains(text, `"backend_runtimes"`) ||
		!strings.Contains(text, `"keep_warm_enabled"`) ||
		!strings.Contains(text, `"restart_count"`) ||
		!strings.Contains(text, `"last_keepwarm_error"`) ||
		!strings.Contains(text, `"restart_backoff_seconds"`) {
		t.Fatalf("status JSON missing llama.cpp runtime fields: %s", respBody)
	}
	for _, forbidden := range []string{"raw_prompt", "prompt_text", "model_output", "output_text", "generated_text", "key_data", "value_data", "query_vector", "tensor_bytes", "raw_tensor", "secret"} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("status JSON contains forbidden marker %q: %s", forbidden, respBody)
		}
	}
}

func TestOperatorAPIStatusEndpointIncludesActiveBackendRuntime(t *testing.T) {
	llamaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health", "/v1/models":
			writeJSON(w, http.StatusOK, map[string]any{"ok": true})
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer llamaServer.Close()

	port := freeOperatorAPITestPort(t)
	state := &operatorRuntime{
		version:         "test",
		hubURL:          "https://api.ryvion.ai",
		deviceType:      "gpu",
		publicKeyHex:    "abc123",
		llamaCppSidecar: testLlamaCppManagerForServer(t, llamaServer.URL),
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	startOperatorAPIServer(ctx, state, port)

	respBody := getOperatorAPITestJSON(t, port, "/api/v1/operator/status")
	var status operatorStatusResponse
	if err := json.Unmarshal(respBody, &status); err != nil {
		t.Fatalf("decode status response: %v\nbody: %s", err, respBody)
	}
	runtime := status.BackendRuntimes.LlamaCPP
	if !runtime.Enabled || !runtime.Available || !runtime.Running || !runtime.Healthy || !runtime.Loaded || !runtime.Warm {
		t.Fatalf("backend_runtimes.llama_cpp = %+v, want active loaded warm sidecar", runtime)
	}
	if runtime.ModelID != "tinyllama.Q4_K_M.gguf" ||
		runtime.ModelFilename != "tinyllama.Q4_K_M.gguf" ||
		!strings.HasSuffix(runtime.ModelPath, "tinyllama.Q4_K_M.gguf") {
		t.Fatalf("backend runtime model metadata = %+v", runtime)
	}
	if runtime.SupportsKVAccess || runtime.SupportsTensorHooks {
		t.Fatalf("backend runtime should not advertise KV/tensor hooks: %+v", runtime)
	}
}

func TestOperatorStatusAndHeartbeatPreviewUseSameBackendRuntimeBuilder(t *testing.T) {
	state := &operatorRuntime{
		version:      "test",
		hubURL:       "https://api.ryvion.ai",
		deviceType:   "gpu",
		publicKeyHex: "abc123",
		caps: hw.CapSet{
			CPUCores: 4,
			RAMBytes: 8 << 30,
		},
		llamaCppSidecar: v7llamacpp.NewManager(v7llamacpp.LlamaCppSidecarConfig{
			Enabled: false,
			Host:    v7llamacpp.DefaultHost,
			Port:    v7llamacpp.DefaultPort,
		}),
	}

	status := state.statusSnapshot(defaultOperatorAPIPort)
	preview, err := state.v7HeartbeatPreview()
	if err != nil {
		t.Fatalf("v7HeartbeatPreview() error = %v", err)
	}
	if !reflect.DeepEqual(status.BackendRuntimes, preview.HeartbeatPreview.V7.BackendRuntimes) {
		t.Fatalf("operator backend_runtimes = %+v, heartbeat backend_runtimes = %+v", status.BackendRuntimes, preview.HeartbeatPreview.V7.BackendRuntimes)
	}
	if len(preview.HeartbeatPreview.V7.SpeculativeProfiles) != 0 {
		t.Fatalf("heartbeat speculative_profiles = %+v, want omitted from heartbeat preview", preview.HeartbeatPreview.V7.SpeculativeProfiles)
	}
}

func TestOperatorAPILlamaCppEndpointsDisabledSafe(t *testing.T) {
	port := freeOperatorAPITestPort(t)
	state := &operatorRuntime{
		version:      "test",
		hubURL:       "https://api.ryvion.ai",
		deviceType:   "gpu",
		publicKeyHex: "abc123",
		llamaCppSidecar: v7llamacpp.NewManager(v7llamacpp.LlamaCppSidecarConfig{
			Enabled: false,
			Host:    v7llamacpp.DefaultHost,
			Port:    v7llamacpp.DefaultPort,
		}),
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	startOperatorAPIServer(ctx, state, port)

	for _, path := range []string{
		"/api/v1/operator/llamacpp/status",
		"/api/v1/operator/llamacpp/start",
		"/api/v1/operator/llamacpp/stop",
		"/api/v1/operator/llamacpp/restart",
	} {
		var body []byte
		if strings.Contains(path, "/status") {
			body = getOperatorAPITestJSON(t, port, path)
		} else {
			body = postOperatorAPITestJSON(t, port, path, nil)
		}
		var status v7llamacpp.LlamaCppSidecarStatus
		if err := json.Unmarshal(body, &status); err != nil {
			t.Fatalf("decode %s response: %v\nbody: %s", path, err, body)
		}
		if status.Enabled || status.Running || status.Healthy {
			t.Fatalf("%s status = %+v, want disabled stopped", path, status)
		}
	}
}

func TestOperatorAPILlamaCppBenchmarkEndpointRecordsSafeStatus(t *testing.T) {
	t.Setenv(v7llamacpp.EnvBenchmark, "1")
	llamaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			writeJSON(w, http.StatusOK, map[string]any{"ok": true})
		case "/v1/chat/completions":
			var req struct {
				Stream bool `json:"stream"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatalf("decode llama.cpp request: %v", err)
			}
			if !req.Stream {
				t.Fatalf("stream = false, want true")
			}
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"content\":\"safe generated output\"}}],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"total_tokens\":10}}\n")
			_, _ = io.WriteString(w, "data: [DONE]\n")
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer llamaServer.Close()

	port := freeOperatorAPITestPort(t)
	state := &operatorRuntime{
		version:           "test",
		hubURL:            "https://api.ryvion.ai",
		deviceType:        "gpu",
		publicKeyHex:      "abc123",
		llamaCppSidecar:   testLlamaCppManagerForServer(t, llamaServer.URL),
		llamaCppBenchmark: v7llamacpp.NewBenchmarkLocalStatus(),
		v7MemoryBenchmark: v7memorybench.NewLocalStatus(),
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	startOperatorAPIServer(ctx, state, port)

	respBody := postOperatorAPITestJSON(t, port, "/api/v1/operator/llamacpp/benchmark", []byte(`{"max_tokens":4,"warmup_runs":0,"measured_runs":1,"timeout_ms":60000}`))
	var snapshot v7llamacpp.BenchmarkStatusSnapshot
	if err := json.Unmarshal(respBody, &snapshot); err != nil {
		t.Fatalf("decode benchmark response: %v\nbody: %s", err, respBody)
	}
	if snapshot.Status != v7llamacpp.BenchmarkStatusCompleted {
		t.Fatalf("benchmark status = %+v, want completed", snapshot)
	}
	if snapshot.Metrics.PromptHash == "" || snapshot.Metrics.OutputHash == "" || snapshot.Metrics.OutputBytes == 0 {
		t.Fatalf("benchmark hashes/bytes missing: %+v", snapshot.Metrics)
	}
	statusBody := getOperatorAPITestJSON(t, port, "/api/v1/operator/status")
	var status operatorStatusResponse
	if err := json.Unmarshal(statusBody, &status); err != nil {
		t.Fatalf("decode status: %v\nbody: %s", err, statusBody)
	}
	if status.LlamaCPPBenchmark.Status != v7llamacpp.BenchmarkStatusCompleted {
		t.Fatalf("status llama_cpp_benchmark = %+v, want completed", status.LlamaCPPBenchmark)
	}
	lower := strings.ToLower(string(respBody) + string(statusBody))
	for _, forbidden := range []string{"write one short sentence about distributed computing", "safe generated output", "raw_prompt", "prompt_text", "generated_text", "output_text", "model_output"} {
		if strings.Contains(lower, forbidden) {
			t.Fatalf("benchmark operator JSON leaked forbidden text %q: response=%s status=%s", forbidden, respBody, statusBody)
		}
	}
}

func TestOperatorAPIDebugV7HeartbeatPreviewEndpoint(t *testing.T) {
	dataDir := setupHeartbeatPreviewInventoryFixture(t)
	t.Setenv("RYV_DATA_DIR", dataDir)
	t.Setenv("RYV_MODEL_CACHE_DIR", filepath.Join(dataDir, "models"))
	t.Setenv("RYV_MODEL_DIR", "")
	t.Setenv("RYVION_MODEL_DIR", "")
	t.Setenv("RYV_LLAMA_CPP_DIR", "")
	t.Setenv("RYVION_LLAMA_CPP_DIR", "")
	t.Setenv("RYV_RUNTIME_DIR", "")
	t.Setenv("RYVION_RUNTIME_DIR", "")
	t.Setenv("RYV_LLAMA_CPP_PROBE_MODEL", "")
	emptyPath := filepath.Join(dataDir, "empty-path")
	if err := os.MkdirAll(emptyPath, 0o755); err != nil {
		t.Fatalf("create empty PATH dir: %v", err)
	}
	t.Setenv("PATH", emptyPath)

	port := freeOperatorAPITestPort(t)
	state := &operatorRuntime{
		version:         "test",
		hubURL:          "https://api.ryvion.ai",
		deviceType:      "gpu",
		declaredCountry: "CA",
		publicKeyHex:    "abc123",
		caps: hw.CapSet{
			CPUCores:  8,
			RAMBytes:  16 << 30,
			GPUModel:  "test-gpu",
			VRAMBytes: 8 << 30,
		},
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	startOperatorAPIServer(ctx, state, port)

	respBody := getOperatorAPITestJSON(t, port, "/api/v1/operator/debug/v7-heartbeat-preview")
	var preview v7HeartbeatPreviewResponse
	if err := json.Unmarshal(respBody, &preview); err != nil {
		t.Fatalf("decode heartbeat preview response: %v\nbody: %s", err, respBody)
	}
	if !preview.OK {
		t.Fatalf("ok = false; body=%s", respBody)
	}
	if preview.NodeID != "abc123" {
		t.Fatalf("node_id = %q, want abc123", preview.NodeID)
	}
	if preview.HeartbeatPreview.V7.SchemaVersion == "" {
		t.Fatalf("heartbeat_preview.v7 missing schema_version: %+v", preview.HeartbeatPreview.V7)
	}
	if !preview.FieldPresence.RuntimeInventoryPresent {
		t.Fatalf("runtime_inventory_present = false; body=%s", respBody)
	}
	if !preview.FieldPresence.HardwareCapacityPresent {
		t.Fatalf("hardware_capacity_present = false; body=%s", respBody)
	}
	if !preview.FieldPresence.BackendCandidatesPresent || preview.FieldPresence.BackendCandidatesLen < 1 {
		t.Fatalf("backend_candidates presence = %+v; body=%s", preview.FieldPresence, respBody)
	}
	if !preview.FieldPresence.GGUFModelsPresent || preview.FieldPresence.GGUFModelsLen != 2 {
		t.Fatalf("gguf_models presence = %+v; body=%s", preview.FieldPresence, respBody)
	}
	if !preview.FieldPresence.ModelPolicyPresent || !preview.FieldPresence.ModelCachePresent || preview.FieldPresence.ModelCacheModelsLen != 2 {
		t.Fatalf("model policy/cache presence = %+v; body=%s", preview.FieldPresence, respBody)
	}
	if !preview.FieldPresence.BackendProbesPresent || !preview.FieldPresence.LlamaCPPProbePresent {
		t.Fatalf("backend probe presence = %+v; body=%s", preview.FieldPresence, respBody)
	}
	if !preview.FieldPresence.BackendRuntimesPresent || !preview.FieldPresence.LlamaCPPRuntimePresent {
		t.Fatalf("backend runtime presence = %+v; body=%s", preview.FieldPresence, respBody)
	}
	if preview.FieldPresence.SpeculativeProfilesPresent || preview.FieldPresence.SpeculativeProfilesLen != 0 {
		t.Fatalf("speculative profile presence = %+v, want omitted by default; body=%s", preview.FieldPresence, respBody)
	}
	if !preview.HeartbeatPreview.V7.BackendProbes.LlamaCPP.Available {
		t.Fatalf("llama_cpp probe = %+v, want available from local fixture", preview.HeartbeatPreview.V7.BackendProbes.LlamaCPP)
	}
	if preview.HeartbeatPreview.V7.BackendRuntimes.LlamaCPP.SupportsKVAccess ||
		preview.HeartbeatPreview.V7.BackendRuntimes.LlamaCPP.SupportsTensorHooks {
		t.Fatalf("llama_cpp runtime should not advertise KV/tensor hooks: %+v", preview.HeartbeatPreview.V7.BackendRuntimes.LlamaCPP)
	}
	if len(preview.HeartbeatPreview.V7.RuntimeInventory.BackendCandidates) == 0 {
		t.Fatalf("heartbeat_preview.v7.runtime_inventory.backend_candidates empty: %s", respBody)
	}
	if len(preview.HeartbeatPreview.V7.RuntimeInventory.GGUFModels) != 2 {
		t.Fatalf("heartbeat_preview.v7.runtime_inventory.gguf_models = %+v", preview.HeartbeatPreview.V7.RuntimeInventory.GGUFModels)
	}
	if preview.HeartbeatPreview.V7.ModelPolicy.CacheDir != filepath.Join(dataDir, "models") {
		t.Fatalf("heartbeat_preview.v7.model_policy = %+v", preview.HeartbeatPreview.V7.ModelPolicy)
	}
	if preview.HeartbeatPreview.V7.ModelPolicy.RuntimePolicy.MaxRuntimeModelBytes == 0 ||
		!preview.HeartbeatPreview.V7.ModelPolicy.RuntimePolicy.AllowRuntimeExecution {
		t.Fatalf("heartbeat_preview.v7.model_policy.runtime_policy = %+v", preview.HeartbeatPreview.V7.ModelPolicy.RuntimePolicy)
	}
	if len(preview.HeartbeatPreview.V7.ModelCache.Models) != 2 {
		t.Fatalf("heartbeat_preview.v7.model_cache = %+v", preview.HeartbeatPreview.V7.ModelCache)
	}
	if preview.HeartbeatPreview.V7.CapabilityProfile.SpeculativeDecoding != nil {
		t.Fatalf("heartbeat_preview.v7.capability_profile.speculative_decoding = %+v, want omitted by default", preview.HeartbeatPreview.V7.CapabilityProfile.SpeculativeDecoding)
	}
	rawV7, err := json.Marshal(preview.HeartbeatPreview.V7)
	if err != nil {
		t.Fatalf("json.Marshal(heartbeat_preview.v7) error = %v", err)
	}
	if strings.Contains(string(rawV7), `"speculative_decoding"`) || strings.Contains(string(rawV7), `"speculative_profiles"`) {
		t.Fatalf("heartbeat preview V7 payload should omit speculative fields by default: %s", rawV7)
	}

	text := strings.ToLower(string(respBody))
	for _, forbidden := range []string{
		"auth_token",
		"bind_token",
		"admin_key",
		"wallet",
		"raw_tensor",
		"tensor_bytes",
		"raw_prompt",
		"prompt_text",
		"output_text",
		"generated_text",
		"key_data",
		"value_data",
		"query_vector",
	} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("heartbeat preview contains forbidden marker %q: %s", forbidden, respBody)
		}
	}
}

func testLlamaCppManagerForServer(t *testing.T, serverURL string) *v7llamacpp.Manager {
	t.Helper()
	return testLlamaCppManagerForServerModel(t, serverURL, "")
}

func testLlamaCppManagerForServerModel(t *testing.T, serverURL string, modelPath string) *v7llamacpp.Manager {
	t.Helper()
	parsed, err := url.Parse(serverURL)
	if err != nil {
		t.Fatalf("parse test server URL: %v", err)
	}
	host, portRaw, err := net.SplitHostPort(parsed.Host)
	if err != nil {
		t.Fatalf("split test server host: %v", err)
	}
	port, err := strconv.Atoi(portRaw)
	if err != nil {
		t.Fatalf("parse test server port: %v", err)
	}
	dir := t.TempDir()
	serverPath := filepath.Join(dir, "llama-server")
	if err := os.WriteFile(serverPath, []byte("test llama-server"), 0o755); err != nil {
		t.Fatalf("write server fixture: %v", err)
	}
	if strings.TrimSpace(modelPath) == "" {
		modelPath = filepath.Join(dir, "tinyllama.Q4_K_M.gguf")
		if err := os.WriteFile(modelPath, []byte("test gguf"), 0o644); err != nil {
			t.Fatalf("write model fixture: %v", err)
		}
	}
	return v7llamacpp.NewManager(v7llamacpp.LlamaCppSidecarConfig{
		Enabled:    true,
		ServerPath: serverPath,
		ModelPath:  modelPath,
		Host:       host,
		Port:       port,
	})
}

func setupHeartbeatPreviewInventoryFixture(t *testing.T) string {
	t.Helper()
	dataDir := t.TempDir()
	binDir := filepath.Join(dataDir, "bin")
	modelDir := filepath.Join(dataDir, "models")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatalf("create bin dir: %v", err)
	}
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatalf("create model dir: %v", err)
	}
	for _, name := range []string{"llama-cli", "llama-server", "llama-bench"} {
		if err := os.WriteFile(filepath.Join(binDir, name), []byte("llama.cpp version test\n"), 0o755); err != nil {
			t.Fatalf("write %s fixture: %v", name, err)
		}
	}
	for _, name := range []string{"llama-3-q4_k_m.gguf", "qwen2.5-q5_k_m.gguf"} {
		if err := os.WriteFile(filepath.Join(modelDir, name), []byte("gguf"), 0o644); err != nil {
			t.Fatalf("write %s fixture: %v", name, err)
		}
	}
	return dataDir
}

func TestStatusTokenParsing(t *testing.T) {
	t.Parallel()

	msg := "docker-cli:present, docker-ready:1, disk_gb:512, native-inference-ready:1"
	if !statusToken(msg, "docker-ready:1") {
		t.Fatal("expected docker-ready token")
	}
	if statusToken(msg, "docker-ready:0") {
		t.Fatal("did not expect docker-ready:0 token")
	}
	if got := statusTokenUint(msg, "disk_gb:"); got != 512 {
		t.Fatalf("expected disk_gb 512, got %d", got)
	}
}

func freeOperatorAPITestPort(t *testing.T) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen on free port: %v", err)
	}
	defer ln.Close()
	addr, ok := ln.Addr().(*net.TCPAddr)
	if !ok {
		t.Fatalf("addr type = %T, want *net.TCPAddr", ln.Addr())
	}
	return strconv.Itoa(addr.Port)
}

func postOperatorAPITestJSON(t *testing.T, port, path string, payload []byte) []byte {
	t.Helper()
	client := &http.Client{Timeout: 2 * time.Second}
	url := "http://127.0.0.1:" + port + path
	var lastErr error
	for i := 0; i < 50; i++ {
		req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(payload))
		if err != nil {
			t.Fatalf("build request: %v", err)
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := client.Do(req)
		if err != nil {
			lastErr = err
			time.Sleep(20 * time.Millisecond)
			continue
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatalf("read response: %v", err)
		}
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, want 200; body=%s", resp.StatusCode, body)
		}
		return body
	}
	t.Fatalf("operator API did not become ready: %v", lastErr)
	return nil
}

func getOperatorAPITestJSON(t *testing.T, port, path string) []byte {
	t.Helper()
	client := &http.Client{Timeout: 2 * time.Second}
	url := "http://127.0.0.1:" + port + path
	var lastErr error
	for i := 0; i < 50; i++ {
		resp, err := client.Get(url)
		if err != nil {
			lastErr = err
			time.Sleep(20 * time.Millisecond)
			continue
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			t.Fatalf("read response: %v", err)
		}
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, want 200; body=%s", resp.StatusCode, body)
		}
		return body
	}
	t.Fatalf("operator API did not become ready: %v", lastErr)
	return nil
}

func TestSplitStatusTokens(t *testing.T) {
	t.Parallel()

	got := splitStatusTokens("docker-cli:present, docker-ready:1, , native-inference-ready:1 ")
	want := []string{"docker-cli:present", "docker-ready:1", "native-inference-ready:1"}
	if len(got) != len(want) {
		t.Fatalf("expected %d tokens, got %d: %#v", len(want), len(got), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token %d: expected %q, got %q", i, want[i], got[i])
		}
	}
}

func TestDeriveSovereignPosture(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name              string
		registered        bool
		declaredCountry   string
		verifiedCountry   string
		locationApproved  bool
		sovereignVerified bool
		trustReason       string
		runtimeReady      bool
		runtimeHealth     string
		nativeReady       bool
		wantReady         bool
		wantStatus        string
		wantDetailPart    string
	}{
		{
			name:           "missing country blocks review",
			registered:     true,
			wantStatus:     "country_missing",
			wantDetailPart: "infer it automatically",
		},
		{
			name:              "registration pending blocks review",
			verifiedCountry:   "CA",
			locationApproved:  true,
			sovereignVerified: true,
			declaredCountry:   "CA",
			runtimeReady:      true,
			wantStatus:        "registration_pending",
		},
		{
			name:              "runtime unavailable blocks review",
			registered:        true,
			verifiedCountry:   "CA",
			locationApproved:  true,
			sovereignVerified: true,
			wantStatus:        "runtime_unavailable",
		},
		{
			name:              "runtime warming surfaces warmup posture",
			registered:        true,
			verifiedCountry:   "CA",
			locationApproved:  true,
			sovereignVerified: true,
			runtimeHealth:     "warming",
			wantStatus:        "runtime_warming",
		},
		{
			name:              "local prerequisites satisfied",
			registered:        true,
			verifiedCountry:   "CA",
			locationApproved:  true,
			sovereignVerified: true,
			runtimeReady:      true,
			wantReady:         true,
			wantStatus:        "review_ready",
			wantDetailPart:    "Hub-verified country CA",
		},
		{
			name:              "native path also satisfies prerequisites",
			registered:        true,
			verifiedCountry:   "DE",
			locationApproved:  true,
			sovereignVerified: true,
			nativeReady:       true,
			wantReady:         true,
			wantStatus:        "review_ready",
		},
		{
			name:              "country mismatch is surfaced explicitly",
			registered:        true,
			declaredCountry:   "CA",
			verifiedCountry:   "US",
			locationApproved:  false,
			sovereignVerified: false,
			wantStatus:        "country_mismatch",
			wantDetailPart:    "Declared country CA does not match hub-verified country US",
		},
		{
			name:            "verified country without approval stays pending",
			registered:      true,
			verifiedCountry: "CA",
			trustReason:     "eligible for public workloads; sovereign approval has not been granted",
			wantStatus:      "trust_review_pending",
			wantDetailPart:  "public workloads",
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			gotReady, gotStatus, gotDetail := deriveSovereignPosture(tc.registered, tc.declaredCountry, tc.verifiedCountry, tc.locationApproved, tc.sovereignVerified, tc.trustReason, tc.runtimeReady, tc.runtimeHealth, tc.nativeReady)
			if gotReady != tc.wantReady {
				t.Fatalf("ready = %v, want %v", gotReady, tc.wantReady)
			}
			if gotStatus != tc.wantStatus {
				t.Fatalf("status = %q, want %q", gotStatus, tc.wantStatus)
			}
			if gotDetail == "" {
				t.Fatal("expected non-empty detail")
			}
			if tc.wantDetailPart != "" && !strings.Contains(gotDetail, tc.wantDetailPart) {
				t.Fatalf("detail = %q, want substring %q", gotDetail, tc.wantDetailPart)
			}
		})
	}
}

func TestDeriveRuntimePosture(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		runtimeReady  bool
		runtimeHealth string
		wantPosture   string
		wantWarming   bool
	}{
		{name: "ready wins", runtimeReady: true, runtimeHealth: "degraded", wantPosture: "ready"},
		{name: "warming posture", runtimeHealth: "warming", wantPosture: "warming", wantWarming: true},
		{name: "missing defaults to unavailable", wantPosture: "unavailable"},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			gotPosture, gotDetail, gotWarming := deriveRuntimePosture(tc.runtimeReady, tc.runtimeHealth)
			if gotPosture != tc.wantPosture {
				t.Fatalf("posture = %q, want %q", gotPosture, tc.wantPosture)
			}
			if gotWarming != tc.wantWarming {
				t.Fatalf("warming = %v, want %v", gotWarming, tc.wantWarming)
			}
			if gotDetail == "" {
				t.Fatal("expected non-empty detail")
			}
		})
	}
}

func TestOperatorStatusSnapshotMarksRuntimeWarmup(t *testing.T) {
	t.Parallel()

	state := &operatorRuntime{
		version:         "dev",
		hubURL:          "https://api.ryvion.ai",
		deviceType:      "desktop",
		declaredCountry: "CA",
		publicKeyHex:    "abc123",
		registered:      true,
		lastHealthReport: hub.HealthReport{
			Message: "runtime-ready:0,runtime-health:warming,runtime-backend:/opt/ryvion/runtime/backend/ryvion-oci,runtime-engine-kind:podman",
		},
	}

	status := state.statusSnapshot("45890")
	if !status.Runtime.RuntimeWarming {
		t.Fatal("expected runtime_warming to be true")
	}
	if status.Runtime.RuntimePosture != "warming" {
		t.Fatalf("runtime_posture = %q, want %q", status.Runtime.RuntimePosture, "warming")
	}
	if status.Runtime.RuntimeDetail == "" {
		t.Fatal("expected runtime detail to be populated")
	}
	if status.Runtime.SovereignStatus != "runtime_warming" {
		t.Fatalf("sovereign_status = %q, want %q", status.Runtime.SovereignStatus, "runtime_warming")
	}
}

func TestOperatorStatusSnapshotRefreshesRuntimeReport(t *testing.T) {
	t.Parallel()

	prevProbe := probeManagedRuntimeStatus
	probeManagedRuntimeStatus = func(_ context.Context, _ string, _ func(string) string, _ string) (runtimeexec.Status, bool) {
		return runtimeexec.Status{
			BinaryPath:   `C:\Program Files\Ryvion\runtime\ryvion-runtime.cmd`,
			BackendPath:  `C:\Program Files\Ryvion\runtime\backend\ryvion-oci.cmd`,
			EnginePath:   `C:\Program Files\RedHat\Podman\podman.exe`,
			EngineKind:   "podman",
			CLIInstalled: true,
			Ready:        true,
			GPUReady:     false,
			Health:       "ready",
		}, true
	}
	defer func() {
		probeManagedRuntimeStatus = prevProbe
	}()

	state := &operatorRuntime{
		version:      "dev",
		hubURL:       "https://api.ryvion.ai",
		deviceType:   "gpu",
		publicKeyHex: "abc123",
		caps: hw.CapSet{
			CPUCores:  16,
			RAMBytes:  32 << 30,
			GPUModel:  "RTX",
			VRAMBytes: 16 << 30,
		},
		runtimeMgr: newRuntimeManager("dev", runtimeContractMetadata{
			Channel:      "managed_oci_v1",
			Version:      "2026.04.15.21",
			Provider:     "oci_desktop_adapter",
			Mode:         "host_package",
			Source:       "ryvion_runtime_kit",
			Artifact:     "ryvion-runtime-kit-windows-amd64-2026.04.15.21.zip",
			Binary:       `C:\Program Files\Ryvion\runtime\ryvion-runtime.cmd`,
			Backend:      `C:\Program Files\Ryvion\runtime\backend\ryvion-oci.cmd`,
			Engine:       `C:\Program Files\RedHat\Podman\podman.exe`,
			EngineKind:   "podman",
			ManifestHash: "freshhash",
		}),
		lastHealthReport: hub.HealthReport{
			Message: "runtime-ready:0,runtime-health:warming,runtime-version:2026.04.15.20,runtime-artifact:ryvion-runtime-kit-windows-amd64-2026.04.15.20.zip",
		},
	}

	status := state.statusSnapshot("45890")
	if !status.Runtime.RuntimeReady {
		t.Fatal("expected live runtime probe to refresh runtime_ready")
	}
	if status.Runtime.RuntimePosture != "ready" {
		t.Fatalf("runtime_posture = %q, want %q", status.Runtime.RuntimePosture, "ready")
	}
	if status.Runtime.RuntimeVersion != "2026.04.15.21" {
		t.Fatalf("runtime_version = %q, want %q", status.Runtime.RuntimeVersion, "2026.04.15.21")
	}
	if status.Runtime.RuntimeArtifact != "ryvion-runtime-kit-windows-amd64-2026.04.15.21.zip" {
		t.Fatalf("runtime_artifact = %q", status.Runtime.RuntimeArtifact)
	}
}

func TestOperatorStatusSnapshotIncludesV7MemoryBenchmarkStatus(t *testing.T) {
	t.Parallel()

	benchStatus := v7memorybench.NewLocalStatus()
	benchStatus.RecordSeen("job-bench-1", "request-bench-1")
	benchStatus.RecordExecuted("job-bench-1")
	benchStatus.RecordReceiptSubmitted("job-bench-1")

	state := &operatorRuntime{
		version:           "dev",
		hubURL:            "https://api.ryvion.ai",
		deviceType:        "gpu",
		publicKeyHex:      "abc123",
		v7MemoryBenchmark: benchStatus,
	}
	work := &hub.WorkAssignment{JobID: "job-bench-1", Kind: "benchmark", SpecJSON: `{"task":"v7_memory_benchmark"}`}
	state.startJob(work)
	state.finishJob(work, &runnerResultSnapshot{
		ResultHashHex: "hash",
		MeteringUnits: 1,
		Metadata: map[string]any{
			v7memorybench.BenchmarkTask: map[string]any{
				"request_id":     "request-bench-1",
				"weighted_value": []float64{1, 2, 3},
				"proof_status":   "synthetic_measured",
			},
		},
	}, nil)

	status := state.statusSnapshot("45890")
	if status.V7MemoryBenchmark.LastSeenBenchmarkJobID != "job-bench-1" {
		t.Fatalf("v7 benchmark status = %+v", status.V7MemoryBenchmark)
	}
	if status.V7MemoryBenchmark.Counters.Seen != 1 || status.V7MemoryBenchmark.Counters.Executed != 1 || status.V7MemoryBenchmark.Counters.ReceiptSubmitted != 1 {
		t.Fatalf("unexpected v7 benchmark counters: %+v", status.V7MemoryBenchmark.Counters)
	}
	encoded, err := json.Marshal(status)
	if err != nil {
		t.Fatalf("json.Marshal(status) error = %v", err)
	}
	text := string(encoded)
	if !strings.Contains(text, `"v7_memory_benchmark"`) {
		t.Fatalf("status JSON missing v7_memory_benchmark: %s", text)
	}
	if strings.Contains(text, "weighted_value") {
		t.Fatalf("operator status JSON includes weighted_value: %s", text)
	}
}

func TestOperatorStatusSnapshotIncludesV7InferenceBenchmarkStatus(t *testing.T) {
	t.Parallel()

	benchStatus := v7inferencebench.NewLocalStatus()
	benchStatus.RecordSeen("job-infer-1")
	benchStatus.RecordExecuted("job-infer-1")
	benchStatus.RecordReceiptSubmitted("job-infer-1")

	state := &operatorRuntime{
		version:              "dev",
		hubURL:               "https://api.ryvion.ai",
		deviceType:           "gpu",
		publicKeyHex:         "abc123",
		v7InferenceBenchmark: benchStatus,
	}

	status := state.statusSnapshot("45890")
	if status.V7InferenceBenchmark.LastJobID != "job-infer-1" {
		t.Fatalf("v7 inference benchmark status = %+v", status.V7InferenceBenchmark)
	}
	if status.V7InferenceBenchmark.Counters.Seen != 1 || status.V7InferenceBenchmark.Counters.Executed != 1 || status.V7InferenceBenchmark.Counters.ReceiptSubmitted != 1 {
		t.Fatalf("unexpected v7 inference benchmark counters: %+v", status.V7InferenceBenchmark.Counters)
	}
	encoded, err := json.Marshal(status)
	if err != nil {
		t.Fatalf("json.Marshal(status) error = %v", err)
	}
	if !strings.Contains(string(encoded), `"v7_inference_benchmark"`) {
		t.Fatalf("status JSON missing v7_inference_benchmark: %s", encoded)
	}
}

func TestOperatorStatusSnapshotIncludesV7DashboardInferenceStatus(t *testing.T) {
	t.Parallel()

	dashboardStatus := v7dashboardinference.NewLocalStatus()
	dashboardStatus.RecordSeen("run-dashboard-1", "job-dashboard-1")
	dashboardStatus.RecordExecuted("run-dashboard-1", "job-dashboard-1")
	dashboardStatus.RecordReceiptSubmitted("run-dashboard-1", "job-dashboard-1")

	state := &operatorRuntime{
		version:              "dev",
		hubURL:               "https://api.ryvion.ai",
		deviceType:           "gpu",
		publicKeyHex:         "abc123",
		v7DashboardInference: dashboardStatus,
	}

	status := state.statusSnapshot("45890")
	if status.V7DashboardInference.LastRunID != "run-dashboard-1" || status.V7DashboardInference.LastJobID != "job-dashboard-1" {
		t.Fatalf("v7 dashboard inference status = %+v", status.V7DashboardInference)
	}
	if status.V7DashboardInference.LastError != "" {
		t.Fatalf("v7 dashboard inference last_error = %q, want empty", status.V7DashboardInference.LastError)
	}
	if status.V7DashboardInference.Counters.Seen != 1 || status.V7DashboardInference.Counters.Executed != 1 || status.V7DashboardInference.Counters.ReceiptSubmitted != 1 {
		t.Fatalf("unexpected v7 dashboard inference counters: %+v", status.V7DashboardInference.Counters)
	}
	encoded, err := json.Marshal(status)
	if err != nil {
		t.Fatalf("json.Marshal(status) error = %v", err)
	}
	text := string(encoded)
	if !strings.Contains(text, `"v7_dashboard_inference"`) {
		t.Fatalf("status JSON missing v7_dashboard_inference: %s", encoded)
	}
	for _, forbidden := range []string{"raw_prompt", "prompt_text", "output_text", "generated_text", "raw_output", "completion"} {
		if strings.Contains(strings.ToLower(text), forbidden) {
			t.Fatalf("status JSON leaked forbidden dashboard inference field %q: %s", forbidden, encoded)
		}
	}
}

func TestNormalizeDeclaredCountry(t *testing.T) {
	t.Parallel()

	if got := normalizeDeclaredCountry("ca"); got != "CA" {
		t.Fatalf("normalizeDeclaredCountry() = %q, want %q", got, "CA")
	}
	if got := normalizeDeclaredCountry(" c1 "); got != "" {
		t.Fatalf("normalizeDeclaredCountry() = %q, want empty", got)
	}
	if got := normalizeDeclaredCountry("CAN"); got != "" {
		t.Fatalf("normalizeDeclaredCountry() = %q, want empty", got)
	}
}

func TestUpdatePublicAIOptInPreservesOtherPreferences(t *testing.T) {
	prevResolver := operatorConfigPathResolver
	configPath := filepath.Join(t.TempDir(), "config.json")
	operatorConfigPathResolver = func() (string, error) {
		return configPath, nil
	}
	defer func() {
		operatorConfigPathResolver = prevResolver
	}()

	if err := saveOperatorPreferences(operatorPreferences{
		PublicAIOptIn:         false,
		DeclaredCountry:       "CA",
		RuntimeChannel:        "managed_oci_v1",
		RuntimeChannelVersion: "2026.04.14",
		RuntimeProvider:       "oci_linux_adapter",
		RuntimeMode:           "host_package",
		RuntimeSource:         "ryvion_runtime_kit",
		RuntimeArtifact:       "artifact.tar.gz",
		RuntimeBackendBinary:  "/opt/ryvion/runtime/backend/ryvion-oci",
		RuntimeManifestHash:   "abc123",
	}); err != nil {
		t.Fatalf("saveOperatorPreferences() error = %v", err)
	}

	state := &operatorRuntime{}
	if err := state.updatePublicAIOptIn(true); err != nil {
		t.Fatalf("updatePublicAIOptIn() error = %v", err)
	}

	got, err := loadOperatorPreferences()
	if err != nil {
		t.Fatalf("loadOperatorPreferences() error = %v", err)
	}
	if !got.PublicAIOptIn {
		t.Fatal("expected public AI opt-in to be updated")
	}
	if got.PublicAIOptOut {
		t.Fatal("expected public AI opt-out marker to be cleared when enabled")
	}
	if got.DeclaredCountry != "CA" || got.RuntimeChannel != "managed_oci_v1" || got.RuntimeArtifact != "artifact.tar.gz" {
		t.Fatalf("expected unrelated preferences to be preserved, got %+v", got)
	}
}

func TestUpdatePublicAIOptInWritesExplicitOptOutMarker(t *testing.T) {
	prevResolver := operatorConfigPathResolver
	configPath := filepath.Join(t.TempDir(), "config.json")
	operatorConfigPathResolver = func() (string, error) {
		return configPath, nil
	}
	defer func() {
		operatorConfigPathResolver = prevResolver
	}()

	state := &operatorRuntime{}
	if err := state.updatePublicAIOptIn(false); err != nil {
		t.Fatalf("updatePublicAIOptIn(false) error = %v", err)
	}
	got, err := loadOperatorPreferences()
	if err != nil {
		t.Fatalf("loadOperatorPreferences() error = %v", err)
	}
	if got.PublicAIOptIn {
		t.Fatal("expected public AI opt-in false")
	}
	if !got.PublicAIOptOut {
		t.Fatal("expected explicit public AI opt-out marker")
	}
}
