package dashboardinference

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Ryvion/ryvion-node/internal/v7/llamacpp"
	"github.com/Ryvion/ryvion-node/internal/v7/modelcache"
	"github.com/Ryvion/ryvion-node/internal/v7/modelpolicy"
)

type fakeSidecar struct {
	status              llamacpp.LlamaCppSidecarStatus
	statusQueue         []llamacpp.LlamaCppSidecarStatus
	starts              int
	restarts            int
	restartPath         string
	restartStatus       llamacpp.LlamaCppSidecarStatus
	fastCUDARestarts    int
	fastCUDAStatus      llamacpp.LlamaCppSidecarStatus
	fastRestartPath     string
	safeCUDAFallbacks   int
	partialGPUFallbacks int
	safeCUDAStatus      llamacpp.LlamaCppSidecarStatus
	partialGPUStatus    llamacpp.LlamaCppSidecarStatus
	fallbackRestartPath string
	startDelay          time.Duration
	stops               int
}

func (f *fakeSidecar) Start(context.Context) llamacpp.LlamaCppSidecarStatus {
	f.starts++
	if f.startDelay > 0 {
		time.Sleep(f.startDelay)
	}
	return f.status
}

func (f *fakeSidecar) Status(context.Context) llamacpp.LlamaCppSidecarStatus {
	if len(f.statusQueue) > 0 {
		next := f.statusQueue[0]
		f.statusQueue = f.statusQueue[1:]
		f.status = next
		return f.status
	}
	return f.status
}

func (f *fakeSidecar) Stop(context.Context) llamacpp.LlamaCppSidecarStatus {
	f.stops++
	return f.status
}

func (f *fakeSidecar) RestartWithModel(_ context.Context, modelPath string) llamacpp.LlamaCppSidecarStatus {
	f.restarts++
	f.restartPath = modelPath
	if f.restartStatus.ModelPath != "" {
		f.status = f.restartStatus
		return f.status
	}
	f.status.ModelPath = modelPath
	f.status.ModelFilename = filepath.Base(modelPath)
	f.status.ModelSizeBytes = int64(len(modelPath))
	f.status.Enabled = true
	f.status.Available = true
	f.status.Running = true
	f.status.Healthy = true
	f.status.Launch = &llamacpp.LlamaCppLaunchConfig{Mode: "managed", Managed: true, ConfiguredGPULayers: llamacpp.DefaultGPULayers, FastDefaultsEnabled: true, Profile: llamacpp.LaunchProfileCUDAFast}
	return f.status
}

func (f *fakeSidecar) RestartWithModelFastCUDA(_ context.Context, modelPath string) llamacpp.LlamaCppSidecarStatus {
	f.fastCUDARestarts++
	f.fastRestartPath = modelPath
	if f.fastCUDAStatus.ModelPath != "" {
		f.status = f.fastCUDAStatus
		return f.status
	}
	return f.RestartWithModel(context.Background(), modelPath)
}

func (f *fakeSidecar) RestartWithModelSafeCUDA(_ context.Context, modelPath string) llamacpp.LlamaCppSidecarStatus {
	f.safeCUDAFallbacks++
	f.fallbackRestartPath = modelPath
	if f.safeCUDAStatus.ModelPath != "" {
		f.status = f.safeCUDAStatus
		return f.status
	}
	return f.RestartWithModel(context.Background(), modelPath)
}

func (f *fakeSidecar) RestartWithModelPartialGPU(_ context.Context, modelPath string) llamacpp.LlamaCppSidecarStatus {
	f.partialGPUFallbacks++
	f.fallbackRestartPath = modelPath
	if f.partialGPUStatus.ModelPath != "" {
		f.status = f.partialGPUStatus
		return f.status
	}
	return f.RestartWithModel(context.Background(), modelPath)
}

type fakeCompletionClient struct {
	result llamacpp.CompletionResult
	err    error
	deltas []string
	reqs   []llamacpp.CompletionRequest
}

func (f *fakeCompletionClient) Complete(_ context.Context, req llamacpp.CompletionRequest) (llamacpp.CompletionResult, error) {
	f.reqs = append(f.reqs, req)
	if f.err != nil {
		return llamacpp.CompletionResult{}, f.err
	}
	for _, delta := range f.deltas {
		if req.OnDelta == nil {
			continue
		}
		if err := req.OnDelta(llamacpp.CompletionDelta{Text: delta}); err != nil {
			return llamacpp.CompletionResult{}, err
		}
	}
	return f.result, nil
}

type fakeSlotClient struct {
	restores []llamacpp.SlotCacheRequest
	saves    []llamacpp.SlotCacheRequest
	restore  llamacpp.SlotCacheResult
	save     llamacpp.SlotCacheResult
	err      error
}

func (f *fakeSlotClient) ListSlots(context.Context, string) ([]llamacpp.SlotState, error) {
	return nil, nil
}

func (f *fakeSlotClient) RestoreSlot(_ context.Context, req llamacpp.SlotCacheRequest) (llamacpp.SlotCacheResult, error) {
	f.restores = append(f.restores, req)
	if f.err != nil {
		return llamacpp.SlotCacheResult{}, f.err
	}
	return f.restore, nil
}

func (f *fakeSlotClient) SaveSlot(_ context.Context, req llamacpp.SlotCacheRequest) (llamacpp.SlotCacheResult, error) {
	f.saves = append(f.saves, req)
	if f.err != nil {
		return llamacpp.SlotCacheResult{}, f.err
	}
	return f.save, nil
}

func (f *fakeSlotClient) EraseSlot(context.Context, llamacpp.SlotCacheRequest) (llamacpp.SlotCacheResult, error) {
	return llamacpp.SlotCacheResult{}, nil
}

type fakeProgressSender struct {
	err     error
	batches []ProgressBatch
}

func (f *fakeProgressSender) SendDashboardInferenceProgress(_ context.Context, batch ProgressBatch) error {
	f.batches = append(f.batches, batch)
	if f.err != nil {
		return f.err
	}
	return nil
}

func TestExecuteAssignmentRunsDashboardInferenceWithMockedLlamaCpp(t *testing.T) {
	sidecar := &fakeSidecar{status: healthySidecarStatus()}
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{
		Output:          []byte("dashboard inference measured output"),
		OutputBytes:     int64(len("dashboard inference measured output")),
		TokensGenerated: 7,
		TTFTMs:          80,
		TotalTimeMs:     430,
		Streamed:        true,
	}}
	receipt, handled, err := ExecuteAssignment(context.Background(), validSpecJSON(t), ExecuteOptions{
		Getenv: getenvEnabled,
		Runner: LlamaCppRunner{
			Sidecar: sidecar,
			Client:  client,
		},
	})
	if err != nil {
		t.Fatalf("ExecuteAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if sidecar.starts != 1 {
		t.Fatalf("sidecar starts = %d, want 1", sidecar.starts)
	}
	if len(client.reqs) != 1 {
		t.Fatalf("client calls = %d, want 1", len(client.reqs))
	}
	if client.reqs[0].Stream {
		t.Fatal("completion request stream = true without return_text/text/stream flags, want false")
	}
	if sidecar.stops != 1 {
		t.Fatalf("sidecar stops = %d, want idle cleanup when keep warm is disabled", sidecar.stops)
	}
	if client.reqs[0].ModelID != "Llama-3.2-3B-Instruct-Q4_K_M.gguf" {
		t.Fatalf("completion model = %q", client.reqs[0].ModelID)
	}
	if receipt.JobID != "v7dashboardinfer_job" || receipt.MeteringUnits != 1 {
		t.Fatalf("receipt = %+v, want measured job receipt", receipt)
	}
	metadata, ok := receipt.Metadata[Task].(map[string]any)
	if !ok {
		t.Fatalf("receipt metadata missing %q: %+v", Task, receipt.Metadata)
	}
	if metadata["proof_status"] != ProofStatusMeasured {
		t.Fatalf("proof_status = %v", metadata["proof_status"])
	}
	if metadata["output_hash"] == "" || metadata["tokens_generated"] != int64(7) {
		t.Fatalf("metadata missing hash/metrics: %+v", metadata)
	}
	if metadata["result_hash_hex"] != receipt.ResultHashHex {
		t.Fatalf("result_hash_hex = %#v, want receipt result hash %q", metadata["result_hash_hex"], receipt.ResultHashHex)
	}
	if metadata["requested_max_tokens"] != int(32) || metadata["finish_reason"] != llamacpp.FinishReasonUnknown || metadata["max_tokens_reached"] != false {
		t.Fatalf("finish metadata = %+v", metadata)
	}
	if _, ok := metadata["generated_text"]; ok {
		t.Fatalf("metadata included generated_text without opt-in: %+v", metadata)
	}
	if metadata["ttft_ms"] != int64(80) || metadata["total_time_ms"] != int64(430) {
		t.Fatalf("timing metadata = %+v", metadata)
	}
	if metadata["decode_tps"] != float64(20) || metadata["end_to_end_tps"] != float64(16.279) {
		t.Fatalf("tps metadata = %+v", metadata)
	}
	if metadata["p50_ttft_ms"] != int64(80) || metadata["p50_decode_tps"] != float64(20) || metadata["tpot_ms"] != float64(50) || metadata["p50_end_to_end_tps"] != float64(16.279) {
		t.Fatalf("normalized runtime metrics = %+v", metadata)
	}
	if metadata["runtime_measurement_status"] != llamacpp.RuntimeMeasurementStatusMeasured || metadata["metadata_parse_status"] != llamacpp.MetadataParseStatusOK {
		t.Fatalf("measurement statuses = %+v", metadata)
	}
	if !ReceiptJSONContainsNoRawText(receipt) {
		raw, _ := json.Marshal(receipt.Metadata)
		t.Fatalf("metadata leaked raw text: %s", raw)
	}
}

func TestExecuteAssignmentAppliesCachePolicyAndSlotSaveRestore(t *testing.T) {
	sidecar := &fakeSidecar{status: healthySidecarStatus()}
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{
		Output:          []byte("cached output"),
		OutputBytes:     int64(len("cached output")),
		TokensGenerated: 2,
		TTFTMs:          20,
		TotalTimeMs:     80,
	}}
	slotID := 0
	spec := validSpec(t)
	spec.ReturnText = true
	spec.Prompt = "Summarize the warm session."
	spec.CachePolicy = CachePolicy{
		SessionID:            "sess-customer-123",
		PrefixHash:           testSHA256("shared prefix"),
		CachePrompt:          true,
		CacheReuseTokens:     64,
		SlotID:               &slotID,
		RestoreSlotBeforeRun: true,
		SaveSlotAfterRun:     true,
		CacheStateID:         "state-alpha",
		AffinityTTLSeconds:   600,
	}
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	slots := &fakeSlotClient{
		restore: llamacpp.SlotCacheResult{RestoredTokens: 42},
		save:    llamacpp.SlotCacheResult{SavedTokens: 55},
	}

	receipt, handled, err := ExecuteAssignment(context.Background(), string(raw), ExecuteOptions{
		Getenv: getenvTextOutputEnabled,
		Runner: LlamaCppRunner{
			Sidecar:    sidecar,
			Client:     client,
			SlotClient: slots,
		},
	})
	if err != nil {
		t.Fatalf("ExecuteAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if len(client.reqs) != 1 {
		t.Fatalf("client calls = %d, want 1", len(client.reqs))
	}
	req := client.reqs[0]
	if !req.CachePrompt || req.CacheReuseTokens != 64 || req.SlotID == nil || *req.SlotID != 0 {
		t.Fatalf("completion cache request = %+v, want cache_prompt/reuse/slot", req)
	}
	if len(slots.restores) != 1 || len(slots.saves) != 1 {
		t.Fatalf("slot operations restores=%d saves=%d, want one each", len(slots.restores), len(slots.saves))
	}
	if slots.restores[0].Filename == "" || strings.Contains(slots.restores[0].Filename, "sess-customer") {
		t.Fatalf("restore filename = %q, want derived non-raw session filename", slots.restores[0].Filename)
	}
	if slots.saves[0].Filename != slots.restores[0].Filename {
		t.Fatalf("save filename = %q, want restore filename %q", slots.saves[0].Filename, slots.restores[0].Filename)
	}
	metadata, ok := receipt.Metadata[Task].(map[string]any)
	if !ok {
		t.Fatalf("receipt metadata missing %q: %+v", Task, receipt.Metadata)
	}
	cache, ok := metadata["cache"].(map[string]any)
	if !ok {
		t.Fatalf("cache metadata missing: %+v", metadata)
	}
	if cache["cache_prompt"] != true || cache["cache_reuse_tokens"] != int(64) || cache["slot_id"] != int(0) {
		t.Fatalf("cache policy metadata = %+v", cache)
	}
	if cache["restore_status"] != "restored" || cache["restored_tokens"] != int64(42) || cache["save_status"] != "saved" || cache["saved_tokens"] != int64(55) {
		t.Fatalf("cache operation metadata = %+v", cache)
	}
	if cache["prefix_hash"] != spec.CachePolicy.PrefixHash || cache["session_id_hash"] == "" || cache["session_id_hash"] == spec.CachePolicy.SessionID {
		t.Fatalf("cache identity metadata = %+v", cache)
	}
	rawReceipt, _ := json.Marshal(receipt.Metadata)
	if strings.Contains(string(rawReceipt), "sess-customer-123") || strings.Contains(string(rawReceipt), spec.Prompt) {
		t.Fatalf("receipt leaked raw cache/session/prompt text: %s", rawReceipt)
	}
}

func TestExecuteAssignmentSwitchesColdResidentModelBeforeInference(t *testing.T) {
	cacheDir := t.TempDir()
	phiPath := filepath.Join(cacheDir, "phi-4-Q4_K_M.gguf")
	if err := os.WriteFile(phiPath, []byte("gguf"), 0o644); err != nil {
		t.Fatalf("write phi fixture: %v", err)
	}
	sidecar := &fakeSidecar{status: healthySidecarStatus()}
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{
		Output:          []byte("phi output"),
		OutputBytes:     int64(len("phi output")),
		TokensGenerated: 2,
		TTFTMs:          12,
		TotalTimeMs:     42,
	}}
	spec := validSpec(t)
	spec.ModelID = "phi-4-Q4_K_M.gguf"
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}

	policy := modelpolicy.Policy{
		AutoDownload:        false,
		MaxSingleModelBytes: 16 << 30,
		MaxCacheBytes:       64 << 30,
		CacheDir:            cacheDir,
		AllowedFamilies:     []string{"llama", "phi"},
		AllowedFormats:      []string{"gguf"},
		RuntimePolicy: modelpolicy.RuntimePolicy{
			AllowRuntimeExecution:            true,
			MaxRuntimeModelBytes:             16 << 30,
			MaxRuntimeParameterCountBillions: 16,
			AllowFamilies:                    []string{"llama", "phi"},
		},
	}
	receipt, handled, err := ExecuteAssignment(context.Background(), string(raw), ExecuteOptions{
		Getenv: getenvEnabled,
		Policy: policy,
		Runner: LlamaCppRunner{
			Sidecar: sidecar,
			Client:  client,
			Policy:  policy,
		},
	})
	if err != nil || !handled {
		t.Fatalf("ExecuteAssignment() handled=%v error=%v, want handled success", handled, err)
	}
	if sidecar.restarts != 1 || sidecar.restartPath != phiPath {
		t.Fatalf("sidecar restarts/path = %d/%q, want one restart to %q", sidecar.restarts, sidecar.restartPath, phiPath)
	}
	if len(client.reqs) != 1 || client.reqs[0].ModelID != "phi-4-Q4_K_M.gguf" {
		t.Fatalf("completion requests = %#v, want Phi inference after model switch", client.reqs)
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["proof_status"] != ProofStatusMeasured || metadata["model_id"] != "phi-4-Q4_K_M.gguf" {
		t.Fatalf("metadata = %+v, want measured Phi receipt", metadata)
	}
}

func TestFindDashboardCachedModelMatchesGemma4QATForCatalogName(t *testing.T) {
	cacheDir := t.TempDir()
	path := filepath.Join(cacheDir, "gemma-4-27b-it-q4_0.gguf")
	status := modelcache.Status{
		CacheDir: cacheDir,
		Models: []modelcache.Model{{
			ModelID:    "gemma-4-27b-it-q4_0.gguf",
			Filename:   "gemma-4-27b-it-q4_0.gguf",
			Path:       path,
			SizeBytes:  18 << 30,
			FamilyHint: "gemma",
			Format:     "gguf",
			Installed:  true,
		}},
	}

	model, ok := findDashboardCachedModel(status, "gemma-4-27b-it-Q4_K_M.gguf")
	if !ok || model.Path != path {
		t.Fatalf("findDashboardCachedModel() = %+v/%v, want local Gemma 4 QAT artifact", model, ok)
	}
}

func TestExecuteAssignmentWaitsForColdModelRestartToBecomeHealthy(t *testing.T) {
	cacheDir := t.TempDir()
	phiPath := filepath.Join(cacheDir, "phi-4-Q4_K_M.gguf")
	if err := os.WriteFile(phiPath, []byte("gguf"), 0o644); err != nil {
		t.Fatalf("write phi fixture: %v", err)
	}
	restarting := healthySidecarStatus()
	restarting.ModelPath = phiPath
	restarting.ModelFilename = filepath.Base(phiPath)
	restarting.ModelSizeBytes = int64(len("gguf"))
	restarting.Healthy = false
	ready := restarting
	ready.Healthy = true
	sidecar := &fakeSidecar{
		status:        healthySidecarStatus(),
		restartStatus: restarting,
		statusQueue:   []llamacpp.LlamaCppSidecarStatus{ready},
	}
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{
		Output:          []byte("phi warmed output"),
		OutputBytes:     int64(len("phi warmed output")),
		TokensGenerated: 3,
		TTFTMs:          30,
		TotalTimeMs:     90,
	}}
	spec := validSpec(t)
	spec.ModelID = "phi-4-Q4_K_M.gguf"
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}

	policy := modelpolicy.Policy{
		AutoDownload:        false,
		MaxSingleModelBytes: 16 << 30,
		MaxCacheBytes:       64 << 30,
		CacheDir:            cacheDir,
		AllowedFamilies:     []string{"llama", "phi"},
		AllowedFormats:      []string{"gguf"},
		RuntimePolicy: modelpolicy.RuntimePolicy{
			AllowRuntimeExecution:            true,
			MaxRuntimeModelBytes:             16 << 30,
			MaxRuntimeParameterCountBillions: 16,
			AllowFamilies:                    []string{"llama", "phi"},
		},
	}
	receipt, handled, err := ExecuteAssignment(context.Background(), string(raw), ExecuteOptions{
		Getenv: getenvEnabled,
		Policy: policy,
		Runner: LlamaCppRunner{
			Sidecar: sidecar,
			Client:  client,
			Policy:  policy,
		},
	})
	if err != nil || !handled {
		t.Fatalf("ExecuteAssignment() handled=%v error=%v, want handled success", handled, err)
	}
	if sidecar.restarts != 1 || sidecar.restartPath != phiPath {
		t.Fatalf("sidecar restarts/path = %d/%q, want one restart to %q", sidecar.restarts, sidecar.restartPath, phiPath)
	}
	if len(client.reqs) != 1 {
		t.Fatalf("completion calls = %d, want 1 after restart became healthy", len(client.reqs))
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["proof_status"] != ProofStatusMeasured || metadata["error_code"] != nil {
		t.Fatalf("metadata = %+v, want measured receipt after delayed sidecar readiness", metadata)
	}
}

func TestExecuteAssignmentRetriesCudaFallbackWhenColdLaunchExits(t *testing.T) {
	cacheDir := t.TempDir()
	phiPath := filepath.Join(cacheDir, "phi-4-Q4_K_M.gguf")
	if err := os.WriteFile(phiPath, []byte("gguf"), 0o644); err != nil {
		t.Fatalf("write phi fixture: %v", err)
	}
	launching := healthySidecarStatus()
	launching.ModelPath = phiPath
	launching.ModelFilename = filepath.Base(phiPath)
	launching.ModelSizeBytes = int64(len("gguf"))
	launching.Healthy = false
	exited := launching
	exited.Running = false
	exited.Healthy = false
	exited.LastError = "exit status 1: CUDA error: out of memory"
	exited.Reason = exited.LastError
	ready := launching
	ready.Healthy = true
	sidecar := &fakeSidecar{
		status:           healthySidecarStatus(),
		restartStatus:    launching,
		statusQueue:      []llamacpp.LlamaCppSidecarStatus{exited},
		partialGPUStatus: ready,
	}
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{
		Output:          []byte("phi recovered output"),
		OutputBytes:     int64(len("phi recovered output")),
		TokensGenerated: 3,
		TTFTMs:          20,
		TotalTimeMs:     80,
	}}
	spec := validSpec(t)
	spec.ModelID = "phi-4-Q4_K_M.gguf"
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}

	policy := modelpolicy.Policy{
		AutoDownload:        false,
		MaxSingleModelBytes: 16 << 30,
		MaxCacheBytes:       64 << 30,
		CacheDir:            cacheDir,
		AllowedFamilies:     []string{"llama", "phi"},
		AllowedFormats:      []string{"gguf"},
		RuntimePolicy: modelpolicy.RuntimePolicy{
			AllowRuntimeExecution:            true,
			MaxRuntimeModelBytes:             16 << 30,
			MaxRuntimeParameterCountBillions: 16,
			AllowFamilies:                    []string{"llama", "phi"},
		},
	}
	receipt, handled, err := ExecuteAssignment(context.Background(), string(raw), ExecuteOptions{
		Getenv: getenvEnabled,
		Policy: policy,
		Runner: LlamaCppRunner{
			Sidecar: sidecar,
			Client:  client,
			Policy:  policy,
		},
	})
	if err != nil || !handled {
		t.Fatalf("ExecuteAssignment() handled=%v error=%v, want handled success", handled, err)
	}
	if sidecar.safeCUDAFallbacks != 0 || sidecar.partialGPUFallbacks != 1 || sidecar.fallbackRestartPath != phiPath {
		t.Fatalf("fallbacks safe/partial/path = %d/%d/%q, want direct partial fallback to %q", sidecar.safeCUDAFallbacks, sidecar.partialGPUFallbacks, sidecar.fallbackRestartPath, phiPath)
	}
	if len(client.reqs) != 1 {
		t.Fatalf("completion calls = %d, want 1 after fallback readiness", len(client.reqs))
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["proof_status"] != ProofStatusMeasured || metadata["error_code"] != nil {
		t.Fatalf("metadata = %+v, want measured receipt after CUDA fallback", metadata)
	}
}

func TestExecuteAssignmentRetriesSafeCUDAForUnsupportedFastFlag(t *testing.T) {
	cacheDir := t.TempDir()
	phiPath := filepath.Join(cacheDir, "phi-4-Q4_K_M.gguf")
	if err := os.WriteFile(phiPath, []byte("gguf"), 0o644); err != nil {
		t.Fatalf("write phi fixture: %v", err)
	}
	launching := healthySidecarStatus()
	launching.ModelPath = phiPath
	launching.ModelFilename = filepath.Base(phiPath)
	launching.ModelSizeBytes = int64(len("gguf"))
	launching.Healthy = false
	exited := launching
	exited.Running = false
	exited.Healthy = false
	exited.LastError = "exit status 1: unknown argument --flash-attn"
	exited.Reason = exited.LastError
	ready := launching
	ready.Healthy = true
	sidecar := &fakeSidecar{
		status:         healthySidecarStatus(),
		restartStatus:  launching,
		statusQueue:    []llamacpp.LlamaCppSidecarStatus{exited},
		safeCUDAStatus: ready,
	}
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{
		Output:          []byte("phi safe cuda output"),
		OutputBytes:     int64(len("phi safe cuda output")),
		TokensGenerated: 4,
		TTFTMs:          20,
		TotalTimeMs:     80,
	}}
	spec := validSpec(t)
	spec.ModelID = "phi-4-Q4_K_M.gguf"
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	policy := modelpolicy.Policy{
		AutoDownload:        false,
		MaxSingleModelBytes: 16 << 30,
		MaxCacheBytes:       64 << 30,
		CacheDir:            cacheDir,
		AllowedFamilies:     []string{"llama", "phi"},
		AllowedFormats:      []string{"gguf"},
		RuntimePolicy: modelpolicy.RuntimePolicy{
			AllowRuntimeExecution:            true,
			MaxRuntimeModelBytes:             16 << 30,
			MaxRuntimeParameterCountBillions: 16,
			AllowFamilies:                    []string{"llama", "phi"},
		},
	}

	receipt, handled, err := ExecuteAssignment(context.Background(), string(raw), ExecuteOptions{
		Getenv: getenvEnabled,
		Policy: policy,
		Runner: LlamaCppRunner{
			Sidecar: sidecar,
			Client:  client,
			Policy:  policy,
		},
	})
	if err != nil || !handled {
		t.Fatalf("ExecuteAssignment() handled=%v error=%v, want handled success", handled, err)
	}
	if sidecar.safeCUDAFallbacks != 1 || sidecar.partialGPUFallbacks != 0 || sidecar.fallbackRestartPath != phiPath {
		t.Fatalf("fallbacks safe/partial/path = %d/%d/%q, want one safe fallback to %q", sidecar.safeCUDAFallbacks, sidecar.partialGPUFallbacks, sidecar.fallbackRestartPath, phiPath)
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["proof_status"] != ProofStatusMeasured || metadata["error_code"] != nil {
		t.Fatalf("metadata = %+v, want measured receipt after safe CUDA fallback", metadata)
	}
}

func TestExecuteAssignmentRestoresFastCUDAFromDegradedWarmSidecar(t *testing.T) {
	cacheDir := t.TempDir()
	phiPath := filepath.Join(cacheDir, "phi-4-Q4_K_M.gguf")
	if err := os.WriteFile(phiPath, []byte("gguf"), 0o644); err != nil {
		t.Fatalf("write phi fixture: %v", err)
	}
	degraded := healthySidecarStatus()
	degraded.ModelPath = phiPath
	degraded.ModelFilename = filepath.Base(phiPath)
	degraded.ModelSizeBytes = int64(len("gguf"))
	degraded.Launch = &llamacpp.LlamaCppLaunchConfig{
		Mode:                "managed",
		Managed:             true,
		ConfiguredGPULayers: 35,
		Profile:             llamacpp.LaunchProfileCUDAPartial,
	}
	fast := degraded
	fast.Launch = &llamacpp.LlamaCppLaunchConfig{
		Mode:                "managed",
		Managed:             true,
		ConfiguredGPULayers: llamacpp.DefaultGPULayers,
		FastDefaultsEnabled: true,
		Profile:             llamacpp.LaunchProfileCUDAFast,
	}
	sidecar := &fakeSidecar{
		status:         degraded,
		fastCUDAStatus: fast,
	}
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{
		Output:          []byte("phi fast cuda output"),
		OutputBytes:     int64(len("phi fast cuda output")),
		TokensGenerated: 4,
		TTFTMs:          20,
		TotalTimeMs:     80,
	}}
	spec := validSpec(t)
	spec.ModelID = "phi-4-Q4_K_M.gguf"
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	policy := modelpolicy.Policy{
		AutoDownload:        false,
		MaxSingleModelBytes: 16 << 30,
		MaxCacheBytes:       64 << 30,
		CacheDir:            cacheDir,
		AllowedFamilies:     []string{"llama", "phi"},
		AllowedFormats:      []string{"gguf"},
		RuntimePolicy: modelpolicy.RuntimePolicy{
			AllowRuntimeExecution:            true,
			MaxRuntimeModelBytes:             16 << 30,
			MaxRuntimeParameterCountBillions: 16,
			AllowFamilies:                    []string{"llama", "phi"},
		},
	}

	receipt, handled, err := ExecuteAssignment(context.Background(), string(raw), ExecuteOptions{
		Getenv: getenvEnabled,
		Policy: policy,
		Runner: LlamaCppRunner{
			Sidecar: sidecar,
			Client:  client,
			Policy:  policy,
		},
	})
	if err != nil || !handled {
		t.Fatalf("ExecuteAssignment() handled=%v error=%v, want handled success", handled, err)
	}
	if sidecar.fastCUDARestarts != 1 || sidecar.fastRestartPath != phiPath {
		t.Fatalf("fast CUDA restarts/path = %d/%q, want one restart to %q", sidecar.fastCUDARestarts, sidecar.fastRestartPath, phiPath)
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["proof_status"] != ProofStatusMeasured || metadata["error_code"] != nil {
		t.Fatalf("metadata = %+v, want measured receipt after fast CUDA restore", metadata)
	}
}

func TestExecuteAssignmentPromotesDefaultCUDAProfileToFastDefaults(t *testing.T) {
	cacheDir := t.TempDir()
	phiPath := filepath.Join(cacheDir, "phi-4-Q4_K_M.gguf")
	if err := os.WriteFile(phiPath, []byte("gguf"), 0o644); err != nil {
		t.Fatalf("write phi fixture: %v", err)
	}
	defaultCUDA := healthySidecarStatus()
	defaultCUDA.ModelPath = phiPath
	defaultCUDA.ModelFilename = filepath.Base(phiPath)
	defaultCUDA.ModelSizeBytes = int64(len("gguf"))
	defaultCUDA.Launch = &llamacpp.LlamaCppLaunchConfig{
		Mode:                "managed",
		Managed:             true,
		ConfiguredGPULayers: llamacpp.DefaultGPULayers,
		FastDefaultsEnabled: false,
		Profile:             llamacpp.LaunchProfileDefault,
	}
	defaultCUDA.ServerProperties = &llamacpp.LlamaCppServerProperties{
		BuildInfo:            "llama.cpp CUDA",
		SystemInfo:           "CUDA enabled",
		ReportedAcceleration: []string{"cuda"},
	}
	fast := defaultCUDA
	fast.Launch = &llamacpp.LlamaCppLaunchConfig{
		Mode:                "managed",
		Managed:             true,
		ConfiguredGPULayers: llamacpp.DefaultGPULayers,
		FastDefaultsEnabled: true,
		Profile:             llamacpp.LaunchProfileCUDAFast,
	}
	sidecar := &fakeSidecar{
		status:         defaultCUDA,
		fastCUDAStatus: fast,
	}
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{
		Output:          []byte("phi fast cuda output"),
		OutputBytes:     int64(len("phi fast cuda output")),
		TokensGenerated: 4,
		TTFTMs:          20,
		TotalTimeMs:     80,
	}}
	spec := validSpec(t)
	spec.ModelID = "phi-4-Q4_K_M.gguf"
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	policy := modelpolicy.Policy{
		AutoDownload:        false,
		MaxSingleModelBytes: 16 << 30,
		MaxCacheBytes:       64 << 30,
		CacheDir:            cacheDir,
		AllowedFamilies:     []string{"llama", "phi"},
		AllowedFormats:      []string{"gguf"},
		RuntimePolicy: modelpolicy.RuntimePolicy{
			AllowRuntimeExecution:            true,
			MaxRuntimeModelBytes:             16 << 30,
			MaxRuntimeParameterCountBillions: 16,
			AllowFamilies:                    []string{"llama", "phi"},
		},
	}

	receipt, handled, err := ExecuteAssignment(context.Background(), string(raw), ExecuteOptions{
		Getenv: getenvEnabled,
		Policy: policy,
		Runner: LlamaCppRunner{
			Sidecar: sidecar,
			Client:  client,
			Policy:  policy,
		},
	})
	if err != nil || !handled {
		t.Fatalf("ExecuteAssignment() handled=%v error=%v, want handled success", handled, err)
	}
	if sidecar.fastCUDARestarts != 1 || sidecar.fastRestartPath != phiPath {
		t.Fatalf("fast CUDA restarts/path = %d/%q, want one restart to %q", sidecar.fastCUDARestarts, sidecar.fastRestartPath, phiPath)
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["proof_status"] != ProofStatusMeasured || metadata["error_code"] != nil {
		t.Fatalf("metadata = %+v, want measured receipt after fast CUDA promotion", metadata)
	}
}

func TestSafeSidecarFailureClassifiesCudaLaunchFailures(t *testing.T) {
	status := healthySidecarStatus()
	status.Running = false
	status.Healthy = false
	status.LastError = `exit status 1: cudart64_12.dll was not found`
	if got := safeSidecarFailure(status); got != "llamacpp_cuda_runtime_missing" {
		t.Fatalf("safeSidecarFailure(runtime) = %q", got)
	}
	status.LastError = "exit status 1: CUDA error: out of memory"
	if got := safeSidecarFailure(status); got != "llamacpp_cuda_out_of_memory" {
		t.Fatalf("safeSidecarFailure(oom) = %q", got)
	}
	status.LastError = "exit status 1: unknown argument --flash-attn"
	if got := safeSidecarFailure(status); got != "llamacpp_launch_arg_unsupported" {
		t.Fatalf("safeSidecarFailure(args) = %q", got)
	}
}

func TestExecuteAssignmentUsesPromptWithoutReturningText(t *testing.T) {
	sidecar := &fakeSidecar{status: healthySidecarStatus()}
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{
		Output:          []byte("private completion text"),
		OutputBytes:     int64(len("private completion text")),
		TokensGenerated: 3,
		TTFTMs:          20,
		TotalTimeMs:     120,
	}}
	spec := validSpec(t)
	spec.Prompt = "private dashboard prompt"
	spec.ReturnText = false
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	receipt, handled, err := ExecuteAssignment(context.Background(), string(raw), ExecuteOptions{
		Getenv: getenvEnabled,
		Runner: LlamaCppRunner{
			Sidecar: sidecar,
			Client:  client,
		},
	})
	if err != nil {
		t.Fatalf("ExecuteAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if len(client.reqs) != 1 || client.reqs[0].Prompt != "private dashboard prompt" {
		t.Fatal("completion request did not use supplied prompt")
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if _, ok := metadata["generated_text"]; ok {
		t.Fatalf("metadata included generated_text without return_text: %+v", metadata)
	}
	encoded, _ := json.Marshal(receipt.Metadata)
	for _, forbidden := range []string{"private dashboard prompt", "private completion text", `"generated_text":`} {
		if strings.Contains(string(encoded), forbidden) {
			t.Fatalf("metadata leaked %q: %s", forbidden, encoded)
		}
	}
}

func TestExecuteAssignmentReturnTextFlagDisabledRejects(t *testing.T) {
	client := &fakeCompletionClient{}
	spec := validSpec(t)
	spec.Prompt = "private dashboard prompt"
	spec.ReturnText = true
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	receipt, handled, err := ExecuteAssignment(context.Background(), string(raw), ExecuteOptions{
		Getenv: getenvTextOutputDisabled,
		Runner: LlamaCppRunner{
			Sidecar: &fakeSidecar{status: healthySidecarStatus()},
			Client:  client,
		},
	})
	if err == nil {
		t.Fatal("ExecuteAssignment() error = nil, want text output disabled error")
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if len(client.reqs) != 0 {
		t.Fatalf("client calls = %d, want 0", len(client.reqs))
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["proof_status"] != ProofStatusRejected || metadata["error_code"] != "text_output_disabled" {
		t.Fatalf("disabled text metadata = %+v", metadata)
	}
	encoded, _ := json.Marshal(receipt.Metadata)
	if strings.Contains(string(encoded), "private dashboard prompt") || strings.Contains(string(encoded), `"generated_text":`) {
		t.Fatalf("disabled text receipt leaked prompt/text field: %s", encoded)
	}
}

func TestExecuteAssignmentReturnTextFlagEnabledIncludesGeneratedText(t *testing.T) {
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{
		Output:          []byte("Ryvion routes AI work to warm, ready nodes."),
		OutputBytes:     int64(len("Ryvion routes AI work to warm, ready nodes.")),
		TokensGenerated: 8,
		TTFTMs:          123,
		TotalTimeMs:     456,
	}}
	spec := validSpec(t)
	spec.Prompt = "Write one short sentence about Ryvion."
	spec.ReturnText = true
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	receipt, handled, err := ExecuteAssignment(context.Background(), string(raw), ExecuteOptions{
		Getenv: getenvTextOutputEnabled,
		Runner: LlamaCppRunner{
			Sidecar: &fakeSidecar{status: healthySidecarStatus()},
			Client:  client,
		},
	})
	if err != nil {
		t.Fatalf("ExecuteAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["generated_text"] != "Ryvion routes AI work to warm, ready nodes." {
		t.Fatalf("generated_text = %#v", metadata["generated_text"])
	}
	if metadata["generated_text_truncated"] != false {
		t.Fatalf("generated_text_truncated = %#v", metadata["generated_text_truncated"])
	}
	if metadata["output_hash"] == "" || metadata["tokens_generated"] != int64(8) || metadata["ttft_ms"] != int64(123) {
		t.Fatalf("metadata missing hash/timing metrics: %+v", metadata)
	}
	if metadata["p50_ttft_ms"] != int64(123) || metadata["p50_decode_tps"] == nil || metadata["proof_status"] != ProofStatusMeasured {
		t.Fatalf("normalized schema missing from text receipt: %+v", metadata)
	}
	if len(receipt.ResultHashHex) != 64 {
		t.Fatalf("result hash = %q, want 64 hex chars", receipt.ResultHashHex)
	}
}

func TestLlamaCppRunnerPassesResolvedSystemPromptAndReportsSafeDebugMetadata(t *testing.T) {
	const systemPrompt = "Ryvion is a warm backend-aware distributed AI execution fabric for DePIN AI execution."
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{
		Output:          []byte("Ryvion routes warm local Llama requests."),
		OutputBytes:     int64(len("Ryvion routes warm local Llama requests.")),
		TokensGenerated: 6,
		TTFTMs:          50,
		TotalTimeMs:     250,
	}}
	spec := validSpec(t)
	spec.Prompt = "Explain Ryvion."
	spec.SystemPrompt = systemPrompt
	spec.UseDefaultRyvionGrounding = true
	spec.ReturnText = true
	runner := LlamaCppRunner{
		Sidecar: &fakeSidecar{status: healthySidecarStatus()},
		Client:  client,
		Getenv:  getenvTextOutputEnabled,
	}
	result, err := runner.RunDashboardInference(context.Background(), spec)
	if err != nil {
		t.Fatalf("RunDashboardInference() error = %v", err)
	}
	if len(client.reqs) != 1 {
		t.Fatalf("client calls = %d, want 1", len(client.reqs))
	}
	if client.reqs[0].SystemPrompt != systemPrompt || client.reqs[0].Prompt != spec.Prompt {
		t.Fatalf("completion request = %+v, want resolved system and user prompts", client.reqs[0])
	}
	wantHash := llamacpp.HashSystemPrompt(systemPrompt)
	if !result.GroundingApplied || result.PromptMode != llamacpp.PromptModeChatMessages || result.SystemPromptHash != wantHash {
		t.Fatalf("result prompt metadata = %+v, want grounding/chat/hash", result)
	}
	receipt, err := BuildReceipt(result)
	if err != nil {
		t.Fatalf("BuildReceipt() error = %v", err)
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["grounding_applied"] != true ||
		metadata["prompt_mode"] != llamacpp.PromptModeChatMessages ||
		metadata["system_prompt_hash"] != wantHash {
		t.Fatalf("receipt prompt metadata = %+v", metadata)
	}
	encoded, _ := json.Marshal(receipt.Metadata)
	if strings.Contains(string(encoded), systemPrompt) || strings.Contains(string(encoded), spec.Prompt) {
		t.Fatalf("receipt metadata leaked prompt text: %s", encoded)
	}
}

func TestLlamaCppRunnerAppliesDefaultGroundingAndPassesMessages(t *testing.T) {
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{
		Output:          []byte("Ryvion reports warm local runtime readiness."),
		OutputBytes:     int64(len("Ryvion reports warm local runtime readiness.")),
		TokensGenerated: 6,
		TTFTMs:          40,
		TotalTimeMs:     240,
	}}
	spec := validSpec(t)
	spec.Prompt = ""
	spec.Messages = []llamacpp.CompletionMessage{
		{Role: "user", Content: "What is Ryvion?"},
		{Role: "assistant", Content: "Ryvion runs local AI workloads."},
		{Role: "user", Content: "Mention warm readiness."},
	}
	spec.UseDefaultRyvionGrounding = true
	spec.ReturnText = true
	runner := LlamaCppRunner{
		Sidecar: &fakeSidecar{status: healthySidecarStatus()},
		Client:  client,
		Getenv:  getenvTextOutputEnabled,
	}
	result, err := runner.RunDashboardInference(context.Background(), spec)
	if err != nil {
		t.Fatalf("RunDashboardInference() error = %v", err)
	}
	if len(client.reqs) != 1 {
		t.Fatalf("client calls = %d, want 1", len(client.reqs))
	}
	req := client.reqs[0]
	if req.Prompt != "" || req.SystemPrompt != defaultRyvionGroundingSystemPrompt || len(req.Messages) != 3 {
		t.Fatalf("completion request = %+v, want default system prompt plus provided messages", req)
	}
	wantHash := llamacpp.HashSystemPrompt(defaultRyvionGroundingSystemPrompt)
	if !result.GroundingApplied || result.SystemPromptHash != wantHash || result.PromptMode != llamacpp.PromptModeChatMessages {
		t.Fatalf("prompt metadata = %+v, want default grounding metadata", result)
	}
	receipt, err := BuildReceipt(result)
	if err != nil {
		t.Fatalf("BuildReceipt() error = %v", err)
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["grounding_applied"] != true || metadata["system_prompt_hash"] != wantHash || metadata["tokens_generated"] != int64(6) || metadata["proof_status"] != ProofStatusMeasured {
		t.Fatalf("receipt metadata = %+v, want safe grounding and finish metrics", metadata)
	}
	encoded, _ := json.Marshal(receipt.Metadata)
	for _, forbidden := range []string{"What is Ryvion?", "Mention warm readiness.", defaultRyvionGroundingSystemPrompt} {
		if strings.Contains(string(encoded), forbidden) {
			t.Fatalf("receipt metadata leaked request text %q: %s", forbidden, encoded)
		}
	}
}

func TestLlamaCppRunnerStreamsReturnTextWithProgressBatches(t *testing.T) {
	output := "Ryvion routes tokens"
	client := &fakeCompletionClient{
		result: llamacpp.CompletionResult{
			Output:          []byte(output),
			OutputBytes:     int64(len(output)),
			TokensGenerated: 3,
			FinishReason:    llamacpp.FinishReasonStop,
			TTFTMs:          90,
			TotalTimeMs:     390,
			Streamed:        true,
		},
		deltas: []string{"Ryvion", " routes", " tokens"},
	}
	progress := &fakeProgressSender{}
	spec := validSpec(t)
	spec.Prompt = "Write a short dashboard sentence."
	spec.ReturnText = true
	runner := LlamaCppRunner{
		Sidecar: &fakeSidecar{status: healthySidecarStatus()},
		Client:  client,
		Getenv:  getenvTextAndStreamingEnabled,
	}
	result, err := runner.RunDashboardInferenceWithProgress(context.Background(), spec, progress)
	if err != nil {
		t.Fatalf("RunDashboardInferenceWithProgress() error = %v", err)
	}
	if len(client.reqs) != 1 || !client.reqs[0].Stream {
		t.Fatalf("stream requests = %+v, want one streaming request", client.reqs)
	}
	if result.ProofStatus != ProofStatusMeasured || result.GeneratedText != output {
		t.Fatalf("result = %+v, want measured generated text", result)
	}
	receipt, err := BuildReceipt(result)
	if err != nil {
		t.Fatalf("BuildReceipt() error = %v", err)
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["tokens_generated"] != int64(3) || metadata["finish_reason"] != llamacpp.FinishReasonStop || metadata["generated_text"] != output {
		t.Fatalf("streaming final receipt metadata = %+v", metadata)
	}
	if len(progress.batches) != 1 {
		t.Fatalf("progress batches = %d, want 1", len(progress.batches))
	}
	batch := progress.batches[0]
	if batch.RunID != spec.RunID || batch.JobID != spec.JobID || batch.NodeID != spec.TargetNodeID || batch.SeqStart != 1 {
		t.Fatalf("batch identity = %+v", batch)
	}
	if len(batch.Chunks) != 4 {
		t.Fatalf("batch chunks = %+v, want 4 including final done", batch.Chunks)
	}
	for idx, chunk := range batch.Chunks[:3] {
		wantSeq := int64(idx + 1)
		if chunk.Seq != wantSeq || chunk.Type != ProgressChunkTypeTokenCommit {
			t.Fatalf("chunk[%d] = %+v, want seq %d token.commit", idx, chunk, wantSeq)
		}
	}
	if got := batch.Chunks[0].Text + batch.Chunks[1].Text + batch.Chunks[2].Text; got != output {
		t.Fatalf("concatenated chunks = %q, want %q", got, output)
	}
	if done := batch.Chunks[3]; done.Seq != 4 || done.Type != ProgressChunkTypeTokenFinalize || done.FinishReason != llamacpp.FinishReasonStop || done.Text != "" {
		t.Fatalf("done chunk = %+v, want final stop reason", done)
	}
}

func TestLlamaCppRunnerRecordsPreInferenceDelaySeparatelyFromBackendTTFT(t *testing.T) {
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{
		Output:          []byte("phi warmed after model load"),
		OutputBytes:     int64(len("phi warmed after model load")),
		TokensGenerated: 5,
		FinishReason:    llamacpp.FinishReasonStop,
		TTFTMs:          88,
		TotalTimeMs:     620,
	}}
	spec := validSpec(t)
	spec.Prompt = "Explain the route."
	spec.ReturnText = true
	runner := LlamaCppRunner{
		Sidecar: &fakeSidecar{
			status:     healthySidecarStatus(),
			startDelay: 5 * time.Millisecond,
		},
		Client: client,
		Getenv: getenvTextAndStreamingEnabled,
	}

	result, err := runner.RunDashboardInferenceWithProgress(context.Background(), spec, nil)
	if err != nil {
		t.Fatalf("RunDashboardInferenceWithProgress() error = %v", err)
	}
	if result.TTFTMs != 88 {
		t.Fatalf("TTFTMs = %d, want backend TTFT unchanged", result.TTFTMs)
	}
	if result.PreInferenceMs <= 0 {
		t.Fatalf("PreInferenceMs = %d, want positive sidecar preparation time", result.PreInferenceMs)
	}
	if result.PerceivedTTFTMs != result.PreInferenceMs+result.TTFTMs {
		t.Fatalf("PerceivedTTFTMs = %d, want pre-inference + TTFT (%d + %d)", result.PerceivedTTFTMs, result.PreInferenceMs, result.TTFTMs)
	}
	receipt, err := BuildReceipt(result)
	if err != nil {
		t.Fatalf("BuildReceipt() error = %v", err)
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["ttft_ms"] != int64(88) {
		t.Fatalf("receipt ttft_ms = %v, want backend TTFT", metadata["ttft_ms"])
	}
	if metadata["pre_inference_ms"] == nil || metadata["perceived_ttft_ms"] == nil {
		t.Fatalf("receipt missing perceived latency fields: %+v", metadata)
	}
}

func TestLlamaCppRunnerCanPostLegacyDeltaChunksWhenV8EventsDisabled(t *testing.T) {
	client := &fakeCompletionClient{
		result: llamacpp.CompletionResult{
			Output:          []byte("legacy chunk"),
			OutputBytes:     int64(len("legacy chunk")),
			TokensGenerated: 2,
			FinishReason:    llamacpp.FinishReasonStop,
			TTFTMs:          25,
			TotalTimeMs:     100,
			Streamed:        true,
		},
		deltas: []string{"legacy ", "chunk"},
	}
	progress := &fakeProgressSender{}
	spec := validSpec(t)
	spec.ReturnText = true
	spec.Prompt = "Return a legacy streaming sentence."
	runner := LlamaCppRunner{
		Sidecar: &fakeSidecar{status: healthySidecarStatus()},
		Client:  client,
		Getenv: func(key string) string {
			if key == V8StreamEventsEnv {
				return "0"
			}
			return getenvTextAndStreamingEnabled(key)
		},
	}
	if _, err := runner.RunDashboardInferenceWithProgress(context.Background(), spec, progress); err != nil {
		t.Fatalf("RunDashboardInferenceWithProgress() error = %v", err)
	}
	if len(progress.batches) != 1 || len(progress.batches[0].Chunks) != 3 {
		t.Fatalf("progress batches = %+v, want one legacy batch", progress.batches)
	}
	if progress.batches[0].Chunks[0].Type != ProgressChunkTypeDelta || progress.batches[0].Chunks[2].Type != ProgressChunkTypeDone {
		t.Fatalf("legacy chunk types = %+v, want delta/done", progress.batches[0].Chunks)
	}
}

func TestLlamaCppRunnerDoesNotPostChunksWhenStreamingFlagDisabled(t *testing.T) {
	client := &fakeCompletionClient{
		result: llamacpp.CompletionResult{
			Output:          []byte("non streaming output"),
			OutputBytes:     int64(len("non streaming output")),
			TokensGenerated: 3,
			TTFTMs:          300,
			TotalTimeMs:     300,
		},
		deltas: []string{"should not post"},
	}
	progress := &fakeProgressSender{}
	spec := validSpec(t)
	spec.Prompt = "Write a short dashboard sentence."
	spec.ReturnText = true
	runner := LlamaCppRunner{
		Sidecar: &fakeSidecar{status: healthySidecarStatus()},
		Client:  client,
		Getenv:  getenvStreamingDisabled,
	}
	result, err := runner.RunDashboardInferenceWithProgress(context.Background(), spec, progress)
	if err != nil {
		t.Fatalf("RunDashboardInferenceWithProgress() error = %v", err)
	}
	if len(client.reqs) != 1 || client.reqs[0].Stream {
		t.Fatalf("stream requests = %+v, want one non-streaming request", client.reqs)
	}
	if len(progress.batches) != 0 {
		t.Fatalf("progress batches = %+v, want none", progress.batches)
	}
	if result.ProofStatus != ProofStatusMeasured {
		t.Fatalf("result = %+v, want measured result", result)
	}
}

func TestLlamaCppRunnerDoesNotPostChunksWhenSpecStreamFalse(t *testing.T) {
	client := &fakeCompletionClient{
		result: llamacpp.CompletionResult{
			Output:          []byte("non streaming output"),
			OutputBytes:     int64(len("non streaming output")),
			TokensGenerated: 3,
			TTFTMs:          300,
			TotalTimeMs:     300,
		},
		deltas: []string{"should not post"},
	}
	progress := &fakeProgressSender{}
	spec := validSpec(t)
	spec.Prompt = "Write a short dashboard sentence."
	spec.ReturnText = true
	spec.Stream = false
	runner := LlamaCppRunner{
		Sidecar: &fakeSidecar{status: healthySidecarStatus()},
		Client:  client,
		Getenv:  getenvTextAndStreamingEnabled,
	}
	result, err := runner.RunDashboardInferenceWithProgress(context.Background(), spec, progress)
	if err != nil {
		t.Fatalf("RunDashboardInferenceWithProgress() error = %v", err)
	}
	if len(client.reqs) != 1 || client.reqs[0].Stream {
		t.Fatalf("stream requests = %+v, want one non-streaming request", client.reqs)
	}
	if len(progress.batches) != 0 {
		t.Fatalf("progress batches = %+v, want none", progress.batches)
	}
	if result.ProofStatus != ProofStatusMeasured {
		t.Fatalf("result = %+v, want measured result", result)
	}
}

func TestLlamaCppRunnerProgressFailureKeepsMeasuredReceipt(t *testing.T) {
	client := &fakeCompletionClient{
		result: llamacpp.CompletionResult{
			Output:          []byte("private streamed output"),
			OutputBytes:     int64(len("private streamed output")),
			TokensGenerated: 3,
			TTFTMs:          40,
			TotalTimeMs:     240,
		},
		deltas: []string{"private", " streamed", " output"},
	}
	progress := &fakeProgressSender{err: errors.New("secret private streamed output")}
	spec := validSpec(t)
	spec.Prompt = "private prompt"
	spec.ReturnText = true
	runner := LlamaCppRunner{
		Sidecar: &fakeSidecar{status: healthySidecarStatus()},
		Client:  client,
		Getenv:  getenvTextAndStreamingEnabled,
	}
	result, err := runner.RunDashboardInferenceWithProgress(context.Background(), spec, progress)
	if err != nil {
		t.Fatalf("RunDashboardInferenceWithProgress() error = %v", err)
	}
	if result.ProofStatus != ProofStatusMeasured || result.ErrorCode != "" {
		t.Fatalf("result = %+v, want measured result despite best-effort progress failure", result)
	}
	if result.GeneratedText != "private streamed output" {
		t.Fatalf("generated text = %q, want final measured output", result.GeneratedText)
	}
}

func TestExecuteAssignmentReturnTextTruncatesDeterministically(t *testing.T) {
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{
		Output:          []byte("abcdef"),
		OutputBytes:     6,
		TokensGenerated: 1,
		TTFTMs:          1,
		TotalTimeMs:     2,
	}}
	spec := validSpec(t)
	spec.Prompt = "Write six letters."
	spec.ReturnText = true
	spec.MaxReturnChars = 3
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	receipt, _, err := ExecuteAssignment(context.Background(), string(raw), ExecuteOptions{
		Getenv: getenvTextOutputEnabled,
		Runner: LlamaCppRunner{
			Sidecar: &fakeSidecar{status: healthySidecarStatus()},
			Client:  client,
		},
	})
	if err != nil {
		t.Fatalf("ExecuteAssignment() error = %v", err)
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["generated_text"] != "abc" || metadata["generated_text_truncated"] != true {
		t.Fatalf("truncated generated metadata = %+v", metadata)
	}
	if metadata["output_hash"] != HashOutput(spec.JobID, []byte("abcdef")) {
		t.Fatalf("output_hash = %v, want hash over full output", metadata["output_hash"])
	}
}

func TestExecuteAssignmentMaxTokenCapMetadataDoesNotMeanDisplayTruncated(t *testing.T) {
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{
		Output:          []byte("partial wor"),
		OutputBytes:     int64(len("partial wor")),
		TokensGenerated: 8,
		TTFTMs:          12,
		TotalTimeMs:     112,
	}}
	spec := validSpec(t)
	spec.MaxTokens = 8
	spec.Prompt = "Write a short dashboard sentence."
	spec.ReturnText = true
	spec.MaxReturnChars = defaultMaxReturnChars
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	receipt, _, err := ExecuteAssignment(context.Background(), string(raw), ExecuteOptions{
		Getenv: getenvTextOutputEnabled,
		Runner: LlamaCppRunner{
			Sidecar: &fakeSidecar{status: healthySidecarStatus()},
			Client:  client,
		},
	})
	if err != nil {
		t.Fatalf("ExecuteAssignment() error = %v", err)
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["requested_max_tokens"] != int(8) || metadata["tokens_generated"] != int64(8) || metadata["max_tokens_reached"] != true {
		t.Fatalf("max token metadata = %+v", metadata)
	}
	if metadata["finish_reason"] != llamacpp.FinishReasonLength {
		t.Fatalf("finish_reason = %#v, want length when max token cap is reached", metadata["finish_reason"])
	}
	if metadata["generated_text"] != "partial wor" || metadata["generated_text_truncated"] != false {
		t.Fatalf("display truncation metadata = %+v", metadata)
	}
	if metadata["max_return_chars"] != int(defaultMaxReturnChars) {
		t.Fatalf("max_return_chars = %#v", metadata["max_return_chars"])
	}
}

func TestExecuteAssignmentBackendNaturalStopMetadata(t *testing.T) {
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{
		Output:              []byte("Ryvion is ready."),
		OutputBytes:         int64(len("Ryvion is ready.")),
		TokensGenerated:     3,
		FinishReason:        llamacpp.FinishReasonStop,
		BackendFinishReason: llamacpp.FinishReasonStop,
		TTFTMs:              15,
		TotalTimeMs:         115,
	}}
	spec := validSpec(t)
	spec.Prompt = "Write a short dashboard sentence."
	spec.ReturnText = true
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	receipt, _, err := ExecuteAssignment(context.Background(), string(raw), ExecuteOptions{
		Getenv: getenvTextOutputEnabled,
		Runner: LlamaCppRunner{
			Sidecar: &fakeSidecar{status: healthySidecarStatus()},
			Client:  client,
		},
	})
	if err != nil {
		t.Fatalf("ExecuteAssignment() error = %v", err)
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["finish_reason"] != llamacpp.FinishReasonStop || metadata["backend_finish_reason"] != llamacpp.FinishReasonStop || metadata["max_tokens_reached"] != false {
		t.Fatalf("natural stop metadata = %+v", metadata)
	}
}

func TestExecuteAssignmentMissingBackendTokenCountUsesPartialEstimate(t *testing.T) {
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{
		Output:      []byte("fallback token estimate"),
		OutputBytes: int64(len("fallback token estimate")),
		TTFTMs:      40,
		TotalTimeMs: 340,
	}}
	spec := validSpec(t)
	spec.Prompt = "Write a short dashboard sentence."
	spec.ReturnText = true
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	receipt, handled, err := ExecuteAssignment(context.Background(), string(raw), ExecuteOptions{
		Getenv: getenvTextOutputEnabled,
		Runner: LlamaCppRunner{
			Sidecar: &fakeSidecar{status: healthySidecarStatus()},
			Client:  client,
		},
	})
	if err != nil {
		t.Fatalf("ExecuteAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["proof_status"] != ProofStatusMeasured || metadata["tokens_generated"] != int64(3) {
		t.Fatalf("fallback token metadata = %+v", metadata)
	}
	if metadata["runtime_measurement_status"] != llamacpp.RuntimeMeasurementStatusPartial || metadata["metadata_parse_status"] != llamacpp.MetadataParseStatusOK || metadata["token_count_estimated"] != true {
		t.Fatalf("fallback measurement status = %+v", metadata)
	}
	if metadata["p50_ttft_ms"] != int64(40) || metadata["p50_decode_tps"] != float64(10) {
		t.Fatalf("fallback runtime metrics = %+v", metadata)
	}
	if metadata["generated_text"] != "fallback token estimate" {
		t.Fatalf("generated_text = %#v", metadata["generated_text"])
	}
}

func TestBuildReceiptNoTokenCountMarksUnknownWithoutFailing(t *testing.T) {
	spec := validSpec(t)
	spec.ReturnText = false
	receipt, err := BuildReceipt(ExecutionResult{
		Spec:               spec,
		Backend:            spec.Backend,
		ModelID:            spec.ModelID,
		OutputHash:         HashOutput(spec.JobID, nil),
		RequestedMaxTokens: spec.MaxTokens,
		TokensGenerated:    0,
		FinishReason:       llamacpp.FinishReasonUnknown,
		ProofStatus:        ProofStatusMeasured,
		MaxReturnChars:     spec.MaxReturnChars,
	})
	if err != nil {
		t.Fatalf("BuildReceipt() error = %v", err)
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["tokens_generated"] != int64(0) {
		t.Fatalf("tokens_generated = %#v, want explicit zero", metadata["tokens_generated"])
	}
	if metadata["runtime_measurement_status"] != llamacpp.RuntimeMeasurementStatusUnknown || metadata["metadata_parse_status"] != llamacpp.MetadataParseStatusPartial {
		t.Fatalf("unknown measurement metadata = %+v", metadata)
	}
	if metadata["proof_status"] != ProofStatusMeasured || metadata["result_hash_hex"] != receipt.ResultHashHex {
		t.Fatalf("stable receipt metadata = %+v", metadata)
	}
}

func TestExecuteAssignmentFlagDisabledReturnsSafeRejection(t *testing.T) {
	client := &fakeCompletionClient{}
	receipt, handled, err := ExecuteAssignment(context.Background(), validSpecJSON(t), ExecuteOptions{
		Getenv: getenvInferenceDisabled,
		Runner: LlamaCppRunner{
			Sidecar: &fakeSidecar{status: healthySidecarStatus()},
			Client:  client,
		},
	})
	if err == nil {
		t.Fatal("ExecuteAssignment() error = nil, want disabled error")
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if len(client.reqs) != 0 {
		t.Fatalf("client calls = %d, want 0", len(client.reqs))
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["proof_status"] != ProofStatusRejected || metadata["error_code"] != "dashboard_inference_disabled" {
		t.Fatalf("disabled metadata = %+v", metadata)
	}
	if !ReceiptJSONContainsNoRawText(receipt) {
		raw, _ := json.Marshal(receipt.Metadata)
		t.Fatalf("disabled metadata leaked raw text: %s", raw)
	}
}

func TestExecuteAssignmentSidecarUnavailableReturnsSafeRejection(t *testing.T) {
	status := healthySidecarStatus()
	status.Available = false
	status.Running = false
	status.Healthy = false
	receipt, handled, err := ExecuteAssignment(context.Background(), validSpecJSON(t), ExecuteOptions{
		Getenv: getenvEnabled,
		Runner: LlamaCppRunner{
			Sidecar: &fakeSidecar{status: status},
			Client:  &fakeCompletionClient{},
		},
	})
	if err != nil {
		t.Fatalf("ExecuteAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["proof_status"] != ProofStatusRejected || metadata["error_code"] != "llamacpp_sidecar_unavailable" {
		t.Fatalf("sidecar unavailable metadata = %+v", metadata)
	}
	if metadata["output_hash"] == "" || metadata["output_bytes"] != int64(0) {
		t.Fatalf("safe rejection output metadata = %+v", metadata)
	}
}

func TestLlamaCppRunnerRejectsPhi4WhenRuntimePolicyBlocksFamily(t *testing.T) {
	status := healthySidecarStatus()
	status.ModelPath = "/models/phi-4-Q4_K_M.gguf"
	status.ModelFilename = "phi-4-Q4_K_M.gguf"
	status.ModelFamilyHint = "phi"
	status.ModelSizeBytes = 4 * 1024 * 1024 * 1024
	client := &fakeCompletionClient{result: llamacpp.CompletionResult{Output: []byte("should not run")}}
	spec := validSpec(t)
	spec.ModelID = "phi-4-Q4_K_M.gguf"
	spec.Prompt = "Write a short dashboard sentence."
	spec.ReturnText = true
	progress := &fakeProgressSender{}
	runner := LlamaCppRunner{
		Sidecar: &fakeSidecar{status: status},
		Client:  client,
		Getenv:  getenvTextAndStreamingEnabled,
		Policy: modelpolicy.Policy{
			CacheDir:            "/models",
			MaxSingleModelBytes: 8 * 1024 * 1024 * 1024,
			MaxCacheBytes:       50 * 1024 * 1024 * 1024,
			AllowedFamilies:     []string{"llama", "phi"},
			AllowedFormats:      []string{"gguf"},
			RuntimePolicy: modelpolicy.RuntimePolicy{
				AllowRuntimeExecution:            true,
				MaxRuntimeModelBytes:             8 * 1024 * 1024 * 1024,
				MaxRuntimeParameterCountBillions: 8,
				AllowCPUOffload:                  true,
				AllowFamilies:                    []string{"llama"},
			},
		},
	}
	result, err := runner.RunDashboardInferenceWithProgress(context.Background(), spec, progress)
	if err != nil {
		t.Fatalf("RunDashboardInference() error = %v", err)
	}
	if result.ProofStatus != ProofStatusRejected || result.ErrorCode != modelpolicy.RuntimeDecisionFamilyNotAllowed {
		t.Fatalf("result = %+v, want runtime policy family rejection", result)
	}
	if len(client.reqs) != 0 {
		t.Fatalf("client calls = %d, want 0", len(client.reqs))
	}
	if len(progress.batches) != 0 {
		t.Fatalf("progress batches = %+v, want none", progress.batches)
	}
}

func TestExecuteAssignmentUnsupportedBackendRejectedSafely(t *testing.T) {
	spec := validSpec(t)
	spec.Backend = "openai"
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	receipt, handled, err := ExecuteAssignment(context.Background(), string(raw), ExecuteOptions{Getenv: getenvEnabled})
	if err == nil {
		t.Fatal("ExecuteAssignment() error = nil, want unsupported backend error")
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	metadata := receipt.Metadata[Task].(map[string]any)
	if metadata["proof_status"] != ProofStatusRejected {
		t.Fatalf("proof_status = %v", metadata["proof_status"])
	}
	if _, ok := metadata["error_code"].(string); !ok {
		t.Fatalf("metadata missing error_code: %+v", metadata)
	}
}

func TestExecuteAssignmentNonDashboardTaskIgnored(t *testing.T) {
	receipt, handled, err := ExecuteAssignment(context.Background(), `{"task":"other"}`, ExecuteOptions{Getenv: getenvEnabled})
	if err != nil {
		t.Fatalf("ExecuteAssignment() error = %v", err)
	}
	if handled {
		t.Fatal("handled = true, want false")
	}
	if receipt.JobID != "" {
		t.Fatalf("receipt = %+v, want empty", receipt)
	}
}

func TestLlamaCppRunnerFallsBackWhenStreamingUnsupported(t *testing.T) {
	client := &fakeCompletionClient{err: llamacpp.ClientError{Code: "llamacpp_stream_unavailable"}}
	runner := LlamaCppRunner{
		Sidecar: &fakeSidecar{status: healthySidecarStatus()},
		Client:  client,
		Getenv:  getenvTextAndStreamingEnabled,
	}
	spec := validSpec(t)
	spec.Prompt = "Write a short dashboard sentence."
	spec.ReturnText = true
	result, err := runner.RunDashboardInference(context.Background(), spec)
	if err != nil {
		t.Fatalf("RunDashboardInference() error = %v", err)
	}
	if result.ProofStatus != ProofStatusRejected || result.ErrorCode != "llamacpp_stream_unavailable" {
		t.Fatalf("result = %+v, want safe rejection", result)
	}
	if len(client.reqs) != 2 {
		t.Fatalf("client calls = %d, want stream attempt and fallback", len(client.reqs))
	}
	if !client.reqs[0].Stream || client.reqs[1].Stream {
		t.Fatalf("stream flags = %+v", []bool{client.reqs[0].Stream, client.reqs[1].Stream})
	}
}

func TestErrorCodeRedactsUnsafeErrorText(t *testing.T) {
	got := ErrorCode(errors.New("secret raw_output should not leak"))
	if got != "dashboard_inference_error_redacted" {
		t.Fatalf("ErrorCode() = %q", got)
	}
}

func validSpecJSON(t *testing.T) string {
	t.Helper()
	raw, err := json.Marshal(validSpec(t))
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	return string(raw)
}

func validSpec(t *testing.T) Spec {
	t.Helper()
	return Spec{
		Task:            Task,
		RequestID:       "dashboardinfer_request",
		RunID:           "dashboardinfer_run",
		JobID:           "v7dashboardinfer_job",
		Backend:         llamacpp.BackendName,
		ModelID:         "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		TargetNodeID:    "node-local",
		MaxTokens:       32,
		Stream:          true,
		CreatedAtUnixMs: 1_800_000_000_123,
		PromptHash:      testSHA256("dashboard prompt profile"),
		PromptProfileID: "dashboard_default",
	}
}

func healthySidecarStatus() llamacpp.LlamaCppSidecarStatus {
	return llamacpp.LlamaCppSidecarStatus{
		Enabled:                true,
		Available:              true,
		Running:                true,
		Healthy:                true,
		BaseURL:                "http://127.0.0.1:45910",
		ModelPath:              "/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		ModelFilename:          "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		Backend:                llamacpp.BackendName,
		SupportsTextGeneration: true,
		SupportsStreaming:      true,
	}
}

func healthySidecarStatusForModel(modelID string) llamacpp.LlamaCppSidecarStatus {
	status := healthySidecarStatus()
	status.ModelPath = filepath.Join("/models", modelID)
	status.ModelFilename = modelID
	return status
}

func getenvEnabled(key string) string {
	if key == FlagEnv {
		return "1"
	}
	return ""
}

func getenvInferenceDisabled(key string) string {
	if key == DisableFlagEnv {
		return "1"
	}
	return ""
}

func getenvTextOutputDisabled(key string) string {
	if key == DisableTextEnv {
		return "1"
	}
	return ""
}

func getenvStreamingDisabled(key string) string {
	if key == DisableStreamEnv {
		return "1"
	}
	return ""
}

func getenvTextOutputEnabled(key string) string {
	switch key {
	case FlagEnv, TextOutputFlagEnv:
		return "1"
	default:
		return ""
	}
}

func getenvTextAndStreamingEnabled(key string) string {
	switch key {
	case FlagEnv, TextOutputFlagEnv, StreamingFlagEnv:
		return "1"
	default:
		return ""
	}
}

func testSHA256(value string) string {
	sum := sha256.Sum256([]byte(value))
	return "sha256:" + hex.EncodeToString(sum[:])
}
