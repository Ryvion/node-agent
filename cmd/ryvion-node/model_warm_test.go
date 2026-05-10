package main

import (
	"context"
	"crypto/ed25519"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/Ryvion/node-agent/internal/diagnostics"
	"github.com/Ryvion/node-agent/internal/hub"
	v7llamacpp "github.com/Ryvion/node-agent/internal/v7/llamacpp"
	v7modelwarm "github.com/Ryvion/node-agent/internal/v7/modelwarm"
)

func TestProcessOptionalV7ModelWarmSubmitsReceiptForWarmRuntime(t *testing.T) {
	t.Setenv(v7modelwarm.WarmFlagEnv, "1")
	cacheDir := t.TempDir()
	t.Setenv("RYV_MODEL_CACHE_DIR", cacheDir)
	modelPath := filepath.Join(cacheDir, "phi-4-Q4_K_M.gguf")
	if err := os.WriteFile(modelPath, []byte("gguf"), 0o644); err != nil {
		t.Fatalf("write cached model: %v", err)
	}

	llamaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health", "/v1/models":
			w.WriteHeader(http.StatusOK)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer llamaServer.Close()

	oldState := operatorRuntimeState
	oldDiagnostics := workLoopDiagnostics
	status := v7modelwarm.NewLocalStatus()
	operatorRuntimeState = &operatorRuntime{
		version:           "test",
		hubURL:            "https://api.ryvion.ai",
		deviceType:        "gpu",
		publicKeyHex:      "abc123",
		v7ModelWarm:       status,
		llamaCppSidecar:   testLlamaCppManagerForServerModel(t, llamaServer.URL, modelPath),
		llamaCppKeeper:    nil,
		llamaCppBenchmark: v7llamacpp.NewBenchmarkLocalStatus(),
	}
	workLoopDiagnostics = diagnostics.NewWorkLoopDiagnostics()
	t.Cleanup(func() {
		_ = operatorRuntimeState.stopLlamaCppSidecar(context.Background())
		operatorRuntimeState = oldState
		workLoopDiagnostics = oldDiagnostics
	})

	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	var receiptCalls atomic.Int32
	receiptServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/node/receipt" {
			t.Errorf("unexpected path: %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		receiptCalls.Add(1)
		var req struct {
			JobID         string         `json:"job_id"`
			ResultHashHex string         `json:"result_hash_hex"`
			Metadata      map[string]any `json:"metadata"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode receipt: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if req.JobID != "job-warm-local" || len(req.ResultHashHex) != 64 {
			t.Errorf("receipt envelope = %+v", req)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		metadata, ok := req.Metadata[v7modelwarm.WarmTask].(map[string]any)
		if !ok {
			t.Errorf("receipt metadata missing %q: %+v", v7modelwarm.WarmTask, req.Metadata)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		if metadata["warm_id"] != "warm-local" ||
			metadata["model_id"] != "phi-4-Q4_K_M.gguf" ||
			metadata["backend"] != "llama.cpp" ||
			metadata["model_path"] != modelPath ||
			metadata["warm"] != true ||
			metadata["proof_status"] != v7modelwarm.ProofStatusModelWarmed {
			t.Errorf("warm metadata = %+v", metadata)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		encoded, _ := json.Marshal(req.Metadata)
		for _, forbidden := range []string{"raw_prompt", "prompt_text", "output_text", "generated_text", "raw_output", "tensor_bytes", "private_key"} {
			if strings.Contains(strings.ToLower(string(encoded)), forbidden) {
				t.Errorf("metadata leaked forbidden material %q: %s", forbidden, encoded)
				w.WriteHeader(http.StatusBadRequest)
				return
			}
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer receiptServer.Close()

	work := &hub.WorkAssignment{
		JobID:    "job-warm-local",
		Kind:     "model_warm",
		SpecJSON: testModelWarmSpecJSON(t),
	}
	handled, result, err := processOptionalV7ModelWarm(context.Background(), hub.New(receiptServer.URL, pub, priv), work, nil, true)
	if err != nil {
		t.Fatalf("processOptionalV7ModelWarm() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if result == nil || result.ResultHashHex == "" || result.ExitCode != 0 {
		t.Fatalf("result = %+v, want successful warm receipt", result)
	}
	if receiptCalls.Load() != 1 {
		t.Fatalf("receipt calls = %d, want 1", receiptCalls.Load())
	}
	snapshot := status.Snapshot()
	if snapshot.LastWarmID != "warm-local" || snapshot.LastModelID != "phi-4-Q4_K_M.gguf" || snapshot.LastError != "" {
		t.Fatalf("status snapshot = %+v", snapshot)
	}
	if snapshot.Counters.Seen != 1 || snapshot.Counters.Executed != 1 || snapshot.Counters.ReceiptSubmitted != 1 || snapshot.Counters.ReceiptFailed != 0 {
		t.Fatalf("status counters = %+v", snapshot.Counters)
	}
	runtime := operatorRuntimeState.backendRuntimesStatus(context.Background()).LlamaCPP
	if !runtime.Warm || runtime.ModelID != "phi-4-Q4_K_M.gguf" || runtime.ModelPath != modelPath {
		t.Fatalf("backend runtime = %+v, want only phi model warm", runtime)
	}
	workSnapshot := workLoopDiagnostics.Snapshot()
	if workSnapshot.ReceiptSubmittedCount != 1 || workSnapshot.ReceiptFailedCount != 0 {
		t.Fatalf("unexpected work-loop receipt counters: %+v", workSnapshot)
	}
}

func testModelWarmSpecJSON(t *testing.T) string {
	t.Helper()
	spec := v7modelwarm.WarmSpec{
		Task:                  v7modelwarm.WarmTask,
		RequestID:             "request-local",
		WarmID:                "warm-local",
		JobID:                 "job-warm-local",
		ModelID:               "phi-4-Q4_K_M.gguf",
		Backend:               "llama.cpp",
		RunBenchmarkAfterWarm: false,
		TimeoutMs:             30_000,
	}
	encoded, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	return string(encoded)
}
