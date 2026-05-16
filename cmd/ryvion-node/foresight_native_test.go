package main

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/Ryvion/node-agent/internal/hub"
	v7llamacpp "github.com/Ryvion/node-agent/internal/v7/llamacpp"
)

func TestDecodeForesightNativeDraftSpecBuildsPackets(t *testing.T) {
	specJSON := `{
		"task":"draft_runner_v8",
		"draft_backend":"native_bridge",
		"workgraph_id":"wg-live",
		"window_id":"win-live",
		"role_id":"draft-worker-live",
		"target_node_id":"node-drafter",
		"prompt":"Write one short sentence.",
		"parent_prefix_hash":"sha256:prefix",
		"branch_count":3,
		"horizon":4,
		"deadline_ms":1000,
		"model_hash":"sha256:model",
		"drafter_model_id":"native-test"
	}`
	spec, ok := decodeForesightNativeDraftSpec(specJSON)
	if !ok {
		t.Fatal("decodeForesightNativeDraftSpec ok = false")
	}
	if spec.DraftBackend != "native_bridge" {
		t.Fatalf("DraftBackend = %q, want native_bridge", spec.DraftBackend)
	}
	packets := buildForesightNativeDraftPackets(spec)
	if len(packets) != 3 {
		t.Fatalf("len(packets) = %d, want 3", len(packets))
	}
	seen := map[string]bool{}
	for _, packet := range packets {
		if packet["window_id"] != "win-live" || packet["workgraph_id"] != "wg-live" ||
			packet["role_id"] != "draft-worker-live" ||
			packet["parent_prefix_hash"] != "sha256:prefix" ||
			packet["model_hash"] != "sha256:model" {
			t.Fatalf("packet identity fields = %#v", packet)
		}
		tokens, ok := packet["candidate_tokens"].([]int)
		if !ok || len(tokens) != 4 {
			t.Fatalf("candidate_tokens = %#v, want 4 ints", packet["candidate_tokens"])
		}
		encoded, _ := json.Marshal(tokens)
		key := string(encoded)
		if seen[key] {
			t.Fatalf("duplicate token branch %s", key)
		}
		seen[key] = true
	}
}

func TestDecodeForesightNativeVerifierSpecAcceptsTree(t *testing.T) {
	specJSON := `{
		"task":"verifier_session_v8",
		"tree":{
			"tree_cid":"sha256:tree",
			"branches":[
				{"candidate_tokens":[1,2,3]},
				{"candidate_tokens":[4,5,6,7,8,9,10,11,12]}
			]
		}
	}`
	accepted, treeCID, backend, ok := decodeForesightNativeVerifierSpec(specJSON)
	if !ok {
		t.Fatal("decodeForesightNativeVerifierSpec ok = false")
	}
	if backend != "" {
		t.Fatalf("backend = %q, want empty bridge default", backend)
	}
	if accepted != 8 {
		t.Fatalf("accepted = %d, want capped 8", accepted)
	}
	if treeCID != "sha256:tree" {
		t.Fatalf("treeCID = %q, want sha256:tree", treeCID)
	}
}

func TestDecodeForesightNativeHotSessionSpecPreservesNativeSGLangFields(t *testing.T) {
	specJSON := `{
		"task":"verifier_session_v8_hot",
		"executor_kind":"native_report",
		"docker_required":false,
		"verifier_backend":"native_sglang",
		"run_id":"flab-native",
		"session_id":"sess-native",
		"workgraph_id":"wg-native",
		"target_node_id":"node-verifier",
		"model_id":"nemotron",
		"model_path":"/models/nemotron",
		"max_tokens":128
	}`
	spec, ok := decodeForesightNativeHotSessionSpec(specJSON, foresightVerifierHotSessionTask)
	if !ok {
		t.Fatal("decodeForesightNativeHotSessionSpec ok = false")
	}
	if spec.VerifierBackend != foresightVerifierBackendSGLang {
		t.Fatalf("VerifierBackend = %q, want %q", spec.VerifierBackend, foresightVerifierBackendSGLang)
	}
	if spec.ModelPath != "/models/nemotron" {
		t.Fatalf("ModelPath = %q", spec.ModelPath)
	}
	if foresightVerifierBackendKind(spec.VerifierBackend) != foresightVerifierBackendSGLang {
		t.Fatalf("verifier backend kind did not select native SGLang")
	}
}

func TestForesightVerifierBackendKindAcceptsNativeLlamaCpp(t *testing.T) {
	for _, backend := range []string{"native_llamacpp", "llamacpp", "llama.cpp", "llama_cpp"} {
		if got := foresightVerifierBackendKind(backend); got != foresightVerifierBackendLlamaCpp {
			t.Fatalf("foresightVerifierBackendKind(%q) = %q, want %q", backend, got, foresightVerifierBackendLlamaCpp)
		}
	}
}

func TestNativeLlamaCppVerifierWaveUsesMeasuredCompletion(t *testing.T) {
	sidecar := &fakeForesightLlamaCppSidecar{status: v7llamacpp.LlamaCppSidecarStatus{
		Enabled:                true,
		Available:              true,
		Running:                true,
		Healthy:                true,
		BaseURL:                "http://127.0.0.1:45910",
		ModelFilename:          "tinyllama.Q4_K_M.gguf",
		Backend:                v7llamacpp.BackendName,
		OpenAICompatible:       true,
		SupportsTextGeneration: true,
		SupportsStreaming:      true,
	}}
	client := &fakeForesightLlamaCppCompletionClient{result: v7llamacpp.CompletionResult{
		Output:                   []byte("Verified local text."),
		OutputBytes:              int64(len("Verified local text.")),
		TokensGenerated:          4,
		CompletionTokens:         4,
		RequestedMaxTokens:       8,
		FinishReason:             "stop",
		RuntimeMeasurementStatus: v7llamacpp.RuntimeMeasurementStatusMeasured,
		MetadataParseStatus:      v7llamacpp.MetadataParseStatusOK,
		TotalTimeMs:              25,
		Streamed:                 true,
	}}
	verifier := nativeLlamaCppVerifier{Sidecar: sidecar, Client: client}

	result, err := verifier.VerifyWave(context.Background(), foresightNativeHotSessionSpec{
		RunID:            "flab_llama",
		SessionID:        "sess_llama",
		WorkGraphID:      "wg_llama",
		ModelID:          "tinyllama",
		Prompt:           "Write one short sentence.",
		ParentPrefixHash: "sha256:prefix",
		MaxTokens:        8,
	}, hub.ForesightLiveLabSessionCommand{
		CommandID:   "cmd_1",
		WindowID:    "win_llama",
		WaveIndex:   1,
		WorkGraphID: "wg_llama",
		Tree: map[string]any{
			"tree_cid": "sha256:tree",
			"branches": []any{
				map[string]any{"candidate_tokens": []any{float64(1), float64(2), float64(3), float64(4), float64(5)}},
			},
		},
	}, 0)
	if err != nil {
		t.Fatalf("VerifyWave() error = %v, want nil", err)
	}
	if !sidecar.started {
		t.Fatal("VerifyWave() did not start llama.cpp sidecar")
	}
	if len(client.requests) != 1 {
		t.Fatalf("completion requests = %d, want 1", len(client.requests))
	}
	req := client.requests[0]
	if req.BaseURL != "http://127.0.0.1:45910" || req.ModelID != "tinyllama" || !req.Stream {
		t.Fatalf("completion request = %+v, want hot llama.cpp request", req)
	}
	if req.Prompt == "" || !strings.Contains(req.Prompt, "sha256:tree") || !strings.Contains(req.Prompt, "branch_count=1") {
		t.Fatalf("completion prompt = %q, want verifier tree summary", req.Prompt)
	}
	if result.AcceptedLen != 4 || result.TreeCID != "sha256:tree" || result.AcceptedText != "Verified local text." || !result.AcceptedTextPublic {
		t.Fatalf("verifier result = %+v, want measured llama.cpp acceptance", result)
	}
	if result.ProbeSummary["source"] != "native_llamacpp_verifier" ||
		result.ProbeSummary["backend"] != v7llamacpp.BackendName ||
		result.ProbeSummary["output_hash"] == "" {
		t.Fatalf("probe summary = %#v, want llama.cpp measured metadata", result.ProbeSummary)
	}
	if encoded, _ := json.Marshal(result.ProbeSummary); strings.Contains(string(encoded), "Write one short sentence") || strings.Contains(string(encoded), "Verified local text") {
		t.Fatalf("probe summary leaked raw prompt/output: %s", encoded)
	}
}

func TestNativeLlamaCppVerifierUnavailableDoesNotUseDeterministicCPUFallback(t *testing.T) {
	sidecar := &fakeForesightLlamaCppSidecar{status: v7llamacpp.LlamaCppSidecarStatus{
		Enabled:   true,
		Available: false,
		Backend:   v7llamacpp.BackendName,
		Reason:    "llama-server binary not detected",
	}}
	client := &fakeForesightLlamaCppCompletionClient{}
	verifier := nativeLlamaCppVerifier{Sidecar: sidecar, Client: client}

	_, err := verifier.VerifyWave(context.Background(), foresightNativeHotSessionSpec{
		RunID:       "flab_llama",
		WorkGraphID: "wg_llama",
		ModelID:     "tinyllama",
	}, hub.ForesightLiveLabSessionCommand{
		CommandID: "cmd_1",
		WindowID:  "win_llama",
		Tree:      map[string]any{"tree_cid": "sha256:tree"},
	}, 0)
	if !errors.Is(err, errNativeLlamaCppUnavailable) {
		t.Fatalf("VerifyWave() error = %v, want errNativeLlamaCppUnavailable", err)
	}
	if len(client.requests) != 0 {
		t.Fatalf("completion requests = %d, want 0 when llama.cpp unavailable", len(client.requests))
	}
}

func TestForesightNativeExternalRuntimeRequestedSkipsManagedOCI(t *testing.T) {
	work := &hub.WorkAssignment{
		Image:        "ghcr.io/ryvion/sglang-verifier-runner-v8:0.1.0",
		ExecutorKind: executorKindManagedOCI,
	}
	if !foresightNativeExternalRuntimeRequested(work, executorKindManagedOCI, work.Image, true) {
		t.Fatal("managed OCI verifier job should not be claimed by native CPU bridge")
	}
	nativeWork := &hub.WorkAssignment{Image: executorKindNativeReport, ExecutorKind: executorKindNativeReport}
	if foresightNativeExternalRuntimeRequested(nativeWork, executorKindNativeReport, "", false) {
		t.Fatal("native report job should be claimable by native Foresight handlers")
	}
}

func TestResolveNativeSGLangVerifierCommandFromEnv(t *testing.T) {
	t.Setenv("RYV_SGLANG_VERIFIER_CMD", "python /opt/ryvion/sglang-verifier/run.py")
	command, ok := resolveNativeSGLangVerifierCommand()
	if !ok {
		t.Fatal("resolveNativeSGLangVerifierCommand ok = false")
	}
	if !command.Shell || command.Original == "" {
		t.Fatalf("command = %+v, want shell command from env", command)
	}
}

type fakeForesightLlamaCppSidecar struct {
	status  v7llamacpp.LlamaCppSidecarStatus
	started bool
}

func (f *fakeForesightLlamaCppSidecar) Start(context.Context) v7llamacpp.LlamaCppSidecarStatus {
	f.started = true
	return f.status
}

func (f *fakeForesightLlamaCppSidecar) Status(context.Context) v7llamacpp.LlamaCppSidecarStatus {
	return f.status
}

type fakeForesightLlamaCppCompletionClient struct {
	result   v7llamacpp.CompletionResult
	err      error
	requests []v7llamacpp.CompletionRequest
}

func (f *fakeForesightLlamaCppCompletionClient) Complete(_ context.Context, req v7llamacpp.CompletionRequest) (v7llamacpp.CompletionResult, error) {
	f.requests = append(f.requests, req)
	if f.err != nil {
		return v7llamacpp.CompletionResult{}, f.err
	}
	return f.result, nil
}
