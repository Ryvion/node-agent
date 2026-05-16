package main

import (
	"encoding/json"
	"testing"

	"github.com/Ryvion/node-agent/internal/hub"
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
