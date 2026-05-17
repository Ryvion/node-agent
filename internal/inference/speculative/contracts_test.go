package speculative

import (
	"encoding/json"
	"testing"
)

func TestDecodeDraftSpecBuildsPackets(t *testing.T) {
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
	spec, ok := DecodeDraftSpec(specJSON)
	if !ok {
		t.Fatal("DecodeDraftSpec ok = false")
	}
	if spec.DraftBackend != DraftBackendNative {
		t.Fatalf("DraftBackend = %q, want %q", spec.DraftBackend, DraftBackendNative)
	}
	packets := BuildDraftPackets(spec)
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

func TestDecodeVerifierSpecAcceptsTree(t *testing.T) {
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
	accepted, treeCID, backend, ok := DecodeVerifierSpec(specJSON)
	if !ok {
		t.Fatal("DecodeVerifierSpec ok = false")
	}
	if backend != VerifierBackendBridge {
		t.Fatalf("backend = %q, want %q", backend, VerifierBackendBridge)
	}
	if accepted != 8 {
		t.Fatalf("accepted = %d, want capped 8", accepted)
	}
	if treeCID != "sha256:tree" {
		t.Fatalf("treeCID = %q, want sha256:tree", treeCID)
	}
}

func TestDecodeHotSessionSpecPreservesNativeSGLangFields(t *testing.T) {
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
	spec, ok := DecodeHotSessionSpec(specJSON, VerifierHotSessionTask)
	if !ok {
		t.Fatal("DecodeHotSessionSpec ok = false")
	}
	if spec.VerifierBackend != VerifierBackendSGLang {
		t.Fatalf("VerifierBackend = %q, want %q", spec.VerifierBackend, VerifierBackendSGLang)
	}
	if spec.ModelPath != "/models/nemotron" {
		t.Fatalf("ModelPath = %q", spec.ModelPath)
	}
	if VerifierBackendKind(spec.VerifierBackend) != VerifierBackendSGLang {
		t.Fatal("VerifierBackendKind did not select native SGLang")
	}
}

func TestVerifierBackendKindAcceptsNativeLlamaCpp(t *testing.T) {
	for _, backend := range []string{"native_llamacpp", "llamacpp", "llama.cpp", "llama_cpp"} {
		if got := VerifierBackendKind(backend); got != VerifierBackendLlamaCpp {
			t.Fatalf("VerifierBackendKind(%q) = %q, want %q", backend, got, VerifierBackendLlamaCpp)
		}
	}
}
