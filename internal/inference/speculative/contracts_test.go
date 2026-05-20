package speculative

import "testing"

func TestDecodeDraftSpecCanonicalizesContractTestBridge(t *testing.T) {
	specJSON := `{
		"task":"draft_runner",
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
	if spec.NodeID != "node-drafter" || spec.Horizon != 4 || spec.BranchCount != 3 {
		t.Fatalf("decoded spec = %+v, want normalized draft contract fields", spec)
	}
}

func TestDecodeVerifierSpecAcceptsTree(t *testing.T) {
	specJSON := `{
		"task":"verifier_session",
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
		"task":"verifier_session_hot",
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
