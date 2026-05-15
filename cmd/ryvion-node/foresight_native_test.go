package main

import (
	"encoding/json"
	"testing"
)

func TestDecodeForesightNativeDraftSpecBuildsPackets(t *testing.T) {
	specJSON := `{
		"task":"draft_runner_v8",
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
	accepted, treeCID, ok := decodeForesightNativeVerifierSpec(specJSON)
	if !ok {
		t.Fatal("decodeForesightNativeVerifierSpec ok = false")
	}
	if accepted != 8 {
		t.Fatalf("accepted = %d, want capped 8", accepted)
	}
	if treeCID != "sha256:tree" {
		t.Fatalf("treeCID = %q, want sha256:tree", treeCID)
	}
}
