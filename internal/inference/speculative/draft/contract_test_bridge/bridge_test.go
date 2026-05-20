package contracttestbridge

import (
	"encoding/json"
	"strings"
	"testing"

	nodespec "github.com/Ryvion/ryvion-node/internal/inference/speculative"
)

func TestBuildPacketsProducesDistinctNonBillableContractTestPackets(t *testing.T) {
	spec := nodespec.DraftSpec{
		WorkGraphID:      "wg-live",
		WindowID:         "win-live",
		RoleID:           "draft-worker-live",
		TargetNodeID:     "node-drafter",
		NodeID:           "node-drafter",
		Prompt:           "Write one short sentence.",
		ParentPrefixHash: "sha256:prefix",
		BranchCount:      3,
		Horizon:          4,
		DeadlineMs:       1000,
		ModelHash:        "sha256:model",
		DrafterModelID:   "native-test",
	}

	packets := BuildPackets(spec)
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
		if packet["production_valid"] != false || packet["test_adapter"] != true || packet["billing_status"] != "not_billable_contract_test" {
			t.Fatalf("packet contract-test labels = %#v, want non-production non-billable", packet)
		}
		packetID, ok := packet["packet_id"].(string)
		if !ok || !strings.HasPrefix(packetID, "pkt_contract_test_") {
			t.Fatalf("packet_id = %#v, want contract-test packet id", packet["packet_id"])
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

func TestIsBackendAcceptsLegacyAliasesAsContractTestBridge(t *testing.T) {
	for _, backend := range []string{"", Backend, "native_bridge", "deterministic_native_bridge"} {
		if !IsBackend(backend) {
			t.Fatalf("IsBackend(%q) = false, want true", backend)
		}
	}
	if IsBackend("native_sglang") {
		t.Fatal("IsBackend(native_sglang) = true, want false")
	}
}
