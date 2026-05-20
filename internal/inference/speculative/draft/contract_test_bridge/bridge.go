package contracttestbridge

import (
	"fmt"
	"strings"

	nodespec "github.com/Ryvion/ryvion-node/internal/inference/speculative"
)

const Backend = nodespec.DraftBackendNative

func IsBackend(backend string) bool {
	return nodespec.DraftBackendIsNativeBridge(backend)
}

func BuildPackets(spec nodespec.DraftSpec) []map[string]any {
	count := spec.BranchCount
	if count <= 0 {
		count = 1
	}
	packets := make([]map[string]any, 0, count)
	seed := strings.Join([]string{
		spec.WorkGraphID,
		spec.WindowID,
		spec.ParentPrefixHash,
		spec.DrafterModelID,
		nodespec.PromptDigest(spec.Prompt),
	}, "|")
	for branch := 0; branch < count; branch++ {
		confidence := nodespec.DefaultNativeDraftConfidenceBPS - int64(branch*250)
		if confidence < 4000 {
			confidence = 4000
		}
		tokens := nodespec.DeterministicTokens(seed, branch, spec.Horizon)
		packetID := "pkt_contract_test_" + shortHash(fmt.Sprintf("%s|%d|%v", spec.WindowID, branch, tokens))
		packets = append(packets, map[string]any{
			"packet_id":          packetID,
			"window_id":          spec.WindowID,
			"workgraph_id":       spec.WorkGraphID,
			"role_id":            spec.RoleID,
			"node_id":            spec.NodeID,
			"parent_prefix_hash": spec.ParentPrefixHash,
			"candidate_tokens":   tokens,
			"model_hash":         spec.ModelHash,
			"drafter_model_id":   firstNonEmpty(spec.DrafterModelID, "contract-test-bridge-drafter"),
			"horizon":            len(tokens),
			"confidence_bps":     confidence,
			"deadline_ms":        spec.DeadlineMs,
			"energy_mwh":         1,
			"production_valid":   false,
			"test_adapter":       true,
			"billing_status":     "not_billable_contract_test",
		})
	}
	return packets
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func shortHash(value string) string {
	hash := nodespec.FullHash(value)
	if len(hash) > 12 {
		return hash[:12]
	}
	return hash
}
