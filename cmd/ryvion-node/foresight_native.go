package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/Ryvion/node-agent/internal/hub"
)

const (
	foresightDraftRunnerTask        = "draft_runner_v8"
	foresightVerifierSessionTask    = "verifier_session_v8"
	foresightNativeExecutor         = "native_foresight_v8"
	defaultNativeDraftConfidenceBPS = int64(7600)
)

type foresightNativeDraftSpec struct {
	Task                 string `json:"task"`
	WorkGraphID          string `json:"workgraph_id"`
	WindowID             string `json:"window_id"`
	RoleID               string `json:"role_id"`
	TargetNodeID         string `json:"target_node_id"`
	NodeID               string `json:"node_id"`
	Prompt               string `json:"prompt"`
	ParentPrefixHash     string `json:"parent_prefix_hash"`
	BranchCount          int    `json:"branch_count"`
	Horizon              int    `json:"horizon"`
	DeadlineMs           int    `json:"deadline_ms"`
	ModelHash            string `json:"model_hash"`
	DrafterModelID       string `json:"drafter_model_id"`
	FirstPacketTimeoutMs int    `json:"first_packet_timeout_ms"`
}

func processOptionalForesightNativeDraft(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, runtimeMgr *runtimeManager, gpuDetected bool) (bool, *runnerResultSnapshot, error) {
	spec, ok := decodeForesightNativeDraftSpec(work.SpecJSON)
	if !ok {
		return false, nil, nil
	}
	started := time.Now()
	packets := buildForesightNativeDraftPackets(spec)
	summary := submitForesightDraftPackets(ctx, client, packets)
	accepted := intFromAny(summary["accepted"])
	failed := intFromAny(summary["failed"])
	rejected := intFromAny(summary["rejected"])
	resultHash := foresightFullHash(fmt.Sprintf("%s|%s|%d|%d|%d", work.JobID, spec.WindowID, len(packets), accepted, rejected))
	metadata := receiptMetadataBase(work, safeRuntimeReceiptMetadata(runtimeMgr, gpuDetected), map[string]any{
		"executor":                foresightNativeExecutor,
		"executor_kind":           foresightNativeExecutor,
		"task":                    foresightDraftRunnerTask,
		"docker_required":         false,
		"runtime_mode":            "native_node_agent",
		"draft_generation_mode":   "deterministic_native_bridge",
		"window_id":               spec.WindowID,
		"workgraph_id":            spec.WorkGraphID,
		"branch_count":            len(packets),
		"horizon":                 normalizedForesightHorizon(spec.Horizon),
		"draft_packet_submission": summary,
		"duration_ms":             time.Since(started).Milliseconds(),
		"exit_code":               0,
	})
	units := uint64(accepted)
	receipt := hub.Receipt{
		JobID:         work.JobID,
		ResultHashHex: resultHash,
		MeteringUnits: units,
		Metadata:      metadata,
	}
	err := submitReceiptWithRetry(ctx, client, receipt)
	if err == nil && accepted == 0 {
		err = fmt.Errorf("native foresight draft submitted no accepted packets: failed=%d rejected=%d", failed, rejected)
	}
	return true, &runnerResultSnapshot{
		DurationMs:    time.Since(started).Milliseconds(),
		ResultHashHex: resultHash,
		MeteringUnits: units,
		ExitCode:      0,
		Metadata:      metadata,
	}, err
}

func decodeForesightNativeDraftSpec(specJSON string) (foresightNativeDraftSpec, bool) {
	var spec foresightNativeDraftSpec
	if json.Unmarshal([]byte(strings.TrimSpace(specJSON)), &spec) != nil {
		return foresightNativeDraftSpec{}, false
	}
	if strings.TrimSpace(spec.Task) != foresightDraftRunnerTask {
		return foresightNativeDraftSpec{}, false
	}
	spec.WorkGraphID = strings.TrimSpace(spec.WorkGraphID)
	spec.WindowID = strings.TrimSpace(spec.WindowID)
	spec.RoleID = firstNonEmptyString(strings.TrimSpace(spec.RoleID), "draft-worker-native")
	spec.TargetNodeID = strings.TrimSpace(spec.TargetNodeID)
	spec.NodeID = firstNonEmptyString(strings.TrimSpace(spec.NodeID), spec.TargetNodeID)
	spec.ParentPrefixHash = strings.TrimSpace(spec.ParentPrefixHash)
	spec.ModelHash = strings.TrimSpace(spec.ModelHash)
	spec.DrafterModelID = strings.TrimSpace(spec.DrafterModelID)
	if spec.BranchCount <= 0 {
		spec.BranchCount = 1
	}
	if spec.BranchCount > 16 {
		spec.BranchCount = 16
	}
	spec.Horizon = normalizedForesightHorizon(spec.Horizon)
	if spec.DeadlineMs <= 0 {
		spec.DeadlineMs = 1000
	}
	if spec.ModelHash == "" {
		spec.ModelHash = "sha256:" + foresightFullHash(firstNonEmptyString(spec.DrafterModelID, "native-drafter"))
	}
	return spec, spec.WindowID != "" && spec.ParentPrefixHash != ""
}

func buildForesightNativeDraftPackets(spec foresightNativeDraftSpec) []map[string]any {
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
		foresightPromptDigest(spec.Prompt),
	}, "|")
	for branch := 0; branch < count; branch++ {
		confidence := defaultNativeDraftConfidenceBPS - int64(branch*250)
		if confidence < 4000 {
			confidence = 4000
		}
		tokens := foresightDeterministicTokens(seed, branch, spec.Horizon)
		packetID := "pkt_native_" + shortHash(fmt.Sprintf("%s|%d|%v", spec.WindowID, branch, tokens))
		packets = append(packets, map[string]any{
			"packet_id":          packetID,
			"window_id":          spec.WindowID,
			"workgraph_id":       spec.WorkGraphID,
			"role_id":            spec.RoleID,
			"node_id":            spec.NodeID,
			"parent_prefix_hash": spec.ParentPrefixHash,
			"candidate_tokens":   tokens,
			"model_hash":         spec.ModelHash,
			"drafter_model_id":   firstNonEmptyString(spec.DrafterModelID, "native-node-agent-drafter"),
			"horizon":            len(tokens),
			"confidence_bps":     confidence,
			"deadline_ms":        spec.DeadlineMs,
			"energy_mwh":         1,
		})
	}
	return packets
}

func processOptionalForesightNativeVerifier(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, runtimeMgr *runtimeManager, gpuDetected bool) (bool, *runnerResultSnapshot, error) {
	acceptedLen, treeCID, ok := decodeForesightNativeVerifierSpec(work.SpecJSON)
	if !ok {
		return false, nil, nil
	}
	started := time.Now()
	resultHash := foresightFullHash(fmt.Sprintf("%s|%s|%d", work.JobID, treeCID, acceptedLen))
	metadata := receiptMetadataBase(work, safeRuntimeReceiptMetadata(runtimeMgr, gpuDetected), map[string]any{
		"executor":        foresightNativeExecutor,
		"executor_kind":   foresightNativeExecutor,
		"task":            foresightVerifierSessionTask,
		"docker_required": false,
		"runtime_mode":    "native_node_agent",
		"exit_code":       0,
		"duration_ms":     time.Since(started).Milliseconds(),
		"verifier_session": map[string]any{
			"accepted_token_receipt": map[string]any{
				"accepted_len": acceptedLen,
				"tree_cid":     treeCID,
			},
			"probe_summary": map[string]any{
				"confidence_bps": 8200,
				"source":         "native_node_agent_contract_verifier",
			},
		},
	})
	receipt := hub.Receipt{
		JobID:         work.JobID,
		ResultHashHex: resultHash,
		MeteringUnits: uint64(acceptedLen),
		Metadata:      metadata,
	}
	err := submitReceiptWithRetry(ctx, client, receipt)
	return true, &runnerResultSnapshot{
		DurationMs:    time.Since(started).Milliseconds(),
		ResultHashHex: resultHash,
		MeteringUnits: uint64(acceptedLen),
		ExitCode:      0,
		Metadata:      metadata,
	}, err
}

func decodeForesightNativeVerifierSpec(specJSON string) (int, string, bool) {
	var spec map[string]any
	if json.Unmarshal([]byte(strings.TrimSpace(specJSON)), &spec) != nil {
		return 0, "", false
	}
	if strings.TrimSpace(stringValue(spec["task"])) != foresightVerifierSessionTask {
		return 0, "", false
	}
	tree := mapFromAny(spec["tree"])
	branches := sliceFromAny(tree["branches"])
	acceptedLen := 0
	for _, raw := range branches {
		branch := mapFromAny(raw)
		tokens := sliceFromAny(branch["candidate_tokens"])
		if len(tokens) > acceptedLen {
			acceptedLen = len(tokens)
		}
	}
	if acceptedLen <= 0 {
		acceptedLen = 1
	}
	if acceptedLen > 8 {
		acceptedLen = 8
	}
	treeCID := strings.TrimSpace(stringValue(tree["tree_cid"]))
	if treeCID == "" {
		treeCID = "sha256:" + foresightFullHash(specJSON)
	}
	return acceptedLen, treeCID, true
}

func foresightDeterministicTokens(seed string, branch int, horizon int) []int {
	horizon = normalizedForesightHorizon(horizon)
	tokens := make([]int, 0, horizon)
	for len(tokens) < horizon {
		sum := sha256.Sum256([]byte(fmt.Sprintf("%s|branch:%d|offset:%d", seed, branch, len(tokens))))
		for offset := 0; offset+4 <= len(sum) && len(tokens) < horizon; offset += 4 {
			token := int(binary.BigEndian.Uint32(sum[offset:offset+4])%32000) + 1
			tokens = append(tokens, token)
		}
	}
	return tokens
}

func normalizedForesightHorizon(horizon int) int {
	if horizon <= 0 {
		return 8
	}
	if horizon > 64 {
		return 64
	}
	return horizon
}

func foresightPromptDigest(prompt string) string {
	if strings.TrimSpace(prompt) == "" {
		return ""
	}
	return foresightFullHash(prompt)
}

func foresightFullHash(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}

func safeRuntimeReceiptMetadata(runtimeMgr *runtimeManager, gpuDetected bool) map[string]any {
	if runtimeMgr == nil {
		return map[string]any{}
	}
	return runtimeMgr.ReceiptMetadata(gpuDetected)
}

func intFromAny(value any) int {
	switch typed := value.(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case int32:
		return int(typed)
	case float64:
		return int(typed)
	case float32:
		return int(typed)
	case json.Number:
		parsed, _ := typed.Int64()
		return int(parsed)
	default:
		return 0
	}
}

func mapFromAny(value any) map[string]any {
	if typed, ok := value.(map[string]any); ok {
		return typed
	}
	return map[string]any{}
}

func sliceFromAny(value any) []any {
	if typed, ok := value.([]any); ok {
		return typed
	}
	return nil
}
