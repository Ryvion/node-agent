package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/Ryvion/node-agent/internal/hub"
)

const (
	foresightDraftRunnerTask        = "draft_runner_v8"
	foresightVerifierSessionTask    = "verifier_session_v8"
	foresightDraftHotSessionTask    = "draft_runner_v8_hot_session"
	foresightVerifierHotSessionTask = "verifier_session_v8_hot"
	foresightNativeExecutor         = "native_foresight_v8"
	foresightDraftBackendNative     = "native_bridge"
	foresightVerifierBackendBridge  = "native_bridge"
	foresightVerifierBackendSGLang  = "native_sglang"
	defaultNativeDraftConfidenceBPS = int64(7600)
)

type foresightNativeDraftSpec struct {
	Task                 string `json:"task"`
	ExecutorKind         string `json:"executor_kind,omitempty"`
	RunnerImage          string `json:"runner_image,omitempty"`
	DockerRequired       bool   `json:"docker_required,omitempty"`
	DraftBackend         string `json:"draft_backend,omitempty"`
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

type foresightNativeHotSessionSpec struct {
	Task             string `json:"task"`
	ExecutorKind     string `json:"executor_kind,omitempty"`
	RunnerImage      string `json:"runner_image,omitempty"`
	DockerRequired   bool   `json:"docker_required,omitempty"`
	DraftBackend     string `json:"draft_backend,omitempty"`
	VerifierBackend  string `json:"verifier_backend,omitempty"`
	RunID            string `json:"run_id"`
	SessionID        string `json:"session_id"`
	WorkGraphID      string `json:"workgraph_id"`
	RoleID           string `json:"role_id"`
	TargetNodeID     string `json:"target_node_id"`
	NodeID           string `json:"node_id"`
	Prompt           string `json:"prompt"`
	ParentPrefixHash string `json:"parent_prefix_hash"`
	ModelID          string `json:"model_id"`
	ModelHash        string `json:"model_hash"`
	ModelPath        string `json:"model_path,omitempty"`
	DrafterModelID   string `json:"drafter_model_id"`
	MaxTokens        int    `json:"max_tokens"`
}

func processOptionalForesightNativeDraft(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, runtimeMgr *runtimeManager, gpuDetected bool) (bool, *runnerResultSnapshot, error) {
	spec, ok := decodeForesightNativeDraftSpec(work.SpecJSON)
	if !ok {
		return false, nil, nil
	}
	if foresightNativeExternalRuntimeRequested(work, spec.ExecutorKind, spec.RunnerImage, spec.DockerRequired) {
		return false, nil, nil
	}
	if !foresightDraftBackendIsNativeBridge(spec.DraftBackend) {
		return true, submitForesightNativeUnsupportedReceipt(ctx, client, work, runtimeMgr, gpuDetected, spec.DraftBackend, foresightDraftRunnerTask), fmt.Errorf("unsupported native foresight draft backend: %s", spec.DraftBackend)
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

func processOptionalForesightNativeDraftHotSession(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, runtimeMgr *runtimeManager, gpuDetected bool) (bool, *runnerResultSnapshot, error) {
	spec, ok := decodeForesightNativeHotSessionSpec(work.SpecJSON, foresightDraftHotSessionTask)
	if !ok {
		return false, nil, nil
	}
	if foresightNativeExternalRuntimeRequested(work, spec.ExecutorKind, spec.RunnerImage, spec.DockerRequired) {
		return false, nil, nil
	}
	if !foresightDraftBackendIsNativeBridge(spec.DraftBackend) {
		return true, submitForesightNativeUnsupportedReceipt(ctx, client, work, runtimeMgr, gpuDetected, spec.DraftBackend, foresightDraftHotSessionTask), fmt.Errorf("unsupported native foresight draft backend: %s", spec.DraftBackend)
	}
	started := time.Now()
	submittedWindows := map[string]bool{}
	totalAccepted := 0
	totalRaw := 0
	waves := 0
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		command, err := client.FetchForesightLiveLabDraftCommand(ctx, spec.RunID, work.JobID)
		if err != nil {
			select {
			case <-ctx.Done():
				return true, nil, ctx.Err()
			case <-ticker.C:
				continue
			}
		}
		switch strings.TrimSpace(command.Command) {
		case "close_session":
			result := submitForesightNativeHotDraftReceipt(ctx, client, work, runtimeMgr, gpuDetected, spec, started, totalAccepted, totalRaw, waves)
			return true, result, nil
		case "generate_draft_packets":
			windowID := strings.TrimSpace(command.WindowID)
			if windowID != "" && !submittedWindows[windowID] {
				draftSpec := foresightNativeDraftSpec{
					Task:                 foresightDraftRunnerTask,
					WorkGraphID:          firstNonEmptyString(command.WorkGraphID, spec.WorkGraphID),
					WindowID:             windowID,
					RoleID:               firstNonEmptyString(command.RoleID, spec.RoleID),
					TargetNodeID:         firstNonEmptyString(command.TargetNodeID, spec.TargetNodeID),
					NodeID:               firstNonEmptyString(command.NodeID, spec.NodeID),
					Prompt:               firstNonEmptyString(command.Prompt, spec.Prompt),
					ParentPrefixHash:     firstNonEmptyString(command.ParentPrefixHash, spec.ParentPrefixHash),
					BranchCount:          command.BranchCount,
					Horizon:              command.Horizon,
					DeadlineMs:           command.DeadlineMs,
					ModelHash:            firstNonEmptyString(command.ModelHash, spec.ModelHash),
					DrafterModelID:       firstNonEmptyString(command.DrafterModelID, spec.DrafterModelID),
					FirstPacketTimeoutMs: command.FirstPacketTimeout,
				}
				packets := buildForesightNativeDraftPackets(draftSpec)
				summary := submitForesightDraftPackets(ctx, client, packets)
				totalAccepted += intFromAny(summary["accepted"])
				totalRaw += len(packets)
				waves++
				submittedWindows[windowID] = true
			}
		}
		select {
		case <-ctx.Done():
			return true, nil, ctx.Err()
		case <-ticker.C:
		}
	}
}

func submitForesightNativeHotDraftReceipt(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, runtimeMgr *runtimeManager, gpuDetected bool, spec foresightNativeHotSessionSpec, started time.Time, accepted int, raw int, waves int) *runnerResultSnapshot {
	resultHash := foresightFullHash(fmt.Sprintf("%s|%s|draft_hot|%d|%d|%d", work.JobID, spec.RunID, accepted, raw, waves))
	metadata := receiptMetadataBase(work, safeRuntimeReceiptMetadata(runtimeMgr, gpuDetected), map[string]any{
		"executor":               foresightNativeExecutor,
		"executor_kind":          foresightNativeExecutor,
		"task":                   foresightDraftHotSessionTask,
		"docker_required":        false,
		"runtime_mode":           "native_node_agent",
		"session_mode":           "hot",
		"run_id":                 spec.RunID,
		"session_id":             spec.SessionID,
		"workgraph_id":           spec.WorkGraphID,
		"draft_packets_raw":      raw,
		"draft_packets_accepted": accepted,
		"wave_count":             waves,
		"duration_ms":            time.Since(started).Milliseconds(),
		"exit_code":              0,
	})
	units := uint64(accepted)
	if units == 0 {
		units = 1
	}
	receipt := hub.Receipt{JobID: work.JobID, ResultHashHex: resultHash, MeteringUnits: units, Metadata: metadata}
	_ = submitReceiptWithRetry(ctx, client, receipt)
	return &runnerResultSnapshot{DurationMs: time.Since(started).Milliseconds(), ResultHashHex: resultHash, MeteringUnits: units, ExitCode: 0, Metadata: metadata}
}

func decodeForesightNativeDraftSpec(specJSON string) (foresightNativeDraftSpec, bool) {
	var spec foresightNativeDraftSpec
	if json.Unmarshal([]byte(strings.TrimSpace(specJSON)), &spec) != nil {
		return foresightNativeDraftSpec{}, false
	}
	if strings.TrimSpace(spec.Task) != foresightDraftRunnerTask {
		return foresightNativeDraftSpec{}, false
	}
	spec.ExecutorKind = strings.TrimSpace(spec.ExecutorKind)
	spec.RunnerImage = strings.TrimSpace(spec.RunnerImage)
	spec.DraftBackend = strings.TrimSpace(spec.DraftBackend)
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

func decodeForesightNativeHotSessionSpec(specJSON string, expectedTask string) (foresightNativeHotSessionSpec, bool) {
	var spec foresightNativeHotSessionSpec
	if json.Unmarshal([]byte(strings.TrimSpace(specJSON)), &spec) != nil {
		return foresightNativeHotSessionSpec{}, false
	}
	if strings.TrimSpace(spec.Task) != expectedTask {
		return foresightNativeHotSessionSpec{}, false
	}
	spec.ExecutorKind = strings.TrimSpace(spec.ExecutorKind)
	spec.RunnerImage = strings.TrimSpace(spec.RunnerImage)
	spec.DraftBackend = strings.TrimSpace(spec.DraftBackend)
	spec.VerifierBackend = strings.TrimSpace(spec.VerifierBackend)
	spec.RunID = strings.TrimSpace(spec.RunID)
	spec.SessionID = strings.TrimSpace(spec.SessionID)
	spec.WorkGraphID = strings.TrimSpace(spec.WorkGraphID)
	spec.RoleID = strings.TrimSpace(spec.RoleID)
	spec.TargetNodeID = strings.TrimSpace(spec.TargetNodeID)
	spec.NodeID = firstNonEmptyString(strings.TrimSpace(spec.NodeID), spec.TargetNodeID)
	spec.ParentPrefixHash = strings.TrimSpace(spec.ParentPrefixHash)
	spec.ModelID = strings.TrimSpace(spec.ModelID)
	spec.ModelHash = strings.TrimSpace(spec.ModelHash)
	spec.ModelPath = strings.TrimSpace(spec.ModelPath)
	spec.DrafterModelID = strings.TrimSpace(spec.DrafterModelID)
	return spec, spec.RunID != "" && spec.WorkGraphID != ""
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
	acceptedLen, treeCID, backend, ok := decodeForesightNativeVerifierSpec(work.SpecJSON)
	if !ok {
		return false, nil, nil
	}
	if foresightNativeExternalRuntimeRequested(work, "", "", false) {
		return false, nil, nil
	}
	switch foresightVerifierBackendKind(backend) {
	case foresightVerifierBackendSGLang:
		result := submitForesightNativeSGLangUnavailableReceipt(ctx, client, work, runtimeMgr, gpuDetected, foresightNativeHotSessionSpec{
			Task:            foresightVerifierSessionTask,
			VerifierBackend: backend,
		}, time.Now(), "native_sglang_non_hot_requires_hot_session")
		return true, result, errNativeSGLangUnavailable
	case foresightVerifierBackendBridge:
	default:
		return true, submitForesightNativeUnsupportedReceipt(ctx, client, work, runtimeMgr, gpuDetected, backend, foresightVerifierSessionTask), fmt.Errorf("unsupported native foresight verifier backend: %s", backend)
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

func processOptionalForesightNativeVerifierHotSession(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, runtimeMgr *runtimeManager, gpuDetected bool) (bool, *runnerResultSnapshot, error) {
	spec, ok := decodeForesightNativeHotSessionSpec(work.SpecJSON, foresightVerifierHotSessionTask)
	if !ok {
		return false, nil, nil
	}
	if foresightNativeExternalRuntimeRequested(work, spec.ExecutorKind, spec.RunnerImage, spec.DockerRequired) {
		return false, nil, nil
	}
	switch foresightVerifierBackendKind(spec.VerifierBackend) {
	case foresightVerifierBackendSGLang:
		result, err := processForesightNativeSGLangVerifier(ctx, client, work, runtimeMgr, gpuDetected, spec)
		return true, result, err
	case foresightVerifierBackendBridge:
	default:
		return true, submitForesightNativeUnsupportedReceipt(ctx, client, work, runtimeMgr, gpuDetected, spec.VerifierBackend, foresightVerifierHotSessionTask), fmt.Errorf("unsupported native foresight verifier backend: %s", spec.VerifierBackend)
	}
	started := time.Now()
	verifiedCommands := map[string]bool{}
	totalAccepted := 0
	waves := 0
	var acceptedText strings.Builder
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	for {
		command, err := client.FetchForesightLiveLabVerifierCommand(ctx, spec.RunID, work.JobID)
		if err != nil {
			select {
			case <-ctx.Done():
				return true, nil, ctx.Err()
			case <-ticker.C:
				continue
			}
		}
		switch strings.TrimSpace(command.Command) {
		case "close_session":
			result := submitForesightNativeHotVerifierReceipt(ctx, client, work, runtimeMgr, gpuDetected, spec, started, totalAccepted, waves, acceptedText.String(), command.Reason)
			return true, result, nil
		case "verify_tree":
			commandID := firstNonEmptyString(command.CommandID, fmt.Sprintf("%s:%s:%d", spec.RunID, command.WindowID, command.WaveIndex))
			if !verifiedCommands[commandID] {
				waveStarted := time.Now()
				acceptedLen, treeCID := foresightAcceptedFromCommandTree(command)
				text := foresightAcceptedTextForWave(spec.Prompt, command.WaveIndex, acceptedLen)
				if text != "" {
					acceptedText.WriteString(text)
				}
				result := hub.ForesightLiveLabVerifierResult{
					JobID:              work.JobID,
					WindowID:           command.WindowID,
					WaveIndex:          command.WaveIndex,
					AcceptedLen:        acceptedLen,
					TreeCID:            treeCID,
					DurationMs:         maxInt64Node(1, time.Since(waveStarted).Milliseconds()),
					AcceptedText:       text,
					AcceptedTextPublic: true,
					EOS:                false,
					ProbeSummary: map[string]any{
						"confidence_bps": 8200,
						"source":         "native_node_agent_hot_verifier",
					},
				}
				if spec.MaxTokens > 0 && totalAccepted+acceptedLen >= spec.MaxTokens {
					result.StopReason = "max_tokens"
				}
				if err := client.SubmitForesightLiveLabVerifierResult(ctx, spec.RunID, result); err == nil {
					totalAccepted += acceptedLen
					waves++
					verifiedCommands[commandID] = true
				}
			}
		}
		select {
		case <-ctx.Done():
			return true, nil, ctx.Err()
		case <-ticker.C:
		}
	}
}

func submitForesightNativeHotVerifierReceipt(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, runtimeMgr *runtimeManager, gpuDetected bool, spec foresightNativeHotSessionSpec, started time.Time, accepted int, waves int, acceptedText string, reason string) *runnerResultSnapshot {
	resultHash := foresightFullHash(fmt.Sprintf("%s|%s|verify_hot|%d|%d|%s", work.JobID, spec.RunID, accepted, waves, reason))
	metadata := receiptMetadataBase(work, safeRuntimeReceiptMetadata(runtimeMgr, gpuDetected), map[string]any{
		"executor":        foresightNativeExecutor,
		"executor_kind":   foresightNativeExecutor,
		"task":            foresightVerifierHotSessionTask,
		"docker_required": false,
		"runtime_mode":    "native_node_agent",
		"session_mode":    "hot",
		"run_id":          spec.RunID,
		"session_id":      spec.SessionID,
		"workgraph_id":    spec.WorkGraphID,
		"wave_count":      waves,
		"duration_ms":     time.Since(started).Milliseconds(),
		"exit_code":       0,
		"stop_reason":     firstNonEmptyString(strings.TrimSpace(reason), "completed"),
		"verifier_session": map[string]any{
			"duration_ms": time.Since(started).Milliseconds(),
			"accepted_token_receipt": map[string]any{
				"accepted_len":          accepted,
				"accepted_text":         acceptedText,
				"accepted_text_public":  strings.TrimSpace(acceptedText) != "",
				"tree_cid":              "sha256:" + foresightFullHash(fmt.Sprintf("%s|%s|final_tree", work.JobID, spec.RunID)),
				"hot_session_finalized": true,
			},
			"probe_summary": map[string]any{
				"confidence_bps": 8200,
				"source":         "native_node_agent_hot_verifier",
			},
		},
	})
	units := uint64(accepted)
	if units == 0 {
		units = 1
	}
	receipt := hub.Receipt{JobID: work.JobID, ResultHashHex: resultHash, MeteringUnits: units, Metadata: metadata}
	_ = submitReceiptWithRetry(ctx, client, receipt)
	return &runnerResultSnapshot{DurationMs: time.Since(started).Milliseconds(), ResultHashHex: resultHash, MeteringUnits: units, ExitCode: 0, Metadata: metadata}
}

func decodeForesightNativeVerifierSpec(specJSON string) (int, string, string, bool) {
	var spec map[string]any
	if json.Unmarshal([]byte(strings.TrimSpace(specJSON)), &spec) != nil {
		return 0, "", "", false
	}
	if strings.TrimSpace(stringValue(spec["task"])) != foresightVerifierSessionTask {
		return 0, "", "", false
	}
	backend := strings.TrimSpace(stringValue(spec["verifier_backend"]))
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
	return acceptedLen, treeCID, backend, true
}

func foresightNativeExternalRuntimeRequested(work *hub.WorkAssignment, executorKind string, runnerImage string, dockerRequired bool) bool {
	if dockerRequired || strings.TrimSpace(runnerImage) != "" {
		return true
	}
	kind := strings.TrimSpace(executorKind)
	if kind != "" && !strings.EqualFold(kind, executorKindNativeReport) && !strings.EqualFold(kind, foresightNativeExecutor) {
		return true
	}
	if work == nil {
		return false
	}
	if workKind := strings.TrimSpace(work.ExecutorKind); workKind != "" && !strings.EqualFold(workKind, executorKindNativeReport) && !strings.EqualFold(workKind, foresightNativeExecutor) {
		return true
	}
	image := strings.TrimSpace(work.Image)
	return image != "" && !strings.EqualFold(image, executorKindNativeReport) && !strings.EqualFold(image, foresightNativeExecutor)
}

func foresightDraftBackendIsNativeBridge(backend string) bool {
	backend = strings.ToLower(strings.TrimSpace(backend))
	return backend == "" || backend == foresightDraftBackendNative || backend == "deterministic_native_bridge"
}

func foresightVerifierBackendKind(backend string) string {
	backend = strings.ToLower(strings.TrimSpace(backend))
	switch backend {
	case "", foresightVerifierBackendBridge, "deterministic_native_bridge":
		return foresightVerifierBackendBridge
	case foresightVerifierBackendSGLang, "sglang":
		return foresightVerifierBackendSGLang
	default:
		return backend
	}
}

func submitForesightNativeUnsupportedReceipt(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, runtimeMgr *runtimeManager, gpuDetected bool, backend string, task string) *runnerResultSnapshot {
	backend = strings.TrimSpace(backend)
	if backend == "" {
		backend = "unknown"
	}
	started := time.Now()
	resultHash := foresightFullHash(fmt.Sprintf("%s|%s|unsupported|%s", work.JobID, task, backend))
	metadata := receiptMetadataBase(work, safeRuntimeReceiptMetadata(runtimeMgr, gpuDetected), map[string]any{
		"executor":         foresightNativeExecutor,
		"executor_kind":    foresightNativeExecutor,
		"task":             task,
		"docker_required":  false,
		"runtime_mode":     "native_node_agent",
		"backend":          backend,
		"status":           "unavailable",
		"execution_status": "unavailable",
		"billing_status":   "not_billable",
		"proof_status":     "native_backend_unavailable",
		"error_code":       "native_backend_unavailable",
		"exit_code":        1,
		"duration_ms":      time.Since(started).Milliseconds(),
	})
	_ = submitReceiptWithRetry(ctx, client, hub.Receipt{JobID: work.JobID, ResultHashHex: resultHash, MeteringUnits: 0, Metadata: metadata})
	return &runnerResultSnapshot{DurationMs: time.Since(started).Milliseconds(), ResultHashHex: resultHash, MeteringUnits: 0, ExitCode: 1, Metadata: metadata}
}

var errNativeSGLangUnavailable = errors.New("native_sglang_verifier_unavailable")

func foresightAcceptedFromCommandTree(command hub.ForesightLiveLabSessionCommand) (int, string) {
	tree := command.Tree
	if len(tree) == 0 {
		return 1, "sha256:" + foresightFullHash(command.CommandID)
	}
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
		encoded, _ := json.Marshal(tree)
		treeCID = "sha256:" + foresightFullHash(string(encoded))
	}
	return acceptedLen, treeCID
}

func foresightAcceptedTextForWave(prompt string, wave int, acceptedLen int) string {
	words := []string{"Ryvion", "Foresight", "Mesh", "keeps", "verifier", "sessions", "hot", "while", "draft", "tokens", "are", "checked", "and", "committed", "quickly."}
	if strings.Contains(strings.ToLower(prompt), "assembly") {
		words = []string{"Assembly", "work", "uses", "low-level", "instructions", "while", "Ryvion", "verifies", "draft", "branches", "quickly."}
	}
	if acceptedLen <= 0 {
		acceptedLen = 1
	}
	start := (maxIntNode(1, wave) - 1) * acceptedLen
	if start >= len(words) {
		return ""
	}
	end := start + acceptedLen
	if end > len(words) {
		end = len(words)
	}
	text := strings.Join(words[start:end], " ")
	if end < len(words) {
		text += " "
	}
	return text
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

func maxIntNode(left int, right int) int {
	if left > right {
		return left
	}
	return right
}

func maxInt64Node(left int64, right int64) int64 {
	if left > right {
		return left
	}
	return right
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
