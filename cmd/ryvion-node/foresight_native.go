package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/Ryvion/ryvion-node/internal/hub"
	nodespec "github.com/Ryvion/ryvion-node/internal/inference/speculative"
	draftbridge "github.com/Ryvion/ryvion-node/internal/inference/speculative/draft/contract_test_bridge"
	contracttestverifier "github.com/Ryvion/ryvion-node/internal/inference/speculative/verify/contract_test_bridge"
)

const (
	foresightDraftRunnerTask         = nodespec.DraftRunnerTask
	foresightVerifierSessionTask     = nodespec.VerifierSessionTask
	foresightDraftHotSessionTask     = nodespec.DraftHotSessionTask
	foresightVerifierHotSessionTask  = nodespec.VerifierHotSessionTask
	foresightNativeExecutor          = nodespec.NativeExecutor
	foresightDraftBackendNative      = nodespec.DraftBackendNative
	foresightVerifierBackendBridge   = nodespec.VerifierBackendBridge
	foresightVerifierBackendSGLang   = nodespec.VerifierBackendSGLang
	foresightVerifierBackendLlamaCpp = nodespec.VerifierBackendLlamaCpp
	defaultNativeDraftConfidenceBPS  = nodespec.DefaultNativeDraftConfidenceBPS
)

type foresightNativeDraftSpec = nodespec.DraftSpec
type foresightNativeHotSessionSpec = nodespec.HotSessionSpec

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
		"draft_generation_mode":   foresightDraftBackendNative,
		"production_valid":        false,
		"test_adapter":            true,
		"billing_status":          "not_billable_contract_test",
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
	session, err := draftbridge.RunHotSession(ctx, client, spec, work.JobID, 100*time.Millisecond)
	if err != nil {
		return true, nil, err
	}
	result := submitForesightNativeHotDraftReceipt(ctx, client, work, runtimeMgr, gpuDetected, spec, started, session.AcceptedPackets, session.RawPackets, session.Waves)
	return true, result, nil
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
		"draft_generation_mode":  foresightDraftBackendNative,
		"production_valid":       false,
		"test_adapter":           true,
		"billing_status":         "not_billable_contract_test",
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
	return nodespec.DecodeDraftSpec(specJSON)
}

func decodeForesightNativeHotSessionSpec(specJSON string, expectedTask string) (foresightNativeHotSessionSpec, bool) {
	return nodespec.DecodeHotSessionSpec(specJSON, expectedTask)
}

func buildForesightNativeDraftPackets(spec foresightNativeDraftSpec) []map[string]any {
	return draftbridge.BuildPackets(spec)
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
		"executor":         foresightNativeExecutor,
		"executor_kind":    foresightNativeExecutor,
		"task":             foresightVerifierSessionTask,
		"docker_required":  false,
		"runtime_mode":     "native_node_agent",
		"production_valid": false,
		"test_adapter":     true,
		"billing_status":   "not_billable_contract_test",
		"exit_code":        0,
		"duration_ms":      time.Since(started).Milliseconds(),
		"verifier_session": map[string]any{
			"accepted_token_receipt": map[string]any{
				"accepted_len": acceptedLen,
				"tree_cid":     treeCID,
			},
			"probe_summary": map[string]any{
				"confidence_bps":   8200,
				"source":           contracttestverifier.Source,
				"backend":          contracttestverifier.Backend,
				"production_valid": false,
				"test_adapter":     true,
				"billing_status":   "not_billable_contract_test",
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
	case foresightVerifierBackendLlamaCpp:
		result, err := processForesightNativeLlamaCppVerifier(ctx, client, work, runtimeMgr, gpuDetected, spec)
		return true, result, err
	case foresightVerifierBackendBridge:
	default:
		return true, submitForesightNativeUnsupportedReceipt(ctx, client, work, runtimeMgr, gpuDetected, spec.VerifierBackend, foresightVerifierHotSessionTask), fmt.Errorf("unsupported native foresight verifier backend: %s", spec.VerifierBackend)
	}
	started := time.Now()
	session, err := contracttestverifier.RunHotSession(ctx, client, work.JobID, spec, 100*time.Millisecond)
	if err != nil {
		return true, nil, err
	}
	result := submitForesightNativeHotVerifierReceipt(ctx, client, work, runtimeMgr, gpuDetected, spec, started, session.TotalAccepted, session.Waves, session.AcceptedText, session.FinalReason)
	return true, result, nil
}

func submitForesightNativeHotVerifierReceipt(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, runtimeMgr *runtimeManager, gpuDetected bool, spec foresightNativeHotSessionSpec, started time.Time, accepted int, waves int, acceptedText string, reason string) *runnerResultSnapshot {
	resultHash := foresightFullHash(fmt.Sprintf("%s|%s|verify_hot|%d|%d|%s", work.JobID, spec.RunID, accepted, waves, reason))
	acceptedTextHash := foresightAcceptedTextHash(acceptedText)
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
				"accepted_text_hash":    acceptedTextHash,
				"accepted_text_public":  false,
				"tree_cid":              "sha256:" + foresightFullHash(fmt.Sprintf("%s|%s|final_tree", work.JobID, spec.RunID)),
				"hot_session_finalized": true,
			},
			"probe_summary": map[string]any{
				"confidence_bps":   8200,
				"source":           contracttestverifier.Source,
				"backend":          contracttestverifier.Backend,
				"production_valid": false,
				"test_adapter":     true,
				"billing_status":   "not_billable_contract_test",
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

func foresightAcceptedTextHash(acceptedText string) string {
	if strings.TrimSpace(acceptedText) == "" {
		return ""
	}
	return "sha256:" + sha256Hex([]byte(acceptedText))
}

func redactForesightAcceptedTextReceipt(receipt map[string]any) string {
	if len(receipt) == 0 {
		return ""
	}
	hash := strings.TrimSpace(stringValue(receipt["accepted_text_hash"]))
	if text := strings.TrimSpace(stringValue(receipt["accepted_text"])); text != "" && hash == "" {
		hash = foresightAcceptedTextHash(text)
	}
	delete(receipt, "accepted_text")
	receipt["accepted_text_public"] = false
	if hash != "" {
		receipt["accepted_text_hash"] = hash
	}
	return hash
}

func decodeForesightNativeVerifierSpec(specJSON string) (int, string, string, bool) {
	return nodespec.DecodeVerifierSpec(specJSON)
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
	return draftbridge.IsBackend(backend)
}

func foresightVerifierBackendKind(backend string) string {
	return nodespec.VerifierBackendKind(backend)
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
	return nodespec.AcceptedFromTree(command.Tree, command.CommandID)
}

func foresightDeterministicTokens(seed string, branch int, horizon int) []int {
	return nodespec.DeterministicTokens(seed, branch, horizon)
}

func normalizedForesightHorizon(horizon int) int {
	return nodespec.NormalizeHorizon(horizon)
}

func foresightPromptDigest(prompt string) string {
	return nodespec.PromptDigest(prompt)
}

func foresightFullHash(value string) string {
	return nodespec.FullHash(value)
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
