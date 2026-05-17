package main

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/Ryvion/ryvion-node/internal/hub"
	llamacppdemo "github.com/Ryvion/ryvion-node/internal/inference/speculative/verify/llamacpp_demo"
	v7llamacpp "github.com/Ryvion/ryvion-node/internal/v7/llamacpp"
)

const nativeLlamaCppVerifierExecutor = llamacppdemo.Executor

var errNativeLlamaCppUnavailable = llamacppdemo.ErrUnavailable

var newForesightNativeLlamaCppVerifier = llamacppdemo.NewVerifierFromEnv

func processForesightNativeLlamaCppVerifier(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, runtimeMgr *runtimeManager, gpuDetected bool, spec foresightNativeHotSessionSpec) (*runnerResultSnapshot, error) {
	started := time.Now()
	verifier := newForesightNativeLlamaCppVerifier()
	session, err := llamacppdemo.RunHotSession(ctx, client, verifier, work.JobID, spec, 100*time.Millisecond)
	if err != nil {
		if errors.Is(err, errNativeLlamaCppUnavailable) {
			unavailable := submitForesightNativeLlamaCppUnavailableReceipt(ctx, client, work, runtimeMgr, gpuDetected, spec, started, llamacppdemo.UnavailableCode(verifier.Status(ctx)))
			return unavailable, err
		}
		receiptCtx := ctx
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			receiptCtx = context.Background()
		}
		failed := submitForesightNativeLlamaCppFinalReceipt(receiptCtx, client, work, runtimeMgr, gpuDetected, spec, started, session.TotalAccepted, session.Waves, session.AcceptedText, session.FinalReason)
		return failed, err
	}
	result := submitForesightNativeLlamaCppFinalReceipt(ctx, client, work, runtimeMgr, gpuDetected, spec, started, session.TotalAccepted, session.Waves, session.AcceptedText, session.FinalReason)
	return result, nil
}

func submitForesightNativeLlamaCppUnavailableReceipt(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, runtimeMgr *runtimeManager, gpuDetected bool, spec foresightNativeHotSessionSpec, started time.Time, errorCode string) *runnerResultSnapshot {
	resultHash := foresightFullHash(fmt.Sprintf("%s|%s|%s", work.JobID, foresightVerifierBackendLlamaCpp, errorCode))
	metadata := receiptMetadataBase(work, safeRuntimeReceiptMetadata(runtimeMgr, gpuDetected), map[string]any{
		"executor":              nativeLlamaCppVerifierExecutor,
		"executor_kind":         nativeLlamaCppVerifierExecutor,
		"task":                  firstNonEmptyString(spec.Task, foresightVerifierHotSessionTask),
		"docker_required":       false,
		"runtime_mode":          "native_node_agent",
		"verifier_backend":      foresightVerifierBackendLlamaCpp,
		"backend":               v7llamacpp.BackendName,
		"status":                "unavailable",
		"execution_status":      "unavailable",
		"billing_status":        "not_billable",
		"proof_status":          "llamacpp_demo_unavailable",
		"error_code":            firstNonEmptyString(strings.TrimSpace(errorCode), "llamacpp_demo_unavailable"),
		"install_hint":          "install llama-server or set RYV_LLAMA_CPP_SERVER_PATH / RYV_LLAMA_CPP_MODEL_PATH",
		"model_path_configured": spec.ModelPath != "",
		"exit_code":             1,
		"duration_ms":           time.Since(started).Milliseconds(),
	})
	_ = submitReceiptWithRetry(ctx, client, hub.Receipt{JobID: work.JobID, ResultHashHex: resultHash, MeteringUnits: 0, Metadata: metadata})
	return &runnerResultSnapshot{DurationMs: time.Since(started).Milliseconds(), ResultHashHex: resultHash, MeteringUnits: 0, ExitCode: 1, Metadata: metadata}
}

func submitForesightNativeLlamaCppFinalReceipt(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, runtimeMgr *runtimeManager, gpuDetected bool, spec foresightNativeHotSessionSpec, started time.Time, accepted int, waves int, acceptedText string, reason string) *runnerResultSnapshot {
	resultHash := foresightFullHash(fmt.Sprintf("%s|%s|llamacpp_demo|%d|%d|%s", work.JobID, spec.RunID, accepted, waves, reason))
	outputHash := ""
	if strings.TrimSpace(acceptedText) != "" {
		outputHash = "sha256:" + sha256Hex([]byte(acceptedText))
	}
	metadata := receiptMetadataBase(work, safeRuntimeReceiptMetadata(runtimeMgr, gpuDetected), map[string]any{
		"executor":         nativeLlamaCppVerifierExecutor,
		"executor_kind":    nativeLlamaCppVerifierExecutor,
		"task":             firstNonEmptyString(spec.Task, foresightVerifierHotSessionTask),
		"docker_required":  false,
		"runtime_mode":     "native_node_agent",
		"verifier_backend": foresightVerifierBackendLlamaCpp,
		"backend":          v7llamacpp.BackendName,
		"session_mode":     "hot",
		"run_id":           spec.RunID,
		"session_id":       spec.SessionID,
		"workgraph_id":     spec.WorkGraphID,
		"wave_count":       waves,
		"duration_ms":      time.Since(started).Milliseconds(),
		"exit_code":        0,
		"stop_reason":      firstNonEmptyString(strings.TrimSpace(reason), "completed"),
		"verifier_session": map[string]any{
			"duration_ms": time.Since(started).Milliseconds(),
			"accepted_token_receipt": map[string]any{
				"accepted_len":          accepted,
				"accepted_text_hash":    outputHash,
				"accepted_text_public":  false,
				"tree_cid":              "sha256:" + foresightFullHash(fmt.Sprintf("%s|%s|llamacpp_demo_final_tree", work.JobID, spec.RunID)),
				"hot_session_finalized": true,
			},
			"probe_summary": map[string]any{
				"source":      "llamacpp_demo_verifier",
				"backend":     v7llamacpp.BackendName,
				"output_hash": outputHash,
			},
		},
	})
	units := uint64(accepted)
	if units == 0 {
		units = 1
	}
	_ = submitReceiptWithRetry(ctx, client, hub.Receipt{JobID: work.JobID, ResultHashHex: resultHash, MeteringUnits: units, Metadata: metadata})
	return &runnerResultSnapshot{DurationMs: time.Since(started).Milliseconds(), ResultHashHex: resultHash, MeteringUnits: units, ExitCode: 0, Metadata: metadata}
}
