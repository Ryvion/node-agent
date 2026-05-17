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
	totalAccepted := 0
	waves := 0
	verifiedCommands := map[string]bool{}
	var acceptedText strings.Builder
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		command, err := client.FetchForesightLiveLabVerifierCommand(ctx, spec.RunID, work.JobID)
		if err != nil {
			select {
			case <-ctx.Done():
				result := submitForesightNativeLlamaCppFinalReceipt(context.Background(), client, work, runtimeMgr, gpuDetected, spec, started, totalAccepted, waves, acceptedText.String(), "aborted")
				return result, ctx.Err()
			case <-ticker.C:
				continue
			}
		}
		switch strings.TrimSpace(command.Command) {
		case "close_session":
			result := submitForesightNativeLlamaCppFinalReceipt(ctx, client, work, runtimeMgr, gpuDetected, spec, started, totalAccepted, waves, acceptedText.String(), command.Reason)
			return result, nil
		case "verify_tree":
			commandID := firstNonEmptyString(command.CommandID, fmt.Sprintf("%s:%s:%d", spec.RunID, command.WindowID, command.WaveIndex))
			if verifiedCommands[commandID] {
				break
			}
			result, err := verifier.VerifyWave(ctx, spec, command, totalAccepted)
			if err != nil {
				if errors.Is(err, errNativeLlamaCppUnavailable) {
					unavailable := submitForesightNativeLlamaCppUnavailableReceipt(ctx, client, work, runtimeMgr, gpuDetected, spec, started, llamacppdemo.UnavailableCode(verifier.Status(ctx)))
					return unavailable, err
				}
				failed := submitForesightNativeLlamaCppFinalReceipt(ctx, client, work, runtimeMgr, gpuDetected, spec, started, totalAccepted, waves, acceptedText.String(), "llamacpp_demo_verify_failed")
				return failed, err
			}
			result.JobID = work.JobID
			if spec.MaxTokens > 0 && totalAccepted+result.AcceptedLen >= spec.MaxTokens && result.StopReason == "" {
				result.StopReason = "max_tokens"
			}
			if err := client.SubmitForesightLiveLabVerifierResult(ctx, spec.RunID, result); err != nil {
				failed := submitForesightNativeLlamaCppFinalReceipt(ctx, client, work, runtimeMgr, gpuDetected, spec, started, totalAccepted, waves, acceptedText.String(), "result_submit_failed")
				return failed, err
			}
			totalAccepted += result.AcceptedLen
			waves++
			if strings.TrimSpace(result.AcceptedText) != "" {
				acceptedText.WriteString(result.AcceptedText)
			}
			verifiedCommands[commandID] = true
		}
		select {
		case <-ctx.Done():
			result := submitForesightNativeLlamaCppFinalReceipt(context.Background(), client, work, runtimeMgr, gpuDetected, spec, started, totalAccepted, waves, acceptedText.String(), "aborted")
			return result, ctx.Err()
		case <-ticker.C:
		}
	}
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
