package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/Ryvion/ryvion-node/internal/hub"
	sglangverify "github.com/Ryvion/ryvion-node/internal/inference/speculative/verify/sglang"
)

const nativeSGLangVerifierExecutor = "native_sglang_verifier"

func processForesightNativeSGLangVerifier(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, runtimeMgr *runtimeManager, gpuDetected bool, spec foresightNativeHotSessionSpec) (*runnerResultSnapshot, error) {
	started := time.Now()
	commandSpec, ok := sglangverify.ResolveVerifierCommand()
	if !ok {
		result := submitForesightNativeSGLangUnavailableReceipt(ctx, client, work, runtimeMgr, gpuDetected, spec, started, "native_sglang_bridge_unavailable")
		return result, errNativeSGLangUnavailable
	}

	workDir, err := os.MkdirTemp("", "ryv_sglang_verifier_*")
	if err != nil {
		result := submitForesightNativeSGLangUnavailableReceipt(ctx, client, work, runtimeMgr, gpuDetected, spec, started, "native_sglang_workdir_failed")
		return result, err
	}
	defer os.RemoveAll(workDir)

	socketPath := filepath.Join(workDir, "verifier_session.sock")
	jobPayload := nativeSGLangJobPayload(work, spec)
	if err := writeJSONFile(filepath.Join(workDir, "job.json"), jobPayload); err != nil {
		result := submitForesightNativeSGLangUnavailableReceipt(ctx, client, work, runtimeMgr, gpuDetected, spec, started, "native_sglang_job_write_failed")
		return result, err
	}

	cmd := sglangverify.Command(ctx, commandSpec)
	cmd.Env = append(os.Environ(),
		"RYV_WORK_DIR="+workDir,
		"RYV_VERIFIER_SESSION_SOCKET="+socketPath,
		"RYV_EAGER_LOAD_SGLANG="+firstNonEmptyString(strings.TrimSpace(os.Getenv("RYV_EAGER_LOAD_SGLANG")), "0"),
		"TRANSFORMERS_OFFLINE=1",
		"HF_HUB_OFFLINE=1",
		"HF_DATASETS_OFFLINE=1",
	)
	if spec.ModelPath != "" {
		cmd.Env = append(cmd.Env, "RYV_MODEL_PATH="+spec.ModelPath)
	}
	if err := cmd.Start(); err != nil {
		result := submitForesightNativeSGLangUnavailableReceipt(ctx, client, work, runtimeMgr, gpuDetected, spec, started, "native_sglang_start_failed")
		return result, err
	}
	defer sglangverify.StopCommand(cmd)

	if err := sglangverify.WaitForSocket(ctx, socketPath, 30*time.Second); err != nil {
		result := submitForesightNativeSGLangUnavailableReceipt(ctx, client, work, runtimeMgr, gpuDetected, spec, started, "native_sglang_socket_timeout")
		return result, err
	}

	session, err := sglangverify.RunHotSession(ctx, client, socketPath, work.JobID, spec, 100*time.Millisecond)
	if err != nil {
		receiptCtx := ctx
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			receiptCtx = context.Background()
		}
		result := submitForesightNativeSGLangFinalReceipt(receiptCtx, client, work, runtimeMgr, gpuDetected, spec, started, workDir, session.TotalAccepted, session.Waves, session.AcceptedText, session.FinalReason)
		return result, err
	}
	result := submitForesightNativeSGLangFinalReceipt(ctx, client, work, runtimeMgr, gpuDetected, spec, started, workDir, session.TotalAccepted, session.Waves, session.AcceptedText, session.FinalReason)
	return result, nil
}

func nativeSGLangJobPayload(work *hub.WorkAssignment, spec foresightNativeHotSessionSpec) map[string]any {
	return map[string]any{
		"schema_version":    "ryvion.native_sglang_verifier_job.v1",
		"task":              firstNonEmptyString(spec.Task, foresightVerifierHotSessionTask),
		"job_id":            work.JobID,
		"workgraph_id":      spec.WorkGraphID,
		"session_id":        spec.SessionID,
		"model_id":          spec.ModelID,
		"model_hash":        spec.ModelHash,
		"model_path":        spec.ModelPath,
		"verifier_backend":  foresightVerifierBackendSGLang,
		"network":           "offline",
		"docker_required":   false,
		"runner_contract":   "VerifierSessionContract.v8",
		"raw_kv_forbidden":  true,
		"raw_prompt_stored": false,
		"session":           nativeSGLangSessionPayload(spec),
	}
}

func nativeSGLangSessionPayload(spec foresightNativeHotSessionSpec) map[string]any {
	return map[string]any{
		"session_id":   spec.SessionID,
		"workgraph_id": spec.WorkGraphID,
		"role_id":      spec.RoleID,
		"model_id":     spec.ModelID,
		"model_hash":   spec.ModelHash,
		"model_path":   spec.ModelPath,
		"prefix_hash":  spec.ParentPrefixHash,
	}
}

func submitForesightNativeSGLangUnavailableReceipt(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, runtimeMgr *runtimeManager, gpuDetected bool, spec foresightNativeHotSessionSpec, started time.Time, errorCode string) *runnerResultSnapshot {
	resultHash := foresightFullHash(fmt.Sprintf("%s|%s|%s", work.JobID, foresightVerifierBackendSGLang, errorCode))
	metadata := receiptMetadataBase(work, safeRuntimeReceiptMetadata(runtimeMgr, gpuDetected), map[string]any{
		"executor":              nativeSGLangVerifierExecutor,
		"executor_kind":         nativeSGLangVerifierExecutor,
		"task":                  firstNonEmptyString(spec.Task, foresightVerifierHotSessionTask),
		"docker_required":       false,
		"runtime_mode":          "native_node_agent",
		"verifier_backend":      foresightVerifierBackendSGLang,
		"status":                "unavailable",
		"execution_status":      "unavailable",
		"billing_status":        "not_billable",
		"proof_status":          "native_sglang_unavailable",
		"error_code":            errorCode,
		"install_hint":          "install native SGLang verifier bridge or set RYV_SGLANG_VERIFIER_CMD / RYV_SGLANG_VERIFIER_SCRIPT",
		"model_path_configured": spec.ModelPath != "",
		"exit_code":             1,
		"duration_ms":           time.Since(started).Milliseconds(),
	})
	_ = submitReceiptWithRetry(ctx, client, hub.Receipt{JobID: work.JobID, ResultHashHex: resultHash, MeteringUnits: 0, Metadata: metadata})
	return &runnerResultSnapshot{DurationMs: time.Since(started).Milliseconds(), ResultHashHex: resultHash, MeteringUnits: 0, ExitCode: 1, Metadata: metadata}
}

func submitForesightNativeSGLangFinalReceipt(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, runtimeMgr *runtimeManager, gpuDetected bool, spec foresightNativeHotSessionSpec, started time.Time, workDir string, accepted int, waves int, acceptedText string, reason string) *runnerResultSnapshot {
	verifierReceipt := readJSONMap(filepath.Join(workDir, "verifier_session_receipt.json"))
	if len(verifierReceipt) == 0 {
		verifierReceipt = readJSONMap(filepath.Join(workDir, "verifier_session_receipt.partial.json"))
	}
	probeSummary := readJSONMap(filepath.Join(workDir, "probe_summary.json"))
	if len(probeSummary) == 0 {
		probeSummary = readJSONMap(filepath.Join(workDir, "probe_summary.partial.json"))
	}
	if accepted <= 0 {
		accepted = intFromAny(verifierReceipt["accepted_len"])
	}
	acceptedTextHash := redactForesightAcceptedTextReceipt(verifierReceipt)
	if acceptedTextHash == "" {
		acceptedTextHash = foresightAcceptedTextHash(acceptedText)
	}
	resultHash := foresightFullHash(fmt.Sprintf("%s|%s|native_sglang|%d|%d|%s", work.JobID, spec.RunID, accepted, waves, reason))
	metadata := receiptMetadataBase(work, safeRuntimeReceiptMetadata(runtimeMgr, gpuDetected), map[string]any{
		"executor":         nativeSGLangVerifierExecutor,
		"executor_kind":    nativeSGLangVerifierExecutor,
		"task":             firstNonEmptyString(spec.Task, foresightVerifierHotSessionTask),
		"docker_required":  false,
		"runtime_mode":     "native_node_agent",
		"verifier_backend": foresightVerifierBackendSGLang,
		"session_mode":     "hot",
		"run_id":           spec.RunID,
		"session_id":       spec.SessionID,
		"workgraph_id":     spec.WorkGraphID,
		"wave_count":       waves,
		"duration_ms":      time.Since(started).Milliseconds(),
		"exit_code":        0,
		"stop_reason":      firstNonEmptyString(strings.TrimSpace(reason), "completed"),
		"verifier_session": map[string]any{
			"duration_ms":            time.Since(started).Milliseconds(),
			"accepted_token_receipt": verifierReceipt,
			"probe_summary":          probeSummary,
		},
	})
	if acceptedTextHash != "" {
		metadata["accepted_text_hash"] = acceptedTextHash
	}
	_ = submitReceiptWithRetry(ctx, client, hub.Receipt{JobID: work.JobID, ResultHashHex: resultHash, MeteringUnits: uint64(maxIntNode(0, accepted)), Metadata: metadata})
	return &runnerResultSnapshot{DurationMs: time.Since(started).Milliseconds(), ResultHashHex: resultHash, MeteringUnits: uint64(maxIntNode(0, accepted)), ExitCode: 0, Metadata: metadata}
}

func writeJSONFile(path string, payload any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	return os.WriteFile(path, raw, 0o600)
}

func readJSONMap(path string) map[string]any {
	raw, err := os.ReadFile(path)
	if err != nil {
		return map[string]any{}
	}
	var out map[string]any
	if json.Unmarshal(raw, &out) != nil {
		return map[string]any{}
	}
	return out
}

func firstPositiveInt64(values ...int64) int64 {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func int64FromAny(value any) int64 {
	switch typed := value.(type) {
	case int:
		return int64(typed)
	case int64:
		return typed
	case int32:
		return int64(typed)
	case float64:
		return int64(typed)
	case float32:
		return int64(typed)
	case json.Number:
		parsed, _ := typed.Int64()
		return parsed
	default:
		return 0
	}
}

func boolFromAny(value any) bool {
	switch typed := value.(type) {
	case bool:
		return typed
	case string:
		switch strings.ToLower(strings.TrimSpace(typed)) {
		case "1", "true", "yes", "on":
			return true
		default:
			return false
		}
	default:
		return false
	}
}
