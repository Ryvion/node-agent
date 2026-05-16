package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/Ryvion/node-agent/internal/hub"
	v7llamacpp "github.com/Ryvion/node-agent/internal/v7/llamacpp"
)

const nativeLlamaCppVerifierExecutor = "native_llamacpp_verifier"

var errNativeLlamaCppUnavailable = errors.New("native_llamacpp_verifier_unavailable")

var newForesightNativeLlamaCppVerifier = func() nativeLlamaCppVerifier {
	return nativeLlamaCppVerifier{
		Sidecar: v7llamacpp.NewManagerFromEnv(),
		Client:  v7llamacpp.OpenAIClient{},
	}
}

type nativeLlamaCppSidecar interface {
	Start(context.Context) v7llamacpp.LlamaCppSidecarStatus
	Status(context.Context) v7llamacpp.LlamaCppSidecarStatus
}

type nativeLlamaCppVerifier struct {
	Sidecar nativeLlamaCppSidecar
	Client  v7llamacpp.CompletionClient
}

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
					unavailable := submitForesightNativeLlamaCppUnavailableReceipt(ctx, client, work, runtimeMgr, gpuDetected, spec, started, nativeLlamaCppUnavailableCode(verifier.sidecar().Status(ctx)))
					return unavailable, err
				}
				failed := submitForesightNativeLlamaCppFinalReceipt(ctx, client, work, runtimeMgr, gpuDetected, spec, started, totalAccepted, waves, acceptedText.String(), "native_llamacpp_verify_failed")
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

func (v nativeLlamaCppVerifier) VerifyWave(ctx context.Context, spec foresightNativeHotSessionSpec, command hub.ForesightLiveLabSessionCommand, acceptedTokensTotal int) (hub.ForesightLiveLabVerifierResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	started := time.Now()
	sidecar := v.sidecar()
	status := sidecar.Start(ctx)
	if !nativeLlamaCppStatusReady(status) {
		return hub.ForesightLiveLabVerifierResult{}, errNativeLlamaCppUnavailable
	}
	acceptedLimit, treeCID := foresightAcceptedFromCommandTree(command)
	remaining := 0
	if spec.MaxTokens > 0 {
		remaining = spec.MaxTokens - maxIntNode(0, acceptedTokensTotal)
		if remaining <= 0 {
			remaining = 1
		}
	}
	maxTokens := acceptedLimit
	if remaining > 0 && remaining < maxTokens {
		maxTokens = remaining
	}
	if maxTokens <= 0 {
		maxTokens = 1
	}
	req := v7llamacpp.CompletionRequest{
		BaseURL:      status.BaseURL,
		ModelID:      firstNonEmptyString(spec.ModelID, status.ModelFilename, v7llamacpp.BackendName),
		SystemPrompt: nativeLlamaCppVerifierSystemPrompt(),
		Prompt:       nativeLlamaCppVerifierPrompt(spec, command, treeCID, acceptedLimit),
		MaxTokens:    maxTokens,
		Temperature:  0,
		Stream:       status.SupportsStreaming,
	}
	completion, err := v.client().Complete(ctx, req)
	if err != nil {
		return hub.ForesightLiveLabVerifierResult{}, err
	}
	text := strings.TrimSpace(string(completion.Output))
	tokensGenerated := int(completion.TokensGenerated)
	if tokensGenerated <= 0 {
		tokensGenerated = len(strings.Fields(text))
	}
	acceptedLen := minPositiveInt(tokensGenerated, acceptedLimit, maxTokens)
	if acceptedLen <= 0 {
		acceptedLen = minPositiveInt(acceptedLimit, maxTokens, 1)
	}
	durationMs := completion.TotalTimeMs
	if durationMs <= 0 {
		durationMs = maxInt64Node(1, time.Since(started).Milliseconds())
	}
	stopReason := firstNonEmptyString(completion.FinishReason, completion.BackendStopReason, completion.BackendFinishReason)
	result := hub.ForesightLiveLabVerifierResult{
		WindowID:           command.WindowID,
		WaveIndex:          command.WaveIndex,
		AcceptedLen:        acceptedLen,
		TreeCID:            treeCID,
		DurationMs:         durationMs,
		AcceptedText:       text,
		AcceptedTextPublic: text != "",
		EOS:                strings.EqualFold(stopReason, "stop") || strings.EqualFold(stopReason, "eos"),
		StopReason:         stopReason,
		ProbeSummary:       nativeLlamaCppProbeSummary(status, req, completion, durationMs),
	}
	if completion.MaxTokensReached && result.StopReason == "" {
		result.StopReason = "max_tokens"
	}
	return result, nil
}

func (v nativeLlamaCppVerifier) sidecar() nativeLlamaCppSidecar {
	if v.Sidecar != nil {
		return v.Sidecar
	}
	return v7llamacpp.NewManagerFromEnv()
}

func (v nativeLlamaCppVerifier) client() v7llamacpp.CompletionClient {
	if v.Client != nil {
		return v.Client
	}
	return v7llamacpp.OpenAIClient{}
}

func nativeLlamaCppStatusReady(status v7llamacpp.LlamaCppSidecarStatus) bool {
	return status.Enabled && status.Available && status.Running && status.Healthy && status.BaseURL != "" && status.SupportsTextGeneration
}

func nativeLlamaCppVerifierSystemPrompt() string {
	return "You are Ryvion's local llama.cpp verifier. Verify the speculative draft tree against the prompt and return only the accepted continuation text."
}

func nativeLlamaCppVerifierPrompt(spec foresightNativeHotSessionSpec, command hub.ForesightLiveLabSessionCommand, treeCID string, acceptedLimit int) string {
	branchCount := len(sliceFromAny(command.Tree["branches"]))
	return strings.Join([]string{
		"Verify this Ryvion speculative draft wave locally.",
		"run_id=" + spec.RunID,
		"session_id=" + spec.SessionID,
		"workgraph_id=" + firstNonEmptyString(command.WorkGraphID, spec.WorkGraphID),
		"window_id=" + command.WindowID,
		"prefix_hash=" + spec.ParentPrefixHash,
		"tree_cid=" + treeCID,
		fmt.Sprintf("branch_count=%d", branchCount),
		fmt.Sprintf("candidate_acceptance_limit=%d", acceptedLimit),
		"User prompt:",
		spec.Prompt,
	}, "\n")
}

func nativeLlamaCppProbeSummary(status v7llamacpp.LlamaCppSidecarStatus, req v7llamacpp.CompletionRequest, completion v7llamacpp.CompletionResult, durationMs int64) map[string]any {
	return map[string]any{
		"source":                     "native_llamacpp_verifier",
		"backend":                    v7llamacpp.BackendName,
		"runtime_mode":               "native_node_agent",
		"model_id":                   req.ModelID,
		"model_filename":             status.ModelFilename,
		"output_hash":                "sha256:" + sha256Hex(completion.Output),
		"output_bytes":               completion.OutputBytes,
		"tokens_generated":           completion.TokensGenerated,
		"completion_tokens":          completion.CompletionTokens,
		"requested_max_tokens":       completion.RequestedMaxTokens,
		"finish_reason":              completion.FinishReason,
		"backend_finish_reason":      completion.BackendFinishReason,
		"backend_stop_reason":        completion.BackendStopReason,
		"max_tokens_reached":         completion.MaxTokensReached,
		"runtime_measurement_status": completion.RuntimeMeasurementStatus,
		"metadata_parse_status":      completion.MetadataParseStatus,
		"ttft_ms":                    completion.TTFTMs,
		"duration_ms":                durationMs,
		"streamed":                   completion.Streamed,
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
		"proof_status":          "native_llamacpp_unavailable",
		"error_code":            firstNonEmptyString(strings.TrimSpace(errorCode), "native_llamacpp_unavailable"),
		"install_hint":          "install llama-server or set RYV_LLAMA_CPP_SERVER_PATH / RYV_LLAMA_CPP_MODEL_PATH",
		"model_path_configured": spec.ModelPath != "",
		"exit_code":             1,
		"duration_ms":           time.Since(started).Milliseconds(),
	})
	_ = submitReceiptWithRetry(ctx, client, hub.Receipt{JobID: work.JobID, ResultHashHex: resultHash, MeteringUnits: 0, Metadata: metadata})
	return &runnerResultSnapshot{DurationMs: time.Since(started).Milliseconds(), ResultHashHex: resultHash, MeteringUnits: 0, ExitCode: 1, Metadata: metadata}
}

func submitForesightNativeLlamaCppFinalReceipt(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, runtimeMgr *runtimeManager, gpuDetected bool, spec foresightNativeHotSessionSpec, started time.Time, accepted int, waves int, acceptedText string, reason string) *runnerResultSnapshot {
	resultHash := foresightFullHash(fmt.Sprintf("%s|%s|native_llamacpp|%d|%d|%s", work.JobID, spec.RunID, accepted, waves, reason))
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
				"accepted_text":         acceptedText,
				"accepted_text_public":  strings.TrimSpace(acceptedText) != "",
				"accepted_text_hash":    outputHash,
				"tree_cid":              "sha256:" + foresightFullHash(fmt.Sprintf("%s|%s|native_llamacpp_final_tree", work.JobID, spec.RunID)),
				"hot_session_finalized": true,
			},
			"probe_summary": map[string]any{
				"source":      "native_llamacpp_verifier",
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

func nativeLlamaCppUnavailableCode(status v7llamacpp.LlamaCppSidecarStatus) string {
	switch {
	case !status.Enabled:
		return "native_llamacpp_disabled"
	case !status.Available:
		return "native_llamacpp_unavailable"
	case !status.Running:
		return "native_llamacpp_not_running"
	case !status.Healthy:
		return "native_llamacpp_unhealthy"
	case status.BaseURL == "":
		return "native_llamacpp_base_url_missing"
	case !status.SupportsTextGeneration:
		return "native_llamacpp_text_generation_unavailable"
	default:
		return "native_llamacpp_unavailable"
	}
}

func minPositiveInt(values ...int) int {
	out := 0
	for _, value := range values {
		if value <= 0 {
			continue
		}
		if out == 0 || value < out {
			out = value
		}
	}
	return out
}

func sha256Hex(value []byte) string {
	sum := sha256.Sum256(value)
	return hex.EncodeToString(sum[:])
}
