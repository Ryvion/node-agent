package sglang

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/Ryvion/ryvion-node/internal/hub"
	nodespec "github.com/Ryvion/ryvion-node/internal/inference/speculative"
)

type LiveLabVerifierClient interface {
	FetchForesightLiveLabVerifierCommand(ctx context.Context, runID string, jobID string) (hub.ForesightLiveLabSessionCommand, error)
	SubmitForesightLiveLabVerifierResult(ctx context.Context, runID string, result hub.ForesightLiveLabVerifierResult) error
}

type HotSessionResult struct {
	AcceptedText  string
	FinalReason   string
	TotalAccepted int
	Waves         int
}

func RunHotSession(ctx context.Context, client LiveLabVerifierClient, socketPath string, jobID string, spec nodespec.HotSessionSpec, pollInterval time.Duration) (HotSessionResult, error) {
	if pollInterval <= 0 {
		pollInterval = 100 * time.Millisecond
	}
	state := HotSessionResult{}
	sessionStarted := false
	prefilled := false
	verifiedCommands := map[string]bool{}
	var acceptedText strings.Builder
	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()

	for {
		command, err := client.FetchForesightLiveLabVerifierCommand(ctx, spec.RunID, jobID)
		if err != nil {
			select {
			case <-ctx.Done():
				_, _ = RPC(context.Background(), socketPath, "abort", map[string]any{"reason": ctx.Err().Error()})
				state.AcceptedText = acceptedText.String()
				state.FinalReason = "aborted"
				return state, ctx.Err()
			case <-ticker.C:
				continue
			}
		}
		switch strings.TrimSpace(command.Command) {
		case "close_session":
			_, _ = RPC(ctx, socketPath, "close_session", map[string]any{"session_id": spec.SessionID, "reason": command.Reason})
			state.AcceptedText = acceptedText.String()
			state.FinalReason = command.Reason
			return state, nil
		case "verify_tree":
			commandID := firstString(command.CommandID, fmt.Sprintf("%s:%s:%d", spec.RunID, command.WindowID, command.WaveIndex))
			if verifiedCommands[commandID] {
				break
			}
			if !sessionStarted {
				if _, err := RPC(ctx, socketPath, "start_session", map[string]any{"session": sessionPayload(spec)}); err != nil {
					state.AcceptedText = acceptedText.String()
					state.FinalReason = "start_session_failed"
					return state, err
				}
				sessionStarted = true
			}
			if !prefilled {
				if _, err := RPC(ctx, socketPath, "prefill", map[string]any{"session_id": spec.SessionID, "prefix_hash": spec.ParentPrefixHash, "prefix_tokens": []int{}}); err != nil {
					state.AcceptedText = acceptedText.String()
					state.FinalReason = "prefill_failed"
					return state, err
				}
				prefilled = true
			}
			waveStarted := time.Now()
			verifyResult, err := RPC(ctx, socketPath, "verify_tree", map[string]any{
				"session_id": spec.SessionID,
				"session":    sessionPayload(spec),
				"tree":       command.Tree,
			})
			if err != nil {
				state.AcceptedText = acceptedText.String()
				state.FinalReason = "verify_tree_failed"
				return state, err
			}
			acceptedReceipt := mapFromAny(verifyResult["accepted_token_receipt"])
			probeSummary := mapFromAny(verifyResult["probe_summary"])
			acceptedLen := intFromAny(acceptedReceipt["accepted_len"])
			treeCID := strings.TrimSpace(stringValue(acceptedReceipt["tree_cid"]))
			acceptedTextChunk := strings.TrimSpace(stringValue(acceptedReceipt["accepted_text"]))
			if acceptedTextChunk != "" {
				acceptedText.WriteString(acceptedTextChunk)
			}
			commitParams := map[string]any{"session_id": spec.SessionID, "accepted_len": acceptedLen}
			if tokens := sliceFromAny(acceptedReceipt["accepted_token_ids"]); len(tokens) > 0 {
				commitParams["accepted_token_ids"] = tokens
			}
			if acceptedLen > 0 {
				_, _ = RPC(ctx, socketPath, "commit", commitParams)
			} else {
				_, _ = RPC(ctx, socketPath, "rollback", map[string]any{"session_id": spec.SessionID, "branch_ids": acceptedReceipt["rollback_branch_ids"]})
			}
			durationMs := maxInt64(1, firstPositiveInt64(int64FromAny(acceptedReceipt["latency_ms"]), time.Since(waveStarted).Milliseconds()))
			result := hub.ForesightLiveLabVerifierResult{
				JobID:              jobID,
				WindowID:           command.WindowID,
				WaveIndex:          command.WaveIndex,
				AcceptedLen:        acceptedLen,
				TreeCID:            treeCID,
				DurationMs:         durationMs,
				AcceptedText:       acceptedTextChunk,
				AcceptedTextPublic: acceptedTextChunk != "",
				EOS:                boolFromAny(acceptedReceipt["eos"]),
				StopReason:         strings.TrimSpace(stringValue(acceptedReceipt["stop_reason"])),
				ProbeSummary:       probeSummary,
			}
			if spec.MaxTokens > 0 && state.TotalAccepted+acceptedLen >= spec.MaxTokens && result.StopReason == "" {
				result.StopReason = "max_tokens"
			}
			if err := client.SubmitForesightLiveLabVerifierResult(ctx, spec.RunID, result); err != nil {
				state.AcceptedText = acceptedText.String()
				state.FinalReason = "result_submit_failed"
				return state, err
			}
			state.TotalAccepted += acceptedLen
			state.Waves++
			verifiedCommands[commandID] = true
		}
		select {
		case <-ctx.Done():
			_, _ = RPC(context.Background(), socketPath, "abort", map[string]any{"reason": ctx.Err().Error()})
			state.AcceptedText = acceptedText.String()
			state.FinalReason = "aborted"
			return state, ctx.Err()
		case <-ticker.C:
		}
	}
}

func sessionPayload(spec nodespec.HotSessionSpec) map[string]any {
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

func firstString(value string, fallback string) string {
	if strings.TrimSpace(value) != "" {
		return strings.TrimSpace(value)
	}
	return strings.TrimSpace(fallback)
}

func firstPositiveInt64(values ...int64) int64 {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
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
	case jsonNumber:
		parsed, _ := typed.Int64()
		return int(parsed)
	default:
		return 0
	}
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
	case jsonNumber:
		parsed, _ := typed.Int64()
		return parsed
	default:
		return 0
	}
}

type jsonNumber interface {
	Int64() (int64, error)
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

func sliceFromAny(value any) []any {
	if typed, ok := value.([]any); ok {
		return typed
	}
	return nil
}

func maxInt64(left int64, right int64) int64 {
	if left > right {
		return left
	}
	return right
}
