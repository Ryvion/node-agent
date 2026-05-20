package llamacppdemo

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/Ryvion/ryvion-node/internal/hub"
	nodespec "github.com/Ryvion/ryvion-node/internal/inference/speculative"
)

type LiveLabVerifierClient interface {
	FetchSpeculativeLiveLabVerifierCommand(ctx context.Context, runID string, jobID string) (hub.SpeculativeLiveLabSessionCommand, error)
	SubmitSpeculativeLiveLabVerifierResult(ctx context.Context, runID string, result hub.SpeculativeLiveLabVerifierResult) error
}

type HotSessionResult struct {
	AcceptedText  string
	FinalReason   string
	TotalAccepted int
	Waves         int
}

func RunHotSession(ctx context.Context, client LiveLabVerifierClient, verifier Verifier, jobID string, spec nodespec.HotSessionSpec, pollInterval time.Duration) (HotSessionResult, error) {
	if pollInterval <= 0 {
		pollInterval = 100 * time.Millisecond
	}
	state := HotSessionResult{}
	verifiedCommands := map[string]bool{}
	var acceptedText strings.Builder
	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()

	for {
		command, err := client.FetchSpeculativeLiveLabVerifierCommand(ctx, spec.RunID, jobID)
		if err != nil {
			select {
			case <-ctx.Done():
				state.AcceptedText = acceptedText.String()
				state.FinalReason = "aborted"
				return state, ctx.Err()
			case <-ticker.C:
				continue
			}
		}
		switch strings.TrimSpace(command.Command) {
		case "close_session":
			state.AcceptedText = acceptedText.String()
			state.FinalReason = command.Reason
			return state, nil
		case "verify_tree":
			commandID := firstString(command.CommandID, fmt.Sprintf("%s:%s:%d", spec.RunID, command.WindowID, command.WaveIndex))
			if verifiedCommands[commandID] {
				break
			}
			result, err := verifier.VerifyWave(ctx, spec, command, state.TotalAccepted)
			if err != nil {
				state.AcceptedText = acceptedText.String()
				state.FinalReason = "llamacpp_demo_verify_failed"
				return state, err
			}
			result.JobID = jobID
			if spec.MaxTokens > 0 && state.TotalAccepted+result.AcceptedLen >= spec.MaxTokens && result.StopReason == "" {
				result.StopReason = "max_tokens"
			}
			if err := client.SubmitSpeculativeLiveLabVerifierResult(ctx, spec.RunID, result); err != nil {
				state.AcceptedText = acceptedText.String()
				state.FinalReason = "result_submit_failed"
				return state, err
			}
			state.TotalAccepted += result.AcceptedLen
			state.Waves++
			if strings.TrimSpace(result.AcceptedText) != "" {
				acceptedText.WriteString(result.AcceptedText)
			}
			verifiedCommands[commandID] = true
		}
		select {
		case <-ctx.Done():
			state.AcceptedText = acceptedText.String()
			state.FinalReason = "aborted"
			return state, ctx.Err()
		case <-ticker.C:
		}
	}
}

func firstString(value string, fallback string) string {
	if strings.TrimSpace(value) != "" {
		return strings.TrimSpace(value)
	}
	return strings.TrimSpace(fallback)
}
