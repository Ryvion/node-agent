package contracttestbridge

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

func RunHotSession(ctx context.Context, client LiveLabVerifierClient, jobID string, spec nodespec.HotSessionSpec, pollInterval time.Duration) (HotSessionResult, error) {
	if pollInterval <= 0 {
		pollInterval = 100 * time.Millisecond
	}
	state := HotSessionResult{}
	verifiedCommands := map[string]bool{}
	var acceptedText strings.Builder
	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()

	for {
		command, err := client.FetchForesightLiveLabVerifierCommand(ctx, spec.RunID, jobID)
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
			if !verifiedCommands[commandID] {
				result := VerifyWave(jobID, spec, command, state.TotalAccepted)
				if err := client.SubmitForesightLiveLabVerifierResult(ctx, spec.RunID, result); err == nil {
					state.TotalAccepted += result.AcceptedLen
					state.Waves++
					if strings.TrimSpace(result.AcceptedText) != "" {
						acceptedText.WriteString(result.AcceptedText)
					}
					verifiedCommands[commandID] = true
				}
			}
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
