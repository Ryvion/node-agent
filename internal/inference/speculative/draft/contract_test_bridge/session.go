package contracttestbridge

import (
	"context"
	"strings"
	"time"

	"github.com/Ryvion/ryvion-node/internal/hub"
	nodespec "github.com/Ryvion/ryvion-node/internal/inference/speculative"
)

type LiveLabDraftClient interface {
	FetchSpeculativeLiveLabDraftCommand(ctx context.Context, runID string, jobID string) (hub.SpeculativeLiveLabSessionCommand, error)
	SubmitSpeculativeDraftPacketBatch(ctx context.Context, windowID string, packets []map[string]any) (hub.DraftPacketBatchDecision, error)
}

type HotSessionResult struct {
	AcceptedPackets int
	RawPackets      int
	Waves           int
}

func RunHotSession(ctx context.Context, client LiveLabDraftClient, spec nodespec.HotSessionSpec, jobID string, pollInterval time.Duration) (HotSessionResult, error) {
	if pollInterval <= 0 {
		pollInterval = 100 * time.Millisecond
	}
	result := HotSessionResult{}
	submittedWindows := map[string]bool{}
	ticker := time.NewTicker(pollInterval)
	defer ticker.Stop()

	for {
		command, err := client.FetchSpeculativeLiveLabDraftCommand(ctx, spec.RunID, jobID)
		if err != nil {
			select {
			case <-ctx.Done():
				return result, ctx.Err()
			case <-ticker.C:
				continue
			}
		}
		switch strings.TrimSpace(command.Command) {
		case "close_session":
			return result, nil
		case "generate_draft_packets":
			windowID := strings.TrimSpace(command.WindowID)
			if windowID != "" && !submittedWindows[windowID] {
				draftSpec := nodespec.DraftSpec{
					Task:                 nodespec.DraftRunnerTask,
					WorkGraphID:          firstString(command.WorkGraphID, spec.WorkGraphID),
					WindowID:             windowID,
					RoleID:               firstString(command.RoleID, spec.RoleID),
					TargetNodeID:         firstString(command.TargetNodeID, spec.TargetNodeID),
					NodeID:               firstString(command.NodeID, spec.NodeID),
					Prompt:               firstString(command.Prompt, spec.Prompt),
					ParentPrefixHash:     firstString(command.ParentPrefixHash, spec.ParentPrefixHash),
					BranchCount:          command.BranchCount,
					Horizon:              command.Horizon,
					DeadlineMs:           command.DeadlineMs,
					ModelHash:            firstString(command.ModelHash, spec.ModelHash),
					DrafterModelID:       firstString(command.DrafterModelID, spec.DrafterModelID),
					FirstPacketTimeoutMs: command.FirstPacketTimeout,
				}
				packets := BuildPackets(draftSpec)
				summary, _ := client.SubmitSpeculativeDraftPacketBatch(ctx, windowID, packets)
				result.AcceptedPackets += summary.Accepted
				result.RawPackets += len(packets)
				result.Waves++
				submittedWindows[windowID] = true
			}
		}
		select {
		case <-ctx.Done():
			return result, ctx.Err()
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
