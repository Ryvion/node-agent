package contracttestbridge

import (
	"time"

	"github.com/Ryvion/ryvion-node/internal/hub"
	nodespec "github.com/Ryvion/ryvion-node/internal/inference/speculative"
)

const (
	Backend = nodespec.VerifierBackendBridge
	Source  = "contract_test_bridge_verifier"
)

func VerifyWave(jobID string, spec nodespec.HotSessionSpec, command hub.ForesightLiveLabSessionCommand, acceptedTokensTotal int) hub.ForesightLiveLabVerifierResult {
	started := time.Now()
	acceptedLen, treeCID := nodespec.AcceptedFromTree(command.Tree, command.CommandID)
	text := nodespec.AcceptedTextForWave(spec.Prompt, command.WaveIndex, acceptedLen)
	result := hub.ForesightLiveLabVerifierResult{
		JobID:              jobID,
		WindowID:           command.WindowID,
		WaveIndex:          command.WaveIndex,
		AcceptedLen:        acceptedLen,
		TreeCID:            treeCID,
		DurationMs:         maxInt64(1, time.Since(started).Milliseconds()),
		AcceptedText:       text,
		AcceptedTextPublic: text != "",
		EOS:                false,
		ProbeSummary:       ProbeSummary(),
	}
	if spec.MaxTokens > 0 && acceptedTokensTotal+acceptedLen >= spec.MaxTokens {
		result.StopReason = "max_tokens"
	}
	return result
}

func ProbeSummary() map[string]any {
	return map[string]any{
		"confidence_bps":   8200,
		"source":           Source,
		"backend":          Backend,
		"production_valid": false,
		"test_adapter":     true,
		"billing_status":   "not_billable_contract_test",
	}
}

func maxInt64(left int64, right int64) int64 {
	if left > right {
		return left
	}
	return right
}
