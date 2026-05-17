package readiness

import (
	"errors"
	"fmt"

	"github.com/Ryvion/ryvion-node/internal/v7/modellease"
)

var (
	ErrInvalidReadinessLocalState = errors.New("readiness: invalid local state")
	ErrReadinessRejected          = errors.New("readiness: rejected challenge")
)

type ReadinessLocalState struct {
	NodeID               string                     `json:"node_id"`
	ResidentModelID      string                     `json:"resident_model_id,omitempty"`
	QuantizationID       string                     `json:"quantization_id,omitempty"`
	ModelLeaseState      modellease.ModelLeaseState `json:"model_lease_state"`
	LastWarmLatencyMs    int64                      `json:"last_warm_latency_ms"`
	SparseLogitCheckHash string                     `json:"sparse_logit_check_hash,omitempty"`
}

type LocalReadinessState = ReadinessLocalState

type ReadinessEvaluator struct{}

func EvaluateReadinessChallenge(challenge ReadinessChallenge, localState ReadinessLocalState, nowUnixMs int64) (ReadinessResponse, error) {
	return ReadinessEvaluator{}.Evaluate(challenge, localState, nowUnixMs)
}

func (e ReadinessEvaluator) EvaluateReadinessChallenge(challenge ReadinessChallenge, localState ReadinessLocalState, nowUnixMs int64) (ReadinessResponse, error) {
	return e.Evaluate(challenge, localState, nowUnixMs)
}

func (ReadinessEvaluator) Evaluate(challenge ReadinessChallenge, localState ReadinessLocalState, nowUnixMs int64) (ReadinessResponse, error) {
	if err := validateReadinessChallenge(challenge); err != nil {
		return ReadinessResponse{}, err
	}

	response := baseReadinessResponse(challenge, localState, nowUnixMs)
	if nowUnixMs < 0 {
		response.Decision = ReadinessDecisionRejected
		response.Reason = ReadinessReasonRejected
		return response, fmt.Errorf("%w: now_unix_ms must be non-negative", ErrInvalidReadinessLocalState)
	}

	localNodeID := readinessStringRequired(localState.NodeID)
	if localNodeID == "" {
		response.Decision = ReadinessDecisionRejected
		response.Reason = ReadinessReasonInvalidLocalState
		return response, fmt.Errorf("%w: node_id required", ErrInvalidReadinessLocalState)
	}
	if localNodeID != readinessStringRequired(challenge.NodeID) {
		response.Decision = ReadinessDecisionRejected
		response.Reason = ReadinessReasonMismatch
		return response, fmt.Errorf("%w: node_id mismatch", ErrReadinessRejected)
	}

	if challenge.ExpiresAtUnixMs <= nowUnixMs {
		response.Decision = ReadinessDecisionExpired
		response.Reason = ReadinessReasonExpired
		return response, nil
	}

	if readinessStringRequired(localState.ResidentModelID) == "" {
		response.Decision = ReadinessDecisionNotReady
		response.Reason = ReadinessReasonNotResident
		return response, nil
	}
	if readinessStringRequired(localState.ResidentModelID) != readinessStringRequired(challenge.ModelID) {
		response.Decision = ReadinessDecisionNotReady
		response.Reason = ReadinessReasonMismatch
		return response, nil
	}
	if challenge.QuantizationID != "" && readinessStringRequired(localState.QuantizationID) != readinessStringRequired(challenge.QuantizationID) {
		response.Decision = ReadinessDecisionNotReady
		response.Reason = ReadinessReasonMismatch
		return response, nil
	}
	if localState.ModelLeaseState != modellease.ModelLeaseStateResident {
		response.Decision = ReadinessDecisionNotReady
		response.Reason = ReadinessReasonNotResident
		return response, nil
	}
	if localState.LastWarmLatencyMs < 0 {
		response.Decision = ReadinessDecisionNotReady
		response.Reason = ReadinessReasonInvalidLocalState
		return response, nil
	}
	if localState.LastWarmLatencyMs > challenge.RequiredLatencyMs {
		response.Decision = ReadinessDecisionNotReady
		response.Reason = ReadinessReasonLatencyTooHigh
		return response, nil
	}
	if challenge.SparseLogitCheckHash != "" &&
		readinessStringRequired(localState.SparseLogitCheckHash) != readinessStringRequired(challenge.SparseLogitCheckHash) {
		response.Decision = ReadinessDecisionNotReady
		response.Reason = ReadinessReasonSparseLogitMismatch
		return response, nil
	}

	response.Decision = ReadinessDecisionReady
	response.Reason = ReadinessReasonReady
	return response, nil
}

func baseReadinessResponse(challenge ReadinessChallenge, localState ReadinessLocalState, nowUnixMs int64) ReadinessResponse {
	latencyMs := localState.LastWarmLatencyMs
	if latencyMs < 0 {
		latencyMs = 0
	}

	nodeID := readinessStringRequired(localState.NodeID)
	if nodeID == "" {
		nodeID = readinessStringRequired(challenge.NodeID)
	}

	return ReadinessResponse{
		ChallengeID:          readinessStringRequired(challenge.ChallengeID),
		NodeID:               nodeID,
		ModelID:              readinessStringRequired(challenge.ModelID),
		QuantizationID:       readinessStringRequired(challenge.QuantizationID),
		RespondedAtUnixMs:    nowUnixMs,
		LatencyMs:            latencyMs,
		SparseLogitCheckHash: readinessStringRequired(localState.SparseLogitCheckHash),
		Decision:             ReadinessDecisionNotReady,
		Reason:               ReadinessReasonNotResident,
		SignaturePlaceholder: SignaturePlaceholderUnsignedV1,
	}
}
