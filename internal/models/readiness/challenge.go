package readiness

import (
	"errors"
	"fmt"
	"strings"
)

var ErrInvalidReadinessChallenge = errors.New("readiness: invalid challenge")

type ReadinessChallenge struct {
	ChallengeID          string `json:"challenge_id"`
	NodeID               string `json:"node_id"`
	ModelID              string `json:"model_id"`
	QuantizationID       string `json:"quantization_id,omitempty"`
	IssuedAtUnixMs       int64  `json:"issued_at_unix_ms"`
	ExpiresAtUnixMs      int64  `json:"expires_at_unix_ms"`
	RequiredLatencyMs    int64  `json:"required_latency_ms"`
	PromptHash           string `json:"prompt_hash,omitempty"`
	SparseLogitCheckHash string `json:"sparse_logit_check_hash,omitempty"`
	Nonce                string `json:"nonce"`
}

func validateReadinessChallenge(challenge ReadinessChallenge) error {
	var errs []error
	if readinessStringRequired(challenge.ChallengeID) == "" {
		errs = append(errs, fmt.Errorf("%w: challenge_id required", ErrInvalidReadinessChallenge))
	}
	if readinessStringRequired(challenge.NodeID) == "" {
		errs = append(errs, fmt.Errorf("%w: node_id required", ErrInvalidReadinessChallenge))
	}
	if readinessStringRequired(challenge.ModelID) == "" {
		errs = append(errs, fmt.Errorf("%w: model_id required", ErrInvalidReadinessChallenge))
	}
	if readinessStringRequired(challenge.Nonce) == "" {
		errs = append(errs, fmt.Errorf("%w: nonce required", ErrInvalidReadinessChallenge))
	}
	if challenge.IssuedAtUnixMs < 0 {
		errs = append(errs, fmt.Errorf("%w: issued_at_unix_ms must be non-negative", ErrInvalidReadinessChallenge))
	}
	if challenge.ExpiresAtUnixMs < 0 {
		errs = append(errs, fmt.Errorf("%w: expires_at_unix_ms must be non-negative", ErrInvalidReadinessChallenge))
	}
	if challenge.RequiredLatencyMs < 0 {
		errs = append(errs, fmt.Errorf("%w: required_latency_ms must be non-negative", ErrInvalidReadinessChallenge))
	}
	return errors.Join(errs...)
}

func readinessStringRequired(value string) string {
	return strings.TrimSpace(value)
}
