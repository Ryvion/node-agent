package readiness

import (
	"errors"
	"testing"

	"github.com/Ryvion/ryvion-node/internal/models/lease"
)

const readinessTestNowUnixMs int64 = 1_800_000_000_000

func TestEvaluateReadinessChallengeValidResidentModelReturnsReady(t *testing.T) {
	response, err := EvaluateReadinessChallenge(validReadinessChallenge(), validReadinessLocalState(), readinessTestNowUnixMs)
	if err != nil {
		t.Fatalf("EvaluateReadinessChallenge() error = %v, want nil", err)
	}

	assertReadinessResponse(t, response, ReadinessDecisionReady, ReadinessReasonReady)
	if response.LatencyMs != 125 {
		t.Fatalf("latency_ms = %d, want 125", response.LatencyMs)
	}
	if response.SignaturePlaceholder == "" {
		t.Fatalf("signature_placeholder is empty")
	}
}

func TestEvaluateReadinessChallengeExpiredReturnsExpired(t *testing.T) {
	challenge := validReadinessChallenge()
	challenge.ExpiresAtUnixMs = readinessTestNowUnixMs

	response, err := EvaluateReadinessChallenge(challenge, validReadinessLocalState(), readinessTestNowUnixMs)
	if err != nil {
		t.Fatalf("EvaluateReadinessChallenge() error = %v, want nil", err)
	}

	assertReadinessResponse(t, response, ReadinessDecisionExpired, ReadinessReasonExpired)
}

func TestEvaluateReadinessChallengeNodeMismatchRejected(t *testing.T) {
	localState := validReadinessLocalState()
	localState.NodeID = "node-2"

	response, err := EvaluateReadinessChallenge(validReadinessChallenge(), localState, readinessTestNowUnixMs)
	if !errors.Is(err, ErrReadinessRejected) {
		t.Fatalf("EvaluateReadinessChallenge() error = %v, want ErrReadinessRejected", err)
	}
	assertReadinessResponse(t, response, ReadinessDecisionRejected, ReadinessReasonMismatch)
}

func TestEvaluateReadinessChallengeModelMismatchNotReady(t *testing.T) {
	localState := validReadinessLocalState()
	localState.ResidentModelID = "mixtral-8x22b"

	response, err := EvaluateReadinessChallenge(validReadinessChallenge(), localState, readinessTestNowUnixMs)
	if err != nil {
		t.Fatalf("EvaluateReadinessChallenge() error = %v, want nil", err)
	}

	assertReadinessResponse(t, response, ReadinessDecisionNotReady, ReadinessReasonMismatch)
}

func TestEvaluateReadinessChallengeLeaseNotResidentNotReady(t *testing.T) {
	localState := validReadinessLocalState()
	localState.ModelLeaseState = modellease.ModelLeaseStateWarmup

	response, err := EvaluateReadinessChallenge(validReadinessChallenge(), localState, readinessTestNowUnixMs)
	if err != nil {
		t.Fatalf("EvaluateReadinessChallenge() error = %v, want nil", err)
	}

	assertReadinessResponse(t, response, ReadinessDecisionNotReady, ReadinessReasonNotResident)
}

func TestEvaluateReadinessChallengeLatencyTooHighNotReady(t *testing.T) {
	localState := validReadinessLocalState()
	localState.LastWarmLatencyMs = 151

	response, err := EvaluateReadinessChallenge(validReadinessChallenge(), localState, readinessTestNowUnixMs)
	if err != nil {
		t.Fatalf("EvaluateReadinessChallenge() error = %v, want nil", err)
	}

	assertReadinessResponse(t, response, ReadinessDecisionNotReady, ReadinessReasonLatencyTooHigh)
}

func TestEvaluateReadinessChallengeSparseHashMismatchNotReady(t *testing.T) {
	localState := validReadinessLocalState()
	localState.SparseLogitCheckHash = "sha256:sparse-logits-other"

	response, err := EvaluateReadinessChallenge(validReadinessChallenge(), localState, readinessTestNowUnixMs)
	if err != nil {
		t.Fatalf("EvaluateReadinessChallenge() error = %v, want nil", err)
	}

	assertReadinessResponse(t, response, ReadinessDecisionNotReady, ReadinessReasonSparseLogitMismatch)
}

func TestEvaluateReadinessChallengeMissingResidentModelNotReady(t *testing.T) {
	localState := validReadinessLocalState()
	localState.ResidentModelID = ""

	response, err := EvaluateReadinessChallenge(validReadinessChallenge(), localState, readinessTestNowUnixMs)
	if err != nil {
		t.Fatalf("EvaluateReadinessChallenge() error = %v, want nil", err)
	}

	assertReadinessResponse(t, response, ReadinessDecisionNotReady, ReadinessReasonNotResident)
}

func TestEvaluateReadinessChallengeSparseHashOptionalWhenChallengeOmitsIt(t *testing.T) {
	challenge := validReadinessChallenge()
	challenge.SparseLogitCheckHash = ""
	localState := validReadinessLocalState()
	localState.SparseLogitCheckHash = ""

	response, err := EvaluateReadinessChallenge(challenge, localState, readinessTestNowUnixMs)
	if err != nil {
		t.Fatalf("EvaluateReadinessChallenge() error = %v, want nil", err)
	}

	assertReadinessResponse(t, response, ReadinessDecisionReady, ReadinessReasonReady)
}

func assertReadinessResponse(t *testing.T, response ReadinessResponse, wantDecision ReadinessDecision, wantReason string) {
	t.Helper()

	if response.Decision != wantDecision {
		t.Fatalf("decision = %q, want %q", response.Decision, wantDecision)
	}
	if response.Reason != wantReason {
		t.Fatalf("reason = %q, want %q", response.Reason, wantReason)
	}
}

func validReadinessChallenge() ReadinessChallenge {
	return ReadinessChallenge{
		ChallengeID:          "challenge-1",
		NodeID:               "node-1",
		ModelID:              "llama-3.1-8b",
		QuantizationID:       "q4_k_m",
		IssuedAtUnixMs:       readinessTestNowUnixMs - 1_000,
		ExpiresAtUnixMs:      readinessTestNowUnixMs + 30_000,
		RequiredLatencyMs:    150,
		PromptHash:           "sha256:prompt",
		SparseLogitCheckHash: "sha256:sparse-logits",
		Nonce:                "nonce-1",
	}
}

func validReadinessLocalState() ReadinessLocalState {
	return ReadinessLocalState{
		NodeID:               "node-1",
		ResidentModelID:      "llama-3.1-8b",
		QuantizationID:       "q4_k_m",
		ModelLeaseState:      modellease.ModelLeaseStateResident,
		LastWarmLatencyMs:    125,
		SparseLogitCheckHash: "sha256:sparse-logits",
	}
}
