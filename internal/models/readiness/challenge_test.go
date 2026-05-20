package readiness

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

func TestReadinessChallengeRequiresChallengeID(t *testing.T) {
	challenge := validReadinessChallenge()
	challenge.ChallengeID = ""

	_, err := EvaluateReadinessChallenge(challenge, validReadinessLocalState(), readinessTestNowUnixMs)
	if !errors.Is(err, ErrInvalidReadinessChallenge) || !strings.Contains(err.Error(), "challenge_id required") {
		t.Fatalf("EvaluateReadinessChallenge() error = %v, want challenge_id required", err)
	}
}

func TestReadinessChallengeRequiresNonce(t *testing.T) {
	challenge := validReadinessChallenge()
	challenge.Nonce = ""

	_, err := EvaluateReadinessChallenge(challenge, validReadinessLocalState(), readinessTestNowUnixMs)
	if !errors.Is(err, ErrInvalidReadinessChallenge) || !strings.Contains(err.Error(), "nonce required") {
		t.Fatalf("EvaluateReadinessChallenge() error = %v, want nonce required", err)
	}
}

func TestReadinessChallengeExcludesRawPromptText(t *testing.T) {
	challengeType := reflect.TypeOf(ReadinessChallenge{})
	for i := 0; i < challengeType.NumField(); i++ {
		fieldName := strings.ToLower(challengeType.Field(i).Name)
		if strings.Contains(fieldName, "prompt") && fieldName != "prompthash" {
			t.Fatalf("ReadinessChallenge field %q exposes raw prompt data", challengeType.Field(i).Name)
		}
	}
}
