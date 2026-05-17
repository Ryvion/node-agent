package contracttestbridge

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/Ryvion/ryvion-node/internal/hub"
	nodespec "github.com/Ryvion/ryvion-node/internal/inference/speculative"
)

func TestVerifyWaveBuildsNonBillableContractTestResult(t *testing.T) {
	result := VerifyWave("job-1", nodespec.HotSessionSpec{
		RunID:            "run-1",
		WorkGraphID:      "wg-1",
		Prompt:           "Write one short sentence.",
		ParentPrefixHash: "sha256:prefix",
		MaxTokens:        16,
	}, hub.SpeculativeLiveLabSessionCommand{
		CommandID: "cmd-1",
		WindowID:  "win-1",
		WaveIndex: 1,
		Tree: map[string]any{
			"tree_cid": "sha256:tree",
			"branches": []any{
				map[string]any{"candidate_tokens": []any{float64(1), float64(2), float64(3)}},
			},
		},
	}, 0)

	if result.JobID != "job-1" || result.WindowID != "win-1" || result.TreeCID != "sha256:tree" {
		t.Fatalf("result identity = %+v, want job/window/tree fields", result)
	}
	if result.AcceptedLen != 3 || result.AcceptedText == "" || !result.AcceptedTextPublic {
		t.Fatalf("accepted result = %+v, want deterministic accepted test text", result)
	}
	if result.StopReason != "" || result.EOS {
		t.Fatalf("terminal state = %q/%v, want non-terminal", result.StopReason, result.EOS)
	}
	if result.ProbeSummary["source"] != Source ||
		result.ProbeSummary["backend"] != Backend ||
		result.ProbeSummary["production_valid"] != false ||
		result.ProbeSummary["test_adapter"] != true ||
		result.ProbeSummary["billing_status"] != "not_billable_contract_test" {
		t.Fatalf("probe summary = %#v, want contract-test labels", result.ProbeSummary)
	}
	if encoded, _ := json.Marshal(result.ProbeSummary); strings.Contains(string(encoded), "Write one short sentence") || strings.Contains(string(encoded), result.AcceptedText) {
		t.Fatalf("probe summary leaked prompt or accepted text: %s", encoded)
	}
}

func TestVerifyWaveMarksMaxTokens(t *testing.T) {
	result := VerifyWave("job-1", nodespec.HotSessionSpec{
		MaxTokens: 4,
	}, hub.SpeculativeLiveLabSessionCommand{
		CommandID: "cmd-1",
		WindowID:  "win-1",
		WaveIndex: 1,
		Tree: map[string]any{
			"branches": []any{
				map[string]any{"candidate_tokens": []any{float64(1), float64(2), float64(3)}},
			},
		},
	}, 1)
	if result.StopReason != "max_tokens" {
		t.Fatalf("StopReason = %q, want max_tokens", result.StopReason)
	}
}
