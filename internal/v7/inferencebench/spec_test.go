package inferencebench

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/Ryvion/node-agent/internal/v7/llamacpp"
)

func TestDecodeBenchmarkSpecAcceptsLlamaCppAliasAndPromptHash(t *testing.T) {
	spec := validInferenceBenchmarkSpec()
	spec.Backend = "llamacpp"
	encoded := mustMarshalInferenceSpec(t, spec)

	got, err := DecodeBenchmarkSpec(encoded)
	if err != nil {
		t.Fatalf("DecodeBenchmarkSpec() error = %v", err)
	}
	if got.Backend != llamacpp.BackendName {
		t.Fatalf("backend = %q, want %q", got.Backend, llamacpp.BackendName)
	}
	if got.PromptHash != llamacpp.HashBenchmarkPrompt() {
		t.Fatalf("prompt_hash = %q, want internal prompt hash", got.PromptHash)
	}
	if got.BenchmarkID != "benchmark-backend-inference-local" || got.TargetNodeID != "node-backend-inference-local" {
		t.Fatalf("benchmark_id/target_node_id = %q/%q, want parsed ids", got.BenchmarkID, got.TargetNodeID)
	}
	if got.PromptProfileID != BenchmarkPromptProfileID {
		t.Fatalf("prompt_profile_id = %q, want %q", got.PromptProfileID, BenchmarkPromptProfileID)
	}
}

func TestDecodeBenchmarkSpecRejectsPromptHashMismatch(t *testing.T) {
	spec := validInferenceBenchmarkSpec()
	spec.PromptHash = "sha256:" + strings.Repeat("a", 64)
	if _, err := DecodeBenchmarkSpec(mustMarshalInferenceSpec(t, spec)); err == nil {
		t.Fatal("DecodeBenchmarkSpec() error = nil, want prompt hash mismatch")
	}
}

func TestBenchmarkAssignmentIdentityFromJSON(t *testing.T) {
	identity, ok := BenchmarkAssignmentIdentityFromJSON(validInferenceBenchmarkSpecJSON(t))
	if !ok {
		t.Fatal("BenchmarkAssignmentIdentityFromJSON() ok = false")
	}
	if identity.JobID != "job-backend-inference-local" || identity.RequestID != "request-backend-inference-local" {
		t.Fatalf("identity = %+v", identity)
	}
}

func mustMarshalInferenceSpec(t *testing.T, spec BenchmarkSpec) string {
	t.Helper()
	encoded, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	return string(encoded)
}
