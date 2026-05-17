package llamacpp

import (
	"context"
	"encoding/json"
	"testing"
	"time"
)

func TestExecuteBackendBenchmarkAssignmentRunsMockedBenchmark(t *testing.T) {
	specJSON := testBackendBenchmarkSpecJSON(t)
	runner := &fakeBackendBenchmarkRunner{
		snapshot: testBackendBenchmarkSnapshot(BenchmarkProofStatusMeasured),
	}

	receipt, handled, err := ExecuteBackendBenchmarkAssignment(context.Background(), specJSON, ExecuteBackendBenchmarkOptions{
		Runner: runner,
		Profile: BackendBenchmarkProfile{
			NodeID:              "node-a",
			Acceleration:        "cuda",
			Warm:                true,
			ContextLengthTokens: 4096,
			StreamingSupported:  true,
		},
	})
	if err != nil {
		t.Fatalf("ExecuteBackendBenchmarkAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if runner.calls != 1 {
		t.Fatalf("runner calls = %d, want 1", runner.calls)
	}
	if got := runner.configs[0]; got.ModelID != "tinyllama.Q4_K_M.gguf" || got.MaxTokens != 16 || got.WarmupRuns != 1 || got.MeasuredRuns != 3 || got.TimeoutMs != 30_000 ||
		got.NodeID != "node-a" || got.Acceleration != "cuda" || !got.Warm || got.ContextLengthTokens != 4096 || !got.StreamingSupported {
		t.Fatalf("runner config = %+v, want spec-derived benchmark config", got)
	}
	metadata := receipt.Metadata[BackendBenchmarkTask].(map[string]any)
	if metadata["proof_status"] != BenchmarkProofStatusMeasured {
		t.Fatalf("proof_status = %v", metadata["proof_status"])
	}
	if metadata["prompt_hash"] == "" || metadata["output_hash"] == "" {
		t.Fatalf("hashes missing: %+v", metadata)
	}
	assertBackendBenchmarkReceiptJSONSafe(t, receipt, "secret llama output")
}

func TestExecuteBackendBenchmarkAssignmentFlagOffDoesNotHandleOrRun(t *testing.T) {
	runner := &fakeBackendBenchmarkRunner{
		snapshot: testBackendBenchmarkSnapshot(BenchmarkProofStatusMeasured),
	}
	receipt, handled, err := ExecuteBackendBenchmarkAssignment(context.Background(), testBackendBenchmarkSpecJSON(t), ExecuteBackendBenchmarkOptions{
		Getenv: func(key string) string {
			if key == BackendBenchmarkFlagEnv {
				return "0"
			}
			return ""
		},
		Runner: runner,
	})
	if err != nil {
		t.Fatalf("ExecuteBackendBenchmarkAssignment() error = %v", err)
	}
	if handled {
		t.Fatal("handled = true, want false when flag is off")
	}
	if receipt.ResultHashHex != "" {
		t.Fatalf("receipt = %+v, want zero receipt when flag is off", receipt)
	}
	if runner.calls != 0 {
		t.Fatalf("runner calls = %d, want 0", runner.calls)
	}
}

func TestExecuteBackendBenchmarkAssignmentIgnoresOtherTasks(t *testing.T) {
	receipt, handled, err := ExecuteBackendBenchmarkAssignment(context.Background(), `{"task":"native_report","job_id":"job-a"}`, ExecuteBackendBenchmarkOptions{
		Getenv: func(string) string { return "1" },
		Runner: &fakeBackendBenchmarkRunner{},
	})
	if err != nil {
		t.Fatalf("ExecuteBackendBenchmarkAssignment() error = %v", err)
	}
	if handled || receipt.ResultHashHex != "" {
		t.Fatalf("handled=%t receipt=%+v, want ignored", handled, receipt)
	}
}

func TestExecuteBackendBenchmarkAssignmentSidecarUnavailableBuildsSafeReceipt(t *testing.T) {
	specJSON := testBackendBenchmarkSpecJSON(t)
	runner := &fakeBackendBenchmarkRunner{
		snapshot: testBackendBenchmarkSnapshot(BenchmarkProofStatusUnavailable),
	}
	receipt, handled, err := ExecuteBackendBenchmarkAssignment(context.Background(), specJSON, ExecuteBackendBenchmarkOptions{
		Runner: runner,
	})
	if err != nil {
		t.Fatalf("ExecuteBackendBenchmarkAssignment() error = %v", err)
	}
	if !handled {
		t.Fatal("handled = false, want true")
	}
	if receipt.MeteringUnits != 0 {
		t.Fatalf("metering_units = %d, want 0 for unavailable backend", receipt.MeteringUnits)
	}
	metadata := receipt.Metadata[BackendBenchmarkTask].(map[string]any)
	if metadata["available"] != false || metadata["sidecar_healthy"] != false {
		t.Fatalf("availability metadata = %+v, want unavailable unhealthy", metadata)
	}
	if metadata["proof_status"] != BenchmarkProofStatusUnavailable {
		t.Fatalf("proof_status = %v, want unavailable", metadata["proof_status"])
	}
	assertBackendBenchmarkReceiptJSONSafe(t, receipt)
}

func TestBackendBenchmarkLocalStatusCounters(t *testing.T) {
	status := NewBackendBenchmarkLocalStatus()
	status.RecordSeen("job-1")
	status.RecordExecuted("job-1")
	status.RecordReceiptSubmitted("job-1")
	status.RecordReceiptFailed("job-2", errTestBackendBenchmarkFailure{})
	snapshot := status.Snapshot()
	if snapshot.LastJobID != "job-2" || snapshot.LastError == "" {
		t.Fatalf("snapshot = %+v, want failed job and error", snapshot)
	}
	if snapshot.Counters.Seen != 1 || snapshot.Counters.Executed != 1 || snapshot.Counters.ReceiptSubmitted != 1 || snapshot.Counters.ReceiptFailed != 1 {
		t.Fatalf("counters = %+v", snapshot.Counters)
	}
}

type fakeBackendBenchmarkRunner struct {
	snapshot BenchmarkStatusSnapshot
	configs  []BenchmarkConfig
	calls    int
}

func (f *fakeBackendBenchmarkRunner) Run(_ context.Context, config BenchmarkConfig) BenchmarkStatusSnapshot {
	f.calls++
	f.configs = append(f.configs, config)
	return f.snapshot
}

func testBackendBenchmarkSpecJSON(t *testing.T) string {
	t.Helper()
	spec := BackendBenchmarkSpec{
		Task:            BackendBenchmarkTask,
		RequestID:       "request-llamacpp-backend-local",
		JobID:           "job-llamacpp-backend-local",
		Backend:         BackendName,
		ModelID:         "tinyllama.Q4_K_M.gguf",
		MaxTokens:       16,
		WarmupRuns:      1,
		MeasuredRuns:    3,
		TimeoutMs:       30_000,
		CreatedAtUnixMs: 1_800_000_000_123,
	}
	encoded, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	return string(encoded)
}

func testBackendBenchmarkSnapshot(proofStatus string) BenchmarkStatusSnapshot {
	available := proofStatus == BenchmarkProofStatusMeasured
	outputHash := "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	outputBytes := int64(len("secret llama output"))
	status := BenchmarkStatusCompleted
	if !available {
		outputHash = ""
		outputBytes = 0
		status = BenchmarkStatusFailed
	}
	return BenchmarkStatusSnapshot{
		LastRunAt: time.Unix(1_800_000_001, 0),
		Status:    status,
		Metrics: BenchmarkMetrics{
			Available:       available,
			SidecarHealthy:  available,
			ModelLoaded:     available,
			ModelID:         "tinyllama.Q4_K_M.gguf",
			PromptHash:      HashBenchmarkPrompt(),
			OutputHash:      outputHash,
			OutputBytes:     outputBytes,
			WarmupRuns:      1,
			MeasuredRuns:    3,
			P50TTFTMs:       100,
			P95TTFTMs:       200,
			P50TotalTimeMs:  800,
			P95TotalTimeMs:  1100,
			P50DecodeTPS:    20.5,
			P95DecodeTPS:    21.5,
			P50EndToEndTPS:  18.25,
			P95EndToEndTPS:  19.75,
			Backend:         BackendName,
			RuntimeKind:     BackendName,
			ProofStatus:     proofStatus,
			Streaming:       true,
			TokensGenerated: 48,
		},
	}
}

type errTestBackendBenchmarkFailure struct{}

func (errTestBackendBenchmarkFailure) Error() string {
	return "receipt failed"
}
