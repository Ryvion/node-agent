package memorybench

import (
	"reflect"
	"testing"
)

func TestGenerateSyntheticAttentionRequestDeterministic(t *testing.T) {
	first := GenerateSyntheticAttentionRequest(123, "shard-a", 8, 4)
	second := GenerateSyntheticAttentionRequest(123, "shard-a", 8, 4)
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("GenerateSyntheticAttentionRequest() not deterministic\nfirst:  %+v\nsecond: %+v", first, second)
	}

	other := GenerateSyntheticAttentionRequest(124, "shard-a", 8, 4)
	if reflect.DeepEqual(first.Logits, other.Logits) {
		t.Fatalf("different seed produced identical logits")
	}
	if len(first.Logits) != 8 || len(first.Values) != 8 {
		t.Fatalf("token count = logits %d values %d, want 8", len(first.Logits), len(first.Values))
	}
	for i, value := range first.Values {
		if len(value) != 4 {
			t.Fatalf("value %d dim = %d, want 4", i, len(value))
		}
	}
}

func TestSyntheticAttentionResponseMetadata(t *testing.T) {
	request := GenerateSyntheticAttentionRequest(456, "shard-b", 16, 6)

	response, err := ComputePartialAttentionSummary(request)
	if err != nil {
		t.Fatalf("ComputePartialAttentionSummary() error = %v", err)
	}
	if response.RequestID != request.RequestID {
		t.Fatalf("request id = %q, want %q", response.RequestID, request.RequestID)
	}
	if response.JobID != request.JobID {
		t.Fatalf("job id = %q, want %q", response.JobID, request.JobID)
	}
	if response.ShardID != request.ShardID {
		t.Fatalf("shard id = %q, want %q", response.ShardID, request.ShardID)
	}
	if response.ComputeTimeMs < 0 {
		t.Fatalf("compute time = %d, want non-negative", response.ComputeTimeMs)
	}
	if response.OutputBytesEstimate <= 0 {
		t.Fatalf("output bytes estimate = %d, want positive", response.OutputBytesEstimate)
	}
	if response.CreatedAtUnixMs != request.CreatedAtUnixMs {
		t.Fatalf("created_at_unix_ms = %d, want %d", response.CreatedAtUnixMs, request.CreatedAtUnixMs)
	}
	if response.Summary.TokenCount != len(request.Logits) {
		t.Fatalf("summary token count = %d, want %d", response.Summary.TokenCount, len(request.Logits))
	}
	if response.Summary.ValueDim != request.ValueDim {
		t.Fatalf("summary value dim = %d, want %d", response.Summary.ValueDim, request.ValueDim)
	}
}
