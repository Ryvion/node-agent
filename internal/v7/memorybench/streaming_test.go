package memorybench

import (
	"context"
	"math"
	"reflect"
	"testing"
)

func TestStreamingPartialAttentionMatchesMaterialized(t *testing.T) {
	for _, tc := range []struct {
		name       string
		seed       int64
		shardID    string
		tokenCount int
		valueDim   int
	}{
		{name: "small", seed: 1, shardID: "shard-a", tokenCount: 4, valueDim: 2},
		{name: "default selftest shape", seed: 42, shardID: "local-selftest", tokenCount: 256, valueDim: 64},
		{name: "odd dimensions", seed: 99, shardID: "shard-z", tokenCount: 33, valueDim: 7},
		{name: "negative seed", seed: -17, shardID: "shard-neg", tokenCount: 19, valueDim: 11},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request := GenerateSyntheticAttentionRequest(tc.seed, tc.shardID, tc.tokenCount, tc.valueDim)
			materialized, err := ComputePartialAttentionSummary(request)
			if err != nil {
				t.Fatalf("ComputePartialAttentionSummary() error = %v", err)
			}
			streaming, err := ComputePartialAttentionSummaryFromSeed(tc.seed, tc.shardID, tc.tokenCount, tc.valueDim)
			if err != nil {
				t.Fatalf("ComputePartialAttentionSummaryFromSeed() error = %v", err)
			}

			if materialized.RequestID != streaming.RequestID || materialized.JobID != streaming.JobID || materialized.ShardID != streaming.ShardID {
				t.Fatalf("identity mismatch\nmaterialized: %+v\nstreaming: %+v", materialized, streaming)
			}
			if materialized.OutputBytesEstimate != streaming.OutputBytesEstimate {
				t.Fatalf("output bytes estimate = %d, want %d", streaming.OutputBytesEstimate, materialized.OutputBytesEstimate)
			}
			assertPartialAttentionSummaryEqual(t, streaming.Summary, materialized.Summary)
		})
	}
}

func TestComputeSyntheticPartialAttentionStreamingUsesSpecIdentity(t *testing.T) {
	spec := validBenchmarkSpec(t)
	spec.TokenCount = 64
	spec.ValueDim = 8

	streaming, err := ComputeSyntheticPartialAttentionStreaming(spec)
	if err != nil {
		t.Fatalf("ComputeSyntheticPartialAttentionStreaming() error = %v", err)
	}
	request := GenerateSyntheticAttentionRequest(spec.Seed, spec.ShardID, spec.TokenCount, spec.ValueDim)
	request.RequestID = spec.RequestID
	request.JobID = spec.JobID
	request.ShardID = spec.ShardID
	request.CreatedAtUnixMs = spec.CreatedAtUnixMs
	materialized, err := ComputePartialAttentionSummary(request)
	if err != nil {
		t.Fatalf("ComputePartialAttentionSummary() error = %v", err)
	}

	if streaming.RequestID != spec.RequestID || streaming.JobID != spec.JobID || streaming.ShardID != spec.ShardID {
		t.Fatalf("streaming response did not use spec identity: %+v", streaming)
	}
	if streaming.CreatedAtUnixMs != spec.CreatedAtUnixMs {
		t.Fatalf("created_at_unix_ms = %d, want %d", streaming.CreatedAtUnixMs, spec.CreatedAtUnixMs)
	}
	assertPartialAttentionSummaryEqual(t, streaming.Summary, materialized.Summary)
}

func TestStreamingPathAllocatesLessThanMaterializedPath(t *testing.T) {
	const (
		seed       int64 = 123
		shardID          = "alloc"
		tokenCount       = 128
		valueDim         = 32
	)

	streamingAllocs := testing.AllocsPerRun(5, func() {
		response, err := ComputePartialAttentionSummaryFromSeed(seed, shardID, tokenCount, valueDim)
		if err != nil {
			panic(err)
		}
		if response.Summary.ValueDim != valueDim {
			panic("unexpected value dim")
		}
	})
	materializedAllocs := testing.AllocsPerRun(5, func() {
		request := GenerateSyntheticAttentionRequest(seed, shardID, tokenCount, valueDim)
		response, err := ComputePartialAttentionSummary(request)
		if err != nil {
			panic(err)
		}
		if response.Summary.ValueDim != valueDim {
			panic("unexpected value dim")
		}
	})

	if streamingAllocs >= materializedAllocs/4 {
		t.Fatalf("streaming allocations = %.0f, materialized allocations = %.0f; want streaming substantially lower", streamingAllocs, materializedAllocs)
	}
}

func TestExecuteBenchmarkSpecStreamingReceiptDiagnostics(t *testing.T) {
	spec := validBenchmarkSpec(t)
	spec.TokenCount = 128
	spec.ValueDim = 16
	spec.SimulatedDelayMs = 0

	receipt, err := ExecuteBenchmarkSpec(context.Background(), spec, ExecuteOptions{})
	if err != nil {
		t.Fatalf("ExecuteBenchmarkSpec() error = %v", err)
	}
	metadata := receipt.Metadata[BenchmarkTask].(map[string]any)

	for _, key := range []string{
		"compute_alloc_bytes_delta",
		"compute_total_alloc_bytes_delta",
		"compute_mallocs_delta",
		"compute_num_gc_delta",
		"compute_gc_pause_total_us_delta",
	} {
		if _, ok := metadata[key]; !ok {
			t.Fatalf("metadata missing streaming diagnostic %q: %+v", key, metadata)
		}
	}
	if got := benchmarkMetadataInt64(t, metadata, "compute_total_alloc_bytes_delta"); got <= 0 {
		t.Fatalf("compute_total_alloc_bytes_delta = %d, want positive", got)
	}
	if got := benchmarkMetadataInt64(t, metadata, "compute_mallocs_delta"); got <= 0 {
		t.Fatalf("compute_mallocs_delta = %d, want positive", got)
	}

	streaming, err := ComputeSyntheticPartialAttentionStreaming(spec)
	if err != nil {
		t.Fatalf("ComputeSyntheticPartialAttentionStreaming() error = %v", err)
	}
	if !nearlyEqual(metadata["local_max"].(float64), streaming.Summary.LocalMax) {
		t.Fatalf("local_max = %v, want %v", metadata["local_max"], streaming.Summary.LocalMax)
	}
	if !nearlyEqual(metadata["exp_sum"].(float64), streaming.Summary.ExpSum) {
		t.Fatalf("exp_sum = %v, want %v", metadata["exp_sum"], streaming.Summary.ExpSum)
	}
}

func TestStreamingDiagnosticsDoNotAffectReceiptHash(t *testing.T) {
	spec := validBenchmarkSpec(t)
	response, err := ComputeSyntheticPartialAttentionStreaming(spec)
	if err != nil {
		t.Fatalf("ComputeSyntheticPartialAttentionStreaming() error = %v", err)
	}

	first, err := BuildBenchmarkReceipt(spec, response)
	if err != nil {
		t.Fatalf("first BuildBenchmarkReceipt() error = %v", err)
	}
	response.ComputeAllocBytesDelta += 4096
	response.ComputeTotalAllocBytesDelta += 8192
	response.ComputeMallocsDelta += 7
	response.ComputeNumGCDelta += 1
	response.ComputeGCPauseTotalUsDelta += 11

	second, err := BuildBenchmarkReceipt(spec, response)
	if err != nil {
		t.Fatalf("second BuildBenchmarkReceipt() error = %v", err)
	}
	if first.ResultHashHex != second.ResultHashHex {
		t.Fatalf("receipt hash changed after diagnostic-only fields changed\nfirst:  %s\nsecond: %s", first.ResultHashHex, second.ResultHashHex)
	}
}

func assertPartialAttentionSummaryEqual(t *testing.T, got, want PartialAttentionSummary) {
	t.Helper()
	if got.LocalMax != want.LocalMax {
		t.Fatalf("local_max = %.17g, want %.17g", got.LocalMax, want.LocalMax)
	}
	if !nearlyEqualFloat64(got.ExpSum, want.ExpSum) {
		t.Fatalf("exp_sum = %.17g, want %.17g", got.ExpSum, want.ExpSum)
	}
	if got.TokenCount != want.TokenCount || got.ValueDim != want.ValueDim || got.DType != want.DType {
		t.Fatalf("summary metadata = %+v, want %+v", got, want)
	}
	if len(got.WeightedValue) != len(want.WeightedValue) {
		t.Fatalf("weighted value length = %d, want %d", len(got.WeightedValue), len(want.WeightedValue))
	}
	for dim := range got.WeightedValue {
		if !nearlyEqualFloat64(got.WeightedValue[dim], want.WeightedValue[dim]) {
			t.Fatalf("weighted_value[%d] = %.17g, want %.17g", dim, got.WeightedValue[dim], want.WeightedValue[dim])
		}
	}
	if !reflect.DeepEqual(got.WeightedValue, want.WeightedValue) {
		for dim := range got.WeightedValue {
			if math.Float64bits(got.WeightedValue[dim]) != math.Float64bits(want.WeightedValue[dim]) {
				t.Fatalf("weighted_value[%d] bits differ: got %.17g want %.17g", dim, got.WeightedValue[dim], want.WeightedValue[dim])
			}
		}
	}
}

func nearlyEqualFloat64(got, want float64) bool {
	const epsilon = 1e-12
	diff := got - want
	if diff < 0 {
		diff = -diff
	}
	return diff <= epsilon
}
