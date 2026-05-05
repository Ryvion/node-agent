package memorybench

import (
	"reflect"
	"testing"
)

func TestBuildBenchmarkReceiptWithTimings(t *testing.T) {
	spec := validBenchmarkSpec(t)
	spec.SimulatedDelayMs = 0
	spec.TokenCount = 64
	spec.ValueDim = 128

	request := GenerateSyntheticAttentionRequest(spec.Seed, spec.ShardID, spec.TokenCount, spec.ValueDim)
	request.RequestID = spec.RequestID
	request.JobID = spec.JobID
	request.ShardID = spec.ShardID
	request.CreatedAtUnixMs = spec.CreatedAtUnixMs
	response, err := ComputePartialAttentionSummary(request)
	if err != nil {
		t.Fatalf("ComputePartialAttentionSummary() error = %v", err)
	}

	receipt, timings, err := BuildBenchmarkReceiptWithTimings(spec, response)
	if err != nil {
		t.Fatalf("BuildBenchmarkReceiptWithTimings() error = %v", err)
	}
	if receipt.JobID != spec.JobID || receipt.MeteringUnits != 1 || len(receipt.ResultHashHex) != 64 {
		t.Fatalf("unexpected receipt: %+v", receipt)
	}
	if _, ok := receipt.Metadata[BenchmarkTask].(map[string]any); !ok {
		t.Fatalf("metadata[%q] type = %T, want map[string]any", BenchmarkTask, receipt.Metadata[BenchmarkTask])
	}
	if timings.HashUs <= 0 {
		t.Fatalf("hash timing = %d us, want positive", timings.HashUs)
	}
	if timings.JSONMeasureUs <= 0 {
		t.Fatalf("JSON measure timing = %d us, want positive", timings.JSONMeasureUs)
	}
	minTotalUs := timings.MetadataBuildUs + timings.HashUs + timings.JSONMeasureUs + timings.EnvelopeBuildUs
	if timings.TotalBuildUs < minTotalUs {
		t.Fatalf("total build timing = %d us, want >= split sum %d; timings=%+v", timings.TotalBuildUs, minTotalUs, timings)
	}

	compatReceipt, err := BuildBenchmarkReceipt(spec, response)
	if err != nil {
		t.Fatalf("BuildBenchmarkReceipt() compatibility wrapper error = %v", err)
	}
	if !reflect.DeepEqual(compatReceipt, receipt) {
		t.Fatalf("BuildBenchmarkReceipt() changed receipt semantics\ncompat: %+v\nwith timings: %+v", compatReceipt, receipt)
	}
}
