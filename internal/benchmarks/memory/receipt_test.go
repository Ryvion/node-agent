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
	if timings.MetadataTotalUs != timings.MetadataBuildUs {
		t.Fatalf("metadata total/build timing = %d/%d us, want aliases", timings.MetadataTotalUs, timings.MetadataBuildUs)
	}
	knownMetadataUs := timings.MetadataStructUs + timings.WeightedValueCopyUs + timings.MetadataDefaultsUs + timings.MetadataValidateUs
	if timings.MetadataTotalUs >= knownMetadataUs {
		if timings.MetadataGapUs != timings.MetadataTotalUs-knownMetadataUs {
			t.Fatalf("metadata gap timing = %d us, want %d; timings=%+v", timings.MetadataGapUs, timings.MetadataTotalUs-knownMetadataUs, timings)
		}
	} else if timings.MetadataGapUs != 0 {
		t.Fatalf("metadata gap timing = %d us, want clamp to 0 when total %d < split sum %d; timings=%+v", timings.MetadataGapUs, timings.MetadataTotalUs, knownMetadataUs, timings)
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

	legacyReceipt, err := buildLegacySemanticBenchmarkReceiptForTest(spec, response)
	if err != nil {
		t.Fatalf("legacy semantic receipt build error = %v", err)
	}
	if receipt.ResultHashHex != legacyReceipt.ResultHashHex {
		t.Fatalf("ResultHashHex changed from legacy semantic receipt\nlegacy: %s\nnew:    %s", legacyReceipt.ResultHashHex, receipt.ResultHashHex)
	}
	if !reflect.DeepEqual(legacyReceipt, receipt) {
		t.Fatalf("receipt semantics changed from legacy semantic build\nlegacy: %+v\nnew:    %+v", legacyReceipt, receipt)
	}
}

func buildLegacySemanticBenchmarkReceiptForTest(spec BenchmarkSpec, response SyntheticAttentionResponse) (BenchmarkReceipt, error) {
	spec = normalizeBenchmarkSpec(spec)
	if err := ValidateBenchmarkSpec(spec); err != nil {
		return BenchmarkReceipt{}, err
	}

	metadata := BenchmarkReceiptMetadata{
		RequestID:                   response.RequestID,
		ShardID:                     response.ShardID,
		LocalMax:                    response.Summary.LocalMax,
		ExpSum:                      response.Summary.ExpSum,
		WeightedValue:               append([]float64(nil), response.Summary.WeightedValue...),
		TokenCount:                  response.Summary.TokenCount,
		ValueDim:                    response.Summary.ValueDim,
		NodeStartedAtUnixMs:         response.NodeStartedAtUnixMs,
		NodeCompletedAtUnixMs:       response.NodeCompletedAtUnixMs,
		ComputeTimeMs:               response.ComputeTimeMs,
		ComputeTimeUs:               response.ComputeTimeUs,
		SimulatedDelayMs:            response.SimulatedDelayMs,
		TotalNodeWallTimeMs:         response.TotalNodeWallTimeMs,
		TotalNodeWallTimeUs:         response.TotalNodeWallTimeUs,
		SummaryPayloadBytesEstimate: response.SummaryPayloadBytesEstimate,
		OutputBytesEstimate:         response.OutputBytesEstimate,
		ProofStatus:                 "synthetic_measured",
		ComputeAllocBytesDelta:      response.ComputeAllocBytesDelta,
		ComputeTotalAllocBytesDelta: response.ComputeTotalAllocBytesDelta,
		ComputeMallocsDelta:         response.ComputeMallocsDelta,
		ComputeNumGCDelta:           response.ComputeNumGCDelta,
		ComputeGCPauseTotalUsDelta:  response.ComputeGCPauseTotalUsDelta,
	}
	if metadata.SummaryPayloadBytesEstimate <= 0 {
		metadata.SummaryPayloadBytesEstimate = estimatePartialAttentionSummaryBytes(response.Summary)
	}
	if metadata.OutputBytesEstimate <= 0 {
		metadata.OutputBytesEstimate = metadata.SummaryPayloadBytesEstimate
	}
	if metadata.RequestID == "" {
		metadata.RequestID = spec.RequestID
	}
	if metadata.ShardID == "" {
		metadata.ShardID = spec.ShardID
	}
	if err := validateBenchmarkReceiptMetadata(spec, metadata); err != nil {
		return BenchmarkReceipt{}, err
	}

	hashHex, err := HashBenchmarkReceiptMetadata(spec.JobID, metadata)
	if err != nil {
		return BenchmarkReceipt{}, err
	}
	if err := populateBenchmarkReceiptJSONByteEstimates(&metadata, spec.JobID, hashHex, 1); err != nil {
		return BenchmarkReceipt{}, err
	}
	return BenchmarkReceipt{
		JobID:         spec.JobID,
		ResultHashHex: hashHex,
		MeteringUnits: 1,
		Metadata: map[string]any{
			BenchmarkTask: metadata.Map(),
		},
	}, nil
}
