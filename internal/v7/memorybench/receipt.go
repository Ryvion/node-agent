package memorybench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

type BenchmarkReceipt struct {
	JobID         string
	ResultHashHex string
	MeteringUnits uint64
	Metadata      map[string]any
}

type BenchmarkReceiptMetadata struct {
	RequestID                   string    `json:"request_id"`
	ShardID                     string    `json:"shard_id"`
	LocalMax                    float64   `json:"local_max"`
	ExpSum                      float64   `json:"exp_sum"`
	WeightedValue               []float64 `json:"weighted_value"`
	TokenCount                  int       `json:"token_count"`
	ValueDim                    int       `json:"value_dim"`
	NodeStartedAtUnixMs         int64     `json:"node_started_at_unix_ms"`
	NodeCompletedAtUnixMs       int64     `json:"node_completed_at_unix_ms"`
	ComputeTimeMs               int64     `json:"compute_time_ms"`
	ComputeTimeUs               int64     `json:"compute_time_us"`
	SimulatedDelayMs            int64     `json:"simulated_delay_ms"`
	TotalNodeWallTimeMs         int64     `json:"total_node_wall_time_ms"`
	TotalNodeWallTimeUs         int64     `json:"total_node_wall_time_us"`
	SummaryPayloadBytesEstimate int64     `json:"summary_payload_bytes_estimate"`
	OutputBytesEstimate         int64     `json:"output_bytes_estimate"`
	ReceiptMetadataJSONBytes    int64     `json:"receipt_metadata_json_bytes"`
	ReceiptEnvelopeJSONBytes    int64     `json:"receipt_envelope_json_bytes"`
	ProofStatus                 string    `json:"proof_status"`
}

type ReceiptBuildTimings struct {
	MetadataBuildMs int64
	HashMs          int64
	JSONMeasureMs   int64
	EnvelopeBuildMs int64
	TotalBuildMs    int64
	MetadataBuildUs int64
	HashUs          int64
	JSONMeasureUs   int64
	EnvelopeBuildUs int64
	TotalBuildUs    int64
}

func BuildBenchmarkReceipt(spec BenchmarkSpec, response SyntheticAttentionResponse) (BenchmarkReceipt, error) {
	receipt, _, err := BuildBenchmarkReceiptWithTimings(spec, response)
	return receipt, err
}

func BuildBenchmarkReceiptWithTimings(spec BenchmarkSpec, response SyntheticAttentionResponse) (receipt BenchmarkReceipt, timings ReceiptBuildTimings, err error) {
	totalStarted := time.Now()
	defer func() {
		timings.TotalBuildMs, timings.TotalBuildUs = receiptBuildDurationFields(time.Since(totalStarted))
	}()

	spec = normalizeBenchmarkSpec(spec)
	if err := ValidateBenchmarkSpec(spec); err != nil {
		return BenchmarkReceipt{}, timings, err
	}

	metadataStarted := time.Now()
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
		timings.MetadataBuildMs, timings.MetadataBuildUs = receiptBuildDurationFields(time.Since(metadataStarted))
		return BenchmarkReceipt{}, timings, err
	}
	timings.MetadataBuildMs, timings.MetadataBuildUs = receiptBuildDurationFields(time.Since(metadataStarted))

	hashStarted := time.Now()
	hashHex, err := HashBenchmarkReceiptMetadata(spec.JobID, metadata)
	if err != nil {
		timings.HashMs, timings.HashUs = receiptBuildDurationFields(time.Since(hashStarted))
		return BenchmarkReceipt{}, timings, err
	}
	timings.HashMs, timings.HashUs = receiptBuildDurationFields(time.Since(hashStarted))

	jsonMeasureStarted := time.Now()
	if err := populateBenchmarkReceiptJSONByteEstimates(&metadata, spec.JobID, hashHex, 1); err != nil {
		timings.JSONMeasureMs, timings.JSONMeasureUs = receiptBuildDurationFields(time.Since(jsonMeasureStarted))
		return BenchmarkReceipt{}, timings, err
	}
	timings.JSONMeasureMs, timings.JSONMeasureUs = receiptBuildDurationFields(time.Since(jsonMeasureStarted))

	envelopeStarted := time.Now()
	receipt = BenchmarkReceipt{
		JobID:         spec.JobID,
		ResultHashHex: hashHex,
		MeteringUnits: 1,
		Metadata: map[string]any{
			BenchmarkTask: metadata.Map(),
		},
	}
	timings.EnvelopeBuildMs, timings.EnvelopeBuildUs = receiptBuildDurationFields(time.Since(envelopeStarted))
	return receipt, timings, nil
}

func BuildBenchmarkRejectionReceipt(jobID string, runErr error) BenchmarkReceipt {
	jobID = strings.TrimSpace(jobID)
	if jobID == "" {
		jobID = "v7-memory-benchmark-rejected"
	}
	reason := "benchmark rejected"
	if runErr != nil {
		reason = strings.TrimSpace(runErr.Error())
	}
	if len(reason) > 256 {
		reason = reason[:256]
	}
	payload := struct {
		Task        string `json:"task"`
		JobID       string `json:"job_id"`
		ProofStatus string `json:"proof_status"`
		Error       string `json:"error"`
	}{
		Task:        BenchmarkTask,
		JobID:       jobID,
		ProofStatus: "rejected",
		Error:       reason,
	}
	encoded, _ := json.Marshal(payload)
	sum := sha256.Sum256(encoded)
	return BenchmarkReceipt{
		JobID:         jobID,
		ResultHashHex: hex.EncodeToString(sum[:]),
		MeteringUnits: 0,
		Metadata: map[string]any{
			BenchmarkTask: map[string]any{
				"proof_status": "rejected",
				"error":        reason,
			},
		},
	}
}

func HashBenchmarkReceiptMetadata(jobID string, metadata BenchmarkReceiptMetadata) (string, error) {
	jobID = strings.TrimSpace(jobID)
	if jobID == "" {
		return "", fmt.Errorf("%w: job_id required for result hash", ErrInvalidBenchmarkSpec)
	}
	metadata = metadata.clone()
	if metadata.SummaryPayloadBytesEstimate <= 0 {
		metadata.SummaryPayloadBytesEstimate = metadata.OutputBytesEstimate
	}
	envelope := struct {
		Task     string                       `json:"task"`
		JobID    string                       `json:"job_id"`
		Metadata benchmarkReceiptHashMetadata `json:"v7_memory_benchmark"`
	}{
		Task:     BenchmarkTask,
		JobID:    jobID,
		Metadata: benchmarkReceiptHashMetadataFromMetadata(metadata),
	}
	encoded, err := json.Marshal(envelope)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:]), nil
}

func (m BenchmarkReceiptMetadata) Map() map[string]any {
	return map[string]any{
		"request_id":                     m.RequestID,
		"shard_id":                       m.ShardID,
		"local_max":                      m.LocalMax,
		"exp_sum":                        m.ExpSum,
		"weighted_value":                 append([]float64(nil), m.WeightedValue...),
		"token_count":                    m.TokenCount,
		"value_dim":                      m.ValueDim,
		"node_started_at_unix_ms":        m.NodeStartedAtUnixMs,
		"node_completed_at_unix_ms":      m.NodeCompletedAtUnixMs,
		"compute_time_ms":                m.ComputeTimeMs,
		"compute_time_us":                m.ComputeTimeUs,
		"simulated_delay_ms":             m.SimulatedDelayMs,
		"total_node_wall_time_ms":        m.TotalNodeWallTimeMs,
		"total_node_wall_time_us":        m.TotalNodeWallTimeUs,
		"summary_payload_bytes_estimate": m.SummaryPayloadBytesEstimate,
		"output_bytes_estimate":          m.OutputBytesEstimate,
		"receipt_metadata_json_bytes":    m.ReceiptMetadataJSONBytes,
		"receipt_envelope_json_bytes":    m.ReceiptEnvelopeJSONBytes,
		"proof_status":                   m.ProofStatus,
	}
}

func (m BenchmarkReceiptMetadata) clone() BenchmarkReceiptMetadata {
	m.WeightedValue = append([]float64(nil), m.WeightedValue...)
	return m
}

func validateBenchmarkReceiptMetadata(spec BenchmarkSpec, metadata BenchmarkReceiptMetadata) error {
	var errs []error
	if strings.TrimSpace(metadata.RequestID) == "" {
		errs = append(errs, fmt.Errorf("%w: receipt request_id required", ErrInvalidBenchmarkSpec))
	}
	if strings.TrimSpace(metadata.ShardID) == "" {
		errs = append(errs, fmt.Errorf("%w: receipt shard_id required", ErrInvalidBenchmarkSpec))
	}
	if metadata.TokenCount != spec.TokenCount {
		errs = append(errs, fmt.Errorf("%w: receipt token_count mismatch", ErrInvalidBenchmarkSpec))
	}
	if metadata.ValueDim != spec.ValueDim {
		errs = append(errs, fmt.Errorf("%w: receipt value_dim mismatch", ErrInvalidBenchmarkSpec))
	}
	if len(metadata.WeightedValue) != spec.ValueDim {
		errs = append(errs, fmt.Errorf("%w: receipt weighted_value length mismatch", ErrInvalidBenchmarkSpec))
	}
	if metadata.ComputeTimeMs < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt compute_time_ms must be non-negative", ErrInvalidBenchmarkSpec))
	}
	if metadata.ComputeTimeUs < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt compute_time_us must be non-negative", ErrInvalidBenchmarkSpec))
	}
	if metadata.SimulatedDelayMs < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt simulated_delay_ms must be non-negative", ErrInvalidBenchmarkSpec))
	}
	if metadata.TotalNodeWallTimeMs < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt total_node_wall_time_ms must be non-negative", ErrInvalidBenchmarkSpec))
	}
	if metadata.TotalNodeWallTimeUs < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt total_node_wall_time_us must be non-negative", ErrInvalidBenchmarkSpec))
	}
	if metadata.NodeStartedAtUnixMs > 0 && metadata.NodeCompletedAtUnixMs > 0 && metadata.NodeCompletedAtUnixMs < metadata.NodeStartedAtUnixMs {
		errs = append(errs, fmt.Errorf("%w: receipt node_completed_at_unix_ms must be at or after node_started_at_unix_ms", ErrInvalidBenchmarkSpec))
	}
	if metadata.SummaryPayloadBytesEstimate <= 0 {
		errs = append(errs, fmt.Errorf("%w: receipt summary_payload_bytes_estimate must be positive", ErrInvalidBenchmarkSpec))
	}
	if metadata.OutputBytesEstimate <= 0 {
		errs = append(errs, fmt.Errorf("%w: receipt output_bytes_estimate must be positive", ErrInvalidBenchmarkSpec))
	}
	if metadata.ReceiptMetadataJSONBytes < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt receipt_metadata_json_bytes must be non-negative", ErrInvalidBenchmarkSpec))
	}
	if metadata.ReceiptEnvelopeJSONBytes < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt receipt_envelope_json_bytes must be non-negative", ErrInvalidBenchmarkSpec))
	}
	if strings.TrimSpace(metadata.ProofStatus) != "synthetic_measured" {
		errs = append(errs, fmt.Errorf("%w: receipt proof_status must be synthetic_measured", ErrInvalidBenchmarkSpec))
	}
	return errors.Join(errs...)
}

type benchmarkReceiptHashMetadata struct {
	RequestID                   string    `json:"request_id"`
	ShardID                     string    `json:"shard_id"`
	LocalMax                    float64   `json:"local_max"`
	ExpSum                      float64   `json:"exp_sum"`
	WeightedValue               []float64 `json:"weighted_value"`
	TokenCount                  int       `json:"token_count"`
	ValueDim                    int       `json:"value_dim"`
	SummaryPayloadBytesEstimate int64     `json:"summary_payload_bytes_estimate"`
	OutputBytesEstimate         int64     `json:"output_bytes_estimate"`
	ProofStatus                 string    `json:"proof_status"`
}

func benchmarkReceiptHashMetadataFromMetadata(metadata BenchmarkReceiptMetadata) benchmarkReceiptHashMetadata {
	return benchmarkReceiptHashMetadata{
		RequestID:                   metadata.RequestID,
		ShardID:                     metadata.ShardID,
		LocalMax:                    metadata.LocalMax,
		ExpSum:                      metadata.ExpSum,
		WeightedValue:               append([]float64(nil), metadata.WeightedValue...),
		TokenCount:                  metadata.TokenCount,
		ValueDim:                    metadata.ValueDim,
		SummaryPayloadBytesEstimate: metadata.SummaryPayloadBytesEstimate,
		OutputBytesEstimate:         metadata.OutputBytesEstimate,
		ProofStatus:                 metadata.ProofStatus,
	}
}

type benchmarkReceiptJSONEnvelope struct {
	JobID         string         `json:"job_id"`
	ResultHashHex string         `json:"result_hash_hex"`
	MeteringUnits uint64         `json:"metering_units"`
	Metadata      map[string]any `json:"metadata"`
}

func populateBenchmarkReceiptJSONByteEstimates(metadata *BenchmarkReceiptMetadata, jobID string, resultHashHex string, meteringUnits uint64) error {
	for i := 0; i < 8; i++ {
		metadataMap := metadata.Map()
		metadataEncoded, err := json.Marshal(metadataMap)
		if err != nil {
			return err
		}
		envelopeEncoded, err := json.Marshal(benchmarkReceiptJSONEnvelope{
			JobID:         jobID,
			ResultHashHex: resultHashHex,
			MeteringUnits: meteringUnits,
			Metadata: map[string]any{
				BenchmarkTask: metadataMap,
			},
		})
		if err != nil {
			return err
		}

		nextMetadataBytes := int64(len(metadataEncoded))
		nextEnvelopeBytes := int64(len(envelopeEncoded))
		if metadata.ReceiptMetadataJSONBytes == nextMetadataBytes && metadata.ReceiptEnvelopeJSONBytes == nextEnvelopeBytes {
			return nil
		}
		metadata.ReceiptMetadataJSONBytes = nextMetadataBytes
		metadata.ReceiptEnvelopeJSONBytes = nextEnvelopeBytes
	}
	return nil
}

func receiptBuildDurationFields(duration time.Duration) (int64, int64) {
	if duration <= 0 {
		return 0, 0
	}
	return duration.Milliseconds(), duration.Microseconds()
}
