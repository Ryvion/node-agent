package llamacpp

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	EnvBenchmark = "RYV_LLAMA_CPP_BENCH"

	DefaultBenchmarkMaxTokens    = 32
	DefaultBenchmarkTimeoutMs    = int64(60_000)
	DefaultBenchmarkMeasuredRuns = 3
	DefaultBenchmarkWarmupRuns   = 1

	BenchmarkStatusCompleted = "completed"
	BenchmarkStatusFailed    = "failed"

	BenchmarkProofStatusMeasured    = "llamacpp_backend_measured"
	BenchmarkProofStatusUnavailable = "llamacpp_backend_unavailable"
	BenchmarkProofStatusFailed      = "llamacpp_backend_failed"
)

const internalBenchmarkPrompt = "Write one short sentence about distributed computing."

type BenchmarkConfig struct {
	NodeID              string  `json:"node_id,omitempty"`
	ModelID             string  `json:"model_id,omitempty"`
	MaxTokens           int     `json:"max_tokens"`
	Temperature         float64 `json:"temperature"`
	TimeoutMs           int64   `json:"timeout_ms"`
	Streaming           bool    `json:"streaming"`
	MeasuredRuns        int     `json:"measured_runs"`
	WarmupRuns          int     `json:"warmup_runs"`
	Acceleration        string  `json:"acceleration,omitempty"`
	Warm                bool    `json:"warm"`
	ContextLengthTokens int     `json:"context_length_tokens,omitempty"`
	StreamingSupported  bool    `json:"streaming_supported"`
}

type BenchmarkMetrics struct {
	NodeID              string  `json:"node_id,omitempty"`
	Available           bool    `json:"available"`
	SidecarHealthy      bool    `json:"sidecar_healthy"`
	ModelLoaded         bool    `json:"model_loaded"`
	ModelID             string  `json:"model_id,omitempty"`
	ModelPath           string  `json:"model_path,omitempty"`
	ModelFilename       string  `json:"model_filename,omitempty"`
	PromptHash          string  `json:"prompt_hash"`
	OutputHash          string  `json:"output_hash,omitempty"`
	OutputBytes         int64   `json:"output_bytes"`
	WarmupRuns          int     `json:"warmup_runs"`
	MeasuredRuns        int     `json:"measured_runs"`
	P50TTFTMs           int64   `json:"p50_ttft_ms"`
	P95TTFTMs           int64   `json:"p95_ttft_ms"`
	P50TotalTimeMs      int64   `json:"p50_total_time_ms"`
	P95TotalTimeMs      int64   `json:"p95_total_time_ms"`
	P50DecodeTPS        float64 `json:"p50_decode_tps"`
	P95DecodeTPS        float64 `json:"p95_decode_tps"`
	P50TPOTMs           float64 `json:"p50_tpot_ms"`
	P95TPOTMs           float64 `json:"p95_tpot_ms"`
	P50EndToEndTPS      float64 `json:"p50_end_to_end_tps"`
	P95EndToEndTPS      float64 `json:"p95_end_to_end_tps"`
	TokensGenerated     int64   `json:"tokens_generated"`
	Backend             string  `json:"backend"`
	RuntimeKind         string  `json:"runtime_kind"`
	Acceleration        string  `json:"acceleration"`
	Warm                bool    `json:"warm"`
	ContextLengthBucket string  `json:"context_length_bucket,omitempty"`
	OutputTokenBucket   string  `json:"output_token_bucket,omitempty"`
	ProofStatus         string  `json:"proof_status"`
	Streaming           bool    `json:"streaming"`
	StreamingSupported  bool    `json:"streaming_supported"`
	PromptTokens        int64   `json:"prompt_tokens,omitempty"`
	CompletionTokens    int64   `json:"completion_tokens,omitempty"`
}

type BenchmarkStatusSnapshot struct {
	LastRunAt time.Time        `json:"last_run_at,omitempty"`
	Status    string           `json:"status"`
	Metrics   BenchmarkMetrics `json:"metrics"`
	LastError string           `json:"last_error"`
}

type BenchmarkLocalStatus struct {
	mu       sync.RWMutex
	snapshot BenchmarkStatusSnapshot
}

type BenchmarkSidecar interface {
	Start(context.Context) LlamaCppSidecarStatus
	Status(context.Context) LlamaCppSidecarStatus
}

type BenchmarkRunner struct {
	Sidecar BenchmarkSidecar
	Client  CompletionClient
	Now     func() time.Time
}

type runMeasurement struct {
	ttftMs           int64
	totalTimeMs      int64
	decodeTPS        float64
	endToEndTPS      float64
	tokensGenerated  int64
	promptTokens     int64
	completionTokens int64
	output           []byte
}

func NewBenchmarkLocalStatus() *BenchmarkLocalStatus {
	return &BenchmarkLocalStatus{}
}

func (s *BenchmarkLocalStatus) Record(snapshot BenchmarkStatusSnapshot) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot = normalizeBenchmarkStatusSnapshot(snapshot)
}

func (s *BenchmarkLocalStatus) Snapshot() BenchmarkStatusSnapshot {
	if s == nil {
		return BenchmarkStatusSnapshot{}
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return normalizeBenchmarkStatusSnapshot(s.snapshot)
}

func BenchmarkEnabledFromEnv(getenv func(string) string) bool {
	if getenv == nil {
		return false
	}
	return strings.TrimSpace(getenv(EnvBenchmark)) == "1"
}

func DefaultBenchmarkConfig() BenchmarkConfig {
	return BenchmarkConfig{
		MaxTokens:    DefaultBenchmarkMaxTokens,
		Temperature:  0,
		TimeoutMs:    DefaultBenchmarkTimeoutMs,
		Streaming:    true,
		MeasuredRuns: DefaultBenchmarkMeasuredRuns,
		WarmupRuns:   DefaultBenchmarkWarmupRuns,
	}
}

func ValidateBenchmarkConfig(config BenchmarkConfig) error {
	if config.MaxTokens < 0 {
		return errors.New("max_tokens must be non-negative")
	}
	if config.TimeoutMs < 0 {
		return errors.New("timeout_ms must be non-negative")
	}
	if config.MeasuredRuns < 0 {
		return errors.New("measured_runs must be non-negative")
	}
	if config.WarmupRuns < 0 {
		return errors.New("warmup_runs must be non-negative")
	}
	if config.Temperature < 0 {
		return errors.New("temperature must be non-negative")
	}
	return nil
}

func NormalizeBenchmarkConfig(config BenchmarkConfig) BenchmarkConfig {
	defaults := DefaultBenchmarkConfig()
	config.NodeID = cleanStatusText(config.NodeID, maxStatusReasonLen)
	config.ModelID = cleanStatusText(config.ModelID, maxStatusReasonLen)
	config.Acceleration = normalizeBenchmarkAcceleration(config.Acceleration)
	if config.MaxTokens == 0 {
		config.MaxTokens = defaults.MaxTokens
	}
	if config.TimeoutMs == 0 {
		config.TimeoutMs = defaults.TimeoutMs
	}
	if config.MeasuredRuns == 0 {
		config.MeasuredRuns = defaults.MeasuredRuns
	}
	if config.WarmupRuns < 0 {
		config.WarmupRuns = 0
	}
	if config.Temperature < 0 {
		config.Temperature = 0
	}
	if config.ContextLengthTokens < 0 {
		config.ContextLengthTokens = 0
	}
	return config
}

func HashBenchmarkPrompt() string {
	sum := sha256.Sum256([]byte("ryvion:v7:llamacpp_benchmark_prompt:v1\n" + internalBenchmarkPrompt))
	return "sha256:" + hex.EncodeToString(sum[:])
}

func KeepWarmEnabledFromEnv(getenv func(string) string) bool {
	if getenv == nil {
		return false
	}
	if envBool(getenv(EnvDisableModelWarm)) {
		return false
	}
	return envBoolDefault(getenv(EnvKeepWarm), true)
}

func CompleteInternalBenchmarkPrompt(ctx context.Context, client CompletionClient, req CompletionRequest) (CompletionResult, bool, error) {
	if client == nil {
		client = OpenAIClient{}
	}
	req.Prompt = internalBenchmarkPrompt
	return completeWithFallback(ctx, client, req)
}

func (r BenchmarkRunner) Run(ctx context.Context, config BenchmarkConfig) BenchmarkStatusSnapshot {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ValidateBenchmarkConfig(config); err != nil {
		return r.failedSnapshot(config, LlamaCppSidecarStatus{Backend: BackendName}, "llamacpp_invalid_config")
	}
	config = NormalizeBenchmarkConfig(config)

	runCtx, cancel := context.WithTimeout(ctx, time.Duration(config.TimeoutMs)*time.Millisecond)
	defer cancel()

	sidecar := r.sidecar()
	status := sidecar.Start(runCtx)
	metrics := baseBenchmarkMetrics(status, config)
	if !status.Enabled || !status.Available || !status.Healthy {
		metrics.ProofStatus = BenchmarkProofStatusUnavailable
		return r.snapshot(BenchmarkStatusFailed, metrics, safeBenchmarkFailure(status))
	}

	client := r.client()
	modelID := config.ModelID
	if modelID == "" {
		modelID = firstNonEmpty(status.ModelFilename, status.ModelPath, BackendName)
	}
	metrics.ModelID = cleanStatusText(modelID, maxStatusReasonLen)
	metrics.ModelLoaded = strings.TrimSpace(status.ModelPath) != "" && status.Healthy

	streaming := config.Streaming
	measurements := make([]runMeasurement, 0, config.MeasuredRuns)
	outputs := make([][]byte, 0, config.MeasuredRuns)
	for i := 0; i < config.WarmupRuns+config.MeasuredRuns; i++ {
		req := CompletionRequest{
			BaseURL:     status.BaseURL,
			ModelID:     modelID,
			Prompt:      internalBenchmarkPrompt,
			MaxTokens:   config.MaxTokens,
			Temperature: config.Temperature,
			Stream:      streaming,
		}
		result, usedStreaming, err := completeWithFallback(runCtx, client, req)
		if err != nil {
			metrics.ProofStatus = BenchmarkProofStatusFailed
			return r.snapshot(BenchmarkStatusFailed, metrics, safeBenchmarkErrorCode(err))
		}
		streaming = usedStreaming
		metrics.Streaming = usedStreaming
		if i < config.WarmupRuns {
			continue
		}
		result = maybeProbeNonStreamingDecode(runCtx, client, req, result, usedStreaming)
		measurement := measurementFromCompletion(result)
		measurements = append(measurements, measurement)
		outputs = append(outputs, append([]byte(nil), result.Output...))
	}
	if len(measurements) == 0 {
		metrics.ProofStatus = BenchmarkProofStatusFailed
		return r.snapshot(BenchmarkStatusFailed, metrics, "llamacpp_no_measured_runs")
	}

	metrics = applyMeasurements(metrics, measurements, outputs)
	metrics.ProofStatus = BenchmarkProofStatusMeasured
	return r.snapshot(BenchmarkStatusCompleted, metrics, "")
}

func completeWithFallback(ctx context.Context, client CompletionClient, req CompletionRequest) (CompletionResult, bool, error) {
	result, err := client.Complete(ctx, req)
	if err == nil {
		return result, req.Stream, nil
	}
	if req.Stream && IsStreamUnavailable(err) {
		req.Stream = false
		result, err = client.Complete(ctx, req)
		if err == nil {
			return result, false, nil
		}
	}
	return CompletionResult{}, req.Stream, err
}

func maybeProbeNonStreamingDecode(ctx context.Context, client CompletionClient, req CompletionRequest, result CompletionResult, usedStreaming bool) CompletionResult {
	if !usedStreaming || client == nil || result.BackendDecodeTPS <= 0 {
		return result
	}
	probeReq := req
	probeReq.Stream = false
	probeReq.OnDelta = nil
	probe, err := client.Complete(ctx, probeReq)
	if err != nil {
		return result
	}
	if probe.BackendDecodeTPS > result.BackendDecodeTPS {
		result.applyBackendDecodeStats(probe.BackendDecodeMs, probe.BackendDecodeTPS)
	}
	return result
}

func (r BenchmarkRunner) failedSnapshot(config BenchmarkConfig, status LlamaCppSidecarStatus, code string) BenchmarkStatusSnapshot {
	config = NormalizeBenchmarkConfig(config)
	metrics := baseBenchmarkMetrics(status, config)
	metrics.ProofStatus = BenchmarkProofStatusFailed
	return r.snapshot(BenchmarkStatusFailed, metrics, code)
}

func (r BenchmarkRunner) snapshot(status string, metrics BenchmarkMetrics, lastError string) BenchmarkStatusSnapshot {
	metrics = normalizeBenchmarkMetrics(metrics)
	return normalizeBenchmarkStatusSnapshot(BenchmarkStatusSnapshot{
		LastRunAt: r.now(),
		Status:    cleanBenchmarkStatus(status),
		Metrics:   metrics,
		LastError: cleanBenchmarkError(lastError),
	})
}

func baseBenchmarkMetrics(status LlamaCppSidecarStatus, config BenchmarkConfig) BenchmarkMetrics {
	modelID := strings.TrimSpace(config.ModelID)
	if modelID == "" {
		modelID = firstNonEmpty(status.ModelFilename, status.ModelPath)
	}
	return normalizeBenchmarkMetrics(BenchmarkMetrics{
		NodeID:              config.NodeID,
		Available:           status.Available,
		SidecarHealthy:      status.Healthy,
		ModelLoaded:         status.Healthy && strings.TrimSpace(status.ModelPath) != "",
		ModelID:             modelID,
		ModelPath:           status.ModelPath,
		ModelFilename:       status.ModelFilename,
		PromptHash:          HashBenchmarkPrompt(),
		WarmupRuns:          config.WarmupRuns,
		MeasuredRuns:        config.MeasuredRuns,
		Backend:             BackendName,
		RuntimeKind:         BackendName,
		Acceleration:        config.Acceleration,
		Warm:                config.Warm,
		ContextLengthBucket: contextLengthBucket(config.ContextLengthTokens),
		OutputTokenBucket:   outputTokenBucket(config.MaxTokens),
		ProofStatus:         BenchmarkProofStatusUnavailable,
		Streaming:           config.Streaming,
		StreamingSupported:  config.StreamingSupported || status.SupportsStreaming,
		TokensGenerated:     0,
	})
}

func measurementFromCompletion(result CompletionResult) runMeasurement {
	total := result.TotalTimeMs
	if total < 0 {
		total = 0
	}
	ttft := result.TTFTMs
	if ttft < 0 {
		ttft = 0
	}
	if ttft > total {
		ttft = total
	}
	tokens := result.TokensGenerated
	if tokens <= 0 {
		tokens = approximateGeneratedTokens(string(result.Output))
	}
	decodeWindowMs := total - ttft
	if decodeWindowMs <= 0 {
		decodeWindowMs = total
	}
	var decodeTPS float64
	if result.BackendDecodeTPS > 0 {
		decodeTPS = result.BackendDecodeTPS
	} else if result.BackendDecodeMs > 0 && tokens > 0 {
		decodeTPS = float64(tokens) / (result.BackendDecodeMs / 1000)
	} else if tokens > 0 && decodeWindowMs > 0 {
		decodeTPS = float64(tokens) / (float64(decodeWindowMs) / 1000)
	}
	var endToEndTPS float64
	if tokens > 0 && total > 0 {
		endToEndTPS = float64(tokens) / (float64(total) / 1000)
	}
	return runMeasurement{
		ttftMs:           ttft,
		totalTimeMs:      total,
		decodeTPS:        decodeTPS,
		endToEndTPS:      endToEndTPS,
		tokensGenerated:  tokens,
		promptTokens:     result.PromptTokens,
		completionTokens: result.CompletionTokens,
		output:           append([]byte(nil), result.Output...),
	}
}

func applyMeasurements(metrics BenchmarkMetrics, measurements []runMeasurement, outputs [][]byte) BenchmarkMetrics {
	ttfts := make([]int64, 0, len(measurements))
	totals := make([]int64, 0, len(measurements))
	decodeTPS := make([]float64, 0, len(measurements))
	endToEndTPS := make([]float64, 0, len(measurements))
	var tokensGenerated int64
	var promptTokens int64
	var completionTokens int64
	var outputBytes int64
	for _, measurement := range measurements {
		ttfts = append(ttfts, measurement.ttftMs)
		totals = append(totals, measurement.totalTimeMs)
		decodeTPS = append(decodeTPS, measurement.decodeTPS)
		endToEndTPS = append(endToEndTPS, measurement.endToEndTPS)
		tokensGenerated += measurement.tokensGenerated
		promptTokens += measurement.promptTokens
		completionTokens += measurement.completionTokens
		outputBytes += int64(len(measurement.output))
	}
	metrics.P50TTFTMs = percentileInt64(ttfts, 50)
	metrics.P95TTFTMs = percentileInt64(ttfts, 95)
	metrics.P50TotalTimeMs = percentileInt64(totals, 50)
	metrics.P95TotalTimeMs = percentileInt64(totals, 95)
	metrics.P50DecodeTPS = roundTPS(percentileFloat64(decodeTPS, 50))
	metrics.P95DecodeTPS = roundTPS(percentileFloat64(decodeTPS, 95))
	metrics.P50TPOTMs = tpotMsFromTPS(metrics.P50DecodeTPS)
	metrics.P95TPOTMs = tpotMsFromTPS(metrics.P95DecodeTPS)
	metrics.P50EndToEndTPS = roundTPS(percentileFloat64(endToEndTPS, 50))
	metrics.P95EndToEndTPS = roundTPS(percentileFloat64(endToEndTPS, 95))
	metrics.TokensGenerated = tokensGenerated
	metrics.PromptTokens = promptTokens
	metrics.CompletionTokens = completionTokens
	metrics.OutputBytes = outputBytes
	metrics.OutputHash = hashBenchmarkOutputs(outputs)
	return normalizeBenchmarkMetrics(metrics)
}

func hashBenchmarkOutputs(outputs [][]byte) string {
	if len(outputs) == 0 {
		return ""
	}
	hash := sha256.New()
	hash.Write([]byte("ryvion:v7:llamacpp_benchmark_output:v1\n"))
	for _, output := range outputs {
		hash.Write([]byte("run-bytes:"))
		hash.Write([]byte(fmt.Sprintf("%d\n", len(output))))
		hash.Write(output)
		hash.Write([]byte("\n"))
	}
	return "sha256:" + hex.EncodeToString(hash.Sum(nil))
}

func percentileInt64(values []int64, percentile float64) int64 {
	if len(values) == 0 {
		return 0
	}
	sorted := append([]int64(nil), values...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	return sorted[percentileIndex(len(sorted), percentile)]
}

func percentileFloat64(values []float64, percentile float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	return sorted[percentileIndex(len(sorted), percentile)]
}

func percentileIndex(length int, percentile float64) int {
	if length <= 1 {
		return 0
	}
	if percentile <= 0 {
		return 0
	}
	if percentile >= 100 {
		return length - 1
	}
	idx := int(math.Ceil((percentile/100)*float64(length))) - 1
	if idx < 0 {
		return 0
	}
	if idx >= length {
		return length - 1
	}
	return idx
}

func roundTPS(value float64) float64 {
	if value <= 0 || math.IsNaN(value) || math.IsInf(value, 0) {
		return 0
	}
	return math.Round(value*1000) / 1000
}

func roundMillis(value float64) float64 {
	if value <= 0 || math.IsNaN(value) || math.IsInf(value, 0) {
		return 0
	}
	return math.Round(value*1000) / 1000
}

func tpotMsFromTPS(tps float64) float64 {
	if tps <= 0 || math.IsNaN(tps) || math.IsInf(tps, 0) {
		return 0
	}
	return roundMillis(1000 / tps)
}

func FormatBenchmarkStatus(snapshot BenchmarkStatusSnapshot, jsonOutput bool) string {
	snapshot = normalizeBenchmarkStatusSnapshot(snapshot)
	if jsonOutput {
		raw, err := json.Marshal(snapshot)
		if err != nil {
			return `{"status":"failed","last_error":"llamacpp_json_encode_failed"}`
		}
		return string(raw)
	}
	return fmt.Sprintf("llama.cpp benchmark status=%s available=%t healthy=%t p50_ttft_ms=%d p50_decode_tps=%.3f proof_status=%s",
		snapshot.Status,
		snapshot.Metrics.Available,
		snapshot.Metrics.SidecarHealthy,
		snapshot.Metrics.P50TTFTMs,
		snapshot.Metrics.P50DecodeTPS,
		snapshot.Metrics.ProofStatus,
	)
}

func normalizeBenchmarkStatusSnapshot(snapshot BenchmarkStatusSnapshot) BenchmarkStatusSnapshot {
	snapshot.Status = cleanBenchmarkStatus(snapshot.Status)
	snapshot.Metrics = normalizeBenchmarkMetrics(snapshot.Metrics)
	snapshot.LastError = cleanBenchmarkError(snapshot.LastError)
	return snapshot
}

func normalizeBenchmarkMetrics(metrics BenchmarkMetrics) BenchmarkMetrics {
	metrics.NodeID = cleanStatusText(metrics.NodeID, maxStatusReasonLen)
	metrics.ModelID = cleanStatusText(metrics.ModelID, maxStatusReasonLen)
	metrics.ModelPath = cleanStatusText(metrics.ModelPath, maxConfigTextLen)
	metrics.ModelFilename = cleanStatusText(metrics.ModelFilename, maxStatusReasonLen)
	metrics.PromptHash = cleanHash(metrics.PromptHash)
	metrics.OutputHash = cleanHash(metrics.OutputHash)
	metrics.Backend = cleanStatusText(firstNonEmpty(metrics.Backend, BackendName), maxStatusReasonLen)
	metrics.RuntimeKind = cleanStatusText(firstNonEmpty(metrics.RuntimeKind, BackendName), maxStatusReasonLen)
	metrics.Acceleration = normalizeBenchmarkAcceleration(metrics.Acceleration)
	metrics.ContextLengthBucket = normalizeLengthBucket(metrics.ContextLengthBucket)
	metrics.OutputTokenBucket = normalizeLengthBucket(metrics.OutputTokenBucket)
	metrics.ProofStatus = cleanStatusText(metrics.ProofStatus, maxStatusReasonLen)
	if metrics.ProofStatus == "" {
		metrics.ProofStatus = BenchmarkProofStatusUnavailable
	}
	if metrics.WarmupRuns < 0 {
		metrics.WarmupRuns = 0
	}
	if metrics.MeasuredRuns < 0 {
		metrics.MeasuredRuns = 0
	}
	if metrics.OutputBytes < 0 {
		metrics.OutputBytes = 0
	}
	if metrics.TokensGenerated < 0 {
		metrics.TokensGenerated = 0
	}
	if metrics.PromptTokens < 0 {
		metrics.PromptTokens = 0
	}
	if metrics.CompletionTokens < 0 {
		metrics.CompletionTokens = 0
	}
	if metrics.Streaming {
		metrics.StreamingSupported = true
	}
	metrics.P50DecodeTPS = roundTPS(metrics.P50DecodeTPS)
	metrics.P95DecodeTPS = roundTPS(metrics.P95DecodeTPS)
	if metrics.P50TPOTMs <= 0 {
		metrics.P50TPOTMs = tpotMsFromTPS(metrics.P50DecodeTPS)
	} else {
		metrics.P50TPOTMs = roundMillis(metrics.P50TPOTMs)
	}
	if metrics.P95TPOTMs <= 0 {
		metrics.P95TPOTMs = tpotMsFromTPS(metrics.P95DecodeTPS)
	} else {
		metrics.P95TPOTMs = roundMillis(metrics.P95TPOTMs)
	}
	metrics.P50EndToEndTPS = roundTPS(metrics.P50EndToEndTPS)
	metrics.P95EndToEndTPS = roundTPS(metrics.P95EndToEndTPS)
	return metrics
}

func normalizeBenchmarkAcceleration(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "cuda":
		return "cuda"
	case "vulkan":
		return "vulkan"
	case "cpu", "":
		return "cpu"
	default:
		return "other"
	}
}

func contextLengthBucket(tokens int) string {
	switch {
	case tokens <= 0:
		return "unknown"
	case tokens <= 2048:
		return "ctx_0_2k"
	case tokens <= 4096:
		return "ctx_2k_4k"
	case tokens <= 8192:
		return "ctx_4k_8k"
	case tokens <= 32768:
		return "ctx_8k_32k"
	default:
		return "ctx_32k_plus"
	}
}

func outputTokenBucket(tokens int) string {
	switch {
	case tokens <= 0:
		return "unknown"
	case tokens <= 64:
		return "out_0_64"
	case tokens <= 256:
		return "out_65_256"
	case tokens <= 1024:
		return "out_257_1024"
	default:
		return "out_1025_plus"
	}
}

func normalizeLengthBucket(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	switch value {
	case "unknown",
		"ctx_0_2k", "ctx_2k_4k", "ctx_4k_8k", "ctx_8k_32k", "ctx_32k_plus",
		"out_0_64", "out_65_256", "out_257_1024", "out_1025_plus":
		return value
	default:
		return ""
	}
}

func cleanHash(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	if !strings.HasPrefix(value, "sha256:") || len(value) != len("sha256:")+64 {
		return ""
	}
	for _, r := range strings.TrimPrefix(value, "sha256:") {
		if (r < '0' || r > '9') && (r < 'a' || r > 'f') {
			return ""
		}
	}
	return value
}

func cleanBenchmarkStatus(status string) string {
	switch strings.TrimSpace(status) {
	case BenchmarkStatusCompleted:
		return BenchmarkStatusCompleted
	case BenchmarkStatusFailed:
		return BenchmarkStatusFailed
	default:
		return strings.TrimSpace(status)
	}
}

func cleanBenchmarkError(value string) string {
	value = cleanStatusText(value, maxStatusReasonLen)
	if value == "" {
		return ""
	}
	lower := strings.ToLower(value)
	if strings.Contains(lower, strings.ToLower(internalBenchmarkPrompt)) {
		return "llamacpp_benchmark_error_redacted"
	}
	return value
}

func safeBenchmarkFailure(status LlamaCppSidecarStatus) string {
	switch {
	case !status.Enabled:
		return "llamacpp_sidecar_disabled"
	case !status.Available:
		return "llamacpp_sidecar_unavailable"
	case !status.Healthy:
		return "llamacpp_sidecar_unhealthy"
	default:
		return "llamacpp_benchmark_unavailable"
	}
}

func safeBenchmarkErrorCode(err error) string {
	if err == nil {
		return ""
	}
	var clientErr ClientError
	if errorAs(err, &clientErr) && strings.TrimSpace(clientErr.Code) != "" {
		return cleanBenchmarkError(clientErr.Code)
	}
	return cleanBenchmarkError(defaultClientErrorCode)
}

func (r BenchmarkRunner) sidecar() BenchmarkSidecar {
	if r.Sidecar != nil {
		return r.Sidecar
	}
	return NewManagerFromEnv()
}

func (r BenchmarkRunner) client() CompletionClient {
	if r.Client != nil {
		return r.Client
	}
	return OpenAIClient{}
}

func (r BenchmarkRunner) now() time.Time {
	if r.Now != nil {
		return r.Now()
	}
	return time.Now()
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func errorAs(err error, target any) bool {
	return errors.As(err, target)
}

func BenchmarkJSONContainsNoRawText(snapshot BenchmarkStatusSnapshot) bool {
	raw, err := json.Marshal(snapshot)
	if err != nil {
		return false
	}
	lower := bytes.ToLower(raw)
	for _, forbidden := range [][]byte{
		[]byte(strings.ToLower(internalBenchmarkPrompt)),
		[]byte("raw_prompt"),
		[]byte("prompt_text"),
		[]byte("generated_text"),
		[]byte("output_text"),
		[]byte("model_output"),
		[]byte("tensor_bytes"),
		[]byte("raw_tensor"),
	} {
		if bytes.Contains(lower, forbidden) {
			return false
		}
	}
	return true
}
