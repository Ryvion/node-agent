package inferencebench

import (
	"context"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"

	"github.com/Ryvion/ryvion-node/internal/v7/llamacpp"
)

const (
	ProofStatusMeasured = "backend_inference_measured"
	ProofStatusRejected = "rejected"
	ProofStatusFailed   = "backend_inference_failed"
)

type BenchmarkRunner interface {
	RunBackendInferenceBenchmark(context.Context, BenchmarkSpec) (BenchmarkExecutionResult, error)
}

type ExecuteOptions struct {
	Getenv func(string) string
	Runner BenchmarkRunner
}

type LlamaCppSidecar interface {
	Start(context.Context) llamacpp.LlamaCppSidecarStatus
	Status(context.Context) llamacpp.LlamaCppSidecarStatus
}

type KeepWarmChecker interface {
	CheckOnce(context.Context) llamacpp.LlamaCppSidecarStatus
}

type LlamaCppStopper interface {
	Stop(context.Context) llamacpp.LlamaCppSidecarStatus
}

type LlamaCppBenchmarkRunner struct {
	Sidecar  LlamaCppSidecar
	KeepWarm KeepWarmChecker
	Client   llamacpp.CompletionClient
	Getenv   func(string) string
}

type BenchmarkExecutionResult struct {
	Spec            BenchmarkSpec
	Backend         string
	ModelID         string
	PromptHash      string
	OutputHash      string
	OutputBytes     int64
	TokensGenerated int64
	TTFTMs          int64
	P95TTFTMs       int64
	TotalTimeMs     int64
	DecodeTPS       float64
	EndToEndTPS     float64
	ProofStatus     string
	ErrorCode       string
}

type LocalStatusCounters struct {
	Seen             uint64 `json:"seen"`
	Executed         uint64 `json:"executed"`
	ReceiptSubmitted uint64 `json:"receipt_submitted"`
	ReceiptFailed    uint64 `json:"receipt_failed"`
}

type LocalStatusSnapshot struct {
	LastJobID string              `json:"last_job_id"`
	LastError string              `json:"last_error"`
	Counters  LocalStatusCounters `json:"counters"`
}

type LocalStatus struct {
	mu       sync.RWMutex
	snapshot LocalStatusSnapshot
}

func NewLocalStatus() *LocalStatus {
	return &LocalStatus{}
}

func (s *LocalStatus) RecordSeen(jobID string) {
	s.recordSeen(jobID)
}

func (s *LocalStatus) RecordExecuted(jobID string) {
	s.recordExecuted(jobID)
}

func (s *LocalStatus) RecordReceiptSubmitted(jobID string) {
	s.recordReceiptSubmitted(jobID)
}

func (s *LocalStatus) RecordReceiptFailed(jobID string, err error) {
	s.recordReceiptFailed(jobID, err)
}

func (s *LocalStatus) RecordError(jobID string, err error) {
	s.recordError(jobID, err)
}

func (s *LocalStatus) Snapshot() LocalStatusSnapshot {
	if s == nil {
		return LocalStatusSnapshot{}
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.snapshot
}

func ExecuteBenchmarkAssignment(ctx context.Context, specJSON string, opts ExecuteOptions) (BenchmarkReceipt, bool, error) {
	if !IsBenchmarkSpecJSON(specJSON) {
		return BenchmarkReceipt{}, false, nil
	}
	if !BenchmarkEnabledFromEnv(opts.Getenv) {
		return BenchmarkReceipt{}, false, nil
	}
	spec, err := DecodeBenchmarkSpec(specJSON)
	if err != nil {
		return BenchmarkReceipt{}, true, err
	}
	receipt, err := ExecuteBenchmarkSpec(ctx, spec, opts)
	return receipt, true, err
}

func ExecuteBenchmarkSpec(ctx context.Context, spec BenchmarkSpec, opts ExecuteOptions) (BenchmarkReceipt, error) {
	spec = normalizeBenchmarkSpec(spec)
	if err := ValidateBenchmarkSpec(spec); err != nil {
		return BenchmarkReceipt{}, err
	}
	runner := opts.Runner
	if runner == nil {
		runner = LlamaCppBenchmarkRunner{
			Sidecar: llamacpp.NewManagerFromEnv(),
			Client:  llamacpp.OpenAIClient{},
			Getenv:  opts.Getenv,
		}
	}
	result, err := runner.RunBackendInferenceBenchmark(ctx, spec)
	if err != nil {
		return BenchmarkReceipt{}, err
	}
	return BuildBenchmarkReceipt(result)
}

func (r LlamaCppBenchmarkRunner) RunBackendInferenceBenchmark(ctx context.Context, spec BenchmarkSpec) (BenchmarkExecutionResult, error) {
	spec = normalizeBenchmarkSpec(spec)
	if err := ValidateBenchmarkSpec(spec); err != nil {
		return BenchmarkExecutionResult{}, err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	runCtx, cancel := context.WithTimeout(ctx, time.Duration(spec.TimeoutMs)*time.Millisecond)
	defer cancel()
	if err := runCtx.Err(); err != nil {
		return failedExecutionResult(spec, "benchmark_context_unavailable"), nil
	}

	sidecar := r.sidecar()
	defer r.stopSidecarAfterRunIfIdle(context.Background(), sidecar)
	status := r.ensureSidecarWith(runCtx, sidecar)
	if !status.Enabled || !status.Available || !status.Running || !status.Healthy {
		return failedExecutionResult(spec, safeSidecarFailure(status)), nil
	}

	client := r.client()
	req := llamacpp.CompletionRequest{
		BaseURL:     status.BaseURL,
		ModelID:     spec.ModelID,
		MaxTokens:   spec.MaxTokens,
		Temperature: 0,
		Stream:      status.SupportsStreaming,
	}
	completion, _, err := llamacpp.CompleteInternalBenchmarkPrompt(runCtx, client, req)
	if err != nil {
		return failedExecutionResult(spec, safeBenchmarkErrorCode(err)), nil
	}
	return measuredExecutionResult(spec, completion), nil
}

func (r LlamaCppBenchmarkRunner) ensureSidecar(ctx context.Context) llamacpp.LlamaCppSidecarStatus {
	return r.ensureSidecarWith(ctx, r.sidecar())
}

func (r LlamaCppBenchmarkRunner) ensureSidecarWith(ctx context.Context, sidecar LlamaCppSidecar) llamacpp.LlamaCppSidecarStatus {
	keepWarm := llamacpp.KeepWarmEnabledFromEnv(r.getenv())
	if keepWarm {
		if r.KeepWarm != nil {
			status := r.KeepWarm.CheckOnce(ctx)
			if status.Enabled && status.Available && status.Running && status.Healthy {
				return status
			}
		}
		return sidecar.Start(ctx)
	}
	status := sidecar.Start(ctx)
	if status.Enabled && status.Available && !status.Healthy {
		status = sidecar.Status(ctx)
	}
	return status
}

func (r LlamaCppBenchmarkRunner) stopSidecarAfterRunIfIdle(ctx context.Context, sidecar LlamaCppSidecar) {
	if llamacpp.KeepWarmEnabledFromEnv(r.getenv()) {
		return
	}
	stopper, ok := sidecar.(LlamaCppStopper)
	if !ok || stopper == nil {
		return
	}
	if ctx == nil {
		ctx = context.Background()
	}
	stopCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	_ = stopper.Stop(stopCtx)
}

func (r LlamaCppBenchmarkRunner) sidecar() LlamaCppSidecar {
	if r.Sidecar != nil {
		return r.Sidecar
	}
	return llamacpp.NewManagerFromEnv()
}

func (r LlamaCppBenchmarkRunner) client() llamacpp.CompletionClient {
	if r.Client != nil {
		return r.Client
	}
	return llamacpp.OpenAIClient{}
}

func (r LlamaCppBenchmarkRunner) getenv() func(string) string {
	if r.Getenv != nil {
		return r.Getenv
	}
	return func(string) string { return "" }
}

func measuredExecutionResult(spec BenchmarkSpec, completion llamacpp.CompletionResult) BenchmarkExecutionResult {
	tokens := completion.TokensGenerated
	if tokens <= 0 {
		tokens = approximateGeneratedTokens(completion.Output)
	}
	outputBytes := completion.OutputBytes
	if outputBytes <= 0 {
		outputBytes = int64(len(completion.Output))
	}
	ttft := completion.TTFTMs
	if ttft < 0 {
		ttft = 0
	}
	total := completion.TotalTimeMs
	if total < 0 {
		total = 0
	}
	if ttft > total {
		ttft = total
	}
	decodeWindowMs := total - ttft
	if decodeWindowMs <= 0 {
		decodeWindowMs = total
	}
	var decodeTPS float64
	if tokens > 0 && decodeWindowMs > 0 {
		decodeTPS = float64(tokens) / (float64(decodeWindowMs) / 1000)
	}
	var endToEndTPS float64
	if tokens > 0 && total > 0 {
		endToEndTPS = float64(tokens) / (float64(total) / 1000)
	}
	return BenchmarkExecutionResult{
		Spec:            spec,
		Backend:         spec.Backend,
		ModelID:         spec.ModelID,
		PromptHash:      spec.PromptHash,
		OutputHash:      hashBenchmarkOutput(spec, completion.Output),
		OutputBytes:     outputBytes,
		TokensGenerated: tokens,
		TTFTMs:          ttft,
		P95TTFTMs:       ttft,
		TotalTimeMs:     total,
		DecodeTPS:       roundTPS(decodeTPS),
		EndToEndTPS:     roundTPS(endToEndTPS),
		ProofStatus:     ProofStatusMeasured,
	}
}

func failedExecutionResult(spec BenchmarkSpec, code string) BenchmarkExecutionResult {
	code = cleanBenchmarkErrorCode(code)
	if code == "" {
		code = "backend_inference_failed"
	}
	return BenchmarkExecutionResult{
		Spec:        spec,
		Backend:     spec.Backend,
		ModelID:     spec.ModelID,
		PromptHash:  spec.PromptHash,
		ProofStatus: ProofStatusRejected,
		ErrorCode:   code,
	}
}

func safeSidecarFailure(status llamacpp.LlamaCppSidecarStatus) string {
	switch {
	case !status.Enabled:
		return "llamacpp_sidecar_disabled"
	case !status.Available:
		return "llamacpp_sidecar_unavailable"
	case !status.Running:
		return "llamacpp_sidecar_not_running"
	case !status.Healthy:
		return "llamacpp_sidecar_unhealthy"
	default:
		return "llamacpp_sidecar_unavailable"
	}
}

func safeBenchmarkErrorCode(err error) string {
	if err == nil {
		return ""
	}
	code := cleanBenchmarkErrorCode(err.Error())
	if code == "" {
		return "backend_inference_failed"
	}
	return code
}

func cleanBenchmarkErrorCode(value string) string {
	value = cleanBenchmarkText(value, maxBenchmarkIDLen)
	if value == "" {
		return ""
	}
	lower := strings.ToLower(value)
	for _, marker := range []string{"raw_prompt", "prompt_text", "generated_text", "output_text", "model_output", "tensor_bytes", "raw_tensor", "auth_token", "bind_token", "secret"} {
		if strings.Contains(lower, marker) {
			return "backend_inference_error_redacted"
		}
	}
	return value
}

func approximateGeneratedTokens(output []byte) int64 {
	count := int64(len(strings.Fields(string(output))))
	if count == 0 && strings.TrimSpace(string(output)) != "" {
		count = 1
	}
	return count
}

func roundTPS(value float64) float64 {
	if value <= 0 || math.IsNaN(value) || math.IsInf(value, 0) {
		return 0
	}
	return math.Round(value*1000) / 1000
}

func finiteTPS(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value >= 0
}

func (s *LocalStatus) recordSeen(jobID string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastJobID = cleanBenchmarkText(jobID, maxBenchmarkIDLen)
	s.snapshot.LastError = ""
	s.snapshot.Counters.Seen++
}

func (s *LocalStatus) recordExecuted(jobID string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastJobID = cleanBenchmarkText(jobID, maxBenchmarkIDLen)
	s.snapshot.LastError = ""
	s.snapshot.Counters.Executed++
}

func (s *LocalStatus) recordReceiptSubmitted(jobID string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastJobID = cleanBenchmarkText(jobID, maxBenchmarkIDLen)
	s.snapshot.LastError = ""
	s.snapshot.Counters.ReceiptSubmitted++
}

func (s *LocalStatus) recordReceiptFailed(jobID string, err error) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastJobID = cleanBenchmarkText(jobID, maxBenchmarkIDLen)
	s.snapshot.LastError = cleanLocalStatusError(err)
	s.snapshot.Counters.ReceiptFailed++
}

func (s *LocalStatus) recordError(jobID string, err error) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastJobID = cleanBenchmarkText(jobID, maxBenchmarkIDLen)
	s.snapshot.LastError = cleanLocalStatusError(err)
}

func cleanLocalStatusError(err error) string {
	if err == nil {
		return ""
	}
	value := cleanBenchmarkErrorCode(err.Error())
	if value == "" {
		return "backend_inference_benchmark_error"
	}
	return value
}

func ensureMeasuredMetrics(result BenchmarkExecutionResult) error {
	var errs []error
	if result.OutputHash == "" {
		errs = append(errs, fmt.Errorf("%w: measured receipt requires output_hash", ErrInvalidBenchmarkReceipt))
	}
	if result.OutputBytes <= 0 {
		errs = append(errs, fmt.Errorf("%w: measured receipt requires output_bytes", ErrInvalidBenchmarkReceipt))
	}
	if result.TokensGenerated <= 0 {
		errs = append(errs, fmt.Errorf("%w: measured receipt requires tokens_generated", ErrInvalidBenchmarkReceipt))
	}
	if result.TotalTimeMs <= 0 {
		errs = append(errs, fmt.Errorf("%w: measured receipt requires total_time_ms", ErrInvalidBenchmarkReceipt))
	}
	if !finiteTPS(result.DecodeTPS) || !finiteTPS(result.EndToEndTPS) {
		errs = append(errs, fmt.Errorf("%w: measured receipt requires finite tps metrics", ErrInvalidBenchmarkReceipt))
	}
	return errorsJoin(errs...)
}
