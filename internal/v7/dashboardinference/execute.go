package dashboardinference

import (
	"context"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/Ryvion/node-agent/internal/v7/llamacpp"
)

const (
	ProofStatusMeasured = "dashboard_inference_measured"
	ProofStatusRejected = "dashboard_inference_rejected"
)

const internalDashboardPrompt = "Answer in one short sentence: confirm the local Ryvion dashboard inference path is ready."

var ErrDisabled = errors.New("dashboardinference: feature disabled")

type Runner interface {
	RunDashboardInference(context.Context, Spec) (ExecutionResult, error)
}

type ExecuteOptions struct {
	Getenv func(string) string
	Runner Runner
}

type LlamaCppSidecar interface {
	Start(context.Context) llamacpp.LlamaCppSidecarStatus
	Status(context.Context) llamacpp.LlamaCppSidecarStatus
}

type KeepWarmChecker interface {
	CheckOnce(context.Context) llamacpp.LlamaCppSidecarStatus
}

type LlamaCppRunner struct {
	Sidecar  LlamaCppSidecar
	KeepWarm KeepWarmChecker
	Client   llamacpp.CompletionClient
	Getenv   func(string) string
}

type ExecutionResult struct {
	Spec            Spec
	Backend         string
	ModelID         string
	OutputHash      string
	OutputBytes     int64
	TokensGenerated int64
	TTFTMs          int64
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
	Rejected         uint64 `json:"rejected"`
}

type LocalStatusSnapshot struct {
	LastRunID string              `json:"last_run_id"`
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

func ExecuteAssignment(ctx context.Context, specJSON string, opts ExecuteOptions) (Receipt, bool, error) {
	identity, isDashboardInference := AssignmentIdentityFromJSON(specJSON)
	if !isDashboardInference {
		return Receipt{}, false, nil
	}
	if !EnabledFromEnv(opts.getenv()) {
		err := codedError{code: "dashboard_inference_disabled", err: ErrDisabled}
		return BuildRejectionReceiptFromIdentity(identity, err), true, err
	}
	spec, err := DecodeSpec(specJSON)
	if err != nil {
		return BuildRejectionReceiptFromIdentity(identity, err), true, err
	}
	receipt, err := ExecuteSpec(ctx, spec, opts)
	if err != nil && strings.TrimSpace(receipt.ResultHashHex) == "" {
		receipt = BuildRejectionReceipt(spec, err)
	}
	return receipt, true, err
}

func ExecuteSpec(ctx context.Context, spec Spec, opts ExecuteOptions) (Receipt, error) {
	spec = normalizeSpec(spec)
	if err := ValidateSpec(spec); err != nil {
		return Receipt{}, err
	}
	if !EnabledFromEnv(opts.getenv()) {
		return Receipt{}, codedError{code: "dashboard_inference_disabled", err: ErrDisabled}
	}
	if ctx == nil {
		ctx = context.Background()
	}
	runner := opts.Runner
	if runner == nil {
		runner = LlamaCppRunner{
			Sidecar: llamacpp.NewManagerFromEnv(),
			Client:  llamacpp.OpenAIClient{},
			Getenv:  opts.getenv(),
		}
	}
	result, err := runner.RunDashboardInference(ctx, spec)
	if err != nil {
		return Receipt{}, err
	}
	return BuildReceipt(result)
}

func (r LlamaCppRunner) RunDashboardInference(ctx context.Context, spec Spec) (ExecutionResult, error) {
	spec = normalizeSpec(spec)
	if err := ValidateSpec(spec); err != nil {
		return ExecutionResult{}, err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	runCtx, cancel := context.WithTimeout(ctx, time.Duration(defaultTimeoutMs)*time.Millisecond)
	defer cancel()
	if err := runCtx.Err(); err != nil {
		return failedExecutionResult(spec, "dashboard_inference_context_unavailable"), nil
	}

	status := r.ensureSidecar(runCtx)
	if !status.Enabled || !status.Available || !status.Running || !status.Healthy {
		return failedExecutionResult(spec, safeSidecarFailure(status)), nil
	}
	if !sidecarModelMatches(status, spec.ModelID) {
		return failedExecutionResult(spec, "llamacpp_model_mismatch"), nil
	}

	req := llamacpp.CompletionRequest{
		BaseURL:     status.BaseURL,
		ModelID:     spec.ModelID,
		Prompt:      internalDashboardPrompt,
		MaxTokens:   spec.MaxTokens,
		Temperature: 0,
		Stream:      spec.Stream && status.SupportsStreaming,
	}
	completion, err := completeWithFallback(runCtx, r.client(), req)
	if err != nil {
		return failedExecutionResult(spec, safeErrorCode(err)), nil
	}
	return measuredExecutionResult(spec, completion), nil
}

func (r LlamaCppRunner) ensureSidecar(ctx context.Context) llamacpp.LlamaCppSidecarStatus {
	sidecar := r.sidecar()
	if llamacpp.KeepWarmEnabledFromEnv(r.getenv()) {
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

func (r LlamaCppRunner) sidecar() LlamaCppSidecar {
	if r.Sidecar != nil {
		return r.Sidecar
	}
	return llamacpp.NewManagerFromEnv()
}

func (r LlamaCppRunner) client() llamacpp.CompletionClient {
	if r.Client != nil {
		return r.Client
	}
	return llamacpp.OpenAIClient{}
}

func (r LlamaCppRunner) getenv() func(string) string {
	if r.Getenv != nil {
		return r.Getenv
	}
	return func(string) string { return "" }
}

func (opts ExecuteOptions) getenv() func(string) string {
	if opts.Getenv != nil {
		return opts.Getenv
	}
	return func(string) string { return "" }
}

func completeWithFallback(ctx context.Context, client llamacpp.CompletionClient, req llamacpp.CompletionRequest) (llamacpp.CompletionResult, error) {
	if client == nil {
		client = llamacpp.OpenAIClient{}
	}
	result, err := client.Complete(ctx, req)
	if err == nil {
		return result, nil
	}
	if req.Stream && llamacpp.IsStreamUnavailable(err) {
		req.Stream = false
		result, err = client.Complete(ctx, req)
		if err == nil {
			return result, nil
		}
	}
	return llamacpp.CompletionResult{}, err
}

func sidecarModelMatches(status llamacpp.LlamaCppSidecarStatus, requested string) bool {
	requested = normalizeModelComparable(requested)
	if requested == "" {
		return false
	}
	for _, candidate := range []string{status.ModelFilename, filepath.Base(status.ModelPath), status.ModelPath} {
		if normalizeModelComparable(candidate) == requested {
			return true
		}
	}
	return false
}

func normalizeModelComparable(value string) string {
	value = strings.TrimSpace(value)
	if value == "" || value == "." || value == string(filepath.Separator) {
		return ""
	}
	return strings.ToLower(filepath.Base(value))
}

func measuredExecutionResult(spec Spec, completion llamacpp.CompletionResult) ExecutionResult {
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
	return ExecutionResult{
		Spec:            spec,
		Backend:         spec.Backend,
		ModelID:         spec.ModelID,
		OutputHash:      HashOutput(spec.JobID, completion.Output),
		OutputBytes:     outputBytes,
		TokensGenerated: tokens,
		TTFTMs:          ttft,
		TotalTimeMs:     total,
		DecodeTPS:       roundTPS(decodeTPS),
		EndToEndTPS:     roundTPS(endToEndTPS),
		ProofStatus:     ProofStatusMeasured,
	}
}

func failedExecutionResult(spec Spec, code string) ExecutionResult {
	code = cleanErrorCode(code)
	if code == "" {
		code = "dashboard_inference_failed"
	}
	return ExecutionResult{
		Spec:        spec,
		Backend:     spec.Backend,
		ModelID:     spec.ModelID,
		OutputHash:  HashOutput(spec.JobID, nil),
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

func safeErrorCode(err error) string {
	if err == nil {
		return ""
	}
	code := ErrorCode(err)
	if code == "" {
		return "dashboard_inference_failed"
	}
	return code
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

func (s *LocalStatus) RecordSeen(runID, jobID string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastRunID = cleanText(runID, maxIDLen)
	s.snapshot.LastJobID = cleanText(jobID, maxIDLen)
	s.snapshot.LastError = ""
	s.snapshot.Counters.Seen++
}

func (s *LocalStatus) RecordExecuted(runID, jobID string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastRunID = cleanText(runID, maxIDLen)
	s.snapshot.LastJobID = cleanText(jobID, maxIDLen)
	s.snapshot.LastError = ""
	s.snapshot.Counters.Executed++
}

func (s *LocalStatus) RecordRejected(runID, jobID string, err error) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastRunID = cleanText(runID, maxIDLen)
	s.snapshot.LastJobID = cleanText(jobID, maxIDLen)
	s.snapshot.LastError = cleanLocalStatusError(err)
	s.snapshot.Counters.Rejected++
}

func (s *LocalStatus) RecordReceiptSubmitted(runID, jobID string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastRunID = cleanText(runID, maxIDLen)
	s.snapshot.LastJobID = cleanText(jobID, maxIDLen)
	s.snapshot.LastError = ""
	s.snapshot.Counters.ReceiptSubmitted++
}

func (s *LocalStatus) RecordReceiptFailed(runID, jobID string, err error) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastRunID = cleanText(runID, maxIDLen)
	s.snapshot.LastJobID = cleanText(jobID, maxIDLen)
	s.snapshot.LastError = cleanLocalStatusError(err)
	s.snapshot.Counters.ReceiptFailed++
}

func (s *LocalStatus) RecordError(runID, jobID string, err error) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastRunID = cleanText(runID, maxIDLen)
	s.snapshot.LastJobID = cleanText(jobID, maxIDLen)
	s.snapshot.LastError = cleanLocalStatusError(err)
}

func (s *LocalStatus) Snapshot() LocalStatusSnapshot {
	if s == nil {
		return LocalStatusSnapshot{}
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.snapshot
}

func cleanLocalStatusError(err error) string {
	if err == nil {
		return ""
	}
	value := ErrorCode(err)
	if value == "" {
		value = "dashboard_inference_error"
	}
	return cleanText(value, maxStatusErrLen)
}

type codedError struct {
	code string
	err  error
}

func (e codedError) Error() string {
	if e.err == nil {
		return e.code
	}
	return fmt.Sprintf("%s: %v", e.code, e.err)
}

func (e codedError) Unwrap() error {
	return e.err
}

func ErrorCode(err error) string {
	if err == nil {
		return ""
	}
	var coded interface{ Code() string }
	if errors.As(err, &coded) {
		return cleanErrorCode(coded.Code())
	}
	var local codedError
	if errors.As(err, &local) {
		return cleanErrorCode(local.code)
	}
	return cleanErrorCode(err.Error())
}

func (e codedError) Code() string {
	return e.code
}

func cleanErrorCode(value string) string {
	value = cleanText(value, maxStatusErrLen)
	if value == "" {
		return ""
	}
	lower := strings.ToLower(value)
	for _, marker := range forbiddenTextMarkers() {
		if strings.Contains(lower, strings.ToLower(marker)) {
			return "dashboard_inference_error_redacted"
		}
	}
	return value
}
