package llamacpp

import (
	"context"
	"fmt"
	"sync"
	"time"
)

type BackendBenchmarkRunner interface {
	Run(context.Context, BenchmarkConfig) BenchmarkStatusSnapshot
}

type ExecuteBackendBenchmarkOptions struct {
	Getenv  func(string) string
	Runner  BackendBenchmarkRunner
	Profile BackendBenchmarkProfile
}

type BackendBenchmarkProfile struct {
	NodeID              string
	Acceleration        string
	Warm                bool
	ContextLengthTokens int
	StreamingSupported  bool
}

type BackendBenchmarkLocalStatusCounters struct {
	Seen             uint64 `json:"seen"`
	Executed         uint64 `json:"executed"`
	ReceiptSubmitted uint64 `json:"receipt_submitted"`
	ReceiptFailed    uint64 `json:"receipt_failed"`
}

type BackendBenchmarkLocalStatusSnapshot struct {
	LastJobID string                              `json:"last_job_id"`
	LastError string                              `json:"last_error"`
	Counters  BackendBenchmarkLocalStatusCounters `json:"counters"`
}

type BackendBenchmarkLocalStatus struct {
	mu       sync.RWMutex
	snapshot BackendBenchmarkLocalStatusSnapshot
}

func NewBackendBenchmarkLocalStatus() *BackendBenchmarkLocalStatus {
	return &BackendBenchmarkLocalStatus{}
}

func ExecuteBackendBenchmarkAssignment(ctx context.Context, specJSON string, opts ExecuteBackendBenchmarkOptions) (BackendBenchmarkReceipt, bool, error) {
	if !IsBackendBenchmarkSpecJSON(specJSON) {
		return BackendBenchmarkReceipt{}, false, nil
	}
	if !BackendBenchmarkEnabledFromEnv(opts.Getenv) {
		return BackendBenchmarkReceipt{}, false, nil
	}
	spec, err := DecodeBackendBenchmarkSpec(specJSON)
	if err != nil {
		return BackendBenchmarkReceipt{}, true, err
	}
	receipt, err := ExecuteBackendBenchmarkSpec(ctx, spec, opts)
	return receipt, true, err
}

func ExecuteBackendBenchmarkSpec(ctx context.Context, spec BackendBenchmarkSpec, opts ExecuteBackendBenchmarkOptions) (BackendBenchmarkReceipt, error) {
	spec = normalizeBackendBenchmarkSpec(spec)
	if err := ValidateBackendBenchmarkSpec(spec); err != nil {
		return BackendBenchmarkReceipt{}, err
	}
	runner := opts.Runner
	if runner == nil {
		runner = BenchmarkRunner{
			Sidecar: NewManagerFromEnv(),
			Client:  OpenAIClient{},
		}
	}
	if ctx == nil {
		ctx = context.Background()
	}
	snapshot := runner.Run(ctx, BenchmarkConfig{
		NodeID:              opts.Profile.NodeID,
		ModelID:             spec.ModelID,
		MaxTokens:           spec.MaxTokens,
		Temperature:         0,
		TimeoutMs:           spec.TimeoutMs,
		Streaming:           true,
		MeasuredRuns:        spec.MeasuredRuns,
		WarmupRuns:          spec.WarmupRuns,
		Acceleration:        opts.Profile.Acceleration,
		Warm:                opts.Profile.Warm,
		ContextLengthTokens: opts.Profile.ContextLengthTokens,
		StreamingSupported:  opts.Profile.StreamingSupported,
	})
	receipt, err := BuildBackendBenchmarkReceipt(spec, snapshot)
	if err != nil {
		return BackendBenchmarkReceipt{}, err
	}
	return receipt, nil
}

func (s *BackendBenchmarkLocalStatus) RecordSeen(jobID string) {
	s.recordSeenAt(jobID, time.Now())
}

func (s *BackendBenchmarkLocalStatus) RecordExecuted(jobID string) {
	s.recordExecutedAt(jobID, time.Now())
}

func (s *BackendBenchmarkLocalStatus) RecordReceiptSubmitted(jobID string) {
	s.recordReceiptSubmittedAt(jobID, time.Now())
}

func (s *BackendBenchmarkLocalStatus) RecordReceiptFailed(jobID string, err error) {
	s.recordReceiptFailedAt(jobID, err, time.Now())
}

func (s *BackendBenchmarkLocalStatus) RecordError(jobID string, err error) {
	s.recordError(jobID, err)
}

func (s *BackendBenchmarkLocalStatus) Snapshot() BackendBenchmarkLocalStatusSnapshot {
	if s == nil {
		return BackendBenchmarkLocalStatusSnapshot{}
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.snapshot
}

func (s *BackendBenchmarkLocalStatus) recordSeenAt(jobID string, _ time.Time) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastJobID = cleanStatusText(jobID, maxBackendBenchmarkIDLen)
	s.snapshot.LastError = ""
	s.snapshot.Counters.Seen++
}

func (s *BackendBenchmarkLocalStatus) recordExecutedAt(jobID string, _ time.Time) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastJobID = cleanStatusText(jobID, maxBackendBenchmarkIDLen)
	s.snapshot.LastError = ""
	s.snapshot.Counters.Executed++
}

func (s *BackendBenchmarkLocalStatus) recordReceiptSubmittedAt(jobID string, _ time.Time) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastJobID = cleanStatusText(jobID, maxBackendBenchmarkIDLen)
	s.snapshot.LastError = ""
	s.snapshot.Counters.ReceiptSubmitted++
}

func (s *BackendBenchmarkLocalStatus) recordReceiptFailedAt(jobID string, err error, _ time.Time) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastJobID = cleanStatusText(jobID, maxBackendBenchmarkIDLen)
	s.snapshot.LastError = cleanBackendBenchmarkStatusError(err)
	s.snapshot.Counters.ReceiptFailed++
}

func (s *BackendBenchmarkLocalStatus) recordError(jobID string, err error) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastJobID = cleanStatusText(jobID, maxBackendBenchmarkIDLen)
	s.snapshot.LastError = cleanBackendBenchmarkStatusError(err)
}

func cleanBackendBenchmarkStatusError(err error) string {
	if err == nil {
		return ""
	}
	value := cleanStatusText(err.Error(), maxStatusReasonLen)
	if value == "" {
		return "llamacpp_backend_benchmark_error"
	}
	return value
}

func BackendBenchmarkError(code string) error {
	code = cleanStatusText(code, maxStatusReasonLen)
	if code == "" {
		code = "llamacpp_backend_benchmark_failed"
	}
	return fmt.Errorf("%w: %s", ErrInvalidBackendBenchmarkReceipt, code)
}
