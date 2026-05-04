package modelbench

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"
)

const (
	ModelBenchmarkFlagEnv = "RYV_NODE_V7_MODEL_BENCH"

	maxLocalStatusIDLen    = 256
	maxLocalStatusErrorLen = 512
)

type LocalStatusCounters struct {
	Seen             uint64 `json:"seen"`
	Executed         uint64 `json:"executed"`
	ReceiptSubmitted uint64 `json:"receipt_submitted"`
	ReceiptFailed    uint64 `json:"receipt_failed"`
}

type LocalStatusSnapshot struct {
	LastSeenBenchmarkJobID    string              `json:"last_seen_benchmark_job_id,omitempty"`
	LastSeenRequestID         string              `json:"last_seen_request_id,omitempty"`
	LastSeenAt                *time.Time          `json:"last_seen_at,omitempty"`
	LastExecutedJobID         string              `json:"last_executed_job_id,omitempty"`
	LastExecutedAt            *time.Time          `json:"last_executed_at,omitempty"`
	LastReceiptSubmittedJobID string              `json:"last_receipt_submitted_job_id,omitempty"`
	LastReceiptSubmittedAt    *time.Time          `json:"last_receipt_submitted_at,omitempty"`
	LastError                 string              `json:"last_error,omitempty"`
	Counters                  LocalStatusCounters `json:"counters"`
}

type LocalStatus struct {
	mu       sync.RWMutex
	snapshot LocalStatusSnapshot
}

type ModelBenchmarkAssignmentIdentity struct {
	JobID     string
	RequestID string
}

func NewLocalStatus() *LocalStatus {
	return &LocalStatus{}
}

func IsModelBenchmarkSpecJSON(specJSON string) bool {
	var header struct {
		Task string `json:"task"`
	}
	if json.Unmarshal([]byte(strings.TrimSpace(specJSON)), &header) != nil {
		return false
	}
	return strings.TrimSpace(header.Task) == ModelBenchmarkTask
}

func ModelBenchmarkEnabledFromEnv(getenv func(string) string) bool {
	if getenv == nil {
		return false
	}
	return strings.TrimSpace(getenv(ModelBenchmarkFlagEnv)) == "1"
}

func ModelBenchmarkAssignmentIdentityFromJSON(specJSON string) (ModelBenchmarkAssignmentIdentity, bool) {
	var header struct {
		Task      string `json:"task"`
		JobID     string `json:"job_id"`
		RequestID string `json:"request_id"`
	}
	if json.Unmarshal([]byte(strings.TrimSpace(specJSON)), &header) != nil {
		return ModelBenchmarkAssignmentIdentity{}, false
	}
	if strings.TrimSpace(header.Task) != ModelBenchmarkTask {
		return ModelBenchmarkAssignmentIdentity{}, false
	}
	return ModelBenchmarkAssignmentIdentity{
		JobID:     cleanLocalStatusText(header.JobID, maxLocalStatusIDLen),
		RequestID: cleanLocalStatusText(header.RequestID, maxLocalStatusIDLen),
	}, true
}

func DecodeModelBenchmarkSpec(specJSON string) (ModelBenchmarkSpec, error) {
	raw := strings.TrimSpace(specJSON)
	if raw == "" {
		return ModelBenchmarkSpec{}, fmt.Errorf("%w: spec_json required", ErrInvalidModelBenchmarkSpec)
	}

	var spec ModelBenchmarkSpec
	if err := json.Unmarshal([]byte(raw), &spec); err != nil {
		return ModelBenchmarkSpec{}, fmt.Errorf("%w: decode spec_json: %v", ErrInvalidModelBenchmarkSpec, err)
	}
	spec = normalizeModelBenchmarkSpec(spec)
	if spec.Task != ModelBenchmarkTask {
		return ModelBenchmarkSpec{}, fmt.Errorf("%w: task must be %q", ErrInvalidModelBenchmarkSpec, ModelBenchmarkTask)
	}
	if err := ValidateModelBenchmarkSpec(spec); err != nil {
		return ModelBenchmarkSpec{}, err
	}
	return spec, nil
}

func ExecuteModelBenchmarkAssignment(ctx context.Context, specJSON string, runner ModelBenchmarkRunner, env func(string) string) (ModelBenchmarkReceipt, bool, error) {
	if !IsModelBenchmarkSpecJSON(specJSON) {
		return ModelBenchmarkReceipt{}, false, nil
	}
	if !ModelBenchmarkEnabledFromEnv(env) {
		return ModelBenchmarkReceipt{}, false, nil
	}

	spec, err := DecodeModelBenchmarkSpec(specJSON)
	if err != nil {
		return ModelBenchmarkReceipt{}, true, err
	}
	receipt, err := ExecuteModelBenchmarkSpec(ctx, spec, runner)
	return receipt, true, err
}

func ExecuteModelBenchmarkSpec(ctx context.Context, spec ModelBenchmarkSpec, runner ModelBenchmarkRunner) (ModelBenchmarkReceipt, error) {
	spec = normalizeModelBenchmarkSpec(spec)
	if err := ValidateModelBenchmarkSpec(spec); err != nil {
		return ModelBenchmarkReceipt{}, err
	}
	if runner == nil {
		return ModelBenchmarkReceipt{}, ModelBenchmarkError{Code: "modelbench_runner_missing", Message: "model benchmark runner is not configured"}
	}

	result, runErr := runner.RunModelBenchmark(ctx, spec)
	receipt, receiptErr := BuildModelBenchmarkReceipt(result)
	if receiptErr != nil {
		if runErr != nil {
			return ModelBenchmarkReceipt{}, fmt.Errorf("%w: %v", runErr, receiptErr)
		}
		return ModelBenchmarkReceipt{}, receiptErr
	}
	return receipt, runErr
}

func (s *LocalStatus) RecordSeen(jobID, requestID string) {
	s.recordSeenAt(jobID, requestID, time.Now())
}

func (s *LocalStatus) RecordExecuted(jobID string) {
	s.recordExecutedAt(jobID, time.Now())
}

func (s *LocalStatus) RecordReceiptSubmitted(jobID string) {
	s.recordReceiptSubmittedAt(jobID, time.Now())
}

func (s *LocalStatus) RecordReceiptFailed(jobID string, err error) {
	s.recordReceiptFailedAt(jobID, err, time.Now())
}

func (s *LocalStatus) RecordError(err error) {
	s.recordError(err)
}

func (s *LocalStatus) Snapshot() LocalStatusSnapshot {
	if s == nil {
		return LocalStatusSnapshot{}
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := s.snapshot
	out.LastSeenAt = cloneStatusTime(s.snapshot.LastSeenAt)
	out.LastExecutedAt = cloneStatusTime(s.snapshot.LastExecutedAt)
	out.LastReceiptSubmittedAt = cloneStatusTime(s.snapshot.LastReceiptSubmittedAt)
	return out
}

func (s *LocalStatus) recordSeenAt(jobID, requestID string, at time.Time) {
	if s == nil {
		return
	}
	at = normalizeLocalStatusTime(at)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastSeenBenchmarkJobID = cleanLocalStatusText(jobID, maxLocalStatusIDLen)
	s.snapshot.LastSeenRequestID = cleanLocalStatusText(requestID, maxLocalStatusIDLen)
	s.snapshot.LastSeenAt = &at
	s.snapshot.LastError = ""
	s.snapshot.Counters.Seen++
}

func (s *LocalStatus) recordExecutedAt(jobID string, at time.Time) {
	if s == nil {
		return
	}
	at = normalizeLocalStatusTime(at)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastExecutedJobID = cleanLocalStatusText(jobID, maxLocalStatusIDLen)
	s.snapshot.LastExecutedAt = &at
	s.snapshot.Counters.Executed++
}

func (s *LocalStatus) recordReceiptSubmittedAt(jobID string, at time.Time) {
	if s == nil {
		return
	}
	at = normalizeLocalStatusTime(at)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastReceiptSubmittedJobID = cleanLocalStatusText(jobID, maxLocalStatusIDLen)
	s.snapshot.LastReceiptSubmittedAt = &at
	s.snapshot.LastError = ""
	s.snapshot.Counters.ReceiptSubmitted++
}

func (s *LocalStatus) recordReceiptFailedAt(jobID string, err error, at time.Time) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastError = cleanLocalStatusError(err)
	s.snapshot.Counters.ReceiptFailed++
}

func (s *LocalStatus) recordError(err error) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastError = cleanLocalStatusError(err)
}

func normalizeLocalStatusTime(at time.Time) time.Time {
	if at.IsZero() {
		return time.Now()
	}
	return at
}

func cloneStatusTime(value *time.Time) *time.Time {
	if value == nil {
		return nil
	}
	cp := *value
	return &cp
}

func cleanLocalStatusError(err error) string {
	if err == nil {
		return ""
	}
	return cleanLocalStatusText(err.Error(), maxLocalStatusErrorLen)
}

func cleanLocalStatusText(value string, maxLen int) string {
	value = strings.TrimSpace(value)
	value = strings.ReplaceAll(value, "\r", " ")
	value = strings.ReplaceAll(value, "\n", " ")
	value = strings.Join(strings.Fields(value), " ")
	if maxLen > 0 && len(value) > maxLen {
		return value[:maxLen]
	}
	return value
}
