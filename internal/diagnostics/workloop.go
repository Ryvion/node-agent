package diagnostics

import (
	"encoding/json"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"
)

const (
	maxWorkLoopIDLen    = 256
	maxWorkLoopKindLen  = 128
	maxWorkLoopTaskLen  = 128
	maxWorkLoopErrorLen = 512
)

type WorkLoopSnapshot struct {
	LastPollStartedAt            string `json:"last_poll_started_at"`
	LastPollCompletedAt          string `json:"last_poll_completed_at"`
	LastPollDurationMs           int64  `json:"last_poll_duration_ms"`
	LastPollError                string `json:"last_poll_error"`
	LastWorkSeenAt               string `json:"last_work_seen_at"`
	LastWorkJobID                string `json:"last_work_job_id"`
	LastWorkKind                 string `json:"last_work_kind"`
	LastWorkSpecTask             string `json:"last_work_spec_task"`
	LastWorkDecodeMs             int64  `json:"last_work_decode_ms"`
	LastExecutionStartedAt       string `json:"last_execution_started_at"`
	LastExecutionCompletedAt     string `json:"last_execution_completed_at"`
	LastExecutionDurationMs      int64  `json:"last_execution_duration_ms"`
	LastReceiptBuildMs           int64  `json:"last_receipt_build_ms"`
	LastReceiptSubmitStartedAt   string `json:"last_receipt_submit_started_at"`
	LastReceiptSubmitCompletedAt string `json:"last_receipt_submit_completed_at"`
	LastReceiptSubmitDurationMs  int64  `json:"last_receipt_submit_duration_ms"`
	LastReceiptSubmitError       string `json:"last_receipt_submit_error"`
	LastReceiptAttempts          int    `json:"last_receipt_attempts"`
	PollCount                    uint64 `json:"poll_count"`
	WorkSeenCount                uint64 `json:"work_seen_count"`
	WorkCompletedCount           uint64 `json:"work_completed_count"`
	ReceiptSubmittedCount        uint64 `json:"receipt_submitted_count"`
	ReceiptFailedCount           uint64 `json:"receipt_failed_count"`
}

type WorkLoopDiagnostics struct {
	mu sync.RWMutex

	snapshot WorkLoopSnapshot

	lastPollStartedAt          time.Time
	lastExecutionStartedAt     time.Time
	lastReceiptSubmitStartedAt time.Time
	lastReceiptSubmitJobID     string
}

func NewWorkLoopDiagnostics() *WorkLoopDiagnostics {
	return &WorkLoopDiagnostics{}
}

func (d *WorkLoopDiagnostics) RecordPollStart() {
	if d == nil {
		return
	}
	now := time.Now().UTC()
	d.mu.Lock()
	defer d.mu.Unlock()
	d.lastPollStartedAt = now
	d.snapshot.LastPollStartedAt = formatWorkLoopTime(now)
	d.snapshot.LastPollError = ""
	d.snapshot.PollCount++
}

func (d *WorkLoopDiagnostics) RecordPollEnd(err error) {
	if d == nil {
		return
	}
	now := time.Now().UTC()
	d.mu.Lock()
	defer d.mu.Unlock()
	d.snapshot.LastPollCompletedAt = formatWorkLoopTime(now)
	if d.lastPollStartedAt.IsZero() {
		d.snapshot.LastPollDurationMs = 0
	} else {
		d.snapshot.LastPollDurationMs = durationMilliseconds(now.Sub(d.lastPollStartedAt))
	}
	d.snapshot.LastPollError = sanitizeWorkLoopError(err)
}

func (d *WorkLoopDiagnostics) RecordWorkSeen(jobID, kind, specTask string) {
	if d == nil {
		return
	}
	now := time.Now().UTC()
	d.mu.Lock()
	defer d.mu.Unlock()
	d.snapshot.LastWorkSeenAt = formatWorkLoopTime(now)
	d.snapshot.LastWorkJobID = cleanWorkLoopText(jobID, maxWorkLoopIDLen)
	d.snapshot.LastWorkKind = cleanWorkLoopText(kind, maxWorkLoopKindLen)
	d.snapshot.LastWorkSpecTask = cleanWorkLoopText(specTask, maxWorkLoopTaskLen)
	d.snapshot.WorkSeenCount++
}

func (d *WorkLoopDiagnostics) RecordWorkDecode(duration time.Duration) {
	if d == nil {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.snapshot.LastWorkDecodeMs = durationMilliseconds(duration)
}

func (d *WorkLoopDiagnostics) RecordExecutionStart(jobID string) {
	if d == nil {
		return
	}
	now := time.Now().UTC()
	d.mu.Lock()
	defer d.mu.Unlock()
	d.lastExecutionStartedAt = now
	d.snapshot.LastExecutionStartedAt = formatWorkLoopTime(now)
	if cleaned := cleanWorkLoopText(jobID, maxWorkLoopIDLen); cleaned != "" {
		d.snapshot.LastWorkJobID = cleaned
	}
}

func (d *WorkLoopDiagnostics) RecordExecutionEnd(duration time.Duration, err error) {
	if d == nil {
		return
	}
	_ = err
	now := time.Now().UTC()
	d.mu.Lock()
	defer d.mu.Unlock()
	if duration <= 0 && !d.lastExecutionStartedAt.IsZero() {
		duration = now.Sub(d.lastExecutionStartedAt)
	}
	d.snapshot.LastExecutionCompletedAt = formatWorkLoopTime(now)
	d.snapshot.LastExecutionDurationMs = durationMilliseconds(duration)
	d.snapshot.WorkCompletedCount++
}

func (d *WorkLoopDiagnostics) RecordReceiptBuild(duration time.Duration) {
	if d == nil {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.snapshot.LastReceiptBuildMs = durationMilliseconds(duration)
}

func (d *WorkLoopDiagnostics) RecordReceiptSubmitStart(jobID string, attempt int) {
	if d == nil {
		return
	}
	if attempt < 1 {
		attempt = 1
	}
	now := time.Now().UTC()
	cleanJobID := cleanWorkLoopText(jobID, maxWorkLoopIDLen)

	d.mu.Lock()
	defer d.mu.Unlock()
	if attempt == 1 || d.lastReceiptSubmitStartedAt.IsZero() || cleanJobID != d.lastReceiptSubmitJobID {
		d.lastReceiptSubmitStartedAt = now
		d.lastReceiptSubmitJobID = cleanJobID
		d.snapshot.LastReceiptSubmitStartedAt = formatWorkLoopTime(now)
		d.snapshot.LastReceiptSubmitError = ""
	}
	d.snapshot.LastReceiptAttempts = attempt
}

func (d *WorkLoopDiagnostics) RecordReceiptSubmitEnd(duration time.Duration, err error) {
	if d == nil {
		return
	}
	now := time.Now().UTC()
	d.mu.Lock()
	defer d.mu.Unlock()
	if duration <= 0 && !d.lastReceiptSubmitStartedAt.IsZero() {
		duration = now.Sub(d.lastReceiptSubmitStartedAt)
	}
	d.snapshot.LastReceiptSubmitCompletedAt = formatWorkLoopTime(now)
	d.snapshot.LastReceiptSubmitDurationMs = durationMilliseconds(duration)
	d.snapshot.LastReceiptSubmitError = sanitizeWorkLoopError(err)
	if err != nil {
		d.snapshot.ReceiptFailedCount++
		return
	}
	d.snapshot.ReceiptSubmittedCount++
}

func (d *WorkLoopDiagnostics) Snapshot() WorkLoopSnapshot {
	if d == nil {
		return WorkLoopSnapshot{}
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.snapshot
}

func WorkSpecTaskFromJSON(specJSON string) string {
	raw := strings.TrimSpace(specJSON)
	if raw == "" {
		return ""
	}
	var header struct {
		Task string `json:"task"`
	}
	if err := json.Unmarshal([]byte(raw), &header); err != nil {
		return ""
	}
	return cleanWorkLoopText(header.Task, maxWorkLoopTaskLen)
}

func sanitizeWorkLoopError(err error) string {
	if err == nil {
		return ""
	}
	return cleanWorkLoopText(err.Error(), maxWorkLoopErrorLen)
}

func formatWorkLoopTime(t time.Time) string {
	if t.IsZero() {
		return ""
	}
	return t.UTC().Format(time.RFC3339Nano)
}

func durationMilliseconds(duration time.Duration) int64 {
	if duration <= 0 {
		return 0
	}
	return duration.Milliseconds()
}

func cleanWorkLoopText(value string, maxLen int) string {
	value = strings.TrimSpace(collapseWorkLoopWhitespace(value))
	if value == "" {
		return ""
	}
	if maxLen <= 0 {
		return value
	}
	if utf8.RuneCountInString(value) <= maxLen {
		return value
	}
	runes := []rune(value)
	return string(runes[:maxLen])
}

func collapseWorkLoopWhitespace(value string) string {
	var b strings.Builder
	b.Grow(len(value))
	previousSpace := false
	for _, r := range value {
		if unicode.IsControl(r) || unicode.IsSpace(r) {
			if !previousSpace {
				b.WriteByte(' ')
				previousSpace = true
			}
			continue
		}
		b.WriteRune(r)
		previousSpace = false
	}
	return b.String()
}
