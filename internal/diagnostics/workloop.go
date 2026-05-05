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
	LastPollStartedAt              string `json:"last_poll_started_at"`
	LastPollCompletedAt            string `json:"last_poll_completed_at"`
	LastPollDurationMs             int64  `json:"last_poll_duration_ms"`
	LastPollCycleDurationMs        int64  `json:"last_poll_cycle_duration_ms"`
	LastPollError                  string `json:"last_poll_error"`
	LastWorkSeenAt                 string `json:"last_work_seen_at"`
	LastWorkJobID                  string `json:"last_work_job_id"`
	LastWorkKind                   string `json:"last_work_kind"`
	LastWorkSpecTask               string `json:"last_work_spec_task"`
	LastWorkDecodeMs               int64  `json:"last_work_decode_ms"`
	LastExecutionStartedAt         string `json:"last_execution_started_at"`
	LastExecutionCompletedAt       string `json:"last_execution_completed_at"`
	LastExecutionDurationMs        int64  `json:"last_execution_duration_ms"`
	LastExecutionDurationUs        int64  `json:"last_execution_duration_us"`
	LastReceiptBuildMs             int64  `json:"last_receipt_build_ms"`
	LastReceiptMetadataBuildMs     int64  `json:"last_receipt_metadata_build_ms"`
	LastReceiptHashMs              int64  `json:"last_receipt_hash_ms"`
	LastReceiptJSONMeasureMs       int64  `json:"last_receipt_json_measure_ms"`
	LastReceiptEnvelopeBuildMs     int64  `json:"last_receipt_envelope_build_ms"`
	LastReceiptTotalBuildMs        int64  `json:"last_receipt_total_build_ms"`
	LastReceiptMetadataBuildUs     int64  `json:"last_receipt_metadata_build_us"`
	LastReceiptMetadataStructUs    int64  `json:"last_receipt_metadata_struct_us"`
	LastReceiptWeightedValueCopyUs int64  `json:"last_receipt_weighted_value_copy_us,omitempty"`
	LastReceiptMetadataDefaultsUs  int64  `json:"last_receipt_metadata_defaults_us"`
	LastReceiptMetadataValidateUs  int64  `json:"last_receipt_metadata_validate_us"`
	LastReceiptMetadataGapUs       int64  `json:"last_receipt_metadata_gap_us"`
	LastReceiptMetadataTotalUs     int64  `json:"last_receipt_metadata_total_us"`
	LastReceiptHashUs              int64  `json:"last_receipt_hash_us"`
	LastReceiptJSONMeasureUs       int64  `json:"last_receipt_json_measure_us"`
	LastReceiptEnvelopeBuildUs     int64  `json:"last_receipt_envelope_build_us"`
	LastReceiptTotalBuildUs        int64  `json:"last_receipt_total_build_us"`
	LastReceiptSubmitStartedAt     string `json:"last_receipt_submit_started_at"`
	LastReceiptSubmitCompletedAt   string `json:"last_receipt_submit_completed_at"`
	LastReceiptSubmitDurationMs    int64  `json:"last_receipt_submit_duration_ms"`
	LastReceiptSubmitDurationUs    int64  `json:"last_receipt_submit_duration_us"`
	LastReceiptSubmitError         string `json:"last_receipt_submit_error"`
	LastReceiptAttempts            int    `json:"last_receipt_attempts"`
	PollCount                      uint64 `json:"poll_count"`
	WorkSeenCount                  uint64 `json:"work_seen_count"`
	WorkCompletedCount             uint64 `json:"work_completed_count"`
	ReceiptSubmittedCount          uint64 `json:"receipt_submitted_count"`
	ReceiptFailedCount             uint64 `json:"receipt_failed_count"`
}

type WorkLoopDiagnostics struct {
	mu sync.RWMutex

	snapshot WorkLoopSnapshot

	lastPollStartedAt          time.Time
	lastExecutionStartedAt     time.Time
	lastReceiptSubmitStartedAt time.Time
	lastReceiptSubmitJobID     string
}

type ReceiptBuildTimings struct {
	MetadataBuildMs     int64
	HashMs              int64
	JSONMeasureMs       int64
	EnvelopeBuildMs     int64
	TotalBuildMs        int64
	MetadataBuildUs     int64
	MetadataStructUs    int64
	WeightedValueCopyUs int64
	MetadataDefaultsUs  int64
	MetadataValidateUs  int64
	MetadataGapUs       int64
	MetadataTotalUs     int64
	HashUs              int64
	JSONMeasureUs       int64
	EnvelopeBuildUs     int64
	TotalBuildUs        int64
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
	d.snapshot.LastPollCompletedAt = formatWorkLoopTime(now)
	d.snapshot.LastPollDurationMs = 0
	d.snapshot.LastPollCycleDurationMs = 0
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
	d.snapshot.LastPollCycleDurationMs = d.snapshot.LastPollDurationMs
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
	d.snapshot.LastExecutionDurationUs = durationMicroseconds(duration)
	d.snapshot.WorkCompletedCount++
}

func (d *WorkLoopDiagnostics) RecordReceiptBuild(duration time.Duration) {
	if d == nil {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	timings := ReceiptBuildTimingsFromDurations(0, 0, 0, duration, duration)
	d.applyReceiptBuildTimingsLocked(timings)
}

func (d *WorkLoopDiagnostics) RecordReceiptBuildTimings(timings ReceiptBuildTimings) {
	if d == nil {
		return
	}
	timings = normalizeReceiptBuildTimings(timings)
	d.mu.Lock()
	defer d.mu.Unlock()
	d.applyReceiptBuildTimingsLocked(timings)
}

func (d *WorkLoopDiagnostics) applyReceiptBuildTimingsLocked(timings ReceiptBuildTimings) {
	d.snapshot.LastReceiptMetadataBuildMs = timings.MetadataBuildMs
	d.snapshot.LastReceiptHashMs = timings.HashMs
	d.snapshot.LastReceiptJSONMeasureMs = timings.JSONMeasureMs
	d.snapshot.LastReceiptEnvelopeBuildMs = timings.EnvelopeBuildMs
	d.snapshot.LastReceiptTotalBuildMs = timings.TotalBuildMs
	d.snapshot.LastReceiptBuildMs = timings.TotalBuildMs
	d.snapshot.LastReceiptMetadataBuildUs = timings.MetadataBuildUs
	d.snapshot.LastReceiptMetadataStructUs = timings.MetadataStructUs
	d.snapshot.LastReceiptWeightedValueCopyUs = timings.WeightedValueCopyUs
	d.snapshot.LastReceiptMetadataDefaultsUs = timings.MetadataDefaultsUs
	d.snapshot.LastReceiptMetadataValidateUs = timings.MetadataValidateUs
	d.snapshot.LastReceiptMetadataGapUs = timings.MetadataGapUs
	d.snapshot.LastReceiptMetadataTotalUs = timings.MetadataTotalUs
	d.snapshot.LastReceiptHashUs = timings.HashUs
	d.snapshot.LastReceiptJSONMeasureUs = timings.JSONMeasureUs
	d.snapshot.LastReceiptEnvelopeBuildUs = timings.EnvelopeBuildUs
	d.snapshot.LastReceiptTotalBuildUs = timings.TotalBuildUs
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
	d.snapshot.LastReceiptSubmitDurationUs = durationMicroseconds(duration)
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

func durationMicroseconds(duration time.Duration) int64 {
	if duration <= 0 {
		return 0
	}
	return duration.Microseconds()
}

func ReceiptBuildTimingsFromDurations(metadataBuild, hash, jsonMeasure, envelopeBuild, totalBuild time.Duration) ReceiptBuildTimings {
	return ReceiptBuildTimings{
		MetadataBuildMs: durationMilliseconds(metadataBuild),
		HashMs:          durationMilliseconds(hash),
		JSONMeasureMs:   durationMilliseconds(jsonMeasure),
		EnvelopeBuildMs: durationMilliseconds(envelopeBuild),
		TotalBuildMs:    durationMilliseconds(totalBuild),
		MetadataBuildUs: durationMicroseconds(metadataBuild),
		MetadataTotalUs: durationMicroseconds(metadataBuild),
		HashUs:          durationMicroseconds(hash),
		JSONMeasureUs:   durationMicroseconds(jsonMeasure),
		EnvelopeBuildUs: durationMicroseconds(envelopeBuild),
		TotalBuildUs:    durationMicroseconds(totalBuild),
	}
}

func ReceiptBuildTimingsFromMicroseconds(metadataBuildUs, hashUs, jsonMeasureUs, envelopeBuildUs, totalBuildUs int64) ReceiptBuildTimings {
	return normalizeReceiptBuildTimings(ReceiptBuildTimings{
		MetadataBuildUs: metadataBuildUs,
		HashUs:          hashUs,
		JSONMeasureUs:   jsonMeasureUs,
		EnvelopeBuildUs: envelopeBuildUs,
		TotalBuildUs:    totalBuildUs,
	})
}

func normalizeReceiptBuildTimings(timings ReceiptBuildTimings) ReceiptBuildTimings {
	if timings.MetadataBuildUs == 0 && timings.MetadataBuildMs > 0 {
		timings.MetadataBuildUs = timings.MetadataBuildMs * 1000
	}
	timings.MetadataBuildUs = nonNegativeInt64(timings.MetadataBuildUs)
	timings.MetadataStructUs = nonNegativeInt64(timings.MetadataStructUs)
	timings.WeightedValueCopyUs = nonNegativeInt64(timings.WeightedValueCopyUs)
	timings.MetadataDefaultsUs = nonNegativeInt64(timings.MetadataDefaultsUs)
	timings.MetadataValidateUs = nonNegativeInt64(timings.MetadataValidateUs)
	timings.MetadataGapUs = nonNegativeInt64(timings.MetadataGapUs)
	timings.MetadataTotalUs = nonNegativeInt64(timings.MetadataTotalUs)
	if timings.MetadataTotalUs == 0 {
		timings.MetadataTotalUs = timings.MetadataBuildUs
	}
	knownMetadataUs := timings.MetadataStructUs + timings.WeightedValueCopyUs + timings.MetadataDefaultsUs + timings.MetadataValidateUs
	if timings.MetadataTotalUs > knownMetadataUs {
		timings.MetadataGapUs = timings.MetadataTotalUs - knownMetadataUs
	} else {
		timings.MetadataGapUs = 0
	}
	timings.MetadataBuildUs = timings.MetadataTotalUs
	timings.HashUs = nonNegativeInt64(timings.HashUs)
	timings.JSONMeasureUs = nonNegativeInt64(timings.JSONMeasureUs)
	timings.EnvelopeBuildUs = nonNegativeInt64(timings.EnvelopeBuildUs)
	timings.TotalBuildUs = nonNegativeInt64(timings.TotalBuildUs)
	timings.MetadataBuildMs = nonNegativeInt64(timings.MetadataBuildUs / 1000)
	timings.HashMs = nonNegativeInt64(firstNonZeroInt64(timings.HashMs, timings.HashUs/1000))
	timings.JSONMeasureMs = nonNegativeInt64(firstNonZeroInt64(timings.JSONMeasureMs, timings.JSONMeasureUs/1000))
	timings.EnvelopeBuildMs = nonNegativeInt64(firstNonZeroInt64(timings.EnvelopeBuildMs, timings.EnvelopeBuildUs/1000))
	timings.TotalBuildMs = nonNegativeInt64(firstNonZeroInt64(timings.TotalBuildMs, timings.TotalBuildUs/1000))
	return timings
}

func firstNonZeroInt64(first, fallback int64) int64 {
	if first != 0 {
		return first
	}
	return fallback
}

func nonNegativeInt64(value int64) int64 {
	if value < 0 {
		return 0
	}
	return value
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
