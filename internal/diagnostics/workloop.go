package diagnostics

import (
	"encoding/json"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"
)

const (
	defaultWorkLoopEventLimit  = 50
	maxWorkLoopIDLen           = 256
	maxWorkLoopKindLen         = 128
	maxWorkLoopTaskLen         = 128
	maxWorkLoopErrorLen        = 512
	maxWorkLoopEventNameLen    = 64
	maxWorkLoopContextValueLen = 128
)

type WorkLoopSnapshot struct {
	LastPollStartedAt              string          `json:"last_poll_started_at"`
	LastPollCompletedAt            string          `json:"last_poll_completed_at"`
	LastPollDurationMs             int64           `json:"last_poll_duration_ms"`
	LastPollCycleDurationMs        int64           `json:"last_poll_cycle_duration_ms"`
	LastPollError                  string          `json:"last_poll_error"`
	LastWorkSeenAt                 string          `json:"last_work_seen_at"`
	LastWorkJobID                  string          `json:"last_work_job_id"`
	LastWorkKind                   string          `json:"last_work_kind"`
	LastWorkSpecTask               string          `json:"last_work_spec_task"`
	LastWorkDecodeMs               int64           `json:"last_work_decode_ms"`
	LastExecutionStartedAt         string          `json:"last_execution_started_at"`
	LastExecutionCompletedAt       string          `json:"last_execution_completed_at"`
	LastExecutionDurationMs        int64           `json:"last_execution_duration_ms"`
	LastExecutionDurationUs        int64           `json:"last_execution_duration_us"`
	LastReceiptBuildMs             int64           `json:"last_receipt_build_ms"`
	LastReceiptMetadataBuildMs     int64           `json:"last_receipt_metadata_build_ms"`
	LastReceiptHashMs              int64           `json:"last_receipt_hash_ms"`
	LastReceiptJSONMeasureMs       int64           `json:"last_receipt_json_measure_ms"`
	LastReceiptEnvelopeBuildMs     int64           `json:"last_receipt_envelope_build_ms"`
	LastReceiptTotalBuildMs        int64           `json:"last_receipt_total_build_ms"`
	LastReceiptMetadataBuildUs     int64           `json:"last_receipt_metadata_build_us"`
	LastReceiptMetadataStructUs    int64           `json:"last_receipt_metadata_struct_us"`
	LastReceiptWeightedValueCopyUs int64           `json:"last_receipt_weighted_value_copy_us,omitempty"`
	LastReceiptMetadataDefaultsUs  int64           `json:"last_receipt_metadata_defaults_us"`
	LastReceiptMetadataValidateUs  int64           `json:"last_receipt_metadata_validate_us"`
	LastReceiptMetadataGapUs       int64           `json:"last_receipt_metadata_gap_us"`
	LastReceiptMetadataTotalUs     int64           `json:"last_receipt_metadata_total_us"`
	LastReceiptHashUs              int64           `json:"last_receipt_hash_us"`
	LastReceiptJSONMeasureUs       int64           `json:"last_receipt_json_measure_us"`
	LastReceiptEnvelopeBuildUs     int64           `json:"last_receipt_envelope_build_us"`
	LastReceiptTotalBuildUs        int64           `json:"last_receipt_total_build_us"`
	LastReceiptReadyAt             string          `json:"last_receipt_ready_at"`
	LastReceiptReadyToSubmitMs     int64           `json:"last_receipt_ready_to_submit_ms"`
	LastReceiptReadyToSubmitUs     int64           `json:"last_receipt_ready_to_submit_us"`
	LastReceiptSubmitQueueGapMs    int64           `json:"last_receipt_submit_queue_gap_ms"`
	LastReceiptSubmitQueueGapUs    int64           `json:"last_receipt_submit_queue_gap_us"`
	LastReceiptSubmitStartedAt     string          `json:"last_receipt_submit_started_at"`
	LastReceiptSubmitCompletedAt   string          `json:"last_receipt_submit_completed_at"`
	LastReceiptSubmitDurationMs    int64           `json:"last_receipt_submit_duration_ms"`
	LastReceiptSubmitDurationUs    int64           `json:"last_receipt_submit_duration_us"`
	LastReceiptSubmitError         string          `json:"last_receipt_submit_error"`
	LastReceiptAttempts            int             `json:"last_receipt_attempts"`
	PollCount                      uint64          `json:"poll_count"`
	WorkSeenCount                  uint64          `json:"work_seen_count"`
	WorkCompletedCount             uint64          `json:"work_completed_count"`
	ReceiptSubmittedCount          uint64          `json:"receipt_submitted_count"`
	ReceiptFailedCount             uint64          `json:"receipt_failed_count"`
	RecentEvents                   []WorkLoopEvent `json:"recent_events"`
}

type WorkLoopEvent struct {
	Name        string            `json:"name"`
	JobID       string            `json:"job_id"`
	Kind        string            `json:"kind"`
	At          string            `json:"at"`
	SincePrevUs int64             `json:"since_prev_us"`
	SafeContext map[string]string `json:"safe_context"`
}

type WorkLoopDiagnostics struct {
	mu sync.RWMutex

	snapshot WorkLoopSnapshot

	lastPollStartedAt          time.Time
	lastExecutionStartedAt     time.Time
	lastReceiptReadyAt         time.Time
	lastReceiptReadyJobID      string
	lastReceiptReadyKind       string
	lastReceiptSubmitStartedAt time.Time
	lastReceiptSubmitJobID     string

	eventLimit  int
	events      []workLoopEventRecord
	eventNext   int
	eventCount  int
	lastEventAt time.Time
}

type workLoopEventRecord struct {
	event WorkLoopEvent
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
	return newWorkLoopDiagnostics(defaultWorkLoopEventLimit)
}

func newWorkLoopDiagnostics(eventLimit int) *WorkLoopDiagnostics {
	if eventLimit <= 0 {
		eventLimit = defaultWorkLoopEventLimit
	}
	return &WorkLoopDiagnostics{
		eventLimit: eventLimit,
		events:     make([]workLoopEventRecord, eventLimit),
	}
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
	d.recordEventLocked(now, "poll_start", "", "", nil)
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
	d.recordEventLocked(now, "poll_end", "", "", nil)
}

func (d *WorkLoopDiagnostics) RecordWorkSeen(jobID, kind, specTask string) {
	if d == nil {
		return
	}
	now := time.Now().UTC()
	d.mu.Lock()
	defer d.mu.Unlock()
	d.snapshot.LastWorkSeenAt = formatWorkLoopTime(now)
	cleanJobID := cleanWorkLoopText(jobID, maxWorkLoopIDLen)
	cleanKind := cleanWorkLoopText(kind, maxWorkLoopKindLen)
	cleanSpecTask := cleanWorkLoopText(specTask, maxWorkLoopTaskLen)
	d.snapshot.LastWorkJobID = cleanJobID
	d.snapshot.LastWorkKind = cleanKind
	d.snapshot.LastWorkSpecTask = cleanSpecTask
	d.snapshot.WorkSeenCount++
	d.recordEventLocked(now, "work_seen", cleanJobID, cleanKind, workLoopSpecContext(cleanSpecTask))
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
	d.recordEventLocked(now, "execution_start", d.snapshot.LastWorkJobID, d.snapshot.LastWorkKind, workLoopSpecContext(d.snapshot.LastWorkSpecTask))
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
	d.recordEventLocked(now, "execution_end", d.snapshot.LastWorkJobID, d.snapshot.LastWorkKind, workLoopSpecContext(d.snapshot.LastWorkSpecTask))
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

func (d *WorkLoopDiagnostics) RecordReceiptReady(jobID, kind string, readyAt time.Time, safeContext map[string]string) {
	if d == nil {
		return
	}
	if readyAt.IsZero() {
		readyAt = time.Now()
	}
	readyAt = readyAt.UTC()
	cleanJobID := cleanWorkLoopText(jobID, maxWorkLoopIDLen)
	cleanKind := cleanWorkLoopText(kind, maxWorkLoopKindLen)

	d.mu.Lock()
	defer d.mu.Unlock()
	if cleanJobID == "" {
		cleanJobID = d.snapshot.LastWorkJobID
	}
	if cleanKind == "" {
		cleanKind = d.snapshot.LastWorkKind
	}
	d.lastReceiptReadyAt = readyAt
	d.lastReceiptReadyJobID = cleanJobID
	d.lastReceiptReadyKind = cleanKind
	d.snapshot.LastReceiptReadyAt = formatWorkLoopTime(readyAt)
	if d.snapshot.LastWorkJobID == "" {
		d.snapshot.LastWorkJobID = cleanJobID
	}
	if d.snapshot.LastWorkKind == "" {
		d.snapshot.LastWorkKind = cleanKind
	}
	if d.snapshot.LastWorkSpecTask == "" {
		if specTask, ok := safeContext["spec_task"]; ok {
			cleanSpecTask := cleanWorkLoopText(specTask, maxWorkLoopTaskLen)
			if isSafeWorkLoopLabel(cleanSpecTask) {
				d.snapshot.LastWorkSpecTask = cleanSpecTask
			}
		}
	}
	d.recordEventLocked(readyAt, "receipt_ready", cleanJobID, cleanKind, safeContext)
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
	shouldRecordSubmitStart := attempt == 1 || d.lastReceiptSubmitStartedAt.IsZero() || cleanJobID != d.lastReceiptSubmitJobID
	if shouldRecordSubmitStart {
		d.lastReceiptSubmitStartedAt = now
		d.lastReceiptSubmitJobID = cleanJobID
		d.snapshot.LastReceiptSubmitStartedAt = formatWorkLoopTime(now)
		d.snapshot.LastReceiptSubmitError = ""
		d.recordReadyToSubmitGapLocked(now, cleanJobID)
	}
	d.snapshot.LastReceiptAttempts = attempt
	eventJobID := cleanJobID
	if eventJobID == "" {
		eventJobID = d.snapshot.LastWorkJobID
	}
	eventKind := d.snapshot.LastWorkKind
	if eventKind == "" {
		eventKind = d.lastReceiptReadyKind
	}
	d.recordEventLocked(now, "receipt_submit_start", eventJobID, eventKind, workLoopSpecContext(d.snapshot.LastWorkSpecTask))
}

func (d *WorkLoopDiagnostics) recordReadyToSubmitGapLocked(now time.Time, cleanJobID string) {
	if d.lastReceiptReadyAt.IsZero() {
		return
	}
	if d.lastReceiptReadyJobID != "" && cleanJobID != "" && cleanJobID != d.lastReceiptReadyJobID {
		return
	}
	gap := now.Sub(d.lastReceiptReadyAt)
	gapMs := durationMilliseconds(gap)
	gapUs := durationMicroseconds(gap)
	d.snapshot.LastReceiptReadyToSubmitMs = gapMs
	d.snapshot.LastReceiptReadyToSubmitUs = gapUs
	d.snapshot.LastReceiptSubmitQueueGapMs = gapMs
	d.snapshot.LastReceiptSubmitQueueGapUs = gapUs

	eventJobID := cleanJobID
	if eventJobID == "" {
		eventJobID = firstNonEmptyWorkLoopString(d.lastReceiptReadyJobID, d.snapshot.LastWorkJobID)
	}
	eventKind := firstNonEmptyWorkLoopString(d.snapshot.LastWorkKind, d.lastReceiptReadyKind)
	context := workLoopSpecContext(d.snapshot.LastWorkSpecTask)
	if context == nil {
		context = map[string]string{}
	}
	context["gap_us"] = strconv.FormatInt(gapUs, 10)
	if eventJobID != "" {
		context["job_id"] = eventJobID
	}
	if eventKind != "" {
		context["kind"] = eventKind
	}
	d.recordEventLocked(now, "receipt_ready_to_submit_gap", eventJobID, eventKind, context)
	d.lastReceiptReadyAt = time.Time{}
	d.lastReceiptReadyJobID = ""
	d.lastReceiptReadyKind = ""
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
	eventJobID := d.lastReceiptSubmitJobID
	if eventJobID == "" {
		eventJobID = d.snapshot.LastWorkJobID
	}
	d.recordEventLocked(now, "receipt_submit_end", eventJobID, d.snapshot.LastWorkKind, workLoopSpecContext(d.snapshot.LastWorkSpecTask))
	if err != nil {
		d.snapshot.ReceiptFailedCount++
		return
	}
	d.snapshot.ReceiptSubmittedCount++
}

func (d *WorkLoopDiagnostics) RecordEvent(name, jobID, kind string, safeContext map[string]string) {
	if d == nil {
		return
	}
	now := time.Now().UTC()
	d.mu.Lock()
	defer d.mu.Unlock()
	d.recordEventLocked(now, name, jobID, kind, safeContext)
}

func (d *WorkLoopDiagnostics) RecordReceiptSubstepEvent(name, jobID, kind string, safeContext map[string]string) {
	d.RecordEvent(name, jobID, kind, safeContext)
}

func (d *WorkLoopDiagnostics) EventTimeline() []WorkLoopEvent {
	if d == nil {
		return []WorkLoopEvent{}
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.eventTimelineLocked()
}

func (d *WorkLoopDiagnostics) Snapshot() WorkLoopSnapshot {
	if d == nil {
		return WorkLoopSnapshot{RecentEvents: []WorkLoopEvent{}}
	}
	d.mu.RLock()
	defer d.mu.RUnlock()
	snapshot := d.snapshot
	snapshot.RecentEvents = d.eventTimelineLocked()
	return snapshot
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

func firstNonEmptyWorkLoopString(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func nonNegativeInt64(value int64) int64 {
	if value < 0 {
		return 0
	}
	return value
}

func (d *WorkLoopDiagnostics) recordEventLocked(now time.Time, name, jobID, kind string, safeContext map[string]string) {
	name = cleanWorkLoopEventName(name)
	if name == "" {
		return
	}
	d.ensureEventBufferLocked()
	event := WorkLoopEvent{
		Name:        name,
		JobID:       cleanWorkLoopText(jobID, maxWorkLoopIDLen),
		Kind:        cleanWorkLoopText(kind, maxWorkLoopKindLen),
		At:          formatWorkLoopTime(now),
		SafeContext: sanitizeWorkLoopEventContext(safeContext),
	}
	if !d.lastEventAt.IsZero() {
		event.SincePrevUs = nonNegativeInt64(now.Sub(d.lastEventAt).Microseconds())
	}
	d.lastEventAt = now
	d.events[d.eventNext] = workLoopEventRecord{event: event}
	d.eventNext = (d.eventNext + 1) % d.eventLimit
	if d.eventCount < d.eventLimit {
		d.eventCount++
	}
}

func (d *WorkLoopDiagnostics) ensureEventBufferLocked() {
	if d.eventLimit <= 0 {
		d.eventLimit = defaultWorkLoopEventLimit
	}
	if len(d.events) == d.eventLimit {
		return
	}
	d.events = make([]workLoopEventRecord, d.eventLimit)
	d.eventNext = 0
	d.eventCount = 0
	d.lastEventAt = time.Time{}
}

func (d *WorkLoopDiagnostics) eventTimelineLocked() []WorkLoopEvent {
	if d.eventCount == 0 {
		return []WorkLoopEvent{}
	}
	out := make([]WorkLoopEvent, 0, d.eventCount)
	start := d.eventNext - d.eventCount
	if start < 0 {
		start += d.eventLimit
	}
	for i := 0; i < d.eventCount; i++ {
		idx := (start + i) % d.eventLimit
		out = append(out, cloneWorkLoopEvent(d.events[idx].event))
	}
	out[0].SincePrevUs = 0
	return out
}

func cloneWorkLoopEvent(event WorkLoopEvent) WorkLoopEvent {
	event.SafeContext = cloneWorkLoopEventContext(event.SafeContext)
	return event
}

func cloneWorkLoopEventContext(input map[string]string) map[string]string {
	out := make(map[string]string, len(input))
	for key, value := range input {
		out[key] = value
	}
	return out
}

func workLoopSpecContext(specTask string) map[string]string {
	if strings.TrimSpace(specTask) == "" {
		return nil
	}
	return map[string]string{"spec_task": specTask}
}

func cleanWorkLoopEventName(name string) string {
	name = cleanWorkLoopText(name, maxWorkLoopEventNameLen)
	if !isAllowedWorkLoopEventName(name) {
		return ""
	}
	return name
}

func isAllowedWorkLoopEventName(name string) bool {
	switch name {
	case "work_seen",
		"execution_start",
		"execution_end",
		"v7_fast_path_start",
		"v7_fast_path_receipt_ready",
		"v7_fast_path_submit_start",
		"v7_fast_path_submit_end",
		"generic_path_entered",
		"pre_submit_block_start",
		"pre_submit_block_end",
		"receipt_build_start",
		"receipt_metadata_start",
		"receipt_metadata_struct_end",
		"receipt_weighted_copy_end",
		"receipt_defaults_end",
		"receipt_validate_end",
		"receipt_metadata_end",
		"receipt_hash_end",
		"receipt_json_measure_end",
		"receipt_envelope_end",
		"receipt_build_end",
		"receipt_ready",
		"receipt_ready_to_submit_gap",
		"receipt_submit_start",
		"receipt_submit_end",
		"poll_start",
		"poll_end":
		return true
	default:
		return false
	}
}

func sanitizeWorkLoopEventContext(input map[string]string) map[string]string {
	out := map[string]string{}
	for key, value := range input {
		key = strings.TrimSpace(key)
		if !isAllowedWorkLoopContextKey(key) {
			continue
		}
		cleanValue := cleanWorkLoopContextValue(key, value)
		if cleanValue == "" {
			continue
		}
		out[key] = cleanValue
	}
	return out
}

func isAllowedWorkLoopContextKey(key string) bool {
	switch key {
	case "spec_task",
		"job_id",
		"kind",
		"token_count",
		"value_dim",
		"gap_us",
		"metadata_total_us",
		"metadata_gap_us",
		"weighted_value_len",
		"receipt_body_bytes":
		return true
	default:
		return false
	}
}

func cleanWorkLoopContextValue(key, value string) string {
	value = cleanWorkLoopText(value, maxWorkLoopContextValueLen)
	if value == "" {
		return ""
	}
	if key == "spec_task" || key == "job_id" || key == "kind" {
		if !isSafeWorkLoopLabel(value) {
			return ""
		}
		return value
	}
	if !isWorkLoopUnsignedInteger(value) {
		return ""
	}
	return value
}

func isSafeWorkLoopLabel(value string) bool {
	for _, r := range value {
		if r >= 'a' && r <= 'z' {
			continue
		}
		if r >= 'A' && r <= 'Z' {
			continue
		}
		if r >= '0' && r <= '9' {
			continue
		}
		if r == '_' || r == '-' || r == '.' || r == ':' {
			continue
		}
		return false
	}
	return true
}

func isWorkLoopUnsignedInteger(value string) bool {
	for _, r := range value {
		if r < '0' || r > '9' {
			return false
		}
	}
	return value != ""
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
