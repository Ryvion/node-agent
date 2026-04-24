package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/Ryvion/node-agent/internal/blob"
	"github.com/Ryvion/node-agent/internal/hub"
)

type nativeReportInput struct {
	Task             string
	SiteName         string
	SiteID           string
	ExternalRef      string
	Notes            string
	Photos           []nativeReportPhoto
	RequestedOutputs []string
}

type nativeReportPhoto struct {
	URL   string `json:"url,omitempty"`
	Label string `json:"label,omitempty"`
}

type nativeReportObservation struct {
	ID         string   `json:"id"`
	Title      string   `json:"title"`
	Severity   string   `json:"severity"`
	Source     string   `json:"source"`
	Evidence   []string `json:"evidence,omitempty"`
	Confidence float64  `json:"confidence"`
}

type nativeInspectionReport struct {
	ReportID         string                    `json:"report_id"`
	ReportType       string                    `json:"report_type"`
	JobID            string                    `json:"job_id"`
	GeneratedAt      string                    `json:"generated_at"`
	SiteID           string                    `json:"site_id,omitempty"`
	SiteName         string                    `json:"site_name,omitempty"`
	ExternalRef      string                    `json:"external_ref,omitempty"`
	Summary          string                    `json:"summary"`
	Observations     []nativeReportObservation `json:"observations"`
	Evidence         []nativeReportPhoto       `json:"evidence"`
	QC               nativeReportQC            `json:"qc"`
	RequestedOutputs []string                  `json:"requested_outputs"`
	InputDigest      string                    `json:"input_digest"`
	Limitations      []string                  `json:"limitations"`
}

type nativeReportQC struct {
	Confidence       float64  `json:"confidence"`
	Flags            []string `json:"flags"`
	PhotoCount       int      `json:"photo_count"`
	ObservationCount int      `json:"observation_count"`
}

func (nativeReportEngine) Execute(ctx context.Context, work *hub.WorkAssignment, execCtx executionContext) (*runnerResultSnapshot, error) {
	start := time.Now()
	if execCtx.client == nil {
		return nil, fmt.Errorf("hub client required")
	}
	input, inputDigest := parseNativeReportInput(work)
	report := buildNativeInspectionReport(work, input, inputDigest, start.UTC())
	reportBytes, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return nil, err
	}
	reportHash := sha256.Sum256(reportBytes)
	reportHashHex := hex.EncodeToString(reportHash[:])

	tmp, err := os.CreateTemp("", "ryvion-native-report-*.json")
	if err != nil {
		return nil, err
	}
	tmpPath := tmp.Name()
	if _, err := tmp.Write(reportBytes); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
		return nil, err
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return nil, err
	}
	defer os.Remove(tmpPath)

	runtimeMeta := map[string]any{}
	if execCtx.runtimeManager != nil {
		runtimeMeta = execCtx.runtimeManager.ReceiptMetadata(execCtx.gpuDetected)
	}
	durationMs := time.Since(start).Milliseconds()
	metadata := receiptMetadataBase(
		work,
		runtimeMeta,
		map[string]any{
			"executor":          "native_report",
			"task":              input.Task,
			"duration_ms":       durationMs,
			"exit_code":         0,
			"content":           report.Summary,
			"report_type":       report.ReportType,
			"report_summary":    report.Summary,
			"qc_confidence":     report.QC.Confidence,
			"qc_flags":          report.QC.Flags,
			"photo_count":       report.QC.PhotoCount,
			"observation_count": report.QC.ObservationCount,
			"report_json":       string(reportBytes),
		},
	)

	resultHash := reportHashHex
	uploadRes, uploadErr := blob.Upload(ctx, execCtx.client, work.JobID, tmpPath)
	if uploadErr == nil && uploadRes != nil {
		metadata["blob_url"] = uploadRes.URL
		metadata["object_key"] = uploadRes.Key
		if strings.TrimSpace(uploadRes.Key) != "" {
			metadata["manifest_key"] = uploadRes.Key + ".manifest.json"
		}
		if strings.TrimSpace(uploadRes.Hash) != "" {
			metadata["artifact_sha256"] = uploadRes.Hash
			resultHash = uploadRes.Hash
		}
	} else if uploadErr != nil {
		metadata["upload_error"] = uploadErr.Error()
	}

	units := uint64(work.Units)
	if units == 0 {
		units = 1
	}
	receipt := hub.Receipt{
		JobID:         work.JobID,
		ResultHashHex: resultHash,
		MeteringUnits: units,
		Metadata:      metadata,
	}
	snapshot := &runnerResultSnapshot{
		DurationMs:    durationMs,
		ResultHashHex: resultHash,
		ExitCode:      0,
		MeteringUnits: units,
		BlobURL:       stringValue(metadata["blob_url"]),
		ObjectKey:     stringValue(metadata["object_key"]),
		Metadata:      metadata,
	}
	if err := submitReceiptWithRetry(ctx, execCtx.client, receipt); err != nil {
		return snapshot, err
	}
	return snapshot, nil
}

func parseNativeReportInput(work *hub.WorkAssignment) (nativeReportInput, string) {
	input := nativeReportInput{
		Task:             "inspection_report",
		RequestedOutputs: []string{"json"},
	}
	raw := ""
	if work != nil {
		raw = strings.TrimSpace(work.SpecJSON)
	}
	digest := sha256.Sum256([]byte(raw))
	inputDigest := hex.EncodeToString(digest[:])

	var root map[string]any
	if err := json.Unmarshal([]byte(raw), &root); err == nil {
		applyNativeReportMap(&input, root)
		if text := stringFromMap(root, "text"); strings.HasPrefix(strings.TrimSpace(text), "{") {
			var nested map[string]any
			if err := json.Unmarshal([]byte(text), &nested); err == nil {
				applyNativeReportMap(&input, nested)
			}
		}
		if input.Notes == "" {
			input.Notes = stringFromMap(root, "text")
		}
	}

	input.Task = normalizeNativeReportTask(input.Task)
	if len(input.RequestedOutputs) == 0 {
		input.RequestedOutputs = []string{"json"}
	}
	return input, inputDigest
}

func applyNativeReportMap(input *nativeReportInput, data map[string]any) {
	if input == nil || data == nil {
		return
	}
	if task := stringFromMap(data, "task"); task != "" {
		input.Task = task
	}
	if siteName := stringFromMap(data, "site_name"); siteName != "" {
		input.SiteName = siteName
	}
	if siteID := stringFromMap(data, "site_id"); siteID != "" {
		input.SiteID = siteID
	}
	if externalRef := stringFromMap(data, "external_ref"); externalRef != "" {
		input.ExternalRef = externalRef
	}
	if notes := stringFromMap(data, "notes"); notes != "" {
		input.Notes = notes
	}
	if outputs := stringSliceFromMap(data, "requested_outputs"); len(outputs) > 0 {
		input.RequestedOutputs = outputs
	}
	if photos := photosFromMap(data, "photos"); len(photos) > 0 {
		input.Photos = photos
	}
	if args, ok := data["args"].([]any); ok {
		for _, rawArg := range args {
			arg, ok := rawArg.(string)
			if !ok {
				continue
			}
			key, value, ok := strings.Cut(strings.TrimSpace(arg), "=")
			if ok && strings.EqualFold(strings.TrimSpace(key), "task") && strings.TrimSpace(value) != "" {
				input.Task = strings.TrimSpace(value)
			}
		}
	}
}

func normalizeNativeReportTask(task string) string {
	task = strings.ToLower(strings.TrimSpace(task))
	switch task {
	case "", "report", "site_report", "property_report":
		return "inspection_report"
	default:
		return task
	}
}

func buildNativeInspectionReport(work *hub.WorkAssignment, input nativeReportInput, inputDigest string, now time.Time) nativeInspectionReport {
	jobID := ""
	if work != nil {
		jobID = work.JobID
	}
	siteName := strings.TrimSpace(input.SiteName)
	if siteName == "" {
		siteName = "Untitled site"
	}
	observations := buildNativeReportObservations(input)
	flags := nativeReportFlags(input, observations)
	confidence := nativeReportConfidence(input, observations, flags)
	summary := fmt.Sprintf(
		"Processed %d photo reference(s) and %d note line(s) for %s. Generated %d structured observation(s) with %.0f%% QC confidence.",
		len(input.Photos),
		countNonEmptyLines(input.Notes),
		siteName,
		len(observations),
		confidence*100,
	)
	if strings.TrimSpace(input.Notes) == "" && len(input.Photos) == 0 {
		summary = "Generated an intake report, but no notes or photo references were provided. Add site notes or evidence links for stronger QC."
	}
	return nativeInspectionReport{
		ReportID:         "rpt_" + shortHash(jobID+inputDigest),
		ReportType:       "inspection_report",
		JobID:            jobID,
		GeneratedAt:      now.Format(time.RFC3339),
		SiteID:           input.SiteID,
		SiteName:         siteName,
		ExternalRef:      input.ExternalRef,
		Summary:          summary,
		Observations:     observations,
		Evidence:         input.Photos,
		RequestedOutputs: input.RequestedOutputs,
		InputDigest:      inputDigest,
		QC: nativeReportQC{
			Confidence:       confidence,
			Flags:            flags,
			PhotoCount:       len(input.Photos),
			ObservationCount: len(observations),
		},
		Limitations: []string{
			"Native report runner performs deterministic intake structuring and QC flagging.",
			"Visual photo interpretation is deferred unless an external vision model runner is attached.",
		},
	}
}

func buildNativeReportObservations(input nativeReportInput) []nativeReportObservation {
	lines := nonEmptyLines(input.Notes)
	if len(lines) == 0 && len(input.Photos) > 0 {
		lines = []string{"Photo evidence provided without written notes."}
	}
	observations := make([]nativeReportObservation, 0, len(lines))
	for i, line := range lines {
		if i >= 12 {
			break
		}
		severity := "info"
		lower := strings.ToLower(line)
		switch {
		case containsAny(lower, "mold", "moisture", "leak", "water intrusion", "structural", "electrical", "active crack"):
			severity = "high"
		case containsAny(lower, "crack", "stain", "rust", "broken", "damage", "missing", "loose"):
			severity = "medium"
		}
		confidence := 0.64
		if len(input.Photos) > 0 {
			confidence += 0.12
		}
		if severity != "info" {
			confidence += 0.08
		}
		if confidence > 0.92 {
			confidence = 0.92
		}
		observations = append(observations, nativeReportObservation{
			ID:         fmt.Sprintf("obs_%02d", i+1),
			Title:      truncateSentence(line, 96),
			Severity:   severity,
			Source:     "operator_notes",
			Evidence:   evidenceLabels(input.Photos, 3),
			Confidence: confidence,
		})
	}
	return observations
}

func nativeReportFlags(input nativeReportInput, observations []nativeReportObservation) []string {
	flags := []string{}
	if strings.TrimSpace(input.Notes) == "" {
		flags = append(flags, "missing_notes")
	}
	if len(input.Photos) == 0 {
		flags = append(flags, "missing_photo_evidence")
	}
	for _, obs := range observations {
		if obs.Severity == "high" {
			flags = append(flags, "high_severity_observation")
			break
		}
	}
	return flags
}

func nativeReportConfidence(input nativeReportInput, observations []nativeReportObservation, flags []string) float64 {
	confidence := 0.58
	if strings.TrimSpace(input.Notes) != "" {
		confidence += 0.14
	}
	if len(input.Photos) > 0 {
		confidence += 0.12
	}
	if len(observations) > 0 {
		confidence += 0.08
	}
	confidence -= float64(len(flags)) * 0.04
	if confidence < 0.35 {
		return 0.35
	}
	if confidence > 0.94 {
		return 0.94
	}
	return confidence
}

func stringFromMap(data map[string]any, key string) string {
	v, ok := data[key]
	if !ok {
		return ""
	}
	switch value := v.(type) {
	case string:
		return strings.TrimSpace(value)
	case fmt.Stringer:
		return strings.TrimSpace(value.String())
	default:
		return ""
	}
}

func stringSliceFromMap(data map[string]any, key string) []string {
	raw, ok := data[key].([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(raw))
	for _, item := range raw {
		if s, ok := item.(string); ok && strings.TrimSpace(s) != "" {
			out = append(out, strings.TrimSpace(s))
		}
	}
	return out
}

func photosFromMap(data map[string]any, key string) []nativeReportPhoto {
	raw, ok := data[key].([]any)
	if !ok {
		return nil
	}
	photos := make([]nativeReportPhoto, 0, len(raw))
	for i, item := range raw {
		switch value := item.(type) {
		case string:
			if strings.TrimSpace(value) != "" {
				photos = append(photos, nativeReportPhoto{URL: strings.TrimSpace(value), Label: fmt.Sprintf("photo_%02d", i+1)})
			}
		case map[string]any:
			url := stringFromMap(value, "url")
			if url == "" {
				url = stringFromMap(value, "href")
			}
			label := stringFromMap(value, "label")
			if label == "" {
				label = fmt.Sprintf("photo_%02d", i+1)
			}
			if url != "" {
				photos = append(photos, nativeReportPhoto{URL: url, Label: label})
			}
		}
	}
	return photos
}

func nonEmptyLines(text string) []string {
	parts := strings.FieldsFunc(text, func(r rune) bool {
		return r == '\n' || r == '\r' || r == ';'
	})
	lines := make([]string, 0, len(parts))
	for _, part := range parts {
		if s := strings.TrimSpace(part); s != "" {
			lines = append(lines, s)
		}
	}
	return lines
}

func countNonEmptyLines(text string) int {
	return len(nonEmptyLines(text))
}

func containsAny(text string, needles ...string) bool {
	for _, needle := range needles {
		if strings.Contains(text, needle) {
			return true
		}
	}
	return false
}

func evidenceLabels(photos []nativeReportPhoto, limit int) []string {
	if limit <= 0 || len(photos) == 0 {
		return nil
	}
	out := make([]string, 0, min(limit, len(photos)))
	for i, photo := range photos {
		if i >= limit {
			break
		}
		label := strings.TrimSpace(photo.Label)
		if label == "" {
			label = fmt.Sprintf("photo_%02d", i+1)
		}
		out = append(out, label)
	}
	return out
}

func truncateSentence(text string, maxLen int) string {
	text = strings.Join(strings.Fields(strings.TrimSpace(text)), " ")
	if maxLen <= 0 || len(text) <= maxLen {
		return text
	}
	return strings.TrimSpace(text[:maxLen-1]) + "..."
}

func shortHash(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:])[:12]
}
