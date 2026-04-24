package main

import (
	"strings"
	"testing"
	"time"

	"github.com/Ryvion/node-agent/internal/hub"
)

func TestExecutorKindForNativeReportAssignment(t *testing.T) {
	work := &hub.WorkAssignment{Image: executorKindNativeReport}
	if got := executorKindForAssignment(work); got != executorKindNativeReport {
		t.Fatalf("expected %s, got %s", executorKindNativeReport, got)
	}
}

func TestParseNativeReportInputNestedJSON(t *testing.T) {
	work := &hub.WorkAssignment{
		JobID: "job_test",
		SpecJSON: `{
			"kind":"data_processing",
			"text":"{\"site_name\":\"Atlantic Property\",\"external_ref\":\"WO-42\",\"notes\":\"Moisture staining in north wall\\nVisible crack near basement corner\",\"photos\":[{\"url\":\"https://example.test/photo-a.jpg\",\"label\":\"north wall\"}],\"requested_outputs\":[\"json\"]}",
			"args":["task=property_report"]
		}`,
	}

	input, digest := parseNativeReportInput(work)
	if digest == "" {
		t.Fatal("expected input digest")
	}
	if input.Task != "inspection_report" {
		t.Fatalf("expected normalized inspection_report task, got %q", input.Task)
	}
	if input.SiteName != "Atlantic Property" {
		t.Fatalf("unexpected site name: %q", input.SiteName)
	}
	if input.ExternalRef != "WO-42" {
		t.Fatalf("unexpected external ref: %q", input.ExternalRef)
	}
	if !strings.Contains(input.Notes, "Moisture staining") {
		t.Fatalf("notes were not extracted: %q", input.Notes)
	}
	if len(input.Photos) != 1 || input.Photos[0].Label != "north wall" {
		t.Fatalf("unexpected photos: %#v", input.Photos)
	}
}

func TestBuildNativeInspectionReportFlagsHighSeverity(t *testing.T) {
	input := nativeReportInput{
		Task:             "inspection_report",
		SiteName:         "Test Site",
		Notes:            "Possible mold and active leak under sink",
		RequestedOutputs: []string{"json"},
	}
	now := time.Date(2026, 4, 24, 12, 0, 0, 0, time.UTC)
	report := buildNativeInspectionReport(&hub.WorkAssignment{JobID: "job_test"}, input, "digest", now)
	if len(report.Observations) != 1 {
		t.Fatalf("expected one observation, got %d", len(report.Observations))
	}
	if report.Observations[0].Severity != "high" {
		t.Fatalf("expected high severity, got %q", report.Observations[0].Severity)
	}
	if !containsAny(strings.Join(report.QC.Flags, ","), "high_severity_observation") {
		t.Fatalf("expected high severity flag, got %#v", report.QC.Flags)
	}
}
