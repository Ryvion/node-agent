package modellease

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

const telemetryFutureUnixMs int64 = 4_102_444_800_000

func TestBuildResidencyReportResidentModelSummarized(t *testing.T) {
	lease := validTelemetryModelLease("lease-1", "llama-3.1-8b", "q4_k_m", ModelLeaseStateResident)

	report, err := BuildResidencyReport("node-1", []ModelLease{lease})
	if err != nil {
		t.Fatalf("BuildResidencyReport() error = %v, want nil", err)
	}
	if report.NodeID != "node-1" {
		t.Fatalf("report node = %q, want node-1", report.NodeID)
	}
	if len(report.InvalidLeases) != 0 {
		t.Fatalf("invalid leases = %+v, want none", report.InvalidLeases)
	}
	if len(report.Summaries) != 1 {
		t.Fatalf("summary length = %d, want 1", len(report.Summaries))
	}

	got := report.Summaries[0]
	want := ModelResidencySummary{
		ModelID:              "llama-3.1-8b",
		QuantizationID:       "q4_k_m",
		State:                ModelLeaseStateResident,
		VRAMReservedBytes:    24_000_000_000,
		CanAcceptHotRequest:  true,
		LeaseExpiresAtUnixMs: telemetryFutureUnixMs,
	}
	if got != want {
		t.Fatalf("summary = %+v, want %+v", got, want)
	}
}

func TestBuildResidencyReportExpiredResidentCannotAcceptHotRequest(t *testing.T) {
	lease := validTelemetryModelLease("lease-1", "llama-3.1-8b", "q4_k_m", ModelLeaseStateResident)
	lease.LeaseExpiresAtUnixMs = 1

	report, err := BuildResidencyReport("node-1", []ModelLease{lease})
	if err != nil {
		t.Fatalf("BuildResidencyReport() error = %v, want nil", err)
	}
	if len(report.Summaries) != 1 {
		t.Fatalf("summary length = %d, want 1", len(report.Summaries))
	}
	if report.Summaries[0].CanAcceptHotRequest {
		t.Fatalf("CanAcceptHotRequest = true, want false")
	}
}

func TestBuildResidencyReportLoadingModelShownButNotHot(t *testing.T) {
	lease := validTelemetryModelLease("lease-1", "mistral-7b", "q5_k_m", ModelLeaseStateLoading)

	report, err := BuildResidencyReport("node-1", []ModelLease{lease})
	if err != nil {
		t.Fatalf("BuildResidencyReport() error = %v, want nil", err)
	}
	if len(report.Summaries) != 1 {
		t.Fatalf("summary length = %d, want 1", len(report.Summaries))
	}

	got := report.Summaries[0]
	if got.State != ModelLeaseStateLoading {
		t.Fatalf("state = %q, want %q", got.State, ModelLeaseStateLoading)
	}
	if got.CanAcceptHotRequest {
		t.Fatalf("CanAcceptHotRequest = true, want false")
	}
}

func TestBuildResidencyReportDeterministicOrdering(t *testing.T) {
	leases := []ModelLease{
		validTelemetryModelLease("lease-c", "zeta-70b", "q4_k_m", ModelLeaseStateResident),
		validTelemetryModelLease("lease-a2", "alpha-8b", "q8_0", ModelLeaseStateResident),
		validTelemetryModelLease("lease-a1", "alpha-8b", "q4_k_m", ModelLeaseStateWarmup),
		validTelemetryModelLease("lease-b", "beta-13b", "q5_k_m", ModelLeaseStateLoading),
	}
	reversed := []ModelLease{leases[3], leases[2], leases[1], leases[0]}

	report, err := BuildResidencyReport("node-1", leases)
	if err != nil {
		t.Fatalf("BuildResidencyReport(leases) error = %v, want nil", err)
	}
	reportReversed, err := BuildResidencyReport("node-1", reversed)
	if err != nil {
		t.Fatalf("BuildResidencyReport(reversed) error = %v, want nil", err)
	}
	if !reflect.DeepEqual(report.Summaries, reportReversed.Summaries) {
		t.Fatalf("summaries are not deterministic:\nfirst:  %+v\nsecond: %+v", report.Summaries, reportReversed.Summaries)
	}

	gotOrder := []string{
		report.Summaries[0].ModelID + "/" + report.Summaries[0].QuantizationID,
		report.Summaries[1].ModelID + "/" + report.Summaries[1].QuantizationID,
		report.Summaries[2].ModelID + "/" + report.Summaries[2].QuantizationID,
		report.Summaries[3].ModelID + "/" + report.Summaries[3].QuantizationID,
	}
	wantOrder := []string{"alpha-8b/q4_k_m", "alpha-8b/q8_0", "beta-13b/q5_k_m", "zeta-70b/q4_k_m"}
	if !reflect.DeepEqual(gotOrder, wantOrder) {
		t.Fatalf("order = %+v, want %+v", gotOrder, wantOrder)
	}
}

func TestBuildResidencyReportMissingNodeRejected(t *testing.T) {
	_, err := BuildResidencyReport(" \t", nil)
	if !errors.Is(err, ErrInvalidResidencyReport) || !strings.Contains(err.Error(), "node_id required") {
		t.Fatalf("BuildResidencyReport() error = %v, want node error", err)
	}
}

func TestBuildResidencyReportInvalidLeaseReportedSafely(t *testing.T) {
	lease := validTelemetryModelLease("secret/path/lease", "llama-3.1-8b", "q4_k_m", ModelLeaseState("raw-secret-state"))

	report, err := BuildResidencyReport("node-1", []ModelLease{lease})
	if err != nil {
		t.Fatalf("BuildResidencyReport() error = %v, want nil", err)
	}
	if len(report.Summaries) != 0 {
		t.Fatalf("summary length = %d, want 0", len(report.Summaries))
	}
	if len(report.InvalidLeases) != 1 {
		t.Fatalf("invalid lease length = %d, want 1", len(report.InvalidLeases))
	}
	if report.InvalidLeases[0].Index != 0 || !reflect.DeepEqual(report.InvalidLeases[0].Reasons, []string{"unknown_state"}) {
		t.Fatalf("invalid lease report = %+v, want index 0 unknown_state", report.InvalidLeases[0])
	}
	if leaked := fmt.Sprintf("%+v", report.InvalidLeases); strings.Contains(leaked, "secret/path") || strings.Contains(leaked, "raw-secret-state") {
		t.Fatalf("invalid lease report leaked unsafe input: %s", leaked)
	}
}

func TestBuildResidencyReportPathLikeModelIDReportedSafely(t *testing.T) {
	lease := validTelemetryModelLease("lease-1", "/Users/caspian/models/local.gguf", "q4_k_m", ModelLeaseStateResident)

	report, err := BuildResidencyReport("node-1", []ModelLease{lease})
	if err != nil {
		t.Fatalf("BuildResidencyReport() error = %v, want nil", err)
	}
	if len(report.Summaries) != 0 {
		t.Fatalf("summary length = %d, want 0", len(report.Summaries))
	}
	if len(report.InvalidLeases) != 1 {
		t.Fatalf("invalid lease length = %d, want 1", len(report.InvalidLeases))
	}
	if report.InvalidLeases[0].Index != 0 || !reflect.DeepEqual(report.InvalidLeases[0].Reasons, []string{"model_id_local_path"}) {
		t.Fatalf("invalid lease report = %+v, want index 0 model_id_local_path", report.InvalidLeases[0])
	}
	if leaked := fmt.Sprintf("%+v", report.InvalidLeases); strings.Contains(leaked, "/Users/caspian/models/local.gguf") {
		t.Fatalf("invalid lease report leaked local path: %s", leaked)
	}
}

func validTelemetryModelLease(leaseID string, modelID string, quantizationID string, state ModelLeaseState) ModelLease {
	return ModelLease{
		LeaseID:                       leaseID,
		NodeID:                        "node-1",
		ModelID:                       modelID,
		QuantizationID:                quantizationID,
		State:                         state,
		VRAMReservedBytes:             24_000_000_000,
		LeaseExpiresAtUnixMs:          telemetryFutureUnixMs,
		ReadinessRewardCentsPerMinute: 1,
		UpdatedAtUnixMs:               1,
	}
}
