package main

import (
	"testing"
	"time"

	"github.com/Ryvion/ryvion-node/internal/hub"
	usefulenergy "github.com/Ryvion/ryvion-node/internal/telemetry/usefulenergy"
)

func TestReceiptMetadataBaseIncludesJobScopedEnergyReceipt(t *testing.T) {
	oldState := operatorRuntimeState
	t.Cleanup(func() { operatorRuntimeState = oldState })

	start := time.Date(2026, 5, 15, 2, 30, 0, 0, time.UTC)
	var acc usefulenergy.Accumulator
	before := acc.RecordPower(start, 120, usefulenergy.TelemetryDeviceSensor, usefulenergy.SourceHeartbeatPower)
	acc.RecordPower(start.Add(time.Minute), 120, usefulenergy.TelemetryDeviceSensor, usefulenergy.SourceHeartbeatPower)
	operatorRuntimeState = &operatorRuntime{
		usefulEnergy: acc,
		jobEnergyStarts: map[string]jobEnergyStart{
			"job-energy": {
				startedAt: start,
				snapshot:  before,
			},
		},
	}

	metadata := receiptMetadataBase(&hub.WorkAssignment{
		JobID: "job-energy",
		Kind:  "inference",
		Units: 24,
	})
	raw, ok := metadata["energy_receipt"].(map[string]any)
	if !ok {
		t.Fatalf("energy_receipt missing from metadata: %#v", metadata)
	}
	if raw["schema_version"] != energyReceiptSchemaVersionV1 {
		t.Fatalf("schema_version = %v, want %s", raw["schema_version"], energyReceiptSchemaVersionV1)
	}
	if raw["status"] != "measured" {
		t.Fatalf("status = %v, want measured: %#v", raw["status"], raw)
	}
	if raw["accepted_value"] != uint32(24) {
		t.Fatalf("accepted_value = %#v, want uint32(24)", raw["accepted_value"])
	}
	if raw["energy_used_wh"] != 2.0 {
		t.Fatalf("energy_used_wh = %#v, want 2.0", raw["energy_used_wh"])
	}
	if raw["useful_energy_scoreable"] != true || raw["raw_energy_reward_allowed"] != false {
		t.Fatalf("useful-energy flags = %#v", raw)
	}
	if raw["settlement_mode"] != "metadata_only_shadow" {
		t.Fatalf("settlement_mode = %v, want metadata_only_shadow", raw["settlement_mode"])
	}
}

func TestBuildJobEnergyReceiptFallsBackToLastPowerEstimate(t *testing.T) {
	start := time.Date(2026, 5, 15, 2, 30, 0, 0, time.UTC)
	before := usefulenergy.Snapshot{
		SchemaVersion:   usefulenergy.SchemaVersionV1,
		Status:          usefulenergy.StatusMeasuring,
		TelemetryTier:   usefulenergy.TelemetryDeviceSensor,
		TelemetrySource: usefulenergy.SourceHeartbeatPower,
		LastPowerWatts:  150,
	}
	after := before
	after.LastPowerWatts = 180

	metadata := buildJobEnergyReceiptMetadata("job-estimate", "batch_inference", 9, start, before, start.Add(30*time.Second), after)

	if metadata["status"] != "estimated" {
		t.Fatalf("status = %v, want estimated: %#v", metadata["status"], metadata)
	}
	if metadata["energy_used_wh"] != 1.5 {
		t.Fatalf("energy_used_wh = %#v, want 1.5", metadata["energy_used_wh"])
	}
	if metadata["useful_energy_scoreable"] != true {
		t.Fatalf("expected estimate to be scoreable with accepted value: %#v", metadata)
	}
}
