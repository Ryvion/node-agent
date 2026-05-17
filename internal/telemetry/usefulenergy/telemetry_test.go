package usefulenergy

import (
	"testing"
	"time"
)

func TestAccumulatorIntegratesPowerSamples(t *testing.T) {
	var acc Accumulator
	start := time.Date(2026, 5, 14, 12, 0, 0, 0, time.UTC)

	first := acc.RecordPower(start, 120, TelemetryDeviceSensor, SourceHeartbeatPower)
	if first.Status != StatusMeasuring {
		t.Fatalf("first status = %q, want measuring", first.Status)
	}
	if first.EstimatedEnergyWh != 0 {
		t.Fatalf("first energy = %v, want 0", first.EstimatedEnergyWh)
	}

	second := acc.RecordPower(start.Add(30*time.Second), 180, TelemetryDeviceSensor, SourceHeartbeatPower)
	if second.Status != StatusMeasured {
		t.Fatalf("second status = %q, want measured: %#v", second.Status, second)
	}
	if second.EstimatedEnergyWh != 1.25 {
		t.Fatalf("energy Wh = %v, want 1.25", second.EstimatedEnergyWh)
	}
	if second.AveragePowerWatts != 150 {
		t.Fatalf("avg power = %v, want 150", second.AveragePowerWatts)
	}
	if !second.UsefulEnergyReady || !second.AcceptedValueRequired {
		t.Fatalf("useful energy flags not set: %#v", second)
	}
}

func TestAccumulatorCapsLongWindows(t *testing.T) {
	var acc Accumulator
	start := time.Date(2026, 5, 14, 12, 0, 0, 0, time.UTC)
	acc.RecordPower(start, 100, TelemetryDeviceSensor, SourceHeartbeatPower)

	got := acc.RecordPower(start.Add(time.Hour), 100, TelemetryDeviceSensor, SourceHeartbeatPower)
	if !got.MeasurementWindowCapped {
		t.Fatalf("measurement window should be capped: %#v", got)
	}
	if got.IntegratedSeconds != 300 {
		t.Fatalf("integrated seconds = %v, want 300", got.IntegratedSeconds)
	}
}

func TestAccumulatorHandlesUnavailablePower(t *testing.T) {
	var acc Accumulator
	got := acc.RecordPower(time.Date(2026, 5, 14, 12, 0, 0, 0, time.UTC), -1, "", "")
	if got.Status != StatusUnavailable {
		t.Fatalf("status = %q, want unavailable", got.Status)
	}
	if got.SampleCount != 0 {
		t.Fatalf("sample count = %d, want 0", got.SampleCount)
	}
	if got.Detail == "" {
		t.Fatalf("detail empty")
	}
}

func TestAccumulatorKeepsZeroPowerSamplesMeasuring(t *testing.T) {
	var acc Accumulator
	start := time.Date(2026, 5, 14, 12, 0, 0, 0, time.UTC)
	acc.RecordPower(start, 0, TelemetryDeviceSensor, SourceHeartbeatPower)
	acc.RecordPower(start.Add(30*time.Second), 0, TelemetryDeviceSensor, SourceHeartbeatPower)

	got := acc.Snapshot()
	if got.Status != StatusMeasuring {
		t.Fatalf("status = %q, want measuring: %#v", got.Status, got)
	}
	if got.SampleCount != 2 {
		t.Fatalf("sample count = %d, want 2", got.SampleCount)
	}
	if got.EstimatedEnergyWh != 0 || got.UsefulEnergyReady {
		t.Fatalf("zero-power snapshot should not be receipt-ready: %#v", got)
	}
}
