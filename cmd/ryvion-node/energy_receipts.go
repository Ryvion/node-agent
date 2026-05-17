package main

import (
	"strings"
	"time"

	"github.com/Ryvion/ryvion-node/internal/hub"
	v8energyplane "github.com/Ryvion/ryvion-node/internal/v8/energyplane"
)

const energyReceiptSchemaVersionV1 = "ryvion.energy_receipt.v1"

type jobEnergyStart struct {
	startedAt time.Time
	snapshot  v8energyplane.Snapshot
}

func jobEnergyReceiptMetadata(work *hub.WorkAssignment) map[string]any {
	if work == nil || operatorRuntimeState == nil {
		return nil
	}
	jobID := strings.TrimSpace(work.JobID)
	if jobID == "" {
		return nil
	}

	operatorRuntimeState.mu.RLock()
	start, ok := operatorRuntimeState.jobEnergyStarts[jobID]
	after := operatorRuntimeState.energyPlane.Snapshot()
	operatorRuntimeState.mu.RUnlock()
	if !ok {
		return nil
	}

	meta := buildJobEnergyReceiptMetadata(jobID, work.Kind, work.Units, start.startedAt, start.snapshot, time.Now(), after)
	if len(meta) == 0 {
		return nil
	}
	return meta
}

func buildJobEnergyReceiptMetadata(jobID string, roleType string, acceptedValue uint32, startedAt time.Time, before v8energyplane.Snapshot, completedAt time.Time, after v8energyplane.Snapshot) map[string]any {
	jobID = strings.TrimSpace(jobID)
	if jobID == "" {
		return nil
	}
	if completedAt.IsZero() {
		completedAt = time.Now()
	}
	if startedAt.IsZero() {
		startedAt = before.LastSampleAt
	}
	if startedAt.IsZero() {
		startedAt = completedAt
	}
	if completedAt.Before(startedAt) {
		completedAt = startedAt
	}

	duration := completedAt.Sub(startedAt)
	energyWh := after.EstimatedEnergyWh - before.EstimatedEnergyWh
	if energyWh < 0 {
		energyWh = 0
	}
	status := "unavailable"
	detail := "job-scoped energy delta unavailable; receipt is not useful-energy scoreable yet"
	if energyWh > 0 {
		status = "measured"
		detail = "job energy delta measured from node EnergyPlane accumulator"
	} else if after.LastPowerWatts > 0 && duration > 0 {
		energyWh = after.LastPowerWatts * duration.Hours()
		status = "estimated"
		detail = "job energy estimated from latest power sample and job wall time"
	}

	avgPower := 0.0
	if duration > 0 && energyWh > 0 {
		avgPower = energyWh / duration.Hours()
	}

	tier := after.TelemetryTier
	if strings.TrimSpace(string(tier)) == "" {
		tier = before.TelemetryTier
	}
	source := strings.TrimSpace(after.TelemetrySource)
	if source == "" {
		source = strings.TrimSpace(before.TelemetrySource)
	}

	receipt := map[string]any{
		"schema_version":             energyReceiptSchemaVersionV1,
		"status":                     status,
		"job_id":                     jobID,
		"role_type":                  strings.TrimSpace(roleType),
		"accepted_value":             acceptedValue,
		"accepted_value_source":      "work_assignment_units",
		"energy_used_wh":             roundEnergyReceiptFloat(energyWh),
		"average_power_watts":        roundEnergyReceiptFloat(avgPower),
		"telemetry_tier":             strings.TrimSpace(string(tier)),
		"telemetry_source":           source,
		"measurement_started_at":     startedAt.UTC().Format(time.RFC3339Nano),
		"measurement_completed_at":   completedAt.UTC().Format(time.RFC3339Nano),
		"measurement_duration_ms":    duration.Milliseconds(),
		"source_schema_version":      after.SchemaVersion,
		"useful_energy_scoreable":    energyWh > 0 && acceptedValue > 0,
		"settlement_mode":            "metadata_only_shadow",
		"accepted_value_required":    energyWh > 0 && acceptedValue == 0,
		"raw_energy_reward_allowed":  false,
		"node_energy_status_before":  strings.TrimSpace(string(before.Status)),
		"node_energy_status_after":   strings.TrimSpace(string(after.Status)),
		"node_energy_sample_count":   after.SampleCount,
		"node_energy_integrated_sec": roundEnergyReceiptFloat(after.IntegratedSeconds - before.IntegratedSeconds),
		"detail":                     detail,
	}
	if receipt["telemetry_tier"] == "" {
		receipt["telemetry_tier"] = string(v8energyplane.TelemetryEstimatedTDP)
	}
	if receipt["source_schema_version"] == "" {
		receipt["source_schema_version"] = v8energyplane.SchemaVersionV1
	}
	return receipt
}

func roundEnergyReceiptFloat(value float64) float64 {
	if value == 0 {
		return 0
	}
	return float64(int64(value*1_000_000+0.5)) / 1_000_000
}
