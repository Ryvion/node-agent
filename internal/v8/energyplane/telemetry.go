package energyplane

import (
	"math"
	"strings"
	"time"
)

type TelemetryStatus string
type TelemetryTier string

const (
	SchemaVersionV1 = "ryvion.node.energyplane.v1"

	StatusUnavailable TelemetryStatus = "unavailable"
	StatusMeasuring   TelemetryStatus = "measuring"
	StatusMeasured    TelemetryStatus = "measured"

	TelemetryEstimatedTDP TelemetryTier = "estimated_tdp"
	TelemetryDeviceSensor TelemetryTier = "device_sensor"
	TelemetryWallMeter    TelemetryTier = "wall_meter"
	TelemetryTrustedMeter TelemetryTier = "trusted_meter"

	SourceHeartbeatPower = "heartbeat_power_watts"

	maxIntegrationWindow = 5 * time.Minute
)

type Accumulator struct {
	startedAt       time.Time
	lastSampleAt    time.Time
	lastPowerWatts  float64
	totalEnergyWh   float64
	integrated      time.Duration
	sampleCount     uint64
	telemetryTier   TelemetryTier
	telemetrySource string
}

type Snapshot struct {
	SchemaVersion           string          `json:"schema_version"`
	Status                  TelemetryStatus `json:"status"`
	TelemetryTier           TelemetryTier   `json:"telemetry_tier,omitempty"`
	TelemetrySource         string          `json:"telemetry_source,omitempty"`
	LastPowerWatts          float64         `json:"last_power_watts,omitempty"`
	SampleCount             uint64          `json:"sample_count,omitempty"`
	WindowStartedAt         time.Time       `json:"window_started_at,omitempty"`
	LastSampleAt            time.Time       `json:"last_sample_at,omitempty"`
	IntegratedSeconds       float64         `json:"integrated_seconds,omitempty"`
	EstimatedEnergyWh       float64         `json:"estimated_energy_wh,omitempty"`
	AveragePowerWatts       float64         `json:"average_power_watts,omitempty"`
	UsefulEnergyReady       bool            `json:"useful_energy_ready"`
	EnergyReceiptCandidate  bool            `json:"energy_receipt_candidate"`
	AcceptedValueRequired   bool            `json:"accepted_value_required"`
	MeasurementWindowCapped bool            `json:"measurement_window_capped,omitempty"`
	Detail                  string          `json:"detail,omitempty"`
}

func (a *Accumulator) RecordPower(sampleAt time.Time, powerWatts float64, tier TelemetryTier, source string) Snapshot {
	if sampleAt.IsZero() {
		sampleAt = time.Now()
	}
	tier = normalizeTelemetryTier(tier)
	source = strings.TrimSpace(source)
	if source == "" {
		source = SourceHeartbeatPower
	}
	if powerWatts < 0 || invalidFloat(powerWatts) {
		return a.SnapshotWithDetail("power_watts unavailable")
	}
	if powerWatts == 0 {
		if a.sampleCount == 0 {
			a.startedAt = sampleAt
		}
		a.lastSampleAt = sampleAt
		a.lastPowerWatts = 0
		a.sampleCount++
		a.telemetryTier = tier
		a.telemetrySource = source
		return a.SnapshotWithDetail("power_watts is zero; waiting for measurable device power")
	}

	capped := false
	if a.sampleCount == 0 || a.lastSampleAt.IsZero() || sampleAt.Before(a.lastSampleAt) {
		a.startedAt = sampleAt
	} else {
		window := sampleAt.Sub(a.lastSampleAt)
		if window > maxIntegrationWindow {
			window = maxIntegrationWindow
			capped = true
		}
		if window > 0 {
			averagePower := (a.lastPowerWatts + powerWatts) / 2
			if a.lastPowerWatts <= 0 {
				averagePower = powerWatts
			}
			a.totalEnergyWh += averagePower * window.Hours()
			a.integrated += window
		}
	}

	a.lastSampleAt = sampleAt
	a.lastPowerWatts = powerWatts
	a.sampleCount++
	a.telemetryTier = tier
	a.telemetrySource = source
	snapshot := a.Snapshot()
	snapshot.MeasurementWindowCapped = capped
	return snapshot
}

func (a Accumulator) Snapshot() Snapshot {
	status := StatusUnavailable
	detail := "no power telemetry samples recorded"
	if a.sampleCount > 0 {
		status = StatusMeasuring
		detail = "power samples recorded; energy integrates after a positive measurement window"
	}
	if a.sampleCount > 1 && a.totalEnergyWh > 0 {
		status = StatusMeasured
		detail = "energy telemetry is ready for useful-energy scoring when accepted value is available"
	}
	avgPower := 0.0
	if a.integrated > 0 {
		avgPower = a.totalEnergyWh / a.integrated.Hours()
	}
	return Snapshot{
		SchemaVersion:          SchemaVersionV1,
		Status:                 status,
		TelemetryTier:          a.telemetryTier,
		TelemetrySource:        a.telemetrySource,
		LastPowerWatts:         round(a.lastPowerWatts),
		SampleCount:            a.sampleCount,
		WindowStartedAt:        a.startedAt,
		LastSampleAt:           a.lastSampleAt,
		IntegratedSeconds:      round(a.integrated.Seconds()),
		EstimatedEnergyWh:      round(a.totalEnergyWh),
		AveragePowerWatts:      round(avgPower),
		UsefulEnergyReady:      status == StatusMeasured,
		EnergyReceiptCandidate: status == StatusMeasured,
		AcceptedValueRequired:  status == StatusMeasured,
		Detail:                 detail,
	}
}

func (a Accumulator) SnapshotWithDetail(detail string) Snapshot {
	snapshot := a.Snapshot()
	snapshot.Detail = strings.TrimSpace(detail)
	return snapshot
}

func normalizeTelemetryTier(tier TelemetryTier) TelemetryTier {
	switch TelemetryTier(strings.TrimSpace(string(tier))) {
	case TelemetryTrustedMeter:
		return TelemetryTrustedMeter
	case TelemetryWallMeter:
		return TelemetryWallMeter
	case TelemetryDeviceSensor:
		return TelemetryDeviceSensor
	case TelemetryEstimatedTDP:
		return TelemetryEstimatedTDP
	default:
		return TelemetryDeviceSensor
	}
}

func invalidFloat(value float64) bool {
	return math.IsNaN(value) || math.IsInf(value, 0)
}

func round(value float64) float64 {
	return math.Round(value*1_000_000) / 1_000_000
}
