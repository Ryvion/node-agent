package hw

import (
	"runtime"
	"testing"
)

func TestSampleMetricsNonNegative(t *testing.T) {
	m := SampleMetrics()
	if m.CPUUtil < 0 {
		t.Errorf("CPUUtil negative: %f", m.CPUUtil)
	}
	if m.MemUtil < 0 {
		t.Errorf("MemUtil negative: %f", m.MemUtil)
	}
	// GPU and power are expected to be 0 on machines without nvidia-smi.
}

func TestSampleMetricsMacOS(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("macOS only")
	}
	m := SampleMetrics()
	if m.CPUUtil <= 0 {
		t.Errorf("CPUUtil should be positive on macOS, got %f", m.CPUUtil)
	}
	if m.MemUtil <= 0 {
		t.Errorf("MemUtil should be positive on macOS, got %f", m.MemUtil)
	}
}

func TestDetectCaps(t *testing.T) {
	caps := DetectCaps("")
	if caps.CPUCores == 0 {
		t.Errorf("CPUCores should be > 0")
	}
	if caps.RAMBytes == 0 {
		t.Errorf("RAMBytes should be > 0")
	}
}

func TestParseDarwinCPU(t *testing.T) {
	user, sys := parseDarwinCPU("CPU usage: 5.26% user, 3.94% sys, 90.78% idle")
	if user < 5.0 || user > 5.5 {
		t.Errorf("expected user ~5.26, got %f", user)
	}
	if sys < 3.5 || sys > 4.5 {
		t.Errorf("expected sys ~3.94, got %f", sys)
	}
}

func TestParseVMStatLine(t *testing.T) {
	if v := parseVMStatLine("Pages active:                            163235.", "Pages active:"); v != 163235 {
		t.Errorf("expected 163235, got %d", v)
	}
	if v := parseVMStatLine("Pages free:                                4599.", "Pages active:"); v != 0 {
		t.Errorf("expected 0 for non-matching prefix, got %d", v)
	}
}
