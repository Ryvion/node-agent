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

func TestParseWindowsSizedMemory(t *testing.T) {
	tests := []struct {
		input string
		want  uint64
	}{
		{input: "24564 MB", want: 24564 * 1024 * 1024},
		{input: "24 GB", want: 24 * 1024 * 1024 * 1024},
		{input: "3,072 MB", want: 3072 * 1024 * 1024},
		{input: "", want: 0},
		{input: "n/a", want: 0},
	}

	for _, tc := range tests {
		if got := parseWindowsSizedMemory(tc.input); got != tc.want {
			t.Fatalf("parseWindowsSizedMemory(%q)=%d want %d", tc.input, got, tc.want)
		}
	}
}

func TestParseDxDiagDisplayDevicesPrefersDiscreteGPU(t *testing.T) {
	xml := []byte(`
<DxDiag>
  <DisplayDevices>
    <DisplayDevice>
      <CardName>Intel(R) Iris(R) Xe Graphics</CardName>
      <DedicatedMemory>128 MB</DedicatedMemory>
      <DisplayMemory>8120 MB</DisplayMemory>
    </DisplayDevice>
    <DisplayDevice>
      <CardName>AMD Radeon RX 7900 XTX</CardName>
      <DedicatedMemory>24564 MB</DedicatedMemory>
      <DisplayMemory>24564 MB</DisplayMemory>
    </DisplayDevice>
  </DisplayDevices>
</DxDiag>`)

	gpu := pickBestWindowsGPU(parseDxDiagDisplayDevices(xml))
	if gpu.Name != "AMD Radeon RX 7900 XTX" {
		t.Fatalf("expected 7900 XTX, got %q", gpu.Name)
	}
	if gpu.VRAMBytes != 24564*1024*1024 {
		t.Fatalf("expected 24564 MB, got %d", gpu.VRAMBytes)
	}
}

func TestParseWindowsGPUWMICSV(t *testing.T) {
	csv := []byte("Node,AdapterRAM,Name\r\nDESKTOP,4293918720,Intel(R) Iris(R) Xe Graphics\r\nDESKTOP,3221225472,AMD Radeon RX 7900 XTX\r\n")

	gpu := pickBestWindowsGPU(parseWindowsGPUWMICSV(csv))
	if gpu.Name != "AMD Radeon RX 7900 XTX" {
		t.Fatalf("expected 7900 XTX, got %q", gpu.Name)
	}
	if gpu.VRAMBytes != 3221225472 {
		t.Fatalf("expected raw WMI bytes, got %d", gpu.VRAMBytes)
	}
}
