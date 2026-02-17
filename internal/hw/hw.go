package hw

import (
	"context"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"time"
)

type CapSet struct {
	GPUModel      string
	CPUCores      uint32
	RAMBytes      uint64
	VRAMBytes     uint64
	Sensors       string
	BandwidthMbps uint64
	GeohashBucket uint64
	Attestation   uint32
}

type Metrics struct {
	CPUUtil    float64
	MemUtil    float64
	GPUUtil    float64
	PowerWatts float64
}

// DetectCaps returns machine capability info.
// Failed probes return zero values instead of synthetic data.
func DetectCaps(_ string) CapSet {
	gpuModel, vramBytes, sensors := detectGPU()
	return CapSet{
		GPUModel:      gpuModel,
		CPUCores:      uint32(runtime.NumCPU()),
		RAMBytes:      detectRAMBytes(),
		VRAMBytes:     vramBytes,
		Sensors:       sensors,
		BandwidthMbps: 100,
		GeohashBucket: 0,
		Attestation:   0,
	}
}

// SampleMetrics collects volatile utilization metrics.
// Failed probes return zero values.
func SampleMetrics() Metrics {
	cpu := sampleCPU()
	if cpu < 0 {
		cpu = 0
	}
	mem := sampleMem()
	if mem < 0 {
		mem = 0
	}
	gpu := sampleGPU(context.Background())
	if gpu < 0 {
		gpu = 0
	}
	power := samplePower()
	if power < 0 {
		power = 0
	}
	return Metrics{CPUUtil: cpu, MemUtil: mem, GPUUtil: gpu, PowerWatts: power}
}

func sampleCPU() float64 {
	out1, err := exec.Command("sh", "-lc", "cat /proc/stat | head -n1").CombinedOutput()
	if err != nil {
		return -1
	}
	time.Sleep(120 * time.Millisecond)
	out2, err := exec.Command("sh", "-lc", "cat /proc/stat | head -n1").CombinedOutput()
	if err != nil {
		return -1
	}
	used1, total1 := parseProcStat(string(out1))
	used2, total2 := parseProcStat(string(out2))
	if total2 <= total1 || used2 < used1 {
		return -1
	}
	return 100 * float64(used2-used1) / float64(total2-total1)
}

func sampleMem() float64 {
	out, err := exec.Command("sh", "-lc", "grep -E 'MemTotal|MemAvailable' /proc/meminfo").CombinedOutput()
	if err != nil {
		return -1
	}
	var totalKB uint64
	var availKB uint64
	for _, line := range strings.Split(string(out), "\n") {
		fields := strings.Fields(strings.TrimSpace(line))
		if len(fields) < 2 {
			continue
		}
		switch fields[0] {
		case "MemTotal:":
			totalKB, _ = strconv.ParseUint(fields[1], 10, 64)
		case "MemAvailable:":
			availKB, _ = strconv.ParseUint(fields[1], 10, 64)
		}
	}
	if totalKB == 0 || availKB > totalKB {
		return -1
	}
	used := totalKB - availKB
	return 100 * float64(used) / float64(totalKB)
}

func sampleGPU(ctx context.Context) float64 {
	if _, err := exec.LookPath("nvidia-smi"); err != nil {
		return -1
	}
	cctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	out, err := exec.CommandContext(cctx, "nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits").CombinedOutput()
	if err != nil {
		return -1
	}
	best := -1.0
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		v, convErr := strconv.Atoi(line)
		if convErr != nil {
			continue
		}
		if float64(v) > best {
			best = float64(v)
		}
	}
	return best
}

func samplePower() float64 {
	if _, err := exec.LookPath("nvidia-smi"); err != nil {
		return -1
	}
	out, err := exec.Command("nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits").CombinedOutput()
	if err != nil {
		return -1
	}
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		v, convErr := strconv.ParseFloat(line, 64)
		if convErr == nil {
			return v
		}
	}
	return -1
}

func parseProcStat(line string) (used uint64, total uint64) {
	fields := strings.Fields(line)
	if len(fields) < 5 {
		return 0, 0
	}
	vals := make([]uint64, 0, len(fields)-1)
	for i := 1; i < len(fields); i++ {
		v, err := strconv.ParseUint(fields[i], 10, 64)
		if err != nil {
			return 0, 0
		}
		vals = append(vals, v)
		total += v
	}
	if len(vals) < 3 {
		return 0, 0
	}
	used = vals[0] + vals[1] + vals[2]
	return used, total
}

func detectGPU() (model string, vramBytes uint64, sensors string) {
	if _, err := exec.LookPath("nvidia-smi"); err != nil {
		return "", 0, ""
	}
	out, err := exec.Command("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits").CombinedOutput()
	if err != nil {
		return "", 0, ""
	}
	line := strings.TrimSpace(strings.Split(string(out), "\n")[0])
	if line == "" {
		return "", 0, ""
	}
	parts := strings.Split(line, ",")
	for i := range parts {
		parts[i] = strings.TrimSpace(parts[i])
	}
	if len(parts) > 0 {
		model = parts[0]
	}
	if len(parts) > 1 {
		if mb, err := strconv.ParseUint(parts[1], 10, 64); err == nil {
			vramBytes = mb * 1024 * 1024
		}
	}
	driver := ""
	if len(parts) > 2 {
		driver = parts[2]
	}
	if model != "" || driver != "" {
		sensors = strings.TrimSpace("nvidia-driver:" + driver + " model:" + model)
	}
	return model, vramBytes, sensors
}

func detectRAMBytes() uint64 {
	if out, err := exec.Command("sh", "-lc", "grep -i MemTotal /proc/meminfo | awk '{print $2}'").CombinedOutput(); err == nil {
		if kb, convErr := strconv.ParseUint(strings.TrimSpace(string(out)), 10, 64); convErr == nil && kb > 0 {
			return kb * 1024
		}
	}
	if out, err := exec.Command("sh", "-lc", "sysctl -n hw.memsize").CombinedOutput(); err == nil {
		if b, convErr := strconv.ParseUint(strings.TrimSpace(string(out)), 10, 64); convErr == nil && b > 0 {
			return b
		}
	}
	if out, err := exec.Command("wmic", "ComputerSystem", "get", "TotalPhysicalMemory").CombinedOutput(); err == nil {
		lines := strings.Split(strings.TrimSpace(string(out)), "\n")
		if len(lines) >= 2 {
			if b, convErr := strconv.ParseUint(strings.TrimSpace(lines[1]), 10, 64); convErr == nil {
				return b
			}
		}
	}
	if info, err := os.ReadFile("/proc/meminfo"); err == nil {
		for _, line := range strings.Split(string(info), "\n") {
			if strings.HasPrefix(line, "MemTotal:") {
				fields := strings.Fields(line)
				if len(fields) >= 2 {
					if kb, convErr := strconv.ParseUint(fields[1], 10, 64); convErr == nil {
						return kb * 1024
					}
				}
			}
		}
	}
	return 0
}
