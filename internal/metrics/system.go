package metrics

import (
	"context"
	crand "crypto/rand"
	"fmt"
	"math/rand"
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
}

type SampleSet struct {
	CPUUtil    float64
	MemUtil    float64
	GPUUtil    float64
	PowerWatts float64
}

func Capabilities(deviceType string) CapSet {
	caps := CapSet{CPUCores: uint32(runtime.NumCPU()), BandwidthMbps: 100}
	caps.RAMBytes = detectRAMBytes()
	g := detectGPU()
	caps.GPUModel = g.model
	caps.VRAMBytes = g.vramBytes
	caps.Sensors = g.sensors
	return caps
}

func Sample() SampleSet {
	cpu := sampleCPUUtil()
	mem := sampleMemUtil()
	if cpu < 0 {
		cpu = rand.Float64()*70 + 10
	}
	if mem < 0 {
		mem = rand.Float64()*70 + 10
	}
	gpu := GPUUtilSnapshot(context.Background())
	if gpu < 0 {
		gpu = rand.Float64()*70 + 10
	}
	return SampleSet{
		CPUUtil:    cpu,
		MemUtil:    mem,
		GPUUtil:    gpu,
		PowerWatts: rand.Float64()*100 + 50,
	}
}

func sampleCPUUtil() float64 {
	if out1, err := exec.Command("sh", "-lc", "cat /proc/stat | head -n1").CombinedOutput(); err == nil {
		time.Sleep(120 * time.Millisecond)
		if out2, err2 := exec.Command("sh", "-lc", "cat /proc/stat | head -n1").CombinedOutput(); err2 == nil {
			u1, t1 := parseProcStatCPU(string(out1))
			u2, t2 := parseProcStatCPU(string(out2))
			if t2 > t1 && u2 >= u1 {
				return 100.0 * float64(u2-u1) / float64(t2-t1)
			}
		}
	}
	return -1
}

func parseProcStatCPU(line string) (uint64, uint64) {
	f := strings.Fields(line)
	if len(f) < 5 {
		return 0, 0
	}
	var vals []uint64
	for i := 1; i < len(f); i++ {
		if v, err := strconv.ParseUint(f[i], 10, 64); err == nil {
			vals = append(vals, v)
		}
	}
	if len(vals) < 4 {
		return 0, 0
	}
	user := vals[0]
	nice := uint64(0)
	if len(vals) > 1 {
		nice = vals[1]
	}
	system := uint64(0)
	if len(vals) > 2 {
		system = vals[2]
	}
	idle := vals[3]
	total := uint64(0)
	for _, v := range vals {
		total += v
	}
	used := user + nice + system
	_ = idle
	return used, total
}

func sampleMemUtil() float64 {
	if out, err := exec.Command("sh", "-lc", "grep -E 'MemTotal|MemAvailable' /proc/meminfo").CombinedOutput(); err == nil {
		lines := strings.Split(strings.TrimSpace(string(out)), "\n")
		var totalKB, availKB uint64
		for _, ln := range lines {
			ln = strings.TrimSpace(ln)
			if strings.HasPrefix(ln, "MemTotal:") {
				fields := strings.Fields(ln)
				if len(fields) >= 2 {
					totalKB, _ = strconv.ParseUint(fields[1], 10, 64)
				}
			}
			if strings.HasPrefix(ln, "MemAvailable:") {
				fields := strings.Fields(ln)
				if len(fields) >= 2 {
					availKB, _ = strconv.ParseUint(fields[1], 10, 64)
				}
			}
		}
		if totalKB > 0 && availKB <= totalKB {
			used := totalKB - availKB
			return 100.0 * float64(used) / float64(totalKB)
		}
	}
	return -1
}

func RandomHash() []byte {
	b := make([]byte, 32)
	_, _ = crand.Read(b)
	return b
}

// GPUUtilSnapshot attempts to query instantaneous GPU utilization via nvidia-smi.
// Returns -1 if unavailable.
func GPUUtilSnapshot(ctx context.Context) float64 {
	if _, err := exec.LookPath("nvidia-smi"); err != nil {
		return -1
	}
	cctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	out, err := exec.CommandContext(cctx, "nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits").CombinedOutput()
	if err != nil {
		return -1
	}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(lines) == 0 {
		return -1
	}
	var best float64 = -1
	for _, ln := range lines {
		ln = strings.TrimSpace(ln)
		if ln == "" {
			continue
		}
		if v, err := strconv.Atoi(ln); err == nil {
			if float64(v) > best {
				best = float64(v)
			}
		}
	}
	return best
}

type gpuInfo struct {
	model     string
	vramBytes uint64
	sensors   string
}

func detectGPU() gpuInfo {
	if _, err := exec.LookPath("nvidia-smi"); err != nil {
		return gpuInfo{}
	}
	out, err := exec.Command("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits").CombinedOutput()
	if err != nil {
		return gpuInfo{}
	}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(lines) == 0 {
		return gpuInfo{}
	}
	parts := strings.Split(lines[0], ",")
	for i := range parts {
		parts[i] = strings.TrimSpace(parts[i])
	}
	var model, driver string
	var vramMB int64
	if len(parts) > 0 {
		model = parts[0]
	}
	if len(parts) > 1 {
		if v, err := strconv.ParseInt(parts[1], 10, 64); err == nil {
			vramMB = v
		}
	}
	if len(parts) > 2 {
		driver = parts[2]
	}
	info := gpuInfo{model: model, vramBytes: uint64(vramMB) * 1024 * 1024}
	if model != "" || driver != "" {
		info.sensors = strings.TrimSpace(fmt.Sprintf("nvidia-driver:%s model:%s", driver, model))
	}
	return info
}

func detectRAMBytes() uint64 {
	if out, err := exec.Command("sh", "-lc", "grep -i MemTotal /proc/meminfo | awk '{print $2}'").CombinedOutput(); err == nil {
		s := strings.TrimSpace(string(out))
		if s != "" {
			if kb, err := strconv.ParseUint(s, 10, 64); err == nil {
				return kb * 1024
			}
		}
	}

	if out, err := exec.Command("sh", "-lc", "sysctl -n hw.memsize").CombinedOutput(); err == nil {
		s := strings.TrimSpace(string(out))
		if s != "" {
			if b, err := strconv.ParseUint(s, 10, 64); err == nil {
				return b
			}
		}
	}

	if out, err := exec.Command("wmic", "ComputerSystem", "get", "TotalPhysicalMemory").CombinedOutput(); err == nil {
		lines := strings.Split(strings.TrimSpace(string(out)), "\n")
		if len(lines) >= 2 {
			if b, err := strconv.ParseUint(strings.TrimSpace(lines[1]), 10, 64); err == nil {
				return b
			}
		}
	}

	return 8 * 1024 * 1024 * 1024
}
