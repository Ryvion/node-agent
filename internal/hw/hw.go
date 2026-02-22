package hw

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// nvidiaSMIPath caches the resolved nvidia-smi binary path.
var nvidiaSMIPath string

func init() {
	nvidiaSMIPath = findNvidiaSMI()
}

// findNvidiaSMI locates nvidia-smi, checking common OS-specific paths
// when it's not in the default PATH (e.g. Windows services).
func findNvidiaSMI() string {
	if p, err := exec.LookPath("nvidia-smi"); err == nil {
		return p
	}
	if runtime.GOOS == "windows" {
		candidates := []string{
			filepath.Join(os.Getenv("SystemRoot"), "System32", "nvidia-smi.exe"),
			filepath.Join(os.Getenv("ProgramFiles"), "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"),
		}
		for _, c := range candidates {
			if _, err := os.Stat(c); err == nil {
				return c
			}
		}
	}
	return ""
}

func hasNvidiaSMI() bool { return nvidiaSMIPath != "" }

type CapSet struct {
	GPUModel      string
	CPUCores      uint32
	RAMBytes      uint64
	VRAMBytes     uint64
	Sensors       string
	BandwidthMbps uint64
	GeohashBucket uint64
	Attestation   uint32
	TEESupported  bool
	TEEType       string
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
	teeSupported, teeType := DetectTEE()
	return CapSet{
		GPUModel:      gpuModel,
		CPUCores:      uint32(runtime.NumCPU()),
		RAMBytes:      detectRAMBytes(),
		VRAMBytes:     vramBytes,
		Sensors:       sensors,
		BandwidthMbps: 100,
		GeohashBucket: 0,
		Attestation:   0,
		TEESupported:  teeSupported,
		TEEType:       teeType,
	}
}

// DetectTEE checks for hardware TEE support.
// Returns (true, type) if a TEE device is found, (false, "") otherwise.
func DetectTEE() (bool, string) {
	// AMD SEV-SNP
	for _, p := range []string{"/dev/sev", "/dev/sev-guest"} {
		if _, err := os.Stat(p); err == nil {
			return true, "sev-snp"
		}
	}
	// Intel SGX
	for _, p := range []string{"/dev/sgx_enclave", "/dev/sgx"} {
		if _, err := os.Stat(p); err == nil {
			return true, "sgx"
		}
	}
	// Intel TDX
	if _, err := os.Stat("/sys/firmware/tdx"); err == nil {
		return true, "tdx"
	}
	return false, ""
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
	// Linux: /proc/stat
	out1, err := exec.Command("sh", "-lc", "cat /proc/stat | head -n1").CombinedOutput()
	if err == nil {
		time.Sleep(120 * time.Millisecond)
		out2, err2 := exec.Command("sh", "-lc", "cat /proc/stat | head -n1").CombinedOutput()
		if err2 == nil {
			used1, total1 := parseProcStat(string(out1))
			used2, total2 := parseProcStat(string(out2))
			if total2 > total1 && used2 >= used1 {
				return 100 * float64(used2-used1) / float64(total2-total1)
			}
		}
	}
	// macOS: top -l 2 to get delta-based CPU usage
	if runtime.GOOS == "darwin" {
		dctx, dcancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer dcancel()
		out, err := exec.CommandContext(dctx, "top", "-l", "2", "-n", "0", "-s", "0").CombinedOutput()
		if err == nil {
			// Parse the second sample's "CPU usage:" line
			lines := strings.Split(string(out), "\n")
			var lastUser, lastSys float64
			found := false
			for _, line := range lines {
				if strings.Contains(line, "CPU usage:") {
					lastUser, lastSys = parseDarwinCPU(line)
					found = true
				}
			}
			if found {
				return lastUser + lastSys
			}
		}
	}
	// Windows: wmic (with timeout to avoid hangs in service context)
	if runtime.GOOS == "windows" {
		wctx, wcancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer wcancel()
		out, err := exec.CommandContext(wctx, "wmic", "cpu", "get", "LoadPercentage").CombinedOutput()
		if err == nil {
			for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
				line = strings.TrimSpace(line)
				if v, convErr := strconv.ParseFloat(line, 64); convErr == nil {
					return v
				}
			}
		}
	}
	return -1
}

// parseDarwinCPU parses a macOS top "CPU usage:" line like:
// "CPU usage: 5.26% user, 3.94% sys, 90.78% idle"
func parseDarwinCPU(line string) (user, sys float64) {
	line = strings.TrimSpace(line)
	if i := strings.Index(line, "CPU usage:"); i >= 0 {
		line = line[i+len("CPU usage:"):]
	}
	for _, part := range strings.Split(line, ",") {
		part = strings.TrimSpace(part)
		if strings.HasSuffix(part, "user") {
			part = strings.TrimSuffix(strings.TrimSpace(strings.TrimSuffix(part, "user")), "%")
			user, _ = strconv.ParseFloat(strings.TrimSpace(part), 64)
		} else if strings.HasSuffix(part, "sys") {
			part = strings.TrimSuffix(strings.TrimSpace(strings.TrimSuffix(part, "sys")), "%")
			sys, _ = strconv.ParseFloat(strings.TrimSpace(part), 64)
		}
	}
	return user, sys
}

func sampleMem() float64 {
	// Linux: /proc/meminfo
	out, err := exec.Command("sh", "-lc", "grep -E 'MemTotal|MemAvailable' /proc/meminfo").CombinedOutput()
	if err == nil {
		var totalKB, availKB uint64
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
		if totalKB > 0 && availKB <= totalKB {
			return 100 * float64(totalKB-availKB) / float64(totalKB)
		}
	}
	// macOS: sysctl + vm_stat
	if runtime.GOOS == "darwin" {
		totalBytes := detectRAMBytes() // uses sysctl hw.memsize
		if totalBytes > 0 {
			if used := darwinUsedMemBytes(); used > 0 {
				return 100 * float64(used) / float64(totalBytes)
			}
		}
	}
	// Windows: wmic (with timeout to avoid hangs in service context)
	if runtime.GOOS == "windows" {
		wctx, wcancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer wcancel()
		out, err := exec.CommandContext(wctx, "wmic", "OS", "get", "TotalVisibleMemorySize,FreePhysicalMemory").CombinedOutput()
		if err == nil {
			lines := strings.Split(strings.TrimSpace(string(out)), "\n")
			for _, line := range lines {
				fields := strings.Fields(strings.TrimSpace(line))
				if len(fields) == 2 {
					free, e1 := strconv.ParseUint(fields[0], 10, 64)
					total, e2 := strconv.ParseUint(fields[1], 10, 64)
					if e1 == nil && e2 == nil && total > 0 {
						return 100 * float64(total-free) / float64(total)
					}
				}
			}
		}
	}
	return -1
}

// darwinUsedMemBytes parses vm_stat output to calculate used memory on macOS.
// vm_stat reports page counts; we multiply by the page size (typically 16384 on ARM64).
func darwinUsedMemBytes() uint64 {
	out, err := exec.Command("vm_stat").CombinedOutput()
	if err != nil {
		return 0
	}
	var pageSize uint64 = 16384 // ARM64 default
	// First line: "Mach Virtual Memory Statistics: (page size of NNNN bytes)"
	lines := strings.Split(string(out), "\n")
	if len(lines) > 0 {
		if i := strings.Index(lines[0], "page size of "); i >= 0 {
			rest := lines[0][i+len("page size of "):]
			if j := strings.IndexByte(rest, ' '); j > 0 {
				if ps, err := strconv.ParseUint(rest[:j], 10, 64); err == nil && ps > 0 {
					pageSize = ps
				}
			}
		}
	}
	var active, wired, compressed uint64
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if v := parseVMStatLine(line, "Pages active:"); v > 0 {
			active = v
		} else if v := parseVMStatLine(line, "Pages wired down:"); v > 0 {
			wired = v
		} else if v := parseVMStatLine(line, "Pages occupied by compressor:"); v > 0 {
			compressed = v
		}
	}
	return (active + wired + compressed) * pageSize
}

func parseVMStatLine(line, prefix string) uint64 {
	if !strings.HasPrefix(line, prefix) {
		return 0
	}
	val := strings.TrimSpace(strings.TrimPrefix(line, prefix))
	val = strings.TrimSuffix(val, ".")
	v, _ := strconv.ParseUint(val, 10, 64)
	return v
}

func sampleGPU(ctx context.Context) float64 {
	if !hasNvidiaSMI() {
		return -1
	}
	cctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	out, err := exec.CommandContext(cctx, nvidiaSMIPath, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits").CombinedOutput()
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
	if !hasNvidiaSMI() {
		return -1
	}
	out, err := exec.Command(nvidiaSMIPath, "--query-gpu=power.draw", "--format=csv,noheader,nounits").CombinedOutput()
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
	// 1) NVIDIA via nvidia-smi
	if hasNvidiaSMI() {
		out, err := exec.Command(nvidiaSMIPath, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits").CombinedOutput()
		if err == nil {
			line := strings.TrimSpace(strings.Split(string(out), "\n")[0])
			if line != "" {
				parts := strings.Split(line, ",")
				for i := range parts {
					parts[i] = strings.TrimSpace(parts[i])
				}
				if len(parts) > 0 {
					model = parts[0]
				}
				if len(parts) > 1 {
					if mb, convErr := strconv.ParseUint(parts[1], 10, 64); convErr == nil {
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
		}
	}

	// 2) AMD via rocm-smi (Linux)
	if p, err := exec.LookPath("rocm-smi"); err == nil {
		out, err := exec.Command(p, "--showproductname", "--csv").CombinedOutput()
		if err == nil {
			for _, line := range strings.Split(string(out), "\n") {
				line = strings.TrimSpace(line)
				if line == "" || strings.HasPrefix(line, "device") || strings.HasPrefix(line, "=") {
					continue
				}
				// CSV format: device,card_series,...
				parts := strings.SplitN(line, ",", 3)
				if len(parts) >= 2 {
					model = strings.TrimSpace(parts[1])
					break
				}
			}
		}
		if model != "" {
			// Try to get VRAM
			out2, err2 := exec.Command(p, "--showmeminfo", "vram", "--csv").CombinedOutput()
			if err2 == nil {
				for _, line := range strings.Split(string(out2), "\n") {
					if strings.Contains(line, "Total") {
						fields := strings.Split(line, ",")
						for _, f := range fields {
							f = strings.TrimSpace(f)
							if v, convErr := strconv.ParseUint(f, 10, 64); convErr == nil && v > 1000 {
								vramBytes = v
								break
							}
						}
					}
				}
			}
			sensors = "amd-rocm model:" + model
			return model, vramBytes, sensors
		}
	}

	// 3) Linux fallback: lspci for AMD/other GPUs without ROCm
	if runtime.GOOS == "linux" {
		if m := detectGPULspci(); m != "" {
			return m, 0, "lspci model:" + m
		}
	}

	// 4) Fallback: WMI on Windows (catches AMD, Intel, any GPU)
	if runtime.GOOS == "windows" {
		m, vram := detectGPUWindows()
		if m != "" {
			return m, vram, "wmi model:" + m
		}
	}

	return "", 0, ""
}

// detectGPULspci uses lspci to find discrete GPUs on Linux when
// neither nvidia-smi nor rocm-smi is available.
func detectGPULspci() string {
	p, err := exec.LookPath("lspci")
	if err != nil {
		return ""
	}
	out, err := exec.Command(p, "-mm").CombinedOutput()
	if err != nil {
		return ""
	}
	// lspci -mm outputs quoted fields: Slot "Class" "Vendor" "Device" ...
	// VGA class = "0300", 3D controller = "0302"
	for _, line := range strings.Split(string(out), "\n") {
		lower := strings.ToLower(line)
		if !strings.Contains(lower, "0300") && !strings.Contains(lower, "0302") {
			continue
		}
		// Skip integrated Intel GPUs
		if strings.Contains(lower, "intel") {
			continue
		}
		// Extract device name from quoted fields
		parts := splitQuoted(line)
		if len(parts) >= 4 {
			vendor := parts[2]
			device := parts[3]
			name := strings.TrimSpace(vendor + " " + device)
			if name != "" {
				return name
			}
		}
	}
	return ""
}

// splitQuoted splits a line by whitespace but keeps quoted strings together.
func splitQuoted(s string) []string {
	var parts []string
	var cur strings.Builder
	inQuote := false
	for _, r := range s {
		switch {
		case r == '"':
			inQuote = !inQuote
		case (r == ' ' || r == '\t') && !inQuote:
			if cur.Len() > 0 {
				parts = append(parts, cur.String())
				cur.Reset()
			}
		default:
			cur.WriteRune(r)
		}
	}
	if cur.Len() > 0 {
		parts = append(parts, cur.String())
	}
	return parts
}

// detectGPUWindows uses WMI to detect any discrete GPU on Windows.
// Note: WMI's AdapterRAM is uint32 and overflows to 0 for GPUs with
// VRAM that is a multiple of 4 GB (8, 12, 16, 24 GB). We select the
// first discrete GPU found and use VRAM only as a tiebreaker.
func detectGPUWindows() (model string, vramBytes uint64) {
	out, err := exec.Command("wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM", "/format:csv").CombinedOutput()
	if err != nil {
		return "", 0
	}
	// CSV: Node,AdapterRAM,Name
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(strings.ToLower(line), "node") {
			continue
		}
		fields := strings.Split(line, ",")
		if len(fields) < 3 {
			continue
		}
		name := strings.TrimSpace(fields[len(fields)-1])
		ramStr := strings.TrimSpace(fields[len(fields)-2])
		// Skip integrated GPUs
		nameLower := strings.ToLower(name)
		if strings.Contains(nameLower, "intel") && (strings.Contains(nameLower, "uhd") || strings.Contains(nameLower, "hd graphics") || strings.Contains(nameLower, "iris")) {
			continue
		}
		if strings.Contains(nameLower, "microsoft basic") || strings.Contains(nameLower, "remote desktop") {
			continue
		}
		if name == "" {
			continue
		}
		ram, _ := strconv.ParseUint(ramStr, 10, 64)
		// Pick first discrete GPU, prefer one with higher reported VRAM
		if model == "" || ram > vramBytes {
			model = name
			vramBytes = ram
		}
	}
	return model, vramBytes
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
