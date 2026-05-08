package hardware

import (
	"bytes"
	"context"
	"encoding/csv"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

func DetectInventory(modelCacheDir string) CapacityInventory {
	return BuildInventory(Detector{ModelCacheDir: modelCacheDir})
}

func BuildInventory(detector Detector) CapacityInventory {
	detector = detector.withDefaults()
	inventory := CapacityInventory{
		OS:           detector.GOOS,
		Arch:         detector.GOARCH,
		GPUVendor:    GPUVendorUnknown,
		GPUName:      "unknown",
		PowerProfile: PowerProfileUnknown,
		ThermalRisk:  ThermalRiskUnknown,
	}
	if detector.CPULogicalCores != nil {
		inventory.CPULogicalCores = detector.CPULogicalCores()
	}
	if inventory.CPULogicalCores < 0 {
		inventory.CPULogicalCores = 0
	}

	switch detector.GOOS {
	case "darwin":
		applyDarwinInventory(&inventory, detector)
	case "windows":
		applyWindowsInventory(&inventory, detector)
	case "linux":
		applyLinuxInventory(&inventory, detector)
	default:
		inventory.SystemRAMBytes, inventory.AvailableRAMBytes = detectProcMeminfo(detector)
	}
	inventory.VulkanAvailable = detectVulkanAvailable(detector)

	if strings.TrimSpace(detector.ModelCacheDir) != "" && detector.DiskFreeBytes != nil {
		if free, err := detector.DiskFreeBytes(detector.ModelCacheDir); err == nil {
			inventory.DiskFreeBytesModelCache = free
		}
	}
	return NormalizeInventory(inventory)
}

func (detector Detector) withDefaults() Detector {
	if strings.TrimSpace(detector.GOOS) == "" {
		detector.GOOS = runtime.GOOS
	}
	if strings.TrimSpace(detector.GOARCH) == "" {
		detector.GOARCH = runtime.GOARCH
	}
	if detector.CommandTimeout <= 0 {
		detector.CommandTimeout = defaultProbeTimeout
	}
	if detector.RunCommand == nil {
		detector.RunCommand = runCommand
	}
	if detector.LookPath == nil {
		detector.LookPath = exec.LookPath
	}
	if detector.ReadFile == nil {
		detector.ReadFile = os.ReadFile
	}
	if detector.ReadDirNames == nil {
		detector.ReadDirNames = readDirNames
	}
	if detector.Stat == nil {
		detector.Stat = os.Stat
	}
	if detector.DiskFreeBytes == nil {
		detector.DiskFreeBytes = defaultDiskFreeBytes
	}
	if detector.CPULogicalCores == nil {
		detector.CPULogicalCores = runtime.NumCPU
	}
	return detector
}

func runCommand(name string, args []string, timeout time.Duration) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	out, err := cmd.Output()
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	return out, err
}

func readDirNames(path string) ([]string, error) {
	entries, err := os.ReadDir(path)
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		names = append(names, entry.Name())
	}
	return names, nil
}

func applyDarwinInventory(inventory *CapacityInventory, detector Detector) {
	inventory.CPULogicalCores = firstPositiveInt(int(parseUintCommand(detector, "sysctl", "-n", "hw.logicalcpu")), inventory.CPULogicalCores)
	inventory.SystemRAMBytes = parseUintCommand(detector, "sysctl", "-n", "hw.memsize")
	if data, err := detector.RunCommand("vm_stat", nil, detector.CommandTimeout); err == nil {
		inventory.AvailableRAMBytes = parseDarwinVMStatAvailable(data)
	}
	if data, err := detector.RunCommand("system_profiler", []string{"SPDisplaysDataType", "SPHardwareDataType"}, 4*time.Second); err == nil {
		applyDarwinSystemProfiler(inventory, data)
	}
	if inventory.GPUDetected && inventory.GPUVendor == GPUVendorApple && inventory.GPUVRAMBytes == 0 {
		inventory.UnifiedMemory = true
	}
	if inventory.GPUVendor == GPUVendorApple {
		inventory.MetalAvailable = true
	}
}

func applyWindowsInventory(inventory *CapacityInventory, detector Detector) {
	total, available := detectWindowsMemory(detector)
	inventory.SystemRAMBytes = total
	inventory.AvailableRAMBytes = available
	if gpu := detectNVIDIAGPU(detector); gpu.name != "" {
		applyGPU(inventory, gpu)
	} else if gpu := detectWindowsWMIGPU(detector); gpu.name != "" {
		applyGPU(inventory, gpu)
	}
	inventory.PowerProfile = detectWindowsPowerProfile(detector)
}

func applyLinuxInventory(inventory *CapacityInventory, detector Detector) {
	total, available := detectProcMeminfo(detector)
	inventory.SystemRAMBytes = total
	inventory.AvailableRAMBytes = available
	if gpu := detectNVIDIAGPU(detector); gpu.name != "" {
		applyGPU(inventory, gpu)
	} else if gpu := detectLinuxLspciGPU(detector); gpu.name != "" {
		applyGPU(inventory, gpu)
	}
	inventory.PowerProfile = detectLinuxPowerProfile(detector)
}

type gpuInfo struct {
	name        string
	vendor      string
	vramBytes   uint64
	cuda        bool
	metal       bool
	unified     bool
	thermalRisk string
}

func applyGPU(inventory *CapacityInventory, gpu gpuInfo) {
	inventory.GPUDetected = true
	inventory.GPUName = gpu.name
	inventory.GPUVendor = firstNonEmpty(gpu.vendor, inferGPUVendor(gpu.name))
	inventory.GPUVRAMBytes = gpu.vramBytes
	inventory.CUDAAvailable = gpu.cuda
	inventory.MetalAvailable = gpu.metal
	inventory.UnifiedMemory = gpu.unified
	if gpu.thermalRisk != "" {
		inventory.ThermalRisk = gpu.thermalRisk
	}
}

func parseUintCommand(detector Detector, name string, args ...string) uint64 {
	data, err := detector.RunCommand(name, args, detector.CommandTimeout)
	if err != nil {
		return 0
	}
	value, _ := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
	return value
}

func firstPositiveInt(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

func parseDarwinVMStatAvailable(data []byte) uint64 {
	pageSize := uint64(16384)
	lines := strings.Split(string(data), "\n")
	if len(lines) > 0 {
		if idx := strings.Index(lines[0], "page size of "); idx >= 0 {
			rest := lines[0][idx+len("page size of "):]
			fields := strings.Fields(rest)
			if len(fields) > 0 {
				if parsed, err := strconv.ParseUint(strings.Trim(fields[0], "."), 10, 64); err == nil && parsed > 0 {
					pageSize = parsed
				}
			}
		}
	}
	var pages uint64
	for _, line := range lines {
		line = strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(line, "Pages free:"):
			pages += parseDarwinVMStatPages(line, "Pages free:")
		case strings.HasPrefix(line, "Pages inactive:"):
			pages += parseDarwinVMStatPages(line, "Pages inactive:")
		case strings.HasPrefix(line, "Pages speculative:"):
			pages += parseDarwinVMStatPages(line, "Pages speculative:")
		}
	}
	return pages * pageSize
}

func parseDarwinVMStatPages(line, prefix string) uint64 {
	value := strings.TrimSpace(strings.TrimPrefix(line, prefix))
	value = strings.TrimSuffix(value, ".")
	parsed, _ := strconv.ParseUint(value, 10, 64)
	return parsed
}

func applyDarwinSystemProfiler(inventory *CapacityInventory, data []byte) {
	lines := strings.Split(string(data), "\n")
	modelName := ""
	chipName := ""
	metal := false
	for _, line := range lines {
		line = strings.TrimSpace(line)
		lower := strings.ToLower(line)
		switch {
		case strings.HasPrefix(line, "Chipset Model:"):
			modelName = cleanName(strings.TrimSpace(strings.TrimPrefix(line, "Chipset Model:")))
		case strings.HasPrefix(line, "Chip:"):
			chipName = cleanName(strings.TrimSpace(strings.TrimPrefix(line, "Chip:")))
		case strings.HasPrefix(line, "Model Name:"):
			inventory.PowerProfile = powerProfileFromDarwinModel(strings.TrimSpace(strings.TrimPrefix(line, "Model Name:")))
		case strings.HasPrefix(line, "Metal Support:") && !strings.Contains(lower, "unsupported"):
			metal = true
		}
	}
	gpuName := firstNonEmpty(modelName, chipName)
	if gpuName != "" {
		inventory.GPUDetected = true
		inventory.GPUName = gpuName
		inventory.GPUVendor = inferGPUVendor(gpuName)
		if inventory.GPUVendor == GPUVendorApple {
			inventory.UnifiedMemory = true
		}
	}
	inventory.MetalAvailable = metal || (inventory.GPUDetected && inventory.GPUVendor == GPUVendorApple)
}

func powerProfileFromDarwinModel(model string) string {
	lower := strings.ToLower(model)
	switch {
	case strings.Contains(lower, "macbook"):
		return PowerProfileLaptop
	case strings.Contains(lower, "mac pro"), strings.Contains(lower, "mac mini"), strings.Contains(lower, "mac studio"), strings.Contains(lower, "imac"):
		return PowerProfileDesktop
	default:
		return PowerProfileUnknown
	}
}

func detectProcMeminfo(detector Detector) (total, available uint64) {
	data, err := detector.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		value, _ := strconv.ParseUint(fields[1], 10, 64)
		switch fields[0] {
		case "MemTotal:":
			total = value * 1024
		case "MemAvailable:":
			available = value * 1024
		}
	}
	return total, available
}

func detectNVIDIAGPU(detector Detector) gpuInfo {
	path, err := detector.LookPath("nvidia-smi")
	if err != nil || strings.TrimSpace(path) == "" {
		return gpuInfo{}
	}
	data, err := detector.RunCommand(path, []string{"--query-gpu=name,memory.total,temperature.gpu", "--format=csv,noheader,nounits"}, detector.CommandTimeout)
	if err != nil {
		return gpuInfo{}
	}
	line := firstNonEmpty(strings.Split(strings.TrimSpace(string(data)), "\n")...)
	if line == "" {
		return gpuInfo{}
	}
	fields := splitCSVLine(line)
	if len(fields) == 0 {
		return gpuInfo{}
	}
	vramBytes := uint64(0)
	if len(fields) > 1 {
		if mb, err := strconv.ParseUint(strings.TrimSpace(fields[1]), 10, 64); err == nil {
			vramBytes = mb * 1024 * 1024
		}
	}
	thermalRisk := ThermalRiskUnknown
	if len(fields) > 2 {
		if temp, err := strconv.ParseFloat(strings.TrimSpace(fields[2]), 64); err == nil {
			thermalRisk = thermalRiskFromCelsius(temp)
		}
	}
	return gpuInfo{
		name:        cleanName(fields[0]),
		vendor:      GPUVendorNVIDIA,
		vramBytes:   vramBytes,
		cuda:        true,
		thermalRisk: thermalRisk,
	}
}

func detectVulkanAvailable(detector Detector) bool {
	if detector.LookPath == nil {
		return false
	}
	for _, name := range executableNameVariants(detector.GOOS, "vulkaninfo") {
		if path, err := detector.LookPath(name); err == nil && strings.TrimSpace(path) != "" {
			return true
		}
	}
	return false
}

func executableNameVariants(goos, name string) []string {
	name = strings.TrimSpace(name)
	if name == "" {
		return nil
	}
	variants := []string{name}
	if strings.EqualFold(strings.TrimSpace(goos), "windows") && !strings.HasSuffix(strings.ToLower(name), ".exe") {
		variants = append(variants, name+".exe")
	}
	return variants
}

func detectLinuxLspciGPU(detector Detector) gpuInfo {
	path, err := detector.LookPath("lspci")
	if err != nil || strings.TrimSpace(path) == "" {
		return gpuInfo{}
	}
	data, err := detector.RunCommand(path, []string{"-mm"}, detector.CommandTimeout)
	if err != nil {
		return gpuInfo{}
	}
	for _, line := range strings.Split(string(data), "\n") {
		lower := strings.ToLower(line)
		if !strings.Contains(lower, "0300") && !strings.Contains(lower, "0302") &&
			!strings.Contains(lower, "vga") && !strings.Contains(lower, "3d controller") {
			continue
		}
		if strings.Contains(lower, "intel") {
			continue
		}
		parts := splitQuoted(line)
		if len(parts) >= 4 {
			name := cleanName(parts[2] + " " + parts[3])
			if name != "" {
				return gpuInfo{name: name, vendor: inferGPUVendor(name)}
			}
		}
	}
	return gpuInfo{}
}

func detectLinuxPowerProfile(detector Detector) string {
	if names, err := detector.ReadDirNames("/sys/class/power_supply"); err == nil {
		for _, name := range names {
			if strings.HasPrefix(strings.ToUpper(strings.TrimSpace(name)), "BAT") {
				return PowerProfileLaptop
			}
		}
	}
	if data, err := detector.ReadFile("/sys/class/dmi/id/chassis_type"); err == nil {
		return powerProfileFromChassisType(string(data))
	}
	return PowerProfileUnknown
}

func detectWindowsMemory(detector Detector) (total, available uint64) {
	data, err := detector.RunCommand("wmic", []string{"OS", "get", "TotalVisibleMemorySize,FreePhysicalMemory", "/format:csv"}, detector.CommandTimeout)
	if err != nil {
		return 0, 0
	}
	records := csvRecords(data)
	header, rows := headerAndRows(records)
	freeIdx := headerIndex(header, "freephysicalmemory")
	totalIdx := headerIndex(header, "totalvisiblememorysize")
	if freeIdx < 0 || totalIdx < 0 {
		return 0, 0
	}
	for _, row := range rows {
		if len(row) <= freeIdx || len(row) <= totalIdx {
			continue
		}
		freeKB, _ := strconv.ParseUint(strings.TrimSpace(row[freeIdx]), 10, 64)
		totalKB, _ := strconv.ParseUint(strings.TrimSpace(row[totalIdx]), 10, 64)
		if totalKB > 0 {
			return totalKB * 1024, freeKB * 1024
		}
	}
	return 0, 0
}

func detectWindowsWMIGPU(detector Detector) gpuInfo {
	data, err := detector.RunCommand("wmic", []string{"path", "win32_VideoController", "get", "Name,AdapterRAM", "/format:csv"}, detector.CommandTimeout)
	if err != nil {
		return gpuInfo{}
	}
	records := csvRecords(data)
	header, rows := headerAndRows(records)
	nameIdx := headerIndex(header, "name")
	ramIdx := headerIndex(header, "adapterram")
	if nameIdx < 0 {
		return gpuInfo{}
	}
	var best gpuInfo
	for _, row := range rows {
		if len(row) <= nameIdx {
			continue
		}
		name := cleanName(row[nameIdx])
		if name == "" || ignoredGPUName(name) {
			continue
		}
		vramBytes := uint64(0)
		if ramIdx >= 0 && len(row) > ramIdx {
			vramBytes, _ = strconv.ParseUint(strings.TrimSpace(row[ramIdx]), 10, 64)
		}
		candidate := gpuInfo{name: name, vendor: inferGPUVendor(name), vramBytes: vramBytes}
		if best.name == "" || candidate.vramBytes > best.vramBytes || gpuTier(candidate.name) > gpuTier(best.name) {
			best = candidate
		}
	}
	return best
}

func detectWindowsPowerProfile(detector Detector) string {
	data, err := detector.RunCommand("wmic", []string{"SystemEnclosure", "get", "ChassisTypes", "/format:csv"}, detector.CommandTimeout)
	if err != nil {
		return PowerProfileUnknown
	}
	return powerProfileFromChassisType(string(data))
}

func csvRecords(data []byte) [][]string {
	reader := csv.NewReader(bytes.NewReader(data))
	reader.FieldsPerRecord = -1
	records, err := reader.ReadAll()
	if err != nil {
		return nil
	}
	out := make([][]string, 0, len(records))
	for _, record := range records {
		empty := true
		for i := range record {
			record[i] = strings.TrimSpace(record[i])
			if record[i] != "" {
				empty = false
			}
		}
		if !empty {
			out = append(out, record)
		}
	}
	return out
}

func headerAndRows(records [][]string) ([]string, [][]string) {
	if len(records) == 0 {
		return nil, nil
	}
	header := records[0]
	return header, records[1:]
}

func headerIndex(header []string, want string) int {
	want = strings.ToLower(strings.TrimSpace(want))
	for i, value := range header {
		if strings.ToLower(strings.TrimSpace(value)) == want {
			return i
		}
	}
	return -1
}

func powerProfileFromChassisType(value string) string {
	for _, field := range strings.FieldsFunc(value, func(r rune) bool {
		return r < '0' || r > '9'
	}) {
		code, err := strconv.Atoi(field)
		if err != nil {
			continue
		}
		switch code {
		case 8, 9, 10, 14, 30, 31, 32:
			return PowerProfileLaptop
		case 17, 23:
			return PowerProfileServer
		case 3, 4, 5, 6, 7, 15, 16:
			return PowerProfileDesktop
		}
	}
	return PowerProfileUnknown
}

func thermalRiskFromCelsius(temp float64) string {
	switch {
	case temp <= 0:
		return ThermalRiskUnknown
	case temp < 75:
		return ThermalRiskLow
	case temp < 88:
		return ThermalRiskMedium
	default:
		return ThermalRiskHigh
	}
}

func splitCSVLine(line string) []string {
	reader := csv.NewReader(strings.NewReader(line))
	reader.FieldsPerRecord = -1
	records, err := reader.Read()
	if err != nil {
		return strings.Split(line, ",")
	}
	for i := range records {
		records[i] = strings.TrimSpace(records[i])
	}
	return records
}

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

func ignoredGPUName(name string) bool {
	lower := strings.ToLower(name)
	return strings.Contains(lower, "microsoft basic") ||
		strings.Contains(lower, "remote desktop") ||
		strings.Contains(lower, "vmware") ||
		strings.Contains(lower, "virtualbox") ||
		strings.Contains(lower, "parallels") ||
		strings.Contains(lower, "hyper-v")
}

func gpuTier(name string) int {
	lower := strings.ToLower(name)
	switch {
	case strings.Contains(lower, "nvidia") || strings.Contains(lower, "geforce") || strings.Contains(lower, "rtx") || strings.Contains(lower, "gtx"):
		return 3
	case strings.Contains(lower, "amd") || strings.Contains(lower, "radeon"):
		return 2
	case strings.Contains(lower, "apple"):
		return 2
	case strings.Contains(lower, "intel"):
		return 1
	default:
		return 0
	}
}

func existingPath(path string, stat func(string) (os.FileInfo, error)) (string, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return "", errors.New("empty path")
	}
	path = filepath.Clean(path)
	for {
		if _, err := stat(path); err == nil {
			return path, nil
		}
		parent := filepath.Dir(path)
		if parent == path || parent == "." {
			return "", os.ErrNotExist
		}
		path = parent
	}
}
