package hardware

import (
	"os"
	"strings"
	"time"
	"unicode"
)

const (
	GPUVendorApple   = "apple"
	GPUVendorNVIDIA  = "nvidia"
	GPUVendorAMD     = "amd"
	GPUVendorUnknown = "unknown"

	PowerProfileLaptop  = "laptop"
	PowerProfileDesktop = "desktop"
	PowerProfileServer  = "server"
	PowerProfileUnknown = "unknown"

	ThermalRiskUnknown = "unknown"
	ThermalRiskLow     = "low"
	ThermalRiskMedium  = "medium"
	ThermalRiskHigh    = "high"
)

const (
	defaultProbeTimeout = 2 * time.Second
	maxNameRunes        = 128
)

type CapacityInventory struct {
	OS                      string `json:"os"`
	Arch                    string `json:"arch"`
	CPULogicalCores         int    `json:"cpu_logical_cores"`
	SystemRAMBytes          uint64 `json:"system_ram_bytes"`
	AvailableRAMBytes       uint64 `json:"available_ram_bytes"`
	GPUDetected             bool   `json:"gpu_detected"`
	GPUVendor               string `json:"gpu_vendor"`
	GPUName                 string `json:"gpu_name"`
	GPUVRAMBytes            uint64 `json:"gpu_vram_bytes"`
	UnifiedMemory           bool   `json:"unified_memory"`
	MetalAvailable          bool   `json:"metal_available"`
	CUDAAvailable           bool   `json:"cuda_available"`
	VulkanAvailable         bool   `json:"vulkan_available"`
	DiskFreeBytesModelCache uint64 `json:"disk_free_bytes_model_cache"`
	PowerProfile            string `json:"power_profile"`
	ThermalRisk             string `json:"thermal_risk"`
}

type CommandRunner func(name string, args []string, timeout time.Duration) ([]byte, error)

type Detector struct {
	GOOS            string
	GOARCH          string
	ModelCacheDir   string
	CommandTimeout  time.Duration
	RunCommand      CommandRunner
	LookPath        func(string) (string, error)
	ReadFile        func(string) ([]byte, error)
	ReadDirNames    func(string) ([]string, error)
	Stat            func(string) (os.FileInfo, error)
	DiskFreeBytes   func(string) (uint64, error)
	CPULogicalCores func() int
}

func NormalizeInventory(inventory CapacityInventory) CapacityInventory {
	inventory.OS = cleanName(strings.ToLower(strings.TrimSpace(inventory.OS)))
	inventory.Arch = cleanName(strings.ToLower(strings.TrimSpace(inventory.Arch)))
	if inventory.CPULogicalCores < 0 {
		inventory.CPULogicalCores = 0
	}
	inventory.GPUName = cleanName(inventory.GPUName)
	inventory.GPUVendor = normalizeGPUVendor(firstNonEmpty(inventory.GPUVendor, inferGPUVendor(inventory.GPUName)))
	if !inventory.GPUDetected && inventory.GPUName != "" {
		inventory.GPUDetected = true
	}
	if !inventory.GPUDetected {
		inventory.GPUVendor = GPUVendorUnknown
		inventory.GPUName = "unknown"
		inventory.GPUVRAMBytes = 0
		inventory.MetalAvailable = false
		inventory.CUDAAvailable = false
		inventory.VulkanAvailable = false
	}
	if inventory.GPUName == "" {
		inventory.GPUName = "unknown"
	}
	inventory.PowerProfile = normalizePowerProfile(inventory.PowerProfile)
	inventory.ThermalRisk = normalizeThermalRisk(inventory.ThermalRisk)
	return inventory
}

func normalizeGPUVendor(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case GPUVendorApple:
		return GPUVendorApple
	case GPUVendorNVIDIA:
		return GPUVendorNVIDIA
	case GPUVendorAMD:
		return GPUVendorAMD
	default:
		return GPUVendorUnknown
	}
}

func normalizePowerProfile(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case PowerProfileLaptop:
		return PowerProfileLaptop
	case PowerProfileDesktop:
		return PowerProfileDesktop
	case PowerProfileServer:
		return PowerProfileServer
	default:
		return PowerProfileUnknown
	}
}

func normalizeThermalRisk(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case ThermalRiskLow:
		return ThermalRiskLow
	case ThermalRiskMedium:
		return ThermalRiskMedium
	case ThermalRiskHigh:
		return ThermalRiskHigh
	default:
		return ThermalRiskUnknown
	}
}

func inferGPUVendor(name string) string {
	lower := strings.ToLower(strings.TrimSpace(name))
	switch {
	case strings.Contains(lower, "apple"):
		return GPUVendorApple
	case strings.Contains(lower, "nvidia") || strings.Contains(lower, "geforce") || strings.Contains(lower, "rtx") || strings.Contains(lower, "gtx"):
		return GPUVendorNVIDIA
	case strings.Contains(lower, "amd") || strings.Contains(lower, "radeon"):
		return GPUVendorAMD
	default:
		return GPUVendorUnknown
	}
}

func cleanName(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	var b strings.Builder
	b.Grow(len(value))
	written := 0
	for _, r := range value {
		if unicode.IsControl(r) {
			continue
		}
		if written >= maxNameRunes {
			break
		}
		b.WriteRune(r)
		written++
	}
	return strings.TrimSpace(b.String())
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			return value
		}
	}
	return ""
}
