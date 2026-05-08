package hardware

import (
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
	"time"
)

func TestBuildInventoryMockedDarwin(t *testing.T) {
	t.Parallel()

	detector := Detector{
		GOOS:          "darwin",
		GOARCH:        "arm64",
		ModelCacheDir: "/Users/test/.ryvion/models",
		CPULogicalCores: func() int {
			return 1
		},
		RunCommand: func(name string, args []string, _ time.Duration) ([]byte, error) {
			switch commandKey(name, args) {
			case "sysctl -n hw.logicalcpu":
				return []byte("10\n"), nil
			case "sysctl -n hw.memsize":
				return []byte("25769803776\n"), nil
			case "vm_stat":
				return []byte("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 100.\nPages inactive: 200.\nPages speculative: 50.\n"), nil
			case "system_profiler SPDisplaysDataType SPHardwareDataType":
				return []byte(`
Hardware:

    Hardware Overview:
      Model Name: MacBook Pro
      Chip: Apple M4 Pro
      Serial Number (system): should-not-appear

Graphics/Displays:

    Apple M4 Pro:
      Chipset Model: Apple M4 Pro
      Type: GPU
      Bus: Built-In
      Metal Support: Metal 3
`), nil
			default:
				return nil, os.ErrNotExist
			}
		},
		DiskFreeBytes: func(path string) (uint64, error) {
			if path != "/Users/test/.ryvion/models" {
				t.Fatalf("disk path = %q", path)
			}
			return 99 * 1024 * 1024 * 1024, nil
		},
	}

	inventory := BuildInventory(detector)
	if inventory.OS != "darwin" || inventory.Arch != "arm64" {
		t.Fatalf("identity = %+v, want darwin/arm64", inventory)
	}
	if inventory.CPULogicalCores != 10 {
		t.Fatalf("cpu_logical_cores = %d, want 10", inventory.CPULogicalCores)
	}
	if inventory.SystemRAMBytes != 25769803776 {
		t.Fatalf("system_ram_bytes = %d", inventory.SystemRAMBytes)
	}
	if inventory.AvailableRAMBytes != (100+200+50)*16384 {
		t.Fatalf("available_ram_bytes = %d", inventory.AvailableRAMBytes)
	}
	if !inventory.GPUDetected || inventory.GPUVendor != GPUVendorApple || inventory.GPUName != "Apple M4 Pro" {
		t.Fatalf("gpu = %+v, want Apple M4 Pro", inventory)
	}
	if !inventory.UnifiedMemory || !inventory.MetalAvailable || inventory.CUDAAvailable || inventory.GPUVRAMBytes != 0 {
		t.Fatalf("accelerator flags = %+v, want unified Metal and no CUDA/dedicated VRAM", inventory)
	}
	if inventory.PowerProfile != PowerProfileLaptop || inventory.ThermalRisk != ThermalRiskUnknown {
		t.Fatalf("power/thermal = %q/%q", inventory.PowerProfile, inventory.ThermalRisk)
	}
	if inventory.DiskFreeBytesModelCache != 99*1024*1024*1024 {
		t.Fatalf("disk_free_bytes_model_cache = %d", inventory.DiskFreeBytesModelCache)
	}
}

func TestBuildInventoryMockedWindows(t *testing.T) {
	t.Parallel()

	detector := Detector{
		GOOS:          "windows",
		GOARCH:        "amd64",
		ModelCacheDir: `C:\ryvion\models`,
		CPULogicalCores: func() int {
			return 24
		},
		LookPath: func(name string) (string, error) {
			if name == "nvidia-smi" {
				return `C:\Windows\System32\nvidia-smi.exe`, nil
			}
			return "", os.ErrNotExist
		},
		RunCommand: func(name string, args []string, _ time.Duration) ([]byte, error) {
			if strings.Contains(strings.ToLower(name), "nvidia-smi") {
				return []byte("NVIDIA GeForce RTX 4070 Ti SUPER, 16384, 68, 8.9\n"), nil
			}
			switch commandKey(name, args) {
			case "wmic CPU get Name /format:csv":
				return []byte("Node,Name\nDESKTOP,AMD Ryzen 9 7900X\n"), nil
			case "wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /format:csv":
				return []byte("Node,FreePhysicalMemory,TotalVisibleMemorySize\nDESKTOP,12582912,33554432\n"), nil
			case "wmic SystemEnclosure get ChassisTypes /format:csv":
				return []byte("Node,ChassisTypes\nDESKTOP,{3}\n"), nil
			default:
				return nil, os.ErrNotExist
			}
		},
		DiskFreeBytes: func(path string) (uint64, error) {
			if path != `C:\ryvion\models` {
				t.Fatalf("disk path = %q", path)
			}
			return 512 * 1024 * 1024 * 1024, nil
		},
		Stat: func(path string) (os.FileInfo, error) {
			if path == `C:\Windows\System32\DirectML.dll` {
				return fakeHardwareFileInfo{name: "DirectML.dll"}, nil
			}
			return nil, os.ErrNotExist
		},
	}

	inventory := BuildInventory(detector)
	if inventory.OS != "windows" || inventory.Arch != "amd64" {
		t.Fatalf("identity = %+v, want windows/amd64", inventory)
	}
	if inventory.CPULogicalCores != 24 {
		t.Fatalf("cpu_logical_cores = %d, want 24", inventory.CPULogicalCores)
	}
	if inventory.CPUName != "AMD Ryzen 9 7900X" {
		t.Fatalf("cpu_name = %q", inventory.CPUName)
	}
	if inventory.SystemRAMBytes != 33554432*1024 || inventory.AvailableRAMBytes != 12582912*1024 {
		t.Fatalf("memory = %+v", inventory)
	}
	if !inventory.GPUDetected || inventory.GPUVendor != GPUVendorNVIDIA || inventory.GPUName != "NVIDIA GeForce RTX 4070 Ti SUPER" {
		t.Fatalf("gpu = %+v, want RTX 4070 Ti SUPER", inventory)
	}
	if inventory.GPUVRAMBytes != 16384*1024*1024 || !inventory.CUDAAvailable || inventory.MetalAvailable || inventory.UnifiedMemory {
		t.Fatalf("gpu capacity flags = %+v", inventory)
	}
	if inventory.ComputeCapability != "8.9" || !inventory.DirectMLAvailable {
		t.Fatalf("windows acceleration fields = %+v", inventory)
	}
	if got := strings.Join(inventory.AccelerationHints, ","); !strings.Contains(got, "cpu") || !strings.Contains(got, "cuda") || !strings.Contains(got, "directml") {
		t.Fatalf("acceleration_hints = %q, want cpu/cuda/directml", got)
	}
	if inventory.PowerProfile != PowerProfileDesktop || inventory.ThermalRisk != ThermalRiskLow {
		t.Fatalf("power/thermal = %q/%q", inventory.PowerProfile, inventory.ThermalRisk)
	}
	if inventory.DiskFreeBytesModelCache != 512*1024*1024*1024 {
		t.Fatalf("disk_free_bytes_model_cache = %d", inventory.DiskFreeBytesModelCache)
	}
}

func TestBuildInventoryDoesNotExposeRawCommandOutput(t *testing.T) {
	t.Parallel()

	detector := Detector{
		GOOS:   "darwin",
		GOARCH: "arm64",
		RunCommand: func(name string, args []string, _ time.Duration) ([]byte, error) {
			switch commandKey(name, args) {
			case "sysctl -n hw.logicalcpu":
				return []byte("8\n"), nil
			case "sysctl -n hw.memsize":
				return []byte("17179869184\n"), nil
			case "vm_stat":
				return []byte("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 1.\n"), nil
			case "system_profiler SPDisplaysDataType SPHardwareDataType":
				return []byte(`
Hardware:
  Hardware Overview:
    Model Name: MacBook Air
    Serial Number (system): demo-key-secret
    Raw Prompt: should-not-appear
Graphics/Displays:
  Apple M4:
    Chipset Model: Apple M4
    Metal Support: Metal 3
`), nil
			default:
				return nil, errors.New("unexpected command")
			}
		},
		DiskFreeBytes: func(string) (uint64, error) {
			return 0, os.ErrNotExist
		},
	}

	inventory := BuildInventory(detector)
	raw, err := json.Marshal(inventory)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	body := strings.ToLower(string(raw))
	for _, forbidden := range []string{
		"demo-key-secret",
		"should-not-appear",
		"raw_prompt",
		"prompt_text",
		"output_text",
		"generated_text",
		"tensor_bytes",
	} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("hardware inventory leaked forbidden marker %q: %s", forbidden, raw)
		}
	}
}

func commandKey(name string, args []string) string {
	if len(args) == 0 {
		return name
	}
	return name + " " + strings.Join(args, " ")
}

type fakeHardwareFileInfo struct {
	name string
}

func (f fakeHardwareFileInfo) Name() string       { return f.name }
func (f fakeHardwareFileInfo) Size() int64        { return 1 }
func (f fakeHardwareFileInfo) Mode() os.FileMode  { return 0 }
func (f fakeHardwareFileInfo) ModTime() time.Time { return time.Unix(1, 0) }
func (f fakeHardwareFileInfo) IsDir() bool        { return false }
func (f fakeHardwareFileInfo) Sys() any           { return nil }
