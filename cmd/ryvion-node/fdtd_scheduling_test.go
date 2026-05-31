package main

import (
	"runtime"
	"testing"

	"github.com/Ryvion/ryvion-node/internal/hw"
)

const gb = uint64(1) << 30

func TestEMGPUAccelerated(t *testing.T) {
	cases := []struct {
		name string
		caps hw.CapSet
		want bool
	}{
		{"nvidia rtx", hw.CapSet{GPUModel: "NVIDIA GeForce RTX 4070 Ti", VRAMBytes: 16 * gb}, true},
		{"nvidia tesla", hw.CapSet{GPUModel: "Tesla T4", VRAMBytes: 16 * gb}, true},
		{"nvidia datacenter h100", hw.CapSet{GPUModel: "NVIDIA H100 PCIe", VRAMBytes: 80 * gb}, true},
		// AMD reports a gfx arch via rocm-smi -> gprMax has no CUDA -> not accel.
		{"amd 7900 xtx gfx", hw.CapSet{GPUModel: "AMD Radeon RX 7900 XTX", GfxVersion: "gfx1100", VRAMBytes: 24 * gb}, false},
		{"amd model only", hw.CapSet{GPUModel: "Radeon RX 7900 XTX"}, false},
		{"empty", hw.CapSet{}, false},
	}
	for _, c := range cases {
		if got := emGPUAccelerated(c.caps); got != c.want {
			t.Errorf("%s: emGPUAccelerated=%v want %v", c.name, got, c.want)
		}
	}
}

func TestFDTDNativeReadyExclusions(t *testing.T) {
	nvidia := hw.CapSet{GPUModel: "NVIDIA GeForce RTX 4070 Ti", VRAMBytes: 16 * gb, RAMBytes: 32 * gb}
	amd := hw.CapSet{GPUModel: "AMD Radeon RX 7900 XTX", GfxVersion: "gfx1100", VRAMBytes: 24 * gb, RAMBytes: 32 * gb}

	// These hold on every OS.
	if !envFlagsClear(t) {
		return
	}
	t.Run("disabled flag", func(t *testing.T) {
		t.Setenv("RYV_DISABLE_NATIVE_EM", "1")
		if fdtdNativeReady(nvidia, true) {
			t.Fatal("RYV_DISABLE_NATIVE_EM must force EM off")
		}
	})
	t.Run("no gpu ready", func(t *testing.T) {
		if fdtdNativeReady(nvidia, false) {
			t.Fatal("not-ready GPU must not advertise EM")
		}
	})
	t.Run("amd without cpu opt-in", func(t *testing.T) {
		if fdtdNativeReady(amd, true) {
			t.Fatal("AMD must not advertise EM by default (gprMax is CUDA-only)")
		}
	})
}

// fdtdNativeReady is OS-aware (darwin excluded), so the positive GPU/CPU-lane
// paths are only asserted off darwin where they actually apply.
func TestFDTDNativeReadyPositive(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("EM is excluded on Apple/darwin for v1")
	}
	nvidiaBig := hw.CapSet{GPUModel: "NVIDIA GeForce RTX 4070 Ti", VRAMBytes: 16 * gb, RAMBytes: 32 * gb}
	nvidiaSmall := hw.CapSet{GPUModel: "NVIDIA GeForce RTX 3050", VRAMBytes: 4 * gb, RAMBytes: 16 * gb}
	amdBigRAM := hw.CapSet{GPUModel: "AMD Radeon RX 7900 XTX", GfxVersion: "gfx1100", VRAMBytes: 24 * gb, RAMBytes: 32 * gb}
	amdLowRAM := hw.CapSet{GPUModel: "AMD Radeon RX 7900 XTX", GfxVersion: "gfx1100", VRAMBytes: 24 * gb, RAMBytes: 4 * gb}

	t.Run("nvidia gpu lane", func(t *testing.T) {
		if !fdtdNativeReady(nvidiaBig, true) {
			t.Fatal("NVIDIA GPU with VRAM headroom should advertise EM")
		}
	})
	t.Run("nvidia below vram floor", func(t *testing.T) {
		if fdtdNativeReady(nvidiaSmall, true) {
			t.Fatal("below VRAM floor should not advertise GPU EM")
		}
	})
	t.Run("amd cpu lane opt-in", func(t *testing.T) {
		t.Setenv("RYV_EM_ALLOW_CPU", "1")
		if !fdtdNativeReady(amdBigRAM, true) {
			t.Fatal("AMD with RYV_EM_ALLOW_CPU=1 + RAM headroom should advertise CPU-lane EM")
		}
		if fdtdNativeReady(amdLowRAM, true) {
			t.Fatal("CPU lane below RAM floor should not advertise EM")
		}
	})
}

// envFlagsClear guards the exclusion tests against an EM flag leaking in from the
// environment; t.Setenv-based subtests below set what they need.
func envFlagsClear(t *testing.T) bool {
	t.Helper()
	t.Setenv("RYV_DISABLE_NATIVE_EM", "")
	t.Setenv("RYV_EM_ALLOW_CPU", "")
	return true
}
