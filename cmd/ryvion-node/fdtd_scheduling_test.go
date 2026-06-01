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

// fdtdNativeReady is OS-aware: linux leads with gprMax (CUDA GPU + opt-in CPU
// lanes), darwin with Meep (opt-in CPU lane, asserted separately below). These
// NVIDIA/AMD GPU-lane paths only apply off darwin.
func TestFDTDNativeReadyPositive(t *testing.T) {
	if runtime.GOOS == "darwin" {
		t.Skip("gprMax GPU/AMD lanes do not apply on darwin; see TestFDTDNativeReadyDarwinMeep")
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

// TestFDTDNativeReadyDarwinMeep asserts the darwin Meep lane: opt-in only, RAM
// gated, gpuReady-independent (Apple Silicon has no CUDA; Meep solves on CPU).
func TestFDTDNativeReadyDarwinMeep(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("darwin Meep lane only applies on macOS")
	}
	t.Setenv("RYV_DISABLE_NATIVE_EM", "")
	t.Setenv("RYV_EM_ENGINE", "")
	mac := hw.CapSet{GPUModel: "Apple M4", RAMBytes: 16 * gb}

	t.Run("requires opt-in", func(t *testing.T) {
		t.Setenv("RYV_EM_ALLOW_CPU", "")
		if fdtdNativeReady(mac, false) || fdtdNativeReady(mac, true) {
			t.Fatal("darwin must not advertise EM without RYV_EM_ALLOW_CPU")
		}
	})
	t.Run("opt-in meep lane (gpuReady irrelevant)", func(t *testing.T) {
		t.Setenv("RYV_EM_ALLOW_CPU", "1")
		if !fdtdNativeReady(mac, false) {
			t.Fatal("darwin + RYV_EM_ALLOW_CPU + RAM headroom should advertise Meep EM regardless of gpuReady")
		}
		low := hw.CapSet{GPUModel: "Apple M4", RAMBytes: 4 * gb}
		if fdtdNativeReady(low, true) {
			t.Fatal("below RAM floor should not advertise EM")
		}
	})
	if eng := fdtdNativeEngine(); eng != "meep" {
		t.Fatalf("darwin fdtdNativeEngine=%q want meep", eng)
	}
}

// envFlagsClear guards the exclusion tests against an EM flag leaking in from the
// environment; t.Setenv-based subtests below set what they need.
func envFlagsClear(t *testing.T) bool {
	t.Helper()
	t.Setenv("RYV_DISABLE_NATIVE_EM", "")
	t.Setenv("RYV_EM_ALLOW_CPU", "")
	return true
}
