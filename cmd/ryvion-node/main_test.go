package main

import (
	"flag"
	"io"
	"os"
	"testing"

	"github.com/Ryvion/ryvion-node/internal/hw"
)

func TestParseConfigUsesLegacyDeviceTypeEnv(t *testing.T) {
	withArgs(t, "ryvion-node")
	t.Setenv("RYV_DEVICE", "")
	t.Setenv("RYV_DEVICE_TYPE", "gpu")

	cfg := parseConfig()
	if cfg.Device != "gpu" {
		t.Fatalf("Device = %q, want legacy RYV_DEVICE_TYPE fallback", cfg.Device)
	}
}

func TestParseConfigPrefersDeviceEnv(t *testing.T) {
	withArgs(t, "ryvion-node")
	t.Setenv("RYV_DEVICE", "cpu")
	t.Setenv("RYV_DEVICE_TYPE", "gpu")

	cfg := parseConfig()
	if cfg.Device != "cpu" {
		t.Fatalf("Device = %q, want RYV_DEVICE to override RYV_DEVICE_TYPE", cfg.Device)
	}
}

func TestParseConfigDeviceFlagOverridesEnv(t *testing.T) {
	withArgs(t, "ryvion-node", "-device", "gpu")
	t.Setenv("RYV_DEVICE", "cpu")
	t.Setenv("RYV_DEVICE_TYPE", "")

	cfg := parseConfig()
	if cfg.Device != "gpu" {
		t.Fatalf("Device = %q, want -device flag", cfg.Device)
	}
}

func TestParseConfigTypeFlagAlias(t *testing.T) {
	withArgs(t, "ryvion-node", "-type", "gpu")
	t.Setenv("RYV_DEVICE", "")
	t.Setenv("RYV_DEVICE_TYPE", "")

	cfg := parseConfig()
	if cfg.Device != "gpu" {
		t.Fatalf("Device = %q, want deprecated -type alias", cfg.Device)
	}
}

func TestBuildHeartbeatPayloadDoesNotAdvertiseMissingOCI(t *testing.T) {
	payload, err := buildNodeHeartbeatPayload("node-test", zeroCaps(), "cpu", "", runtimeHealthSnapshot{Health: "missing"})
	if err != nil {
		t.Fatalf("buildNodeHeartbeatPayload() error = %v", err)
	}
	runtimeProfile := payload.CapabilityPassport.RuntimeProfile
	if runtimeProfile.OCIAvailable {
		t.Fatal("OCIAvailable = true, want false for missing runtime")
	}
	if len(runtimeProfile.SupportedRunnerKinds) != 0 {
		t.Fatalf("SupportedRunnerKinds = %v, want empty", runtimeProfile.SupportedRunnerKinds)
	}
}

func zeroCaps() hw.CapSet { return hw.CapSet{CPUCores: 1, RAMBytes: 1 << 30} }

func withArgs(t *testing.T, args ...string) {
	t.Helper()

	oldArgs := os.Args
	oldCommandLine := flag.CommandLine
	fs := flag.NewFlagSet(args[0], flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	flag.CommandLine = fs
	os.Args = args

	t.Cleanup(func() {
		os.Args = oldArgs
		flag.CommandLine = oldCommandLine
	})
}
