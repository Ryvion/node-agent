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

func TestRuntimeHealthMissingDoesNotAdvertiseOCI(t *testing.T) {
	t.Setenv("RYV_RUNTIME_BINARY", "/definitely/not/present")
	t.Setenv("RYV_RUNTIME_BACKEND_BINARY", "/definitely/not/present")
	t.Setenv("RYV_RUNTIME_ENGINE_BINARY", "/definitely/not/present")
	t.Setenv("RYV_ALLOW_DOCKER_FALLBACK", "0")

	snapshot := detectRuntimeHealth(hw.CapSet{})
	if snapshot.OCIAvailable {
		t.Fatal("OCIAvailable = true, want false when no runtime is present")
	}
	if snapshot.Health != "missing" {
		t.Fatalf("Health = %q, want missing", snapshot.Health)
	}
	if len(snapshot.SupportedRunnerKinds) != 0 {
		t.Fatalf("SupportedRunnerKinds = %v, want empty", snapshot.SupportedRunnerKinds)
	}
}

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
