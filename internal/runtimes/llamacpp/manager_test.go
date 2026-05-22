package llamacpp

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	capshardware "github.com/Ryvion/ryvion-node/internal/capabilities/hardware"
	"github.com/Ryvion/ryvion-node/internal/runtimes/inventory"
)

func TestConfigFromEnv(t *testing.T) {
	t.Parallel()

	serverPath := filepath.Join(t.TempDir(), "llama-server")
	modelPath := filepath.Join(t.TempDir(), "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
	env := map[string]string{
		EnvEnabled:   "1",
		EnvServer:    serverPath,
		EnvModel:     modelPath,
		EnvHost:      "0.0.0.0",
		EnvPort:      "45911",
		EnvCtxSize:   "8192",
		EnvThreads:   "6",
		EnvGPULayers: "12",
		EnvExtraArgs: "--host 0.0.0.0 --parallel 4 --no-webui --model secret.gguf --cache-type-k q8_0 --batch-size nope",
	}

	cfg := ConfigFromEnvWith(ConfigSource{
		Getenv: func(name string) string {
			return env[name]
		},
	})

	if !cfg.Enabled {
		t.Fatalf("enabled = false, want true")
	}
	if cfg.ServerPath != serverPath || cfg.ModelPath != modelPath {
		t.Fatalf("paths = %q/%q, want explicit env paths", cfg.ServerPath, cfg.ModelPath)
	}
	if !cfg.ServerPathExplicit {
		t.Fatalf("server_path_explicit = false, want true for env path")
	}
	if cfg.Host != DefaultHost {
		t.Fatalf("host = %q, want safe default loopback", cfg.Host)
	}
	if cfg.Port != 45911 || cfg.ContextSize != 8192 || cfg.Threads != 6 || cfg.GPULayers != 12 {
		t.Fatalf("numeric config = %+v", cfg)
	}
	gotArgs := strings.Join(cfg.ExtraArgs, " ")
	if gotArgs != "--parallel 4 --no-webui --cache-type-k q8_0" {
		t.Fatalf("extra args = %q, want sanitized allowlist", gotArgs)
	}
}

func TestConfigFromEnvAllowsSlotSavePathExtraArg(t *testing.T) {
	t.Parallel()

	env := map[string]string{
		EnvEnabled:   "1",
		EnvModel:     filepath.Join(t.TempDir(), "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
		EnvExtraArgs: "--slot-save-path /tmp/ryvion-slots --slot-save-path ../unsafe --parallel 2",
	}
	cfg := ConfigFromEnvWith(ConfigSource{
		Getenv: func(name string) string {
			return env[name]
		},
	})
	gotArgs := strings.Join(cfg.ExtraArgs, " ")
	if gotArgs != "--slot-save-path /tmp/ryvion-slots --parallel 2" {
		t.Fatalf("extra args = %q, want safe slot save path only", gotArgs)
	}
}

func TestConfigFromEnvDefaultsNativeSidecarEnabled(t *testing.T) {
	t.Parallel()

	cfg := ConfigFromEnvWith(ConfigSource{
		Getenv: func(string) string {
			return ""
		},
	})
	if !cfg.Enabled {
		t.Fatal("enabled = false, want native llama.cpp enabled by default")
	}

	cfg = ConfigFromEnvWith(ConfigSource{
		Getenv: func(name string) string {
			if name == EnvEnabled {
				return "0"
			}
			return ""
		},
	})
	if cfg.Enabled {
		t.Fatal("enabled = true, want explicit RYV_LLAMA_CPP_ENABLED=0 to disable native llama.cpp")
	}
}

func TestConfigFromEnvDefaultsToGPUOffloadAndAllowsExplicitOptOut(t *testing.T) {
	t.Parallel()

	cfg := ConfigFromEnvWith(ConfigSource{
		Getenv: func(name string) string {
			return ""
		},
		LookPath: func(string) (string, error) {
			return "", os.ErrNotExist
		},
		Stat: func(string) (os.FileInfo, error) {
			return nil, os.ErrNotExist
		},
		GOOS: "linux",
	})
	if cfg.GPULayers != DefaultGPULayers {
		t.Fatalf("default gpu layers = %d, want %d", cfg.GPULayers, DefaultGPULayers)
	}
	if cfg.ServerPathExplicit {
		t.Fatalf("server_path_explicit = true, want false for discovered/default path")
	}
	if cfg.FastDefaults {
		t.Fatalf("fast defaults = true, want false without CUDA signal")
	}
	if cfg.DraftGPULayers != 0 {
		t.Fatalf("default draft gpu layers = %d, want 0", cfg.DraftGPULayers)
	}

	cfg = ConfigFromEnvWith(ConfigSource{
		Getenv: func(name string) string {
			if name == EnvGPULayers {
				return "0"
			}
			return ""
		},
		LookPath: func(name string) (string, error) {
			if name == "nvidia-smi" {
				return "/usr/bin/nvidia-smi", nil
			}
			return "", os.ErrNotExist
		},
		GOOS: "linux",
	})
	if cfg.GPULayers != 0 {
		t.Fatalf("explicit gpu opt-out = %d, want 0", cfg.GPULayers)
	}
	if cfg.FastDefaults {
		t.Fatalf("fast defaults = true, want false when GPU layers are disabled")
	}

	cfg = ConfigFromEnvWith(ConfigSource{
		Getenv: func(name string) string {
			switch name {
			case EnvGPULayers, EnvDraftGPULayers:
				return "99"
			default:
				return ""
			}
		},
		LookPath: func(name string) (string, error) {
			if name == "nvidia-smi" {
				return "/usr/bin/nvidia-smi", nil
			}
			return "", os.ErrNotExist
		},
		GOOS: "linux",
	})
	if cfg.GPULayers != 99 || cfg.DraftGPULayers != 99 {
		t.Fatalf("explicit gpu config = %+v, want 99 GPU layers", cfg)
	}
	if !cfg.FastDefaults {
		t.Fatalf("fast defaults = false, want automatic CUDA fast defaults")
	}

	cfg = ConfigFromEnvWith(ConfigSource{
		Getenv: func(name string) string {
			if name == EnvFastDefaults {
				return "0"
			}
			return ""
		},
		LookPath: func(name string) (string, error) {
			if name == "nvidia-smi" {
				return "/usr/bin/nvidia-smi", nil
			}
			return "", os.ErrNotExist
		},
		GOOS: "linux",
	})
	if cfg.FastDefaults {
		t.Fatalf("fast defaults = true, want explicit opt-out to win")
	}

	cfg = ConfigFromEnvWith(ConfigSource{
		Getenv: func(name string) string {
			if name == EnvFastDefaults {
				return "1"
			}
			return ""
		},
		LookPath: func(string) (string, error) {
			return "", os.ErrNotExist
		},
		GOOS: "linux",
	})
	if !cfg.FastDefaults {
		t.Fatalf("fast defaults = false, want explicit opt-in")
	}
}

func TestConfigFromEnvEnablesFastDefaultsForWindowsNVIDIAServicePath(t *testing.T) {
	t.Parallel()

	cfg := ConfigFromEnvWith(ConfigSource{
		Getenv: func(name string) string {
			if name == "SystemRoot" {
				return `C:\Windows`
			}
			return ""
		},
		LookPath: func(string) (string, error) {
			return "", os.ErrNotExist
		},
		Stat: func(path string) (os.FileInfo, error) {
			if strings.EqualFold(path, `C:\Windows\System32\nvidia-smi.exe`) {
				return fakeWindowsBundleFileInfo{name: "nvidia-smi.exe"}, nil
			}
			return nil, os.ErrNotExist
		},
		GOOS: "windows",
	})
	if !cfg.FastDefaults {
		t.Fatalf("fast defaults = false, want CUDA fast defaults from known nvidia-smi path")
	}
}

func TestConfigFromEnvEnablesFastDefaultsForLinuxNVIDIAServicePath(t *testing.T) {
	t.Parallel()

	cfg := ConfigFromEnvWith(ConfigSource{
		Getenv: func(string) string {
			return ""
		},
		LookPath: func(string) (string, error) {
			return "", os.ErrNotExist
		},
		Stat: func(path string) (os.FileInfo, error) {
			if path == "/usr/bin/nvidia-smi" {
				return fakeWindowsBundleFileInfo{name: "nvidia-smi"}, nil
			}
			return nil, os.ErrNotExist
		},
		GOOS: "linux",
	})
	if !cfg.FastDefaults {
		t.Fatalf("fast defaults = false, want CUDA fast defaults from known Linux nvidia-smi path")
	}
}

func TestConfigFromEnvUsesHardwareInventoryForCUDAFastDefaults(t *testing.T) {
	t.Parallel()

	cfg := ConfigFromEnvWith(ConfigSource{
		Getenv: func(string) string {
			return ""
		},
		LookPath: func(string) (string, error) {
			return "", os.ErrNotExist
		},
		Stat: func(string) (os.FileInfo, error) {
			return nil, os.ErrNotExist
		},
		GOOS: "linux",
		HardwareCapacity: &capshardware.CapacityInventory{
			GPUDetected:       true,
			GPUVendor:         capshardware.GPUVendorNVIDIA,
			GPUName:           "NVIDIA GeForce RTX 4090",
			CUDAAvailable:     true,
			ComputeCapability: "8.9",
		},
	})

	if cfg.GPULayers != DefaultGPULayers {
		t.Fatalf("gpu layers = %d, want default full offload", cfg.GPULayers)
	}
	if !cfg.FastDefaults || cfg.LaunchProfile != LaunchProfileCUDAFast {
		t.Fatalf("fast launch config = enabled:%t profile:%q, want CUDA fast defaults from hardware inventory", cfg.FastDefaults, cfg.LaunchProfile)
	}
	if got := strings.Join(cfg.AccelerationHints, ","); !strings.Contains(got, "cuda") {
		t.Fatalf("acceleration hints = %q, want cuda from hardware inventory", got)
	}
}

func TestConfigFromEnvUsesVulkanHintForWindowsAMD(t *testing.T) {
	t.Parallel()

	cfg := ConfigFromEnvWith(ConfigSource{
		Getenv: func(string) string {
			return ""
		},
		LookPath: func(string) (string, error) {
			return "", os.ErrNotExist
		},
		Stat: func(string) (os.FileInfo, error) {
			return nil, os.ErrNotExist
		},
		GOOS: "windows",
		HardwareCapacity: &capshardware.CapacityInventory{
			GPUDetected:       true,
			GPUVendor:         capshardware.GPUVendorAMD,
			GPUName:           "AMD Radeon RX 7900 XTX",
			DirectMLAvailable: true,
		},
	})

	if got := strings.Join(cfg.AccelerationHints, ","); !strings.Contains(got, "vulkan") {
		t.Fatalf("acceleration hints = %q, want vulkan for Windows AMD native llama.cpp", got)
	}
	if got := strings.Join(cfg.AccelerationHints, ","); !strings.Contains(got, "directml") {
		t.Fatalf("acceleration hints = %q, want existing DirectML hint preserved", got)
	}
}

func TestBuildServerArgsAddsReasoningFormatForQwen(t *testing.T) {
	t.Parallel()

	args := buildServerArgs(LlamaCppSidecarConfig{
		Host:        DefaultHost,
		Port:        freePortForConfig(t),
		ModelPath:   "/models/Qwen3-8B-Q4_K_M.gguf",
		ContextSize: DefaultContextSize,
	})
	joined := strings.Join(args, " ")
	for _, want := range []string{"--jinja", "--reasoning-format deepseek"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("args = %q, missing Qwen reasoning arg %q", joined, want)
		}
	}
}

func TestBuildServerArgsAddsHarmonyJinjaForGPTOSS(t *testing.T) {
	t.Parallel()

	args := buildServerArgs(LlamaCppSidecarConfig{
		Host:        DefaultHost,
		Port:        freePortForConfig(t),
		ModelPath:   "/models/gpt-oss-20b-mxfp4.gguf",
		ContextSize: DefaultContextSize,
	})
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "--jinja") {
		t.Fatalf("args = %q, missing GPT-OSS Jinja chat-template flag", joined)
	}
	if strings.Contains(joined, "--reasoning-format") {
		t.Fatalf("args = %q, GPT-OSS should use Harmony/Jinja without DeepSeek reasoning format", joined)
	}
}

func TestConfigDiscoversKnownDirServerAndModel(t *testing.T) {
	t.Parallel()

	root := t.TempDir()
	binDir := filepath.Join(root, "bin")
	modelDir := filepath.Join(root, "models")
	serverPath := filepath.Join(binDir, "llama-server")
	modelPath := filepath.Join(modelDir, "phi-4-Q4_K_M.gguf")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatalf("mkdir bin: %v", err)
	}
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatalf("mkdir models: %v", err)
	}
	if err := os.WriteFile(serverPath, []byte("server"), 0o755); err != nil {
		t.Fatalf("write server: %v", err)
	}
	if err := os.WriteFile(modelPath, []byte("gguf"), 0o644); err != nil {
		t.Fatalf("write model: %v", err)
	}

	cfg := ConfigFromEnvWith(ConfigSource{
		Getenv: func(name string) string {
			return ""
		},
		LookPath: func(name string) (string, error) {
			return "", errors.New("not on path")
		},
		ConfiguredBinaryDirs: []string{binDir},
		ConfiguredModelDirs:  []string{modelDir},
	})

	if cfg.ServerPath != serverPath || cfg.ModelPath != modelPath {
		t.Fatalf("discovered paths = %q/%q, want %q/%q", cfg.ServerPath, cfg.ModelPath, serverPath, modelPath)
	}
}

func TestDisabledModeDoesNotStartSidecar(t *testing.T) {
	t.Parallel()

	serverPath, modelPath := sidecarFixtureFiles(t)
	starts := 0
	manager := NewManager(LlamaCppSidecarConfig{
		Enabled:     false,
		ServerPath:  serverPath,
		ModelPath:   modelPath,
		Host:        DefaultHost,
		Port:        freePortForConfig(t),
		ContextSize: DefaultContextSize,
	}, WithProcessStarter(func(ctx context.Context, binary string, args []string, output io.Writer) (managedProcess, error) {
		starts++
		return newFakeProcess(123), nil
	}))

	status := manager.Start(context.Background())
	if starts != 0 {
		t.Fatalf("process starts = %d, want 0 when disabled", starts)
	}
	if status.Enabled || status.Running || status.Healthy {
		t.Fatalf("disabled status = %+v, want stopped disabled", status)
	}
}

func TestMissingBinaryReturnsUnavailableStatus(t *testing.T) {
	t.Parallel()

	_, modelPath := sidecarFixtureFiles(t)
	manager := NewManager(LlamaCppSidecarConfig{
		Enabled:     true,
		ServerPath:  filepath.Join(t.TempDir(), "missing-llama-server"),
		ModelPath:   modelPath,
		Host:        DefaultHost,
		Port:        freePortForConfig(t),
		ContextSize: DefaultContextSize,
	})

	status := manager.Status(context.Background())
	if status.Available || status.Running || status.Healthy {
		t.Fatalf("status = %+v, want unavailable stopped", status)
	}
	if status.Reason != "llama-server binary not detected" {
		t.Fatalf("reason = %q", status.Reason)
	}
}

func TestStoppedOnDemandStatusSuppressesProbeFailure(t *testing.T) {
	t.Parallel()

	serverPath, modelPath := sidecarFixtureFiles(t)
	manager := NewManager(LlamaCppSidecarConfig{
		Enabled:     true,
		ServerPath:  serverPath,
		ModelPath:   modelPath,
		Host:        DefaultHost,
		Port:        freePortForConfig(t),
		ContextSize: DefaultContextSize,
	}, WithHealthClient(errorHealthClient{err: errors.New("connection refused")}), WithHealthTimeout(time.Millisecond))

	status := manager.Status(context.Background())
	if !status.Enabled || !status.Available || status.Running || status.Healthy {
		t.Fatalf("status = %+v, want available stopped on-demand sidecar", status)
	}
	if status.LastError != "" {
		t.Fatalf("last_error = %q, want empty for idle on-demand sidecar", status.LastError)
	}
	if status.Reason != "llama.cpp sidecar stopped; starts on inference demand" {
		t.Fatalf("reason = %q", status.Reason)
	}
}

func TestMockedBinaryStartRecordsRunningState(t *testing.T) {
	t.Parallel()

	serverPath, modelPath := sidecarFixtureFiles(t)
	proc := newFakeProcess(4321)
	manager := NewManager(LlamaCppSidecarConfig{
		Enabled:     true,
		ServerPath:  serverPath,
		ModelPath:   modelPath,
		Host:        DefaultHost,
		Port:        freePortForConfig(t),
		ContextSize: DefaultContextSize,
		Threads:     2,
		GPULayers:   1,
	}, WithProcessStarter(func(ctx context.Context, binary string, args []string, output io.Writer) (managedProcess, error) {
		if binary != serverPath {
			t.Fatalf("binary = %q, want %q", binary, serverPath)
		}
		argText := strings.Join(args, " ")
		for _, want := range []string{"--host " + DefaultHost, "--model " + modelPath, "--ctx-size 4096", "--threads 2", "--n-gpu-layers 1"} {
			if !strings.Contains(argText, want) {
				t.Fatalf("args = %q, missing %q", argText, want)
			}
		}
		return proc, nil
	}), WithHealthClient(errorHealthClient{}), WithHealthTimeout(time.Millisecond))
	t.Cleanup(func() {
		_ = manager.Stop(context.Background())
	})

	status := manager.Start(context.Background())
	if !status.Enabled || !status.Available || !status.Running || status.PID != 4321 {
		t.Fatalf("status = %+v, want running managed process", status)
	}
	if status.Healthy {
		t.Fatalf("healthy = true, want health unconfirmed with failing mock client")
	}
}

func TestProcessExitPreservesStartupLogInLastError(t *testing.T) {
	serverPath, modelPath := sidecarFixtureFiles(t)
	proc := newFakeProcess(2222)
	manager := NewManager(LlamaCppSidecarConfig{
		Enabled:     true,
		ServerPath:  serverPath,
		ModelPath:   modelPath,
		Host:        DefaultHost,
		Port:        freePortForConfig(t),
		ContextSize: DefaultContextSize,
	}, WithProcessStarter(func(ctx context.Context, binary string, args []string, output io.Writer) (managedProcess, error) {
		if output != nil {
			_, _ = output.Write([]byte("ggml_vulkan: missing runtime library\n"))
		}
		return proc, nil
	}), WithHealthClient(errorHealthClient{err: errors.New("connection refused")}))

	_ = manager.Start(context.Background())
	go func() {
		proc.done <- errors.New("exit status 1")
	}()

	var status LlamaCppSidecarStatus
	for i := 0; i < 20; i++ {
		status = manager.Status(context.Background())
		if !status.Running && strings.Contains(status.LastError, "missing runtime library") {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("status.LastError = %q, running=%t; want process exit plus startup log", status.LastError, status.Running)
}

func TestBuildServerArgsAddsGPUFastDefaults(t *testing.T) {
	t.Parallel()

	args := buildServerArgs(LlamaCppSidecarConfig{
		Host:         DefaultHost,
		Port:         freePortForConfig(t),
		ModelPath:    "/models/llama.gguf",
		ContextSize:  DefaultContextSize,
		GPULayers:    DefaultGPULayers,
		FastDefaults: true,
		ExtraArgs:    []string{"--batch-size", "256"},
	})
	joined := strings.Join(args, " ")
	for _, want := range []string{"--flash-attn", "--ubatch-size 512", "--cache-type-k q8_0", "--cache-type-v q8_0"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("args = %q, missing fast default %q", joined, want)
		}
	}
	if got := argValue(args, "--batch-size"); got != "256" {
		t.Fatalf("--batch-size = %q, want explicit extra arg to override default; args=%v", got, args)
	}
}

func TestAppendGPUFastDefaultsForWindowsAvoidsUnsupportedBackendFlags(t *testing.T) {
	t.Parallel()

	args := appendGPUFastDefaultsForGOOS([]string{}, nil, "windows")
	joined := strings.Join(args, " ")
	for _, forbidden := range []string{"--flash-attn", "--cache-type-k", "--cache-type-v"} {
		if strings.Contains(joined, forbidden) {
			t.Fatalf("windows fast defaults = %q, should not include backend-sensitive flag %q", joined, forbidden)
		}
	}
	for _, want := range []string{"--batch-size 512", "--ubatch-size 512"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("windows fast defaults = %q, missing safe launch flag %q", joined, want)
		}
	}
}

func TestRestartWithModelSafeCUDAKeepsGPUOffloadWithoutFastFlags(t *testing.T) {
	t.Parallel()

	serverPath, modelPath := sidecarFixtureFiles(t)
	draftPath := filepath.Join(t.TempDir(), "tinyllama.gguf")
	if err := os.WriteFile(draftPath, []byte("draft"), 0o644); err != nil {
		t.Fatalf("write draft fixture: %v", err)
	}
	proc := newFakeProcess(333)
	var gotArgs []string
	manager := NewManager(LlamaCppSidecarConfig{
		Enabled:        true,
		ServerPath:     serverPath,
		ModelPath:      modelPath,
		Host:           DefaultHost,
		Port:           freePortForConfig(t),
		ContextSize:    DefaultContextSize,
		GPULayers:      DefaultGPULayers,
		FastDefaults:   true,
		DraftModelPath: draftPath,
	}, WithProcessStarter(func(ctx context.Context, binary string, args []string, output io.Writer) (managedProcess, error) {
		gotArgs = append([]string(nil), args...)
		return proc, nil
	}), WithHealthClient(errorHealthClient{}), WithHealthTimeout(time.Millisecond))
	t.Cleanup(func() {
		_ = manager.Stop(context.Background())
	})

	status := manager.RestartWithModelSafeCUDA(context.Background(), modelPath)
	if !status.Running {
		t.Fatalf("status = %+v, want managed process running", status)
	}
	joined := strings.Join(gotArgs, " ")
	if !strings.Contains(joined, "--n-gpu-layers 999") {
		t.Fatalf("args = %q, want CUDA GPU layers preserved", joined)
	}
	if status.Launch == nil || status.Launch.Profile != LaunchProfileCUDASafe {
		t.Fatalf("launch = %+v, want safe CUDA profile", status.Launch)
	}
	for _, forbidden := range []string{"--flash-attn", "--cache-type-k", "--cache-type-v", "--batch-size 512", "--ubatch-size 512", "--model-draft"} {
		if strings.Contains(joined, forbidden) {
			t.Fatalf("args = %q, should not include safe-fallback forbidden flag %q", joined, forbidden)
		}
	}
}

func TestRestartWithModelFastCUDARestoresFullFastProfile(t *testing.T) {
	t.Parallel()

	serverPath, modelPath := sidecarFixtureFiles(t)
	proc := newFakeProcess(335)
	var gotArgs []string
	manager := NewManager(LlamaCppSidecarConfig{
		Enabled:       true,
		ServerPath:    serverPath,
		ModelPath:     modelPath,
		Host:          DefaultHost,
		Port:          freePortForConfig(t),
		ContextSize:   DefaultContextSize,
		GPULayers:     fallbackPartialGPULayers,
		FastDefaults:  false,
		LaunchProfile: LaunchProfileCUDAPartial,
	}, WithProcessStarter(func(ctx context.Context, binary string, args []string, output io.Writer) (managedProcess, error) {
		gotArgs = append([]string(nil), args...)
		return proc, nil
	}), WithHealthClient(errorHealthClient{}), WithHealthTimeout(time.Millisecond))
	t.Cleanup(func() {
		_ = manager.Stop(context.Background())
	})

	status := manager.RestartWithModelFastCUDA(context.Background(), modelPath)
	if !status.Running {
		t.Fatalf("status = %+v, want managed process running", status)
	}
	joined := strings.Join(gotArgs, " ")
	for _, want := range []string{"--n-gpu-layers 999", "--flash-attn", "--ubatch-size 512", "--cache-type-k q8_0", "--cache-type-v q8_0"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("args = %q, missing restored fast CUDA arg %q", joined, want)
		}
	}
	if status.Launch == nil || status.Launch.Profile != LaunchProfileCUDAFast || !status.Launch.FastDefaultsEnabled || status.Launch.ConfiguredGPULayers != DefaultGPULayers {
		t.Fatalf("launch = %+v, want restored fast CUDA profile", status.Launch)
	}
}

func TestRestartWithModelPartialGPUClampsGPUOffload(t *testing.T) {
	t.Parallel()

	serverPath, modelPath := sidecarFixtureFiles(t)
	proc := newFakeProcess(334)
	var gotArgs []string
	manager := NewManager(LlamaCppSidecarConfig{
		Enabled:      true,
		ServerPath:   serverPath,
		ModelPath:    modelPath,
		Host:         DefaultHost,
		Port:         freePortForConfig(t),
		ContextSize:  DefaultContextSize,
		GPULayers:    DefaultGPULayers,
		FastDefaults: true,
	}, WithProcessStarter(func(ctx context.Context, binary string, args []string, output io.Writer) (managedProcess, error) {
		gotArgs = append([]string(nil), args...)
		return proc, nil
	}), WithHealthClient(errorHealthClient{}), WithHealthTimeout(time.Millisecond))
	t.Cleanup(func() {
		_ = manager.Stop(context.Background())
	})

	status := manager.RestartWithModelPartialGPU(context.Background(), modelPath)
	if !status.Running {
		t.Fatalf("status = %+v, want managed process running", status)
	}
	joined := strings.Join(gotArgs, " ")
	if !strings.Contains(joined, "--n-gpu-layers 35") {
		t.Fatalf("args = %q, want partial GPU fallback layers", joined)
	}
	if status.Launch == nil || status.Launch.Profile != LaunchProfileCUDAPartial {
		t.Fatalf("launch = %+v, want partial CUDA profile", status.Launch)
	}
	if strings.Contains(joined, "--flash-attn") || strings.Contains(joined, "--cache-type-k") {
		t.Fatalf("args = %q, want partial fallback without fast flags", joined)
	}
}

func TestStopOnlyTerminatesManagedProcess(t *testing.T) {
	t.Parallel()

	serverPath, modelPath := sidecarFixtureFiles(t)
	proc := newFakeProcess(987)
	manager := NewManager(LlamaCppSidecarConfig{
		Enabled:     true,
		ServerPath:  serverPath,
		ModelPath:   modelPath,
		Host:        DefaultHost,
		Port:        freePortForConfig(t),
		ContextSize: DefaultContextSize,
	}, WithProcessStarter(func(ctx context.Context, binary string, args []string, output io.Writer) (managedProcess, error) {
		return proc, nil
	}), WithHealthClient(errorHealthClient{}), WithHealthTimeout(time.Millisecond))

	if status := manager.Start(context.Background()); !status.Running {
		t.Fatalf("start status = %+v, want running", status)
	}
	if status := manager.Stop(context.Background()); status.Running {
		t.Fatalf("stop status = %+v, want stopped", status)
	}
	if proc.killCount() != 1 {
		t.Fatalf("managed process kill count = %d, want 1", proc.killCount())
	}

	attachedServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(attachedServer.Close)
	host, port := hostPortFromURL(t, attachedServer.URL)
	attached := NewManager(LlamaCppSidecarConfig{
		Enabled:     true,
		ServerPath:  serverPath,
		ModelPath:   modelPath,
		Host:        host,
		Port:        port,
		ContextSize: DefaultContextSize,
	}, WithProcessStarter(func(ctx context.Context, binary string, args []string, output io.Writer) (managedProcess, error) {
		t.Fatalf("attached server should not start a new process")
		return nil, nil
	}))

	status := attached.Start(context.Background())
	if !status.Running || !status.Healthy || !status.Attached || status.PID != 0 {
		t.Fatalf("attached status = %+v, want healthy attached external server", status)
	}
	status = attached.Stop(context.Background())
	if status.Running || status.Attached {
		t.Fatalf("attached stop status = %+v, want detached without killing external process", status)
	}
}

func TestRestartWithModelRehomesAttachedServerBeforeManagedStart(t *testing.T) {
	t.Parallel()

	serverPath, modelPath := sidecarFixtureFiles(t)
	nextModelPath := filepath.Join(t.TempDir(), "Phi-4-Q4_K_M.gguf")
	if err := os.WriteFile(nextModelPath, []byte("phi-gguf"), 0o644); err != nil {
		t.Fatalf("write next model fixture: %v", err)
	}

	attachedServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(attachedServer.Close)
	host, oldPort := hostPortFromURL(t, attachedServer.URL)

	proc := newFakeProcess(6543)
	starts := 0
	manager := NewManager(LlamaCppSidecarConfig{
		Enabled:     true,
		ServerPath:  serverPath,
		ModelPath:   modelPath,
		Host:        host,
		Port:        oldPort,
		ContextSize: DefaultContextSize,
		GPULayers:   DefaultGPULayers,
	}, WithProcessStarter(func(ctx context.Context, binary string, args []string, output io.Writer) (managedProcess, error) {
		starts++
		if binary != serverPath {
			t.Fatalf("binary = %q, want %q", binary, serverPath)
		}
		if got := argValue(args, "--model"); got != nextModelPath {
			t.Fatalf("--model = %q, want %q; args=%v", got, nextModelPath, args)
		}
		if got := argValue(args, "--n-gpu-layers"); got != strconv.Itoa(DefaultGPULayers) {
			t.Fatalf("--n-gpu-layers = %q, want %d; args=%v", got, DefaultGPULayers, args)
		}
		newPort, err := strconv.Atoi(argValue(args, "--port"))
		if err != nil {
			t.Fatalf("--port parse error: %v; args=%v", err, args)
		}
		if newPort == oldPort {
			t.Fatalf("--port = %d, want a fresh managed sidecar port distinct from attached port", newPort)
		}
		return proc, nil
	}), WithHealthTimeout(25*time.Millisecond))
	t.Cleanup(func() {
		_ = manager.Stop(context.Background())
	})

	status := manager.Start(context.Background())
	if !status.Running || !status.Healthy || !status.Attached || status.PID != 0 {
		t.Fatalf("attached status = %+v, want healthy attached external server", status)
	}

	status = manager.RestartWithModel(context.Background(), nextModelPath)
	if starts != 1 {
		t.Fatalf("managed starts = %d, want 1", starts)
	}
	if !status.Running || status.Attached || status.PID != 6543 {
		t.Fatalf("restart status = %+v, want managed process running without attached server", status)
	}
	if status.ModelPath != nextModelPath || status.ModelFilename != "Phi-4-Q4_K_M.gguf" {
		t.Fatalf("restart model metadata = %q/%q, want next model", status.ModelPath, status.ModelFilename)
	}
	if status.BaseURL == attachedServer.URL {
		t.Fatalf("restart base_url = %q, want fresh managed sidecar URL", status.BaseURL)
	}
}

func TestRestartWithModelAvoidsUntrackedHealthyDefaultServer(t *testing.T) {
	t.Parallel()

	serverPath, modelPath := sidecarFixtureFiles(t)
	healthyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(healthyServer.Close)
	host, oldPort := hostPortFromURL(t, healthyServer.URL)

	proc := newFakeProcess(7654)
	starts := 0
	manager := NewManager(LlamaCppSidecarConfig{
		Enabled:     true,
		ServerPath:  serverPath,
		Host:        host,
		Port:        oldPort,
		ContextSize: DefaultContextSize,
		GPULayers:   DefaultGPULayers,
	}, WithProcessStarter(func(ctx context.Context, binary string, args []string, output io.Writer) (managedProcess, error) {
		starts++
		if binary != serverPath {
			t.Fatalf("binary = %q, want %q", binary, serverPath)
		}
		if got := argValue(args, "--model"); got != modelPath {
			t.Fatalf("--model = %q, want %q; args=%v", got, modelPath, args)
		}
		newPort, err := strconv.Atoi(argValue(args, "--port"))
		if err != nil {
			t.Fatalf("--port parse error: %v; args=%v", err, args)
		}
		if newPort == oldPort {
			t.Fatalf("--port = %d, want fresh managed port distinct from untracked server", newPort)
		}
		return proc, nil
	}), WithHealthTimeout(25*time.Millisecond))
	t.Cleanup(func() {
		_ = manager.Stop(context.Background())
	})

	status := manager.RestartWithModel(context.Background(), modelPath)
	if starts != 1 {
		t.Fatalf("managed starts = %d, want 1", starts)
	}
	if !status.Running || status.Attached || status.PID != 7654 {
		t.Fatalf("restart status = %+v, want managed process instead of attached default server", status)
	}
	if status.BaseURL == healthyServer.URL {
		t.Fatalf("restart base_url = %q, want fresh managed sidecar URL", status.BaseURL)
	}
}

func TestConfigureProcessCommandUsesBinaryDirForBackendLibraries(t *testing.T) {
	t.Parallel()

	binDir := t.TempDir()
	binary := filepath.Join(binDir, "llama-server")
	cmd := exec.Command(binary)

	configureProcessCommand(cmd, binary)

	if cmd.Dir != binDir {
		t.Fatalf("cmd.Dir = %q, want llama.cpp binary dir %q", cmd.Dir, binDir)
	}
	if firstEnvSearchPath(cmd.Env, "PATH") != binDir {
		t.Fatalf("PATH first entry = %q, want %q; env=%v", firstEnvSearchPath(cmd.Env, "PATH"), binDir, cmd.Env)
	}
	if runtime.GOOS != "windows" {
		if firstEnvSearchPath(cmd.Env, "LD_LIBRARY_PATH") != binDir {
			t.Fatalf("LD_LIBRARY_PATH first entry = %q, want %q", firstEnvSearchPath(cmd.Env, "LD_LIBRARY_PATH"), binDir)
		}
		if firstEnvSearchPath(cmd.Env, "DYLD_LIBRARY_PATH") != binDir {
			t.Fatalf("DYLD_LIBRARY_PATH first entry = %q, want %q", firstEnvSearchPath(cmd.Env, "DYLD_LIBRARY_PATH"), binDir)
		}
	}
}

func TestStatusJSONExcludesPromptOutputAuthAndTensorData(t *testing.T) {
	t.Parallel()

	serverPath, modelPath := sidecarFixtureFiles(t)
	manager := NewManager(LlamaCppSidecarConfig{
		Enabled:     true,
		ServerPath:  serverPath,
		ModelPath:   modelPath,
		Host:        DefaultHost,
		Port:        freePortForConfig(t),
		ContextSize: DefaultContextSize,
	}, WithHealthClient(errorHealthClient{err: errors.New("raw_prompt output_text auth_token tensor_bytes")}), WithHealthTimeout(time.Millisecond))

	status := manager.Status(context.Background())
	raw, err := json.Marshal(status)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	body := strings.ToLower(string(raw))
	for _, forbidden := range []string{"raw_prompt", "prompt_text", "model_output", "output_text", "generated_text", "key_data", "value_data", "query_vector", "tensor_bytes", "raw_tensor", "auth_token", "bind_token"} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("status JSON contains forbidden marker %q: %s", forbidden, raw)
		}
	}
	if status.SupportsKVAccess || status.SupportsTensorHooks {
		t.Fatalf("sidecar status should not advertise KV/tensor hooks: %+v", status)
	}
}

func TestBuildBackendRuntimesMarksHealthySidecarLoadedWarm(t *testing.T) {
	t.Parallel()

	lastHealth := time.Unix(123, 456000000)
	runtimes := BuildBackendRuntimes(LlamaCppSidecarStatus{
		Enabled:    true,
		Available:  true,
		Running:    true,
		Healthy:    true,
		ServerPath: "/opt/ryvion/bin/llama-server",
		Launch: &LlamaCppLaunchConfig{
			Mode:                     "managed",
			Managed:                  true,
			ServerPath:               "/opt/ryvion/bin/llama-server",
			ServerFilename:           "llama-server",
			ConfiguredGPULayers:      999,
			FastDefaultsEnabled:      true,
			ConfiguredDraftGPULayers: 12,
		},
		ServerProperties: &LlamaCppServerProperties{
			BuildInfo:            "llama.cpp b999 CUDA",
			SystemInfo:           "CUDA enabled",
			ReportedGPULayers:    999,
			ReportedAcceleration: []string{"cuda"},
		},
		BaseURL:                "http://127.0.0.1:45910",
		ModelPath:              "/tmp/ryvion-models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		ModelFilename:          "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		ModelSizeBytes:         2019377696,
		ModelFamilyHint:        "llama",
		QuantizationHint:       "Q4_K_M",
		LastHealthAt:           lastHealth,
		Backend:                BackendName,
		OpenAICompatible:       true,
		SupportsTextGeneration: true,
		SupportsStreaming:      true,
	})

	runtime := runtimes.LlamaCPP
	if !runtime.Enabled || !runtime.Available || !runtime.Running || !runtime.Healthy || !runtime.Loaded || !runtime.Warm {
		t.Fatalf("backend runtime = %+v, want active loaded warm status", runtime)
	}
	if runtime.ModelID != "Llama-3.2-3B-Instruct-Q4_K_M.gguf" ||
		runtime.ModelPath != "/tmp/ryvion-models/Llama-3.2-3B-Instruct-Q4_K_M.gguf" ||
		runtime.ModelSizeBytes != 2019377696 ||
		runtime.ModelFamilyHint != "llama" ||
		runtime.QuantizationHint != "Q4_K_M" {
		t.Fatalf("model residency metadata = %+v", runtime)
	}
	if runtime.Launch == nil ||
		runtime.Launch.Mode != "managed" ||
		!runtime.Launch.Managed ||
		runtime.Launch.Attached ||
		runtime.Launch.ServerFilename != "llama-server" ||
		runtime.Launch.ConfiguredGPULayers != 999 ||
		!runtime.Launch.FastDefaultsEnabled ||
		runtime.Launch.ConfiguredDraftGPULayers != 12 {
		t.Fatalf("launch telemetry = %+v, want managed GPU launch config", runtime.Launch)
	}
	if runtime.ServerProperties == nil ||
		runtime.ServerProperties.BuildInfo != "llama.cpp b999 CUDA" ||
		runtime.ServerProperties.ReportedGPULayers != 999 ||
		len(runtime.ServerProperties.ReportedAcceleration) != 1 ||
		runtime.ServerProperties.ReportedAcceleration[0] != "cuda" {
		t.Fatalf("server properties = %+v, want safe CUDA build hints", runtime.ServerProperties)
	}
	if len(runtime.Acceleration) != 1 || runtime.Acceleration[0] != "cuda" || !strings.Contains(runtime.AccelerationReason, "server_reported_cuda") {
		t.Fatalf("runtime acceleration = %+v reason=%q, want reported CUDA status", runtime.Acceleration, runtime.AccelerationReason)
	}
	if runtime.LastHealthAtUnixMs != lastHealth.UnixMilli() {
		t.Fatalf("last_health_at_unix_ms = %d, want %d", runtime.LastHealthAtUnixMs, lastHealth.UnixMilli())
	}
	if runtime.SupportsKVAccess || runtime.SupportsTensorHooks {
		t.Fatalf("backend runtime should not advertise KV/tensor hooks: %+v", runtime)
	}
}

func TestGPUOffloadRequestedButServerReportsCPUOnly(t *testing.T) {
	t.Parallel()

	if !gpuOffloadRequestedButServerReportsCPUOnly(LlamaCppSidecarConfig{GPULayers: 999}, &LlamaCppServerProperties{
		BuildInfo:            "llama.cpp b9180",
		ReportedAcceleration: []string{"cpu"},
	}) {
		t.Fatal("gpuOffloadRequestedButServerReportsCPUOnly() = false, want CPU-only warning")
	}
	if gpuOffloadRequestedButServerReportsCPUOnly(LlamaCppSidecarConfig{GPULayers: 0}, &LlamaCppServerProperties{
		ReportedAcceleration: []string{"cpu"},
	}) {
		t.Fatal("gpuOffloadRequestedButServerReportsCPUOnly() = true with GPU layers disabled")
	}
	if gpuOffloadRequestedButServerReportsCPUOnly(LlamaCppSidecarConfig{GPULayers: 999}, &LlamaCppServerProperties{
		ReportedAcceleration: []string{"cuda"},
	}) {
		t.Fatal("gpuOffloadRequestedButServerReportsCPUOnly() = true for CUDA server")
	}
}

func TestSidecarAccelerationStatusReportsVulkan(t *testing.T) {
	t.Parallel()

	acceleration, reason := sidecarAccelerationStatus(LlamaCppSidecarConfig{GPULayers: 999}, &LlamaCppServerProperties{
		ReportedAcceleration: []string{"vulkan"},
	})
	if len(acceleration) != 1 || acceleration[0] != "vulkan" || reason != "server_reported_vulkan" {
		t.Fatalf("sidecarAccelerationStatus() = %+v/%q, want reported Vulkan", acceleration, reason)
	}

	acceleration, reason = sidecarAccelerationStatus(LlamaCppSidecarConfig{
		GPULayers:         999,
		AccelerationHints: []string{"vulkan"},
	}, nil)
	if len(acceleration) != 1 || acceleration[0] != "vulkan" || reason != "configured_vulkan_unconfirmed" {
		t.Fatalf("sidecarAccelerationStatus(unconfirmed) = %+v/%q, want configured Vulkan", acceleration, reason)
	}
}

func TestEnrichBackendRuntimesDoesNotPromoteActiveCPUOnlySidecarToCUDA(t *testing.T) {
	t.Parallel()

	runtimes := BuildBackendRuntimes(LlamaCppSidecarStatus{
		Enabled:   true,
		Available: true,
		Running:   true,
		Healthy:   true,
		Launch: &LlamaCppLaunchConfig{
			Mode:                "managed",
			Managed:             true,
			ConfiguredGPULayers: DefaultGPULayers,
		},
		ServerProperties: &LlamaCppServerProperties{
			BuildInfo:            "llama.cpp b9180",
			ReportedAcceleration: []string{"cpu"},
		},
		BaseURL:                "http://127.0.0.1:45910",
		ModelPath:              "/tmp/ryvion-models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		ModelFilename:          "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		Backend:                BackendName,
		OpenAICompatible:       true,
		SupportsTextGeneration: true,
		SupportsStreaming:      true,
	})

	enriched := EnrichBackendRuntimes(runtimes, runtimeinventory.Inventory{}, capshardware.CapacityInventory{
		GPUDetected:       true,
		GPUVendor:         capshardware.GPUVendorNVIDIA,
		GPUName:           "NVIDIA GeForce RTX 4090",
		CUDAAvailable:     true,
		ComputeCapability: "8.9",
	})
	runtime := enriched.LlamaCPP
	if len(runtime.Acceleration) != 1 || runtime.Acceleration[0] != "cpu" {
		t.Fatalf("active acceleration = %+v, want CPU-only server report to remain authoritative", runtime.Acceleration)
	}
	if !strings.Contains(runtime.AccelerationReason, "server_reported_cpu") {
		t.Fatalf("acceleration_reason = %q, want CPU-only explanation", runtime.AccelerationReason)
	}
}

func TestEnrichBackendRuntimesPreservesActiveSGLangSidecar(t *testing.T) {
	t.Parallel()

	runtimes := NormalizeBackendRuntimes(BackendRuntimes{
		SGLang: BackendRuntimeStatus{
			Enabled:                  true,
			Available:                true,
			Running:                  true,
			Healthy:                  true,
			Backend:                  runtimeinventory.BackendCandidateSGLang,
			BaseURL:                  "http://127.0.0.1:45921",
			ModelID:                  "qwen2.5-7b-instruct",
			ModelPath:                "/models/qwen2.5-7b-instruct",
			ModelFilename:            "qwen2.5-7b-instruct",
			OpenAICompatible:         true,
			SupportsTextGeneration:   true,
			SupportsStreaming:        true,
			SupportsStatefulSessions: true,
			Acceleration:             []string{"cuda"},
		},
	})

	enriched := EnrichBackendRuntimes(runtimes, runtimeinventory.Inventory{
		BackendCandidates: []runtimeinventory.BackendCandidate{{
			Backend:                        runtimeinventory.BackendCandidateSGLang,
			Detected:                       true,
			SupportsTextGeneration:         true,
			SupportsStreaming:              true,
			SupportsOpenAICompatibleServer: true,
		}},
	}, capshardware.CapacityInventory{
		GPUDetected:       true,
		GPUVendor:         capshardware.GPUVendorNVIDIA,
		CUDAAvailable:     true,
		ComputeCapability: "8.9",
	})

	runtime := enriched.SGLang
	if !runtime.Running || !runtime.Healthy || !runtime.Loaded || !runtime.Warm {
		t.Fatalf("sglang runtime = %+v, want active managed sidecar preserved", runtime)
	}
	if runtime.BaseURL != "http://127.0.0.1:45921" || runtime.ModelID != "qwen2.5-7b-instruct" {
		t.Fatalf("sglang active metadata = %+v, want managed sidecar URL/model preserved", runtime)
	}
}

func TestBuildBackendRuntimesDisabledSidecarNotLoadedWarm(t *testing.T) {
	t.Parallel()

	runtimes := BuildBackendRuntimes(LlamaCppSidecarStatus{
		Enabled:                false,
		Available:              true,
		Running:                true,
		Healthy:                true,
		BaseURL:                "http://127.0.0.1:45910",
		ModelPath:              "/tmp/ryvion-models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		ModelFilename:          "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		Backend:                BackendName,
		OpenAICompatible:       true,
		SupportsTextGeneration: true,
		SupportsStreaming:      true,
	})

	runtime := runtimes.LlamaCPP
	if runtime.Enabled || runtime.Available || runtime.Running || runtime.Healthy || runtime.Loaded || runtime.Warm {
		t.Fatalf("disabled backend runtime = %+v, want inactive unloaded state", runtime)
	}
	if runtime.ModelID != "Llama-3.2-3B-Instruct-Q4_K_M.gguf" {
		t.Fatalf("model_id = %q, want configured filename metadata preserved", runtime.ModelID)
	}
}

func TestNormalizeBackendRuntimesSanitizesUnsafeFields(t *testing.T) {
	t.Parallel()

	longPath := "/tmp/" + strings.Repeat("p", 600)
	runtimes := NormalizeBackendRuntimes(BackendRuntimes{
		LlamaCPP: BackendRuntimeStatus{
			Enabled:                true,
			Available:              true,
			Running:                true,
			Healthy:                true,
			Backend:                BackendName,
			BaseURL:                "http://127.0.0.1:45910",
			ModelPath:              longPath + "\n",
			ModelFilename:          "model.Q4_K_M.gguf",
			ModelSizeBytes:         -1,
			Loaded:                 true,
			Warm:                   true,
			OpenAICompatible:       true,
			SupportsTextGeneration: true,
			SupportsStreaming:      true,
			SupportsKVAccess:       true,
			SupportsTensorHooks:    true,
			LastHealthAtUnixMs:     99,
			LastError:              "raw_prompt output_text auth_token tensor_bytes",
		},
	})

	runtime := runtimes.LlamaCPP
	if len(runtime.ModelPath) != maxRuntimePathLen {
		t.Fatalf("model_path length = %d, want %d", len(runtime.ModelPath), maxRuntimePathLen)
	}
	if runtime.ModelSizeBytes != 0 {
		t.Fatalf("model_size_bytes = %d, want 0 for negative input", runtime.ModelSizeBytes)
	}
	if runtime.LastError != "llama.cpp sidecar status redacted" {
		t.Fatalf("last_error = %q, want redacted", runtime.LastError)
	}
	if runtime.SupportsKVAccess || runtime.SupportsTensorHooks {
		t.Fatalf("normalized runtime should not advertise KV/tensor hooks: %+v", runtime)
	}
	raw, err := json.Marshal(runtimes)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	body := strings.ToLower(string(raw))
	for _, forbidden := range []string{"server_path", "pid", "stdout", "stderr", "raw_prompt", "prompt_text", "model_output", "output_text", "generated_text", "key_data", "value_data", "query_vector", "tensor_bytes", "raw_tensor", "auth_token", "bind_token"} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("backend runtime JSON contains forbidden marker %q: %s", forbidden, raw)
		}
	}
}

func TestEnrichBackendRuntimesRepresentsBlackwellGVRCapability(t *testing.T) {
	t.Parallel()

	runtimes := EnrichBackendRuntimes(BackendRuntimes{}, runtimeinventory.Inventory{
		BackendCandidates: []runtimeinventory.BackendCandidate{{
			Backend:                        runtimeinventory.BackendCandidateTensorRTLLM,
			Detected:                       true,
			SupportsTextGeneration:         true,
			SupportsOpenAICompatibleServer: true,
			SupportsStreaming:              true,
		}},
	}, capshardware.CapacityInventory{
		GPUDetected:       true,
		GPUVendor:         capshardware.GPUVendorNVIDIA,
		GPUName:           "NVIDIA B200",
		CUDAAvailable:     true,
		ComputeCapability: "10.0",
	})

	runtime := runtimes.TensorRTLLM
	if runtime.GPUArchitecture != "blackwell_sm100_plus" || runtime.GPUComputeCapability != "10.0" {
		t.Fatalf("gpu metadata = %q/%q, want blackwell_sm100_plus/10.0", runtime.GPUArchitecture, runtime.GPUComputeCapability)
	}
	if len(runtime.OptimizationCapabilities) != 1 ||
		runtime.OptimizationCapabilities[0].Name != "gvr_topk" ||
		!runtime.OptimizationCapabilities[0].Supported ||
		!runtime.OptimizationCapabilities[0].Enabled ||
		runtime.OptimizationCapabilities[0].RequiresAttention != "deepseek_sparse_attention" ||
		runtime.OptimizationCapabilities[0].RequiresGPUArch != "blackwell_sm100_plus" ||
		runtime.OptimizationCapabilities[0].ContextMinTokens != 16384 {
		t.Fatalf("optimization_capabilities = %#v, want enabled gvr_topk", runtime.OptimizationCapabilities)
	}
}

func TestEnrichBackendRuntimesDoesNotFalselyClaimGVRForNonBlackwellRTX(t *testing.T) {
	t.Parallel()

	runtimes := EnrichBackendRuntimes(BackendRuntimes{}, runtimeinventory.Inventory{
		BackendCandidates: []runtimeinventory.BackendCandidate{{
			Backend:                runtimeinventory.BackendCandidateTensorRTLLM,
			Detected:               true,
			SupportsTextGeneration: true,
		}},
	}, capshardware.CapacityInventory{
		GPUDetected:       true,
		GPUVendor:         capshardware.GPUVendorNVIDIA,
		GPUName:           "NVIDIA GeForce RTX 4090",
		CUDAAvailable:     true,
		ComputeCapability: "8.9",
	})

	caps := runtimes.TensorRTLLM.OptimizationCapabilities
	if len(caps) != 1 || caps[0].Name != "gvr_topk" {
		t.Fatalf("optimization_capabilities = %#v, want represented gvr_topk capability row", caps)
	}
	if caps[0].Supported || caps[0].Enabled {
		t.Fatalf("non-Blackwell RTX should not claim enabled GVR: %#v", caps[0])
	}
	if runtimes.LlamaCPP.OptimizationCapabilities != nil && len(runtimes.LlamaCPP.OptimizationCapabilities) != 0 {
		t.Fatalf("llama.cpp should not inherit TensorRT-only GVR capability: %#v", runtimes.LlamaCPP.OptimizationCapabilities)
	}
}

func sidecarFixtureFiles(t *testing.T) (string, string) {
	t.Helper()
	dir := t.TempDir()
	serverPath := filepath.Join(dir, "llama-server")
	modelPath := filepath.Join(dir, "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
	if err := os.WriteFile(serverPath, []byte("server"), 0o755); err != nil {
		t.Fatalf("write server fixture: %v", err)
	}
	if err := os.WriteFile(modelPath, []byte("gguf"), 0o644); err != nil {
		t.Fatalf("write model fixture: %v", err)
	}
	return serverPath, modelPath
}

func freePortForConfig(t *testing.T) int {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	defer server.Close()
	_, port := hostPortFromURL(t, server.URL)
	return port
}

func hostPortFromURL(t *testing.T, raw string) (string, int) {
	t.Helper()
	parsed, err := url.Parse(raw)
	if err != nil {
		t.Fatalf("parse url: %v", err)
	}
	port, err := strconv.Atoi(parsed.Port())
	if err != nil {
		t.Fatalf("parse port %q: %v", parsed.Port(), err)
	}
	return parsed.Hostname(), port
}

func argValue(args []string, name string) string {
	for i := 0; i < len(args)-1; i++ {
		if args[i] == name {
			return args[i+1]
		}
	}
	return ""
}

func firstEnvSearchPath(env []string, key string) string {
	for _, item := range env {
		name, value, ok := strings.Cut(item, "=")
		if !ok || !strings.EqualFold(name, key) {
			continue
		}
		parts := strings.Split(value, string(os.PathListSeparator))
		if len(parts) == 0 {
			return ""
		}
		return parts[0]
	}
	return ""
}

type errorHealthClient struct {
	err error
}

func (c errorHealthClient) Do(*http.Request) (*http.Response, error) {
	if c.err != nil {
		return nil, c.err
	}
	return nil, errors.New("connection refused")
}

type fakeProcess struct {
	pid    int
	done   chan error
	once   sync.Once
	kills  int
	killsM sync.Mutex
}

func newFakeProcess(pid int) *fakeProcess {
	return &fakeProcess{pid: pid, done: make(chan error)}
}

func (p *fakeProcess) PID() int {
	return p.pid
}

func (p *fakeProcess) Wait() error {
	return <-p.done
}

func (p *fakeProcess) Kill() error {
	p.killsM.Lock()
	p.kills++
	p.killsM.Unlock()
	p.once.Do(func() {
		close(p.done)
	})
	return nil
}

func (p *fakeProcess) killCount() int {
	p.killsM.Lock()
	defer p.killsM.Unlock()
	return p.kills
}
