package runtimeexec

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

type Executor struct {
	Command    string
	PrefixArgs []string
	BinaryPath string
	ViaWrapper bool
}

type Status struct {
	BinaryPath   string `json:"binary_path"`
	BackendPath  string `json:"backend_path"`
	EnginePath   string `json:"engine_path"`
	EngineKind   string `json:"engine_kind"`
	CLIInstalled bool   `json:"cli_installed"`
	Ready        bool   `json:"ready"`
	GPUReady     bool   `json:"gpu_ready"`
	Health       string `json:"health"`
}

func ResolveBinaryPath(goos string, getenv func(string) string) string {
	if getenv == nil {
		getenv = os.Getenv
	}
	if value := strings.TrimSpace(getenv("RYV_RUNTIME_BINARY")); value != "" {
		return value
	}
	switch goos {
	case "windows":
		programFiles := strings.TrimSpace(getenv("ProgramFiles"))
		if programFiles == "" {
			programFiles = `C:\Program Files`
		}
		return filepath.Join(programFiles, "Ryvion", "runtime", "ryvion-runtime.ps1")
	case "linux":
		return "/opt/ryvion/runtime/ryvion-runtime"
	default:
		return ""
	}
}

func ResolveBackendPath(goos string, getenv func(string) string) string {
	if getenv == nil {
		getenv = os.Getenv
	}
	if value := strings.TrimSpace(getenv("RYV_RUNTIME_BACKEND_BINARY")); value != "" {
		return value
	}
	switch goos {
	case "windows":
		programFiles := strings.TrimSpace(getenv("ProgramFiles"))
		if programFiles == "" {
			programFiles = `C:\Program Files`
		}
		return filepath.Join(programFiles, "Ryvion", "runtime", "backend", "ryvion-oci.cmd")
	case "linux":
		return "/opt/ryvion/runtime/backend/ryvion-oci"
	default:
		return ""
	}
}

func ResolveEnginePath(goos string, getenv func(string) string) string {
	if getenv == nil {
		getenv = os.Getenv
	}
	if value := strings.TrimSpace(getenv("RYV_RUNTIME_ENGINE_BINARY")); value != "" {
		if _, err := os.Stat(value); err == nil {
			return value
		}
	}
	engine, err := resolveOCIEngineCLI(goos, getenv)
	if err != nil {
		return ""
	}
	return engine
}

func EngineKind(path string) string {
	name := strings.ToLower(strings.TrimSpace(filepath.Base(path)))
	if name == "" || name == "." {
		return ""
	}
	switch {
	case strings.HasPrefix(name, "podman"):
		return "podman"
	case strings.HasPrefix(name, "docker"):
		return "docker"
	case strings.HasPrefix(name, "nerdctl"):
		return "nerdctl"
	case name == "":
		return ""
	default:
		return "unknown"
	}
}

func ResolveBackendCommand(goos string, getenv func(string) string) (string, error) {
	if backend := strings.TrimSpace(ResolveBackendPath(goos, getenv)); backend != "" {
		if _, err := os.Stat(backend); err == nil {
			return backend, nil
		}
	}
	return resolveOCIEngineCLI(goos, getenv)
}

func ResolveExecutor(goos string, getenv func(string) string) (Executor, error) {
	binary := ResolveBinaryPath(goos, getenv)
	if binary != "" {
		if _, err := os.Stat(binary); err == nil {
			command, prefixArgs, err := wrapperCommand(goos, getenv, binary, "oci")
			if err == nil {
				return Executor{
					Command:    command,
					PrefixArgs: prefixArgs,
					BinaryPath: binary,
					ViaWrapper: true,
				}, nil
			}
		}
	}

	backendBin, err := ResolveBackendCommand(goos, getenv)
	if err != nil {
		return Executor{}, err
	}
	return Executor{
		Command:    backendBin,
		BinaryPath: backendBin,
		ViaWrapper: false,
	}, nil
}

func ProbeStatus(ctx context.Context, goos string, getenv func(string) string, binary string) (Status, bool) {
	if getenv == nil {
		getenv = os.Getenv
	}
	if strings.TrimSpace(binary) == "" {
		binary = ResolveBinaryPath(goos, getenv)
	}
	if strings.TrimSpace(binary) == "" {
		return Status{}, false
	}
	if _, err := os.Stat(binary); err != nil {
		return Status{}, false
	}

	command, args, err := wrapperCommand(goos, getenv, binary, "status-json")
	if err != nil {
		return Status{}, false
	}
	out, err := exec.CommandContext(ctx, command, args...).Output()
	if err != nil {
		return Status{}, false
	}
	var status Status
	if err := json.Unmarshal(out, &status); err != nil {
		return Status{}, false
	}
	if strings.TrimSpace(status.BinaryPath) == "" {
		status.BinaryPath = binary
	}
	if strings.TrimSpace(status.EnginePath) == "" {
		status.EnginePath = ResolveEnginePath(goos, getenv)
	}
	if strings.TrimSpace(status.EngineKind) == "" {
		status.EngineKind = EngineKind(status.EnginePath)
	}
	status.Health = strings.TrimSpace(status.Health)
	if status.Health == "" {
		switch {
		case status.Ready:
			status.Health = "ready"
		case status.CLIInstalled:
			status.Health = "degraded"
		default:
			status.Health = "missing"
		}
	}
	return status, true
}

func wrapperCommand(goos string, getenv func(string) string, binary string, subcommand string) (string, []string, error) {
	if strings.TrimSpace(binary) == "" {
		return "", nil, fmt.Errorf("runtime wrapper binary required")
	}
	switch goos {
	case "windows":
		shell, err := resolvePowerShell(getenv)
		if err != nil {
			return "", nil, err
		}
		return shell, []string{"-NoProfile", "-ExecutionPolicy", "Bypass", "-File", binary, subcommand}, nil
	default:
		return binary, []string{subcommand}, nil
	}
}

func resolvePowerShell(getenv func(string) string) (string, error) {
	if p, err := exec.LookPath("pwsh"); err == nil {
		return p, nil
	}
	if p, err := exec.LookPath("powershell"); err == nil {
		return p, nil
	}
	if getenv == nil {
		getenv = os.Getenv
	}
	candidates := []string{
		filepath.Join(strings.TrimSpace(getenv("SystemRoot")), "System32", "WindowsPowerShell", "v1.0", "powershell.exe"),
		filepath.Join(strings.TrimSpace(getenv("WINDIR")), "System32", "WindowsPowerShell", "v1.0", "powershell.exe"),
	}
	for _, candidate := range candidates {
		if candidate == "" {
			continue
		}
		if _, err := os.Stat(candidate); err == nil {
			return candidate, nil
		}
	}
	return "", fmt.Errorf("powershell not found")
}

func resolveOCIEngineCLI(goos string, getenv func(string) string) (string, error) {
	if getenv == nil {
		getenv = os.Getenv
	}
	if value := strings.TrimSpace(getenv("RYV_RUNTIME_ENGINE_BINARY")); value != "" {
		if _, err := os.Stat(value); err == nil {
			return value, nil
		}
	}
	engineCandidates := []string{"docker", "nerdctl", "podman"}
	if goos == "linux" {
		engineCandidates = []string{"nerdctl", "podman", "docker"}
	}
	if goos == "windows" {
		engineCandidates = []string{"podman"}
	}
	for _, candidate := range engineCandidates {
		if p, err := exec.LookPath(candidate); err == nil {
			return p, nil
		}
	}
	candidates := []string{
		"/usr/local/bin/nerdctl",
		"/usr/local/bin/podman",
		"/usr/local/bin/docker",
		"/opt/homebrew/bin/nerdctl",
		"/opt/homebrew/bin/podman",
		"/opt/homebrew/bin/docker",
		"/usr/bin/nerdctl",
		"/usr/bin/podman",
		"/usr/bin/docker",
		"/snap/bin/nerdctl",
		"/snap/bin/podman",
		"/snap/bin/docker",
	}
	if goos == "windows" {
		candidates = []string{
			filepath.Join(strings.TrimSpace(getenv("ProgramFiles")), "RedHat", "Podman", "podman.exe"),
			filepath.Join(strings.TrimSpace(getenv("ProgramW6432")), "RedHat", "Podman", "podman.exe"),
			filepath.Join(strings.TrimSpace(getenv("ProgramFiles")), "RedHat", "Podman Desktop", "podman.exe"),
			filepath.Join(strings.TrimSpace(getenv("ProgramW6432")), "RedHat", "Podman Desktop", "podman.exe"),
			filepath.Join(strings.TrimSpace(getenv("LOCALAPPDATA")), "Programs", "RedHat", "Podman", "podman.exe"),
			filepath.Join(strings.TrimSpace(getenv("LOCALAPPDATA")), "Programs", "RedHat", "Podman Desktop", "podman.exe"),
			filepath.Join(strings.TrimSpace(getenv("LOCALAPPDATA")), "Programs", "Podman", "podman.exe"),
			filepath.Join(strings.TrimSpace(getenv("LOCALAPPDATA")), "Programs", "Podman Desktop", "podman.exe"),
			filepath.Join(strings.TrimSpace(getenv("LOCALAPPDATA")), "Microsoft", "WinGet", "Links", "podman.exe"),
		}
	}
	for _, candidate := range candidates {
		if candidate == "" {
			continue
		}
		if _, err := os.Stat(candidate); err == nil {
			return candidate, nil
		}
	}
	return "", fmt.Errorf("managed OCI engine not found on %s", goos)
}
