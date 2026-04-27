package runtimeexec

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestResolveBackendPathUsesEnvOverride(t *testing.T) {
	t.Setenv("RYV_RUNTIME_BACKEND_BINARY", "/tmp/ryvion/runtime/backend/custom-oci")
	got := ResolveBackendPath("linux", os.Getenv)
	if got != "/tmp/ryvion/runtime/backend/custom-oci" {
		t.Fatalf("ResolveBackendPath() = %q", got)
	}
}

func TestResolveBackendCommandPrefersRyvionBackendShim(t *testing.T) {
	temp := t.TempDir()
	backend := filepath.Join(temp, "backend", "ryvion-oci")
	if err := os.MkdirAll(filepath.Dir(backend), 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	if err := os.WriteFile(backend, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	getenv := func(key string) string {
		switch key {
		case "RYV_RUNTIME_BACKEND_BINARY":
			return backend
		default:
			return ""
		}
	}

	got, err := ResolveBackendCommand("linux", getenv)
	if err != nil {
		t.Fatalf("ResolveBackendCommand() error = %v", err)
	}
	if got != backend {
		t.Fatalf("ResolveBackendCommand() = %q, want %q", got, backend)
	}
}

func TestEngineKindDetectsKnownBackends(t *testing.T) {
	cases := map[string]string{
		"/usr/bin/podman":   "podman",
		"/usr/bin/docker":   "docker",
		"/usr/bin/nerdctl":  "nerdctl",
		"/opt/ryvion/thing": "unknown",
		"":                  "",
	}
	for input, want := range cases {
		if got := EngineKind(input); got != want {
			t.Fatalf("EngineKind(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestResolveOCIEngineCLIWindowsFindsLocalAppDataPodman(t *testing.T) {
	temp := t.TempDir()
	podman := filepath.Join(temp, "Programs", "RedHat", "Podman Desktop", "podman.exe")
	if err := os.MkdirAll(filepath.Dir(podman), 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	if err := os.WriteFile(podman, []byte(""), 0o755); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	getenv := func(key string) string {
		switch key {
		case "LOCALAPPDATA":
			return temp
		default:
			return ""
		}
	}

	got, err := resolveOCIEngineCLI("windows", getenv)
	if err != nil {
		t.Fatalf("resolveOCIEngineCLI() error = %v", err)
	}
	if got != podman {
		t.Fatalf("resolveOCIEngineCLI() = %q, want %q", got, podman)
	}
}

func TestResolveOCIEngineCLIDoesNotUseDockerByDefault(t *testing.T) {
	temp := t.TempDir()
	docker := filepath.Join(temp, "docker")
	if err := os.WriteFile(docker, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	t.Setenv("PATH", temp)
	t.Setenv("RYV_ALLOW_DOCKER_FALLBACK", "")

	got, err := resolveOCIEngineCLI("linux", os.Getenv)
	if err == nil {
		if strings.HasPrefix(strings.ToLower(filepath.Base(got)), "docker") {
			t.Fatalf("docker engine %q should not be selected by default", got)
		}
		t.Skipf("host has a non-Docker managed runtime engine available: %s", got)
	}
	if !strings.Contains(err.Error(), "RYV_ALLOW_DOCKER_FALLBACK=1") {
		t.Fatalf("error = %q, want Docker fallback guidance", err.Error())
	}
}

func TestResolveOCIEngineCLIAllowsDockerFallbackWhenExplicit(t *testing.T) {
	temp := t.TempDir()
	docker := filepath.Join(temp, "docker")
	if err := os.WriteFile(docker, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	t.Setenv("PATH", temp)
	t.Setenv("RYV_ALLOW_DOCKER_FALLBACK", "1")

	got, err := resolveOCIEngineCLI("linux", os.Getenv)
	if err != nil {
		t.Fatalf("resolveOCIEngineCLI() error = %v", err)
	}
	if !strings.HasPrefix(strings.ToLower(filepath.Base(got)), "docker") {
		t.Skipf("host has a managed runtime engine available; Docker remains a fallback only: %s", got)
	}
}
