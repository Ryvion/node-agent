package runtimeexec

import (
	"os"
	"path/filepath"
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
