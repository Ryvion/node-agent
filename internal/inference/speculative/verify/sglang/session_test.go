package sglang

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestResolveVerifierCommandUsesExplicitEnvCommand(t *testing.T) {
	t.Setenv("RYV_SGLANG_VERIFIER_CMD", "python /opt/ryvion/sglang-verifier/run.py")
	command, ok := ResolveVerifierCommand()
	if !ok {
		t.Fatal("ResolveVerifierCommand ok = false")
	}
	if !command.Shell || command.Original != "python /opt/ryvion/sglang-verifier/run.py" {
		t.Fatalf("command = %+v, want shell command from env", command)
	}
}

func TestWaitForSocketAcceptsExistingPath(t *testing.T) {
	path := filepath.Join(t.TempDir(), "verifier.sock")
	if err := os.WriteFile(path, []byte{}, 0o600); err != nil {
		t.Fatalf("write socket marker: %v", err)
	}
	if err := WaitForSocket(context.Background(), path, time.Second); err != nil {
		t.Fatalf("WaitForSocket() error = %v", err)
	}
}
