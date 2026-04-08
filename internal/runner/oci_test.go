package runner

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestResolveWorkBasePrefersExplicitEnv(t *testing.T) {
	t.Parallel()

	got := resolveWorkBase("windows", func(key string) string {
		switch key {
		case "RYV_WORK_DIR":
			return `D:\Ryvion\custom-work`
		case "ProgramData":
			return `C:\ProgramData`
		default:
			return ""
		}
	})

	if got != `D:\Ryvion\custom-work` {
		t.Fatalf("expected explicit work dir, got %q", got)
	}
}

func TestResolveWorkBaseDefaultsToProgramDataOnWindows(t *testing.T) {
	t.Parallel()

	got := resolveWorkBase("windows", func(key string) string {
		switch key {
		case "ProgramData":
			return `C:\ProgramData`
		default:
			return ""
		}
	})

	want := filepath.Join(`C:\ProgramData`, "Ryvion", "work")
	if got != want {
		t.Fatalf("expected %q, got %q", want, got)
	}
}

func TestResolveWorkBaseFallsBackToSystemDefaultOffWindows(t *testing.T) {
	t.Parallel()

	got := resolveWorkBase("linux", func(string) string { return "" })
	if got != "" {
		t.Fatalf("expected empty work base on non-windows, got %q", got)
	}
}

func TestCopyArtifactAcceptsSymlinkedWorkDir(t *testing.T) {
	t.Parallel()

	tmp := t.TempDir()
	realBase := filepath.Join(tmp, "real-base")
	if err := os.MkdirAll(realBase, 0o755); err != nil {
		t.Fatalf("mkdir real base: %v", err)
	}

	linkBase := filepath.Join(tmp, "link-base")
	if err := os.Symlink(realBase, linkBase); err != nil {
		t.Skipf("symlink not supported: %v", err)
	}

	workDir := filepath.Join(linkBase, "job")
	if err := os.MkdirAll(workDir, 0o755); err != nil {
		t.Fatalf("mkdir work dir: %v", err)
	}
	input := []byte("ffmpeg-artifact")
	if err := os.WriteFile(filepath.Join(workDir, "output"), input, 0o644); err != nil {
		t.Fatalf("write output: %v", err)
	}

	outBase := filepath.Join(tmp, "out")
	if err := os.MkdirAll(outBase, 0o755); err != nil {
		t.Fatalf("mkdir out base: %v", err)
	}

	path, err := copyArtifact(workDir, outBase)
	if err != nil {
		t.Fatalf("copyArtifact returned error: %v", err)
	}
	if path == "" {
		t.Fatalf("copyArtifact returned empty path")
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read copied artifact: %v", err)
	}
	if string(got) != string(input) {
		t.Fatalf("artifact mismatch: got=%q want=%q", string(got), string(input))
	}
}

func TestCopyArtifactBlocksTraversal(t *testing.T) {
	t.Parallel()

	tmp := t.TempDir()
	workDir := filepath.Join(tmp, "work")
	if err := os.MkdirAll(workDir, 0o755); err != nil {
		t.Fatalf("mkdir work dir: %v", err)
	}

	outside := filepath.Join(tmp, "outside.bin")
	if err := os.WriteFile(outside, []byte("outside"), 0o644); err != nil {
		t.Fatalf("write outside file: %v", err)
	}
	if err := os.Symlink(outside, filepath.Join(workDir, "output")); err != nil {
		t.Skipf("symlink not supported: %v", err)
	}

	outBase := filepath.Join(tmp, "out")
	if err := os.MkdirAll(outBase, 0o755); err != nil {
		t.Fatalf("mkdir out base: %v", err)
	}

	path, err := copyArtifact(workDir, outBase)
	if err != nil {
		t.Fatalf("copyArtifact returned error: %v", err)
	}
	if path != "" {
		t.Fatalf("expected traversal artifact to be blocked, got %q", path)
	}
}

func TestCopyArtifactFindsNamedOutputFromMetrics(t *testing.T) {
	t.Parallel()

	tmp := t.TempDir()
	workDir := filepath.Join(tmp, "work")
	if err := os.MkdirAll(workDir, 0o755); err != nil {
		t.Fatalf("mkdir work dir: %v", err)
	}

	output := []byte("mp4-bytes")
	if err := os.WriteFile(filepath.Join(workDir, "output.mp4"), output, 0o644); err != nil {
		t.Fatalf("write output.mp4: %v", err)
	}
	if err := os.WriteFile(filepath.Join(workDir, "metrics.json"), []byte(`{"output_name":"output.mp4"}`), 0o644); err != nil {
		t.Fatalf("write metrics.json: %v", err)
	}

	outBase := filepath.Join(tmp, "out")
	if err := os.MkdirAll(outBase, 0o755); err != nil {
		t.Fatalf("mkdir out base: %v", err)
	}

	path, err := copyArtifact(workDir, outBase)
	if err != nil {
		t.Fatalf("copyArtifact returned error: %v", err)
	}
	if path == "" {
		t.Fatal("expected artifact path")
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read copied artifact: %v", err)
	}
	if string(got) != string(output) {
		t.Fatalf("artifact mismatch: got=%q want=%q", string(got), string(output))
	}
}

func TestAgentHealthIntervalClampsOperatorOverride(t *testing.T) {
	t.Setenv("RYV_AGENT_HEALTH_INTERVAL_SECONDS", "1")
	if got := agentHealthInterval(); got != 5*time.Second {
		t.Fatalf("expected minimum 5s interval, got %v", got)
	}

	t.Setenv("RYV_AGENT_HEALTH_INTERVAL_SECONDS", "999")
	if got := agentHealthInterval(); got != 300*time.Second {
		t.Fatalf("expected maximum 300s interval, got %v", got)
	}
}

func TestValidateDownloadURLRejectsLoopbackTargets(t *testing.T) {
	t.Parallel()

	if err := validateDownloadURL("https://127.0.0.1/file", false); err == nil {
		t.Fatal("expected loopback download target to be rejected")
	}
	if err := validateDownloadURL("http://127.0.0.1/file", true); err != nil {
		t.Fatalf("expected loopback download target to be allowed when explicitly enabled, got %v", err)
	}
}

func TestValidateAgentImageRefRequiresDigestOrManagedVersionedTag(t *testing.T) {
	t.Parallel()

	if err := validateAgentImageRef("ghcr.io/ryvion/agent-runner:0.1.0"); err != nil {
		t.Fatalf("expected managed versioned tag to be allowed, got %v", err)
	}
	if err := validateAgentImageRef("ghcr.io/ryvion/agent-runner:latest"); err == nil {
		t.Fatal("expected managed latest tag to be rejected")
	}
	if err := validateAgentImageRef("docker.io/library/python:3.12"); err == nil {
		t.Fatal("expected unpinned third-party tag to be rejected")
	}
	if err := validateAgentImageRef("docker.io/library/python@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"); err != nil {
		t.Fatalf("expected digest-pinned third-party image to be allowed, got %v", err)
	}
}

func TestVerifyAgentImageSignatureUsesKeylessDefaultsForManagedImages(t *testing.T) {
	tmp := t.TempDir()
	logPath := filepath.Join(tmp, "cosign-args.txt")
	cosignPath := filepath.Join(tmp, "cosign")
	script := "#!/bin/sh\nprintf '%s\n' \"$@\" > \"" + logPath + "\"\n"
	if err := os.WriteFile(cosignPath, []byte(script), 0o755); err != nil {
		t.Fatalf("write fake cosign: %v", err)
	}

	t.Setenv("PATH", tmp+string(os.PathListSeparator)+os.Getenv("PATH"))
	if err := verifyAgentImageSignature(context.Background(), "ghcr.io/ryvion/agent-runner:0.1.1"); err != nil {
		t.Fatalf("expected verification to succeed, got %v", err)
	}

	got, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("read cosign args: %v", err)
	}
	args := strings.Split(strings.TrimSpace(string(got)), "\n")
	joined := strings.Join(args, " ")
	for _, want := range []string{
		"verify",
		"--output",
		"json",
		"--certificate-identity-regexp",
		agentCosignIdentityRegex(),
		"--certificate-oidc-issuer",
		agentCosignOIDCIssuer(),
		"ghcr.io/ryvion/agent-runner:0.1.1",
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("expected cosign args %q in %q", want, joined)
		}
	}
}

func TestVerifyAgentImageSignatureCanBeDisabled(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	t.Setenv("RYV_REQUIRE_AGENT_SIGNATURES", "0")
	if err := verifyAgentImageSignature(context.Background(), "ghcr.io/ryvion/agent-runner:0.1.1"); err != nil {
		t.Fatalf("expected signature verification to be skipped, got %v", err)
	}
}

func TestVerifyAgentImageSignatureSkipsLegacyManagedTagByDefault(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	if err := verifyAgentImageSignature(context.Background(), "ghcr.io/ryvion/agent-runner:0.1.0"); err != nil {
		t.Fatalf("expected legacy managed tag to skip signature verification, got %v", err)
	}
}
