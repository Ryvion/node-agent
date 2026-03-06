package runner

import (
	"os"
	"path/filepath"
	"testing"
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
