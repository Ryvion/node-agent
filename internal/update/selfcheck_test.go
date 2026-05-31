package update

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// makeScript writes an executable shell script and returns its path. Unix only
// (the self-check just runs `<path> --version`; a shell script is a stand-in for
// a real release binary so we can exercise the gate without building one).
func makeScript(t *testing.T, dir, name, body string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestSelfCheckBinaryAcceptsWorking(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell-script stand-in is unix-only")
	}
	p := makeScript(t, t.TempDir(), "good", "#!/bin/sh\necho 'ryvion-node 1.2.3'\n")
	if err := selfCheckBinary(p); err != nil {
		t.Fatalf("a binary that prints its version should pass: %v", err)
	}
}

func TestSelfCheckBinaryRejectsCrashing(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell-script stand-in is unix-only")
	}
	p := makeScript(t, t.TempDir(), "bad", "#!/bin/sh\nexit 1\n")
	if err := selfCheckBinary(p); err == nil {
		t.Fatal("a binary that exits non-zero on --version must be rejected")
	}
}

func TestSelfCheckBinaryRejectsWrongOutput(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("shell-script stand-in is unix-only")
	}
	p := makeScript(t, t.TempDir(), "weird", "#!/bin/sh\necho 'not the right program'\n")
	if err := selfCheckBinary(p); err == nil {
		t.Fatal("a binary whose --version isn't ryvion-node must be rejected")
	}
}

func TestSelfCheckBinaryDisabledByEnv(t *testing.T) {
	t.Setenv("RYV_DISABLE_UPDATE_SELFCHECK", "1")
	if err := selfCheckBinary(filepath.Join(t.TempDir(), "does-not-exist")); err != nil {
		t.Fatalf("opt-out env should skip the self-check, got %v", err)
	}
}

// The whole point: a BROKEN staged binary must leave the current working binary
// in place, so the node keeps running and can pull the next (fixed) release.
func TestReplaceUnixKeepsCurrentBinaryWhenStagedBad(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("replaceUnix is the non-Windows path")
	}
	dir := t.TempDir()
	const working = "#!/bin/sh\necho 'ryvion-node 9.9.9'\n"
	current := makeScript(t, dir, "ryvion-node", working)

	// Staged "binary" that cannot start cleanly (exits non-zero on --version).
	bad := []byte("#!/bin/sh\nexit 3\n")
	err := replaceUnix(current, bad)
	if err == nil {
		t.Fatal("replaceUnix must refuse a staged binary that fails self-check")
	}

	got, readErr := os.ReadFile(current)
	if readErr != nil {
		t.Fatalf("current binary should still exist: %v", readErr)
	}
	if string(got) != working {
		t.Fatal("current working binary was modified despite a failed update — brick risk")
	}
	// No leftover temp binaries in the dir.
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if e.Name() != "ryvion-node" {
			t.Fatalf("leftover staged file after rejected update: %s", e.Name())
		}
	}
}
