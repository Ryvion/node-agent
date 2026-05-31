package runner

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestNativeEMTimeoutClamps(t *testing.T) {
	if got := nativeEMTimeout(0); got != nativeEMDefaultTimeout {
		t.Fatalf("zero budget should use default, got %v", got)
	}
	// 2x margin, clamped to >= 1m.
	if got := nativeEMTimeout(10); got.Minutes() != 1 {
		t.Fatalf("tiny budget should clamp to 1m, got %v", got)
	}
	// Hard cap at 4h.
	if got := nativeEMTimeout(100000); got.Hours() != 4 {
		t.Fatalf("huge budget should clamp to 4h, got %v", got)
	}
}

func TestDeriveEMBundleURL(t *testing.T) {
	t.Setenv("RYV_EM_BUNDLE_BASE_URL", "https://cdn.example.com/em/")
	m := emBundleManifest{Engine: "gprmax", EngineVersion: "4.2.0", OS: "linux", Arch: "amd64"}
	got := deriveEMBundleURL(m)
	want := "https://cdn.example.com/em/gprmax/4.2.0/linux-amd64.tar.gz"
	if got != want {
		t.Fatalf("derive url: got %q want %q", got, want)
	}
	mwin := emBundleManifest{Engine: "openems", EngineVersion: "0.0.36", OS: "windows", Arch: "amd64"}
	if got := deriveEMBundleURL(mwin); filepath.Ext(got) != ".zip" {
		t.Fatalf("windows bundle should be .zip, got %q", got)
	}
}

func TestDeriveEMBundleURLEmptyWithoutBase(t *testing.T) {
	t.Setenv("RYV_EM_BUNDLE_BASE_URL", "")
	if got := deriveEMBundleURL(emBundleManifest{Engine: "gprmax", EngineVersion: "1"}); got != "" {
		t.Fatalf("no base url should yield empty, got %q", got)
	}
}

func TestVerifyManifestSignatureUnsignedGate(t *testing.T) {
	t.Setenv("RYV_EM_BUNDLE_PUBKEY", "")
	t.Setenv("RYV_EM_ALLOW_UNSIGNED_BUNDLE", "")
	if err := verifyManifestSignature(emBundleManifest{Engine: "gprmax"}); err == nil {
		t.Fatal("unsigned bundle without pubkey/override must be rejected")
	}
	t.Setenv("RYV_EM_ALLOW_UNSIGNED_BUNDLE", "1")
	if err := verifyManifestSignature(emBundleManifest{Engine: "gprmax"}); err != nil {
		t.Fatalf("override should allow unsigned in dev, got %v", err)
	}
}

func TestEnsureEMBundleCacheHit(t *testing.T) {
	root := t.TempDir()
	t.Setenv("RYVION_EM_RUNTIME_ROOT", root)
	t.Setenv("RYV_EM_ALLOW_UNSIGNED_BUNDLE", "1")

	entryName := "run.py"
	bundleDir := filepath.Join(root, "gprmax-4.2.0")
	if err := os.MkdirAll(bundleDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(bundleDir, entryName), []byte("# stub"), 0o644); err != nil {
		t.Fatal(err)
	}
	// Ready marker with empty sha -> treated as ready.
	if err := os.WriteFile(filepath.Join(bundleDir, emBundleReadyMarker), []byte(""), 0o644); err != nil {
		t.Fatal(err)
	}

	m := emBundleManifest{Engine: "gprmax", EngineVersion: "4.2.0", Entrypoint: entryName}
	got, err := ensureEMBundle(context.Background(), m, "")
	if err != nil {
		t.Fatalf("cache hit should not error: %v", err)
	}
	if got != filepath.Join(bundleDir, entryName) {
		t.Fatalf("unexpected entrypoint: %q", got)
	}
}

func TestEnsureEMBundleMissingNoURL(t *testing.T) {
	root := t.TempDir()
	t.Setenv("RYVION_EM_RUNTIME_ROOT", root)
	t.Setenv("RYV_EM_ALLOW_UNSIGNED_BUNDLE", "1")
	m := emBundleManifest{Engine: "gprmax", EngineVersion: "9.9.9", Entrypoint: "run.py"}
	if _, err := ensureEMBundle(context.Background(), m, ""); err == nil {
		t.Fatal("uncached bundle with no URL must error")
	}
}

func TestSafeJoinRejectsTraversal(t *testing.T) {
	dest := t.TempDir()
	if _, err := safeJoin(dest, "../escape.txt"); err == nil {
		t.Fatal("path traversal must be rejected")
	}
	if _, err := safeJoin(dest, "ok/inside.txt"); err != nil {
		t.Fatalf("legit path should be allowed: %v", err)
	}
}

func TestNativeEMProcessEnvIsolated(t *testing.T) {
	env := nativeEMProcessEnv("/work/x", "", "all", emBudget{EstVRAMMB: 8000, MaxCells: 1000})
	var hasOffline, hasReceipt, hasGpus, hasWorkDir bool
	for _, e := range env {
		switch {
		case e == "RYV_EM_OFFLINE=1":
			hasOffline = true
		case e == "RYV_RECEIPT_PATH="+filepath.Join("/work/x", "receipt.json"):
			hasReceipt = true
		case e == "RYV_EM_GPUS=all":
			hasGpus = true
		case e == "RYVION_WORK_DIR=/work/x":
			hasWorkDir = true
		}
	}
	if !hasOffline || !hasReceipt || !hasGpus || !hasWorkDir {
		t.Fatalf("env missing required keys: offline=%v receipt=%v gpus=%v workdir=%v",
			hasOffline, hasReceipt, hasGpus, hasWorkDir)
	}
}

func TestEMCudaVisibleDevices(t *testing.T) {
	cases := map[string]string{
		"": "", "auto": "", "all": "", "none": "-1",
		"0": "0", "0,1": "0,1", "device=0": "0",
	}
	for in, want := range cases {
		if got := emCudaVisibleDevices(in); got != want {
			t.Fatalf("emCudaVisibleDevices(%q)=%q want %q", in, got, want)
		}
	}
}

func TestEMBundleRootAndEmbeddedPython(t *testing.T) {
	root := t.TempDir()
	entry := filepath.Join(root, "runner", "run.py")
	if got := emBundleRoot(entry); got != root {
		t.Fatalf("emBundleRoot(%q)=%q want %q", entry, got, root)
	}
	if got := emEmbeddedPython(root); got != "" {
		t.Fatalf("expected no embedded python, got %q", got)
	}
	if runtime.GOOS != "windows" {
		py := filepath.Join(root, "python", "bin", "python3")
		if err := os.MkdirAll(filepath.Dir(py), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(py, []byte("#!/bin/sh\n"), 0o755); err != nil {
			t.Fatal(err)
		}
		if got := emEmbeddedPython(root); got != py {
			t.Fatalf("embedded python: got %q want %q", got, py)
		}
	}
}

func TestDefaultEMEntrypointIsRunnerRunPy(t *testing.T) {
	if got := defaultEMEntrypoint(); got != "runner/run.py" {
		t.Fatalf("default entrypoint should be runner/run.py, got %q", got)
	}
}

func TestDefaultEMEntrypoint(t *testing.T) {
	if defaultEMEntrypoint() == "" {
		t.Fatal("default entrypoint must be non-empty")
	}
	_ = runtime.GOOS
}

func TestEMMemoryCapBytes(t *testing.T) {
	t.Setenv("RYV_EM_RAM_CAP_MB", "")
	// No VRAM budget -> no derivable cap (caller skips the cap).
	if got := emMemoryCapBytes(emBudget{}); got != 0 {
		t.Fatalf("zero budget should yield no cap, got %d", got)
	}
	// 8000 MB VRAM -> 2x + 1 GiB floor headroom = 17024 MB.
	want := uint64(8000*2+1024) << 20
	if got := emMemoryCapBytes(emBudget{EstVRAMMB: 8000}); got != want {
		t.Fatalf("vram-derived cap: got %d want %d", got, want)
	}
	// Explicit override wins regardless of budget.
	t.Setenv("RYV_EM_RAM_CAP_MB", "2048")
	if got := emMemoryCapBytes(emBudget{EstVRAMMB: 8000}); got != uint64(2048)<<20 {
		t.Fatalf("override should win, got %d", got)
	}
}

func TestEMCPUQuotaPercent(t *testing.T) {
	t.Setenv("RYV_EM_CPU_CORES", "")
	if got := emCPUQuotaPercent(); got != 400 {
		t.Fatalf("default should be 4 cores (400%%), got %d", got)
	}
	t.Setenv("RYV_EM_CPU_CORES", "2")
	if got := emCPUQuotaPercent(); got != 200 {
		t.Fatalf("2 cores should be 200%%, got %d", got)
	}
	t.Setenv("RYV_EM_CPU_CORES", "0.5")
	if got := emCPUQuotaPercent(); got != 50 {
		t.Fatalf("half core should be 50%%, got %d", got)
	}
}
