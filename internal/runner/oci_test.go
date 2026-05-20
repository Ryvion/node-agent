package runner

import (
	"context"
	"net/http"
	"net/http/httptest"
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

func TestReceiptReaderPrefersCommittedPartialFilesAndIgnoresTmp(t *testing.T) {
	t.Parallel()

	tmp := t.TempDir()
	partialReceipt := filepath.Join(tmp, "receipt.partial.json")
	if err := os.WriteFile(filepath.Join(tmp, "receipt.partial.json.tmp"), []byte(`{"output_hash":"tmp-should-not-win"}`), 0o644); err != nil {
		t.Fatalf("write tmp receipt: %v", err)
	}
	if err := os.WriteFile(partialReceipt, []byte(`{"output_hash":"sha256:partial-committed"}`), 0o644); err != nil {
		t.Fatalf("write partial receipt: %v", err)
	}

	hash := readReceiptHash(filepath.Join(tmp, "receipt.json"), partialReceipt, filepath.Join(tmp, "receipt.partial.json.tmp"))
	if hash != "partial-committed" {
		t.Fatalf("receipt hash = %q, want partial-committed", hash)
	}
}

func TestAbortAwareOCIArgsExposePartialReceiptPathsAndArtifactCandidatesSkipPartials(t *testing.T) {
	t.Parallel()

	args := baseOCIRunArgs("job-name", "/tmp/work", "512m", "1", "--network=none")
	joined := strings.Join(args, " ")
	for _, want := range []string{
		"RYV_RECEIPT_PATH=/work/receipt.json",
		"RYV_PARTIAL_RECEIPT_PATH=/work/receipt.partial.json",
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("OCI args missing %s: %v", want, args)
		}
	}

	tmp := t.TempDir()
	for _, name := range []string{"receipt.partial.json", "metrics.partial.json"} {
		if err := os.WriteFile(filepath.Join(tmp, name), []byte("control"), 0o644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}
	if err := os.WriteFile(filepath.Join(tmp, "output.bin"), []byte("artifact"), 0o644); err != nil {
		t.Fatalf("write output: %v", err)
	}
	candidates := artifactCandidates(tmp)
	for _, candidate := range candidates {
		if strings.Contains(filepath.Base(candidate), "partial") {
			t.Fatalf("partial control file considered artifact: %v", candidates)
		}
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

func TestValidateDownloadURLRejectsPrivateAndUnsafeTargets(t *testing.T) {
	t.Parallel()

	for _, rawURL := range []string{
		"http://example.com/file",
		"file:///tmp/input.bin",
		"https://10.0.0.1/file",
		"https://169.254.169.254/latest/meta-data",
		"https://100.64.0.1/file",
	} {
		if err := validateDownloadURL(rawURL, false); err == nil {
			t.Fatalf("expected %q to be rejected", rawURL)
		}
	}
}

func TestRestrictedHTTPClientRejectsPublicHTTPRedirect(t *testing.T) {
	t.Parallel()

	source := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "http://example.com/input.bin", http.StatusFound)
	}))
	defer source.Close()

	client := restrictedHTTPClient(time.Second, true)
	if transport, ok := client.Transport.(*http.Transport); ok {
		transport.TLSClientConfig = source.Client().Transport.(*http.Transport).TLSClientConfig
	}
	resp, err := client.Get(source.URL)
	if resp != nil {
		_ = resp.Body.Close()
	}
	if err == nil || !strings.Contains(err.Error(), "download url must use https") {
		t.Fatalf("expected http redirect to be rejected, got resp=%v err=%v", resp, err)
	}
}

func TestDownloadToFileEnforcesMaxPrefetchBytes(t *testing.T) {
	t.Setenv("RYV_ALLOW_LOOPBACK_DOWNLOADS", "1")
	t.Setenv("RYV_MAX_PREFETCH_BYTES", "4")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("abcdef"))
	}))
	defer server.Close()

	dest := filepath.Join(t.TempDir(), "input.bin")
	err := downloadToFile(context.Background(), server.URL+"/input.bin", dest)
	if err == nil || !strings.Contains(err.Error(), "download exceeds max size") {
		t.Fatalf("expected max size error, got %v", err)
	}
	if _, statErr := os.Stat(dest); !os.IsNotExist(statErr) {
		t.Fatalf("expected oversized partial file to be removed, statErr=%v", statErr)
	}
}
