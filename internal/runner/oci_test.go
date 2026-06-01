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

// TestCopyArtifactFindsArtifactInOutputSubdir covers the EM/OCI runner contract:
// the runner writes its artifact into WORK_DIR/output/<name> and reports
// output_name as just the basename. copyArtifact must still find it (regression
// for the EM dataset-delivery bug where artifacts in output/ were never found).
func TestCopyArtifactFindsArtifactInOutputSubdir(t *testing.T) {
	t.Parallel()

	tmp := t.TempDir()
	workDir := filepath.Join(tmp, "work")
	outputDir := filepath.Join(workDir, "output")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		t.Fatalf("mkdir output dir: %v", err)
	}

	artifact := []byte("npz-bytes")
	if err := os.WriteFile(filepath.Join(outputDir, "result.npz"), artifact, 0o644); err != nil {
		t.Fatalf("write result.npz: %v", err)
	}
	// metrics.json reports the basename only (as the runner does).
	if err := os.WriteFile(filepath.Join(workDir, "metrics.json"), []byte(`{"output_name":"result.npz"}`), 0o644); err != nil {
		t.Fatalf("write metrics.json: %v", err)
	}
	// A control file in the work dir must not be picked as the artifact.
	if err := os.WriteFile(filepath.Join(workDir, "receipt.json"), []byte(`{"output_hash":"x"}`), 0o644); err != nil {
		t.Fatalf("write receipt.json: %v", err)
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
		t.Fatal("expected artifact path from output/ subdir, got empty")
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read copied artifact: %v", err)
	}
	if string(got) != string(artifact) {
		t.Fatalf("artifact mismatch: got=%q want=%q", string(got), string(artifact))
	}
}

// TestCopyArtifactFromPrefersJSONMirror verifies the EM lane uploads the JSON
// mirror (which the hub's em.result.v1 reader parses) over the binary .npz when
// both are present — the fix for the EM dataset format mismatch.
func TestCopyArtifactFromPrefersJSONMirror(t *testing.T) {
	t.Parallel()

	tmp := t.TempDir()
	workDir := filepath.Join(tmp, "work")
	outputDir := filepath.Join(workDir, "output")
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		t.Fatalf("mkdir output dir: %v", err)
	}
	jsonBytes := []byte(`{"point_index":0,"fom":{"transmission_mag":0.5}}`)
	if err := os.WriteFile(filepath.Join(outputDir, "result.json"), jsonBytes, 0o644); err != nil {
		t.Fatalf("write result.json: %v", err)
	}
	if err := os.WriteFile(filepath.Join(outputDir, "result.npz"), []byte("PK\x03\x04binary-npz"), 0o644); err != nil {
		t.Fatalf("write result.npz: %v", err)
	}
	if err := os.WriteFile(filepath.Join(workDir, "metrics.json"), []byte(`{"output_name":"result.npz"}`), 0o644); err != nil {
		t.Fatalf("write metrics.json: %v", err)
	}

	outBase := filepath.Join(tmp, "out")
	if err := os.MkdirAll(outBase, 0o755); err != nil {
		t.Fatalf("mkdir out base: %v", err)
	}

	prefer := []string{filepath.Join(workDir, "output", "result.json"), filepath.Join(workDir, "result.json")}
	candidates := append(prefer, artifactCandidates(workDir)...)
	path, err := copyArtifactFrom(workDir, outBase, candidates)
	if err != nil {
		t.Fatalf("copyArtifactFrom returned error: %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read copied artifact: %v", err)
	}
	if string(got) != string(jsonBytes) {
		t.Fatalf("expected JSON mirror, got %q", string(got))
	}
}

func TestReadProbeSummaryKeepsScaledEvidenceAndDropsRawInternals(t *testing.T) {
	t.Parallel()

	tmp := t.TempDir()
	path := filepath.Join(tmp, "probe_summary.json")
	if err := os.WriteFile(path, []byte(`{
		"workgraph_id": "wg-1",
		"role_id": "target-verifier",
		"model_hash": "sha256:model",
		"probe_pack_cid": "sha256:probe",
		"feature_scores_bps": {"hallucination": 8800},
		"confidence_bps": 2500,
		"risk_flags": ["hallucination_risk"],
		"raw_activation": [0.1, 0.2],
		"raw_logits": [1, 2, 3]
	}`), 0o644); err != nil {
		t.Fatalf("write probe summary: %v", err)
	}

	got := readProbeSummary(path)
	if got["model_hash"] != "sha256:model" || got["probe_pack_cid"] != "sha256:probe" {
		t.Fatalf("safe probe summary fields missing: %#v", got)
	}
	if _, ok := got["raw_activation"]; ok {
		t.Fatalf("raw activation leaked: %#v", got)
	}
	if _, ok := got["raw_logits"]; ok {
		t.Fatalf("raw logits leaked: %#v", got)
	}
	scores, ok := got["feature_scores_bps"].(map[string]any)
	if !ok || scores["hallucination"] == nil {
		t.Fatalf("feature_scores_bps missing: %#v", got)
	}
}

func TestReceiptAndProbeReadersPreferCommittedPartialFilesAndIgnoreTmp(t *testing.T) {
	t.Parallel()

	tmp := t.TempDir()
	partialReceipt := filepath.Join(tmp, "receipt.partial.json")
	partialProbe := filepath.Join(tmp, "probe_summary.partial.json")
	if err := os.WriteFile(filepath.Join(tmp, "receipt.partial.json.tmp"), []byte(`{"output_hash":"tmp-should-not-win"}`), 0o644); err != nil {
		t.Fatalf("write tmp receipt: %v", err)
	}
	if err := os.WriteFile(partialReceipt, []byte(`{"output_hash":"sha256:partial-committed"}`), 0o644); err != nil {
		t.Fatalf("write partial receipt: %v", err)
	}
	if err := os.WriteFile(filepath.Join(tmp, "probe_summary.partial.json.tmp"), []byte(`{"raw_activation":[1,2,3]}`), 0o644); err != nil {
		t.Fatalf("write tmp probe: %v", err)
	}
	if err := os.WriteFile(partialProbe, []byte(`{"model_hash":"sha256:model","probe_pack_cid":"sha256:probe","feature_scores_bps":{"confidence":9100}}`), 0o644); err != nil {
		t.Fatalf("write partial probe: %v", err)
	}

	hash := readReceiptHash(filepath.Join(tmp, "receipt.json"), partialReceipt, filepath.Join(tmp, "receipt.partial.json.tmp"))
	if hash != "partial-committed" {
		t.Fatalf("receipt hash = %q, want partial-committed", hash)
	}
	probe := readProbeSummary(filepath.Join(tmp, "probe_summary.json"), partialProbe, filepath.Join(tmp, "probe_summary.partial.json.tmp"))
	if probe["model_hash"] != "sha256:model" {
		t.Fatalf("partial probe not read: %#v", probe)
	}
	if _, ok := probe["raw_activation"]; ok {
		t.Fatalf("tmp/raw probe leaked: %#v", probe)
	}
}

func TestAbortAwareOCIArgsExposePartialReceiptPathsAndArtifactCandidatesSkipPartials(t *testing.T) {
	t.Parallel()

	args := baseOCIRunArgs("job-name", "/tmp/work", "512m", "1", "--network=none")
	joined := strings.Join(args, " ")
	for _, want := range []string{
		"RYV_RECEIPT_PATH=/work/receipt.json",
		"RYV_PARTIAL_RECEIPT_PATH=/work/receipt.partial.json",
		"RYV_PROBE_SUMMARY_PATH=/work/probe_summary.json",
		"RYV_PARTIAL_PROBE_SUMMARY_PATH=/work/probe_summary.partial.json",
		"RYV_VERIFIER_SESSION_RECEIPT_PATH=/work/verifier_session_receipt.json",
		"RYV_PARTIAL_VERIFIER_SESSION_RECEIPT_PATH=/work/verifier_session_receipt.partial.json",
	} {
		if !strings.Contains(joined, want) {
			t.Fatalf("OCI args missing %s: %v", want, args)
		}
	}

	tmp := t.TempDir()
	for _, name := range []string{"receipt.partial.json", "probe_summary.partial.json", "metrics.partial.json", "verifier_session_receipt.json", "verifier_session_receipt.partial.json"} {
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

func TestReadVerifierSessionReceiptKeepsCommitRollbackEvidenceAndDropsRawKV(t *testing.T) {
	t.Parallel()

	tmp := t.TempDir()
	path := filepath.Join(tmp, "verifier_session_receipt.partial.json")
	if err := os.WriteFile(path, []byte(`{
		"schema_version":"ryvion.verifier_wave_receipt.v1",
		"method":"verify_tree",
		"session_id":"sess-1",
		"workgraph_id":"wg-1",
		"window_id":"win-1",
		"tree_cid":"sha256:tree",
		"kv_epoch":7,
		"accepted_len":4,
		"commit_range":{"start":0,"end":4},
		"rollback_branch_ids":["br-reject"],
		"verifier_signature":"sig",
		"raw_kv_cache":[1,2,3],
		"candidate_text":"secret text"
	}`), 0o644); err != nil {
		t.Fatalf("write verifier receipt: %v", err)
	}

	got := readVerifierSessionReceipt(filepath.Join(tmp, "verifier_session_receipt.json"), path)
	if got["method"] != "verify_tree" || got["tree_cid"] != "sha256:tree" {
		t.Fatalf("safe verifier receipt missing: %#v", got)
	}
	if _, ok := got["raw_kv_cache"]; ok {
		t.Fatalf("raw kv leaked: %#v", got)
	}
	if _, ok := got["candidate_text"]; ok {
		t.Fatalf("candidate text leaked: %#v", got)
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
