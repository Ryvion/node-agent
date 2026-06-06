package llamacpp

import (
	"strings"
	"testing"
)

// setSpecFlagSupport forces the binary-capability probe to a fixed result and
// clears the per-path cache, restoring both afterwards.
func setSpecFlagSupport(t *testing.T, supported bool) {
	t.Helper()
	specFlagSupportMu.Lock()
	prev := specFlagSupportProbe
	specFlagSupportProbe = func(string) bool { return supported }
	specFlagSupportCache = map[string]bool{}
	specFlagSupportMu.Unlock()
	t.Cleanup(func() {
		specFlagSupportMu.Lock()
		specFlagSupportProbe = prev
		specFlagSupportCache = map[string]bool{}
		specFlagSupportMu.Unlock()
	})
}

func TestHasUnsupportedSpecFlag(t *testing.T) {
	if !hasUnsupportedSpecFlag([]string{"--host", "127.0.0.1", "--spec-type", "ngram-simple"}) {
		t.Fatal("expected --spec-type to be detected")
	}
	if !hasUnsupportedSpecFlag([]string{"--spec-draft-n-max=16"}) {
		t.Fatal("expected --spec-draft-n-max= to be detected")
	}
	if hasUnsupportedSpecFlag([]string{"--host", "127.0.0.1", "--draft-max", "16"}) {
		t.Fatal("stock --draft-max must NOT be treated as unsupported")
	}
}

func TestStripUnsupportedSpecFlags(t *testing.T) {
	in := []string{
		"--host", "127.0.0.1", "--port", "8081",
		"--spec-type", "ngram-simple",
		"--spec-draft-n-max", "16",
		"--spec-draft-n-min=2",
		"--draft-max", "8", // stock flag: must survive
	}
	got := strings.Join(stripUnsupportedSpecFlags(in), " ")
	for _, bad := range []string{"--spec-type", "--spec-draft-n-max", "--spec-draft-n-min", "ngram-simple"} {
		if strings.Contains(got, bad) {
			t.Fatalf("stripped args still contain %q: %s", bad, got)
		}
	}
	for _, keep := range []string{"--host 127.0.0.1", "--port 8081", "--draft-max 8"} {
		if !strings.Contains(got, keep) {
			t.Fatalf("stripped args dropped %q: %s", keep, got)
		}
	}
}

func TestSpecCompatibleArgsStripsWhenUnsupported(t *testing.T) {
	setSpecFlagSupport(t, false)
	args := []string{"--model", "m.gguf", "--spec-type", "ngram-simple", "--spec-draft-n-max", "16"}
	if got := specCompatibleArgs("/path/stock-llama-server", args); hasUnsupportedSpecFlag(got) {
		t.Fatalf("unsupported spec flags should have been stripped: %v", got)
	}
}

func TestSpecCompatibleArgsKeepsWhenSupported(t *testing.T) {
	setSpecFlagSupport(t, true)
	args := []string{"--model", "m.gguf", "--spec-type", "ngram-simple", "--spec-draft-n-max", "16"}
	if got := specCompatibleArgs("/path/fork-llama-server", args); !hasUnsupportedSpecFlag(got) {
		t.Fatalf("spec flags should be preserved for a supporting binary: %v", got)
	}
}

func TestServerSupportsSpecFlagsEmptyPathFailsClosed(t *testing.T) {
	if serverSupportsSpecFlags("") {
		t.Fatal("empty server path must report unsupported (fail-closed)")
	}
}

// TestNgramDefaultStrippedForStockBinary reproduces the production regression:
// the default ngram-simple config makes buildServerArgs emit the fork-only
// --spec-* flags; on a stock llama.cpp binary they must be stripped so the
// server starts instead of exiting with "invalid argument: --spec-draft-n-max".
func TestNgramDefaultStrippedForStockBinary(t *testing.T) {
	setSpecFlagSupport(t, false)
	cfg := LlamaCppSidecarConfig{
		ServerPath: "/path/stock-llama-server",
		ModelPath:  "/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		Port:       8081,
		SpecType:   SpeculativeMethodNGramSimple,
	}
	args := buildServerArgs(cfg)
	if !hasUnsupportedSpecFlag(args) {
		t.Fatalf("precondition: ngram config should emit --spec-* flags, got %v", args)
	}
	if got := specCompatibleArgs(cfg.ServerPath, args); hasUnsupportedSpecFlag(got) {
		t.Fatalf("stock binary still receives fork-only spec flags: %v", got)
	}
}
