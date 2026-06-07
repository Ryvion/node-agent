package inference

import (
	"strings"
	"testing"
)

func withSpecProbe(t *testing.T, supported bool) {
	t.Helper()
	prev := specFlagSupportProbe
	prevLegacy := legacySpecFlagSupportProbe
	specFlagSupportMu.Lock()
	specFlagSupportCache = map[string]bool{}
	specFlagSupportMu.Unlock()
	legacySpecFlagSupportMu.Lock()
	legacySpecFlagSupportMap = map[string]bool{}
	legacySpecFlagSupportMu.Unlock()
	specFlagSupportProbe = func(string) bool { return supported }
	legacySpecFlagSupportProbe = func(string) bool { return supported }
	t.Cleanup(func() {
		specFlagSupportProbe = prev
		legacySpecFlagSupportProbe = prevLegacy
		specFlagSupportMu.Lock()
		specFlagSupportCache = map[string]bool{}
		specFlagSupportMu.Unlock()
		legacySpecFlagSupportMu.Lock()
		legacySpecFlagSupportMap = map[string]bool{}
		legacySpecFlagSupportMu.Unlock()
	})
}

func joined(args []string) string { return " " + strings.Join(args, " ") + " " }

func TestSpecCompatibleArgs_StripsWhenUnsupported(t *testing.T) {
	withSpecProbe(t, false)
	args := []string{
		"--model", "/m/phi-4.gguf", "--port", "8081",
		"--spec-type", "ngram-simple", "--draft-max", "8",
		"--n-gpu-layers", "99",
	}
	got := specCompatibleArgs("/bin/llama-server", args)
	for _, removed := range []string{"--spec-type", "--draft-max", "ngram-simple"} {
		if strings.Contains(joined(got), " "+removed+" ") {
			t.Fatalf("expected legacy speculative flag/value %q stripped for incompatible binary, got %v", removed, got)
		}
	}
	// Unrelated flags must remain.
	for _, must := range []string{"--model", "--port", "--n-gpu-layers"} {
		if !strings.Contains(joined(got), " "+must+" ") {
			t.Errorf("stripping removed unrelated flag %q: %v", must, got)
		}
	}
}

func TestSpecCompatibleArgs_KeepsWhenSupported(t *testing.T) {
	withSpecProbe(t, true)
	args := []string{"--model", "/m/phi-4.gguf", "--spec-type", "ngram-simple", "--draft-max", "8"}
	got := specCompatibleArgs("/bin/llama-server-fork", args)
	if !strings.Contains(joined(got), " --spec-type ") {
		t.Fatalf("expected --spec-type kept for fork binary, got %v", got)
	}
}

func TestSpecCompatibleArgs_EqualsForm(t *testing.T) {
	withSpecProbe(t, false)
	args := []string{"--model", "/m/x.gguf", "--spec-type=ngram-simple", "--draft-max=8", "--ctx-size", "16384"}
	got := specCompatibleArgs("/bin/llama-server", args)
	if strings.Contains(joined(got), "--spec-type") {
		t.Fatalf("expected --spec-type=... stripped, got %v", got)
	}
	if strings.Contains(joined(got), "--draft-max") {
		t.Fatalf("expected --draft-max=... stripped, got %v", got)
	}
	if !strings.Contains(joined(got), " --ctx-size ") {
		t.Errorf("expected --ctx-size preserved, got %v", got)
	}
}

func TestSpecCompatibleArgs_NoSpecFlagsIsNoOp(t *testing.T) {
	withSpecProbe(t, false)
	args := []string{"--model", "/m/x.gguf", "--port", "8081", "--n-gpu-layers", "99"}
	got := specCompatibleArgs("/bin/llama-server", args)
	if len(got) != len(args) {
		t.Fatalf("expected no-op when no spec flags present, got %v", got)
	}
}

func TestSpecCompatibleArgs_EmptyPathFailsClosed(t *testing.T) {
	// Empty server path must be treated as unsupported (fail-closed) and strip.
	prev := specFlagSupportProbe
	prevLegacy := legacySpecFlagSupportProbe
	specFlagSupportProbe = func(string) bool { return true } // even if probe would say yes
	legacySpecFlagSupportProbe = func(string) bool { return true }
	t.Cleanup(func() {
		specFlagSupportProbe = prev
		legacySpecFlagSupportProbe = prevLegacy
	})
	args := []string{"--spec-type", "ngram-simple", "--draft-max", "8", "--model", "/m/x.gguf"}
	got := specCompatibleArgs("", args)
	if strings.Contains(joined(got), " --spec-type ") || strings.Contains(joined(got), " --draft-max ") {
		t.Fatalf("expected fail-closed strip for empty server path, got %v", got)
	}
}
