package inference

import (
	"strings"
	"testing"
)

func withSpecProbe(t *testing.T, supported bool) {
	t.Helper()
	prev := specFlagSupportProbe
	specFlagSupportMu.Lock()
	specFlagSupportCache = map[string]bool{}
	specFlagSupportMu.Unlock()
	specFlagSupportProbe = func(string) bool { return supported }
	t.Cleanup(func() {
		specFlagSupportProbe = prev
		specFlagSupportMu.Lock()
		specFlagSupportCache = map[string]bool{}
		specFlagSupportMu.Unlock()
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
	if strings.Contains(joined(got), " --spec-type ") {
		t.Fatalf("expected --spec-type stripped for stock binary, got %v", got)
	}
	// The fork-only flag's value must go too, but unrelated flags must remain.
	for _, must := range []string{"--model", "--port", "--draft-max", "--n-gpu-layers"} {
		if !strings.Contains(joined(got), " "+must+" ") {
			t.Errorf("stripping removed unrelated flag %q: %v", must, got)
		}
	}
	if strings.Contains(joined(got), " ngram-simple ") {
		t.Errorf("expected --spec-type value dropped, got %v", got)
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
	args := []string{"--model", "/m/x.gguf", "--spec-type=ngram-simple", "--ctx-size", "16384"}
	got := specCompatibleArgs("/bin/llama-server", args)
	if strings.Contains(joined(got), "--spec-type") {
		t.Fatalf("expected --spec-type=... stripped, got %v", got)
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
	specFlagSupportProbe = func(string) bool { return true } // even if probe would say yes
	t.Cleanup(func() { specFlagSupportProbe = prev })
	args := []string{"--spec-type", "ngram-simple", "--model", "/m/x.gguf"}
	got := specCompatibleArgs("", args)
	if strings.Contains(joined(got), " --spec-type ") {
		t.Fatalf("expected fail-closed strip for empty server path, got %v", got)
	}
}
