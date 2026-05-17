package kvprobe

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestProbeNativeRuntimeUnsupportedWithoutHooks(t *testing.T) {
	result := ProbeNativeRuntime(ProbeInput{
		RuntimeAvailable: true,
		RuntimeKind:      RuntimeKindNative,
		Backend:          BackendLlamaCPP,
		ModelID:          "ryvion-llama-3.2-3b",
		ModelLoaded:      true,
	})

	if result.KVAccessSupported ||
		result.KVSnapshotSupported ||
		result.HiddenStateAccessSupported ||
		result.LogitsAccessSupported ||
		result.AttentionHookSupported {
		t.Fatalf("expected all tensor/KV hooks to be unsupported: %+v", result)
	}
	if result.Backend != BackendLlamaCPP {
		t.Fatalf("backend = %q, want %q", result.Backend, BackendLlamaCPP)
	}
	if result.ModelID != "ryvion-llama-3.2-3b" {
		t.Fatalf("model_id = %q", result.ModelID)
	}
	if !result.ModelLoaded {
		t.Fatal("model_loaded = false, want true")
	}
	if result.RuntimeKind != RuntimeKindNative {
		t.Fatalf("runtime_kind = %q, want native", result.RuntimeKind)
	}
	if result.Reason != ReasonTextGenerationOnly {
		t.Fatalf("reason = %q, want %q", result.Reason, ReasonTextGenerationOnly)
	}
}

func TestProbeNativeRuntimeReflectsModelLoadedState(t *testing.T) {
	result := ProbeNativeRuntime(ProbeInput{
		RuntimeAvailable: true,
		ModelID:          "ryvion-llama-3.2-3b",
		ModelLoaded:      false,
	})

	if result.ModelLoaded {
		t.Fatal("model_loaded = true, want false")
	}
	if result.ModelID != "ryvion-llama-3.2-3b" {
		t.Fatalf("model_id = %q", result.ModelID)
	}
	if result.Reason != ReasonTextGenerationOnly {
		t.Fatalf("reason = %q, want %q", result.Reason, ReasonTextGenerationOnly)
	}
}

func TestProbeNativeRuntimeUnavailableIsSafeUnsupported(t *testing.T) {
	result := ProbeNativeRuntime(ProbeInput{
		RuntimeAvailable: false,
		ModelID:          "ryvion-llama-3.2-3b",
		ModelLoaded:      true,
	})

	if result.Backend != BackendUnknown {
		t.Fatalf("backend = %q, want unknown", result.Backend)
	}
	if result.ModelLoaded {
		t.Fatal("model_loaded = true when runtime is unavailable")
	}
	if result.KVAccessSupported || result.KVSnapshotSupported {
		t.Fatalf("KV support should be false when runtime is unavailable: %+v", result)
	}
	if result.Reason != ReasonNativeRuntimeUnavailable {
		t.Fatalf("reason = %q, want %q", result.Reason, ReasonNativeRuntimeUnavailable)
	}
}

func TestProbeNativeRuntimeJSONFieldsAndNoPromptOrOutputText(t *testing.T) {
	result := ProbeNativeRuntime(ProbeInput{
		RuntimeAvailable: true,
		Backend:          "llama-server",
		ModelID:          "ryvion-llama-3.2-3b",
		ModelLoaded:      true,
	})
	encoded, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	text := string(encoded)
	for _, want := range []string{
		`"kv_access_supported"`,
		`"kv_snapshot_supported"`,
		`"hidden_state_access_supported"`,
		`"logits_access_supported"`,
		`"attention_hook_supported"`,
		`"backend"`,
		`"model_id"`,
		`"model_loaded"`,
		`"runtime_kind"`,
		`"reason"`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("probe JSON missing %s: %s", want, text)
		}
	}
	lower := strings.ToLower(text)
	for _, forbidden := range []string{"prompt", "output", "completion"} {
		if strings.Contains(lower, forbidden) {
			t.Fatalf("probe JSON contains forbidden raw text marker %q: %s", forbidden, text)
		}
	}
}
