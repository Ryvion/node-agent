package tensoraccess

import (
	"strings"
	"testing"
)

func TestNormalizeCapabilitySanitizesCompactFields(t *testing.T) {
	capability := NormalizeCapability(TensorAccessCapability{
		Provider:                   "  NOOP  ",
		Backend:                    "  NATIVE  ",
		RuntimeKind:                "  NATIVE  ",
		ModelID:                    "  " + strings.Repeat("m", 160) + "\n",
		Reason:                     "  " + strings.Repeat("r", 300) + "\t",
		TensorPlaneDemoSupported:   true,
		KVAccessSupported:          false,
		KVSnapshotSupported:        false,
		HiddenStateAccessSupported: false,
		LogitsAccessSupported:      false,
		AttentionHookSupported:     false,
	})

	if capability.Provider != ProviderNoop {
		t.Fatalf("provider = %q, want %q", capability.Provider, ProviderNoop)
	}
	if capability.Backend != BackendNative {
		t.Fatalf("backend = %q, want %q", capability.Backend, BackendNative)
	}
	if capability.RuntimeKind != RuntimeKindNative {
		t.Fatalf("runtime_kind = %q, want %q", capability.RuntimeKind, RuntimeKindNative)
	}
	if len(capability.ModelID) != maxCapabilityModelIDLen {
		t.Fatalf("model_id length = %d, want %d", len(capability.ModelID), maxCapabilityModelIDLen)
	}
	if len(capability.Reason) != maxCapabilityReasonLen {
		t.Fatalf("reason length = %d, want %d", len(capability.Reason), maxCapabilityReasonLen)
	}
	if strings.ContainsAny(capability.ModelID, " \n\t") || strings.ContainsAny(capability.Reason, "\n\t") {
		t.Fatalf("capability text contains unsanitized whitespace: %+v", capability)
	}
	if !capability.TensorPlaneDemoSupported {
		t.Fatalf("tensorplane_demo_supported was not preserved: %+v", capability)
	}
}

func TestNormalizeCapabilityDefaultsUnavailableNoop(t *testing.T) {
	capability := NormalizeCapability(TensorAccessCapability{
		Provider:    "unsupported-provider",
		Backend:     "unsupported-backend",
		RuntimeKind: "unsupported-runtime",
		ModelLoaded: true,
	})

	if capability.Provider != ProviderNoop {
		t.Fatalf("provider = %q, want noop", capability.Provider)
	}
	if capability.Backend != BackendUnknown {
		t.Fatalf("backend = %q, want unknown", capability.Backend)
	}
	if capability.RuntimeKind != RuntimeKindUnknown {
		t.Fatalf("runtime_kind = %q, want unknown", capability.RuntimeKind)
	}
	if capability.ModelLoaded {
		t.Fatalf("model_loaded = true for unknown backend: %+v", capability)
	}
	if capability.Reason != ReasonNativeRuntimeUnavailable {
		t.Fatalf("reason = %q, want %q", capability.Reason, ReasonNativeRuntimeUnavailable)
	}
}
