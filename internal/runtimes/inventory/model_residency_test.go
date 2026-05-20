package runtimeinventory

import (
	"strings"
	"testing"

	"github.com/Ryvion/ryvion-node/internal/runtimes/tensoraccess"
)

func TestBuildModelResidencySnapshotReflectsNativeLoadedWarmModel(t *testing.T) {
	t.Parallel()

	models := BuildModelResidencySnapshot(RuntimeStatus{
		RuntimeKind:             " NATIVE ",
		Backend:                 " NATIVE ",
		Provider:                " NOOP ",
		ProcessMode:             ProcessModeSidecar,
		NativeInferenceReady:    true,
		NativeModel:             " ryvion-llama-3.2-3b ",
		ModelLoaded:             true,
		SupportsTextGeneration:  true,
		SupportsStreaming:       true,
		SupportsTensorPlaneDemo: true,
	})

	if len(models) != 1 {
		t.Fatalf("models length = %d, want 1", len(models))
	}
	model := models[0]
	if model.ModelID != "ryvion-llama-3.2-3b" {
		t.Fatalf("model_id = %q", model.ModelID)
	}
	if model.RuntimeKind != RuntimeKindNative || model.Backend != BackendNative {
		t.Fatalf("runtime/backend = %+v", model)
	}
	if !model.Loaded || !model.Warm || !model.SupportsTextGeneration || !model.SupportsStreaming {
		t.Fatalf("loaded/warm/text support = %+v", model)
	}
	if model.SupportsKVAccess || model.SupportsTensorPlane {
		t.Fatalf("model residency should not claim real KV/TensorPlane hooks: %+v", model)
	}
	if !model.SupportsTensorPlaneDemo {
		t.Fatalf("supports_tensorplane_demo = false: %+v", model)
	}
	if model.Reason != tensoraccess.ReasonTextGenerationOnly {
		t.Fatalf("reason = %q, want %q", model.Reason, tensoraccess.ReasonTextGenerationOnly)
	}
}

func TestBuildModelResidencySnapshotSanitizesIdentifiers(t *testing.T) {
	t.Parallel()

	models := BuildModelResidencySnapshot(RuntimeStatus{
		RuntimeKind: RuntimeKindNative,
		Backend:     BackendNative,
		NativeModel: strings.Repeat("m", 140) + "\nsecret",
		ModelLoaded: true,
		ProcessMode: ProcessModeSidecar,
	})

	if len(models) != 1 {
		t.Fatalf("models length = %d, want 1", len(models))
	}
	if len(models[0].ModelID) != maxInventoryModelIDLen {
		t.Fatalf("model_id length = %d, want %d", len(models[0].ModelID), maxInventoryModelIDLen)
	}
	if strings.ContainsAny(models[0].ModelID, "\n\t") {
		t.Fatalf("model_id contains control whitespace: %q", models[0].ModelID)
	}
}

func TestBuildModelResidencySnapshotEmptyWhenModelUnknown(t *testing.T) {
	t.Parallel()

	models := BuildModelResidencySnapshot(RuntimeStatus{
		RuntimeKind:          RuntimeKindUnknown,
		Backend:              BackendUnknown,
		NativeInferenceReady: false,
		ModelLoaded:          false,
	})

	if len(models) != 0 {
		t.Fatalf("models = %+v, want empty", models)
	}
}
