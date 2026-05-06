package tensoraccess

import (
	"context"
	"errors"
	"testing"

	"github.com/Ryvion/node-agent/internal/v7/tensorplane"
)

func TestNoopProviderReturnsUnsupportedCapabilityAndErrors(t *testing.T) {
	provider := NewNativeNoopProvider("ryvion-llama-3.2-3b", true, true)

	capability := provider.Capability(context.Background())
	if capability.Provider != ProviderNoop {
		t.Fatalf("provider = %q, want %q", capability.Provider, ProviderNoop)
	}
	if capability.Backend != BackendNative {
		t.Fatalf("backend = %q, want %q", capability.Backend, BackendNative)
	}
	if capability.KVAccessSupported ||
		capability.KVSnapshotSupported ||
		capability.HiddenStateAccessSupported ||
		capability.LogitsAccessSupported ||
		capability.AttentionHookSupported ||
		capability.TensorPlaneDemoSupported {
		t.Fatalf("noop capability should not advertise tensor support: %+v", capability)
	}
	if !capability.ModelLoaded {
		t.Fatal("model_loaded = false, want true for loaded native model")
	}
	if capability.Reason != ReasonTextGenerationOnly {
		t.Fatalf("reason = %q, want %q", capability.Reason, ReasonTextGenerationOnly)
	}

	_, err := provider.GetPage(context.Background(), TensorPageRequest{
		ModelID:    "ryvion-llama-3.2-3b",
		LayerIndex: 0,
		HeadStart:  0,
		HeadCount:  1,
		TokenStart: 0,
		TokenCount: 4,
		DType:      tensorplane.TensorDTypeFloat32,
		PageID:     "page-1",
	})
	if !errors.Is(err, ErrTensorAccessUnsupported) {
		t.Fatalf("GetPage() error = %v, want unsupported", err)
	}

	_, err = provider.GetQuery(context.Background(), TensorQueryRequest{
		RequestID:  "req-1",
		JobID:      "job-1",
		ModelID:    "ryvion-llama-3.2-3b",
		LayerIndex: 0,
		HeadIndex:  0,
		DType:      tensorplane.TensorDTypeFloat32,
		HeadDim:    4,
	})
	if !errors.Is(err, ErrTensorAccessUnsupported) {
		t.Fatalf("GetQuery() error = %v, want unsupported", err)
	}
}

func TestNoopProviderListLoadedModelsDoesNotClaimKVOrTensorPlane(t *testing.T) {
	provider := NewNativeNoopProvider("ryvion-llama-3.2-3b", true, true)

	models, err := provider.ListLoadedModels(context.Background())
	if err != nil {
		t.Fatalf("ListLoadedModels() error = %v", err)
	}
	if len(models) != 1 {
		t.Fatalf("models length = %d, want 1", len(models))
	}
	if models[0].SupportsKV || models[0].SupportsTensorPlane {
		t.Fatalf("noop loaded model should not claim tensor support: %+v", models[0])
	}
}
