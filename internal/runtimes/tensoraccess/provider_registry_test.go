package tensoraccess

import (
	"context"
	"testing"

	"github.com/Ryvion/ryvion-node/internal/v7/tensorplane"
)

func TestGetTensorAccessProviderDefaultsToNoopForRuntimeAccess(t *testing.T) {
	provider := GetTensorAccessProvider(TensorAccessProviderConfig{
		Getenv: func(string) string { return "" },
	})

	if provider.Name() != ProviderNoop {
		t.Fatalf("provider = %q, want %q", provider.Name(), ProviderNoop)
	}
	if provider.Capability(context.Background()).TensorPlaneDemoSupported {
		t.Fatalf("default runtime provider should not enable demo tensor access: %+v", provider.Capability(context.Background()))
	}
}

func TestGetTensorAccessProviderSelectsDemoFromEnv(t *testing.T) {
	provider := GetTensorAccessProvider(TensorAccessProviderConfig{
		Getenv: func(key string) string {
			if key == TensorAccessProviderEnv {
				return ProviderTensorPlaneDemo
			}
			return ""
		},
		ModelID:  "tensorplane/demo-model",
		Tokens:   8,
		HeadDim:  4,
		ValueDim: 3,
		DType:    tensorplane.TensorDTypeFloat32,
		Seed:     42,
	})

	if provider.Name() != ProviderTensorPlaneDemo {
		t.Fatalf("provider = %q, want %q", provider.Name(), ProviderTensorPlaneDemo)
	}
	capability := provider.Capability(context.Background())
	if capability.Provider != ProviderTensorPlaneDemo || !capability.TensorPlaneDemoSupported {
		t.Fatalf("capability = %+v, want tensorplane demo support", capability)
	}
}

func TestGetTensorAccessProviderExplicitProviderOverridesEnv(t *testing.T) {
	provider := GetTensorAccessProvider(TensorAccessProviderConfig{
		Provider: ProviderNoop,
		Getenv: func(key string) string {
			if key == TensorAccessProviderEnv {
				return ProviderTensorPlaneDemo
			}
			return ""
		},
		DefaultProvider: ProviderTensorPlaneDemo,
	})

	if provider.Name() != ProviderNoop {
		t.Fatalf("provider = %q, want explicit noop", provider.Name())
	}
}

func TestGetTensorAccessProviderUsesProbeDefaultWhenConfigured(t *testing.T) {
	provider := GetTensorAccessProvider(TensorAccessProviderConfig{
		DefaultProvider: ProviderTensorPlaneDemo,
		Getenv:          func(string) string { return "" },
		ModelID:         "tensorplane/demo-model",
		Tokens:          4,
		HeadDim:         2,
		ValueDim:        2,
		DType:           tensorplane.TensorDTypeFloat32,
		Seed:            7,
	})

	if provider.Name() != ProviderTensorPlaneDemo {
		t.Fatalf("provider = %q, want probe default demo", provider.Name())
	}
}

func TestGetTensorAccessProviderUnknownSelectionFallsBackToNoop(t *testing.T) {
	provider := GetTensorAccessProvider(TensorAccessProviderConfig{
		Provider: "future-provider",
		Getenv:   func(string) string { return "" },
	})

	if provider.Name() != ProviderNoop {
		t.Fatalf("provider = %q, want noop fallback", provider.Name())
	}
}
