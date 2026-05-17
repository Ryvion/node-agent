package tensoraccess

import (
	"context"
	"fmt"

	"github.com/Ryvion/ryvion-node/internal/v7/tensorplane"
)

type NoopProviderConfig struct {
	Backend     string
	RuntimeKind string
	ModelID     string
	ModelLoaded bool
	Reason      string
}

type NoopProvider struct {
	capability TensorAccessCapability
}

func NewNoopProvider(config NoopProviderConfig) *NoopProvider {
	return &NoopProvider{
		capability: NormalizeCapability(TensorAccessCapability{
			Provider:                   ProviderNoop,
			Backend:                    config.Backend,
			KVAccessSupported:          false,
			KVSnapshotSupported:        false,
			HiddenStateAccessSupported: false,
			LogitsAccessSupported:      false,
			AttentionHookSupported:     false,
			TensorPlaneDemoSupported:   false,
			ModelLoaded:                config.ModelLoaded,
			RuntimeKind:                config.RuntimeKind,
			ModelID:                    config.ModelID,
			Reason:                     config.Reason,
		}),
	}
}

func NewNativeNoopProvider(modelID string, modelLoaded bool, runtimeAvailable bool) *NoopProvider {
	backend := BackendUnknown
	reason := ReasonNativeRuntimeUnavailable
	if runtimeAvailable {
		backend = BackendNative
		reason = ReasonTextGenerationOnly
	}
	return NewNoopProvider(NoopProviderConfig{
		Backend:     backend,
		RuntimeKind: RuntimeKindNative,
		ModelID:     modelID,
		ModelLoaded: modelLoaded,
		Reason:      reason,
	})
}

func (p *NoopProvider) Name() string {
	return ProviderNoop
}

func (p *NoopProvider) Backend() string {
	return p.Capability(context.Background()).Backend
}

func (p *NoopProvider) Capability(ctx context.Context) TensorAccessCapability {
	_ = ctx
	return NormalizeCapability(p.capability)
}

func (p *NoopProvider) ListLoadedModels(ctx context.Context) ([]LoadedTensorModel, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	capability := p.Capability(ctx)
	if capability.ModelID == "" {
		return []LoadedTensorModel{}, nil
	}
	return []LoadedTensorModel{{
		ModelID:             capability.ModelID,
		RuntimeKind:         capability.RuntimeKind,
		Backend:             capability.Backend,
		Loaded:              capability.ModelLoaded,
		SupportsKV:          false,
		SupportsTensorPlane: false,
	}}, nil
}

func (p *NoopProvider) GetPage(ctx context.Context, req TensorPageRequest) (tensorplane.TensorPage, error) {
	_ = req
	if err := ctx.Err(); err != nil {
		return tensorplane.TensorPage{}, err
	}
	return tensorplane.TensorPage{}, fmt.Errorf("%w: %s", ErrTensorAccessUnsupported, p.Capability(ctx).Reason)
}

func (p *NoopProvider) GetQuery(ctx context.Context, req TensorQueryRequest) (tensorplane.AttentionQuery, error) {
	_ = req
	if err := ctx.Err(); err != nil {
		return tensorplane.AttentionQuery{}, err
	}
	return tensorplane.AttentionQuery{}, fmt.Errorf("%w: %s", ErrTensorAccessUnsupported, p.Capability(ctx).Reason)
}
