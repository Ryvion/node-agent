package tensoraccess

import (
	"context"
	"os"
	"strings"

	"github.com/Ryvion/ryvion-node/internal/v7/tensorplane"
)

const TensorAccessProviderEnv = "RYV_TENSOR_ACCESS_PROVIDER"

type TensorAccessProviderConfig struct {
	Provider              string
	DefaultProvider       string
	Getenv                func(string) string
	ModelID               string
	RuntimeKind           string
	Tokens                int
	HeadDim               int
	ValueDim              int
	DType                 tensorplane.TensorDType
	Seed                  int64
	NoopConfig            NoopProviderConfig
	TensorPlaneDemoConfig TensorPlaneDemoProviderConfig
}

func init() {
	tensorplane.RegisterProviderBackedProbeResolver(resolveTensorPlaneProbeProvider)
}

func GetTensorAccessProvider(config TensorAccessProviderConfig) TensorAccessProvider {
	switch selectedTensorAccessProvider(config) {
	case ProviderTensorPlaneDemo:
		return NewTensorPlaneDemoProvider(tensorPlaneDemoProviderConfigFromRegistry(config))
	default:
		return NewNoopProvider(noopProviderConfigFromRegistry(config))
	}
}

func selectedTensorAccessProvider(config TensorAccessProviderConfig) string {
	if selected := strings.TrimSpace(config.Provider); selected != "" {
		return normalizeProvider(selected)
	}
	getenv := config.Getenv
	if getenv == nil {
		getenv = os.Getenv
	}
	if selected := strings.TrimSpace(getenv(TensorAccessProviderEnv)); selected != "" {
		return normalizeProvider(selected)
	}
	if selected := strings.TrimSpace(config.DefaultProvider); selected != "" {
		return normalizeProvider(selected)
	}
	return ProviderNoop
}

func tensorPlaneDemoProviderConfigFromRegistry(config TensorAccessProviderConfig) TensorPlaneDemoProviderConfig {
	demo := config.TensorPlaneDemoConfig
	if strings.TrimSpace(demo.ModelID) == "" {
		demo.ModelID = config.ModelID
	}
	if strings.TrimSpace(demo.RuntimeKind) == "" {
		demo.RuntimeKind = firstNonEmpty(config.RuntimeKind, RuntimeKindDemo)
	}
	if demo.Tokens <= 0 {
		demo.Tokens = config.Tokens
	}
	if demo.HeadDim <= 0 {
		demo.HeadDim = config.HeadDim
	}
	if demo.ValueDim <= 0 {
		demo.ValueDim = config.ValueDim
	}
	if demo.DType == "" {
		demo.DType = config.DType
	}
	if demo.Seed == 0 {
		demo.Seed = config.Seed
	}
	if demo.ContextLength <= 0 {
		demo.ContextLength = demo.Tokens
	}
	if demo.Heads <= 0 {
		demo.Heads = 1
	}
	if demo.Layers <= 0 {
		demo.Layers = 1
	}
	return demo
}

func noopProviderConfigFromRegistry(config TensorAccessProviderConfig) NoopProviderConfig {
	noop := config.NoopConfig
	if strings.TrimSpace(noop.ModelID) == "" {
		noop.ModelID = config.ModelID
	}
	if strings.TrimSpace(noop.RuntimeKind) == "" {
		noop.RuntimeKind = firstNonEmpty(config.RuntimeKind, RuntimeKindNative)
	}
	return noop
}

func resolveTensorPlaneProbeProvider(config tensorplane.ProviderBackedProbeProviderConfig) tensorplane.ProviderBackedProbeProvider {
	provider := GetTensorAccessProvider(TensorAccessProviderConfig{
		Provider:        config.Provider,
		DefaultProvider: config.DefaultProvider,
		Getenv:          config.Getenv,
		ModelID:         config.ModelID,
		RuntimeKind:     RuntimeKindDemo,
		Tokens:          config.Tokens,
		HeadDim:         config.HeadDim,
		ValueDim:        config.ValueDim,
		DType:           config.DType,
		Seed:            config.Seed,
		NoopConfig: NoopProviderConfig{
			ModelID: config.ModelID,
		},
		TensorPlaneDemoConfig: TensorPlaneDemoProviderConfig{
			ModelID:     config.ModelID,
			RuntimeKind: RuntimeKindDemo,
			Tokens:      config.Tokens,
			HeadDim:     config.HeadDim,
			ValueDim:    config.ValueDim,
			DType:       config.DType,
			Seed:        config.Seed,
		},
	})
	return tensorPlaneProbeProviderAdapter{provider: provider}
}

type tensorPlaneProbeProviderAdapter struct {
	provider TensorAccessProvider
}

func (a tensorPlaneProbeProviderAdapter) Name() string {
	if a.provider == nil {
		return ProviderNoop
	}
	return a.provider.Name()
}

func (a tensorPlaneProbeProviderAdapter) Backend() string {
	if a.provider == nil {
		return BackendUnknown
	}
	return a.provider.Backend()
}

func (a tensorPlaneProbeProviderAdapter) Capability(ctx context.Context) tensorplane.ProviderBackedProbeProviderCapability {
	if a.provider == nil {
		return tensorplane.ProviderBackedProbeProviderCapability{
			Provider: ProviderNoop,
			Backend:  BackendUnknown,
		}
	}
	capability := a.provider.Capability(ctx)
	return tensorplane.ProviderBackedProbeProviderCapability{
		Provider:                 capability.Provider,
		Backend:                  capability.Backend,
		KVAccessSupported:        capability.KVAccessSupported,
		TensorPlaneDemoSupported: capability.TensorPlaneDemoSupported,
	}
}

func (a tensorPlaneProbeProviderAdapter) GetPage(ctx context.Context, req tensorplane.ProviderBackedProbePageRequest) (tensorplane.TensorPage, error) {
	if a.provider == nil {
		return tensorplane.TensorPage{}, ErrTensorAccessUnsupported
	}
	return a.provider.GetPage(ctx, TensorPageRequest{
		ModelID:    req.ModelID,
		LayerIndex: req.LayerIndex,
		HeadStart:  req.HeadStart,
		HeadCount:  req.HeadCount,
		TokenStart: req.TokenStart,
		TokenCount: req.TokenCount,
		DType:      req.DType,
		PageID:     req.PageID,
		Seed:       req.Seed,
	})
}

func (a tensorPlaneProbeProviderAdapter) GetQuery(ctx context.Context, req tensorplane.ProviderBackedProbeQueryRequest) (tensorplane.AttentionQuery, error) {
	if a.provider == nil {
		return tensorplane.AttentionQuery{}, ErrTensorAccessUnsupported
	}
	return a.provider.GetQuery(ctx, TensorQueryRequest{
		RequestID:  req.RequestID,
		JobID:      req.JobID,
		ModelID:    req.ModelID,
		LayerIndex: req.LayerIndex,
		HeadIndex:  req.HeadIndex,
		DType:      req.DType,
		HeadDim:    req.HeadDim,
		Seed:       req.Seed,
	})
}
