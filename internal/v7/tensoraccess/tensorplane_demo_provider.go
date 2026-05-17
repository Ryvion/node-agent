package tensoraccess

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/Ryvion/ryvion-node/internal/v7/tensorplane"
)

const defaultTensorPlaneDemoModelID = "tensorplane-demo-fixture"

type TensorPlaneDemoProviderConfig struct {
	ModelID       string
	RuntimeKind   string
	Tokens        int
	HeadDim       int
	ValueDim      int
	DType         tensorplane.TensorDType
	Seed          int64
	ContextLength int
	Layers        int
	Heads         int
}

type TensorPlaneDemoProvider struct {
	config TensorPlaneDemoProviderConfig
}

func DefaultTensorPlaneDemoProviderConfig() TensorPlaneDemoProviderConfig {
	defaults := tensorplane.DefaultTensorPlaneFixtureConfig()
	return TensorPlaneDemoProviderConfig{
		ModelID:       defaultTensorPlaneDemoModelID,
		RuntimeKind:   RuntimeKindDemo,
		Tokens:        defaults.Tokens,
		HeadDim:       defaults.HeadDim,
		ValueDim:      defaults.ValueDim,
		DType:         defaults.DType,
		Seed:          defaults.Seed,
		ContextLength: defaults.Tokens,
		Layers:        1,
		Heads:         1,
	}
}

func NewTensorPlaneDemoProvider(config TensorPlaneDemoProviderConfig) *TensorPlaneDemoProvider {
	return &TensorPlaneDemoProvider{config: normalizeTensorPlaneDemoProviderConfig(config)}
}

func (p *TensorPlaneDemoProvider) Name() string {
	return ProviderTensorPlaneDemo
}

func (p *TensorPlaneDemoProvider) Backend() string {
	return BackendDemo
}

func (p *TensorPlaneDemoProvider) Capability(ctx context.Context) TensorAccessCapability {
	_ = ctx
	return NormalizeCapability(TensorAccessCapability{
		Provider:                   ProviderTensorPlaneDemo,
		Backend:                    BackendDemo,
		KVAccessSupported:          false,
		KVSnapshotSupported:        false,
		HiddenStateAccessSupported: false,
		LogitsAccessSupported:      false,
		AttentionHookSupported:     false,
		TensorPlaneDemoSupported:   true,
		ModelLoaded:                true,
		RuntimeKind:                p.config.RuntimeKind,
		ModelID:                    p.config.ModelID,
		Reason:                     ReasonTensorPlaneDemoOnly,
	})
}

func (p *TensorPlaneDemoProvider) ListLoadedModels(ctx context.Context) ([]LoadedTensorModel, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return []LoadedTensorModel{{
		ModelID:             p.config.ModelID,
		RuntimeKind:         p.config.RuntimeKind,
		Backend:             BackendDemo,
		Loaded:              true,
		SupportsKV:          false,
		SupportsTensorPlane: true,
		ContextLength:       p.config.ContextLength,
		Layers:              p.config.Layers,
		Heads:               p.config.Heads,
		HeadDim:             p.config.HeadDim,
	}}, nil
}

func (p *TensorPlaneDemoProvider) GetPage(ctx context.Context, req TensorPageRequest) (tensorplane.TensorPage, error) {
	if err := ctx.Err(); err != nil {
		return tensorplane.TensorPage{}, err
	}
	req = normalizeTensorPageRequest(req)
	if err := validateTensorPlaneDemoPageRequest(req); err != nil {
		return tensorplane.TensorPage{}, err
	}
	config := p.fixtureConfig(req.DType, req.TokenCount, p.config.HeadDim, req.Seed)
	fixture, err := tensorplane.BuildTensorPlaneFixture(config)
	if err != nil {
		return tensorplane.TensorPage{}, err
	}
	page, _, err := tensorplane.TensorPlaneFixturePageAndQuery(fixture)
	if err != nil {
		return tensorplane.TensorPage{}, err
	}
	page.PageID = tensorplane.TensorPageID{
		ModelID:       firstNonEmpty(req.ModelID, p.config.ModelID),
		LayerIndex:    req.LayerIndex,
		HeadStart:     req.HeadStart,
		HeadCount:     req.HeadCount,
		TokenStart:    req.TokenStart,
		TokenCount:    req.TokenCount,
		PageID:        firstNonEmpty(req.PageID, demoPageID(req, config.Seed)),
		DType:         config.DType,
		LayoutVersion: tensorplane.TensorLayoutSimpleContiguousV1,
	}
	page.DType = config.DType
	page.Shape = tensorplane.TensorShape{
		Heads:    req.HeadCount,
		Tokens:   req.TokenCount,
		HeadDim:  p.config.HeadDim,
		ValueDim: p.config.ValueDim,
		PageSize: req.TokenCount,
	}
	pageHash, err := tensorplane.HashTensorPage(page)
	if err != nil {
		return tensorplane.TensorPage{}, err
	}
	page.Hash = pageHash
	return page, nil
}

func (p *TensorPlaneDemoProvider) GetQuery(ctx context.Context, req TensorQueryRequest) (tensorplane.AttentionQuery, error) {
	if err := ctx.Err(); err != nil {
		return tensorplane.AttentionQuery{}, err
	}
	req = normalizeTensorQueryRequest(req)
	if err := validateTensorPlaneDemoQueryRequest(req); err != nil {
		return tensorplane.AttentionQuery{}, err
	}
	headDim := req.HeadDim
	if headDim == 0 {
		headDim = p.config.HeadDim
	}
	config := p.fixtureConfig(req.DType, p.config.Tokens, headDim, req.Seed)
	fixture, err := tensorplane.BuildTensorPlaneFixture(config)
	if err != nil {
		return tensorplane.AttentionQuery{}, err
	}
	_, query, err := tensorplane.TensorPlaneFixturePageAndQuery(fixture)
	if err != nil {
		return tensorplane.AttentionQuery{}, err
	}
	query.RequestID = req.RequestID
	query.JobID = req.JobID
	query.ModelID = firstNonEmpty(req.ModelID, p.config.ModelID)
	query.LayerIndex = req.LayerIndex
	query.HeadIndex = req.HeadIndex
	query.HeadStart = 0
	query.HeadCount = 0
	query.DType = config.DType
	return query, nil
}

func (p *TensorPlaneDemoProvider) fixtureConfig(dtype tensorplane.TensorDType, tokens int, headDim int, seed int64) tensorplane.TensorPlaneFixtureConfig {
	if seed == 0 {
		seed = p.config.Seed
	}
	if dtype == "" {
		dtype = p.config.DType
	}
	return tensorplane.TensorPlaneFixtureConfig{
		Tokens:   tokens,
		HeadDim:  headDim,
		ValueDim: p.config.ValueDim,
		DType:    tensorplane.NormalizeTensorDType(dtype),
		Seed:     seed,
	}
}

func normalizeTensorPlaneDemoProviderConfig(config TensorPlaneDemoProviderConfig) TensorPlaneDemoProviderConfig {
	defaults := DefaultTensorPlaneDemoProviderConfig()
	config.ModelID = cleanCapabilityText(firstNonEmpty(config.ModelID, defaults.ModelID))
	config.RuntimeKind = normalizeRuntimeKind(firstNonEmpty(config.RuntimeKind, defaults.RuntimeKind))
	if config.Tokens <= 0 {
		config.Tokens = defaults.Tokens
	}
	if config.HeadDim <= 0 {
		config.HeadDim = defaults.HeadDim
	}
	if config.ValueDim <= 0 {
		config.ValueDim = defaults.ValueDim
	}
	config.DType = tensorplane.NormalizeTensorDType(config.DType)
	if config.DType == "" {
		config.DType = defaults.DType
	}
	if config.Seed == 0 {
		config.Seed = defaults.Seed
	}
	if config.ContextLength <= 0 {
		config.ContextLength = config.Tokens
	}
	if config.Layers <= 0 {
		config.Layers = defaults.Layers
	}
	if config.Heads <= 0 {
		config.Heads = defaults.Heads
	}
	return config
}

func normalizeTensorPageRequest(req TensorPageRequest) TensorPageRequest {
	req.ModelID = cleanCapabilityText(req.ModelID)
	req.PageID = cleanCapabilityText(req.PageID)
	req.DType = tensorplane.NormalizeTensorDType(req.DType)
	return req
}

func validateTensorPlaneDemoPageRequest(req TensorPageRequest) error {
	var errs []error
	if strings.TrimSpace(req.ModelID) == "" {
		errs = append(errs, fmt.Errorf("%w: model_id required", ErrInvalidTensorAccessRequest))
	}
	if req.LayerIndex < 0 {
		errs = append(errs, fmt.Errorf("%w: layer_index must be non-negative", ErrInvalidTensorAccessRequest))
	}
	if req.HeadStart < 0 {
		errs = append(errs, fmt.Errorf("%w: head_start must be non-negative", ErrInvalidTensorAccessRequest))
	}
	if req.HeadCount != 1 {
		errs = append(errs, fmt.Errorf("%w: tensorplane demo provider supports exactly one head per page", ErrInvalidTensorAccessRequest))
	}
	if req.TokenStart < 0 {
		errs = append(errs, fmt.Errorf("%w: token_start must be non-negative", ErrInvalidTensorAccessRequest))
	}
	if req.TokenCount <= 0 {
		errs = append(errs, fmt.Errorf("%w: token_count must be greater than zero", ErrInvalidTensorAccessRequest))
	}
	if err := tensorplane.ValidateTensorDType(req.DType); err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

func normalizeTensorQueryRequest(req TensorQueryRequest) TensorQueryRequest {
	req.RequestID = cleanCapabilityText(req.RequestID)
	req.JobID = cleanCapabilityText(req.JobID)
	req.ModelID = cleanCapabilityText(req.ModelID)
	req.DType = tensorplane.NormalizeTensorDType(req.DType)
	return req
}

func validateTensorPlaneDemoQueryRequest(req TensorQueryRequest) error {
	var errs []error
	if strings.TrimSpace(req.RequestID) == "" {
		errs = append(errs, fmt.Errorf("%w: request_id required", ErrInvalidTensorAccessRequest))
	}
	if strings.TrimSpace(req.JobID) == "" {
		errs = append(errs, fmt.Errorf("%w: job_id required", ErrInvalidTensorAccessRequest))
	}
	if strings.TrimSpace(req.ModelID) == "" {
		errs = append(errs, fmt.Errorf("%w: model_id required", ErrInvalidTensorAccessRequest))
	}
	if req.LayerIndex < 0 {
		errs = append(errs, fmt.Errorf("%w: layer_index must be non-negative", ErrInvalidTensorAccessRequest))
	}
	if req.HeadIndex < 0 {
		errs = append(errs, fmt.Errorf("%w: head_index must be non-negative", ErrInvalidTensorAccessRequest))
	}
	if req.HeadDim < 0 {
		errs = append(errs, fmt.Errorf("%w: head_dim must be non-negative", ErrInvalidTensorAccessRequest))
	}
	if err := tensorplane.ValidateTensorDType(req.DType); err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

func demoPageID(req TensorPageRequest, seed int64) string {
	return fmt.Sprintf("tensorplane-demo-page-%s-%d-%d-%d-%d-%d-%s", req.ModelID, seed, req.LayerIndex, req.HeadStart, req.TokenStart, req.TokenCount, req.DType)
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			return trimmed
		}
	}
	return ""
}
