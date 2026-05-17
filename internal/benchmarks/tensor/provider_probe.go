package tensorplane

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
)

const (
	defaultProviderBackedProbeModelID = "tensorplane/demo-model"
)

var (
	ErrInvalidProviderBackedProbe     = errors.New("tensorplane: invalid provider-backed probe")
	ErrProviderBackedProbeUnsupported = errors.New("tensorplane: provider-backed probe unsupported")
)

type ProviderBackedProbeRequest struct {
	Provider   string              `json:"provider"`
	ModelID    string              `json:"model_id"`
	LayerIndex int                 `json:"layer_index"`
	DType      TensorDType         `json:"dtype"`
	Tokens     int                 `json:"tokens"`
	HeadDim    int                 `json:"head_dim"`
	ValueDim   int                 `json:"value_dim"`
	Seed       int64               `json:"seed"`
	Getenv     func(string) string `json:"-"`
}

type ProviderBackedProbeResult struct {
	Provider              string      `json:"provider"`
	Backend               string      `json:"backend"`
	ModelID               string      `json:"model_id"`
	DType                 TensorDType `json:"dtype"`
	Tokens                int         `json:"tokens"`
	HeadDim               int         `json:"head_dim"`
	ValueDim              int         `json:"value_dim"`
	PageHash              string      `json:"page_hash"`
	QueryHash             string      `json:"query_hash"`
	SummaryHash           string      `json:"summary_hash"`
	WeightedValueLength   int         `json:"weighted_value_length"`
	ComputeTimeUs         int64       `json:"compute_time_us"`
	PayloadBytesEstimate  int64       `json:"payload_bytes_estimate"`
	CorrectnessStatus     string      `json:"correctness_status,omitempty"`
	RealKVCache           bool        `json:"real_kv_cache"`
	MaxAbsDiffVsReference float64     `json:"max_abs_diff_vs_reference,omitempty"`
}

type ProviderBackedProbeProviderConfig struct {
	Provider        string
	DefaultProvider string
	ModelID         string
	DType           TensorDType
	Tokens          int
	HeadDim         int
	ValueDim        int
	Seed            int64
	Getenv          func(string) string
}

type ProviderBackedProbeProviderCapability struct {
	Provider                 string
	Backend                  string
	KVAccessSupported        bool
	TensorPlaneDemoSupported bool
}

type ProviderBackedProbePageRequest struct {
	ModelID    string
	LayerIndex int
	HeadStart  int
	HeadCount  int
	TokenStart int
	TokenCount int
	DType      TensorDType
	PageID     string
	Seed       int64
}

type ProviderBackedProbeQueryRequest struct {
	RequestID  string
	JobID      string
	ModelID    string
	LayerIndex int
	HeadIndex  int
	DType      TensorDType
	HeadDim    int
	Seed       int64
}

type ProviderBackedProbeProvider interface {
	Name() string
	Backend() string
	Capability(ctx context.Context) ProviderBackedProbeProviderCapability
	GetPage(ctx context.Context, req ProviderBackedProbePageRequest) (TensorPage, error)
	GetQuery(ctx context.Context, req ProviderBackedProbeQueryRequest) (AttentionQuery, error)
}

type ProviderBackedProbeResolver func(config ProviderBackedProbeProviderConfig) ProviderBackedProbeProvider

var (
	providerBackedProbeResolverMu sync.RWMutex
	providerBackedProbeResolver   ProviderBackedProbeResolver
)

func RegisterProviderBackedProbeResolver(resolver ProviderBackedProbeResolver) {
	if resolver == nil {
		return
	}
	providerBackedProbeResolverMu.Lock()
	defer providerBackedProbeResolverMu.Unlock()
	providerBackedProbeResolver = resolver
}

func DefaultProviderBackedProbeRequest() ProviderBackedProbeRequest {
	defaults := DefaultTensorPlaneFixtureConfig()
	return ProviderBackedProbeRequest{
		ModelID:    defaultProviderBackedProbeModelID,
		LayerIndex: 0,
		DType:      defaults.DType,
		Tokens:     defaults.Tokens,
		HeadDim:    defaults.HeadDim,
		ValueDim:   defaults.ValueDim,
		Seed:       defaults.Seed,
	}
}

func RunProviderBackedTensorPlaneProbe(ctx context.Context, req ProviderBackedProbeRequest) (ProviderBackedProbeResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	req = normalizeProviderBackedProbeRequest(req)
	if err := ValidateProviderBackedProbeRequest(req); err != nil {
		return ProviderBackedProbeResult{}, err
	}
	provider := resolveProviderBackedProbeProvider(ProviderBackedProbeProviderConfig{
		Provider:        req.Provider,
		DefaultProvider: "tensorplane_demo",
		ModelID:         req.ModelID,
		DType:           req.DType,
		Tokens:          req.Tokens,
		HeadDim:         req.HeadDim,
		ValueDim:        req.ValueDim,
		Seed:            req.Seed,
		Getenv:          req.Getenv,
	})
	if provider == nil {
		return ProviderBackedProbeResult{}, fmt.Errorf("%w: tensor access provider resolver not registered", ErrProviderBackedProbeUnsupported)
	}
	capability := provider.Capability(ctx)
	page, err := provider.GetPage(ctx, ProviderBackedProbePageRequest{
		ModelID:    req.ModelID,
		LayerIndex: req.LayerIndex,
		HeadStart:  0,
		HeadCount:  1,
		TokenStart: 0,
		TokenCount: req.Tokens,
		DType:      req.DType,
		PageID:     providerBackedProbePageID(req),
		Seed:       req.Seed,
	})
	if err != nil {
		return ProviderBackedProbeResult{}, err
	}
	query, err := provider.GetQuery(ctx, ProviderBackedProbeQueryRequest{
		RequestID:  providerBackedProbeRequestID(req),
		JobID:      providerBackedProbeJobID(req),
		ModelID:    req.ModelID,
		LayerIndex: req.LayerIndex,
		HeadIndex:  0,
		DType:      req.DType,
		HeadDim:    req.HeadDim,
		Seed:       req.Seed,
	})
	if err != nil {
		return ProviderBackedProbeResult{}, err
	}
	summary, err := ComputeTensorPartialAttentionSummary(query, page)
	if err != nil {
		return ProviderBackedProbeResult{}, err
	}
	queryHash, err := HashBenchmarkQuery(query)
	if err != nil {
		return ProviderBackedProbeResult{}, err
	}

	correctnessStatus := CorrectnessStatusNotChecked
	maxDiff := 0.0
	if providerBackedProbeReferenceAvailable(capability) {
		reference, err := computeNaiveTensorPartialAttentionReference(query, page)
		if err != nil {
			return ProviderBackedProbeResult{}, err
		}
		maxDiff = maxAbsDiffVsReference(summary, reference)
		correctnessStatus = CorrectnessStatusMatched
		if maxDiff > TensorPlaneProbeTolerance(summary.DType) {
			correctnessStatus = CorrectnessStatusMismatch
		}
	}

	providerName := strings.TrimSpace(capability.Provider)
	if providerName == "" {
		providerName = strings.TrimSpace(provider.Name())
	}
	backend := strings.TrimSpace(capability.Backend)
	if backend == "" {
		backend = strings.TrimSpace(provider.Backend())
	}

	return ProviderBackedProbeResult{
		Provider:              providerName,
		Backend:               backend,
		ModelID:               page.PageID.ModelID,
		DType:                 summary.DType,
		Tokens:                page.Shape.Tokens,
		HeadDim:               page.Shape.HeadDim,
		ValueDim:              page.Shape.ValueDim,
		PageHash:              sha256ID(summary.PageHash),
		QueryHash:             queryHash,
		SummaryHash:           sha256ID(summary.SummaryHash),
		WeightedValueLength:   len(summary.WeightedValue),
		ComputeTimeUs:         summary.ComputeTimeUs,
		PayloadBytesEstimate:  summary.PayloadBytesEstimate,
		CorrectnessStatus:     correctnessStatus,
		RealKVCache:           capability.KVAccessSupported && !capability.TensorPlaneDemoSupported,
		MaxAbsDiffVsReference: maxDiff,
	}, nil
}

func ValidateProviderBackedProbeRequest(req ProviderBackedProbeRequest) error {
	req = normalizeProviderBackedProbeRequest(req)
	var errs []error
	if req.ModelID == "" {
		errs = append(errs, fmt.Errorf("%w: model_id required", ErrInvalidProviderBackedProbe))
	}
	if req.LayerIndex < 0 {
		errs = append(errs, fmt.Errorf("%w: layer_index must be non-negative", ErrInvalidProviderBackedProbe))
	}
	if err := ValidateTensorDType(req.DType); err != nil {
		errs = append(errs, err)
	}
	if req.Tokens <= 0 {
		errs = append(errs, fmt.Errorf("%w: tokens must be greater than zero", ErrInvalidProviderBackedProbe))
	} else if req.Tokens > maxBenchmarkTokens {
		errs = append(errs, fmt.Errorf("%w: tokens exceeds maximum %d", ErrInvalidProviderBackedProbe, maxBenchmarkTokens))
	}
	if req.HeadDim <= 0 {
		errs = append(errs, fmt.Errorf("%w: head_dim must be greater than zero", ErrInvalidProviderBackedProbe))
	} else if req.HeadDim > maxBenchmarkDim {
		errs = append(errs, fmt.Errorf("%w: head_dim exceeds maximum %d", ErrInvalidProviderBackedProbe, maxBenchmarkDim))
	}
	if req.ValueDim <= 0 {
		errs = append(errs, fmt.Errorf("%w: value_dim must be greater than zero", ErrInvalidProviderBackedProbe))
	} else if req.ValueDim > maxBenchmarkDim {
		errs = append(errs, fmt.Errorf("%w: value_dim exceeds maximum %d", ErrInvalidProviderBackedProbe, maxBenchmarkDim))
	}
	if req.Tokens > 0 && req.HeadDim > 0 && req.Tokens > maxBenchmarkTensorElements/req.HeadDim {
		errs = append(errs, fmt.Errorf("%w: tokens * head_dim exceeds maximum %d", ErrInvalidProviderBackedProbe, maxBenchmarkTensorElements))
	}
	if req.Tokens > 0 && req.ValueDim > 0 && req.Tokens > maxBenchmarkTensorElements/req.ValueDim {
		errs = append(errs, fmt.Errorf("%w: tokens * value_dim exceeds maximum %d", ErrInvalidProviderBackedProbe, maxBenchmarkTensorElements))
	}
	if req.Tokens > 0 && req.HeadDim > 0 {
		if elements, ok := checkedMultiply(req.Tokens, req.HeadDim); !ok {
			errs = append(errs, fmt.Errorf("%w: key tensor element count overflow", ErrInvalidProviderBackedProbe))
		} else if elementBytes, err := tensorDTypeElementBytes(req.DType); err == nil {
			if _, ok := checkedMultiply(elements, elementBytes); !ok {
				errs = append(errs, fmt.Errorf("%w: key tensor byte count overflow", ErrInvalidProviderBackedProbe))
			}
		}
	}
	if req.Tokens > 0 && req.ValueDim > 0 {
		if elements, ok := checkedMultiply(req.Tokens, req.ValueDim); !ok {
			errs = append(errs, fmt.Errorf("%w: value tensor element count overflow", ErrInvalidProviderBackedProbe))
		} else if elementBytes, err := tensorDTypeElementBytes(req.DType); err == nil {
			if _, ok := checkedMultiply(elements, elementBytes); !ok {
				errs = append(errs, fmt.Errorf("%w: value tensor byte count overflow", ErrInvalidProviderBackedProbe))
			}
		}
	}
	return errors.Join(errs...)
}

func FormatProviderBackedTensorPlaneProbeResult(result ProviderBackedProbeResult, jsonOutput bool) string {
	if jsonOutput {
		encoded, err := json.MarshalIndent(result, "", "  ")
		if err != nil {
			return `{"error":"format provider-backed tensorplane probe result"}`
		}
		return string(encoded)
	}
	return fmt.Sprintf(
		"provider: %s\nbackend: %s\nmodel_id: %s\ndtype: %s\ntokens: %d\nhead_dim: %d\nvalue_dim: %d\npage_hash: %s\nquery_hash: %s\nsummary_hash: %s\nweighted_value_length: %d\ncompute_time_us: %d\npayload_bytes_estimate: %d\ncorrectness_status: %s\nreal_kv_cache: %t",
		result.Provider,
		result.Backend,
		result.ModelID,
		result.DType,
		result.Tokens,
		result.HeadDim,
		result.ValueDim,
		result.PageHash,
		result.QueryHash,
		result.SummaryHash,
		result.WeightedValueLength,
		result.ComputeTimeUs,
		result.PayloadBytesEstimate,
		result.CorrectnessStatus,
		result.RealKVCache,
	)
}

func normalizeProviderBackedProbeRequest(req ProviderBackedProbeRequest) ProviderBackedProbeRequest {
	defaults := DefaultProviderBackedProbeRequest()
	req.Provider = strings.TrimSpace(req.Provider)
	req.ModelID = strings.TrimSpace(req.ModelID)
	if req.ModelID == "" {
		req.ModelID = defaults.ModelID
	}
	req.DType = NormalizeTensorDType(req.DType)
	if req.DType == "" {
		req.DType = defaults.DType
	}
	if req.Tokens <= 0 {
		req.Tokens = defaults.Tokens
	}
	if req.HeadDim <= 0 {
		req.HeadDim = defaults.HeadDim
	}
	if req.ValueDim <= 0 {
		req.ValueDim = defaults.ValueDim
	}
	if req.Seed == 0 {
		req.Seed = defaults.Seed
	}
	return req
}

func resolveProviderBackedProbeProvider(config ProviderBackedProbeProviderConfig) ProviderBackedProbeProvider {
	providerBackedProbeResolverMu.RLock()
	resolver := providerBackedProbeResolver
	providerBackedProbeResolverMu.RUnlock()
	if resolver == nil {
		return nil
	}
	return resolver(config)
}

func providerBackedProbeReferenceAvailable(capability ProviderBackedProbeProviderCapability) bool {
	return capability.TensorPlaneDemoSupported
}

func providerBackedProbePageID(req ProviderBackedProbeRequest) string {
	return "tensorplane-provider-page-" + providerBackedProbeHashPrefix(req)
}

func providerBackedProbeRequestID(req ProviderBackedProbeRequest) string {
	return "tensorplane-provider-probe-" + providerBackedProbeHashPrefix(req)
}

func providerBackedProbeJobID(req ProviderBackedProbeRequest) string {
	return "tensorplane-provider-probe-job-" + providerBackedProbeHashPrefix(req)
}

func providerBackedProbeHashPrefix(req ProviderBackedProbeRequest) string {
	hash, err := sha256HexJSON(struct {
		Provider   string      `json:"provider"`
		ModelID    string      `json:"model_id"`
		LayerIndex int         `json:"layer_index"`
		DType      TensorDType `json:"dtype"`
		Tokens     int         `json:"tokens"`
		HeadDim    int         `json:"head_dim"`
		ValueDim   int         `json:"value_dim"`
		Seed       int64       `json:"seed"`
	}{
		Provider:   req.Provider,
		ModelID:    req.ModelID,
		LayerIndex: req.LayerIndex,
		DType:      req.DType,
		Tokens:     req.Tokens,
		HeadDim:    req.HeadDim,
		ValueDim:   req.ValueDim,
		Seed:       req.Seed,
	})
	if err != nil || len(hash) < 24 {
		return "unavailable"
	}
	return hash[:24]
}
