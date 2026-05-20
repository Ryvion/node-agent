package tensoraccess

import (
	"bytes"
	"context"
	"reflect"
	"testing"

	"github.com/Ryvion/ryvion-node/internal/benchmarks/tensor"
)

func TestTensorPlaneDemoProviderReturnsDeterministicPageAndQuery(t *testing.T) {
	provider := NewTensorPlaneDemoProvider(TensorPlaneDemoProviderConfig{
		ModelID:       "ryvion-llama-3.2-3b",
		RuntimeKind:   RuntimeKindNative,
		Tokens:        8,
		HeadDim:       4,
		ValueDim:      4,
		DType:         tensorplane.TensorDTypeFloat32,
		Seed:          7,
		ContextLength: 8,
		Layers:        2,
		Heads:         1,
	})
	pageReq := TensorPageRequest{
		ModelID:    "ryvion-llama-3.2-3b",
		LayerIndex: 1,
		HeadStart:  0,
		HeadCount:  1,
		TokenStart: 0,
		TokenCount: 8,
		DType:      tensorplane.TensorDTypeFloat32,
		PageID:     "demo-page-1",
		Seed:       99,
	}
	queryReq := TensorQueryRequest{
		RequestID:  "req-demo-1",
		JobID:      "job-demo-1",
		ModelID:    "ryvion-llama-3.2-3b",
		LayerIndex: 1,
		HeadIndex:  0,
		DType:      tensorplane.TensorDTypeFloat32,
		HeadDim:    4,
		Seed:       99,
	}

	firstPage, err := provider.GetPage(context.Background(), pageReq)
	if err != nil {
		t.Fatalf("first GetPage() error = %v", err)
	}
	secondPage, err := provider.GetPage(context.Background(), pageReq)
	if err != nil {
		t.Fatalf("second GetPage() error = %v", err)
	}
	if firstPage.Hash == "" || firstPage.Hash != secondPage.Hash {
		t.Fatalf("page hash not deterministic: first=%q second=%q", firstPage.Hash, secondPage.Hash)
	}
	if !bytes.Equal(firstPage.KeyData, secondPage.KeyData) || !bytes.Equal(firstPage.ValueData, secondPage.ValueData) {
		t.Fatal("page tensor bytes differ for identical demo request")
	}

	firstQuery, err := provider.GetQuery(context.Background(), queryReq)
	if err != nil {
		t.Fatalf("first GetQuery() error = %v", err)
	}
	secondQuery, err := provider.GetQuery(context.Background(), queryReq)
	if err != nil {
		t.Fatalf("second GetQuery() error = %v", err)
	}
	if !reflect.DeepEqual(firstQuery, secondQuery) {
		t.Fatalf("query not deterministic:\nfirst:  %+v\nsecond: %+v", firstQuery, secondQuery)
	}
}

func TestTensorPlaneDemoProviderPageComputesAttentionSummary(t *testing.T) {
	provider := NewTensorPlaneDemoProvider(TensorPlaneDemoProviderConfig{
		ModelID:  "ryvion-llama-3.2-3b",
		Tokens:   8,
		HeadDim:  4,
		ValueDim: 4,
		DType:    tensorplane.TensorDTypeFloat32,
		Seed:     42,
	})

	page, err := provider.GetPage(context.Background(), TensorPageRequest{
		ModelID:    "ryvion-llama-3.2-3b",
		LayerIndex: 0,
		HeadStart:  0,
		HeadCount:  1,
		TokenStart: 0,
		TokenCount: 8,
		DType:      tensorplane.TensorDTypeFloat32,
		PageID:     "demo-page-summary",
		Seed:       42,
	})
	if err != nil {
		t.Fatalf("GetPage() error = %v", err)
	}
	query, err := provider.GetQuery(context.Background(), TensorQueryRequest{
		RequestID:  "req-summary",
		JobID:      "job-summary",
		ModelID:    "ryvion-llama-3.2-3b",
		LayerIndex: 0,
		HeadIndex:  0,
		DType:      tensorplane.TensorDTypeFloat32,
		HeadDim:    4,
		Seed:       42,
	})
	if err != nil {
		t.Fatalf("GetQuery() error = %v", err)
	}

	summary, err := tensorplane.ComputeTensorPartialAttentionSummary(query, page)
	if err != nil {
		t.Fatalf("ComputeTensorPartialAttentionSummary() error = %v", err)
	}
	if summary.SummaryHash == "" || summary.PageHash == "" {
		t.Fatalf("summary hashes missing: %+v", summary)
	}
	if len(summary.WeightedValue) != 4 {
		t.Fatalf("weighted value length = %d, want 4", len(summary.WeightedValue))
	}
}

func TestTensorPlaneDemoProviderCapabilityDoesNotClaimRealKV(t *testing.T) {
	provider := NewTensorPlaneDemoProvider(TensorPlaneDemoProviderConfig{
		ModelID:     "ryvion-llama-3.2-3b",
		RuntimeKind: RuntimeKindNative,
	})
	capability := provider.Capability(context.Background())

	if capability.Provider != ProviderTensorPlaneDemo {
		t.Fatalf("provider = %q", capability.Provider)
	}
	if capability.Backend != BackendDemo {
		t.Fatalf("backend = %q", capability.Backend)
	}
	if capability.KVAccessSupported ||
		capability.KVSnapshotSupported ||
		capability.HiddenStateAccessSupported ||
		capability.LogitsAccessSupported ||
		capability.AttentionHookSupported {
		t.Fatalf("demo capability should not claim real tensor hooks: %+v", capability)
	}
	if !capability.TensorPlaneDemoSupported {
		t.Fatalf("tensorplane_demo_supported = false: %+v", capability)
	}
}
