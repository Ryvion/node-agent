package tensorplane_test

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/Ryvion/ryvion-node/internal/benchmarks/tensor"
	"github.com/Ryvion/ryvion-node/internal/runtimes/tensoraccess"
)

func TestProviderBackedTensorPlaneProbeDemoComputesMatchedResult(t *testing.T) {
	result, err := tensorplane.RunProviderBackedTensorPlaneProbe(context.Background(), providerBackedProbeTestRequest(tensoraccess.ProviderTensorPlaneDemo))
	if err != nil {
		t.Fatalf("RunProviderBackedTensorPlaneProbe() error = %v", err)
	}

	if result.Provider != tensoraccess.ProviderTensorPlaneDemo {
		t.Fatalf("provider = %q, want %q", result.Provider, tensoraccess.ProviderTensorPlaneDemo)
	}
	if result.Backend != tensoraccess.BackendDemo {
		t.Fatalf("backend = %q, want %q", result.Backend, tensoraccess.BackendDemo)
	}
	if result.ModelID != "tensorplane/demo-model" {
		t.Fatalf("model_id = %q, want tensorplane/demo-model", result.ModelID)
	}
	if result.DType != tensorplane.TensorDTypeFloat32 || result.Tokens != 8 || result.HeadDim != 4 || result.ValueDim != 3 {
		t.Fatalf("result dimensions = %+v", result)
	}
	if !looksLikeSHA256ID(result.PageHash) || !looksLikeSHA256ID(result.QueryHash) || !looksLikeSHA256ID(result.SummaryHash) {
		t.Fatalf("hashes missing or malformed: page=%q query=%q summary=%q", result.PageHash, result.QueryHash, result.SummaryHash)
	}
	if result.WeightedValueLength != result.ValueDim {
		t.Fatalf("weighted_value_length = %d, want %d", result.WeightedValueLength, result.ValueDim)
	}
	if result.PayloadBytesEstimate <= 0 {
		t.Fatalf("payload_bytes_estimate = %d, want > 0", result.PayloadBytesEstimate)
	}
	if result.CorrectnessStatus != tensorplane.CorrectnessStatusMatched {
		t.Fatalf("correctness_status = %q, want matched", result.CorrectnessStatus)
	}
	if result.RealKVCache {
		t.Fatalf("real_kv_cache = true for tensorplane_demo result: %+v", result)
	}
}

func TestProviderBackedTensorPlaneProbeIsDeterministicForDemoProvider(t *testing.T) {
	req := providerBackedProbeTestRequest(tensoraccess.ProviderTensorPlaneDemo)
	first, err := tensorplane.RunProviderBackedTensorPlaneProbe(context.Background(), req)
	if err != nil {
		t.Fatalf("first RunProviderBackedTensorPlaneProbe() error = %v", err)
	}
	second, err := tensorplane.RunProviderBackedTensorPlaneProbe(context.Background(), req)
	if err != nil {
		t.Fatalf("second RunProviderBackedTensorPlaneProbe() error = %v", err)
	}

	if first.PageHash != second.PageHash || first.QueryHash != second.QueryHash || first.SummaryHash != second.SummaryHash {
		t.Fatalf("demo probe hashes differ:\nfirst:  %+v\nsecond: %+v", first, second)
	}
}

func TestProviderBackedTensorPlaneProbeNoopProviderReturnsUnsupported(t *testing.T) {
	_, err := tensorplane.RunProviderBackedTensorPlaneProbe(context.Background(), providerBackedProbeTestRequest(tensoraccess.ProviderNoop))
	if !errors.Is(err, tensoraccess.ErrTensorAccessUnsupported) {
		t.Fatalf("RunProviderBackedTensorPlaneProbe() error = %v, want unsupported tensor access", err)
	}
}

func TestProviderBackedTensorPlaneProbeResultHasNoRawTensorPromptOrOutputJSON(t *testing.T) {
	result, err := tensorplane.RunProviderBackedTensorPlaneProbe(context.Background(), providerBackedProbeTestRequest(tensoraccess.ProviderTensorPlaneDemo))
	if err != nil {
		t.Fatalf("RunProviderBackedTensorPlaneProbe() error = %v", err)
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &fields); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	for _, key := range []string{"key_data", "value_data", "query_vector", "weighted_value", "prompt", "prompt_text", "generated_text", "output", "output_text", "tensor_bytes", "raw_tensor"} {
		if _, ok := fields[key]; ok {
			t.Fatalf("provider-backed probe JSON exposes forbidden field %q: %s", key, encoded)
		}
	}
	lower := strings.ToLower(string(encoded))
	for _, marker := range []string{"key_data", "value_data", "query_vector", "prompt_text", "generated_text", "output_text", "tensor_bytes", "raw_tensor"} {
		if strings.Contains(lower, marker) {
			t.Fatalf("provider-backed probe JSON contains forbidden marker %q: %s", marker, encoded)
		}
	}
}

func providerBackedProbeTestRequest(provider string) tensorplane.ProviderBackedProbeRequest {
	return tensorplane.ProviderBackedProbeRequest{
		Provider:   provider,
		ModelID:    "tensorplane/demo-model",
		LayerIndex: 0,
		DType:      tensorplane.TensorDTypeFloat32,
		Tokens:     8,
		HeadDim:    4,
		ValueDim:   3,
		Seed:       42,
		Getenv:     func(string) string { return "" },
	}
}

func looksLikeSHA256ID(value string) bool {
	if !strings.HasPrefix(value, "sha256:") {
		return false
	}
	return len(strings.TrimPrefix(value, "sha256:")) == 64
}
