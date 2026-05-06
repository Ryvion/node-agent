package tensoraccess

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func TestProviderImplementationsSatisfyInterface(t *testing.T) {
	var _ TensorAccessProvider = NewNoopProvider(NoopProviderConfig{})
	var _ TensorAccessProvider = NewTensorPlaneDemoProvider(DefaultTensorPlaneDemoProviderConfig())
}

func TestCapabilityJSONHasNoRawTensorPromptOrOutputFields(t *testing.T) {
	capability := NewTensorPlaneDemoProvider(TensorPlaneDemoProviderConfig{
		ModelID:     "ryvion-llama-3.2-3b",
		RuntimeKind: RuntimeKindNative,
	}).Capability(context.Background())

	encoded, err := json.Marshal(capability)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	var keys map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &keys); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	for _, want := range []string{
		"provider",
		"backend",
		"kv_access_supported",
		"kv_snapshot_supported",
		"hidden_state_access_supported",
		"logits_access_supported",
		"attention_hook_supported",
		"tensorplane_demo_supported",
		"model_loaded",
		"runtime_kind",
		"model_id",
		"reason",
	} {
		if _, ok := keys[want]; !ok {
			t.Fatalf("capability JSON missing %q: %s", want, encoded)
		}
	}
	for key := range keys {
		lower := strings.ToLower(key)
		for _, forbidden := range []string{"prompt", "output", "key_data", "value_data", "tensor_bytes", "raw_tensor", "query_vector", "weighted_value"} {
			if strings.Contains(lower, forbidden) {
				t.Fatalf("capability JSON exposes forbidden field %q: %s", key, encoded)
			}
		}
	}
}
