package inference

import "testing"

func TestMergeReceiptMetadataPreservesEnergyReceiptExtras(t *testing.T) {
	meta := mergeReceiptMetadata(
		map[string]any{"energy_receipt": map[string]any{"energy_used_wh": 1.25}},
		map[string]any{"executor": "llama-server"},
	)
	if meta["executor"] != "llama-server" {
		t.Fatalf("executor = %v, want llama-server", meta["executor"])
	}
	energy, ok := meta["energy_receipt"].(map[string]any)
	if !ok {
		t.Fatalf("energy_receipt missing: %#v", meta)
	}
	if energy["energy_used_wh"] != 1.25 {
		t.Fatalf("energy_used_wh = %v, want 1.25", energy["energy_used_wh"])
	}
}
