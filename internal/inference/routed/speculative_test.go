package dashboardinference

import (
	"encoding/json"
	"testing"

	"github.com/Ryvion/ryvion-node/internal/runtimes/llamacpp"
)

func TestSpeculativeMetadataFromStatusEmpty(t *testing.T) {
	t.Parallel()
	got := SpeculativeMetadataFromStatus(llamacpp.LlamaCppSidecarStatus{})
	if !got.IsZero() {
		t.Fatalf("expected zero metadata for empty status, got %+v", got)
	}
	if got.Map() != nil {
		t.Errorf("zero metadata Map() should be nil")
	}
}

func TestSpeculativeMetadataFromStatusActive(t *testing.T) {
	t.Parallel()
	st := llamacpp.LlamaCppSidecarStatus{
		SpeculativeEnabled:   true,
		DraftModelFilename:   "tinyllama-1.1b-Q4_K_M.gguf",
		DraftModelFamilyHint: "llama",
		DraftModelSizeBytes:  650 * 1024 * 1024,
		DraftMaxTokens:       8,
		DraftMinTokens:       2,
	}
	got := SpeculativeMetadataFromStatus(st)
	if !got.Enabled {
		t.Fatalf("Enabled = false, want true")
	}
	if got.Method != speculativeMethodBackendLocalDraft {
		t.Errorf("Method = %q, want %q", got.Method, speculativeMethodBackendLocalDraft)
	}
	if got.DrafterFilename != "tinyllama-1.1b-Q4_K_M.gguf" {
		t.Errorf("DrafterFilename = %q", got.DrafterFilename)
	}
	if got.DraftMaxTokens != 8 || got.DraftMinTokens != 2 {
		t.Errorf("Draft tokens = %d/%d, want 8/2", got.DraftMaxTokens, got.DraftMinTokens)
	}
	m := got.Map()
	if m == nil {
		t.Fatalf("Map should not be nil for active speculation")
	}
	if m["enabled"] != true {
		t.Errorf("Map enabled = %v, want true", m["enabled"])
	}
}

func TestSpeculativeMetadataFromStatusNativeMTP(t *testing.T) {
	t.Parallel()
	st := llamacpp.LlamaCppSidecarStatus{
		SpeculativeEnabled: true,
		SpeculativeMethod:  llamacpp.SpeculativeMethodNativeMTP,
		NativeMTP:          true,
		DraftMaxTokens:     3,
	}
	got := SpeculativeMetadataFromStatus(st)
	if !got.Enabled {
		t.Fatal("Enabled = false, want true")
	}
	if got.Method != speculativeMethodNativeMTP {
		t.Fatalf("Method = %q, want %q", got.Method, speculativeMethodNativeMTP)
	}
	if got.DrafterFilename != "" || got.DrafterSizeBytes != 0 {
		t.Fatalf("native MTP metadata must not invent a drafter: %+v", got)
	}
	m := got.Map()
	if m["method"] != speculativeMethodNativeMTP {
		t.Fatalf("map method = %v, want %q", m["method"], speculativeMethodNativeMTP)
	}
	if _, ok := m["drafter_filename"]; ok {
		t.Fatalf("native MTP map should omit drafter_filename: %v", m)
	}
}

func TestMergeRuntimeCounts(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name        string
		drafted     int64
		accepted    int64
		generated   int64
		wantRate    float64
		wantSpeedup float64
	}{
		{"all-accepted", 12, 12, 30, 1.0, 1.667},
		{"half-accepted", 12, 6, 24, 0.5, 1.333},
		{"none-accepted", 12, 0, 24, 0, 1.0},
		{"clamp-over-drafted", 12, 99, 30, 1.0, 1.667}, // accepted clamped to drafted
		{"zero-generated", 12, 6, 0, 0.5, 0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			base := SpeculativeMetadata{Enabled: true, Method: speculativeMethodBackendLocalDraft}
			got := base.MergeRuntimeCounts(tc.drafted, tc.accepted, tc.generated)
			// AcceptanceRate uses 3-decimal rounding per roundTPS.
			if abs(got.AcceptanceRate-tc.wantRate) > 0.002 {
				t.Errorf("AcceptanceRate = %v, want ~%v", got.AcceptanceRate, tc.wantRate)
			}
			if abs(got.EstimatedSpeedupRatio-tc.wantSpeedup) > 0.005 {
				t.Errorf("EstimatedSpeedupRatio = %v, want ~%v", got.EstimatedSpeedupRatio, tc.wantSpeedup)
			}
		})
	}
}

func TestReceiptMetadataMapEmitsSpeculativeBlock(t *testing.T) {
	t.Parallel()
	meta := ReceiptMetadata{
		RequestID:                "req",
		RunID:                    "run",
		JobID:                    "job",
		Backend:                  llamacpp.BackendName,
		ModelID:                  "ryvion-llama-3.2-3b",
		OutputHash:               "sha256:" + "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
		FinishReason:             llamacpp.FinishReasonStop,
		BackendFinishReason:      llamacpp.FinishReasonStop,
		BackendStopReason:        llamacpp.FinishReasonStop,
		TokensGenerated:          30,
		ProofStatus:              ProofStatusMeasured,
		RuntimeMeasurementStatus: llamacpp.RuntimeMeasurementStatusMeasured,
		MetadataParseStatus:      llamacpp.MetadataParseStatusOK,
		MaxReturnChars:           defaultMaxReturnChars,
		Speculative: SpeculativeMetadata{
			Enabled:               true,
			Method:                speculativeMethodBackendLocalDraft,
			DrafterFilename:       "tinyllama.gguf",
			DraftMaxTokens:        8,
			TokensDrafted:         12,
			TokensAccepted:        9,
			AcceptanceRate:        0.75,
			EstimatedSpeedupRatio: 1.428,
		},
	}
	out := meta.Map()
	specRaw, ok := out["speculative"]
	if !ok {
		t.Fatalf("expected speculative key in map: %v", out)
	}
	spec, ok := specRaw.(map[string]any)
	if !ok {
		t.Fatalf("speculative not a map: %T", specRaw)
	}
	if spec["enabled"] != true {
		t.Errorf("speculative.enabled = %v", spec["enabled"])
	}
	if spec["drafter_filename"] != "tinyllama.gguf" {
		t.Errorf("speculative.drafter_filename = %v", spec["drafter_filename"])
	}
	// Empty speculative block must not emit the key (legacy receipts).
	meta.Speculative = SpeculativeMetadata{}
	out2 := meta.Map()
	if _, has := out2["speculative"]; has {
		t.Errorf("empty speculative should not appear in map")
	}
}

func TestReceiptHashStableWithoutSpeculative(t *testing.T) {
	t.Parallel()
	// Backwards-compat regression guard: a receipt with empty
	// SpeculativeMetadata must produce the same hash as one whose
	// Speculative field has never been touched. Otherwise legacy
	// non-speculative receipts would all change hash on this upgrade.
	base := ReceiptMetadata{
		RequestID:                "req",
		RunID:                    "run",
		JobID:                    "job",
		Backend:                  llamacpp.BackendName,
		ModelID:                  "model",
		OutputHash:               "sha256:" + "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
		FinishReason:             llamacpp.FinishReasonStop,
		BackendFinishReason:      llamacpp.FinishReasonStop,
		BackendStopReason:        llamacpp.FinishReasonStop,
		TokensGenerated:          1,
		ProofStatus:              ProofStatusMeasured,
		RuntimeMeasurementStatus: llamacpp.RuntimeMeasurementStatusMeasured,
		MetadataParseStatus:      llamacpp.MetadataParseStatusOK,
		MaxReturnChars:           defaultMaxReturnChars,
	}
	withZero := base
	withZero.Speculative = SpeculativeMetadata{}

	h1, err := HashReceiptMetadata(base.JobID, base)
	if err != nil {
		t.Fatalf("hash base: %v", err)
	}
	h2, err := HashReceiptMetadata(withZero.JobID, withZero)
	if err != nil {
		t.Fatalf("hash zero: %v", err)
	}
	if h1 != h2 {
		t.Fatalf("hash drift: legacy non-speculative receipts changed hash on upgrade\nbase=%s\nzero=%s", h1, h2)
	}

	// JSON of map must not contain the "speculative" key when zero.
	encoded, err := json.Marshal(base.Map())
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if want := `"speculative"`; containsByteSlice(encoded, want) {
		t.Errorf("legacy non-speculative receipt JSON contains %q: %s", want, string(encoded))
	}
}

func abs(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}

func containsByteSlice(haystack []byte, needle string) bool {
	if len(needle) == 0 {
		return true
	}
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if string(haystack[i:i+len(needle)]) == needle {
			return true
		}
	}
	return false
}
