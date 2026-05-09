package dashboardinference

import (
	"strings"

	"github.com/Ryvion/node-agent/internal/v7/llamacpp"
)

// SpeculativeMetadata captures backend-local speculative-decoding state
// for the receipt and the streamed dashboard payload.
//
// V7.2 Level 0 / V8 Phase 1.2: when the llama.cpp sidecar runs with
// --model-draft, this block records the drafter pairing and any
// runtime metrics the backend exposed so the receipt and dashboard
// can prove acceleration without re-measuring per request.
type SpeculativeMetadata struct {
	Enabled               bool    `json:"enabled"`
	Method                string  `json:"method,omitempty"`
	DrafterFilename       string  `json:"drafter_filename,omitempty"`
	DrafterFamily         string  `json:"drafter_family,omitempty"`
	DrafterSizeBytes      int64   `json:"drafter_size_bytes,omitempty"`
	DraftMaxTokens        int     `json:"draft_max_tokens,omitempty"`
	DraftMinTokens        int     `json:"draft_min_tokens,omitempty"`
	TokensDrafted         int64   `json:"tokens_drafted,omitempty"`
	TokensAccepted        int64   `json:"tokens_accepted,omitempty"`
	AcceptanceRate        float64 `json:"acceptance_rate,omitempty"`
	EstimatedSpeedupRatio float64 `json:"estimated_speedup_ratio,omitempty"`
}

const (
	speculativeMethodBackendLocalDraft = "backend_local_draft_model"
)

// SpeculativeMetadataFromStatus derives the static portion of the
// speculative block from the sidecar status. Runtime metrics
// (tokens_drafted, tokens_accepted) remain zero here; they are set
// later when a future llama-server build surfaces them via the
// completion timings.
func SpeculativeMetadataFromStatus(status llamacpp.LlamaCppSidecarStatus) SpeculativeMetadata {
	if !status.SpeculativeEnabled {
		return SpeculativeMetadata{}
	}
	meta := SpeculativeMetadata{
		Enabled:          true,
		Method:           speculativeMethodBackendLocalDraft,
		DrafterFilename:  cleanText(status.DraftModelFilename, maxModelIDLen),
		DrafterFamily:    strings.ToLower(strings.TrimSpace(status.DraftModelFamilyHint)),
		DrafterSizeBytes: status.DraftModelSizeBytes,
		DraftMaxTokens:   status.DraftMaxTokens,
		DraftMinTokens:   status.DraftMinTokens,
	}
	if meta.DrafterSizeBytes < 0 {
		meta.DrafterSizeBytes = 0
	}
	if meta.DraftMaxTokens < 0 {
		meta.DraftMaxTokens = 0
	}
	if meta.DraftMinTokens < 0 {
		meta.DraftMinTokens = 0
	}
	return meta
}

// MergeRuntimeCounts updates the dynamic acceptance counters when the
// backend has reported them. Acceptance rate and speedup ratio are
// derived deterministically so the receipt remains reproducible.
func (m SpeculativeMetadata) MergeRuntimeCounts(drafted, accepted int64, tokensGenerated int64) SpeculativeMetadata {
	if drafted < 0 {
		drafted = 0
	}
	if accepted < 0 {
		accepted = 0
	}
	if accepted > drafted {
		accepted = drafted
	}
	m.TokensDrafted = drafted
	m.TokensAccepted = accepted
	if drafted > 0 {
		m.AcceptanceRate = roundTPS(float64(accepted) / float64(drafted))
	}
	// Speedup model: every accepted draft token saves one target step.
	// Effective tokens-per-step = 1 + (accepted / target_steps).
	// target_steps = max(1, tokens_generated - accepted).
	if tokensGenerated > 0 {
		target := tokensGenerated - accepted
		if target < 1 {
			target = 1
		}
		m.EstimatedSpeedupRatio = roundTPS(float64(tokensGenerated) / float64(target))
	}
	return m
}

func (m SpeculativeMetadata) IsZero() bool {
	return !m.Enabled &&
		m.DrafterFilename == "" &&
		m.DrafterFamily == "" &&
		m.DrafterSizeBytes == 0 &&
		m.DraftMaxTokens == 0 &&
		m.DraftMinTokens == 0 &&
		m.TokensDrafted == 0 &&
		m.TokensAccepted == 0 &&
		m.AcceptanceRate == 0 &&
		m.EstimatedSpeedupRatio == 0
}

func (m SpeculativeMetadata) Map() map[string]any {
	if m.IsZero() {
		return nil
	}
	out := map[string]any{
		"enabled": m.Enabled,
	}
	if m.Method != "" {
		out["method"] = m.Method
	}
	if m.DrafterFilename != "" {
		out["drafter_filename"] = m.DrafterFilename
	}
	if m.DrafterFamily != "" {
		out["drafter_family"] = m.DrafterFamily
	}
	if m.DrafterSizeBytes > 0 {
		out["drafter_size_bytes"] = m.DrafterSizeBytes
	}
	if m.DraftMaxTokens > 0 {
		out["draft_max_tokens"] = m.DraftMaxTokens
	}
	if m.DraftMinTokens > 0 {
		out["draft_min_tokens"] = m.DraftMinTokens
	}
	if m.TokensDrafted > 0 {
		out["tokens_drafted"] = m.TokensDrafted
	}
	if m.TokensAccepted > 0 {
		out["tokens_accepted"] = m.TokensAccepted
	}
	if m.AcceptanceRate > 0 {
		out["acceptance_rate"] = m.AcceptanceRate
	}
	if m.EstimatedSpeedupRatio > 0 {
		out["estimated_speedup_ratio"] = m.EstimatedSpeedupRatio
	}
	return out
}
