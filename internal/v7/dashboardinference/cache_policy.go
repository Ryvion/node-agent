package dashboardinference

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"strings"

	"github.com/Ryvion/ryvion-node/internal/runtimes/llamacpp"
)

type CacheMetadata struct {
	CachePrompt        bool   `json:"cache_prompt,omitempty"`
	CacheReuseTokens   int    `json:"cache_reuse_tokens,omitempty"`
	SlotID             *int   `json:"slot_id,omitempty"`
	SessionIDHash      string `json:"session_id_hash,omitempty"`
	PrefixHash         string `json:"prefix_hash,omitempty"`
	CacheStateID       string `json:"cache_state_id,omitempty"`
	AffinityTTLSeconds int    `json:"affinity_ttl_seconds,omitempty"`
	RestoreRequested   bool   `json:"restore_requested,omitempty"`
	RestoreStatus      string `json:"restore_status,omitempty"`
	RestoredTokens     int64  `json:"restored_tokens,omitempty"`
	SaveRequested      bool   `json:"save_requested,omitempty"`
	SaveStatus         string `json:"save_status,omitempty"`
	SavedTokens        int64  `json:"saved_tokens,omitempty"`
}

func (m CacheMetadata) IsZero() bool {
	return !m.CachePrompt &&
		m.CacheReuseTokens == 0 &&
		m.SlotID == nil &&
		m.SessionIDHash == "" &&
		m.PrefixHash == "" &&
		m.CacheStateID == "" &&
		m.AffinityTTLSeconds == 0 &&
		!m.RestoreRequested &&
		m.RestoreStatus == "" &&
		m.RestoredTokens == 0 &&
		!m.SaveRequested &&
		m.SaveStatus == "" &&
		m.SavedTokens == 0
}

func (m CacheMetadata) Map() map[string]any {
	if m.IsZero() {
		return nil
	}
	out := map[string]any{}
	if m.CachePrompt {
		out["cache_prompt"] = true
	}
	if m.CacheReuseTokens > 0 {
		out["cache_reuse_tokens"] = m.CacheReuseTokens
	}
	if m.SlotID != nil {
		out["slot_id"] = *m.SlotID
	}
	if m.SessionIDHash != "" {
		out["session_id_hash"] = m.SessionIDHash
	}
	if m.PrefixHash != "" {
		out["prefix_hash"] = m.PrefixHash
	}
	if m.CacheStateID != "" {
		out["cache_state_id"] = m.CacheStateID
	}
	if m.AffinityTTLSeconds > 0 {
		out["affinity_ttl_seconds"] = m.AffinityTTLSeconds
	}
	if m.RestoreRequested {
		out["restore_requested"] = true
	}
	if m.RestoreStatus != "" {
		out["restore_status"] = m.RestoreStatus
	}
	if m.RestoredTokens > 0 {
		out["restored_tokens"] = m.RestoredTokens
	}
	if m.SaveRequested {
		out["save_requested"] = true
	}
	if m.SaveStatus != "" {
		out["save_status"] = m.SaveStatus
	}
	if m.SavedTokens > 0 {
		out["saved_tokens"] = m.SavedTokens
	}
	return out
}

func (m CacheMetadata) clone() CacheMetadata {
	m.SessionIDHash = cleanHash(m.SessionIDHash)
	m.PrefixHash = cleanHash(m.PrefixHash)
	m.CacheStateID = cleanCacheStateID(m.CacheStateID)
	m.RestoreStatus = cleanErrorCode(m.RestoreStatus)
	m.SaveStatus = cleanErrorCode(m.SaveStatus)
	if m.CacheReuseTokens < 0 {
		m.CacheReuseTokens = 0
	}
	if m.AffinityTTLSeconds < 0 {
		m.AffinityTTLSeconds = 0
	}
	if m.RestoredTokens < 0 {
		m.RestoredTokens = 0
	}
	if m.SavedTokens < 0 {
		m.SavedTokens = 0
	}
	if m.SlotID != nil {
		slotID := *m.SlotID
		if slotID < 0 || slotID > maxSlotID {
			m.SlotID = nil
		} else {
			m.SlotID = &slotID
		}
	}
	return m
}

func initialCacheMetadata(spec Spec) CacheMetadata {
	policy := normalizeCachePolicy(spec.CachePolicy)
	meta := CacheMetadata{
		CachePrompt:        policy.CachePrompt,
		CacheReuseTokens:   policy.CacheReuseTokens,
		SlotID:             cloneIntPtr(policy.SlotID),
		SessionIDHash:      hashCacheSessionID(policy.SessionID),
		PrefixHash:         cleanHash(policy.PrefixHash),
		CacheStateID:       policy.CacheStateID,
		AffinityTTLSeconds: policy.AffinityTTLSeconds,
		RestoreRequested:   policy.RestoreSlotBeforeRun,
		SaveRequested:      policy.SaveSlotAfterRun,
	}
	return meta.clone()
}

func applyCompletionCachePolicy(req llamacpp.CompletionRequest, policy CachePolicy) llamacpp.CompletionRequest {
	policy = normalizeCachePolicy(policy)
	req.CachePrompt = policy.CachePrompt
	req.CacheReuseTokens = policy.CacheReuseTokens
	req.SlotID = cloneIntPtr(policy.SlotID)
	return req
}

func restoreSlotCacheIfRequested(ctx context.Context, client llamacpp.SlotCacheClient, baseURL string, spec Spec, meta CacheMetadata) CacheMetadata {
	policy := normalizeCachePolicy(spec.CachePolicy)
	if !policy.RestoreSlotBeforeRun {
		return meta.clone()
	}
	meta.RestoreRequested = true
	if client == nil || policy.SlotID == nil {
		meta.RestoreStatus = "skipped"
		return meta.clone()
	}
	filename := cachePolicySlotFilename(spec)
	if filename == "" {
		meta.RestoreStatus = "skipped"
		return meta.clone()
	}
	result, err := client.RestoreSlot(ctx, llamacpp.SlotCacheRequest{
		BaseURL:  baseURL,
		SlotID:   *policy.SlotID,
		Filename: filename,
	})
	if err != nil {
		meta.RestoreStatus = cleanErrorCode(ErrorCode(err))
		if meta.RestoreStatus == "" {
			meta.RestoreStatus = "failed"
		}
		return meta.clone()
	}
	meta.RestoreStatus = "restored"
	meta.RestoredTokens = result.RestoredTokens
	return meta.clone()
}

func saveSlotCacheIfRequested(ctx context.Context, client llamacpp.SlotCacheClient, baseURL string, spec Spec, meta CacheMetadata) CacheMetadata {
	policy := normalizeCachePolicy(spec.CachePolicy)
	if !policy.SaveSlotAfterRun {
		return meta.clone()
	}
	meta.SaveRequested = true
	if client == nil || policy.SlotID == nil {
		meta.SaveStatus = "skipped"
		return meta.clone()
	}
	filename := cachePolicySlotFilename(spec)
	if filename == "" {
		meta.SaveStatus = "skipped"
		return meta.clone()
	}
	result, err := client.SaveSlot(ctx, llamacpp.SlotCacheRequest{
		BaseURL:  baseURL,
		SlotID:   *policy.SlotID,
		Filename: filename,
	})
	if err != nil {
		meta.SaveStatus = cleanErrorCode(ErrorCode(err))
		if meta.SaveStatus == "" {
			meta.SaveStatus = "failed"
		}
		return meta.clone()
	}
	meta.SaveStatus = "saved"
	meta.SavedTokens = result.SavedTokens
	return meta.clone()
}

func cachePolicySlotFilename(spec Spec) string {
	spec = normalizeSpec(spec)
	policy := spec.CachePolicy
	if policy.SessionID == "" && policy.PrefixHash == "" && policy.CacheStateID == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(strings.Join([]string{
		"ryvion:v7:dashboard_inference_slot_cache:v1",
		spec.ModelID,
		policy.SessionID,
		policy.PrefixHash,
		policy.CacheStateID,
	}, "\n")))
	return "ryvion_slot_" + hex.EncodeToString(sum[:12]) + ".bin"
}

func hashCacheSessionID(sessionID string) string {
	sessionID = cleanText(sessionID, maxSessionIDLen)
	if sessionID == "" {
		return ""
	}
	sum := sha256.Sum256([]byte("ryvion:v7:dashboard_inference_session:v1\n" + sessionID))
	return "sha256:" + hex.EncodeToString(sum[:])
}

func cloneIntPtr(value *int) *int {
	if value == nil {
		return nil
	}
	out := *value
	return &out
}
