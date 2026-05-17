package llamacpp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
)

const maxSlotFilenameLen = 255

type SlotCacheClient interface {
	ListSlots(context.Context, string) ([]SlotState, error)
	SaveSlot(context.Context, SlotCacheRequest) (SlotCacheResult, error)
	RestoreSlot(context.Context, SlotCacheRequest) (SlotCacheResult, error)
	EraseSlot(context.Context, SlotCacheRequest) (SlotCacheResult, error)
}

type SlotCacheRequest struct {
	BaseURL  string
	SlotID   int
	Filename string
}

type SlotCacheResult struct {
	Action         string `json:"action"`
	SlotID         int    `json:"slot_id"`
	Filename       string `json:"filename,omitempty"`
	SavedTokens    int64  `json:"saved_tokens,omitempty"`
	WrittenBytes   int64  `json:"written_bytes,omitempty"`
	RestoredTokens int64  `json:"restored_tokens,omitempty"`
	ReadBytes      int64  `json:"read_bytes,omitempty"`
	ErasedTokens   int64  `json:"erased_tokens,omitempty"`
}

type SlotState struct {
	SlotID          int    `json:"slot_id"`
	State           int    `json:"state,omitempty"`
	TaskID          int    `json:"task_id,omitempty"`
	PromptTokens    int64  `json:"prompt_tokens,omitempty"`
	TokensCached    int64  `json:"tokens_cached,omitempty"`
	TokensEvaluated int64  `json:"tokens_evaluated,omitempty"`
	RawStatus       string `json:"raw_status,omitempty"`
}

func (c OpenAIClient) ListSlots(ctx context.Context, baseURL string) ([]SlotState, error) {
	baseURL = strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if baseURL == "" || !isLocalBaseURL(baseURL) {
		return nil, ClientError{Code: "llamacpp_invalid_base_url"}
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/slots", nil)
	if err != nil {
		return nil, ClientError{Code: "llamacpp_slot_request_build_failed"}
	}
	resp, err := c.httpClient().Do(httpReq)
	if err != nil {
		return nil, ClientError{Code: slotRequestErrorCode(ctx, "llamacpp_slot_list_failed")}
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
		return nil, ClientError{Code: "llamacpp_slot_list_failed", StatusCode: resp.StatusCode}
	}
	var raw []map[string]any
	if err := json.NewDecoder(io.LimitReader(resp.Body, 2*1024*1024)).Decode(&raw); err != nil {
		return nil, ClientError{Code: "llamacpp_slot_list_decode_failed"}
	}
	slots := make([]SlotState, 0, len(raw))
	for _, item := range raw {
		slot := SlotState{
			SlotID:          intFromSlotMap(item, "id_slot", "slot_id", "id"),
			State:           intFromSlotMap(item, "state"),
			TaskID:          intFromSlotMap(item, "task_id"),
			PromptTokens:    int64FromSlotMap(item, "prompt_tokens", "n_prompt_tokens"),
			TokensCached:    int64FromSlotMap(item, "tokens_cached", "n_cached"),
			TokensEvaluated: int64FromSlotMap(item, "tokens_evaluated", "n_past"),
		}
		if status, _ := item["status"].(string); strings.TrimSpace(status) != "" {
			slot.RawStatus = cleanStatusText(status, maxStatusReasonLen)
		}
		slots = append(slots, slot)
	}
	return slots, nil
}

func (c OpenAIClient) SaveSlot(ctx context.Context, req SlotCacheRequest) (SlotCacheResult, error) {
	return c.postSlotAction(ctx, req, "save")
}

func (c OpenAIClient) RestoreSlot(ctx context.Context, req SlotCacheRequest) (SlotCacheResult, error) {
	return c.postSlotAction(ctx, req, "restore")
}

func (c OpenAIClient) EraseSlot(ctx context.Context, req SlotCacheRequest) (SlotCacheResult, error) {
	return c.postSlotAction(ctx, req, "erase")
}

func (c OpenAIClient) postSlotAction(ctx context.Context, req SlotCacheRequest, action string) (SlotCacheResult, error) {
	req.BaseURL = strings.TrimRight(strings.TrimSpace(req.BaseURL), "/")
	req.Filename = cleanSlotFilename(req.Filename)
	action = strings.TrimSpace(action)
	if req.BaseURL == "" || !isLocalBaseURL(req.BaseURL) {
		return SlotCacheResult{}, ClientError{Code: "llamacpp_invalid_base_url"}
	}
	if req.SlotID < 0 {
		return SlotCacheResult{}, ClientError{Code: "llamacpp_slot_id_invalid"}
	}
	if action == "save" || action == "restore" {
		if req.Filename == "" {
			return SlotCacheResult{}, ClientError{Code: "llamacpp_slot_filename_invalid"}
		}
	}

	var body io.Reader
	if action == "save" || action == "restore" {
		encoded, err := json.Marshal(slotActionRequest{Filename: req.Filename})
		if err != nil {
			return SlotCacheResult{}, ClientError{Code: "llamacpp_slot_request_marshal_failed"}
		}
		body = bytes.NewReader(encoded)
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("%s/slots/%d?action=%s", req.BaseURL, req.SlotID, action), body)
	if err != nil {
		return SlotCacheResult{}, ClientError{Code: "llamacpp_slot_request_build_failed"}
	}
	if body != nil {
		httpReq.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.httpClient().Do(httpReq)
	if err != nil {
		return SlotCacheResult{}, ClientError{Code: slotRequestErrorCode(ctx, "llamacpp_slot_"+action+"_failed")}
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
		return SlotCacheResult{}, ClientError{Code: "llamacpp_slot_" + action + "_failed", StatusCode: resp.StatusCode}
	}
	var raw slotActionResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, 2*1024*1024)).Decode(&raw); err != nil {
		return SlotCacheResult{}, ClientError{Code: "llamacpp_slot_" + action + "_decode_failed"}
	}
	return raw.result(action), nil
}

func (c OpenAIClient) httpClient() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return http.DefaultClient
}

type slotActionRequest struct {
	Filename string `json:"filename"`
}

type slotActionResponse struct {
	IDSlot    int    `json:"id_slot"`
	SlotID    int    `json:"slot_id"`
	Filename  string `json:"filename"`
	NSaved    int64  `json:"n_saved"`
	NWritten  int64  `json:"n_written"`
	NRestored int64  `json:"n_restored"`
	NRead     int64  `json:"n_read"`
	NErased   int64  `json:"n_erased"`
}

func (r slotActionResponse) result(action string) SlotCacheResult {
	slotID := r.IDSlot
	if slotID == 0 && r.SlotID != 0 {
		slotID = r.SlotID
	}
	return SlotCacheResult{
		Action:         strings.TrimSpace(action),
		SlotID:         slotID,
		Filename:       cleanSlotFilename(r.Filename),
		SavedTokens:    r.NSaved,
		WrittenBytes:   r.NWritten,
		RestoredTokens: r.NRestored,
		ReadBytes:      r.NRead,
		ErasedTokens:   r.NErased,
	}
}

func cleanSlotFilename(value string) string {
	value = strings.TrimSpace(value)
	if value == "" || len(value) > maxSlotFilenameLen {
		return ""
	}
	for _, r := range value {
		switch {
		case r >= 'a' && r <= 'z':
		case r >= 'A' && r <= 'Z':
		case r >= '0' && r <= '9':
		case r == '_' || r == '-' || r == '.':
		default:
			return ""
		}
	}
	if strings.Contains(value, "..") || strings.Contains(value, "/") || strings.Contains(value, `\`) {
		return ""
	}
	return value
}

func slotRequestErrorCode(ctx context.Context, fallback string) string {
	if ctx != nil && ctx.Err() != nil {
		return "llamacpp_timeout"
	}
	return fallback
}

func intFromSlotMap(item map[string]any, keys ...string) int {
	value := int64FromSlotMap(item, keys...)
	if value > int64(^uint(0)>>1) {
		return 0
	}
	return int(value)
}

func int64FromSlotMap(item map[string]any, keys ...string) int64 {
	for _, key := range keys {
		value, ok := item[key]
		if !ok {
			continue
		}
		switch typed := value.(type) {
		case float64:
			if typed >= 0 {
				return int64(typed)
			}
		case int64:
			if typed >= 0 {
				return typed
			}
		case int:
			if typed >= 0 {
				return int64(typed)
			}
		case string:
			parsed, err := strconv.ParseInt(strings.TrimSpace(typed), 10, 64)
			if err == nil && parsed >= 0 {
				return parsed
			}
		}
	}
	return 0
}
