package dashboardinference

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/Ryvion/ryvion-node/internal/runtimes/llamacpp"
	"github.com/Ryvion/ryvion-node/internal/v7/inferenceconfig"
)

const (
	Task              = "v7_dashboard_inference"
	FlagEnv           = "RYV_NODE_V7_DASHBOARD_INFERENCE"
	TextOutputFlagEnv = "RYV_NODE_V7_INFERENCE_TEXT_OUTPUT"
	StreamingFlagEnv  = "RYV_NODE_V7_INFERENCE_STREAMING"
	V8StreamEventsEnv = "RYV_NODE_V8_STREAM_EVENTS"
	DisableFlagEnv    = inferenceconfig.EnvDisableV7Inference
	DisableTextEnv    = inferenceconfig.EnvDisableTextOutput
	DisableStreamEnv  = inferenceconfig.EnvDisableStreaming

	defaultTimeoutMs = int64(2 * 60 * 1000)

	defaultMaxReturnChars = 8192
	maxIDLen              = 256
	maxModelIDLen         = 256
	maxModelPathLen       = 2048
	maxPromptChars        = 32768
	maxSystemPromptChars  = 4096
	maxMessages           = 32
	maxMaxTokens          = 8192
	maxStatusErrLen       = 512
	maxSessionIDLen       = 256
	maxCacheStateIDLen    = 256
	maxSlotID             = 1024
	maxCacheReuseTokens   = 1_000_000
	maxAffinityTTLSeconds = 24 * 60 * 60
)

var ErrInvalidSpec = errors.New("dashboardinference: invalid spec")

const defaultRyvionGroundingSystemPrompt = "You are Ryvion's local dashboard inference backend. Ground answers in these facts: Ryvion runs AI workloads on operator-controlled nodes, reports local runtime capability and warm model readiness, and returns privacy-safe hashes, metrics, receipts, and status metadata. Do not describe Ryvion as generic cloud hosting or claim unsupported managed-cloud features."

type Spec struct {
	Task                      string                       `json:"task"`
	RequestID                 string                       `json:"request_id"`
	RunID                     string                       `json:"run_id"`
	JobID                     string                       `json:"job_id"`
	Backend                   string                       `json:"backend"`
	ModelID                   string                       `json:"model_id"`
	ModelPath                 string                       `json:"model_path,omitempty"`
	TargetNodeID              string                       `json:"target_node_id"`
	Prompt                    string                       `json:"prompt,omitempty"`
	SystemPrompt              string                       `json:"system_prompt,omitempty"`
	Messages                  []llamacpp.CompletionMessage `json:"messages,omitempty"`
	UseDefaultRyvionGrounding bool                         `json:"use_default_ryvion_grounding,omitempty"`
	ReturnText                bool                         `json:"return_text,omitempty"`
	MaxReturnChars            int                          `json:"max_return_chars,omitempty"`
	MaxTokens                 int                          `json:"max_tokens"`
	Stream                    bool                         `json:"stream"`
	CreatedAtUnixMs           int64                        `json:"created_at_unix_ms"`
	PromptHash                string                       `json:"prompt_hash,omitempty"`
	PromptProfileID           string                       `json:"prompt_profile_id,omitempty"`
	CachePolicy               CachePolicy                  `json:"cache_policy,omitempty"`
}

type CachePolicy struct {
	SessionID            string `json:"session_id,omitempty"`
	PrefixHash           string `json:"prefix_hash,omitempty"`
	CachePrompt          bool   `json:"cache_prompt,omitempty"`
	CacheReuseTokens     int    `json:"cache_reuse_tokens,omitempty"`
	SlotID               *int   `json:"slot_id,omitempty"`
	SaveSlotAfterRun     bool   `json:"save_slot_after_run,omitempty"`
	RestoreSlotBeforeRun bool   `json:"restore_slot_before_run,omitempty"`
	CacheStateID         string `json:"cache_state_id,omitempty"`
	PreferredNodeID      string `json:"preferred_node_id,omitempty"`
	AffinityTTLSeconds   int    `json:"affinity_ttl_seconds,omitempty"`
}

type AssignmentIdentity struct {
	RequestID       string
	RunID           string
	JobID           string
	Backend         string
	ModelID         string
	PromptHash      string
	PromptProfileID string
}

func IsSpecJSON(specJSON string) bool {
	var header struct {
		Task string `json:"task"`
	}
	if json.Unmarshal([]byte(strings.TrimSpace(specJSON)), &header) != nil {
		return false
	}
	return strings.TrimSpace(header.Task) == Task
}

func EnabledFromEnv(getenv func(string) string) bool {
	return inferenceconfig.V7InferenceEnabled(getenv)
}

func TextOutputEnabledFromEnv(getenv func(string) string) bool {
	return inferenceconfig.TextOutputEnabled(getenv)
}

func StreamingEnabledFromEnv(getenv func(string) string) bool {
	return inferenceconfig.StreamingEnabled(getenv)
}

func V8StreamEventsEnabledFromEnv(getenv func(string) string) bool {
	if getenv == nil {
		getenv = func(string) string { return "" }
	}
	value := strings.ToLower(strings.TrimSpace(getenv(V8StreamEventsEnv)))
	switch value {
	case "", "1", "true", "yes", "on", "enabled":
		return true
	default:
		return false
	}
}

func AssignmentIdentityFromJSON(specJSON string) (AssignmentIdentity, bool) {
	var header struct {
		Task            string `json:"task"`
		RequestID       string `json:"request_id"`
		RunID           string `json:"run_id"`
		JobID           string `json:"job_id"`
		Backend         string `json:"backend"`
		ModelID         string `json:"model_id"`
		PromptHash      string `json:"prompt_hash"`
		PromptProfileID string `json:"prompt_profile_id"`
	}
	if json.Unmarshal([]byte(strings.TrimSpace(specJSON)), &header) != nil {
		return AssignmentIdentity{}, false
	}
	if strings.TrimSpace(header.Task) != Task {
		return AssignmentIdentity{}, false
	}
	return AssignmentIdentity{
		RequestID:       cleanText(header.RequestID, maxIDLen),
		RunID:           cleanText(header.RunID, maxIDLen),
		JobID:           cleanText(header.JobID, maxIDLen),
		Backend:         normalizeBackendName(header.Backend),
		ModelID:         cleanText(header.ModelID, maxModelIDLen),
		PromptHash:      strings.TrimSpace(header.PromptHash),
		PromptProfileID: cleanText(header.PromptProfileID, maxIDLen),
	}, true
}

func DecodeSpec(specJSON string) (Spec, error) {
	raw := strings.TrimSpace(specJSON)
	if raw == "" {
		return Spec{}, fmt.Errorf("%w: spec_json required", ErrInvalidSpec)
	}
	var spec Spec
	if err := json.Unmarshal([]byte(raw), &spec); err != nil {
		return Spec{}, fmt.Errorf("%w: decode spec_json: %v", ErrInvalidSpec, err)
	}
	spec = normalizeSpec(spec)
	if err := ValidateSpec(spec); err != nil {
		return Spec{}, err
	}
	return spec, nil
}

func ValidateSpec(spec Spec) error {
	spec = normalizeSpec(spec)
	var errs []error
	if spec.Task == "" {
		errs = append(errs, fmt.Errorf("%w: task required", ErrInvalidSpec))
	} else if spec.Task != Task {
		errs = append(errs, fmt.Errorf("%w: task must be %q", ErrInvalidSpec, Task))
	}
	if spec.RequestID == "" {
		errs = append(errs, fmt.Errorf("%w: request_id required", ErrInvalidSpec))
	}
	if spec.RunID == "" {
		errs = append(errs, fmt.Errorf("%w: run_id required", ErrInvalidSpec))
	}
	if spec.JobID == "" {
		errs = append(errs, fmt.Errorf("%w: job_id required", ErrInvalidSpec))
	}
	if spec.Backend == "" {
		errs = append(errs, fmt.Errorf("%w: backend required", ErrInvalidSpec))
	} else if spec.Backend != llamacpp.BackendName {
		errs = append(errs, fmt.Errorf("%w: unsupported backend %q", ErrInvalidSpec, spec.Backend))
	}
	if spec.ModelID == "" {
		errs = append(errs, fmt.Errorf("%w: model_id required", ErrInvalidSpec))
	}
	if spec.TargetNodeID == "" {
		errs = append(errs, fmt.Errorf("%w: target_node_id required", ErrInvalidSpec))
	}
	if spec.ReturnText && spec.Prompt == "" && len(spec.Messages) == 0 {
		errs = append(errs, fmt.Errorf("%w: prompt or messages required when return_text is true", ErrInvalidSpec))
	}
	if spec.MaxTokens <= 0 {
		errs = append(errs, fmt.Errorf("%w: max_tokens must be greater than zero", ErrInvalidSpec))
	} else if spec.MaxTokens > maxMaxTokens {
		errs = append(errs, fmt.Errorf("%w: max_tokens exceeds maximum %d", ErrInvalidSpec, maxMaxTokens))
	}
	if spec.CreatedAtUnixMs <= 0 {
		errs = append(errs, fmt.Errorf("%w: created_at_unix_ms must be greater than zero", ErrInvalidSpec))
	}
	if spec.PromptHash != "" {
		if err := validateSHA256ID(spec.PromptHash, "prompt_hash", ErrInvalidSpec); err != nil {
			errs = append(errs, err)
		}
	}
	if err := validateCachePolicy(spec.CachePolicy); err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

func normalizeSpec(spec Spec) Spec {
	spec.Task = strings.TrimSpace(spec.Task)
	spec.RequestID = cleanText(spec.RequestID, maxIDLen)
	spec.RunID = cleanText(spec.RunID, maxIDLen)
	spec.JobID = cleanText(spec.JobID, maxIDLen)
	spec.Backend = normalizeBackendName(spec.Backend)
	spec.ModelID = cleanText(spec.ModelID, maxModelIDLen)
	spec.ModelPath = cleanPath(spec.ModelPath, maxModelPathLen)
	spec.TargetNodeID = cleanText(spec.TargetNodeID, maxIDLen)
	spec.Prompt = cleanPrompt(spec.Prompt, maxPromptChars)
	spec.SystemPrompt = cleanPrompt(spec.SystemPrompt, maxSystemPromptChars)
	spec.Messages = normalizeSpecMessages(spec.Messages)
	if spec.UseDefaultRyvionGrounding && spec.SystemPrompt == "" {
		spec.SystemPrompt = defaultRyvionGroundingSystemPrompt
	}
	if spec.MaxReturnChars <= 0 || spec.MaxReturnChars > defaultMaxReturnChars {
		spec.MaxReturnChars = defaultMaxReturnChars
	}
	spec.PromptHash = strings.TrimSpace(spec.PromptHash)
	spec.PromptProfileID = cleanText(spec.PromptProfileID, maxIDLen)
	spec.CachePolicy = normalizeCachePolicy(spec.CachePolicy)
	return spec
}

func normalizeCachePolicy(policy CachePolicy) CachePolicy {
	policy.SessionID = cleanText(policy.SessionID, maxSessionIDLen)
	policy.PrefixHash = strings.TrimSpace(policy.PrefixHash)
	policy.CacheStateID = cleanCacheStateID(policy.CacheStateID)
	policy.PreferredNodeID = cleanText(policy.PreferredNodeID, maxIDLen)
	if policy.CacheReuseTokens < 0 {
		policy.CacheReuseTokens = 0
	}
	if policy.CacheReuseTokens > maxCacheReuseTokens {
		policy.CacheReuseTokens = maxCacheReuseTokens
	}
	if policy.AffinityTTLSeconds < 0 {
		policy.AffinityTTLSeconds = 0
	}
	if policy.AffinityTTLSeconds > maxAffinityTTLSeconds {
		policy.AffinityTTLSeconds = maxAffinityTTLSeconds
	}
	if policy.SlotID != nil {
		slotID := *policy.SlotID
		if slotID < 0 || slotID > maxSlotID {
			policy.SlotID = nil
		} else {
			policy.SlotID = &slotID
		}
	}
	if policy.CachePrompt && policy.AffinityTTLSeconds == 0 && policy.SessionID != "" {
		policy.AffinityTTLSeconds = 10 * 60
	}
	return policy
}

func validateCachePolicy(policy CachePolicy) error {
	policy = normalizeCachePolicy(policy)
	var errs []error
	if policy.PrefixHash != "" {
		if err := validateSHA256ID(policy.PrefixHash, "cache_policy.prefix_hash", ErrInvalidSpec); err != nil {
			errs = append(errs, err)
		}
	}
	if policy.CacheStateID != "" && cleanCacheStateID(policy.CacheStateID) != policy.CacheStateID {
		errs = append(errs, fmt.Errorf("%w: cache_policy.cache_state_id contains unsafe characters", ErrInvalidSpec))
	}
	if policy.SlotID != nil && (*policy.SlotID < 0 || *policy.SlotID > maxSlotID) {
		errs = append(errs, fmt.Errorf("%w: cache_policy.slot_id out of range", ErrInvalidSpec))
	}
	if policy.CacheReuseTokens < 0 || policy.CacheReuseTokens > maxCacheReuseTokens {
		errs = append(errs, fmt.Errorf("%w: cache_policy.cache_reuse_tokens out of range", ErrInvalidSpec))
	}
	return errors.Join(errs...)
}

func normalizeSpecMessages(messages []llamacpp.CompletionMessage) []llamacpp.CompletionMessage {
	if len(messages) == 0 {
		return nil
	}
	if len(messages) > maxMessages {
		messages = messages[:maxMessages]
	}
	normalized := make([]llamacpp.CompletionMessage, 0, len(messages))
	for _, message := range messages {
		role := strings.ToLower(strings.TrimSpace(message.Role))
		switch role {
		case "system", "user", "assistant":
		default:
			continue
		}
		content := cleanPrompt(message.Content, maxPromptChars)
		if content == "" {
			continue
		}
		normalized = append(normalized, llamacpp.CompletionMessage{Role: role, Content: content})
	}
	return normalized
}

func normalizeBackendName(value string) string {
	value = strings.TrimSpace(value)
	normalized := strings.ToLower(value)
	normalized = strings.NewReplacer(".", "", "_", "", "-", "", " ", "").Replace(normalized)
	if normalized == "llamacpp" {
		return llamacpp.BackendName
	}
	return cleanText(value, maxIDLen)
}

func validateSHA256ID(value string, field string, base error) error {
	value = strings.TrimSpace(value)
	if value == "" {
		return fmt.Errorf("%w: %s required", base, field)
	}
	if !strings.HasPrefix(value, "sha256:") || len(value) != len("sha256:")+64 {
		return fmt.Errorf("%w: %s must be sha256:<64 hex>", base, field)
	}
	for _, r := range strings.TrimPrefix(value, "sha256:") {
		if (r < '0' || r > '9') && (r < 'a' || r > 'f') {
			return fmt.Errorf("%w: %s must be lowercase hex", base, field)
		}
	}
	return nil
}

func cleanText(value string, maxRunes int) string {
	value = strings.TrimSpace(value)
	value = strings.ReplaceAll(value, "\r", " ")
	value = strings.ReplaceAll(value, "\n", " ")
	value = strings.Join(strings.Fields(value), " ")
	if maxRunes <= 0 {
		return value
	}
	runes := []rune(value)
	if len(runes) <= maxRunes {
		return value
	}
	return string(runes[:maxRunes])
}

func cleanPrompt(value string, maxRunes int) string {
	value = strings.TrimSpace(value)
	value = strings.ReplaceAll(value, "\r\n", "\n")
	value = strings.ReplaceAll(value, "\r", "\n")
	if maxRunes <= 0 {
		return value
	}
	runes := []rune(value)
	if len(runes) <= maxRunes {
		return value
	}
	return string(runes[:maxRunes])
}

func cleanPath(value string, maxRunes int) string {
	value = strings.TrimSpace(value)
	value = strings.ReplaceAll(value, "\r", "")
	value = strings.ReplaceAll(value, "\n", "")
	if maxRunes <= 0 {
		return value
	}
	runes := []rune(value)
	if len(runes) <= maxRunes {
		return value
	}
	return string(runes[:maxRunes])
}

func cleanCacheStateID(value string) string {
	value = cleanText(value, maxCacheStateIDLen)
	if value == "" {
		return ""
	}
	var b strings.Builder
	for _, r := range value {
		switch {
		case r >= 'a' && r <= 'z':
			b.WriteRune(r)
		case r >= 'A' && r <= 'Z':
			b.WriteRune(r)
		case r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '_' || r == '-' || r == '.' || r == ':':
			b.WriteRune(r)
		}
	}
	return strings.TrimSpace(b.String())
}
