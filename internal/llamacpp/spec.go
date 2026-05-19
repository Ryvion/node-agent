package llamacpp

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

const (
	ExecutorKind = "llama_cpp"
	Task         = "llama_cpp_inference"
	Schema       = "ryvion.llama_cpp_inference.v1"

	DefaultMaxTokens   = 256
	MaxAllowedTokens   = 8192
	MaxPromptBytes     = 256 * 1024
	MaxMessageCount    = 64
	MaxMessageRoleLen  = 32
	MaxModelIDLen      = 256
	DefaultTemperature = 0.2
	DefaultTopP        = 0.95
)

var ErrInvalidSpec = errors.New("llamacpp: invalid spec")

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Spec struct {
	Version        string    `json:"version,omitempty"`
	Task           string    `json:"task"`
	Model          string    `json:"model,omitempty"`
	Prompt         string    `json:"prompt,omitempty"`
	Messages       []Message `json:"messages,omitempty"`
	MaxTokens      int       `json:"max_tokens,omitempty"`
	Temperature    float64   `json:"temperature,omitempty"`
	TopP           float64   `json:"top_p,omitempty"`
	Seed           int       `json:"seed,omitempty"`
	TimeoutSeconds int       `json:"timeout_seconds,omitempty"`
	OutputArtifact bool      `json:"output_artifact,omitempty"`
}

func ShouldHandle(kind, executorKind, specJSON string) bool {
	if isLlamaName(executorKind) || isLlamaName(kind) {
		return true
	}
	var header struct {
		Task         string `json:"task"`
		ExecutorKind string `json:"executor_kind"`
	}
	if json.Unmarshal([]byte(strings.TrimSpace(specJSON)), &header) != nil {
		return false
	}
	return isLlamaName(header.Task) || isLlamaName(header.ExecutorKind)
}

func DecodeSpec(raw string) (Spec, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return Spec{}, fmt.Errorf("%w: spec_json required", ErrInvalidSpec)
	}
	if len(raw) > MaxPromptBytes {
		return Spec{}, fmt.Errorf("%w: spec_json too large", ErrInvalidSpec)
	}
	var spec Spec
	if err := json.Unmarshal([]byte(raw), &spec); err != nil {
		return Spec{}, fmt.Errorf("%w: decode spec_json: %v", ErrInvalidSpec, err)
	}
	spec = NormalizeSpec(spec)
	if err := ValidateSpec(spec); err != nil {
		return Spec{}, err
	}
	return spec, nil
}

func NormalizeSpec(spec Spec) Spec {
	spec.Version = strings.TrimSpace(spec.Version)
	spec.Task = normalizeName(spec.Task)
	spec.Model = cleanText(spec.Model, MaxModelIDLen)
	spec.Prompt = strings.TrimSpace(spec.Prompt)
	if spec.MaxTokens == 0 {
		spec.MaxTokens = DefaultMaxTokens
	}
	if spec.Temperature == 0 {
		spec.Temperature = DefaultTemperature
	}
	if spec.TopP == 0 {
		spec.TopP = DefaultTopP
	}
	if !spec.OutputArtifact {
		spec.OutputArtifact = true
	}
	out := spec.Messages[:0]
	for _, msg := range spec.Messages {
		role := cleanText(msg.Role, MaxMessageRoleLen)
		content := strings.TrimSpace(msg.Content)
		if role == "" && content == "" {
			continue
		}
		if role == "" {
			role = "user"
		}
		out = append(out, Message{Role: role, Content: content})
	}
	spec.Messages = out
	return spec
}

func ValidateSpec(spec Spec) error {
	var errs []error
	if !isLlamaName(spec.Task) {
		errs = append(errs, fmt.Errorf("%w: task must be %q", ErrInvalidSpec, Task))
	}
	if strings.TrimSpace(spec.Prompt) == "" && len(spec.Messages) == 0 {
		errs = append(errs, fmt.Errorf("%w: prompt or messages required", ErrInvalidSpec))
	}
	if len(spec.Prompt) > MaxPromptBytes {
		errs = append(errs, fmt.Errorf("%w: prompt too large", ErrInvalidSpec))
	}
	if len(spec.Messages) > MaxMessageCount {
		errs = append(errs, fmt.Errorf("%w: too many messages", ErrInvalidSpec))
	}
	for i, msg := range spec.Messages {
		if strings.TrimSpace(msg.Content) == "" {
			errs = append(errs, fmt.Errorf("%w: messages[%d].content required", ErrInvalidSpec, i))
		}
		if len(msg.Content) > MaxPromptBytes {
			errs = append(errs, fmt.Errorf("%w: messages[%d].content too large", ErrInvalidSpec, i))
		}
	}
	if spec.MaxTokens <= 0 || spec.MaxTokens > MaxAllowedTokens {
		errs = append(errs, fmt.Errorf("%w: max_tokens out of range", ErrInvalidSpec))
	}
	if spec.Temperature < 0 || spec.Temperature > 2 {
		errs = append(errs, fmt.Errorf("%w: temperature out of range", ErrInvalidSpec))
	}
	if spec.TopP < 0 || spec.TopP > 1 {
		errs = append(errs, fmt.Errorf("%w: top_p out of range", ErrInvalidSpec))
	}
	return errors.Join(errs...)
}

func PromptHash(spec Spec) string {
	spec = NormalizeSpec(spec)
	payload := struct {
		Prompt   string    `json:"prompt,omitempty"`
		Messages []Message `json:"messages,omitempty"`
	}{
		Prompt:   spec.Prompt,
		Messages: spec.Messages,
	}
	encoded, _ := json.Marshal(payload)
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:])
}

func OutputHash(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:])
}

func MessagesForRequest(spec Spec) []Message {
	spec = NormalizeSpec(spec)
	if len(spec.Messages) > 0 {
		out := make([]Message, len(spec.Messages))
		copy(out, spec.Messages)
		return out
	}
	return []Message{{Role: "user", Content: spec.Prompt}}
}

func isLlamaName(value string) bool {
	switch normalizeName(value) {
	case Task, ExecutorKind, "llamacpp", "llama-cpp", "ai_inference":
		return true
	default:
		return false
	}
}

func normalizeName(value string) string {
	return strings.ToLower(strings.TrimSpace(value))
}

func cleanText(value string, maxLen int) string {
	value = strings.TrimSpace(value)
	if maxLen > 0 && len(value) > maxLen {
		return value[:maxLen]
	}
	return value
}
