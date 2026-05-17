package llamacppdemo

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/Ryvion/ryvion-node/internal/hub"
	nodespec "github.com/Ryvion/ryvion-node/internal/inference/speculative"
	llamacpp "github.com/Ryvion/ryvion-node/internal/runtimes/llamacpp"
)

const Executor = "llamacpp_demo_verifier"

var ErrUnavailable = errors.New("llamacpp_demo_verifier_unavailable")

type Sidecar interface {
	Start(context.Context) llamacpp.LlamaCppSidecarStatus
	Status(context.Context) llamacpp.LlamaCppSidecarStatus
}

type Verifier struct {
	Sidecar Sidecar
	Client  llamacpp.CompletionClient
}

func NewVerifierFromEnv() Verifier {
	return Verifier{
		Sidecar: llamacpp.NewManagerFromEnv(),
		Client:  llamacpp.OpenAIClient{},
	}
}

func (v Verifier) VerifyWave(ctx context.Context, spec nodespec.HotSessionSpec, command hub.SpeculativeLiveLabSessionCommand, acceptedTokensTotal int) (hub.SpeculativeLiveLabVerifierResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	started := time.Now()
	sidecar := v.sidecar()
	status := sidecar.Start(ctx)
	if !StatusReady(status) {
		return hub.SpeculativeLiveLabVerifierResult{}, ErrUnavailable
	}
	acceptedLimit, treeCID := nodespec.AcceptedFromTree(command.Tree, command.CommandID)
	remaining := 0
	if spec.MaxTokens > 0 {
		remaining = spec.MaxTokens - maxInt(0, acceptedTokensTotal)
		if remaining <= 0 {
			remaining = 1
		}
	}
	maxTokens := acceptedLimit
	if remaining > 0 && remaining < maxTokens {
		maxTokens = remaining
	}
	if maxTokens <= 0 {
		maxTokens = 1
	}
	req := llamacpp.CompletionRequest{
		BaseURL:      status.BaseURL,
		ModelID:      firstNonEmpty(spec.ModelID, status.ModelFilename, llamacpp.BackendName),
		SystemPrompt: SystemPrompt(),
		Prompt:       VerifierPrompt(spec, command, treeCID, acceptedLimit),
		MaxTokens:    maxTokens,
		Temperature:  0,
		Stream:       status.SupportsStreaming,
	}
	completion, err := v.client().Complete(ctx, req)
	if err != nil {
		return hub.SpeculativeLiveLabVerifierResult{}, err
	}
	text := strings.TrimSpace(string(completion.Output))
	tokensGenerated := int(completion.TokensGenerated)
	if tokensGenerated <= 0 {
		tokensGenerated = len(strings.Fields(text))
	}
	acceptedLen := minPositiveInt(tokensGenerated, acceptedLimit, maxTokens)
	if acceptedLen <= 0 {
		acceptedLen = minPositiveInt(acceptedLimit, maxTokens, 1)
	}
	durationMs := completion.TotalTimeMs
	if durationMs <= 0 {
		durationMs = maxInt64(1, time.Since(started).Milliseconds())
	}
	stopReason, eos := LabStopReason(completion)
	result := hub.SpeculativeLiveLabVerifierResult{
		WindowID:           command.WindowID,
		WaveIndex:          command.WaveIndex,
		AcceptedLen:        acceptedLen,
		TreeCID:            treeCID,
		DurationMs:         durationMs,
		AcceptedText:       text,
		AcceptedTextPublic: text != "",
		EOS:                eos,
		StopReason:         stopReason,
		ProbeSummary:       ProbeSummary(status, req, completion, durationMs),
	}
	if completion.MaxTokensReached && result.StopReason == "" {
		result.StopReason = "max_tokens"
	}
	return result, nil
}

func (v Verifier) Status(ctx context.Context) llamacpp.LlamaCppSidecarStatus {
	return v.sidecar().Status(ctx)
}

func (v Verifier) sidecar() Sidecar {
	if v.Sidecar != nil {
		return v.Sidecar
	}
	return llamacpp.NewManagerFromEnv()
}

func (v Verifier) client() llamacpp.CompletionClient {
	if v.Client != nil {
		return v.Client
	}
	return llamacpp.OpenAIClient{}
}

func StatusReady(status llamacpp.LlamaCppSidecarStatus) bool {
	return status.Enabled && status.Available && status.Running && status.Healthy && status.BaseURL != "" && status.SupportsTextGeneration
}

func SystemPrompt() string {
	return "You are Ryvion's local llama.cpp verifier. Verify the speculative draft tree against the prompt and return only the accepted continuation text."
}

func VerifierPrompt(spec nodespec.HotSessionSpec, command hub.SpeculativeLiveLabSessionCommand, treeCID string, acceptedLimit int) string {
	branchCount := len(sliceFromAny(command.Tree["branches"]))
	return strings.Join([]string{
		"Verify this Ryvion speculative draft wave locally.",
		"run_id=" + spec.RunID,
		"session_id=" + spec.SessionID,
		"workgraph_id=" + firstNonEmpty(command.WorkGraphID, spec.WorkGraphID),
		"window_id=" + command.WindowID,
		"prefix_hash=" + spec.ParentPrefixHash,
		"tree_cid=" + treeCID,
		fmt.Sprintf("branch_count=%d", branchCount),
		fmt.Sprintf("candidate_acceptance_limit=%d", acceptedLimit),
		"User prompt:",
		spec.Prompt,
	}, "\n")
}

func ProbeSummary(status llamacpp.LlamaCppSidecarStatus, req llamacpp.CompletionRequest, completion llamacpp.CompletionResult, durationMs int64) map[string]any {
	return map[string]any{
		"source":                     Executor,
		"backend":                    llamacpp.BackendName,
		"runtime_mode":               "native_node_agent",
		"model_id":                   req.ModelID,
		"model_filename":             status.ModelFilename,
		"output_hash":                "sha256:" + sha256Hex(completion.Output),
		"output_bytes":               completion.OutputBytes,
		"tokens_generated":           completion.TokensGenerated,
		"completion_tokens":          completion.CompletionTokens,
		"requested_max_tokens":       completion.RequestedMaxTokens,
		"finish_reason":              completion.FinishReason,
		"backend_finish_reason":      completion.BackendFinishReason,
		"backend_stop_reason":        completion.BackendStopReason,
		"max_tokens_reached":         completion.MaxTokensReached,
		"runtime_measurement_status": completion.RuntimeMeasurementStatus,
		"metadata_parse_status":      completion.MetadataParseStatus,
		"ttft_ms":                    completion.TTFTMs,
		"duration_ms":                durationMs,
		"streamed":                   completion.Streamed,
	}
}

func LabStopReason(completion llamacpp.CompletionResult) (string, bool) {
	if completion.MaxTokensReached {
		return "max_tokens", false
	}
	for _, value := range []string{completion.FinishReason, completion.BackendStopReason, completion.BackendFinishReason} {
		switch strings.ToLower(strings.TrimSpace(value)) {
		case llamacpp.FinishReasonLength, llamacpp.FinishReasonMaxTokens, "limit", "stopped_limit", "token_limit", "context_length", "max_new_tokens", "max_token", "max_tokens_reached":
			return "max_tokens", false
		case llamacpp.FinishReasonTimeout, "timed_out", "deadline", "deadline_exceeded":
			return "timeout", false
		case llamacpp.FinishReasonError:
			return "llamacpp_demo_completion_error", false
		case "eos", "stopped_eos", "end_of_sequence", "end-of-sequence", "end_of_text", "eos_token":
			return "eos", true
		}
	}
	return "", false
}

func UnavailableCode(status llamacpp.LlamaCppSidecarStatus) string {
	switch {
	case !status.Enabled:
		return "llamacpp_demo_disabled"
	case !status.Available:
		return "llamacpp_demo_unavailable"
	case !status.Running:
		return "llamacpp_demo_not_running"
	case !status.Healthy:
		return "llamacpp_demo_unhealthy"
	case status.BaseURL == "":
		return "llamacpp_demo_base_url_missing"
	case !status.SupportsTextGeneration:
		return "llamacpp_demo_text_generation_unavailable"
	default:
		return "llamacpp_demo_unavailable"
	}
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func sliceFromAny(value any) []any {
	if typed, ok := value.([]any); ok {
		return typed
	}
	return nil
}

func minPositiveInt(values ...int) int {
	out := 0
	for _, value := range values {
		if value <= 0 {
			continue
		}
		if out == 0 || value < out {
			out = value
		}
	}
	return out
}

func maxInt(left int, right int) int {
	if left > right {
		return left
	}
	return right
}

func maxInt64(left int64, right int64) int64 {
	if left > right {
		return left
	}
	return right
}

func sha256Hex(value []byte) string {
	sum := sha256.Sum256(value)
	return hex.EncodeToString(sum[:])
}
