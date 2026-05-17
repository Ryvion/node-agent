package speculative

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
)

const (
	DraftRunnerTask        = "draft_runner_v8"
	VerifierSessionTask    = "verifier_session_v8"
	DraftHotSessionTask    = "draft_runner_v8_hot_session"
	VerifierHotSessionTask = "verifier_session_v8_hot"

	NativeExecutor = "native_foresight_v8"

	DraftBackendNative      = "contract_test_bridge"
	VerifierBackendBridge   = "contract_test_bridge"
	VerifierBackendSGLang   = "native_sglang"
	VerifierBackendLlamaCpp = "llamacpp_demo_verifier"

	DefaultNativeDraftConfidenceBPS = int64(7600)
)

type DraftSpec struct {
	Task                 string `json:"task"`
	ExecutorKind         string `json:"executor_kind,omitempty"`
	RunnerImage          string `json:"runner_image,omitempty"`
	DockerRequired       bool   `json:"docker_required,omitempty"`
	DraftBackend         string `json:"draft_backend,omitempty"`
	WorkGraphID          string `json:"workgraph_id"`
	WindowID             string `json:"window_id"`
	RoleID               string `json:"role_id"`
	TargetNodeID         string `json:"target_node_id"`
	NodeID               string `json:"node_id"`
	Prompt               string `json:"prompt"`
	ParentPrefixHash     string `json:"parent_prefix_hash"`
	BranchCount          int    `json:"branch_count"`
	Horizon              int    `json:"horizon"`
	DeadlineMs           int    `json:"deadline_ms"`
	ModelHash            string `json:"model_hash"`
	DrafterModelID       string `json:"drafter_model_id"`
	FirstPacketTimeoutMs int    `json:"first_packet_timeout_ms"`
}

type HotSessionSpec struct {
	Task             string `json:"task"`
	ExecutorKind     string `json:"executor_kind,omitempty"`
	RunnerImage      string `json:"runner_image,omitempty"`
	DockerRequired   bool   `json:"docker_required,omitempty"`
	DraftBackend     string `json:"draft_backend,omitempty"`
	VerifierBackend  string `json:"verifier_backend,omitempty"`
	RunID            string `json:"run_id"`
	SessionID        string `json:"session_id"`
	WorkGraphID      string `json:"workgraph_id"`
	RoleID           string `json:"role_id"`
	TargetNodeID     string `json:"target_node_id"`
	NodeID           string `json:"node_id"`
	Prompt           string `json:"prompt"`
	ParentPrefixHash string `json:"parent_prefix_hash"`
	ModelID          string `json:"model_id"`
	ModelHash        string `json:"model_hash"`
	ModelPath        string `json:"model_path,omitempty"`
	DrafterModelID   string `json:"drafter_model_id"`
	MaxTokens        int    `json:"max_tokens"`
}

func DecodeDraftSpec(specJSON string) (DraftSpec, bool) {
	var spec DraftSpec
	if json.Unmarshal([]byte(strings.TrimSpace(specJSON)), &spec) != nil {
		return DraftSpec{}, false
	}
	if strings.TrimSpace(spec.Task) != DraftRunnerTask {
		return DraftSpec{}, false
	}
	spec.ExecutorKind = strings.TrimSpace(spec.ExecutorKind)
	spec.RunnerImage = strings.TrimSpace(spec.RunnerImage)
	spec.DraftBackend = CanonicalDraftBackend(spec.DraftBackend)
	spec.WorkGraphID = strings.TrimSpace(spec.WorkGraphID)
	spec.WindowID = strings.TrimSpace(spec.WindowID)
	spec.RoleID = firstNonEmpty(strings.TrimSpace(spec.RoleID), "draft-worker-native")
	spec.TargetNodeID = strings.TrimSpace(spec.TargetNodeID)
	spec.NodeID = firstNonEmpty(strings.TrimSpace(spec.NodeID), spec.TargetNodeID)
	spec.ParentPrefixHash = strings.TrimSpace(spec.ParentPrefixHash)
	spec.ModelHash = strings.TrimSpace(spec.ModelHash)
	spec.DrafterModelID = strings.TrimSpace(spec.DrafterModelID)
	if spec.BranchCount <= 0 {
		spec.BranchCount = 1
	}
	if spec.BranchCount > 16 {
		spec.BranchCount = 16
	}
	spec.Horizon = NormalizeHorizon(spec.Horizon)
	if spec.DeadlineMs <= 0 {
		spec.DeadlineMs = 1000
	}
	if spec.ModelHash == "" {
		spec.ModelHash = "sha256:" + FullHash(firstNonEmpty(spec.DrafterModelID, "native-drafter"))
	}
	return spec, spec.WindowID != "" && spec.ParentPrefixHash != ""
}

func DecodeHotSessionSpec(specJSON string, expectedTask string) (HotSessionSpec, bool) {
	var spec HotSessionSpec
	if json.Unmarshal([]byte(strings.TrimSpace(specJSON)), &spec) != nil {
		return HotSessionSpec{}, false
	}
	if strings.TrimSpace(spec.Task) != expectedTask {
		return HotSessionSpec{}, false
	}
	spec.ExecutorKind = strings.TrimSpace(spec.ExecutorKind)
	spec.RunnerImage = strings.TrimSpace(spec.RunnerImage)
	spec.DraftBackend = CanonicalDraftBackend(spec.DraftBackend)
	spec.VerifierBackend = VerifierBackendKind(spec.VerifierBackend)
	spec.RunID = strings.TrimSpace(spec.RunID)
	spec.SessionID = strings.TrimSpace(spec.SessionID)
	spec.WorkGraphID = strings.TrimSpace(spec.WorkGraphID)
	spec.RoleID = strings.TrimSpace(spec.RoleID)
	spec.TargetNodeID = strings.TrimSpace(spec.TargetNodeID)
	spec.NodeID = firstNonEmpty(strings.TrimSpace(spec.NodeID), spec.TargetNodeID)
	spec.ParentPrefixHash = strings.TrimSpace(spec.ParentPrefixHash)
	spec.ModelID = strings.TrimSpace(spec.ModelID)
	spec.ModelHash = strings.TrimSpace(spec.ModelHash)
	spec.ModelPath = strings.TrimSpace(spec.ModelPath)
	spec.DrafterModelID = strings.TrimSpace(spec.DrafterModelID)
	return spec, spec.RunID != "" && spec.WorkGraphID != ""
}

func DecodeVerifierSpec(specJSON string) (int, string, string, bool) {
	var spec map[string]any
	if json.Unmarshal([]byte(strings.TrimSpace(specJSON)), &spec) != nil {
		return 0, "", "", false
	}
	if strings.TrimSpace(stringValue(spec["task"])) != VerifierSessionTask {
		return 0, "", "", false
	}
	backend := VerifierBackendKind(stringValue(spec["verifier_backend"]))
	tree := mapFromAny(spec["tree"])
	acceptedLen, treeCID := AcceptedFromTree(tree, specJSON)
	return acceptedLen, treeCID, backend, true
}

func CanonicalDraftBackend(backend string) string {
	backend = strings.ToLower(strings.TrimSpace(backend))
	switch backend {
	case "", DraftBackendNative, "native_bridge", "deterministic_native_bridge":
		return DraftBackendNative
	default:
		return strings.TrimSpace(backend)
	}
}

func DraftBackendIsNativeBridge(backend string) bool {
	return CanonicalDraftBackend(backend) == DraftBackendNative
}

func VerifierBackendKind(backend string) string {
	backend = strings.ToLower(strings.TrimSpace(backend))
	switch backend {
	case "", VerifierBackendBridge, "native_bridge", "deterministic_native_bridge":
		return VerifierBackendBridge
	case VerifierBackendSGLang, "sglang":
		return VerifierBackendSGLang
	case VerifierBackendLlamaCpp, "native_llamacpp", "llamacpp", "llama.cpp", "llama_cpp", "llama-cpp":
		return VerifierBackendLlamaCpp
	default:
		return backend
	}
}

func AcceptedFromTree(tree map[string]any, fallback string) (int, string) {
	if len(tree) == 0 {
		return 1, "sha256:" + FullHash(fallback)
	}
	branches := sliceFromAny(tree["branches"])
	acceptedLen := 0
	for _, raw := range branches {
		branch := mapFromAny(raw)
		tokens := sliceFromAny(branch["candidate_tokens"])
		if len(tokens) > acceptedLen {
			acceptedLen = len(tokens)
		}
	}
	if acceptedLen <= 0 {
		acceptedLen = 1
	}
	if acceptedLen > 8 {
		acceptedLen = 8
	}
	treeCID := strings.TrimSpace(stringValue(tree["tree_cid"]))
	if treeCID == "" {
		encoded, _ := json.Marshal(tree)
		treeCID = "sha256:" + FullHash(string(encoded))
	}
	return acceptedLen, treeCID
}

func AcceptedTextForWave(prompt string, wave int, acceptedLen int) string {
	words := []string{"Ryvion", "Foresight", "Mesh", "keeps", "verifier", "sessions", "hot", "while", "draft", "tokens", "are", "checked", "and", "committed", "quickly."}
	if strings.Contains(strings.ToLower(prompt), "assembly") {
		words = []string{"Assembly", "work", "uses", "low-level", "instructions", "while", "Ryvion", "verifies", "draft", "branches", "quickly."}
	}
	if acceptedLen <= 0 {
		acceptedLen = 1
	}
	start := (maxInt(1, wave) - 1) * acceptedLen
	if start >= len(words) {
		return ""
	}
	end := start + acceptedLen
	if end > len(words) {
		end = len(words)
	}
	text := strings.Join(words[start:end], " ")
	if end < len(words) {
		text += " "
	}
	return text
}

func DeterministicTokens(seed string, branch int, horizon int) []int {
	horizon = NormalizeHorizon(horizon)
	tokens := make([]int, 0, horizon)
	for len(tokens) < horizon {
		sum := sha256.Sum256([]byte(fmt.Sprintf("%s|branch:%d|offset:%d", seed, branch, len(tokens))))
		for offset := 0; offset+4 <= len(sum) && len(tokens) < horizon; offset += 4 {
			token := int(binary.BigEndian.Uint32(sum[offset:offset+4])%32000) + 1
			tokens = append(tokens, token)
		}
	}
	return tokens
}

func NormalizeHorizon(horizon int) int {
	if horizon <= 0 {
		return 8
	}
	if horizon > 64 {
		return 64
	}
	return horizon
}

func PromptDigest(prompt string) string {
	if strings.TrimSpace(prompt) == "" {
		return ""
	}
	return FullHash(prompt)
}

func FullHash(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func maxInt(left int, right int) int {
	if left > right {
		return left
	}
	return right
}

func mapFromAny(value any) map[string]any {
	if typed, ok := value.(map[string]any); ok {
		return typed
	}
	return map[string]any{}
}

func sliceFromAny(value any) []any {
	if typed, ok := value.([]any); ok {
		return typed
	}
	return nil
}

func stringValue(value any) string {
	switch typed := value.(type) {
	case string:
		return typed
	case fmt.Stringer:
		return typed.String()
	default:
		return ""
	}
}
