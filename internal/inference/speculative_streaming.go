package inference

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

const (
	envStreamingDraftAuto      = "RYV_LLAMA_CPP_AUTO_DRAFT"
	envStreamingDraftModel     = "RYV_LLAMA_CPP_DRAFT_MODEL_PATH"
	envStreamingDraftMaxTokens = "RYV_LLAMA_CPP_DRAFT_MAX_TOKENS"
	envStreamingDraftMinTokens = "RYV_LLAMA_CPP_DRAFT_MIN_TOKENS"
	envStreamingDraftPMin      = "RYV_LLAMA_CPP_DRAFT_P_MIN"
	envStreamingDraftGPULayers = "RYV_LLAMA_CPP_DRAFT_GPU_LAYERS"
	envStreamingNativeMTP      = "RYV_LLAMA_CPP_NATIVE_MTP"

	speculativeMethodBackendLocalDraft = "backend_local_draft_model"
	speculativeMethodNativeMTP         = "native_mtp"

	defaultStreamingDraftMaxTokens     = 16
	defaultStreamingNativeMTPMaxTokens = 3
)

var streamingMTPModelPattern = regexp.MustCompile(`(?i)(?:^|[._\-\s])mtp(?:[._\-\s]|$)`)

type streamingSpeculativeLaunch struct {
	Method             string
	Args               []string
	DraftModelPath     string
	DraftModelFilename string
	DraftMaxTokens     int
	DraftMinTokens     int
}

type streamingSpeculativeTimings struct {
	NDrafted       int64 `json:"n_drafted,omitempty"`
	NAccepted      int64 `json:"n_accepted,omitempty"`
	DraftN         int64 `json:"draft_n,omitempty"`
	DraftNAccepted int64 `json:"draft_n_accepted,omitempty"`
}

func streamingSpeculativeCountsFromPayload(payload []byte) (int64, int64) {
	var chunk struct {
		Timings *streamingSpeculativeTimings `json:"timings,omitempty"`
	}
	if len(payload) == 0 || json.Unmarshal(payload, &chunk) != nil || chunk.Timings == nil {
		return 0, 0
	}
	drafted := chunk.Timings.NDrafted
	if drafted == 0 {
		drafted = chunk.Timings.DraftN
	}
	accepted := chunk.Timings.NAccepted
	if accepted == 0 {
		accepted = chunk.Timings.DraftNAccepted
	}
	if drafted < 0 {
		drafted = 0
	}
	if accepted < 0 {
		accepted = 0
	}
	if accepted > drafted && drafted > 0 {
		accepted = drafted
	}
	return drafted, accepted
}

func streamingSpeculativeLaunchForModel(modelPath string, modelDir string, getenv func(string) string) streamingSpeculativeLaunch {
	if getenv == nil {
		getenv = os.Getenv
	}
	modelPath = strings.TrimSpace(modelPath)
	nativeMTP, nativeMTPAuto := parseStreamingNativeMTPSetting(getenv(envStreamingNativeMTP))
	draftMax := envIntDefault(getenv(envStreamingDraftMaxTokens), 0)
	draftMin := envIntDefault(getenv(envStreamingDraftMinTokens), 0)
	if draftMin < 0 {
		draftMin = 0
	}
	if draftMax < 0 {
		draftMax = 0
	}
	if draftMax > 0 && draftMin > draftMax {
		draftMin = draftMax
	}
	if (nativeMTP || nativeMTPAuto) && streamingModelSupportsNativeMTP(modelPath) {
		if draftMax == 0 {
			draftMax = defaultStreamingNativeMTPMaxTokens
		}
		args := []string{"--spec-type", "draft-mtp", "--spec-draft-n-max", strconv.Itoa(draftMax)}
		if draftMin > 0 {
			args = append(args, "--spec-draft-n-min", strconv.Itoa(draftMin))
		}
		return streamingSpeculativeLaunch{
			Method:         speculativeMethodNativeMTP,
			Args:           args,
			DraftMaxTokens: draftMax,
			DraftMinTokens: draftMin,
		}
	}

	draftPath := strings.TrimSpace(getenv(envStreamingDraftModel))
	if draftPath == "" && envBoolDefault(getenv(envStreamingDraftAuto), false) {
		draftPath = discoverStreamingDraftModel(modelDir, modelPath)
	}
	if !fileReadable(draftPath) {
		return streamingSpeculativeLaunch{}
	}
	if draftMax == 0 {
		draftMax = defaultStreamingDraftMaxTokens
	}
	args := []string{"--model-draft", draftPath, "--spec-draft-n-max", strconv.Itoa(draftMax)}
	if draftMin > 0 {
		args = append(args, "--spec-draft-n-min", strconv.Itoa(draftMin))
	}
	if pMin := strings.TrimSpace(getenv(envStreamingDraftPMin)); pMin != "" {
		args = append(args, "--draft-p-min", pMin)
	}
	if draftGPULayers := envIntDefault(getenv(envStreamingDraftGPULayers), 0); draftGPULayers > 0 {
		args = append(args, "--n-gpu-layers-draft", strconv.Itoa(draftGPULayers))
	}
	return streamingSpeculativeLaunch{
		Method:             speculativeMethodBackendLocalDraft,
		Args:               args,
		DraftModelPath:     draftPath,
		DraftModelFilename: filepath.Base(draftPath),
		DraftMaxTokens:     draftMax,
		DraftMinTokens:     draftMin,
	}
}

func parseStreamingNativeMTPSetting(raw string) (enabled bool, auto bool) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "auto", "default", "0":
		return false, true
	case "1", "true", "t", "yes", "y", "on", "enabled", "enable":
		return true, true
	case "false", "f", "no", "n", "off", "disabled", "disable":
		return false, false
	default:
		return false, true
	}
}

func streamingModelSupportsNativeMTP(modelPath string) bool {
	modelPath = strings.TrimSpace(modelPath)
	if modelPath == "" {
		return false
	}
	return streamingMTPModelPattern.MatchString(filepath.Base(modelPath))
}

func discoverStreamingDraftModel(modelDir string, targetModelPath string) string {
	if strings.TrimSpace(modelDir) == "" || !strings.Contains(strings.ToLower(filepath.Base(targetModelPath)), "llama") {
		return ""
	}
	names := []string{
		"tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
		"tinyllama-1.1b.Q4_K_M.gguf",
		"tinyllama.Q4_K_M.gguf",
		"llama-3.2-1b-instruct-q4_k_m.gguf",
	}
	for _, name := range names {
		candidate := filepath.Join(modelDir, name)
		if fileReadable(candidate) {
			return candidate
		}
	}
	return ""
}

func fileReadable(path string) bool {
	path = strings.TrimSpace(path)
	if path == "" {
		return false
	}
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func envBoolDefault(raw string, fallback bool) bool {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "1", "true", "t", "yes", "y", "on", "enabled", "enable":
		return true
	case "0", "false", "f", "no", "n", "off", "disabled", "disable":
		return false
	default:
		return fallback
	}
}

func envIntDefault(raw string, fallback int) int {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	return value
}

func (m *Manager) setStreamingSpeculativeLaunch(state streamingSpeculativeLaunch) {
	if m == nil {
		return
	}
	m.mu.Lock()
	m.speculative = state
	m.mu.Unlock()
}

func (m *Manager) streamingSpeculativeLaunch() streamingSpeculativeLaunch {
	if m == nil {
		return streamingSpeculativeLaunch{}
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.speculative
}

func (s streamingSpeculativeLaunch) receiptMetadata(metrics StreamingMetrics) map[string]any {
	if strings.TrimSpace(s.Method) == "" {
		return nil
	}
	out := map[string]any{
		"enabled": true,
		"method":  s.Method,
	}
	if s.DraftModelFilename != "" {
		out["drafter_filename"] = s.DraftModelFilename
	}
	if s.DraftMaxTokens > 0 {
		out["draft_max_tokens"] = s.DraftMaxTokens
	}
	if s.DraftMinTokens > 0 {
		out["draft_min_tokens"] = s.DraftMinTokens
	}
	if metrics.SpeculativeTokensDrafted > 0 {
		out["tokens_drafted"] = metrics.SpeculativeTokensDrafted
	}
	if metrics.SpeculativeTokensAccepted > 0 {
		out["tokens_accepted"] = metrics.SpeculativeTokensAccepted
	}
	if metrics.SpeculativeTokensDrafted > 0 && metrics.SpeculativeTokensAccepted >= 0 {
		out["acceptance_rate"] = roundFloat(float64(metrics.SpeculativeTokensAccepted) / float64(metrics.SpeculativeTokensDrafted))
	}
	if metrics.CompletionTokens > 0 && metrics.SpeculativeTokensAccepted > 0 {
		targetSteps := metrics.CompletionTokens - metrics.SpeculativeTokensAccepted
		if targetSteps < 1 {
			targetSteps = 1
		}
		out["estimated_speedup_ratio"] = roundFloat(float64(metrics.CompletionTokens) / float64(targetSteps))
	}
	return out
}

func applyStreamingSpeculativeMetadata(meta map[string]any, launch streamingSpeculativeLaunch, metrics StreamingMetrics) {
	if meta == nil {
		return
	}
	block := launch.receiptMetadata(metrics)
	if block == nil {
		return
	}
	meta["speculative"] = block
	meta["speculative_enabled"] = true
	if method, _ := block["method"].(string); method != "" {
		meta["speculative_method"] = method
	}
	if speedup, ok := block["estimated_speedup_ratio"]; ok {
		meta["speculative_speedup"] = speedup
	}
}

func roundFloat(value float64) float64 {
	if value <= 0 {
		return 0
	}
	return float64(int(value*1000+0.5)) / 1000
}
