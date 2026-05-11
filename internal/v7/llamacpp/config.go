package llamacpp

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/Ryvion/node-agent/internal/v7/runtimeinventory"
)

const maxConfigTextLen = 1024

const (
	EnvKeepWarm               = "RYV_LLAMA_CPP_KEEP_WARM"
	EnvHealthIntervalSecs     = "RYV_LLAMA_CPP_HEALTH_INTERVAL_SECONDS"
	EnvRestartBackoffSecs     = "RYV_LLAMA_CPP_RESTART_BACKOFF_SECONDS"
	EnvMaxRestartsPerHour     = "RYV_LLAMA_CPP_MAX_RESTARTS_PER_HOUR"
	DefaultHealthInterval     = 30 * time.Second
	DefaultRestartBackoff     = 10 * time.Second
	DefaultMaxRestartsPerHour = 10
)

type ConfigSource struct {
	Getenv               func(string) string
	LookPath             func(string) (string, error)
	Stat                 func(string) (os.FileInfo, error)
	ReadDirNames         func(string, int) ([]string, error)
	UserHomeDir          func() (string, error)
	GOOS                 string
	ConfiguredBinaryDirs []string
	ConfiguredModelDirs  []string
	RuntimeInventory     *runtimeinventory.Inventory
}

type ResidencyKeeperConfig struct {
	Enabled            bool
	HealthInterval     time.Duration
	RestartBackoff     time.Duration
	MaxRestartsPerHour int
}

func ConfigFromEnv() LlamaCppSidecarConfig {
	return ConfigFromEnvWith(ConfigSource{})
}

func ResidencyKeeperConfigFromEnv() ResidencyKeeperConfig {
	return ResidencyKeeperConfigFromEnvWith(ConfigSource{})
}

func ResidencyKeeperConfigFromEnvWith(source ConfigSource) ResidencyKeeperConfig {
	source = normalizeConfigSource(source)
	return normalizeResidencyKeeperConfig(ResidencyKeeperConfig{
		Enabled:            envBool(source.Getenv(EnvKeepWarm)),
		HealthInterval:     envDurationSeconds(source.Getenv(EnvHealthIntervalSecs), DefaultHealthInterval),
		RestartBackoff:     envDurationSeconds(source.Getenv(EnvRestartBackoffSecs), DefaultRestartBackoff),
		MaxRestartsPerHour: envInt(source.Getenv(EnvMaxRestartsPerHour), DefaultMaxRestartsPerHour),
	})
}

func ConfigFromEnvWith(source ConfigSource) LlamaCppSidecarConfig {
	source = normalizeConfigSource(source)
	cfg := LlamaCppSidecarConfig{
		Enabled:        envBool(source.Getenv(EnvEnabled)),
		ServerPath:     cleanConfigText(source.Getenv(EnvServer), maxConfigTextLen),
		ModelPath:      cleanConfigText(source.Getenv(EnvModel), maxConfigTextLen),
		Host:           normalizeHost(source.Getenv(EnvHost)),
		Port:           envInt(source.Getenv(EnvPort), DefaultPort),
		ContextSize:    envInt(source.Getenv(EnvCtxSize), DefaultContextSize),
		Threads:        envInt(source.Getenv(EnvThreads), 0),
		GPULayers:      envInt(source.Getenv(EnvGPULayers), DefaultGPULayers),
		ExtraArgs:      sanitizeExtraArgs(source.Getenv(EnvExtraArgs)),
		DraftModelPath: cleanConfigText(source.Getenv(EnvDraftModel), maxConfigTextLen),
		DraftMaxTokens: envInt(source.Getenv(EnvDraftMaxTokens), 0),
		DraftMinTokens: envInt(source.Getenv(EnvDraftMinTokens), 0),
		DraftPMin:      envFloat(source.Getenv(EnvDraftPMin), 0),
		DraftGPULayers: envInt(source.Getenv(EnvDraftGPULayers), 0),
	}
	if cfg.ServerPath == "" {
		cfg.ServerPath = discoverServerPath(source)
	}
	if cfg.ModelPath == "" {
		cfg.ModelPath = discoverModelPath(source)
	}
	if cfg.DraftModelPath == "" && envBool(source.Getenv(EnvDraftAuto)) {
		cfg.DraftModelPath = discoverDraftModelPath(source, cfg.ModelPath)
	}
	return normalizeConfig(cfg)
}

func normalizeConfig(cfg LlamaCppSidecarConfig) LlamaCppSidecarConfig {
	cfg.ServerPath = cleanConfigText(cfg.ServerPath, maxConfigTextLen)
	cfg.ModelPath = cleanConfigText(cfg.ModelPath, maxConfigTextLen)
	cfg.Host = normalizeHost(cfg.Host)
	if cfg.Port <= 0 || cfg.Port > 65535 {
		cfg.Port = DefaultPort
	}
	if cfg.ContextSize <= 0 {
		cfg.ContextSize = DefaultContextSize
	}
	if cfg.Threads < 0 {
		cfg.Threads = 0
	}
	if cfg.GPULayers < 0 {
		cfg.GPULayers = 0
	}
	cfg.ExtraArgs = sanitizeExtraArgs(strings.Join(cfg.ExtraArgs, " "))

	// V8 speculative decoding (Level 0): clamp draft parameters.
	cfg.DraftModelPath = cleanConfigText(cfg.DraftModelPath, maxConfigTextLen)
	if cfg.DraftMaxTokens < 0 || cfg.DraftMaxTokens > 256 {
		cfg.DraftMaxTokens = 0
	}
	if cfg.DraftMinTokens < 0 || cfg.DraftMinTokens > 256 {
		cfg.DraftMinTokens = 0
	}
	if cfg.DraftMinTokens > cfg.DraftMaxTokens && cfg.DraftMaxTokens > 0 {
		cfg.DraftMinTokens = cfg.DraftMaxTokens
	}
	if cfg.DraftPMin < 0 || cfg.DraftPMin >= 1 {
		cfg.DraftPMin = 0
	}
	if cfg.DraftGPULayers < 0 {
		cfg.DraftGPULayers = 0
	}
	// If draft model is set without explicit max tokens, default to 16
	// (the llama.cpp default and a safe consumer-hardware setting).
	if cfg.DraftModelPath != "" && cfg.DraftMaxTokens == 0 {
		cfg.DraftMaxTokens = DefaultDraftMaxTokens
	}
	return cfg
}

func normalizeResidencyKeeperConfig(cfg ResidencyKeeperConfig) ResidencyKeeperConfig {
	if cfg.HealthInterval <= 0 {
		cfg.HealthInterval = DefaultHealthInterval
	}
	if cfg.RestartBackoff < 0 {
		cfg.RestartBackoff = DefaultRestartBackoff
	}
	if cfg.MaxRestartsPerHour <= 0 {
		cfg.MaxRestartsPerHour = DefaultMaxRestartsPerHour
	}
	return cfg
}

func normalizeConfigSource(source ConfigSource) ConfigSource {
	if source.Getenv == nil {
		source.Getenv = os.Getenv
	}
	if source.LookPath == nil {
		source.LookPath = exec.LookPath
	}
	if source.Stat == nil {
		source.Stat = os.Stat
	}
	if source.ReadDirNames == nil {
		source.ReadDirNames = nil
	}
	if source.UserHomeDir == nil {
		source.UserHomeDir = os.UserHomeDir
	}
	if strings.TrimSpace(source.GOOS) == "" {
		source.GOOS = runtime.GOOS
	}
	return source
}

func discoverServerPath(source ConfigSource) string {
	if source.RuntimeInventory != nil {
		for _, candidate := range source.RuntimeInventory.BackendCandidates {
			if candidate.Backend == runtimeinventory.BackendCandidateLlamaCPP && strings.TrimSpace(candidate.ServerBinaryPath) != "" {
				return cleanConfigText(candidate.ServerBinaryPath, maxConfigTextLen)
			}
		}
	}

	knownDirDetector := source.candidateDetector()
	knownDirDetector.LookPath = func(string) (string, error) {
		return "", errors.New("path lookup disabled for known-dir detection")
	}
	if candidate := runtimeinventory.DetectLlamaCPPBackendCandidate(knownDirDetector); strings.TrimSpace(candidate.ServerBinaryPath) != "" {
		return cleanConfigText(candidate.ServerBinaryPath, maxConfigTextLen)
	}

	if source.LookPath != nil {
		for _, name := range executableNameVariants(source.GOOS, "llama-server") {
			if path, err := source.LookPath(name); err == nil && strings.TrimSpace(path) != "" {
				return cleanConfigText(path, maxConfigTextLen)
			}
		}
	}
	return ""
}

func discoverModelPath(source ConfigSource) string {
	if source.RuntimeInventory != nil && len(source.RuntimeInventory.GGUFModels) > 0 {
		return cleanConfigText(source.RuntimeInventory.GGUFModels[0].Path, maxConfigTextLen)
	}
	models := runtimeinventory.DetectGGUFModels(source.candidateDetector())
	if len(models) == 0 {
		return ""
	}
	return cleanConfigText(models[0].Path, maxConfigTextLen)
}

func (source ConfigSource) candidateDetector() runtimeinventory.CandidateBackendDetector {
	return runtimeinventory.CandidateBackendDetector{
		LookPath:             source.LookPath,
		Stat:                 source.Stat,
		ReadDirNames:         source.ReadDirNames,
		Getenv:               source.Getenv,
		UserHomeDir:          source.UserHomeDir,
		GOOS:                 source.GOOS,
		ConfiguredBinaryDirs: source.ConfiguredBinaryDirs,
		ConfiguredModelDirs:  source.ConfiguredModelDirs,
	}
}

func envBool(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "yes", "on", "enabled":
		return true
	default:
		return false
	}
}

func envInt(value string, fallback int) int {
	value = strings.TrimSpace(value)
	if value == "" {
		return fallback
	}
	n, err := strconv.Atoi(value)
	if err != nil {
		return fallback
	}
	return n
}

func envFloat(value string, fallback float64) float64 {
	value = strings.TrimSpace(value)
	if value == "" {
		return fallback
	}
	f, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return fallback
	}
	return f
}

// discoverDraftModelPath looks for a known-pairing draft model on disk.
// On Ryvion native nodes, tinyllama (1.1B) is paired with llama-3.2-3B as
// the canonical drafter+target pair. The discovery walks the model
// inventory and prefers a same-family, smaller-quantization GGUF.
func discoverDraftModelPath(source ConfigSource, targetModelPath string) string {
	target := strings.ToLower(filepath.Base(targetModelPath))
	if target == "" || !strings.Contains(target, "llama") {
		return ""
	}
	candidates := source.candidateDetector()
	models := runtimeinventory.DetectGGUFModels(candidates)
	for _, m := range models {
		name := strings.ToLower(filepath.Base(m.Path))
		if name == target {
			continue
		}
		// Tokenizer-compatible draft heuristic: same family, smaller size.
		if strings.Contains(name, "tinyllama") || strings.Contains(name, "llama-3.2-1b") {
			return cleanConfigText(m.Path, maxConfigTextLen)
		}
	}
	return ""
}

func envDurationSeconds(value string, fallback time.Duration) time.Duration {
	n := envInt(value, int(fallback/time.Second))
	if n <= 0 {
		return fallback
	}
	return time.Duration(n) * time.Second
}

func normalizeHost(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", "127.0.0.1", "localhost", "::1":
		if strings.TrimSpace(value) == "" {
			return DefaultHost
		}
		return strings.TrimSpace(value)
	default:
		return DefaultHost
	}
}

func executableNameVariants(goos, name string) []string {
	name = strings.TrimSpace(name)
	if name == "" {
		return nil
	}
	variants := []string{name}
	if strings.EqualFold(strings.TrimSpace(goos), "windows") && !strings.HasSuffix(strings.ToLower(name), ".exe") {
		variants = append(variants, name+".exe")
	}
	return variants
}

func sanitizeExtraArgs(raw string) []string {
	fields := strings.Fields(raw)
	if len(fields) == 0 {
		return nil
	}
	noValue := map[string]bool{
		"--cont-batching":     true,
		"--flash-attn":        true,
		"--mlock":             true,
		"--no-context-shift":  true,
		"--no-mmap":           true,
		"--no-webui":          true,
		"--override-kv-cache": true,
	}
	withValue := map[string]func(string) bool{
		"--batch-size":   positiveIntToken,
		"--cache-type-k": cacheTypeToken,
		"--cache-type-v": cacheTypeToken,
		"--parallel":     positiveIntToken,
		"--ubatch-size":  positiveIntToken,
	}

	out := make([]string, 0, len(fields))
	for i := 0; i < len(fields); i++ {
		token := cleanArgToken(fields[i])
		if token == "" {
			continue
		}
		if noValue[token] {
			out = append(out, token)
			continue
		}
		validator := withValue[token]
		if validator == nil || i+1 >= len(fields) {
			continue
		}
		value := cleanArgToken(fields[i+1])
		if validator(value) {
			out = append(out, token, value)
			i++
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func positiveIntToken(value string) bool {
	n, err := strconv.Atoi(value)
	return err == nil && n > 0 && n <= 1_000_000
}

func cacheTypeToken(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "f16", "q8_0", "q4_0":
		return true
	default:
		return false
	}
}

func cleanArgToken(value string) string {
	value = strings.TrimSpace(value)
	if value == "" || len(value) > 80 {
		return ""
	}
	for _, r := range value {
		if r < 33 || r > 126 || r == '\'' || r == '"' || r == '`' || r == '$' || r == '\\' || r == ';' || r == '&' || r == '|' {
			return ""
		}
	}
	return value
}

func cleanConfigText(value string, maxRunes int) string {
	value = strings.TrimSpace(value)
	if value == "" || maxRunes <= 0 {
		return ""
	}
	if filepath.Clean(value) == "." && (strings.Contains(value, "/") || strings.Contains(value, `\`)) {
		return ""
	}
	var b strings.Builder
	b.Grow(len(value))
	written := 0
	for _, r := range value {
		if unicode.IsControl(r) {
			continue
		}
		if written >= maxRunes {
			break
		}
		b.WriteRune(r)
		written++
	}
	return strings.TrimSpace(b.String())
}
