package llamacpp

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"unicode"

	"github.com/Ryvion/node-agent/internal/v7/runtimeinventory"
)

const maxConfigTextLen = 1024

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

func ConfigFromEnv() LlamaCppSidecarConfig {
	return ConfigFromEnvWith(ConfigSource{})
}

func ConfigFromEnvWith(source ConfigSource) LlamaCppSidecarConfig {
	source = normalizeConfigSource(source)
	cfg := LlamaCppSidecarConfig{
		Enabled:     envBool(source.Getenv(EnvEnabled)),
		ServerPath:  cleanConfigText(source.Getenv(EnvServer), maxConfigTextLen),
		ModelPath:   cleanConfigText(source.Getenv(EnvModel), maxConfigTextLen),
		Host:        normalizeHost(source.Getenv(EnvHost)),
		Port:        envInt(source.Getenv(EnvPort), DefaultPort),
		ContextSize: envInt(source.Getenv(EnvCtxSize), DefaultContextSize),
		Threads:     envInt(source.Getenv(EnvThreads), 0),
		GPULayers:   envInt(source.Getenv(EnvGPULayers), 0),
		ExtraArgs:   sanitizeExtraArgs(source.Getenv(EnvExtraArgs)),
	}
	if cfg.ServerPath == "" {
		cfg.ServerPath = discoverServerPath(source)
	}
	if cfg.ModelPath == "" {
		cfg.ModelPath = discoverModelPath(source)
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
