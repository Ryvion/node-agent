package sglang

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"unicode"

	v7hardware "github.com/Ryvion/ryvion-node/internal/v7/hardware"
	"github.com/Ryvion/ryvion-node/internal/v7/runtimeinventory"
)

const maxConfigTextLen = 1024

func ConfigFromEnv() SGLangSidecarConfig {
	return ConfigFromEnvWith(ConfigSource{})
}

func ConfigFromEnvWith(source ConfigSource) SGLangSidecarConfig {
	source = normalizeConfigSource(source)
	explicitServerPath := strings.TrimSpace(source.Getenv(EnvServer)) != ""
	cfg := SGLangSidecarConfig{
		Enabled:              envBool(source.Getenv(EnvEnabled)),
		ServerPath:           cleanConfigText(source.Getenv(EnvServer), maxConfigTextLen),
		ServerPathExplicit:   explicitServerPath,
		ModelPath:            cleanConfigText(source.Getenv(EnvModel), maxConfigTextLen),
		ModelID:              cleanConfigText(source.Getenv(EnvModelID), maxConfigTextLen),
		Host:                 normalizeHost(source.Getenv(EnvHost)),
		Port:                 envInt(source.Getenv(EnvPort), DefaultPort),
		ContextLength:        envInt(source.Getenv(EnvContextLength), DefaultContextLength),
		TPSize:               envInt(source.Getenv(EnvTPSize), DefaultTPSize),
		MemFractionStatic:    envFloat(source.Getenv(EnvMemFractionStatic), 0),
		ExtraArgs:            sanitizeExtraArgs(source.Getenv(EnvExtraArgs)),
		AccelerationHints:    configAccelerationHints(source),
		LaunchProfile:        "python_module",
		ModelPathMustBeLocal: true,
	}
	if cfg.ServerPath == "" {
		cfg.ServerPath = discoverServerPath(source)
	}
	return normalizeConfig(cfg)
}

func normalizeConfig(cfg SGLangSidecarConfig) SGLangSidecarConfig {
	cfg.ServerPath = cleanConfigText(cfg.ServerPath, maxConfigTextLen)
	cfg.ModelPath = cleanConfigText(cfg.ModelPath, maxConfigTextLen)
	cfg.ModelID = cleanConfigText(cfg.ModelID, maxConfigTextLen)
	cfg.Host = normalizeHost(cfg.Host)
	if cfg.Port <= 0 || cfg.Port > 65535 {
		cfg.Port = DefaultPort
	}
	if cfg.ContextLength <= 0 {
		cfg.ContextLength = DefaultContextLength
	}
	if cfg.TPSize <= 0 {
		cfg.TPSize = DefaultTPSize
	}
	if cfg.MemFractionStatic < 0 || cfg.MemFractionStatic >= 1 {
		cfg.MemFractionStatic = 0
	}
	cfg.ExtraArgs = sanitizeExtraArgs(strings.Join(cfg.ExtraArgs, " "))
	cfg.AccelerationHints = normalizeAcceleration(cfg.AccelerationHints)
	cfg.LaunchProfile = cleanRuntimeCompactText(strings.ToLower(strings.TrimSpace(cfg.LaunchProfile)), 64)
	if cfg.LaunchProfile == "" {
		cfg.LaunchProfile = launcherKind(cfg.ServerPath)
	}
	if cfg.ModelID == "" {
		cfg.ModelID = modelIDFromPath(cfg.ModelPath)
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
	if strings.TrimSpace(source.GOOS) == "" {
		source.GOOS = runtime.GOOS
	}
	return source
}

func discoverServerPath(source ConfigSource) string {
	if source.RuntimeInventory != nil {
		for _, candidate := range source.RuntimeInventory.BackendCandidates {
			if candidate.Backend == BackendName && strings.TrimSpace(candidate.BinaryPath) != "" {
				return cleanConfigText(candidate.BinaryPath, maxConfigTextLen)
			}
		}
	}
	if source.LookPath != nil {
		for _, name := range []string{"sglang", "python3", "python"} {
			for _, variant := range executableNameVariants(source.GOOS, name) {
				if path, err := source.LookPath(variant); err == nil && strings.TrimSpace(path) != "" {
					return cleanConfigText(path, maxConfigTextLen)
				}
			}
		}
	}
	return ""
}

func configAccelerationHints(source ConfigSource) []string {
	source = normalizeConfigSource(source)
	hints := []string{}
	if source.HardwareCapacity != nil {
		hardware := v7hardware.NormalizeInventory(*source.HardwareCapacity)
		if len(hardware.AccelerationHints) > 0 {
			hints = append(hints, hardware.AccelerationHints...)
		}
		if hardware.CUDAAvailable {
			hints = append(hints, "cuda")
		}
		if hardware.MetalAvailable {
			hints = append(hints, "metal")
		}
		if hardware.VulkanAvailable {
			hints = append(hints, "vulkan")
		}
	}
	if len(hints) == 0 {
		hints = append(hints, "cpu")
	}
	return normalizeAcceleration(hints)
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
	n, err := strconv.ParseFloat(value, 64)
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
		"--disable-cuda-graph":   true,
		"--disable-radix-cache":  true,
		"--enable-p2p-check":     true,
		"--enable-torch-compile": true,
	}
	withValue := map[string]func(string) bool{
		"--chunked-prefill-size": positiveIntToken,
		"--kv-cache-dtype":       simpleToken,
		"--attention-backend":    simpleToken,
		"--dtype":                simpleToken,
		"--quantization":         simpleToken,
		"--schedule-policy":      simpleToken,
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

func simpleToken(value string) bool {
	value = strings.TrimSpace(value)
	if value == "" || len(value) > 80 {
		return false
	}
	for _, r := range value {
		if !(r == '_' || r == '-' || r == '.' || r == '/' || r == ':' || r == '+' || r == '=' || r == ',' || r == '@' || r == '%' ||
			(r >= '0' && r <= '9') || (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z')) {
			return false
		}
	}
	return true
}

func cleanArgToken(value string) string {
	value = strings.TrimSpace(value)
	if value == "" || len(value) > 120 {
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

func (source ConfigSource) candidateDetector() runtimeinventory.CandidateBackendDetector {
	return runtimeinventory.CandidateBackendDetector{
		LookPath: source.LookPath,
		Stat:     source.Stat,
		Getenv:   source.Getenv,
		GOOS:     source.GOOS,
	}
}
