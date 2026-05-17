package backendprobe

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
	"unicode"

	"github.com/Ryvion/ryvion-node/internal/v7/runtimeinventory"
)

const (
	defaultVersionTimeout    = 2 * time.Second
	defaultModelProbeTimeout = 5 * time.Second
	unknownVersion           = "unknown"
	maxProbePathLen          = 512
	maxProbeVersionLen       = 128
	maxProbeReasonLen        = 256
)

func ProbeAll(detector Detector) Probes {
	return NormalizeProbes(Probes{
		LlamaCPP: ProbeLlamaCPP(detector),
	})
}

func ProbeLlamaCPP(detector Detector) LlamaCPPProbe {
	detector = normalizeDetector(detector)
	candidateDetector := detector.candidateDetector()
	candidate := runtimeinventory.DetectLlamaCPPBackendCandidate(candidateDetector)
	ggufModels := runtimeinventory.DetectGGUFModels(candidateDetector)

	probeModelPath := cleanProbeText(detector.Getenv(EnvLlamaCPPProbeModel), maxProbePathLen)
	probeModelConfigured := probeModelPath != ""
	probeModelReadable := false
	if probeModelConfigured {
		probeModelReadable = probeModelIsReadableGGUF(detector, probeModelPath)
	}
	if probeModelReadable && candidate.Detected {
		runConfiguredModelProbe(detector, candidate, probeModelPath)
	}

	return LlamaCPPProbe{
		Available:                      candidate.Detected,
		BinaryPath:                     candidate.BinaryPath,
		ServerBinaryPath:               candidate.ServerBinaryPath,
		BenchBinaryPath:                candidate.BenchBinaryPath,
		Version:                        normalizeProbeVersion(candidate.Version),
		GGUFModelsDetected:             len(ggufModels) > 0 || probeModelReadable,
		ProbeModelConfigured:           probeModelConfigured,
		SupportsTextGeneration:         candidate.SupportsTextGeneration,
		SupportsStreaming:              candidate.SupportsStreaming,
		SupportsOpenAICompatibleServer: candidate.SupportsOpenAICompatibleServer,
		SupportsKVAccess:               false,
		SupportsTensorHooks:            false,
		CandidateForFastTextRuntime:    candidate.CandidateForFastTextRuntime,
		CandidateForRealTensorAccess:   candidate.CandidateForRealTensorAccess,
		Reason:                         llamaCPPProbeReason(candidate, probeModelConfigured, probeModelReadable),
	}
}

func (detector Detector) candidateDetector() runtimeinventory.CandidateBackendDetector {
	return runtimeinventory.CandidateBackendDetector{
		LookPath:              detector.LookPath,
		Stat:                  detector.Stat,
		ReadDirNames:          detector.ReadDirNames,
		VersionCommand:        detector.VersionCommand,
		Getenv:                detector.Getenv,
		UserHomeDir:           detector.UserHomeDir,
		GOOS:                  detector.GOOS,
		ConfiguredBinaryDirs:  detector.ConfiguredBinaryDirs,
		ConfiguredModelDirs:   detector.ConfiguredModelDirs,
		VersionTimeout:        detector.VersionTimeout,
		PythonModuleAvailable: nil,
	}
}

func normalizeDetector(detector Detector) Detector {
	if detector.Stat == nil {
		detector.Stat = os.Stat
	}
	if detector.Getenv == nil {
		detector.Getenv = os.Getenv
	}
	if detector.UserHomeDir == nil {
		detector.UserHomeDir = os.UserHomeDir
	}
	if strings.TrimSpace(detector.GOOS) == "" {
		detector.GOOS = runtime.GOOS
	}
	if detector.VersionTimeout <= 0 {
		detector.VersionTimeout = defaultVersionTimeout
	}
	if detector.ModelProbeTimeout <= 0 {
		detector.ModelProbeTimeout = defaultModelProbeTimeout
	}
	if detector.ModelProbeCommand == nil {
		detector.ModelProbeCommand = llamaCPPMetadataProbe
	}
	return detector
}

func probeModelIsReadableGGUF(detector Detector, path string) bool {
	if !strings.EqualFold(filepath.Ext(path), ".gguf") {
		return false
	}
	if detector.Stat == nil {
		return false
	}
	info, err := detector.Stat(path)
	return err == nil && !info.IsDir()
}

func runConfiguredModelProbe(detector Detector, candidate runtimeinventory.BackendCandidate, modelPath string) {
	binary := llamaCLIBinaryPath(candidate)
	if binary == "" || detector.ModelProbeCommand == nil {
		return
	}
	_ = detector.ModelProbeCommand(binary, modelPath, detector.ModelProbeTimeout)
}

func llamaCLIBinaryPath(candidate runtimeinventory.BackendCandidate) string {
	binary := strings.TrimSpace(candidate.BinaryPath)
	if binary == "" || binary == strings.TrimSpace(candidate.ServerBinaryPath) || binary == strings.TrimSpace(candidate.BenchBinaryPath) {
		return ""
	}
	return binary
}

func llamaCPPMetadataProbe(binary, modelPath string, timeout time.Duration) error {
	binary = strings.TrimSpace(binary)
	modelPath = strings.TrimSpace(modelPath)
	if binary == "" || modelPath == "" {
		return os.ErrNotExist
	}
	if timeout <= 0 {
		timeout = defaultModelProbeTimeout
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, binary, "--model", modelPath, "--n-predict", "0", "--prompt", "", "--no-display-prompt")
	_, err := cmd.CombinedOutput()
	if ctx.Err() != nil {
		return ctx.Err()
	}
	return err
}

func llamaCPPProbeReason(candidate runtimeinventory.BackendCandidate, probeModelConfigured bool, probeModelReadable bool) string {
	if !candidate.Detected {
		return "llama.cpp binary not detected"
	}
	if probeModelConfigured && !probeModelReadable {
		return "configured RYV_LLAMA_CPP_PROBE_MODEL is not a readable GGUF file"
	}
	if !candidate.SupportsTextGeneration {
		return "llama.cpp benchmark binary detected; text runtime binary not detected"
	}
	return "llama.cpp detected; real KV/tensor hooks require adapter implementation"
}

func NormalizeProbes(probes Probes) Probes {
	return Probes{
		LlamaCPP: normalizeLlamaCPPProbe(probes.LlamaCPP),
	}
}

func normalizeLlamaCPPProbe(probe LlamaCPPProbe) LlamaCPPProbe {
	probe.BinaryPath = cleanProbeText(probe.BinaryPath, maxProbePathLen)
	probe.ServerBinaryPath = cleanProbeText(probe.ServerBinaryPath, maxProbePathLen)
	probe.BenchBinaryPath = cleanProbeText(probe.BenchBinaryPath, maxProbePathLen)
	probe.Version = normalizeProbeVersion(probe.Version)
	probe.Reason = cleanProbeText(probe.Reason, maxProbeReasonLen)
	if probe.Reason == "" {
		if probe.Available {
			probe.Reason = "llama.cpp detected; real KV/tensor hooks require adapter implementation"
		} else {
			probe.Reason = "llama.cpp binary not detected"
		}
	}
	probe.SupportsKVAccess = false
	probe.SupportsTensorHooks = false
	if !probe.Available {
		probe.SupportsTextGeneration = false
		probe.SupportsStreaming = false
		probe.SupportsOpenAICompatibleServer = false
		probe.CandidateForFastTextRuntime = false
		probe.CandidateForRealTensorAccess = false
	}
	return probe
}

func normalizeProbeVersion(value string) string {
	version := firstVersionLookingLine(value)
	if version == "" {
		return unknownVersion
	}
	return version
}

func firstVersionLookingLine(value string) string {
	for _, line := range strings.FieldsFunc(value, func(r rune) bool { return r == '\n' || r == '\r' }) {
		line = cleanProbeText(line, maxProbeVersionLen)
		if line == "" {
			continue
		}
		if version := trimToVersionAnchor(line); version != "" {
			return version
		}
	}
	line := cleanProbeText(value, maxProbeVersionLen)
	return trimToVersionAnchor(line)
}

func trimToVersionAnchor(line string) string {
	lower := strings.ToLower(line)
	anchor, genericAnchor := versionAnchorIndex(lower)
	if anchor < 0 {
		return ""
	}
	if genericAnchor {
		prefix := lower[:anchor]
		if strings.Contains(prefix, "ggml") || strings.Contains(prefix, "metal") {
			return ""
		}
	}
	version := strings.TrimSpace(line[anchor:])
	if version == "" {
		return ""
	}
	lowerVersion := strings.ToLower(version)
	if strings.HasPrefix(lowerVersion, "ggml_") ||
		strings.HasPrefix(lowerVersion, "ggml-") ||
		strings.HasPrefix(lowerVersion, "metal ") ||
		strings.HasPrefix(lowerVersion, "ggml metal") {
		return ""
	}
	return cleanProbeText(version, maxProbeVersionLen)
}

func versionAnchorIndex(lower string) (int, bool) {
	best := -1
	for _, anchor := range []string{"llama.cpp", "llama-cpp"} {
		if idx := strings.Index(lower, anchor); idx >= 0 && (best < 0 || idx < best) {
			best = idx
		}
	}
	if best >= 0 {
		return best, false
	}
	for _, anchor := range []string{"version", "build"} {
		if idx := strings.Index(lower, anchor); idx >= 0 && (best < 0 || idx < best) {
			best = idx
		}
	}
	return best, true
}

func cleanProbeText(value string, maxRunes int) string {
	value = strings.TrimSpace(value)
	if value == "" || maxRunes <= 0 {
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
