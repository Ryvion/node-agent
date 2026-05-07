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

	"github.com/Ryvion/node-agent/internal/v7/runtimeinventory"
)

const (
	defaultVersionTimeout    = 2 * time.Second
	defaultModelProbeTimeout = 5 * time.Second
	unknownVersion           = "unknown"
	maxProbePathLen          = 512
	maxProbeReasonLen        = 256
)

func ProbeAll(detector Detector) Probes {
	return Probes{
		LlamaCPP: ProbeLlamaCPP(detector),
	}
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

func normalizeProbeVersion(value string) string {
	value = cleanProbeText(value, maxProbeReasonLen)
	if value == "" {
		return unknownVersion
	}
	lower := strings.ToLower(value)
	if strings.Contains(lower, "llama") || strings.Contains(lower, "version") || strings.Contains(lower, "build") {
		return value
	}
	return unknownVersion
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
