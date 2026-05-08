package runtimeinventory

import (
	"context"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"time"
)

const (
	BackendCandidateLlamaCPP           = "llama.cpp"
	BackendCandidateOllama             = "ollama"
	BackendCandidateTensorRTLLM        = "tensorrt_llm"
	BackendCandidateVLLM               = "vllm"
	BackendCandidateSGLang             = "sglang"
	BackendCandidatePythonTransformers = "python_transformers"

	ggufModelFamilyLlama   = "llama"
	ggufModelFamilyPhi     = "phi"
	ggufModelFamilyGemma   = "gemma"
	ggufModelFamilyQwen    = "qwen"
	ggufModelFamilyUnknown = "unknown"

	maxBackendCandidates       = 8
	maxGGUFModels              = 20
	maxGGUFDirRead             = 256
	maxInventoryPathLen        = 512
	maxInventoryVersionLen     = 128
	maxInventoryBackendNameLen = 64
	defaultVersionTimeout      = 2 * time.Second
	unknownVersion             = "unknown"
)

var ggufQuantizationPattern = regexp.MustCompile(`(?i)(?:^|[._\-\s])((?:IQ|Q)[0-9](?:_[A-Z0-9]+){0,3}|BF16|F16|F32)(?:[._\-\s]|$)`)

type backendCandidateInventory struct {
	CandidateBackends CandidateBackends
	BackendCandidates []BackendCandidate
	GGUFModels        []GGUFModelCandidate
}

func DefaultCandidateBackendDetector() CandidateBackendDetector {
	return CandidateBackendDetector{}
}

func DetectCandidateBackends(detector CandidateBackendDetector) CandidateBackends {
	return detectBackendCandidateInventory(detector).CandidateBackends
}

func DetectBackendCandidates(detector CandidateBackendDetector) []BackendCandidate {
	return detectBackendCandidateInventory(detector).BackendCandidates
}

func DetectLlamaCPPBackendCandidate(detector CandidateBackendDetector) BackendCandidate {
	detector = normalizeCandidateBackendDetector(detector)
	candidates := normalizeBackendCandidates([]BackendCandidate{detectLlamaCPPCandidate(detector)})
	if len(candidates) == 0 {
		return BackendCandidate{
			Backend: BackendCandidateLlamaCPP,
			Version: unknownVersion,
			Reason:  "llama.cpp binary not detected",
		}
	}
	return candidates[0]
}

func DetectGGUFModels(detector CandidateBackendDetector) []GGUFModelCandidate {
	return normalizeGGUFModels(detectGGUFModels(normalizeCandidateBackendDetector(detector)))
}

func detectBackendCandidateInventory(detector CandidateBackendDetector) backendCandidateInventory {
	detector = normalizeCandidateBackendDetector(detector)

	llama := detectLlamaCPPCandidate(detector)
	ollama := detectCommandBackendCandidate(detector, BackendCandidateOllama, []string{"ollama"}, "ollama binary")
	tensorRTLLM := detectCommandBackendCandidate(detector, BackendCandidateTensorRTLLM, []string{"trtllm-serve", "trtllm-build"}, "TensorRT-LLM binary")
	vllm := detectCommandBackendCandidate(detector, BackendCandidateVLLM, []string{"vllm"}, "vLLM binary")
	sglang := detectCommandBackendCandidate(detector, BackendCandidateSGLang, []string{"sglang"}, "SGLang binary")
	python := detectPythonTransformersCandidate(detector)
	ggufModels := detectGGUFModels(detector)

	return backendCandidateInventory{
		CandidateBackends: CandidateBackends{
			LlamaCPPDetected:           llama.Detected,
			OllamaDetected:             ollama.Detected,
			TensorRTLLMDetected:        tensorRTLLM.Detected,
			VLLMDetected:               vllm.Detected,
			SGLangDetected:             sglang.Detected,
			PythonTransformersDetected: python.Detected,
			GGUFModelsDetected:         len(ggufModels) > 0,
		},
		BackendCandidates: normalizeBackendCandidates([]BackendCandidate{llama, ollama, tensorRTLLM, vllm, sglang, python}),
		GGUFModels:        normalizeGGUFModels(ggufModels),
	}
}

func detectLlamaCPPCandidate(detector CandidateBackendDetector) BackendCandidate {
	cliPath := findExecutable(detector, "llama-cli", true)
	serverPath := findExecutable(detector, "llama-server", true)
	benchPath := findExecutable(detector, "llama-bench", true)
	detected := cliPath != "" || serverPath != "" || benchPath != ""
	textRuntimeDetected := cliPath != "" || serverPath != ""

	version := unknownVersion
	if detected && detector.VersionCommand != nil {
		if probed, err := detector.VersionCommand(firstNonEmptyString(serverPath, cliPath, benchPath), detector.VersionTimeout); err == nil && strings.TrimSpace(probed) != "" {
			version = probed
		}
	}

	reason := "llama.cpp binary not detected"
	if detected {
		reason = "llama.cpp binary detected; KV/tensor hook support not confirmed"
	}
	if detected && !textRuntimeDetected {
		reason = "llama.cpp benchmark binary detected; text runtime binary not detected"
	}

	return BackendCandidate{
		Backend:                        BackendCandidateLlamaCPP,
		Detected:                       detected,
		BinaryPath:                     firstNonEmptyString(cliPath, serverPath, benchPath),
		ServerBinaryPath:               serverPath,
		BenchBinaryPath:                benchPath,
		Version:                        version,
		SupportsTextGeneration:         textRuntimeDetected,
		SupportsStreaming:              serverPath != "",
		SupportsOpenAICompatibleServer: serverPath != "",
		SupportsKVAccess:               false,
		SupportsTensorHooks:            false,
		SupportsSpeculativeDecode:      false,
		CandidateForRealTensorAccess:   textRuntimeDetected,
		CandidateForFastTextRuntime:    textRuntimeDetected,
		Reason:                         reason,
	}
}

func detectPythonTransformersCandidate(detector CandidateBackendDetector) BackendCandidate {
	pythonPath := findExecutable(detector, "python", false)
	python3Path := findExecutable(detector, "python3", false)
	binaryPath := firstNonEmptyString(python3Path, pythonPath)
	detected := binaryPath != ""

	importAttempted := false
	importAvailable := false
	reason := "python not detected"
	if detected {
		reason = "python detected; transformers package availability not fully probed"
		if detector.PythonModuleAvailable != nil {
			importAttempted = true
			if available, err := detector.PythonModuleAvailable(binaryPath, "transformers", detector.VersionTimeout); err == nil {
				importAvailable = available
				if available {
					reason = "python detected; transformers package import is available"
				} else {
					reason = "python detected; transformers package import not available"
				}
			} else {
				reason = "python detected; transformers package availability probe failed"
			}
		}
	}

	return BackendCandidate{
		Backend:                              BackendCandidatePythonTransformers,
		Detected:                             detected,
		BinaryPath:                           binaryPath,
		Version:                              unknownVersion,
		SupportsTextGeneration:               detected,
		SupportsStreaming:                    false,
		SupportsOpenAICompatibleServer:       false,
		SupportsKVAccess:                     false,
		SupportsTensorHooks:                  false,
		SupportsSpeculativeDecode:            false,
		CandidateForRealTensorAccess:         importAvailable,
		CandidateForFastTextRuntime:          false,
		PythonTransformersImportAvailable:    importAvailable,
		PythonTransformersImportProbeAttempt: importAttempted,
		Reason:                               reason,
	}
}

func detectCommandBackendCandidate(detector CandidateBackendDetector, backend string, binaryNames []string, label string) BackendCandidate {
	binaryPath := ""
	for _, name := range binaryNames {
		if path := findExecutable(detector, name, true); path != "" {
			binaryPath = path
			break
		}
	}
	detected := binaryPath != ""
	version := unknownVersion
	if detected && detector.VersionCommand != nil {
		if probed, err := detector.VersionCommand(binaryPath, detector.VersionTimeout); err == nil && strings.TrimSpace(probed) != "" {
			version = probed
		}
	}
	reason := strings.TrimSpace(label) + " not detected"
	if detected {
		reason = strings.TrimSpace(label) + " detected; runtime health not probed"
	}
	supportsText := detected
	supportsStreaming := detected && backend != BackendCandidateTensorRTLLM
	return BackendCandidate{
		Backend:                        backend,
		Detected:                       detected,
		BinaryPath:                     binaryPath,
		Version:                        version,
		SupportsTextGeneration:         supportsText,
		SupportsStreaming:              supportsStreaming,
		SupportsOpenAICompatibleServer: detected && (backend == BackendCandidateOllama || backend == BackendCandidateVLLM || backend == BackendCandidateSGLang),
		SupportsKVAccess:               false,
		SupportsTensorHooks:            false,
		SupportsSpeculativeDecode:      false,
		CandidateForRealTensorAccess:   detected && (backend == BackendCandidateVLLM || backend == BackendCandidateSGLang),
		CandidateForFastTextRuntime:    supportsText,
		Reason:                         reason,
	}
}

func detectGGUFModels(detector CandidateBackendDetector) []GGUFModelCandidate {
	detector = normalizeCandidateBackendDetector(detector)
	out := make([]GGUFModelCandidate, 0, maxGGUFModels)
	seen := map[string]struct{}{}
	for _, dir := range configuredModelDirs(detector) {
		names, err := detector.ReadDirNames(dir, maxGGUFDirRead)
		if err != nil {
			continue
		}
		sort.Strings(names)
		for _, name := range names {
			if len(out) >= maxGGUFModels {
				return out
			}
			name = strings.TrimSpace(name)
			if name == "" || !strings.EqualFold(filepath.Ext(name), ".gguf") {
				continue
			}
			path := joinConfiguredPath(detector.GOOS, dir, name)
			if _, ok := seen[path]; ok {
				continue
			}
			seen[path] = struct{}{}
			sizeBytes := int64(0)
			if detector.Stat != nil {
				info, statErr := detector.Stat(path)
				if statErr == nil {
					if info.IsDir() {
						continue
					}
					sizeBytes = info.Size()
					if sizeBytes < 0 {
						sizeBytes = 0
					}
				}
			}
			out = append(out, GGUFModelCandidate{
				Path:             path,
				Filename:         name,
				SizeBytes:        sizeBytes,
				ModelFamilyHint:  inferGGUFModelFamily(name),
				QuantizationHint: inferGGUFQuantization(name),
			})
		}
	}
	return out
}

func normalizeBackendCandidates(candidates []BackendCandidate) []BackendCandidate {
	if len(candidates) == 0 {
		return []BackendCandidate{}
	}
	out := make([]BackendCandidate, 0, min(len(candidates), maxBackendCandidates))
	for _, candidate := range candidates {
		if len(out) >= maxBackendCandidates {
			break
		}
		candidate.Backend = cleanInventoryText(candidate.Backend, maxInventoryBackendNameLen)
		if candidate.Backend == "" {
			continue
		}
		candidate.BinaryPath = cleanInventoryText(candidate.BinaryPath, maxInventoryPathLen)
		candidate.ServerBinaryPath = cleanInventoryText(candidate.ServerBinaryPath, maxInventoryPathLen)
		candidate.BenchBinaryPath = cleanInventoryText(candidate.BenchBinaryPath, maxInventoryPathLen)
		candidate.Version = cleanInventoryText(candidate.Version, maxInventoryVersionLen)
		if candidate.Version == "" {
			candidate.Version = unknownVersion
		}
		candidate.Reason = cleanInventoryText(candidate.Reason, maxInventoryReasonLen)
		if candidate.Reason == "" {
			candidate.Reason = candidate.Backend + " detection unavailable"
		}
		out = append(out, candidate)
	}
	if len(out) == 0 {
		return []BackendCandidate{}
	}
	return out
}

func normalizeGGUFModels(models []GGUFModelCandidate) []GGUFModelCandidate {
	if len(models) == 0 {
		return []GGUFModelCandidate{}
	}
	out := make([]GGUFModelCandidate, 0, min(len(models), maxGGUFModels))
	for _, model := range models {
		if len(out) >= maxGGUFModels {
			break
		}
		model.Path = cleanInventoryText(model.Path, maxInventoryPathLen)
		model.Filename = cleanInventoryText(model.Filename, maxInventoryPathLen)
		if model.Path == "" || model.Filename == "" {
			continue
		}
		if model.SizeBytes < 0 {
			model.SizeBytes = 0
		}
		model.ModelFamilyHint = normalizeGGUFModelFamily(model.ModelFamilyHint)
		model.QuantizationHint = cleanInventoryText(strings.ToUpper(strings.TrimSpace(model.QuantizationHint)), maxInventoryCompactFieldLen)
		if model.QuantizationHint == "" {
			model.QuantizationHint = "unknown"
		}
		out = append(out, model)
	}
	if len(out) == 0 {
		return []GGUFModelCandidate{}
	}
	return out
}

func normalizeCandidateBackendDetector(detector CandidateBackendDetector) CandidateBackendDetector {
	if detector.LookPath == nil {
		detector.LookPath = exec.LookPath
	}
	if detector.Stat == nil {
		detector.Stat = os.Stat
	}
	if detector.ReadDirNames == nil {
		detector.ReadDirNames = readDirNames
	}
	if detector.VersionCommand == nil {
		detector.VersionCommand = versionCommand
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
	return detector
}

func commandDetected(detector CandidateBackendDetector, name string) bool {
	return findExecutable(detector, name, false) != ""
}

func findExecutable(detector CandidateBackendDetector, name string, includeKnownDirs bool) string {
	name = strings.TrimSpace(name)
	if name == "" {
		return ""
	}
	for _, candidateName := range executableNameVariants(detector.GOOS, name) {
		if detector.LookPath != nil {
			path, err := detector.LookPath(candidateName)
			if err == nil && strings.TrimSpace(path) != "" {
				return strings.TrimSpace(path)
			}
		}
	}
	if !includeKnownDirs {
		return ""
	}
	for _, dir := range configuredBinaryDirs(detector) {
		for _, candidateName := range executableNameVariants(detector.GOOS, name) {
			path := joinConfiguredPath(detector.GOOS, dir, candidateName)
			info, err := detector.Stat(path)
			if err == nil && !info.IsDir() {
				return path
			}
		}
	}
	return ""
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

func configuredBinaryDirs(detector CandidateBackendDetector) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, 8)
	add := func(path string) {
		addConfiguredPath(&out, seen, path)
	}
	addWithBin := func(path string) {
		path = strings.TrimSpace(path)
		if path == "" {
			return
		}
		add(path)
		add(joinConfiguredPath(detector.GOOS, path, "bin"))
	}

	for _, dir := range detector.ConfiguredBinaryDirs {
		add(dir)
	}
	for _, envName := range []string{"RYV_LLAMA_CPP_DIR", "RYVION_LLAMA_CPP_DIR"} {
		addWithBin(detector.Getenv(envName))
	}
	for _, envName := range []string{"RYV_RUNTIME_DIR", "RYVION_RUNTIME_DIR"} {
		addWithBin(detector.Getenv(envName))
	}
	if dataDir := strings.TrimSpace(detector.Getenv("RYV_DATA_DIR")); dataDir != "" {
		add(joinConfiguredPath(detector.GOOS, dataDir, "bin"))
		addWithBin(joinConfiguredPath(detector.GOOS, dataDir, "runtime"))
	} else if home, err := detector.UserHomeDir(); err == nil && strings.TrimSpace(home) != "" {
		add(joinConfiguredPath(detector.GOOS, home, ".ryvion", "bin"))
		addWithBin(joinConfiguredPath(detector.GOOS, home, ".ryvion", "runtime"))
	}
	return out
}

func configuredModelDirs(detector CandidateBackendDetector) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, 4)
	add := func(path string) {
		addConfiguredPath(&out, seen, path)
	}

	for _, dir := range detector.ConfiguredModelDirs {
		add(dir)
	}
	for _, envName := range []string{"RYV_MODEL_DIR", "RYVION_MODEL_DIR"} {
		add(detector.Getenv(envName))
	}
	if dataDir := strings.TrimSpace(detector.Getenv("RYV_DATA_DIR")); dataDir != "" {
		add(joinConfiguredPath(detector.GOOS, dataDir, "models"))
	} else if home, err := detector.UserHomeDir(); err == nil && strings.TrimSpace(home) != "" {
		add(joinConfiguredPath(detector.GOOS, home, ".ryvion", "models"))
	}
	return out
}

func addConfiguredPath(out *[]string, seen map[string]struct{}, path string) {
	path = strings.TrimSpace(path)
	if path == "" {
		return
	}
	cleaned := filepath.Clean(path)
	if cleaned == "." || cleaned == string(filepath.Separator) {
		return
	}
	if _, ok := seen[cleaned]; ok {
		return
	}
	seen[cleaned] = struct{}{}
	*out = append(*out, cleaned)
}

func joinConfiguredPath(goos, dir string, elems ...string) string {
	dir = strings.TrimSpace(dir)
	if dir == "" {
		return ""
	}
	if strings.EqualFold(strings.TrimSpace(goos), "windows") && strings.Contains(dir, `\`) {
		path := strings.TrimRight(dir, `\/`)
		for _, elem := range elems {
			elem = strings.Trim(strings.TrimSpace(elem), `\/`)
			if elem == "" {
				continue
			}
			path += `\` + elem
		}
		return path
	}
	parts := append([]string{dir}, elems...)
	return filepath.Join(parts...)
}

func readDirNames(dir string, limit int) ([]string, error) {
	if limit <= 0 {
		limit = maxGGUFDirRead
	}
	f, err := os.Open(dir)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	names, err := f.Readdirnames(limit)
	if err != nil && err != io.EOF {
		return names, err
	}
	return names, nil
}

func versionCommand(binary string, timeout time.Duration) (string, error) {
	binary = strings.TrimSpace(binary)
	if binary == "" {
		return "", os.ErrNotExist
	}
	if timeout <= 0 {
		timeout = defaultVersionTimeout
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, binary, "--version")
	output, err := cmd.CombinedOutput()
	if ctx.Err() != nil {
		return "", ctx.Err()
	}
	text := sanitizeProbeOutput(string(output), maxInventoryVersionLen)
	if text != "" {
		return text, nil
	}
	return "", err
}

func pythonModuleAvailable(binary, module string, timeout time.Duration) (bool, error) {
	binary = strings.TrimSpace(binary)
	module = strings.TrimSpace(module)
	if binary == "" || module == "" {
		return false, os.ErrNotExist
	}
	if timeout <= 0 {
		timeout = defaultVersionTimeout
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	script := "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)"
	cmd := exec.CommandContext(ctx, binary, "-c", script, module)
	if err := cmd.Run(); err != nil {
		if ctx.Err() != nil {
			return false, ctx.Err()
		}
		return false, err
	}
	return true, nil
}

func sanitizeProbeOutput(value string, maxRunes int) string {
	value = strings.Join(strings.Fields(value), " ")
	return cleanInventoryText(value, maxRunes)
}

func firstNonEmptyString(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func inferGGUFModelFamily(filename string) string {
	lower := strings.ToLower(filename)
	switch {
	case strings.Contains(lower, "llama"):
		return ggufModelFamilyLlama
	case strings.Contains(lower, "phi"):
		return ggufModelFamilyPhi
	case strings.Contains(lower, "gemma"):
		return ggufModelFamilyGemma
	case strings.Contains(lower, "qwen"):
		return ggufModelFamilyQwen
	default:
		return ggufModelFamilyUnknown
	}
}

func normalizeGGUFModelFamily(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case ggufModelFamilyLlama:
		return ggufModelFamilyLlama
	case ggufModelFamilyPhi:
		return ggufModelFamilyPhi
	case ggufModelFamilyGemma:
		return ggufModelFamilyGemma
	case ggufModelFamilyQwen:
		return ggufModelFamilyQwen
	default:
		return ggufModelFamilyUnknown
	}
}

func inferGGUFQuantization(filename string) string {
	matches := ggufQuantizationPattern.FindStringSubmatch(filename)
	if len(matches) < 2 {
		return "unknown"
	}
	return strings.ToUpper(matches[1])
}
