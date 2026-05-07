package runtimeinventory

import (
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"unicode"

	"github.com/Ryvion/node-agent/internal/v7/tensoraccess"
)

const (
	maxInventoryCompactFieldLen = 64
	maxInventoryModelIDLen      = 128
	maxInventoryReasonLen       = 256
	maxInventoryLoadedModels    = 32
)

func BuildInventory(status RuntimeStatus, detector CandidateBackendDetector) Inventory {
	normalized := normalizeRuntimeStatus(status)
	return NormalizeInventory(Inventory{
		RuntimeKind:          normalized.RuntimeKind,
		Backend:              normalized.Backend,
		Provider:             normalized.Provider,
		ProcessMode:          normalized.ProcessMode,
		NativeInferenceReady: normalized.NativeInferenceReady,
		NativeModel:          normalized.NativeModel,
		LoadedModels:         BuildModelResidencySnapshot(normalized),
		CandidateBackends:    DetectCandidateBackends(detector),
	})
}

func NormalizeInventory(inventory Inventory) Inventory {
	inventory.RuntimeKind = normalizeRuntimeKind(cleanInventoryText(inventory.RuntimeKind, maxInventoryCompactFieldLen))
	inventory.Backend = normalizeBackend(cleanInventoryText(inventory.Backend, maxInventoryCompactFieldLen))
	inventory.Provider = normalizeProvider(cleanInventoryText(inventory.Provider, maxInventoryCompactFieldLen))
	inventory.ProcessMode = normalizeProcessMode(cleanInventoryText(inventory.ProcessMode, maxInventoryCompactFieldLen))
	inventory.NativeModel = cleanInventoryText(inventory.NativeModel, maxInventoryModelIDLen)
	inventory.LoadedModels = normalizeLoadedModels(inventory.LoadedModels)
	return inventory
}

func DetectCandidateBackends(detector CandidateBackendDetector) CandidateBackends {
	detector = normalizeCandidateBackendDetector(detector)
	return CandidateBackends{
		LlamaCPPDetected:           commandDetected(detector, "llama-server") || commandDetected(detector, "llama-cli"),
		OllamaDetected:             commandDetected(detector, "ollama"),
		VLLMDetected:               commandDetected(detector, "vllm"),
		PythonTransformersDetected: commandDetected(detector, "python") || commandDetected(detector, "python3"),
		GGUFModelsDetected:         ggufModelDetected(detector),
	}
}

func DefaultCandidateBackendDetector() CandidateBackendDetector {
	return CandidateBackendDetector{}
}

func normalizeRuntimeStatus(status RuntimeStatus) RuntimeStatus {
	status.RuntimeKind = normalizeRuntimeKind(status.RuntimeKind)
	status.Backend = normalizeBackend(status.Backend)
	status.Provider = normalizeProvider(status.Provider)
	status.ProcessMode = normalizeProcessMode(status.ProcessMode)
	status.NativeModel = cleanInventoryText(status.NativeModel, maxInventoryModelIDLen)
	status.Reason = cleanInventoryText(status.Reason, maxInventoryReasonLen)
	if status.Reason == "" {
		status.Reason = defaultInventoryReason(status.Backend)
	}
	return status
}

func normalizeLoadedModels(models []ModelResidencySnapshot) []ModelResidencySnapshot {
	if len(models) == 0 {
		return []ModelResidencySnapshot{}
	}
	out := make([]ModelResidencySnapshot, 0, min(len(models), maxInventoryLoadedModels))
	for _, model := range models {
		if len(out) >= maxInventoryLoadedModels {
			break
		}
		model.ModelID = cleanInventoryText(model.ModelID, maxInventoryModelIDLen)
		if model.ModelID == "" {
			continue
		}
		model.RuntimeKind = normalizeRuntimeKind(cleanInventoryText(model.RuntimeKind, maxInventoryCompactFieldLen))
		model.Backend = normalizeBackend(cleanInventoryText(model.Backend, maxInventoryCompactFieldLen))
		model.Reason = cleanInventoryText(model.Reason, maxInventoryReasonLen)
		if model.Reason == "" {
			model.Reason = defaultInventoryReason(model.Backend)
		}
		out = append(out, model)
	}
	if len(out) == 0 {
		return []ModelResidencySnapshot{}
	}
	return out
}

func defaultInventoryReason(backend string) string {
	if backend == BackendNative {
		return tensoraccess.ReasonTextGenerationOnly
	}
	return tensoraccess.ReasonNativeRuntimeUnavailable
}

func normalizeRuntimeKind(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case RuntimeKindNative:
		return RuntimeKindNative
	case RuntimeKindUnknown, "":
		return RuntimeKindUnknown
	default:
		return RuntimeKindUnknown
	}
}

func normalizeBackend(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case BackendNative:
		return BackendNative
	case BackendUnknown, "":
		return BackendUnknown
	default:
		return BackendUnknown
	}
}

func normalizeProvider(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case tensoraccess.ProviderNoop, "":
		return tensoraccess.ProviderNoop
	case tensoraccess.ProviderTensorPlaneDemo:
		return tensoraccess.ProviderTensorPlaneDemo
	case tensoraccess.ProviderLlamaCPP:
		return tensoraccess.ProviderLlamaCPP
	case tensoraccess.ProviderVLLM:
		return tensoraccess.ProviderVLLM
	case tensoraccess.ProviderRyvionRuntime:
		return tensoraccess.ProviderRyvionRuntime
	default:
		return ProviderNoop
	}
}

func normalizeProcessMode(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case ProcessModeEmbedded:
		return ProcessModeEmbedded
	case ProcessModeSidecar:
		return ProcessModeSidecar
	case ProcessModeUnknown, "":
		return ProcessModeUnknown
	default:
		return ProcessModeUnknown
	}
}

func normalizeCandidateBackendDetector(detector CandidateBackendDetector) CandidateBackendDetector {
	if detector.LookPath == nil {
		detector.LookPath = exec.LookPath
	}
	if detector.ReadDirNames == nil {
		detector.ReadDirNames = readDirNames
	}
	if detector.Getenv == nil {
		detector.Getenv = os.Getenv
	}
	if detector.UserHomeDir == nil {
		detector.UserHomeDir = os.UserHomeDir
	}
	return detector
}

func commandDetected(detector CandidateBackendDetector, name string) bool {
	if strings.TrimSpace(name) == "" || detector.LookPath == nil {
		return false
	}
	path, err := detector.LookPath(name)
	return err == nil && strings.TrimSpace(path) != ""
}

func ggufModelDetected(detector CandidateBackendDetector) bool {
	for _, dir := range configuredModelDirs(detector) {
		names, err := detector.ReadDirNames(dir, 256)
		if err != nil {
			continue
		}
		for _, name := range names {
			if strings.EqualFold(filepath.Ext(strings.TrimSpace(name)), ".gguf") {
				return true
			}
		}
	}
	return false
}

func configuredModelDirs(detector CandidateBackendDetector) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, 4)
	add := func(path string) {
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
		out = append(out, cleaned)
	}

	for _, dir := range detector.ConfiguredModelDirs {
		add(dir)
	}
	for _, envName := range []string{"RYV_MODEL_DIR", "RYVION_MODEL_DIR"} {
		add(detector.Getenv(envName))
	}
	if dataDir := strings.TrimSpace(detector.Getenv("RYV_DATA_DIR")); dataDir != "" {
		add(filepath.Join(dataDir, "models"))
	} else if home, err := detector.UserHomeDir(); err == nil && strings.TrimSpace(home) != "" {
		add(filepath.Join(home, ".ryvion", "models"))
	}
	return out
}

func readDirNames(dir string, limit int) ([]string, error) {
	if limit <= 0 {
		limit = 256
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

func cleanInventoryText(value string, maxRunes int) string {
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
