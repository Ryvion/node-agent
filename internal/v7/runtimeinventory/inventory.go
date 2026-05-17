package runtimeinventory

import (
	"strings"
	"unicode"

	"github.com/Ryvion/ryvion-node/internal/v7/tensoraccess"
)

const (
	maxInventoryCompactFieldLen = 64
	maxInventoryModelIDLen      = 128
	maxInventoryReasonLen       = 256
	maxInventoryLoadedModels    = 32
)

func BuildInventory(status RuntimeStatus, detector CandidateBackendDetector) Inventory {
	normalized := normalizeRuntimeStatus(status)
	backendInventory := detectBackendCandidateInventory(detector)
	return NormalizeInventory(Inventory{
		RuntimeKind:          normalized.RuntimeKind,
		Backend:              normalized.Backend,
		Provider:             normalized.Provider,
		ProcessMode:          normalized.ProcessMode,
		NativeInferenceReady: normalized.NativeInferenceReady,
		NativeModel:          normalized.NativeModel,
		LoadedModels:         BuildModelResidencySnapshot(normalized),
		CandidateBackends:    backendInventory.CandidateBackends,
		BackendCandidates:    backendInventory.BackendCandidates,
		GGUFModels:           backendInventory.GGUFModels,
	})
}

func NormalizeInventory(inventory Inventory) Inventory {
	inventory.RuntimeKind = normalizeRuntimeKind(cleanInventoryText(inventory.RuntimeKind, maxInventoryCompactFieldLen))
	inventory.Backend = normalizeBackend(cleanInventoryText(inventory.Backend, maxInventoryCompactFieldLen))
	inventory.Provider = normalizeProvider(cleanInventoryText(inventory.Provider, maxInventoryCompactFieldLen))
	inventory.ProcessMode = normalizeProcessMode(cleanInventoryText(inventory.ProcessMode, maxInventoryCompactFieldLen))
	inventory.NativeModel = cleanInventoryText(inventory.NativeModel, maxInventoryModelIDLen)
	inventory.LoadedModels = normalizeLoadedModels(inventory.LoadedModels)
	inventory.BackendCandidates = normalizeBackendCandidates(inventory.BackendCandidates)
	inventory.GGUFModels = normalizeGGUFModels(inventory.GGUFModels)
	return inventory
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
