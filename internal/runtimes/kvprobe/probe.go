package kvprobe

import "strings"

const maxCapabilityTextLen = 256

func ProbeNativeRuntime(input ProbeInput) Capability {
	capability := Capability{
		KVAccessSupported:          input.Hooks.KVAccessSupported,
		KVSnapshotSupported:        input.Hooks.KVSnapshotSupported,
		HiddenStateAccessSupported: input.Hooks.HiddenStateAccessSupported,
		LogitsAccessSupported:      input.Hooks.LogitsAccessSupported,
		AttentionHookSupported:     input.Hooks.AttentionHookSupported,
		Backend:                    normalizeBackend(input.Backend, input.RuntimeAvailable),
		ModelID:                    cleanCapabilityText(input.ModelID),
		ModelLoaded:                input.RuntimeAvailable && input.ModelLoaded,
		RuntimeKind:                normalizeRuntimeKind(input.RuntimeKind),
		Reason:                     cleanCapabilityText(input.Reason),
	}
	if capability.Reason == "" {
		capability.Reason = defaultReason(input.RuntimeAvailable, capability.hasAnyTensorHook())
	}
	return NormalizeCapability(capability)
}

func NormalizeCapability(capability Capability) Capability {
	capability.Backend = normalizeBackend(capability.Backend, false)
	capability.ModelID = cleanCapabilityText(capability.ModelID)
	capability.RuntimeKind = normalizeRuntimeKind(capability.RuntimeKind)
	capability.Reason = cleanCapabilityText(capability.Reason)
	if capability.Reason == "" {
		capability.Reason = defaultReason(capability.Backend != BackendUnknown, capability.hasAnyTensorHook())
	}
	if capability.Backend == BackendUnknown {
		capability.ModelLoaded = false
	}
	return capability
}

func (capability Capability) hasAnyTensorHook() bool {
	return capability.KVAccessSupported ||
		capability.KVSnapshotSupported ||
		capability.HiddenStateAccessSupported ||
		capability.LogitsAccessSupported ||
		capability.AttentionHookSupported
}

func defaultReason(runtimeAvailable bool, hasHooks bool) string {
	if hasHooks {
		return ReasonTensorHooksAvailable
	}
	if !runtimeAvailable {
		return ReasonNativeRuntimeUnavailable
	}
	return ReasonTextGenerationOnly
}

func normalizeRuntimeKind(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", RuntimeKindNative:
		return RuntimeKindNative
	default:
		return RuntimeKindUnknown
	}
}

func normalizeBackend(value string, runtimeAvailable bool) string {
	normalized := strings.ToLower(strings.TrimSpace(value))
	switch normalized {
	case BackendNative:
		return BackendNative
	case BackendLlamaCPP, "llamacpp", "llama-cpp", "llama_server", "llama-server":
		return BackendLlamaCPP
	case BackendUnknown:
		return BackendUnknown
	case "":
		if runtimeAvailable {
			return BackendLlamaCPP
		}
		return BackendUnknown
	default:
		return BackendUnknown
	}
}

func cleanCapabilityText(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	var b strings.Builder
	b.Grow(len(value))
	for _, r := range value {
		if r < 0x20 || r == 0x7f {
			continue
		}
		b.WriteRune(r)
		if b.Len() >= maxCapabilityTextLen {
			break
		}
	}
	return strings.TrimSpace(b.String())
}
