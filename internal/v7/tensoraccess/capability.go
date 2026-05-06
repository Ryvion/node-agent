package tensoraccess

import "strings"

const maxCapabilityTextLen = 256

func NormalizeCapability(capability TensorAccessCapability) TensorAccessCapability {
	capability.Provider = normalizeProvider(capability.Provider)
	capability.Backend = normalizeBackend(capability.Backend)
	capability.RuntimeKind = normalizeRuntimeKind(capability.RuntimeKind)
	capability.ModelID = cleanCapabilityText(capability.ModelID)
	capability.Reason = cleanCapabilityText(capability.Reason)
	if capability.Reason == "" {
		capability.Reason = defaultReason(capability)
	}
	if capability.Provider == ProviderNoop {
		capability.TensorPlaneDemoSupported = false
	}
	if capability.Provider == ProviderTensorPlaneDemo {
		capability.TensorPlaneDemoSupported = true
	}
	if capability.Backend == BackendUnknown {
		capability.ModelLoaded = false
	}
	return capability
}

func normalizeProvider(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case ProviderNoop, "":
		return ProviderNoop
	case ProviderTensorPlaneDemo:
		return ProviderTensorPlaneDemo
	case ProviderLlamaCPP, "llama.cpp", "llamacpp", "llama-cpp":
		return ProviderLlamaCPP
	case ProviderVLLM:
		return ProviderVLLM
	case ProviderRyvionRuntime:
		return ProviderRyvionRuntime
	default:
		return ProviderNoop
	}
}

func normalizeBackend(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case BackendNative:
		return BackendNative
	case BackendDemo:
		return BackendDemo
	case BackendUnknown, "":
		return BackendUnknown
	default:
		return BackendUnknown
	}
}

func normalizeRuntimeKind(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case RuntimeKindNative, "":
		return RuntimeKindNative
	case RuntimeKindDemo:
		return RuntimeKindDemo
	case RuntimeKindUnknown:
		return RuntimeKindUnknown
	default:
		return RuntimeKindUnknown
	}
}

func defaultReason(capability TensorAccessCapability) string {
	if capability.Provider == ProviderTensorPlaneDemo {
		return ReasonTensorPlaneDemoOnly
	}
	if capability.Backend == BackendUnknown {
		return ReasonNativeRuntimeUnavailable
	}
	return ReasonTextGenerationOnly
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
