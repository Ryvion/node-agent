package tensoraccess

import "strings"

const (
	maxCapabilityModelIDLen = 128
	maxCapabilityReasonLen  = 256
)

func NormalizeCapability(capability TensorAccessCapability) TensorAccessCapability {
	capability.Provider = normalizeProvider(capability.Provider)
	capability.Backend = normalizeBackend(capability.Backend)
	capability.RuntimeKind = normalizeRuntimeKind(capability.RuntimeKind)
	capability.ModelID = cleanCapabilityText(capability.ModelID, maxCapabilityModelIDLen)
	capability.Reason = cleanCapabilityText(capability.Reason, maxCapabilityReasonLen)
	if capability.Reason == "" {
		capability.Reason = defaultReason(capability)
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

func cleanCapabilityText(value string, limit ...int) string {
	value = strings.TrimSpace(value)
	maxRunes := maxCapabilityReasonLen
	if len(limit) > 0 {
		maxRunes = limit[0]
	}
	if value == "" || maxRunes <= 0 {
		return ""
	}
	var b strings.Builder
	b.Grow(len(value))
	written := 0
	for _, r := range value {
		if r < 0x20 || r == 0x7f {
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
