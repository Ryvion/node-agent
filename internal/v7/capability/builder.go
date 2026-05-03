package capability

import (
	goruntime "runtime"
	"strings"
	"time"

	"github.com/Ryvion/node-agent/internal/hw"
)

type BuildPassportInput struct {
	SchemaVersion   string
	AgentVersion    string
	NodePublicKey   string
	OS              string
	Arch            string
	DeviceType      string
	DeclaredCountry string

	HardwareCapabilities hw.CapSet
	HardwareProfile      HardwareProfile
	RuntimeProfile       RuntimeProfile

	NetworkCapabilitySummary  NetworkCapabilitySummary
	ModelCapabilitySummary    ModelCapabilitySummary
	SandboxCapabilitySummary  SandboxCapabilitySummary
	CASCapabilitySummary      CASCapabilitySummary
	EvidenceCapabilitySummary EvidenceCapabilitySummary

	CreatedAtUnixMs int64
}

func BuildCapabilityPassport(input BuildPassportInput) (CapabilityPassport, error) {
	schemaVersion := strings.TrimSpace(input.SchemaVersion)
	if schemaVersion == "" {
		schemaVersion = SchemaVersionV1
	}

	osName := strings.TrimSpace(input.OS)
	if osName == "" {
		osName = goruntime.GOOS
	}
	arch := strings.TrimSpace(input.Arch)
	if arch == "" {
		arch = goruntime.GOARCH
	}

	createdAtUnixMs := input.CreatedAtUnixMs
	if createdAtUnixMs == 0 {
		createdAtUnixMs = time.Now().UnixMilli()
	}

	runtimeProfile := input.RuntimeProfile
	runtimeProfile.SupportedRunnerKinds = cloneCleanStrings(runtimeProfile.SupportedRunnerKinds)

	modelSummary := input.ModelCapabilitySummary
	if len(modelSummary.SupportedModelFormats) == 0 {
		modelSummary.SupportedModelFormats = defaultModelFormats(runtimeProfile)
	} else {
		modelSummary.SupportedModelFormats = cloneModelFormats(modelSummary.SupportedModelFormats)
	}
	modelSummary.ResidentModelIDs = cloneCleanStrings(modelSummary.ResidentModelIDs)

	sandboxSummary := input.SandboxCapabilitySummary
	sandboxSummary.RejectsUnsafePickle = true
	if runtimeProfile.OCIAvailable {
		sandboxSummary.NetworkIsolationSupported = true
	}

	passport := CapabilityPassport{
		SchemaVersion:             schemaVersion,
		AgentVersion:              strings.TrimSpace(input.AgentVersion),
		NodePublicKey:             strings.TrimSpace(input.NodePublicKey),
		OS:                        osName,
		Arch:                      arch,
		DeviceType:                strings.TrimSpace(input.DeviceType),
		DeclaredCountry:           strings.ToUpper(strings.TrimSpace(input.DeclaredCountry)),
		HardwareProfile:           buildHardwareProfile(input.HardwareCapabilities, input.HardwareProfile),
		RuntimeProfile:            runtimeProfile,
		NetworkCapabilitySummary:  input.NetworkCapabilitySummary,
		ModelCapabilitySummary:    modelSummary,
		SandboxCapabilitySummary:  sandboxSummary,
		CASCapabilitySummary:      input.CASCapabilitySummary,
		EvidenceCapabilitySummary: input.EvidenceCapabilitySummary,
		CreatedAtUnixMs:           createdAtUnixMs,
	}

	return passport, ValidateCapabilityPassport(passport)
}

func buildHardwareProfile(caps hw.CapSet, override HardwareProfile) HardwareProfile {
	out := override
	if out.CPUCores == 0 {
		out.CPUCores = caps.CPUCores
	}
	if out.RAMBytes == 0 {
		out.RAMBytes = caps.RAMBytes
	}
	if strings.TrimSpace(out.GPUModel) == "" {
		out.GPUModel = strings.TrimSpace(caps.GPUModel)
	}
	if out.VRAMBytes == 0 {
		out.VRAMBytes = caps.VRAMBytes
	}
	if strings.TrimSpace(out.GPUVendor) == "" {
		out.GPUVendor = inferGPUVendor(out.GPUModel, caps.Sensors, caps.GfxVersion)
	}
	if strings.TrimSpace(out.DriverVersion) == "" {
		out.DriverVersion = driverVersionFromSensors(caps.Sensors)
	}
	if !out.TEESupported {
		out.TEESupported = caps.TEESupported
	}
	if strings.TrimSpace(out.TEEType) == "" {
		out.TEEType = strings.TrimSpace(caps.TEEType)
	}
	return out
}

func defaultModelFormats(runtimeProfile RuntimeProfile) []string {
	formats := make([]string, 0, 2)
	if runtimeProfile.NativeInferenceSupported || runtimeProfile.LlamaServerAvailable {
		formats = append(formats, "gguf")
	}
	if runtimeProfile.ImageRuntimeAvailable {
		formats = append(formats, "safetensors")
	}
	return formats
}

func cloneCleanStrings(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			out = append(out, value)
		}
	}
	return out
}

func cloneModelFormats(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	out := make([]string, 0, len(values))
	seen := map[string]struct{}{}
	for _, value := range values {
		value = normalizedModelFormat(value)
		if value == "" {
			continue
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}

func inferGPUVendor(parts ...string) string {
	text := strings.ToLower(strings.Join(parts, " "))
	switch {
	case strings.Contains(text, "nvidia") || strings.Contains(text, "rtx") ||
		strings.Contains(text, "gtx") || strings.Contains(text, "quadro") ||
		strings.Contains(text, "tesla"):
		return "nvidia"
	case strings.Contains(text, "amd") || strings.Contains(text, "rocm") ||
		strings.Contains(text, "radeon") || strings.Contains(text, "instinct") ||
		strings.Contains(text, "gfx"):
		return "amd"
	case strings.Contains(text, "intel") || strings.Contains(text, " arc "):
		return "intel"
	case strings.Contains(text, "apple") || strings.Contains(text, "m1") ||
		strings.Contains(text, "m2") || strings.Contains(text, "m3") ||
		strings.Contains(text, "m4"):
		return "apple"
	default:
		return ""
	}
}

func driverVersionFromSensors(sensors string) string {
	sensors = strings.TrimSpace(sensors)
	for _, prefix := range []string{"nvidia-driver:", "driver:"} {
		idx := strings.Index(strings.ToLower(sensors), prefix)
		if idx < 0 {
			continue
		}
		value := sensors[idx+len(prefix):]
		if end := strings.IndexAny(value, " \t,;"); end >= 0 {
			value = value[:end]
		}
		return strings.TrimSpace(value)
	}
	return ""
}
