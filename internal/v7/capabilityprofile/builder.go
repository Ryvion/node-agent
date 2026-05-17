package capabilityprofile

import (
	"strings"
	"unicode"

	"github.com/Ryvion/ryvion-node/internal/runtimes/llamacpp"
	"github.com/Ryvion/ryvion-node/internal/v7/backendprobe"
	v7hardware "github.com/Ryvion/ryvion-node/internal/v7/hardware"
	"github.com/Ryvion/ryvion-node/internal/v7/inferenceconfig"
	"github.com/Ryvion/ryvion-node/internal/v7/kvprobe"
	"github.com/Ryvion/ryvion-node/internal/v7/modelcache"
	"github.com/Ryvion/ryvion-node/internal/v7/modelpolicy"
	"github.com/Ryvion/ryvion-node/internal/v7/runtimeinventory"
	"github.com/Ryvion/ryvion-node/internal/v7/speculative"
	"github.com/Ryvion/ryvion-node/internal/v7/tensoraccess"
)

const (
	maxProfileModels      = 16
	maxProfileTextRunes   = 128
	maxProfileReasonRunes = 192
)

type BuildInput struct {
	Hardware            v7hardware.CapacityInventory
	Policy              modelpolicy.Policy
	ModelCache          modelcache.Status
	BackendProbes       backendprobe.Probes
	BackendRuntimes     llamacpp.BackendRuntimes
	RuntimeInventory    runtimeinventory.Inventory
	SpeculativeDecoding *speculative.DecodingCapability
	KVCapability        *kvprobe.Capability
	TensorAccess        tensoraccess.TensorAccessCapability
	Getenv              func(string) string
}

func BuildProfile(input BuildInput) Profile {
	hardware := v7hardware.NormalizeInventory(input.Hardware)
	policy := modelpolicy.NormalizePolicy(input.Policy)
	cache := modelcache.NormalizeStatus(input.ModelCache)
	probes := backendprobe.NormalizeProbes(input.BackendProbes)
	runtimes := llamacpp.NormalizeBackendRuntimes(input.BackendRuntimes)
	tensor := tensoraccess.NormalizeCapability(input.TensorAccess)

	backend := backendSummary(probes, runtimes)
	hardwareOK := hasHardwareCapacity(hardware)
	inferenceEnabled := inferenceconfig.V7InferenceEnabled(input.Getenv)
	textEnabled := inferenceconfig.TextOutputEnabled(input.Getenv)
	streamingEnabled := inferenceconfig.StreamingEnabled(input.Getenv)
	warmEnabled := inferenceconfig.ModelWarmEnabled(input.Getenv)

	ggufBackendText := ggufBackendSupportsText(probes, runtimes)
	cache = modelcache.AnnotateRuntimeStatus(modelcache.RuntimeAnnotationInput{
		Status:                         cache,
		Policy:                         policy,
		HardwareCapacityAvailable:      hardwareOK,
		BackendTextGenerationAvailable: ggufBackendText,
		V7InferenceEnabled:             inferenceEnabled,
	})
	models, anyRunnable := modelCapabilities(cache, policy, hardwareOK, ggufBackendText, inferenceEnabled, runtimes)
	models = appendRuntimeModelIfMissing(models, policy, hardwareOK, ggufBackendText, inferenceEnabled, runtimes)
	anyRunnable = anyRunnable || anyRunnableModel(models)
	speculativeDecoding := speculativeDecodingSummary(input, hardware, policy, cache, probes, runtimes)

	backendText := hardwareOK && backend.SupportsTextGeneration && inferenceEnabled
	ready := hardwareOK && inferenceEnabled && anyRunnable
	hashMetrics := backendText
	textOutput := backendText && textEnabled
	streaming := textOutput && streamingEnabled && backend.SupportsStreaming
	backendWarm := hardwareOK && backend.SupportsWarmResidency && warmEnabled
	modelPrepare := backendText && policy.AutoDownload

	reason := readinessReason(hardwareOK, backend, inferenceEnabled, anyRunnable, policy)
	return NormalizeProfile(Profile{
		SchemaVersion:         SchemaVersionV1,
		V7DashboardInference:  backendText,
		TextOutput:            textOutput,
		Streaming:             streaming,
		HashMetricsReceipts:   hashMetrics,
		WarmBackend:           backendWarm,
		ModelPrepare:          modelPrepare,
		BackendTextGeneration: hardwareOK && backend.SupportsTextGeneration && inferenceEnabled,
		BackendWarm:           backendWarm,
		SpeculativeDecoding:   &speculativeDecoding,
		StatefulSession:       false,
		KVAccess:              input.KVCapability != nil && input.KVCapability.KVAccessSupported,
		TensorHooks:           tensor.KVAccessSupported || tensor.KVSnapshotSupported || tensor.HiddenStateAccessSupported || tensor.LogitsAccessSupported || tensor.AttentionHookSupported,
		TensorPlaneDemo:       tensor.TensorPlaneDemoSupported,
		Ready:                 ready,
		Reason:                reason,
		Hardware:              hardwareSummary(hardware),
		Policy:                policySummary(policy),
		BackendRuntime:        backend,
		WarmModel:             warmModelSummary(runtimes),
		Models:                models,
	})
}

func NormalizeProfile(profile Profile) Profile {
	if strings.TrimSpace(profile.SchemaVersion) == "" {
		profile.SchemaVersion = SchemaVersionV1
	}
	profile.Reason = cleanProfileText(profile.Reason, maxProfileReasonRunes)
	profile.Hardware.OS = cleanProfileText(strings.ToLower(profile.Hardware.OS), maxProfileTextRunes)
	profile.Hardware.Arch = cleanProfileText(strings.ToLower(profile.Hardware.Arch), maxProfileTextRunes)
	profile.Hardware.GPUVendor = cleanProfileText(strings.ToLower(profile.Hardware.GPUVendor), maxProfileTextRunes)
	profile.Hardware.GPUName = cleanProfileText(profile.Hardware.GPUName, maxProfileTextRunes)
	profile.Policy.AllowedFormats = cleanProfileList(profile.Policy.AllowedFormats)
	profile.Policy.AllowedFamilies = cleanProfileList(profile.Policy.AllowedFamilies)
	profile.Policy.DeniedModelIDs = cleanProfileList(profile.Policy.DeniedModelIDs)
	profile.Policy.DeniedFamilies = cleanProfileList(profile.Policy.DeniedFamilies)
	if profile.Policy.MaxWarmModels < 0 {
		profile.Policy.MaxWarmModels = 0
	}
	if profile.Policy.MaxConcurrentInferenceJobs < 0 {
		profile.Policy.MaxConcurrentInferenceJobs = 0
	}
	profile.BackendRuntime.Backend = cleanProfileText(profile.BackendRuntime.Backend, maxProfileTextRunes)
	profile.BackendRuntime.Reason = cleanProfileText(profile.BackendRuntime.Reason, maxProfileReasonRunes)
	profile.BackendRuntime.Acceleration = cleanProfileList(profile.BackendRuntime.Acceleration)
	if profile.SpeculativeDecoding != nil {
		speculativeDecoding := speculative.NormalizeCapability(*profile.SpeculativeDecoding)
		profile.SpeculativeDecoding = &speculativeDecoding
	}
	profile.WarmModel.Backend = cleanProfileText(profile.WarmModel.Backend, maxProfileTextRunes)
	profile.WarmModel.ModelID = cleanProfileText(profile.WarmModel.ModelID, maxProfileTextRunes)
	profile.Models = normalizeModels(profile.Models)
	return profile
}

func speculativeDecodingSummary(input BuildInput, hardware v7hardware.CapacityInventory, policy modelpolicy.Policy, cache modelcache.Status, probes backendprobe.Probes, runtimes llamacpp.BackendRuntimes) speculative.DecodingCapability {
	if input.SpeculativeDecoding != nil {
		return speculative.NormalizeCapability(*input.SpeculativeDecoding)
	}
	report := speculative.BuildReport(speculative.BuildInput{
		Hardware:         hardware,
		Policy:           policy,
		ModelCache:       cache,
		BackendProbes:    probes,
		BackendRuntimes:  runtimes,
		RuntimeInventory: input.RuntimeInventory,
		Getenv:           input.Getenv,
	})
	return report.SpeculativeDecoding
}

func hardwareSummary(hardware v7hardware.CapacityInventory) HardwareSummary {
	return HardwareSummary{
		OS:                hardware.OS,
		Arch:              hardware.Arch,
		CPULogicalCores:   hardware.CPULogicalCores,
		SystemRAMBytes:    hardware.SystemRAMBytes,
		AvailableRAMBytes: hardware.AvailableRAMBytes,
		GPUDetected:       hardware.GPUDetected,
		GPUVendor:         hardware.GPUVendor,
		GPUName:           hardware.GPUName,
		GPUVRAMBytes:      hardware.GPUVRAMBytes,
		UnifiedMemory:     hardware.UnifiedMemory,
		CUDAAvailable:     hardware.CUDAAvailable,
		MetalAvailable:    hardware.MetalAvailable,
		VulkanAvailable:   hardware.VulkanAvailable,
	}
}

func policySummary(policy modelpolicy.Policy) PolicySummary {
	return PolicySummary{
		MaxRuntimeModelBytes:             policy.RuntimePolicy.MaxRuntimeModelBytes,
		MaxRuntimeParameterCountBillions: policy.RuntimePolicy.MaxRuntimeParameterCountBillions,
		AllowedFormats:                   cloneStrings(policy.AllowedFormats),
		AllowedFamilies:                  cloneStrings(policy.RuntimePolicy.AllowFamilies),
		DeniedModelIDs:                   cloneStrings(policy.RuntimePolicy.DenyModelIDs),
		DeniedFamilies:                   cloneStrings(policy.RuntimePolicy.DenyFamilies),
		AllowLargeModels:                 policy.RuntimePolicy.AllowLargeModels,
		AllowManagedPrepareDownload:      policy.AutoDownload,
		MaxWarmModels:                    policy.RuntimePolicy.MaxWarmModels,
		MaxConcurrentInferenceJobs:       policy.RuntimePolicy.MaxConcurrentInferenceJobs,
	}
}

func backendSummary(probes backendprobe.Probes, runtimes llamacpp.BackendRuntimes) BackendSummary {
	runtime := runtimes.LlamaCPP
	probe := probes.LlamaCPP
	selected := selectBackendRuntime(runtimes)
	backend := firstNonEmpty(selected.Backend, llamacpp.BackendName)
	available := (runtime.Available && runtime.SupportsTextGeneration) ||
		(probe.Available && probe.SupportsTextGeneration) ||
		(selected.Available && selected.SupportsTextGeneration)
	supportsText := available
	supportsStreaming := (runtime.Available && runtime.SupportsStreaming) ||
		(probe.Available && probe.SupportsStreaming) ||
		(selected.Available && selected.SupportsStreaming)
	supportsWarm := available && ((runtime.Available && runtime.OpenAICompatible) || probe.SupportsOpenAICompatibleServer || supportsStreaming)
	reason := firstNonEmpty(runtime.LastError, probe.Reason)
	if supportsText {
		reason = "backend text generation available"
	} else if reason == "" {
		reason = "backend text generation unavailable"
	}
	return BackendSummary{
		Backend:                backend,
		Available:              available,
		Running:                selected.Running,
		Healthy:                selected.Healthy,
		SupportsTextGeneration: supportsText,
		SupportsStreaming:      supportsStreaming,
		SupportsWarmResidency:  supportsWarm,
		Acceleration:           selected.Acceleration,
		Reason:                 reason,
	}
}

func selectBackendRuntime(runtimes llamacpp.BackendRuntimes) llamacpp.BackendRuntimeStatus {
	candidates := []llamacpp.BackendRuntimeStatus{runtimes.LlamaCPP, runtimes.TensorRTLLM, runtimes.VLLM, runtimes.SGLang}
	candidates = append(candidates, runtimes.Other...)
	for _, candidate := range candidates {
		if candidate.Running && candidate.Healthy && candidate.SupportsTextGeneration {
			return candidate
		}
	}
	for _, candidate := range candidates {
		if candidate.Available && candidate.SupportsTextGeneration {
			return candidate
		}
	}
	return runtimes.LlamaCPP
}

func ggufBackendSupportsText(probes backendprobe.Probes, runtimes llamacpp.BackendRuntimes) bool {
	return (runtimes.LlamaCPP.Available && runtimes.LlamaCPP.SupportsTextGeneration) ||
		(probes.LlamaCPP.Available && probes.LlamaCPP.SupportsTextGeneration)
}

func warmModelSummary(runtimes llamacpp.BackendRuntimes) WarmModelSummary {
	runtime := runtimes.LlamaCPP
	return WarmModelSummary{
		Backend: runtime.Backend,
		ModelID: firstNonEmpty(runtime.ModelID, runtime.ModelFilename),
		Warm:    runtime.Warm,
		Healthy: runtime.Healthy,
	}
}

func modelCapabilities(cache modelcache.Status, policy modelpolicy.Policy, hardwareOK bool, backendText bool, inferenceEnabled bool, runtimes llamacpp.BackendRuntimes) ([]ModelCapability, bool) {
	models := make([]ModelCapability, 0, min(len(cache.Models), maxProfileModels))
	anyRunnable := false
	for _, model := range cache.Models {
		if len(models) >= maxProfileModels {
			break
		}
		if strings.TrimSpace(model.ModelID) == "" {
			continue
		}
		capability := buildModelCapability(model, policy, hardwareOK, backendText, inferenceEnabled, runtimes)
		if capability.Runnable {
			anyRunnable = true
		}
		models = append(models, capability)
	}
	return models, anyRunnable
}

func buildModelCapability(model modelcache.Model, policy modelpolicy.Policy, hardwareOK bool, backendText bool, inferenceEnabled bool, runtimes llamacpp.BackendRuntimes) ModelCapability {
	decision := modelpolicy.EvaluateRuntimeRequest(policy, modelpolicy.RuntimeRequest{
		ModelID:                model.ModelID,
		ModelSizeBytes:         uint64NonNegative(model.SizeBytes),
		ParameterCountBillions: model.ParameterCountBillions,
		Family:                 model.FamilyHint,
	})
	warm := modelMatchesRuntime(model, runtimes.LlamaCPP) && runtimes.LlamaCPP.Warm
	blockedReasons := model.BlockedReasons
	reason := firstNonEmpty(blockedReasons...)
	if reason == "" {
		reason = decision.Reason
	}
	runnable := model.Runnable && decision.Allowed && hardwareOK && backendText && inferenceEnabled
	switch {
	case !inferenceEnabled:
		reason = "v7_inference_disabled"
	case !hardwareOK:
		reason = "hardware_capacity_missing"
	case !backendText:
		reason = "backend_text_generation_unavailable"
	case decision.Allowed:
		reason = "runnable"
	}
	return ModelCapability{
		ModelID:        model.ModelID,
		Family:         model.FamilyHint,
		Format:         model.Format,
		SizeBytes:      model.SizeBytes,
		Resident:       model.Installed,
		Warm:           warm,
		Runnable:       runnable,
		BlockedReasons: blockedReasons,
		Reason:         reason,
	}
}

func appendRuntimeModelIfMissing(models []ModelCapability, policy modelpolicy.Policy, hardwareOK bool, backendText bool, inferenceEnabled bool, runtimes llamacpp.BackendRuntimes) []ModelCapability {
	runtime := runtimes.LlamaCPP
	modelID := firstNonEmpty(runtime.ModelID, runtime.ModelFilename)
	if strings.TrimSpace(modelID) == "" || !runtime.Loaded {
		return models
	}
	for _, model := range models {
		if strings.EqualFold(strings.TrimSpace(model.ModelID), strings.TrimSpace(modelID)) {
			return models
		}
	}
	model := modelcache.Model{
		ModelID:                modelID,
		Filename:               runtime.ModelFilename,
		SizeBytes:              runtime.ModelSizeBytes,
		FamilyHint:             runtime.ModelFamilyHint,
		ParameterCountBillions: modelcache.InferParameterCountBillions(firstNonEmpty(runtime.ModelFilename, runtime.ModelID)),
		Format:                 modelcache.DefaultFormat,
		Installed:              true,
		Resident:               true,
	}
	return append(models, buildModelCapability(model, policy, hardwareOK, backendText, inferenceEnabled, runtimes))
}

func anyRunnableModel(models []ModelCapability) bool {
	for _, model := range models {
		if model.Runnable {
			return true
		}
	}
	return false
}

func modelMatchesRuntime(model modelcache.Model, runtime llamacpp.BackendRuntimeStatus) bool {
	if !runtime.Loaded {
		return false
	}
	candidates := []string{model.ModelID, model.Filename}
	for _, left := range candidates {
		for _, right := range []string{runtime.ModelID, runtime.ModelFilename} {
			if strings.EqualFold(strings.TrimSpace(left), strings.TrimSpace(right)) && strings.TrimSpace(left) != "" {
				return true
			}
		}
	}
	return strings.TrimSpace(model.Path) != "" && strings.TrimSpace(runtime.ModelPath) != "" && strings.TrimSpace(model.Path) == strings.TrimSpace(runtime.ModelPath)
}

func readinessReason(hardwareOK bool, backend BackendSummary, inferenceEnabled bool, anyRunnable bool, policy modelpolicy.Policy) string {
	switch {
	case !inferenceEnabled:
		return "v7_inference_disabled"
	case !hardwareOK:
		return "hardware_capacity_missing"
	case !backend.SupportsTextGeneration:
		return "backend_text_generation_unavailable"
	case anyRunnable:
		return "ready"
	case policy.AutoDownload:
		return "prepare_required"
	default:
		return "no_runnable_model"
	}
}

func hasHardwareCapacity(hardware v7hardware.CapacityInventory) bool {
	return strings.TrimSpace(hardware.OS) != "" &&
		strings.TrimSpace(hardware.Arch) != "" &&
		hardware.CPULogicalCores > 0 &&
		hardware.SystemRAMBytes > 0
}

func normalizeModels(models []ModelCapability) []ModelCapability {
	if len(models) == 0 {
		return []ModelCapability{}
	}
	out := make([]ModelCapability, 0, min(len(models), maxProfileModels))
	for _, model := range models {
		if len(out) >= maxProfileModels {
			break
		}
		model.ModelID = cleanProfileText(model.ModelID, maxProfileTextRunes)
		if model.ModelID == "" {
			continue
		}
		model.Family = cleanProfileText(strings.ToLower(model.Family), maxProfileTextRunes)
		if model.Family == "" {
			model.Family = "unknown"
		}
		model.Format = cleanProfileText(strings.ToLower(model.Format), maxProfileTextRunes)
		if model.Format == "" {
			model.Format = modelcache.DefaultFormat
		}
		if model.SizeBytes < 0 {
			model.SizeBytes = 0
		}
		model.Reason = cleanProfileText(model.Reason, maxProfileReasonRunes)
		model.BlockedReasons = cleanProfileList(model.BlockedReasons)
		out = append(out, model)
	}
	if len(out) == 0 {
		return []ModelCapability{}
	}
	return out
}

func cleanProfileList(values []string) []string {
	if len(values) == 0 {
		return []string{}
	}
	seen := map[string]struct{}{}
	out := make([]string, 0, min(len(values), 32))
	for _, value := range values {
		if len(out) >= 32 {
			break
		}
		value = cleanProfileText(value, maxProfileTextRunes)
		if value == "" {
			continue
		}
		key := strings.ToLower(value)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, value)
	}
	if out == nil {
		return []string{}
	}
	return out
}

func cleanProfileText(value string, maxRunes int) string {
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

func cloneStrings(values []string) []string {
	if len(values) == 0 {
		return []string{}
	}
	out := make([]string, len(values))
	copy(out, values)
	return out
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func uint64NonNegative(value int64) uint64 {
	if value <= 0 {
		return 0
	}
	return uint64(value)
}
