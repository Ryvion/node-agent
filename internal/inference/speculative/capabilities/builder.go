package speculative

import (
	"os"
	"sort"
	"strings"
	"unicode"

	capshardware "github.com/Ryvion/ryvion-node/internal/capabilities/hardware"
	"github.com/Ryvion/ryvion-node/internal/inference/config"
	"github.com/Ryvion/ryvion-node/internal/models/cache"
	"github.com/Ryvion/ryvion-node/internal/models/policy"
	"github.com/Ryvion/ryvion-node/internal/runtimes/inventory"
	"github.com/Ryvion/ryvion-node/internal/runtimes/llamacpp"
	"github.com/Ryvion/ryvion-node/internal/runtimes/probe"
)

const (
	maxProfiles       = 16
	maxMethods        = 12
	maxAcceleration   = 8
	maxModelIDRunes   = 128
	maxReasonRunes    = 96
	maxBenchmarkState = 32
)

type BuildInput struct {
	Hardware         capshardware.CapacityInventory
	Policy           modelpolicy.Policy
	ModelCache       modelcache.Status
	BackendProbes    backendprobe.Probes
	BackendRuntimes  llamacpp.BackendRuntimes
	RuntimeInventory runtimeinventory.Inventory
	Getenv           func(string) string
}

type backendMethodSupport struct {
	Backend           string
	Available         bool
	SupportsStreaming bool
	Acceleration      []string
	Methods           []string
}

type targetModel struct {
	ModelID                string
	Family                 string
	Format                 string
	SizeBytes              int64
	ParameterCountBillions float64
	Resident               bool
	Warm                   bool
	Runnable               bool
	BlockedReasons         []string
}

func BuildReport(input BuildInput) Report {
	hardware := capshardware.NormalizeInventory(input.Hardware)
	policy := modelpolicy.NormalizePolicy(input.Policy)
	cache := modelcache.NormalizeStatus(input.ModelCache)
	probes := backendprobe.NormalizeProbes(input.BackendProbes)
	runtimes := llamacpp.NormalizeBackendRuntimes(input.BackendRuntimes)
	inventory := runtimeinventory.NormalizeInventory(input.RuntimeInventory)
	getenv := input.Getenv
	if getenv == nil {
		getenv = os.Getenv
	}

	hardwareOK := hasHardwareCapacity(hardware)
	backendText := backendTextGenerationAvailable(probes, runtimes, inventory)
	inferenceEnabled := inferenceconfig.V7InferenceEnabled(getenv)
	cache = modelcache.AnnotateRuntimeStatus(modelcache.RuntimeAnnotationInput{
		Status:                         cache,
		Policy:                         policy,
		HardwareCapacityAvailable:      hardwareOK,
		BackendTextGenerationAvailable: backendText,
		V7InferenceEnabled:             inferenceEnabled,
	})

	targets := buildTargetModels(cache, policy, hardwareOK, backendText, inferenceEnabled, runtimes)
	methods := buildBackendMethodSupport(probes, runtimes, inventory, hardware)
	profiles := make([]Profile, 0, maxProfiles)
	for _, backend := range methods {
		if !backend.Available {
			continue
		}
		for _, target := range targets {
			if len(profiles) >= maxProfiles {
				break
			}
			for _, method := range backend.Methods {
				if len(profiles) >= maxProfiles {
					break
				}
				switch method {
				case MethodNgram:
					profiles = append(profiles, buildNgramProfile(target, backend, getenv))
				case MethodDraftModel:
					profiles = append(profiles, buildDraftModelProfile(target, targets, backend, policy, getenv))
				default:
					profiles = append(profiles, buildBackendLocalProfile(target, backend, method, getenv))
				}
			}
		}
	}
	profiles = NormalizeProfiles(profiles)
	return Report{
		SpeculativeDecoding: BuildCapabilityFromProfiles(profiles),
		SpeculativeProfiles: profiles,
	}
}

func BuildCapabilityFromProfiles(profiles []Profile) DecodingCapability {
	profiles = NormalizeProfiles(profiles)
	methods := make([]string, 0, maxMethods)
	seen := map[string]struct{}{}
	supported := false
	enabled := false
	supportsStreaming := false
	supportsLossless := false
	maxTokens := 0
	defaultMethod := ""
	for _, profile := range profiles {
		potential := profile.Runnable || onlyOptOutBlocked(profile.BlockedReasons)
		if !potential {
			continue
		}
		supported = true
		if _, ok := seen[profile.Method]; !ok && len(methods) < maxMethods {
			seen[profile.Method] = struct{}{}
			methods = append(methods, profile.Method)
		}
		if profile.Runnable {
			enabled = true
			if defaultMethod == "" {
				defaultMethod = profile.Method
			}
			supportsStreaming = supportsStreaming || backendMethodSupportsStreaming(profile.Backend)
			supportsLossless = true
			if tokens := maxTokensForMethod(profile.Method); tokens > maxTokens {
				maxTokens = tokens
			}
		}
	}
	return NormalizeCapability(DecodingCapability{
		Supported:                    supported,
		Enabled:                      enabled,
		Methods:                      methods,
		DefaultMethod:                defaultMethod,
		SupportsStreaming:            supportsStreaming,
		SupportsLosslessVerification: supportsLossless,
		MaxSpeculativeTokens:         maxTokens,
	})
}

func NormalizeCapability(capability DecodingCapability) DecodingCapability {
	capability.Methods = normalizeMethods(capability.Methods)
	capability.DefaultMethod = normalizeMethod(capability.DefaultMethod)
	if !containsString(capability.Methods, capability.DefaultMethod) {
		capability.DefaultMethod = ""
	}
	if capability.MaxSpeculativeTokens < 0 {
		capability.MaxSpeculativeTokens = 0
	}
	if !capability.Enabled {
		capability.DefaultMethod = ""
		capability.SupportsStreaming = false
		capability.SupportsLosslessVerification = false
		capability.MaxSpeculativeTokens = 0
	}
	if len(capability.Methods) == 0 {
		capability.Supported = false
		capability.Enabled = false
	}
	if !capability.Supported {
		capability.Enabled = false
	}
	return capability
}

func NormalizeProfiles(profiles []Profile) []Profile {
	if len(profiles) == 0 {
		return []Profile{}
	}
	out := make([]Profile, 0, min(len(profiles), maxProfiles))
	for _, profile := range profiles {
		if len(out) >= maxProfiles {
			break
		}
		profile.TargetModelID = cleanText(profile.TargetModelID, maxModelIDRunes)
		profile.DraftModelID = cleanText(profile.DraftModelID, maxModelIDRunes)
		profile.Method = normalizeMethod(profile.Method)
		profile.Backend = normalizeBackend(profile.Backend)
		if profile.TargetModelID == "" || profile.Method == "" || profile.Backend == "" {
			continue
		}
		profile.Acceleration = normalizeAcceleration(profile.Acceleration)
		profile.BlockedReasons = normalizeReasons(profile.BlockedReasons)
		profile.Runnable = len(profile.BlockedReasons) == 0 && profile.Runnable
		if profile.Method != MethodDraftModel {
			profile.DraftModelID = ""
			profile.DraftResident = false
			profile.WarmPair = profile.TargetResident && profile.WarmPair
		}
		profile.LastBenchmark = normalizeBenchmark(profile.LastBenchmark)
		out = append(out, profile)
	}
	if len(out) == 0 {
		return []Profile{}
	}
	return out
}

func buildNgramProfile(target targetModel, backend backendMethodSupport, getenv func(string) string) Profile {
	reasons := cloneStrings(target.BlockedReasons)
	if envBool(getenv(EnvDisableSpeculativeDecoding)) {
		reasons = append(reasons, "speculative_decoding_disabled")
	}
	if envBool(getenv(EnvDisableNgramSpeculation)) {
		reasons = append(reasons, "ngram_speculation_disabled")
	}
	return normalizeProfile(Profile{
		TargetModelID:       target.ModelID,
		Method:              MethodNgram,
		Backend:             backend.Backend,
		Acceleration:        backend.Acceleration,
		Runnable:            len(reasons) == 0,
		TargetResident:      target.Resident,
		WarmPair:            target.Warm,
		TokenizerCompatible: true,
		MemoryEstimateBytes: uint64NonNegative(target.SizeBytes),
		BlockedReasons:      reasons,
		LastBenchmark:       BenchmarkSummary{Status: BenchmarkStatusNotRun},
	})
}

func buildDraftModelProfile(target targetModel, targets []targetModel, backend backendMethodSupport, policy modelpolicy.Policy, getenv func(string) string) Profile {
	draft, ok := compatibleDraftModel(target, targets)
	reasons := cloneStrings(target.BlockedReasons)
	if envBool(getenv(EnvDisableSpeculativeDecoding)) {
		reasons = append(reasons, "speculative_decoding_disabled")
	}
	if envBool(getenv(EnvDisableDraftModels)) {
		reasons = append(reasons, "draft_models_disabled")
	}
	memoryEstimate := uint64NonNegative(target.SizeBytes)
	tokenizerCompatible := false
	if !ok {
		reasons = append(reasons, "compatible_draft_model_not_found")
	} else {
		memoryEstimate += uint64NonNegative(draft.SizeBytes)
		tokenizerCompatible = true
		for _, reason := range draft.BlockedReasons {
			reasons = append(reasons, "draft_"+reason)
		}
		if capBytes := policy.RuntimePolicy.MaxRuntimeModelBytes; capBytes > 0 && memoryEstimate > capBytes {
			reasons = append(reasons, "speculative_pair_memory_exceeds_policy")
		}
	}
	profile := Profile{
		TargetModelID:       target.ModelID,
		Method:              MethodDraftModel,
		Backend:             backend.Backend,
		Acceleration:        backend.Acceleration,
		Runnable:            len(reasons) == 0,
		TargetResident:      target.Resident,
		DraftResident:       ok && draft.Resident,
		WarmPair:            ok && target.Warm && draft.Warm,
		TokenizerCompatible: tokenizerCompatible,
		MemoryEstimateBytes: memoryEstimate,
		BlockedReasons:      reasons,
		LastBenchmark:       BenchmarkSummary{Status: BenchmarkStatusNotRun},
	}
	if ok {
		profile.DraftModelID = draft.ModelID
	}
	return normalizeProfile(profile)
}

func buildBackendLocalProfile(target targetModel, backend backendMethodSupport, method string, getenv func(string) string) Profile {
	reasons := cloneStrings(target.BlockedReasons)
	if envBool(getenv(EnvDisableSpeculativeDecoding)) {
		reasons = append(reasons, "speculative_decoding_disabled")
	}
	return normalizeProfile(Profile{
		TargetModelID:       target.ModelID,
		Method:              method,
		Backend:             backend.Backend,
		Acceleration:        backend.Acceleration,
		Runnable:            len(reasons) == 0,
		TargetResident:      target.Resident,
		WarmPair:            target.Warm,
		TokenizerCompatible: true,
		MemoryEstimateBytes: uint64NonNegative(target.SizeBytes),
		BlockedReasons:      reasons,
		LastBenchmark:       BenchmarkSummary{Status: BenchmarkStatusNotRun},
	})
}

func buildTargetModels(cache modelcache.Status, policy modelpolicy.Policy, hardwareOK bool, backendText bool, inferenceEnabled bool, runtimes llamacpp.BackendRuntimes) []targetModel {
	out := make([]targetModel, 0, min(len(cache.Models)+1, maxProfiles))
	seen := map[string]struct{}{}
	for _, model := range cache.Models {
		target := targetFromCacheModel(model)
		if runtimeMatchesTarget(runtimes.LlamaCPP, target) {
			target.Warm = runtimes.LlamaCPP.Warm
		}
		if target.ModelID == "" {
			continue
		}
		out = appendTargetModel(out, seen, target)
	}
	runtime := runtimes.LlamaCPP
	if runtime.Loaded && runtime.ModelID != "" {
		modelID := firstNonEmpty(runtime.ModelID, runtime.ModelFilename)
		if _, ok := seen[strings.ToLower(modelID)]; !ok {
			target := targetModel{
				ModelID:        modelID,
				Family:         normalizeFamily(runtime.ModelFamilyHint),
				Format:         "gguf",
				SizeBytes:      runtime.ModelSizeBytes,
				Resident:       true,
				Warm:           runtime.Warm,
				Runnable:       true,
				BlockedReasons: []string{},
			}
			target.BlockedReasons = runtimeTargetBlockedReasons(target, policy, hardwareOK, backendText, inferenceEnabled)
			target.Runnable = len(target.BlockedReasons) == 0
			out = appendTargetModel(out, seen, target)
		}
	}
	if out == nil {
		return []targetModel{}
	}
	return out
}

func targetFromCacheModel(model modelcache.Model) targetModel {
	reasons := cloneStrings(model.BlockedReasons)
	if !model.Runnable && len(reasons) == 0 {
		reasons = append(reasons, "model_not_runnable")
	}
	return targetModel{
		ModelID:                firstNonEmpty(model.ModelID, model.Filename),
		Family:                 normalizeFamily(model.FamilyHint),
		Format:                 strings.ToLower(strings.TrimSpace(model.Format)),
		SizeBytes:              model.SizeBytes,
		ParameterCountBillions: model.ParameterCountBillions,
		Resident:               model.Resident || model.Installed,
		Warm:                   false,
		Runnable:               model.Runnable && len(reasons) == 0,
		BlockedReasons:         reasons,
	}
}

func runtimeMatchesTarget(runtime llamacpp.BackendRuntimeStatus, target targetModel) bool {
	if !runtime.Loaded || target.ModelID == "" {
		return false
	}
	return strings.EqualFold(runtime.ModelID, target.ModelID) ||
		strings.EqualFold(runtime.ModelFilename, target.ModelID) ||
		strings.EqualFold(runtime.WarmModelID, target.ModelID)
}

func runtimeTargetBlockedReasons(target targetModel, policy modelpolicy.Policy, hardwareOK bool, backendText bool, inferenceEnabled bool) []string {
	reasons := make([]string, 0, 4)
	if !inferenceEnabled {
		reasons = append(reasons, "v7_inference_disabled")
	}
	if !hardwareOK {
		reasons = append(reasons, "hardware_capacity_missing")
	}
	if !backendText {
		reasons = append(reasons, "backend_text_generation_unavailable")
	}
	decision := modelpolicy.EvaluateRuntimeRequest(policy, modelpolicy.RuntimeRequest{
		ModelID:                target.ModelID,
		ModelSizeBytes:         uint64NonNegative(target.SizeBytes),
		ParameterCountBillions: target.ParameterCountBillions,
		Family:                 target.Family,
	})
	if !decision.Allowed {
		reasons = append(reasons, decision.Reason)
	}
	return normalizeReasons(reasons)
}

func appendTargetModel(out []targetModel, seen map[string]struct{}, target targetModel) []targetModel {
	target.ModelID = cleanText(target.ModelID, maxModelIDRunes)
	target.Family = normalizeFamily(target.Family)
	target.Format = strings.ToLower(strings.TrimSpace(target.Format))
	target.BlockedReasons = normalizeReasons(target.BlockedReasons)
	target.Runnable = target.Runnable && len(target.BlockedReasons) == 0
	if target.ModelID == "" {
		return out
	}
	key := strings.ToLower(target.ModelID)
	if _, ok := seen[key]; ok {
		return out
	}
	seen[key] = struct{}{}
	return append(out, target)
}

func compatibleDraftModel(target targetModel, targets []targetModel) (targetModel, bool) {
	if target.Family == "" || target.Family == "unknown" || target.SizeBytes <= 0 {
		return targetModel{}, false
	}
	candidates := make([]targetModel, 0, len(targets))
	for _, draft := range targets {
		if strings.EqualFold(draft.ModelID, target.ModelID) {
			continue
		}
		if draft.Family != target.Family || draft.SizeBytes <= 0 || draft.SizeBytes >= target.SizeBytes {
			continue
		}
		candidates = append(candidates, draft)
	}
	if len(candidates) == 0 {
		return targetModel{}, false
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		return candidates[i].SizeBytes < candidates[j].SizeBytes
	})
	return candidates[0], true
}

func buildBackendMethodSupport(probes backendprobe.Probes, runtimes llamacpp.BackendRuntimes, inventory runtimeinventory.Inventory, hardware capshardware.CapacityInventory) []backendMethodSupport {
	out := []backendMethodSupport{}
	if support := llamaMethodSupport(probes, runtimes, inventory, hardware); support.Available {
		out = append(out, support)
	}
	for _, support := range []backendMethodSupport{
		runtimeMethodSupport(runtimes.VLLM, runtimeinventory.BackendCandidateVLLM, []string{MethodNativeMTP, MethodDraftModel, MethodEAGLE, MethodMedusa}),
		runtimeMethodSupport(runtimes.SGLang, runtimeinventory.BackendCandidateSGLang, []string{MethodNativeMTP, MethodEAGLE, MethodEAGLE3}),
		runtimeMethodSupport(runtimes.TensorRTLLM, runtimeinventory.BackendCandidateTensorRTLLM, []string{MethodDraftModel, MethodMedusa, MethodReDrafter, MethodEAGLE}),
	} {
		if support.Available && len(support.Methods) > 0 {
			out = append(out, support)
		}
	}
	return out
}

func llamaMethodSupport(probes backendprobe.Probes, runtimes llamacpp.BackendRuntimes, inventory runtimeinventory.Inventory, hardware capshardware.CapacityInventory) backendMethodSupport {
	runtime := runtimes.LlamaCPP
	probe := probes.LlamaCPP
	candidate := backendCandidate(inventory, runtimeinventory.BackendCandidateLlamaCPP)
	available := (runtime.Available && runtime.SupportsTextGeneration) ||
		(probe.Available && probe.SupportsTextGeneration) ||
		(candidate.Detected && candidate.SupportsTextGeneration)
	streaming := (runtime.Available && runtime.SupportsStreaming) ||
		(probe.Available && probe.SupportsStreaming) ||
		(candidate.Detected && candidate.SupportsStreaming)
	serverAvailable := runtime.OpenAICompatible || probe.SupportsOpenAICompatibleServer ||
		strings.TrimSpace(candidate.ServerBinaryPath) != "" || candidate.SupportsOpenAICompatibleServer
	methods := []string{}
	if available && serverAvailable {
		methods = append(methods, MethodNgram, MethodDraftModel)
		if runtimeOptimizationSupports(runtime, MethodNativeMTP) {
			methods = append(methods, MethodNativeMTP)
		}
	}
	return backendMethodSupport{
		Backend:           llamacpp.BackendName,
		Available:         available && serverAvailable,
		SupportsStreaming: streaming,
		Acceleration:      firstNonEmptyAcceleration(runtime.Acceleration, hardware.AccelerationHints, []string{"cpu"}),
		Methods:           methods,
	}
}

func runtimeMethodSupport(runtime llamacpp.BackendRuntimeStatus, backend string, possibleMethods []string) backendMethodSupport {
	available := runtime.Available && runtime.SupportsTextGeneration
	methods := make([]string, 0, len(possibleMethods))
	for _, method := range possibleMethods {
		if runtimeOptimizationSupports(runtime, method) {
			methods = append(methods, method)
		}
	}
	return backendMethodSupport{
		Backend:           backend,
		Available:         available,
		SupportsStreaming: runtime.SupportsStreaming,
		Acceleration:      runtime.Acceleration,
		Methods:           methods,
	}
}

func backendCandidate(inventory runtimeinventory.Inventory, backend string) runtimeinventory.BackendCandidate {
	for _, candidate := range inventory.BackendCandidates {
		if candidate.Backend == backend {
			return candidate
		}
	}
	return runtimeinventory.BackendCandidate{}
}

func backendTextGenerationAvailable(probes backendprobe.Probes, runtimes llamacpp.BackendRuntimes, inventory runtimeinventory.Inventory) bool {
	if runtimes.LlamaCPP.Available && runtimes.LlamaCPP.SupportsTextGeneration {
		return true
	}
	if probes.LlamaCPP.Available && probes.LlamaCPP.SupportsTextGeneration {
		return true
	}
	for _, candidate := range inventory.BackendCandidates {
		if candidate.Detected && candidate.SupportsTextGeneration {
			return true
		}
	}
	return false
}

func runtimeOptimizationSupports(runtime llamacpp.BackendRuntimeStatus, method string) bool {
	method = normalizeMethod(method)
	for _, capability := range runtime.OptimizationCapabilities {
		if !capability.Supported {
			continue
		}
		name := normalizeOptimizationName(capability.Name)
		switch method {
		case MethodNgram:
			if name == "ngram" || name == "speculative_decode" {
				return true
			}
		case MethodDraftModel:
			if name == "draft_model" || name == "draft_target" || name == "speculative_decode" {
				return true
			}
		default:
			if name == method {
				return true
			}
		}
	}
	return false
}

func normalizeProfile(profile Profile) Profile {
	profiles := NormalizeProfiles([]Profile{profile})
	if len(profiles) == 0 {
		return Profile{}
	}
	return profiles[0]
}

func hasHardwareCapacity(hardware capshardware.CapacityInventory) bool {
	return strings.TrimSpace(hardware.OS) != "" &&
		strings.TrimSpace(hardware.Arch) != "" &&
		hardware.CPULogicalCores > 0 &&
		hardware.SystemRAMBytes > 0
}

func firstNonEmptyAcceleration(values ...[]string) []string {
	for _, value := range values {
		normalized := normalizeAcceleration(value)
		if len(normalized) > 0 {
			return normalized
		}
	}
	return []string{}
}

func backendMethodSupportsStreaming(backend string) bool {
	switch normalizeBackend(backend) {
	case llamacpp.BackendName, runtimeinventory.BackendCandidateVLLM, runtimeinventory.BackendCandidateSGLang:
		return true
	default:
		return false
	}
}

func maxTokensForMethod(method string) int {
	switch normalizeMethod(method) {
	case MethodNgram:
		return 8
	case MethodDraftModel, MethodNativeMTP, MethodEAGLE, MethodEAGLE3, MethodMedusa, MethodReDrafter:
		return 16
	default:
		return 0
	}
}

func onlyOptOutBlocked(reasons []string) bool {
	reasons = normalizeReasons(reasons)
	if len(reasons) == 0 {
		return false
	}
	for _, reason := range reasons {
		switch reason {
		case "speculative_decoding_disabled", "draft_models_disabled", "ngram_speculation_disabled":
		default:
			return false
		}
	}
	return true
}

func envBool(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "yes", "on", "enabled":
		return true
	default:
		return false
	}
}

func normalizeMethods(methods []string) []string {
	if len(methods) == 0 {
		return []string{}
	}
	seen := map[string]struct{}{}
	out := make([]string, 0, min(len(methods), maxMethods))
	for _, method := range methods {
		method = normalizeMethod(method)
		if method == "" {
			continue
		}
		if _, ok := seen[method]; ok {
			continue
		}
		seen[method] = struct{}{}
		out = append(out, method)
		if len(out) >= maxMethods {
			break
		}
	}
	if len(out) == 0 {
		return []string{}
	}
	return out
}

func normalizeMethod(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	value = strings.ReplaceAll(value, "-", "_")
	value = strings.ReplaceAll(value, " ", "_")
	switch value {
	case MethodNgram, "ngram_speculation":
		return MethodNgram
	case MethodDraftModel, "draft_target", "draft_model_speculation":
		return MethodDraftModel
	case MethodNativeMTP, "mtp":
		return MethodNativeMTP
	case MethodEAGLE:
		return MethodEAGLE
	case MethodEAGLE3:
		return MethodEAGLE3
	case MethodMedusa:
		return MethodMedusa
	case MethodReDrafter:
		return MethodReDrafter
	default:
		return ""
	}
}

func normalizeOptimizationName(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	value = strings.ReplaceAll(value, "-", "_")
	value = strings.ReplaceAll(value, " ", "_")
	switch value {
	case "speculative_decode", "draft_target":
		return value
	default:
		return normalizeMethod(value)
	}
}

func normalizeBackend(value string) string {
	value = strings.TrimSpace(value)
	switch value {
	case llamacpp.BackendName, runtimeinventory.BackendCandidateTensorRTLLM, runtimeinventory.BackendCandidateVLLM, runtimeinventory.BackendCandidateSGLang:
		return value
	default:
		return cleanText(value, 64)
	}
}

func normalizeAcceleration(values []string) []string {
	if len(values) == 0 {
		return []string{}
	}
	seen := map[string]struct{}{}
	out := make([]string, 0, min(len(values), maxAcceleration))
	for _, value := range values {
		value = strings.ToLower(cleanText(value, 32))
		switch value {
		case "cpu", "cuda", "vulkan", "directml", "metal", "rocm", "other":
		default:
			value = "other"
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
		if len(out) >= maxAcceleration {
			break
		}
	}
	if len(out) == 0 {
		return []string{}
	}
	return out
}

func normalizeBenchmark(summary BenchmarkSummary) BenchmarkSummary {
	summary.Status = cleanText(strings.ToLower(strings.TrimSpace(summary.Status)), maxBenchmarkState)
	if summary.Status == "" {
		summary.Status = BenchmarkStatusNotRun
	}
	if summary.DecodeTokensPerSecond < 0 {
		summary.DecodeTokensPerSecond = 0
	}
	if summary.EndToEndTokensPerSecond < 0 {
		summary.EndToEndTokensPerSecond = 0
	}
	if summary.ImprovementRatio < 0 {
		summary.ImprovementRatio = 0
	}
	if summary.UpdatedAtUnixMs < 0 {
		summary.UpdatedAtUnixMs = 0
	}
	return summary
}

func normalizeFamily(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "llama", "phi", "qwen", "gemma", "gpt-oss", "gptoss":
		if strings.ToLower(strings.TrimSpace(value)) == "gptoss" {
			return "gpt-oss"
		}
		return strings.ToLower(strings.TrimSpace(value))
	default:
		return "unknown"
	}
}

func normalizeReasons(reasons []string) []string {
	if len(reasons) == 0 {
		return []string{}
	}
	seen := map[string]struct{}{}
	out := make([]string, 0, min(len(reasons), 8))
	for _, reason := range reasons {
		reason = strings.ToLower(cleanText(reason, maxReasonRunes))
		reason = strings.ReplaceAll(reason, "-", "_")
		reason = strings.ReplaceAll(reason, " ", "_")
		if reason == "" || reason == "allowed" || reason == "runnable" {
			continue
		}
		if _, ok := seen[reason]; ok {
			continue
		}
		seen[reason] = struct{}{}
		out = append(out, reason)
		if len(out) >= 8 {
			break
		}
	}
	if len(out) == 0 {
		return []string{}
	}
	return out
}

func cleanText(value string, maxRunes int) string {
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

func containsString(values []string, value string) bool {
	value = strings.TrimSpace(value)
	if value == "" {
		return false
	}
	for _, item := range values {
		if item == value {
			return true
		}
	}
	return false
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			return value
		}
	}
	return ""
}

func cloneStrings(values []string) []string {
	if len(values) == 0 {
		return []string{}
	}
	out := make([]string, len(values))
	copy(out, values)
	return out
}

func uint64NonNegative(value int64) uint64 {
	if value <= 0 {
		return 0
	}
	return uint64(value)
}
