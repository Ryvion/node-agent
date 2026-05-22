package llamacpp

import "time"

const (
	BackendName = "llama.cpp"

	EnvEnabled      = "RYV_LLAMA_CPP_ENABLED"
	EnvServer       = "RYV_LLAMA_CPP_SERVER_PATH"
	EnvModel        = "RYV_LLAMA_CPP_MODEL_PATH"
	EnvHost         = "RYV_LLAMA_CPP_HOST"
	EnvPort         = "RYV_LLAMA_CPP_PORT"
	EnvCtxSize      = "RYV_LLAMA_CPP_CTX_SIZE"
	EnvThreads      = "RYV_LLAMA_CPP_THREADS"
	EnvGPULayers    = "RYV_LLAMA_CPP_GPU_LAYERS"
	EnvExtraArgs    = "RYV_LLAMA_CPP_EXTRA_ARGS"
	EnvFastDefaults = "RYV_LLAMA_CPP_FAST_DEFAULTS"

	// V8 speculative decoding (Level 0 - backend-local).
	// When DraftModelPath is set, llama-server runs the target+draft pair
	// and produces speculative-accelerated tokens via llama.cpp's built-in
	// draft/target loop. Typical pairing on Ryvion native models:
	// target=ryvion-llama-3.2-3b, draft=tinyllama-1.1b (Llama family,
	// tokenizer-compatible).
	EnvDraftAuto      = "RYV_LLAMA_CPP_AUTO_DRAFT"
	EnvDraftModel     = "RYV_LLAMA_CPP_DRAFT_MODEL_PATH"
	EnvDraftMaxTokens = "RYV_LLAMA_CPP_DRAFT_MAX_TOKENS"
	EnvDraftMinTokens = "RYV_LLAMA_CPP_DRAFT_MIN_TOKENS"
	EnvDraftPMin      = "RYV_LLAMA_CPP_DRAFT_P_MIN"
	EnvDraftGPULayers = "RYV_LLAMA_CPP_DRAFT_GPU_LAYERS"
	EnvNativeMTP      = "RYV_LLAMA_CPP_NATIVE_MTP"
	EnvSpecType       = "RYV_LLAMA_CPP_SPEC_TYPE"
	EnvAutoNGram      = "RYV_LLAMA_CPP_AUTO_NGRAM"
	EnvNGramMaxTokens = "RYV_LLAMA_CPP_NGRAM_MAX_TOKENS"

	SpeculativeMethodBackendLocalDraft = "backend_local_draft_model"
	SpeculativeMethodNativeMTP         = "native_mtp"
	SpeculativeMethodNGramSimple       = "ngram-simple"
	SpeculativeMethodNGramMapK         = "ngram-map-k"
	SpeculativeMethodNGramMapK4V       = "ngram-map-k4v"
	SpeculativeMethodNGramMod          = "ngram-mod"
	SpeculativeMethodNGramCache        = "ngram-cache"

	DefaultHost               = "127.0.0.1"
	DefaultPort               = 45910
	DefaultContextSize        = 4096
	DefaultGPULayers          = 999
	DefaultDraftMaxTokens     = 16
	DefaultNativeMTPMaxTokens = 3
	DefaultNGramMaxTokens     = 16
	DefaultDraftMinTokens     = 0
	DefaultDraftPMinMillis    = 0 // 0 = use llama.cpp default (0.75)

	LaunchProfileDefault     = "default"
	LaunchProfileCUDAFast    = "cuda_fast"
	LaunchProfileCUDASafe    = "cuda_safe"
	LaunchProfileCUDAPartial = "cuda_partial"
)

type LlamaCppSidecarConfig struct {
	Enabled              bool
	ServerPath           string
	ServerPathExplicit   bool
	ModelPath            string
	Host                 string
	Port                 int
	ContextSize          int
	Threads              int
	GPULayers            int
	GPULayersExplicit    bool
	ExtraArgs            []string
	FastDefaults         bool
	FastDefaultsExplicit bool
	LaunchProfile        string
	AccelerationHints    []string

	// V8 speculative decoding (Level 0).
	// DraftModelPath, when non-empty, enables backend-local speculative
	// decoding by passing --model-draft to llama-server.
	DraftModelPath string
	// NativeMTP enables llama.cpp's native MTP-head speculative path via
	// --spec-type draft-mtp. It does not require a separate draft model.
	NativeMTP      bool
	NativeMTPAuto  bool
	SpecType       string
	DraftMaxTokens int     // --spec-draft-n-max (default 16 when DraftModelPath set)
	DraftMinTokens int     // --spec-draft-n-min
	DraftPMin      float64 // --draft-p-min (0 = llama.cpp default 0.75)
	DraftGPULayers int     // --n-gpu-layers-draft (0 = auto)
}

type LlamaCppSidecarStatus struct {
	Enabled                bool                      `json:"enabled"`
	Available              bool                      `json:"available"`
	Running                bool                      `json:"running"`
	Healthy                bool                      `json:"healthy"`
	Attached               bool                      `json:"attached,omitempty"`
	PID                    int                       `json:"pid,omitempty"`
	BaseURL                string                    `json:"base_url"`
	ServerPath             string                    `json:"server_path,omitempty"`
	ModelPath              string                    `json:"model_path,omitempty"`
	ContextSize            int                       `json:"context_size,omitempty"`
	ModelFilename          string                    `json:"model_filename,omitempty"`
	ModelSizeBytes         int64                     `json:"model_size_bytes"`
	ModelFamilyHint        string                    `json:"model_family_hint,omitempty"`
	QuantizationHint       string                    `json:"quantization_hint,omitempty"`
	StartedAt              time.Time                 `json:"started_at,omitempty"`
	LastHealthAt           time.Time                 `json:"last_health_at,omitempty"`
	LastError              string                    `json:"last_error"`
	Backend                string                    `json:"backend"`
	Launch                 *LlamaCppLaunchConfig     `json:"launch,omitempty"`
	ServerProperties       *LlamaCppServerProperties `json:"server_properties,omitempty"`
	Acceleration           []string                  `json:"acceleration"`
	AccelerationReason     string                    `json:"acceleration_reason,omitempty"`
	OpenAICompatible       bool                      `json:"openai_compatible"`
	SupportsTextGeneration bool                      `json:"supports_text_generation"`
	SupportsStreaming      bool                      `json:"supports_streaming"`
	SupportsKVAccess       bool                      `json:"supports_kv_access"`
	SupportsTensorHooks    bool                      `json:"supports_tensor_hooks"`
	Reason                 string                    `json:"reason"`

	// V8 speculative decoding (Level 0 - backend-local).
	SpeculativeEnabled   bool   `json:"speculative_enabled"`
	SpeculativeActive    bool   `json:"speculative_active"`
	SpeculativeMethod    string `json:"speculative_method,omitempty"`
	NativeMTP            bool   `json:"native_mtp,omitempty"`
	DraftModelPath       string `json:"draft_model_path,omitempty"`
	DraftModelFilename   string `json:"draft_model_filename,omitempty"`
	DraftModelSizeBytes  int64  `json:"draft_model_size_bytes,omitempty"`
	DraftModelFamilyHint string `json:"draft_model_family_hint,omitempty"`
	DraftMaxTokens       int    `json:"draft_max_tokens,omitempty"`
	DraftMinTokens       int    `json:"draft_min_tokens,omitempty"`
}

type LlamaCppLaunchConfig struct {
	Mode                     string `json:"mode,omitempty"`
	Managed                  bool   `json:"managed"`
	Attached                 bool   `json:"attached"`
	ServerPath               string `json:"server_path,omitempty"`
	ServerFilename           string `json:"server_filename,omitempty"`
	ConfiguredGPULayers      int    `json:"configured_gpu_layers"`
	FastDefaultsEnabled      bool   `json:"fast_defaults_enabled"`
	ConfiguredDraftGPULayers int    `json:"configured_draft_gpu_layers,omitempty"`
	Profile                  string `json:"profile,omitempty"`
}

type LlamaCppServerProperties struct {
	BuildInfo            string   `json:"build_info,omitempty"`
	SystemInfo           string   `json:"system_info,omitempty"`
	ReportedGPULayers    int      `json:"reported_gpu_layers,omitempty"`
	ReportedAcceleration []string `json:"reported_acceleration,omitempty"`
}

type BackendRuntimes struct {
	LlamaCPP    BackendRuntimeStatus   `json:"llama_cpp"`
	TensorRTLLM BackendRuntimeStatus   `json:"tensorrt_llm"`
	VLLM        BackendRuntimeStatus   `json:"vllm"`
	SGLang      BackendRuntimeStatus   `json:"sglang"`
	Other       []BackendRuntimeStatus `json:"other"`
}

type BackendRuntimeStatus struct {
	Enabled                  bool                      `json:"enabled"`
	Available                bool                      `json:"available"`
	Running                  bool                      `json:"running"`
	Healthy                  bool                      `json:"healthy"`
	Health                   string                    `json:"health"`
	Backend                  string                    `json:"backend"`
	BaseURL                  string                    `json:"base_url"`
	ModelID                  string                    `json:"model_id,omitempty"`
	ModelPath                string                    `json:"model_path,omitempty"`
	ModelFilename            string                    `json:"model_filename,omitempty"`
	ModelSizeBytes           int64                     `json:"model_size_bytes"`
	ModelFamilyHint          string                    `json:"model_family_hint,omitempty"`
	QuantizationHint         string                    `json:"quantization_hint,omitempty"`
	Loaded                   bool                      `json:"loaded"`
	Warm                     bool                      `json:"warm"`
	WarmModelID              string                    `json:"warm_model_id,omitempty"`
	Acceleration             []string                  `json:"acceleration"`
	AccelerationReason       string                    `json:"acceleration_reason,omitempty"`
	GPUArchitecture          string                    `json:"gpu_architecture,omitempty"`
	GPUComputeCapability     string                    `json:"gpu_compute_capability,omitempty"`
	OpenAICompatible         bool                      `json:"openai_compatible"`
	SupportsTextGeneration   bool                      `json:"supports_text_generation"`
	SupportsStreaming        bool                      `json:"supports_streaming"`
	SupportsStatefulSessions bool                      `json:"supports_stateful_sessions"`
	SupportsKVAccess         bool                      `json:"supports_kv_access"`
	SupportsKVHooks          bool                      `json:"supports_kv_hooks"`
	SupportsTensorHooks      bool                      `json:"supports_tensor_hooks"`
	SupportsDistributedKV    bool                      `json:"supports_distributed_kv"`
	OptimizationCapabilities []OptimizationCapability  `json:"optimization_capabilities,omitempty"`
	MaxContextTokens         int                       `json:"max_context_tokens,omitempty"`
	LastHealthAtUnixMs       int64                     `json:"last_health_at_unix_ms"`
	LastError                string                    `json:"last_error"`
	Launch                   *LlamaCppLaunchConfig     `json:"launch,omitempty"`
	ServerProperties         *LlamaCppServerProperties `json:"server_properties,omitempty"`
}

type OptimizationCapability struct {
	Name              string `json:"name"`
	Supported         bool   `json:"supported"`
	Enabled           bool   `json:"enabled"`
	Backend           string `json:"backend,omitempty"`
	RequiresAttention string `json:"requires_attention,omitempty"`
	RequiresGPUArch   string `json:"requires_gpu_arch,omitempty"`
	ContextMinTokens  int    `json:"context_min_tokens,omitempty"`
	Notes             string `json:"notes,omitempty"`
}

type HealthResult struct {
	Healthy   bool
	Endpoint  string
	CheckedAt time.Time
	Error     string
}
