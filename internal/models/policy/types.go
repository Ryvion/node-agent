package modelpolicy

const (
	EnvModelAutoDownload           = "RYV_MODEL_AUTO_DOWNLOAD"
	EnvModelMaxSingleGB            = "RYV_MODEL_MAX_SINGLE_GB"
	EnvModelMaxCacheGB             = "RYV_MODEL_MAX_CACHE_GB"
	EnvModelCacheDir               = "RYV_MODEL_CACHE_DIR"
	EnvModelAllowedFamilies        = "RYV_MODEL_ALLOWED_FAMILIES"
	EnvModelAllowedFormats         = "RYV_MODEL_ALLOWED_FORMATS"
	EnvModelKeepWarmIDs            = "RYV_MODEL_KEEP_WARM_IDS"
	EnvModelEvictionPolicy         = "RYV_MODEL_EVICTION_POLICY"
	EnvModelAllowLicenseRestricted = "RYV_MODEL_ALLOW_LICENSE_RESTRICTED"
	EnvModelRuntimeMaxSingleGB     = "RYV_MODEL_RUNTIME_MAX_SINGLE_GB"
	EnvModelRuntimeMaxParamsB      = "RYV_MODEL_RUNTIME_MAX_PARAMS_B"
	EnvModelDenyIDs                = "RYV_MODEL_DENY_IDS"
	EnvModelAllowIDs               = "RYV_MODEL_ALLOW_IDS"
	EnvModelRuntimeAllowLarge      = "RYV_MODEL_RUNTIME_ALLOW_LARGE"
	EnvModelRequireExplicitLarge   = "RYV_MODEL_REQUIRE_EXPLICIT_ALLOW_LARGE"
	EnvModelMaxWarmModels          = "RYV_MODEL_MAX_WARM_MODELS"
	EnvModelMaxConcurrentInference = "RYV_MODEL_MAX_CONCURRENT_INFERENCE_JOBS"

	DefaultMaxSingleModelGB  = 8
	DefaultMaxCacheGB        = 50
	DefaultEvictionPolicy    = "lru"
	DefaultRuntimeMaxModelGB = 8
	DefaultRuntimeMaxParamsB = 8
	DefaultMaxWarmModels     = 1
	DefaultMaxConcurrentJobs = 1
)

var (
	DefaultAllowedFamilies        = []string{"llama", "phi", "qwen", "gemma", "gpt-oss"}
	DefaultAllowedFormats         = []string{"gguf"}
	DefaultRuntimeAllowedFamilies = []string{"llama"}
)

type Policy struct {
	AutoDownload           bool          `json:"auto_download"`
	MaxSingleModelBytes    uint64        `json:"max_single_model_bytes"`
	MaxCacheBytes          uint64        `json:"max_cache_bytes"`
	CacheDir               string        `json:"cache_dir"`
	AllowedFamilies        []string      `json:"allowed_families"`
	AllowedFormats         []string      `json:"allowed_formats"`
	KeepWarmModelIDs       []string      `json:"keep_warm_model_ids"`
	EvictionPolicy         string        `json:"eviction_policy"`
	AllowLicenseRestricted bool          `json:"allow_license_restricted"`
	RuntimePolicy          RuntimePolicy `json:"runtime_policy"`
}

type RuntimePolicy struct {
	AllowRuntimeExecution              bool     `json:"allow_runtime_execution"`
	MaxRuntimeModelBytes               uint64   `json:"max_runtime_model_bytes"`
	MaxRuntimeParameterCountBillions   float64  `json:"max_runtime_parameter_count_billions"`
	AllowCPUOffload                    bool     `json:"allow_cpu_offload"`
	AllowLargeModels                   bool     `json:"allow_large_models"`
	DenyModelIDs                       []string `json:"deny_model_ids"`
	AllowModelIDs                      []string `json:"allow_model_ids"`
	DenyFamilies                       []string `json:"deny_families"`
	AllowFamilies                      []string `json:"allow_families"`
	RequireExplicitAllowForLargeModels bool     `json:"require_explicit_allow_for_large_models"`
	MaxWarmModels                      int      `json:"max_warm_models"`
	MaxConcurrentInferenceJobs         int      `json:"max_concurrent_inference_jobs"`
}

type Status = Policy

type ConfigSource struct {
	Getenv      func(string) string
	UserHomeDir func() (string, error)
	GOOS        string
}
