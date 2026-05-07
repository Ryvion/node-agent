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

	DefaultMaxSingleModelGB = 8
	DefaultMaxCacheGB       = 50
	DefaultEvictionPolicy   = "lru"
)

var (
	DefaultAllowedFamilies = []string{"llama", "phi", "qwen", "gemma"}
	DefaultAllowedFormats  = []string{"gguf"}
)

type Policy struct {
	AutoDownload           bool     `json:"auto_download"`
	MaxSingleModelBytes    uint64   `json:"max_single_model_bytes"`
	MaxCacheBytes          uint64   `json:"max_cache_bytes"`
	CacheDir               string   `json:"cache_dir"`
	AllowedFamilies        []string `json:"allowed_families"`
	AllowedFormats         []string `json:"allowed_formats"`
	KeepWarmModelIDs       []string `json:"keep_warm_model_ids"`
	EvictionPolicy         string   `json:"eviction_policy"`
	AllowLicenseRestricted bool     `json:"allow_license_restricted"`
}

type Status = Policy

type ConfigSource struct {
	Getenv      func(string) string
	UserHomeDir func() (string, error)
	GOOS        string
}
