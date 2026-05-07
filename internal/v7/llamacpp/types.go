package llamacpp

import "time"

const (
	BackendName = "llama.cpp"

	EnvEnabled   = "RYV_LLAMA_CPP_ENABLED"
	EnvServer    = "RYV_LLAMA_CPP_SERVER_PATH"
	EnvModel     = "RYV_LLAMA_CPP_MODEL_PATH"
	EnvHost      = "RYV_LLAMA_CPP_HOST"
	EnvPort      = "RYV_LLAMA_CPP_PORT"
	EnvCtxSize   = "RYV_LLAMA_CPP_CTX_SIZE"
	EnvThreads   = "RYV_LLAMA_CPP_THREADS"
	EnvGPULayers = "RYV_LLAMA_CPP_GPU_LAYERS"
	EnvExtraArgs = "RYV_LLAMA_CPP_EXTRA_ARGS"

	DefaultHost        = "127.0.0.1"
	DefaultPort        = 45910
	DefaultContextSize = 4096
)

type LlamaCppSidecarConfig struct {
	Enabled     bool
	ServerPath  string
	ModelPath   string
	Host        string
	Port        int
	ContextSize int
	Threads     int
	GPULayers   int
	ExtraArgs   []string
}

type LlamaCppSidecarStatus struct {
	Enabled                bool      `json:"enabled"`
	Available              bool      `json:"available"`
	Running                bool      `json:"running"`
	Healthy                bool      `json:"healthy"`
	Attached               bool      `json:"attached,omitempty"`
	PID                    int       `json:"pid,omitempty"`
	BaseURL                string    `json:"base_url"`
	ServerPath             string    `json:"server_path,omitempty"`
	ModelPath              string    `json:"model_path,omitempty"`
	ModelFilename          string    `json:"model_filename,omitempty"`
	ModelSizeBytes         int64     `json:"model_size_bytes"`
	ModelFamilyHint        string    `json:"model_family_hint,omitempty"`
	QuantizationHint       string    `json:"quantization_hint,omitempty"`
	StartedAt              time.Time `json:"started_at,omitempty"`
	LastHealthAt           time.Time `json:"last_health_at,omitempty"`
	LastError              string    `json:"last_error"`
	Backend                string    `json:"backend"`
	OpenAICompatible       bool      `json:"openai_compatible"`
	SupportsTextGeneration bool      `json:"supports_text_generation"`
	SupportsStreaming      bool      `json:"supports_streaming"`
	SupportsKVAccess       bool      `json:"supports_kv_access"`
	SupportsTensorHooks    bool      `json:"supports_tensor_hooks"`
	Reason                 string    `json:"reason"`
}

type HealthResult struct {
	Healthy   bool
	Endpoint  string
	CheckedAt time.Time
	Error     string
}
