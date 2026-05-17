package sglang

import (
	"os"
	"time"

	v7hardware "github.com/Ryvion/ryvion-node/internal/v7/hardware"
	"github.com/Ryvion/ryvion-node/internal/v7/runtimeinventory"
)

const (
	BackendName = runtimeinventory.BackendCandidateSGLang

	EnvEnabled           = "RYV_SGLANG_ENABLED"
	EnvServer            = "RYV_SGLANG_SERVER_PATH"
	EnvModel             = "RYV_SGLANG_MODEL_PATH"
	EnvModelID           = "RYV_SGLANG_MODEL_ID"
	EnvHost              = "RYV_SGLANG_HOST"
	EnvPort              = "RYV_SGLANG_PORT"
	EnvContextLength     = "RYV_SGLANG_CONTEXT_LENGTH"
	EnvTPSize            = "RYV_SGLANG_TP_SIZE"
	EnvMemFractionStatic = "RYV_SGLANG_MEM_FRACTION_STATIC"
	EnvExtraArgs         = "RYV_SGLANG_EXTRA_ARGS"

	DefaultHost          = "127.0.0.1"
	DefaultPort          = 45921
	DefaultContextLength = 4096
	DefaultTPSize        = 1
)

type ConfigSource struct {
	Getenv           func(string) string
	LookPath         func(string) (string, error)
	Stat             func(string) (os.FileInfo, error)
	GOOS             string
	RuntimeInventory *runtimeinventory.Inventory
	HardwareCapacity *v7hardware.CapacityInventory
}

type SGLangSidecarConfig struct {
	Enabled              bool
	ServerPath           string
	ServerPathExplicit   bool
	ModelPath            string
	ModelID              string
	Host                 string
	Port                 int
	ContextLength        int
	TPSize               int
	MemFractionStatic    float64
	ExtraArgs            []string
	AccelerationHints    []string
	LaunchProfile        string
	ModelPathMustBeLocal bool
}

type SGLangSidecarStatus struct {
	Enabled                  bool          `json:"enabled"`
	Available                bool          `json:"available"`
	Running                  bool          `json:"running"`
	Healthy                  bool          `json:"healthy"`
	Attached                 bool          `json:"attached,omitempty"`
	PID                      int           `json:"pid,omitempty"`
	BaseURL                  string        `json:"base_url"`
	ServerPath               string        `json:"server_path,omitempty"`
	ModelPath                string        `json:"model_path,omitempty"`
	ModelID                  string        `json:"model_id,omitempty"`
	ContextLength            int           `json:"context_length,omitempty"`
	TPSize                   int           `json:"tp_size,omitempty"`
	MemFractionStatic        float64       `json:"mem_fraction_static,omitempty"`
	ModelFilename            string        `json:"model_filename,omitempty"`
	ModelSizeBytes           int64         `json:"model_size_bytes"`
	StartedAt                time.Time     `json:"started_at,omitempty"`
	LastHealthAt             time.Time     `json:"last_health_at,omitempty"`
	LastError                string        `json:"last_error"`
	Backend                  string        `json:"backend"`
	Launch                   *LaunchConfig `json:"launch,omitempty"`
	Acceleration             []string      `json:"acceleration"`
	AccelerationReason       string        `json:"acceleration_reason,omitempty"`
	OpenAICompatible         bool          `json:"openai_compatible"`
	SupportsTextGeneration   bool          `json:"supports_text_generation"`
	SupportsStreaming        bool          `json:"supports_streaming"`
	SupportsStatefulSessions bool          `json:"supports_stateful_sessions"`
	SupportsKVAccess         bool          `json:"supports_kv_access"`
	SupportsTensorHooks      bool          `json:"supports_tensor_hooks"`
	Reason                   string        `json:"reason"`
}

type LaunchConfig struct {
	Mode              string  `json:"mode,omitempty"`
	Managed           bool    `json:"managed"`
	Attached          bool    `json:"attached"`
	ServerPath        string  `json:"server_path,omitempty"`
	ServerFilename    string  `json:"server_filename,omitempty"`
	Launcher          string  `json:"launcher,omitempty"`
	ContextLength     int     `json:"context_length,omitempty"`
	TPSize            int     `json:"tp_size,omitempty"`
	MemFractionStatic float64 `json:"mem_fraction_static,omitempty"`
	Profile           string  `json:"profile,omitempty"`
}

type HealthResult struct {
	Healthy   bool
	Endpoint  string
	CheckedAt time.Time
	Error     string
}
