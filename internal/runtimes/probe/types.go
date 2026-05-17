package backendprobe

import (
	"os"
	"time"
)

const EnvLlamaCPPProbeModel = "RYV_LLAMA_CPP_PROBE_MODEL"

type Probes struct {
	LlamaCPP LlamaCPPProbe `json:"llama_cpp"`
}

type LlamaCPPProbe struct {
	Available                      bool   `json:"available"`
	BinaryPath                     string `json:"binary_path"`
	ServerBinaryPath               string `json:"server_binary_path"`
	BenchBinaryPath                string `json:"bench_binary_path"`
	Version                        string `json:"version"`
	GGUFModelsDetected             bool   `json:"gguf_models_detected"`
	ProbeModelConfigured           bool   `json:"probe_model_configured"`
	SupportsTextGeneration         bool   `json:"supports_text_generation"`
	SupportsStreaming              bool   `json:"supports_streaming"`
	SupportsOpenAICompatibleServer bool   `json:"supports_openai_compatible_server"`
	SupportsKVAccess               bool   `json:"supports_kv_access"`
	SupportsTensorHooks            bool   `json:"supports_tensor_hooks"`
	CandidateForFastTextRuntime    bool   `json:"candidate_for_fast_text_runtime"`
	CandidateForRealTensorAccess   bool   `json:"candidate_for_real_tensor_access"`
	Reason                         string `json:"reason"`
}

type Detector struct {
	LookPath             func(string) (string, error)
	Stat                 func(string) (os.FileInfo, error)
	ReadDirNames         func(string, int) ([]string, error)
	VersionCommand       func(string, time.Duration) (string, error)
	ModelProbeCommand    func(string, string, time.Duration) error
	Getenv               func(string) string
	UserHomeDir          func() (string, error)
	GOOS                 string
	ConfiguredBinaryDirs []string
	ConfiguredModelDirs  []string
	VersionTimeout       time.Duration
	ModelProbeTimeout    time.Duration
}
