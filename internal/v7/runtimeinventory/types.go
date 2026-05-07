package runtimeinventory

import (
	"os"
	"time"
)

const (
	RuntimeKindNative  = "native"
	RuntimeKindUnknown = "unknown"

	BackendNative  = "native"
	BackendUnknown = "unknown"

	ProviderNoop = "noop"

	ProcessModeEmbedded = "embedded"
	ProcessModeSidecar  = "sidecar"
	ProcessModeUnknown  = "unknown"
)

type Inventory struct {
	RuntimeKind          string                   `json:"runtime_kind"`
	Backend              string                   `json:"backend"`
	Provider             string                   `json:"provider"`
	ProcessMode          string                   `json:"process_mode"`
	NativeInferenceReady bool                     `json:"native_inference_ready"`
	NativeModel          string                   `json:"native_model,omitempty"`
	LoadedModels         []ModelResidencySnapshot `json:"loaded_models"`
	CandidateBackends    CandidateBackends        `json:"candidate_backends"`
	BackendCandidates    []BackendCandidate       `json:"backend_candidates"`
	GGUFModels           []GGUFModelCandidate     `json:"gguf_models"`
}

type RuntimeStatus struct {
	RuntimeKind             string
	Backend                 string
	Provider                string
	ProcessMode             string
	NativeInferenceReady    bool
	NativeModel             string
	ModelLoaded             bool
	SupportsTextGeneration  bool
	SupportsStreaming       bool
	SupportsTensorPlaneDemo bool
	Reason                  string
}

type ModelResidencySnapshot struct {
	ModelID                 string `json:"model_id"`
	RuntimeKind             string `json:"runtime_kind"`
	Backend                 string `json:"backend"`
	Loaded                  bool   `json:"loaded"`
	Warm                    bool   `json:"warm"`
	SupportsTextGeneration  bool   `json:"supports_text_generation"`
	SupportsStreaming       bool   `json:"supports_streaming"`
	SupportsKVAccess        bool   `json:"supports_kv_access"`
	SupportsTensorPlane     bool   `json:"supports_tensorplane"`
	SupportsTensorPlaneDemo bool   `json:"supports_tensorplane_demo"`
	Reason                  string `json:"reason,omitempty"`
}

type CandidateBackends struct {
	LlamaCPPDetected           bool `json:"llama_cpp_detected"`
	OllamaDetected             bool `json:"ollama_detected"`
	VLLMDetected               bool `json:"vllm_detected"`
	PythonTransformersDetected bool `json:"python_transformers_detected"`
	GGUFModelsDetected         bool `json:"gguf_models_detected"`
}

type BackendCandidate struct {
	Backend                              string `json:"backend"`
	Detected                             bool   `json:"detected"`
	BinaryPath                           string `json:"binary_path,omitempty"`
	ServerBinaryPath                     string `json:"server_binary_path,omitempty"`
	BenchBinaryPath                      string `json:"bench_binary_path,omitempty"`
	Version                              string `json:"version"`
	SupportsTextGeneration               bool   `json:"supports_text_generation"`
	SupportsStreaming                    bool   `json:"supports_streaming"`
	SupportsOpenAICompatibleServer       bool   `json:"supports_openai_compatible_server"`
	SupportsKVAccess                     bool   `json:"supports_kv_access"`
	SupportsTensorHooks                  bool   `json:"supports_tensor_hooks"`
	SupportsSpeculativeDecode            bool   `json:"supports_speculative_decode"`
	CandidateForRealTensorAccess         bool   `json:"candidate_for_real_tensor_access"`
	CandidateForFastTextRuntime          bool   `json:"candidate_for_fast_text_runtime"`
	PythonTransformersImportAvailable    bool   `json:"python_transformers_import_available,omitempty"`
	PythonTransformersImportProbeAttempt bool   `json:"python_transformers_import_probe_attempted,omitempty"`
	Reason                               string `json:"reason"`
}

type GGUFModelCandidate struct {
	Path             string `json:"path"`
	Filename         string `json:"filename"`
	SizeBytes        int64  `json:"size_bytes"`
	ModelFamilyHint  string `json:"model_family_hint"`
	QuantizationHint string `json:"quantization_hint"`
}

type CandidateBackendDetector struct {
	LookPath              func(string) (string, error)
	Stat                  func(string) (os.FileInfo, error)
	ReadDirNames          func(string, int) ([]string, error)
	VersionCommand        func(string, time.Duration) (string, error)
	PythonModuleAvailable func(string, string, time.Duration) (bool, error)
	Getenv                func(string) string
	UserHomeDir           func() (string, error)
	GOOS                  string
	ConfiguredBinaryDirs  []string
	ConfiguredModelDirs   []string
	VersionTimeout        time.Duration
}
