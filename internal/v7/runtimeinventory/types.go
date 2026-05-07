package runtimeinventory

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

type CandidateBackendDetector struct {
	LookPath            func(string) (string, error)
	ReadDirNames        func(string, int) ([]string, error)
	Getenv              func(string) string
	UserHomeDir         func() (string, error)
	ConfiguredModelDirs []string
}
