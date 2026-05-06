package tensoraccess

import (
	"errors"

	"github.com/Ryvion/node-agent/internal/v7/tensorplane"
)

const (
	ProviderNoop            = "noop"
	ProviderTensorPlaneDemo = "tensorplane_demo"
	ProviderLlamaCPP        = "llama_cpp"
	ProviderVLLM            = "vllm"
	ProviderRyvionRuntime   = "ryvion_runtime"

	BackendNative  = "native"
	BackendDemo    = "demo"
	BackendUnknown = "unknown"

	RuntimeKindNative  = "native"
	RuntimeKindDemo    = "demo"
	RuntimeKindUnknown = "unknown"

	ReasonTextGenerationOnly       = "native runtime currently exposes text generation only"
	ReasonNativeRuntimeUnavailable = "native runtime is not available"
	ReasonTensorPlaneDemoOnly      = "tensorplane demo provider exposes deterministic fixture tensors only; no real KV access"
)

var (
	ErrTensorAccessUnsupported    = errors.New("tensoraccess: unsupported tensor access")
	ErrInvalidTensorAccessRequest = errors.New("tensoraccess: invalid request")
)

type TensorAccessCapability struct {
	Provider                   string `json:"provider"`
	Backend                    string `json:"backend"`
	KVAccessSupported          bool   `json:"kv_access_supported"`
	KVSnapshotSupported        bool   `json:"kv_snapshot_supported"`
	HiddenStateAccessSupported bool   `json:"hidden_state_access_supported"`
	LogitsAccessSupported      bool   `json:"logits_access_supported"`
	AttentionHookSupported     bool   `json:"attention_hook_supported"`
	TensorPlaneDemoSupported   bool   `json:"tensorplane_demo_supported"`
	ModelLoaded                bool   `json:"model_loaded"`
	RuntimeKind                string `json:"runtime_kind"`
	ModelID                    string `json:"model_id"`
	Reason                     string `json:"reason"`
}

type LoadedTensorModel struct {
	ModelID             string `json:"model_id"`
	RuntimeKind         string `json:"runtime_kind"`
	Backend             string `json:"backend"`
	Loaded              bool   `json:"loaded"`
	SupportsKV          bool   `json:"supports_kv"`
	SupportsTensorPlane bool   `json:"supports_tensorplane"`
	ContextLength       int    `json:"context_length,omitempty"`
	Layers              int    `json:"layers,omitempty"`
	Heads               int    `json:"heads,omitempty"`
	HeadDim             int    `json:"head_dim,omitempty"`
}

type TensorPageRequest struct {
	ModelID    string                  `json:"model_id"`
	LayerIndex int                     `json:"layer_index"`
	HeadStart  int                     `json:"head_start"`
	HeadCount  int                     `json:"head_count"`
	TokenStart int                     `json:"token_start"`
	TokenCount int                     `json:"token_count"`
	DType      tensorplane.TensorDType `json:"dtype"`
	PageID     string                  `json:"page_id"`
	Seed       int64                   `json:"seed,omitempty"`
}

type TensorQueryRequest struct {
	RequestID  string                  `json:"request_id"`
	JobID      string                  `json:"job_id"`
	ModelID    string                  `json:"model_id"`
	LayerIndex int                     `json:"layer_index"`
	HeadIndex  int                     `json:"head_index"`
	DType      tensorplane.TensorDType `json:"dtype"`
	HeadDim    int                     `json:"head_dim"`
	Seed       int64                   `json:"seed,omitempty"`
}
