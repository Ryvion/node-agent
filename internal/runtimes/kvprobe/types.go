package kvprobe

const (
	RuntimeKindNative  = "native"
	RuntimeKindUnknown = "unknown"

	BackendNative   = "native"
	BackendLlamaCPP = "llama.cpp"
	BackendUnknown  = "unknown"

	ReasonTextGenerationOnly       = "native runtime currently exposes text generation only"
	ReasonNativeRuntimeUnavailable = "native runtime is not available"
	ReasonTensorHooksAvailable     = "native runtime exposes safe tensor access hooks"
)

type Capability struct {
	KVAccessSupported          bool   `json:"kv_access_supported"`
	KVSnapshotSupported        bool   `json:"kv_snapshot_supported"`
	HiddenStateAccessSupported bool   `json:"hidden_state_access_supported"`
	LogitsAccessSupported      bool   `json:"logits_access_supported"`
	AttentionHookSupported     bool   `json:"attention_hook_supported"`
	Backend                    string `json:"backend"`
	ModelID                    string `json:"model_id"`
	ModelLoaded                bool   `json:"model_loaded"`
	RuntimeKind                string `json:"runtime_kind"`
	Reason                     string `json:"reason"`
}

type HookSupport struct {
	KVAccessSupported          bool
	KVSnapshotSupported        bool
	HiddenStateAccessSupported bool
	LogitsAccessSupported      bool
	AttentionHookSupported     bool
}

type ProbeInput struct {
	RuntimeAvailable bool
	RuntimeKind      string
	Backend          string
	ModelID          string
	ModelLoaded      bool
	Hooks            HookSupport
	Reason           string
}
