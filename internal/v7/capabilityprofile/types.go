package capabilityprofile

const SchemaVersionV1 = "v7.capability-profile.v1"

type Profile struct {
	SchemaVersion         string            `json:"schema_version"`
	V7DashboardInference  bool              `json:"v7_dashboard_inference"`
	TextOutput            bool              `json:"text_output"`
	Streaming             bool              `json:"streaming"`
	HashMetricsReceipts   bool              `json:"hash_metrics_receipts"`
	BackendTextGeneration bool              `json:"backend_text_generation"`
	BackendWarm           bool              `json:"backend_warm"`
	StatefulSession       bool              `json:"stateful_session"`
	KVAccess              bool              `json:"kv_access"`
	TensorHooks           bool              `json:"tensor_hooks"`
	TensorPlaneDemo       bool              `json:"tensorplane_demo"`
	Ready                 bool              `json:"ready"`
	Reason                string            `json:"reason,omitempty"`
	Hardware              HardwareSummary   `json:"hardware"`
	Policy                PolicySummary     `json:"policy"`
	BackendRuntime        BackendSummary    `json:"backend_runtime"`
	WarmModel             WarmModelSummary  `json:"warm_model"`
	Models                []ModelCapability `json:"models"`
}

type HardwareSummary struct {
	OS                string `json:"os"`
	Arch              string `json:"arch"`
	CPULogicalCores   int    `json:"cpu_logical_cores"`
	SystemRAMBytes    uint64 `json:"system_ram_bytes"`
	AvailableRAMBytes uint64 `json:"available_ram_bytes"`
	GPUDetected       bool   `json:"gpu_detected"`
	GPUVendor         string `json:"gpu_vendor"`
	GPUName           string `json:"gpu_name"`
	GPUVRAMBytes      uint64 `json:"gpu_vram_bytes"`
	UnifiedMemory     bool   `json:"unified_memory"`
	CUDAAvailable     bool   `json:"cuda_available"`
	MetalAvailable    bool   `json:"metal_available"`
	VulkanAvailable   bool   `json:"vulkan_available"`
}

type PolicySummary struct {
	MaxRuntimeModelBytes             uint64   `json:"max_runtime_model_bytes"`
	MaxRuntimeParameterCountBillions float64  `json:"max_runtime_parameter_count_billions"`
	AllowedFormats                   []string `json:"allowed_formats"`
	AllowedFamilies                  []string `json:"allowed_families"`
	DeniedModelIDs                   []string `json:"denied_model_ids"`
	DeniedFamilies                   []string `json:"denied_families"`
	AllowManagedPrepareDownload      bool     `json:"allow_managed_prepare_download"`
	MaxWarmModels                    int      `json:"max_warm_models"`
	MaxConcurrentInferenceJobs       int      `json:"max_concurrent_inference_jobs"`
}

type BackendSummary struct {
	Backend                string `json:"backend"`
	Available              bool   `json:"available"`
	Running                bool   `json:"running"`
	Healthy                bool   `json:"healthy"`
	SupportsTextGeneration bool   `json:"supports_text_generation"`
	SupportsStreaming      bool   `json:"supports_streaming"`
	SupportsWarmResidency  bool   `json:"supports_warm_residency"`
	Reason                 string `json:"reason,omitempty"`
}

type WarmModelSummary struct {
	Backend string `json:"backend,omitempty"`
	ModelID string `json:"model_id,omitempty"`
	Warm    bool   `json:"warm"`
	Healthy bool   `json:"healthy"`
}

type ModelCapability struct {
	ModelID   string `json:"model_id"`
	Family    string `json:"family"`
	Format    string `json:"format"`
	SizeBytes int64  `json:"size_bytes"`
	Resident  bool   `json:"resident"`
	Warm      bool   `json:"warm"`
	Runnable  bool   `json:"runnable"`
	Reason    string `json:"reason,omitempty"`
}
