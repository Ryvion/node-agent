package capability

const SchemaVersionV1 = "v7.capability-passport.v1"

type CapabilityPassport struct {
	SchemaVersion             string                    `json:"schema_version"`
	AgentVersion              string                    `json:"agent_version"`
	NodePublicKey             string                    `json:"node_public_key,omitempty"`
	OS                        string                    `json:"os"`
	Arch                      string                    `json:"arch"`
	DeviceType                string                    `json:"device_type,omitempty"`
	DeclaredCountry           string                    `json:"declared_country,omitempty"`
	HardwareProfile           HardwareProfile           `json:"hardware_profile"`
	RuntimeProfile            RuntimeProfile            `json:"runtime_profile"`
	NetworkCapabilitySummary  NetworkCapabilitySummary  `json:"network_capability_summary"`
	ModelCapabilitySummary    ModelCapabilitySummary    `json:"model_capability_summary"`
	SandboxCapabilitySummary  SandboxCapabilitySummary  `json:"sandbox_capability_summary"`
	CASCapabilitySummary      CASCapabilitySummary      `json:"cas_capability_summary"`
	EvidenceCapabilitySummary EvidenceCapabilitySummary `json:"evidence_capability_summary"`
	CreatedAtUnixMs           int64                     `json:"created_at_unix_ms"`
}

type HardwareProfile struct {
	CPUCores      uint32 `json:"cpu_cores"`
	RAMBytes      uint64 `json:"ram_bytes"`
	GPUModel      string `json:"gpu_model,omitempty"`
	VRAMBytes     uint64 `json:"vram_bytes,omitempty"`
	GPUVendor     string `json:"gpu_vendor,omitempty"`
	DriverVersion string `json:"driver_version,omitempty"`
	TEESupported  bool   `json:"tee_supported"`
	TEEType       string `json:"tee_type,omitempty"`
}

type RuntimeProfile struct {
	NativeInferenceSupported bool     `json:"native_inference_supported"`
	OCIAvailable             bool     `json:"oci_available"`
	LlamaServerAvailable     bool     `json:"llama_server_available"`
	ImageRuntimeAvailable    bool     `json:"image_runtime_available"`
	SupportedRunnerKinds     []string `json:"supported_runner_kinds,omitempty"`
}

type NetworkCapabilitySummary struct {
	UploadMbpsP50   float64 `json:"upload_mbps_p50"`
	UploadMbpsP95   float64 `json:"upload_mbps_p95"`
	DownloadMbpsP50 float64 `json:"download_mbps_p50"`
	DownloadMbpsP95 float64 `json:"download_mbps_p95"`
	RTTMsP50        float64 `json:"rtt_ms_p50"`
	RTTMsP95        float64 `json:"rtt_ms_p95"`
	JitterMsP95     float64 `json:"jitter_ms_p95"`
	LossRateP95     float64 `json:"loss_rate_p95"`
}

type ModelCapabilitySummary struct {
	SupportedModelFormats []string `json:"supported_model_formats,omitempty"`
	ResidentModelIDs      []string `json:"resident_model_ids,omitempty"`
	MaxResidentModelBytes uint64   `json:"max_resident_model_bytes,omitempty"`
	SupportsModelLease    bool     `json:"supports_model_lease"`
}

type SandboxCapabilitySummary struct {
	RejectsUnsafePickle        bool `json:"rejects_unsafe_pickle"`
	RunnerAllowlistEnabled     bool `json:"runner_allowlist_enabled"`
	FilesystemIsolationPlanned bool `json:"filesystem_isolation_planned"`
	NetworkIsolationSupported  bool `json:"network_isolation_supported"`
}

type CASCapabilitySummary struct {
	Enabled  bool   `json:"enabled"`
	RootDir  string `json:"root_dir,omitempty"`
	MaxBytes uint64 `json:"max_bytes,omitempty"`
}

type EvidenceCapabilitySummary struct {
	SupportsArtifactManifest    bool `json:"supports_artifact_manifest"`
	SupportsRYV3EvidencePayload bool `json:"supports_ryv3_evidence_payload"`
	SupportsRuntimeHash         bool `json:"supports_runtime_hash"`
}
