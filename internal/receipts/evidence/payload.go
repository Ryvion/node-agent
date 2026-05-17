package evidence

const SchemaVersionV1 = "ryvion.node.evidence_payload.v1"

type RYV3EvidencePayload struct {
	SchemaVersion             string             `json:"schema_version"`
	JobID                     string             `json:"job_id"`
	AssignmentID              string             `json:"assignment_id"`
	NodeID                    string             `json:"node_id"`
	ExecutionStartedAtUnixMs  int64              `json:"execution_started_at_unix_ms"`
	ExecutionFinishedAtUnixMs int64              `json:"execution_finished_at_unix_ms"`
	RuntimeEvidence           RuntimeEvidence    `json:"runtime_evidence"`
	ModelEvidence             ModelEvidence      `json:"model_evidence"`
	OutputEvidence            OutputEvidence     `json:"output_evidence"`
	Artifacts                 []ArtifactEvidence `json:"artifacts,omitempty"`
}

type RuntimeEvidence struct {
	AgentVersion string `json:"agent_version"`
	RunnerKind   string `json:"runner_kind"`
	RuntimeHash  string `json:"runtime_hash,omitempty"`
	OS           string `json:"os"`
	Arch         string `json:"arch"`
}

type ModelEvidence struct {
	ModelID        string `json:"model_id"`
	ModelRevision  string `json:"model_revision"`
	QuantizationID string `json:"quantization_id"`
	ModelObjectID  string `json:"model_object_id,omitempty"`
	ModelFormat    string `json:"model_format"`
}

type OutputEvidence struct {
	OutputHash      string `json:"output_hash"`
	OutputHashCodec string `json:"output_hash_codec"`
	OutputBytes     int64  `json:"output_bytes"`
	MeteringUnits   int64  `json:"metering_units"`
}

type ArtifactEvidence struct {
	ArtifactID string `json:"artifact_id"`
	Kind       string `json:"kind"`
	ObjectID   string `json:"object_id"`
	SHA256Hex  string `json:"sha256_hex"`
	SizeBytes  int64  `json:"size_bytes"`
}
