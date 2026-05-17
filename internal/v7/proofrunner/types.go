package proofrunner

import (
	"github.com/Ryvion/ryvion-node/internal/v7/artifact"
	"github.com/Ryvion/ryvion-node/internal/v7/evidence"
)

type RunnerResultInput struct {
	JobID            string
	AssignmentID     string
	NodeID           string
	RunnerKind       string
	AgentVersion     string
	ModelID          string
	ModelRevision    string
	QuantizationID   string
	OutputBytes      []byte
	MeteringUnits    int64
	ArtifactKind     artifact.ArtifactKind
	StartedAtUnixMs  int64
	FinishedAtUnixMs int64
}

type ProofCarryingRunnerOutput struct {
	ArtifactManifest    artifact.ArtifactManifest    `json:"artifact_manifest"`
	EvidencePayload     evidence.RYV3EvidencePayload `json:"evidence_payload"`
	OutputHash          string                       `json:"output_hash"`
	OutputBytes         int64                        `json:"output_bytes"`
	ArtifactObjectID    string                       `json:"artifact_object_id"`
	CASObjectReferences []string                     `json:"cas_object_references,omitempty"`
}
