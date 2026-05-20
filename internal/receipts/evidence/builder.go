package evidence

import (
	"strings"

	"github.com/Ryvion/ryvion-node/internal/sandbox"
)

type BuildEvidencePayloadInput struct {
	SchemaVersion             string
	JobID                     string
	AssignmentID              string
	NodeID                    string
	ExecutionStartedAtUnixMs  int64
	ExecutionFinishedAtUnixMs int64
	RuntimeEvidence           RuntimeEvidence
	ModelEvidence             ModelEvidence
	OutputEvidence            OutputEvidence
	Artifacts                 []ArtifactEvidence
}

func BuildEvidencePayload(input BuildEvidencePayloadInput) (RYV3EvidencePayload, error) {
	schemaVersion := strings.TrimSpace(input.SchemaVersion)
	if schemaVersion == "" {
		schemaVersion = SchemaVersionV1
	}

	payload := RYV3EvidencePayload{
		SchemaVersion:             schemaVersion,
		JobID:                     strings.TrimSpace(input.JobID),
		AssignmentID:              strings.TrimSpace(input.AssignmentID),
		NodeID:                    strings.TrimSpace(input.NodeID),
		ExecutionStartedAtUnixMs:  input.ExecutionStartedAtUnixMs,
		ExecutionFinishedAtUnixMs: input.ExecutionFinishedAtUnixMs,
		RuntimeEvidence:           normalizeRuntimeEvidence(input.RuntimeEvidence),
		ModelEvidence:             normalizeModelEvidence(input.ModelEvidence),
		OutputEvidence:            normalizeOutputEvidence(input.OutputEvidence),
		Artifacts:                 cloneArtifactEvidence(input.Artifacts),
	}
	if err := ValidateEvidencePayload(payload); err != nil {
		return RYV3EvidencePayload{}, err
	}
	return payload, nil
}

func normalizeRuntimeEvidence(runtimeEvidence RuntimeEvidence) RuntimeEvidence {
	return RuntimeEvidence{
		AgentVersion: strings.TrimSpace(runtimeEvidence.AgentVersion),
		RunnerKind:   strings.TrimSpace(runtimeEvidence.RunnerKind),
		RuntimeHash:  strings.TrimSpace(runtimeEvidence.RuntimeHash),
		OS:           strings.TrimSpace(runtimeEvidence.OS),
		Arch:         strings.TrimSpace(runtimeEvidence.Arch),
	}
}

func normalizeModelEvidence(modelEvidence ModelEvidence) ModelEvidence {
	modelFormat := strings.TrimSpace(modelEvidence.ModelFormat)
	if evaluated := sandbox.EvaluateModelFormat("", modelFormat); evaluated != sandbox.ModelFormatUnknown {
		modelFormat = string(evaluated)
	}
	return ModelEvidence{
		ModelID:        strings.TrimSpace(modelEvidence.ModelID),
		ModelRevision:  strings.TrimSpace(modelEvidence.ModelRevision),
		QuantizationID: strings.TrimSpace(modelEvidence.QuantizationID),
		ModelObjectID:  strings.TrimSpace(modelEvidence.ModelObjectID),
		ModelFormat:    modelFormat,
	}
}

func normalizeOutputEvidence(outputEvidence OutputEvidence) OutputEvidence {
	return OutputEvidence{
		OutputHash:      strings.TrimSpace(outputEvidence.OutputHash),
		OutputHashCodec: strings.TrimSpace(outputEvidence.OutputHashCodec),
		OutputBytes:     outputEvidence.OutputBytes,
		MeteringUnits:   outputEvidence.MeteringUnits,
	}
}

func cloneArtifactEvidence(artifacts []ArtifactEvidence) []ArtifactEvidence {
	if artifacts == nil {
		return nil
	}
	cloned := make([]ArtifactEvidence, len(artifacts))
	for i, artifact := range artifacts {
		cloned[i] = ArtifactEvidence{
			ArtifactID: strings.TrimSpace(artifact.ArtifactID),
			Kind:       strings.TrimSpace(artifact.Kind),
			ObjectID:   strings.TrimSpace(artifact.ObjectID),
			SHA256Hex:  strings.TrimSpace(artifact.SHA256Hex),
			SizeBytes:  artifact.SizeBytes,
		}
	}
	return cloned
}
