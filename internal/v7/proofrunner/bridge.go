package proofrunner

import (
	"errors"
	"fmt"
	"runtime"
	"strings"

	"github.com/Ryvion/node-agent/internal/v7/artifact"
	"github.com/Ryvion/node-agent/internal/v7/evidence"
	"github.com/Ryvion/node-agent/internal/v7/localcas"
	"github.com/Ryvion/node-agent/internal/v7/sandbox"
)

var ErrInvalidRunnerResult = errors.New("proofrunner: invalid runner result")

func BuildProofCarryingOutput(input RunnerResultInput) (ProofCarryingRunnerOutput, error) {
	normalized, err := normalizeRunnerResultInput(input)
	if err != nil {
		return ProofCarryingRunnerOutput{}, err
	}

	outputObjectID := string(localcas.HashBytes(normalized.OutputBytes))
	outputBytes := int64(len(normalized.OutputBytes))
	outputHashHex := strings.TrimPrefix(outputObjectID, "sha256:")

	manifest, err := artifact.BuildArtifactManifestFromBytes(normalized.OutputBytes, artifact.ArtifactManifestOptions{
		ArtifactID:      outputObjectID,
		Kind:            normalized.ArtifactKind,
		CreatedAtUnixMs: normalized.FinishedAtUnixMs,
		RuntimeID:       normalized.RunnerKind,
		ModelID:         normalized.ModelID,
	})
	if err != nil {
		return ProofCarryingRunnerOutput{}, fmt.Errorf("%w: artifact manifest: %w", ErrInvalidRunnerResult, err)
	}
	if manifest.ObjectID != outputObjectID {
		return ProofCarryingRunnerOutput{}, fmt.Errorf("%w: artifact object id does not match output hash", ErrInvalidRunnerResult)
	}

	payload, err := evidence.BuildEvidencePayload(evidence.BuildEvidencePayloadInput{
		JobID:                     normalized.JobID,
		AssignmentID:              normalized.AssignmentID,
		NodeID:                    normalized.NodeID,
		ExecutionStartedAtUnixMs:  normalized.StartedAtUnixMs,
		ExecutionFinishedAtUnixMs: normalized.FinishedAtUnixMs,
		RuntimeEvidence: evidence.RuntimeEvidence{
			AgentVersion: normalized.AgentVersion,
			RunnerKind:   normalized.RunnerKind,
			OS:           runtime.GOOS,
			Arch:         runtime.GOARCH,
		},
		ModelEvidence: evidence.ModelEvidence{
			ModelID:        normalized.ModelID,
			ModelRevision:  normalized.ModelRevision,
			QuantizationID: normalized.QuantizationID,
			ModelFormat:    modelFormatForRunner(normalized),
		},
		OutputEvidence: evidence.OutputEvidence{
			OutputHash:      outputObjectID,
			OutputHashCodec: "sha256",
			OutputBytes:     outputBytes,
			MeteringUnits:   normalized.MeteringUnits,
		},
		Artifacts: []evidence.ArtifactEvidence{
			{
				ArtifactID: manifest.ArtifactID,
				Kind:       string(manifest.Kind),
				ObjectID:   manifest.ObjectID,
				SHA256Hex:  outputHashHex,
				SizeBytes:  manifest.SizeBytes,
			},
		},
	})
	if err != nil {
		return ProofCarryingRunnerOutput{}, fmt.Errorf("%w: evidence payload: %w", ErrInvalidRunnerResult, err)
	}

	return ProofCarryingRunnerOutput{
		ArtifactManifest:    manifest,
		EvidencePayload:     payload,
		OutputHash:          outputObjectID,
		OutputBytes:         outputBytes,
		ArtifactObjectID:    manifest.ObjectID,
		CASObjectReferences: []string{manifest.ObjectID},
	}, nil
}

func normalizeRunnerResultInput(input RunnerResultInput) (RunnerResultInput, error) {
	input.JobID = strings.TrimSpace(input.JobID)
	input.AssignmentID = strings.TrimSpace(input.AssignmentID)
	input.NodeID = strings.TrimSpace(input.NodeID)
	input.RunnerKind = strings.TrimSpace(input.RunnerKind)
	input.AgentVersion = strings.TrimSpace(input.AgentVersion)
	input.ModelID = strings.TrimSpace(input.ModelID)
	input.ModelRevision = strings.TrimSpace(input.ModelRevision)
	input.QuantizationID = strings.TrimSpace(input.QuantizationID)

	var errs []error
	if input.JobID == "" {
		errs = append(errs, fmt.Errorf("%w: job_id required", ErrInvalidRunnerResult))
	}
	if input.AssignmentID == "" {
		errs = append(errs, fmt.Errorf("%w: assignment_id required", ErrInvalidRunnerResult))
	}
	if input.NodeID == "" {
		errs = append(errs, fmt.Errorf("%w: node_id required", ErrInvalidRunnerResult))
	}
	if input.RunnerKind == "" {
		errs = append(errs, fmt.Errorf("%w: runner_kind required", ErrInvalidRunnerResult))
	}
	if input.AgentVersion == "" {
		errs = append(errs, fmt.Errorf("%w: agent_version required", ErrInvalidRunnerResult))
	}
	if input.ModelID == "" {
		errs = append(errs, fmt.Errorf("%w: model_id required", ErrInvalidRunnerResult))
	}
	if input.ModelRevision == "" {
		errs = append(errs, fmt.Errorf("%w: model_revision required", ErrInvalidRunnerResult))
	}
	if input.QuantizationID == "" {
		errs = append(errs, fmt.Errorf("%w: quantization_id required", ErrInvalidRunnerResult))
	}
	if input.MeteringUnits < 0 {
		errs = append(errs, fmt.Errorf("%w: metering_units must be non-negative", ErrInvalidRunnerResult))
	}
	if input.StartedAtUnixMs <= 0 {
		errs = append(errs, fmt.Errorf("%w: started_at_unix_ms must be positive", ErrInvalidRunnerResult))
	}
	if input.FinishedAtUnixMs <= 0 {
		errs = append(errs, fmt.Errorf("%w: finished_at_unix_ms must be positive", ErrInvalidRunnerResult))
	}
	if input.StartedAtUnixMs > 0 && input.FinishedAtUnixMs > 0 && input.FinishedAtUnixMs < input.StartedAtUnixMs {
		errs = append(errs, fmt.Errorf("%w: finished_at_unix_ms must be greater than or equal to started_at_unix_ms", ErrInvalidRunnerResult))
	}
	if len(errs) > 0 {
		return RunnerResultInput{}, errors.Join(errs...)
	}
	if input.ArtifactKind == "" {
		input.ArtifactKind = artifact.ArtifactKindGenericBlob
	}
	return input, nil
}

func modelFormatForRunner(input RunnerResultInput) string {
	if format := sandbox.EvaluateModelFormat(input.ModelID, ""); isEvidenceSafeModelFormat(format) {
		return string(format)
	}

	runnerKind := strings.ToLower(input.RunnerKind)
	switch {
	case strings.Contains(runnerKind, "onnx"):
		return string(sandbox.ModelFormatONNX)
	case strings.Contains(runnerKind, "safetensor"):
		return string(sandbox.ModelFormatSafetensors)
	default:
		return string(sandbox.ModelFormatGGUF)
	}
}

func isEvidenceSafeModelFormat(format sandbox.ModelFormat) bool {
	switch format {
	case sandbox.ModelFormatGGUF, sandbox.ModelFormatSafetensors, sandbox.ModelFormatONNX:
		return true
	default:
		return false
	}
}
