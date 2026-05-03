package evidence

import (
	"errors"
	"fmt"
	"strings"

	"github.com/Ryvion/node-agent/internal/v7/localcas"
	"github.com/Ryvion/node-agent/internal/v7/sandbox"
)

const (
	outputHashPrefix = "sha256:"
	sha256HexLength  = 64
)

var ErrInvalidEvidencePayload = errors.New("evidence: invalid payload")

func ValidateEvidencePayload(payload RYV3EvidencePayload) error {
	var errs []error
	if strings.TrimSpace(payload.SchemaVersion) == "" {
		errs = append(errs, fmt.Errorf("%w: schema_version required", ErrInvalidEvidencePayload))
	}
	if strings.TrimSpace(payload.JobID) == "" {
		errs = append(errs, fmt.Errorf("%w: job_id required", ErrInvalidEvidencePayload))
	}
	if strings.TrimSpace(payload.AssignmentID) == "" {
		errs = append(errs, fmt.Errorf("%w: assignment_id required", ErrInvalidEvidencePayload))
	}
	if strings.TrimSpace(payload.NodeID) == "" {
		errs = append(errs, fmt.Errorf("%w: node_id required", ErrInvalidEvidencePayload))
	}
	if payload.ExecutionStartedAtUnixMs < 0 {
		errs = append(errs, fmt.Errorf("%w: execution_started_at_unix_ms must be non-negative", ErrInvalidEvidencePayload))
	}
	if payload.ExecutionFinishedAtUnixMs < 0 {
		errs = append(errs, fmt.Errorf("%w: execution_finished_at_unix_ms must be non-negative", ErrInvalidEvidencePayload))
	}
	if payload.ExecutionStartedAtUnixMs != 0 &&
		payload.ExecutionFinishedAtUnixMs != 0 &&
		payload.ExecutionFinishedAtUnixMs < payload.ExecutionStartedAtUnixMs {
		errs = append(errs, fmt.Errorf("%w: execution_finished_at_unix_ms must be greater than or equal to execution_started_at_unix_ms", ErrInvalidEvidencePayload))
	}
	if err := validateRuntimeEvidence(payload.RuntimeEvidence); err != nil {
		errs = append(errs, err)
	}
	if err := validateModelEvidence(payload.ModelEvidence); err != nil {
		errs = append(errs, err)
	}
	if err := validateOutputEvidence(payload.OutputEvidence); err != nil {
		errs = append(errs, err)
	}
	for i, artifact := range payload.Artifacts {
		if err := validateArtifactEvidence(artifact); err != nil {
			errs = append(errs, fmt.Errorf("artifact[%d]: %w", i, err))
		}
	}
	return errors.Join(errs...)
}

func validateRuntimeEvidence(runtimeEvidence RuntimeEvidence) error {
	var errs []error
	if strings.TrimSpace(runtimeEvidence.AgentVersion) == "" {
		errs = append(errs, fmt.Errorf("%w: runtime_evidence.agent_version required", ErrInvalidEvidencePayload))
	}
	if strings.TrimSpace(runtimeEvidence.RunnerKind) == "" {
		errs = append(errs, fmt.Errorf("%w: runtime_evidence.runner_kind required", ErrInvalidEvidencePayload))
	}
	if strings.TrimSpace(runtimeEvidence.OS) == "" {
		errs = append(errs, fmt.Errorf("%w: runtime_evidence.os required", ErrInvalidEvidencePayload))
	}
	if strings.TrimSpace(runtimeEvidence.Arch) == "" {
		errs = append(errs, fmt.Errorf("%w: runtime_evidence.arch required", ErrInvalidEvidencePayload))
	}
	if runtimeHash := strings.TrimSpace(runtimeEvidence.RuntimeHash); runtimeHash != "" {
		if err := validateHashID(runtimeHash, "runtime_evidence.runtime_hash"); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func validateModelEvidence(modelEvidence ModelEvidence) error {
	var errs []error
	if strings.TrimSpace(modelEvidence.ModelID) == "" {
		errs = append(errs, fmt.Errorf("%w: model_evidence.model_id required", ErrInvalidEvidencePayload))
	}
	if strings.TrimSpace(modelEvidence.ModelRevision) == "" {
		errs = append(errs, fmt.Errorf("%w: model_evidence.model_revision required", ErrInvalidEvidencePayload))
	}
	if strings.TrimSpace(modelEvidence.QuantizationID) == "" {
		errs = append(errs, fmt.Errorf("%w: model_evidence.quantization_id required", ErrInvalidEvidencePayload))
	}
	if modelObjectID := strings.TrimSpace(modelEvidence.ModelObjectID); modelObjectID != "" {
		if err := validateHashID(modelObjectID, "model_evidence.model_object_id"); err != nil {
			errs = append(errs, err)
		}
	}
	format := strings.TrimSpace(modelEvidence.ModelFormat)
	if format == "" {
		errs = append(errs, fmt.Errorf("%w: model_evidence.model_format required", ErrInvalidEvidencePayload))
	} else if !isSandboxSafeModelFormat(format) {
		errs = append(errs, fmt.Errorf("%w: model_evidence.model_format %q is not sandbox-safe", ErrInvalidEvidencePayload, modelEvidence.ModelFormat))
	}
	return errors.Join(errs...)
}

func validateOutputEvidence(outputEvidence OutputEvidence) error {
	var errs []error
	if err := validateHashID(outputEvidence.OutputHash, "output_evidence.output_hash"); err != nil {
		errs = append(errs, err)
	}
	if strings.TrimSpace(outputEvidence.OutputHashCodec) == "" {
		errs = append(errs, fmt.Errorf("%w: output_evidence.output_hash_codec required", ErrInvalidEvidencePayload))
	}
	if outputEvidence.OutputBytes < 0 {
		errs = append(errs, fmt.Errorf("%w: output_evidence.output_bytes must be non-negative", ErrInvalidEvidencePayload))
	}
	if outputEvidence.MeteringUnits < 0 {
		errs = append(errs, fmt.Errorf("%w: output_evidence.metering_units must be non-negative", ErrInvalidEvidencePayload))
	}
	return errors.Join(errs...)
}

func validateArtifactEvidence(artifact ArtifactEvidence) error {
	var errs []error
	if strings.TrimSpace(artifact.ArtifactID) == "" {
		errs = append(errs, fmt.Errorf("%w: artifact_id required", ErrInvalidEvidencePayload))
	}
	if strings.TrimSpace(artifact.Kind) == "" {
		errs = append(errs, fmt.Errorf("%w: kind required", ErrInvalidEvidencePayload))
	}
	if err := validateHashID(artifact.ObjectID, "object_id"); err != nil {
		errs = append(errs, err)
	}
	if err := validateSHA256Hex(artifact.SHA256Hex, "sha256_hex"); err != nil {
		errs = append(errs, err)
	}
	if artifact.ObjectID != "" && artifact.SHA256Hex != "" && artifact.ObjectID != outputHashPrefix+artifact.SHA256Hex {
		errs = append(errs, fmt.Errorf("%w: object_id must match sha256_hex", ErrInvalidEvidencePayload))
	}
	if artifact.SizeBytes < 0 {
		errs = append(errs, fmt.Errorf("%w: size_bytes must be non-negative", ErrInvalidEvidencePayload))
	}
	return errors.Join(errs...)
}

func validateHashID(value string, field string) error {
	value = strings.TrimSpace(value)
	if err := localcas.ValidateObjectID(localcas.ObjectID(value)); err != nil {
		return fmt.Errorf("%w: %s must be sha256:<64 lowercase hex>", ErrInvalidEvidencePayload, field)
	}
	return nil
}

func validateSHA256Hex(value string, field string) error {
	value = strings.TrimSpace(value)
	if len(value) != sha256HexLength {
		return fmt.Errorf("%w: %s must be 64 lowercase hex characters", ErrInvalidEvidencePayload, field)
	}
	for _, ch := range value {
		if (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') {
			continue
		}
		return fmt.Errorf("%w: %s must be 64 lowercase hex characters", ErrInvalidEvidencePayload, field)
	}
	return nil
}

func isSandboxSafeModelFormat(format string) bool {
	switch sandbox.EvaluateModelFormat("", format) {
	case sandbox.ModelFormatGGUF, sandbox.ModelFormatSafetensors, sandbox.ModelFormatONNX:
		return true
	default:
		return false
	}
}
