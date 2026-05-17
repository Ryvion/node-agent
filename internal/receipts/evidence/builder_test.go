package evidence

import (
	"reflect"
	"testing"
)

func TestBuildEvidencePayloadStableAndClonesArtifacts(t *testing.T) {
	input := validEvidenceInput()

	first, err := BuildEvidencePayload(input)
	if err != nil {
		t.Fatalf("BuildEvidencePayload() error = %v", err)
	}
	second, err := BuildEvidencePayload(input)
	if err != nil {
		t.Fatalf("second BuildEvidencePayload() error = %v", err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("BuildEvidencePayload() was not stable:\nfirst=%+v\nsecond=%+v", first, second)
	}
	if first.SchemaVersion != SchemaVersionV1 {
		t.Fatalf("schema version = %q, want %q", first.SchemaVersion, SchemaVersionV1)
	}
	if len(first.Artifacts) != 1 {
		t.Fatalf("artifacts length = %d, want 1", len(first.Artifacts))
	}

	input.Artifacts[0].ArtifactID = "mutated"
	if first.Artifacts[0].ArtifactID != "artifact-1" {
		t.Fatalf("payload artifact mutated through input slice: %+v", first.Artifacts[0])
	}
}

func TestBuildEvidencePayloadTrimsAndNormalizesModelFormat(t *testing.T) {
	input := validEvidenceInput()
	input.JobID = " job-1 "
	input.ModelEvidence.ModelFormat = " safe tensors "

	payload, err := BuildEvidencePayload(input)
	if err != nil {
		t.Fatalf("BuildEvidencePayload() error = %v", err)
	}
	if payload.JobID != "job-1" {
		t.Fatalf("JobID = %q, want trimmed job-1", payload.JobID)
	}
	if payload.ModelEvidence.ModelFormat != "safetensors" {
		t.Fatalf("ModelFormat = %q, want safetensors", payload.ModelEvidence.ModelFormat)
	}
}

func validEvidenceInput() BuildEvidencePayloadInput {
	payload := validEvidencePayload()
	return BuildEvidencePayloadInput{
		JobID:                     payload.JobID,
		AssignmentID:              payload.AssignmentID,
		NodeID:                    payload.NodeID,
		ExecutionStartedAtUnixMs:  payload.ExecutionStartedAtUnixMs,
		ExecutionFinishedAtUnixMs: payload.ExecutionFinishedAtUnixMs,
		RuntimeEvidence:           payload.RuntimeEvidence,
		ModelEvidence:             payload.ModelEvidence,
		OutputEvidence:            payload.OutputEvidence,
		Artifacts:                 payload.Artifacts,
	}
}
