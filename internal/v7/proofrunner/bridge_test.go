package proofrunner

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/Ryvion/ryvion-node/internal/v7/artifact"
	"github.com/Ryvion/ryvion-node/internal/v7/evidence"
)

func TestBuildProofCarryingOutputValidRunnerResult(t *testing.T) {
	output, err := BuildProofCarryingOutput(validRunnerResultInput())
	if err != nil {
		t.Fatalf("BuildProofCarryingOutput() error = %v", err)
	}

	if output.OutputBytes != int64(len(validRunnerResultInput().OutputBytes)) {
		t.Fatalf("OutputBytes = %d, want %d", output.OutputBytes, len(validRunnerResultInput().OutputBytes))
	}
	if output.OutputHash == "" {
		t.Fatalf("OutputHash is empty")
	}
	if output.ArtifactObjectID != output.OutputHash {
		t.Fatalf("ArtifactObjectID = %q, want output hash %q", output.ArtifactObjectID, output.OutputHash)
	}
	if output.ArtifactManifest.ObjectID != output.ArtifactObjectID {
		t.Fatalf("manifest ObjectID = %q, want %q", output.ArtifactManifest.ObjectID, output.ArtifactObjectID)
	}
	if output.EvidencePayload.OutputEvidence.OutputHash != output.OutputHash {
		t.Fatalf("evidence OutputHash = %q, want %q", output.EvidencePayload.OutputEvidence.OutputHash, output.OutputHash)
	}
}

func TestBuildProofCarryingOutputChangingOutputChangesHash(t *testing.T) {
	firstInput := validRunnerResultInput()
	secondInput := validRunnerResultInput()
	secondInput.OutputBytes = []byte(`{"result":"changed"}`)

	first, err := BuildProofCarryingOutput(firstInput)
	if err != nil {
		t.Fatalf("first BuildProofCarryingOutput() error = %v", err)
	}
	second, err := BuildProofCarryingOutput(secondInput)
	if err != nil {
		t.Fatalf("second BuildProofCarryingOutput() error = %v", err)
	}

	if first.OutputHash == second.OutputHash {
		t.Fatalf("changed output produced same hash %q", first.OutputHash)
	}
	if first.ArtifactObjectID == second.ArtifactObjectID {
		t.Fatalf("changed output produced same artifact object id %q", first.ArtifactObjectID)
	}
}

func TestBuildProofCarryingOutputInvalidTimestampsRejected(t *testing.T) {
	input := validRunnerResultInput()
	input.FinishedAtUnixMs = input.StartedAtUnixMs - 1

	err := buildProofCarryingOutputError(input)
	if !errors.Is(err, ErrInvalidRunnerResult) || !strings.Contains(err.Error(), "finished_at_unix_ms") {
		t.Fatalf("BuildProofCarryingOutput() error = %v, want invalid timestamp error", err)
	}
}

func TestBuildProofCarryingOutputMissingJobAssignmentNodeRejected(t *testing.T) {
	tests := []struct {
		name string
		edit func(*RunnerResultInput)
		want string
	}{
		{name: "job", edit: func(input *RunnerResultInput) { input.JobID = "" }, want: "job_id required"},
		{name: "assignment", edit: func(input *RunnerResultInput) { input.AssignmentID = "" }, want: "assignment_id required"},
		{name: "node", edit: func(input *RunnerResultInput) { input.NodeID = "" }, want: "node_id required"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			input := validRunnerResultInput()
			tc.edit(&input)

			err := buildProofCarryingOutputError(input)
			if !errors.Is(err, ErrInvalidRunnerResult) || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("BuildProofCarryingOutput() error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestProofCarryingOutputDoesNotStoreRawOutputBytes(t *testing.T) {
	checkNoByteSlices(t, reflect.TypeOf(ProofCarryingRunnerOutput{}))

	input := validRunnerResultInput()
	input.OutputBytes = []byte("raw runner output secret material")
	output, err := BuildProofCarryingOutput(input)
	if err != nil {
		t.Fatalf("BuildProofCarryingOutput() error = %v", err)
	}

	encoded, err := json.Marshal(output)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	if strings.Contains(string(encoded), string(input.OutputBytes)) {
		t.Fatalf("proof output JSON contains raw output bytes: %s", encoded)
	}
}

func TestBuildProofCarryingOutputArtifactsAndEvidenceValidate(t *testing.T) {
	output, err := BuildProofCarryingOutput(validRunnerResultInput())
	if err != nil {
		t.Fatalf("BuildProofCarryingOutput() error = %v", err)
	}

	if err := artifact.ValidateArtifactManifest(output.ArtifactManifest); err != nil {
		t.Fatalf("ValidateArtifactManifest() error = %v", err)
	}
	if err := evidence.ValidateEvidencePayload(output.EvidencePayload); err != nil {
		t.Fatalf("ValidateEvidencePayload() error = %v", err)
	}
	if len(output.CASObjectReferences) != 1 || output.CASObjectReferences[0] != output.ArtifactObjectID {
		t.Fatalf("CASObjectReferences = %v, want artifact object id %q", output.CASObjectReferences, output.ArtifactObjectID)
	}
}

func buildProofCarryingOutputError(input RunnerResultInput) error {
	_, err := BuildProofCarryingOutput(input)
	return err
}

func checkNoByteSlices(t *testing.T, typ reflect.Type) {
	t.Helper()

	for i := 0; i < typ.NumField(); i++ {
		field := typ.Field(i)
		if field.Type.Kind() == reflect.Slice && field.Type.Elem().Kind() == reflect.Uint8 {
			t.Fatalf("%s field %q stores raw bytes", typ.Name(), field.Name)
		}
	}
}

func validRunnerResultInput() RunnerResultInput {
	return RunnerResultInput{
		JobID:            "job-1",
		AssignmentID:     "assignment-1",
		NodeID:           "node-1",
		RunnerKind:       "native_llama",
		AgentVersion:     "dev",
		ModelID:          "llama-3.1-8b.gguf",
		ModelRevision:    "rev-1",
		QuantizationID:   "q4_k_m",
		OutputBytes:      []byte(`{"result":"ok","tokens":12}`),
		MeteringUnits:    12,
		ArtifactKind:     artifact.ArtifactKindResultJSON,
		StartedAtUnixMs:  1_800_000_000_000,
		FinishedAtUnixMs: 1_800_000_001_000,
	}
}
