package evidence

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
)

const (
	evidenceHashHex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	evidenceHashID  = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
)

func TestValidateEvidencePayloadValid(t *testing.T) {
	if err := ValidateEvidencePayload(validEvidencePayload()); err != nil {
		t.Fatalf("ValidateEvidencePayload() error = %v, want nil", err)
	}
}

func TestValidateEvidencePayloadMissingJobAssignmentNodeRejected(t *testing.T) {
	tests := []struct {
		name string
		edit func(*RYV3EvidencePayload)
		want string
	}{
		{name: "job", edit: func(payload *RYV3EvidencePayload) { payload.JobID = "" }, want: "job_id required"},
		{name: "assignment", edit: func(payload *RYV3EvidencePayload) { payload.AssignmentID = "" }, want: "assignment_id required"},
		{name: "node", edit: func(payload *RYV3EvidencePayload) { payload.NodeID = "" }, want: "node_id required"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			payload := validEvidencePayload()
			tc.edit(&payload)

			err := ValidateEvidencePayload(payload)
			if !errors.Is(err, ErrInvalidEvidencePayload) || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("ValidateEvidencePayload() error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestValidateEvidencePayloadMalformedOutputHashRejected(t *testing.T) {
	hashes := []string{
		"",
		"sha256:",
		"sha256:" + strings.Repeat("0", 63),
		"sha256:" + strings.Repeat("0", 65),
		"sha256:" + strings.Repeat("A", 64),
		"md5:" + strings.Repeat("0", 64),
	}
	for _, hash := range hashes {
		t.Run(hash, func(t *testing.T) {
			payload := validEvidencePayload()
			payload.OutputEvidence.OutputHash = hash

			err := ValidateEvidencePayload(payload)
			if !errors.Is(err, ErrInvalidEvidencePayload) || !strings.Contains(err.Error(), "output_evidence.output_hash") {
				t.Fatalf("ValidateEvidencePayload() error = %v, want output hash error", err)
			}
		})
	}
}

func TestValidateEvidencePayloadInvalidTimestampOrderRejected(t *testing.T) {
	payload := validEvidencePayload()
	payload.ExecutionStartedAtUnixMs = 1_800_000_000_000
	payload.ExecutionFinishedAtUnixMs = 1_799_999_999_999

	err := ValidateEvidencePayload(payload)
	if !errors.Is(err, ErrInvalidEvidencePayload) || !strings.Contains(err.Error(), "execution_finished_at_unix_ms") {
		t.Fatalf("ValidateEvidencePayload() error = %v, want timestamp order error", err)
	}
}

func TestValidateEvidencePayloadArtifactHashValidation(t *testing.T) {
	tests := []struct {
		name string
		edit func(*ArtifactEvidence)
		want string
	}{
		{
			name: "bad sha hex",
			edit: func(artifact *ArtifactEvidence) {
				artifact.SHA256Hex = strings.Repeat("g", 64)
			},
			want: "sha256_hex",
		},
		{
			name: "bad object id",
			edit: func(artifact *ArtifactEvidence) {
				artifact.ObjectID = "sha256:" + strings.Repeat("0", 63)
			},
			want: "object_id",
		},
		{
			name: "mismatch",
			edit: func(artifact *ArtifactEvidence) {
				artifact.SHA256Hex = strings.Repeat("1", 64)
			},
			want: "object_id must match sha256_hex",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			payload := validEvidencePayload()
			tc.edit(&payload.Artifacts[0])

			err := ValidateEvidencePayload(payload)
			if !errors.Is(err, ErrInvalidEvidencePayload) || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("ValidateEvidencePayload() error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestEvidencePayloadDoesNotExposeRawPromptTranscriptOrOutputBytes(t *testing.T) {
	checkNoRawPromptTranscriptFields(t, reflect.TypeOf(RYV3EvidencePayload{}))
	checkNoRawPromptTranscriptFields(t, reflect.TypeOf(RuntimeEvidence{}))
	checkNoRawPromptTranscriptFields(t, reflect.TypeOf(ModelEvidence{}))
	checkNoRawPromptTranscriptFields(t, reflect.TypeOf(OutputEvidence{}))
	checkNoRawPromptTranscriptFields(t, reflect.TypeOf(ArtifactEvidence{}))

	payload := validEvidencePayload()
	encoded, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	for _, forbidden := range []string{"raw customer prompt", "raw transcript", "prompt_text", "transcript"} {
		if strings.Contains(string(encoded), forbidden) {
			t.Fatalf("payload JSON contains forbidden raw field/data %q: %s", forbidden, encoded)
		}
	}
}

func TestValidateEvidencePayloadRejectsUnsafeModelFormat(t *testing.T) {
	payload := validEvidencePayload()
	payload.ModelEvidence.ModelFormat = "pytorch_pickle"

	err := ValidateEvidencePayload(payload)
	if !errors.Is(err, ErrInvalidEvidencePayload) || !strings.Contains(err.Error(), "model_evidence.model_format") {
		t.Fatalf("ValidateEvidencePayload() error = %v, want model format error", err)
	}
}

func checkNoRawPromptTranscriptFields(t *testing.T, typ reflect.Type) {
	t.Helper()

	for i := 0; i < typ.NumField(); i++ {
		field := typ.Field(i)
		fieldName := strings.ToLower(field.Name)
		jsonName := strings.ToLower(strings.Split(field.Tag.Get("json"), ",")[0])
		if strings.Contains(fieldName, "prompt") || strings.Contains(jsonName, "prompt") ||
			strings.Contains(fieldName, "transcript") || strings.Contains(jsonName, "transcript") {
			t.Fatalf("%s field %q exposes raw prompt/transcript data", typ.Name(), field.Name)
		}
		if field.Type.Kind() == reflect.Slice && field.Type.Elem().Kind() == reflect.Uint8 {
			t.Fatalf("%s field %q exposes raw byte content", typ.Name(), field.Name)
		}
	}
}

func validEvidencePayload() RYV3EvidencePayload {
	return RYV3EvidencePayload{
		SchemaVersion:             SchemaVersionV1,
		JobID:                     "job-1",
		AssignmentID:              "assignment-1",
		NodeID:                    "node-1",
		ExecutionStartedAtUnixMs:  1_800_000_000_000,
		ExecutionFinishedAtUnixMs: 1_800_000_001_000,
		RuntimeEvidence: RuntimeEvidence{
			AgentVersion: "dev",
			RunnerKind:   "native_llama",
			RuntimeHash:  evidenceHashID,
			OS:           "linux",
			Arch:         "amd64",
		},
		ModelEvidence: ModelEvidence{
			ModelID:        "llama-3.1-8b",
			ModelRevision:  "rev-1",
			QuantizationID: "q4_k_m",
			ModelObjectID:  evidenceHashID,
			ModelFormat:    "gguf",
		},
		OutputEvidence: OutputEvidence{
			OutputHash:      evidenceHashID,
			OutputHashCodec: "sha256",
			OutputBytes:     128,
			MeteringUnits:   42,
		},
		Artifacts: []ArtifactEvidence{
			{
				ArtifactID: "artifact-1",
				Kind:       "result_json",
				ObjectID:   evidenceHashID,
				SHA256Hex:  evidenceHashHex,
				SizeBytes:  128,
			},
		},
	}
}
