package artifact

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestBuildArtifactManifestFromBytesValid(t *testing.T) {
	data := []byte(`{"result":"ok","score":1}`)
	manifest, err := BuildArtifactManifestFromBytes(data, ArtifactManifestOptions{
		Kind:            ArtifactKindResultJSON,
		ChunkSizeBytes:  9,
		CreatedAtUnixMs: 1700000000000,
		RuntimeID:       "runtime-1",
		ModelID:         "model-1",
	})
	if err != nil {
		t.Fatalf("BuildArtifactManifestFromBytes() error = %v", err)
	}
	if err := ValidateArtifactManifest(manifest); err != nil {
		t.Fatalf("ValidateArtifactManifest() error = %v", err)
	}
	if manifest.Kind != ArtifactKindResultJSON {
		t.Fatalf("Kind = %q, want %q", manifest.Kind, ArtifactKindResultJSON)
	}
	if manifest.SizeBytes != int64(len(data)) {
		t.Fatalf("SizeBytes = %d, want %d", manifest.SizeBytes, len(data))
	}
	if manifest.ObjectID != objectIDFromHashHex(manifest.SHA256Hex) {
		t.Fatalf("ObjectID = %q, want sha256 object id for %q", manifest.ObjectID, manifest.SHA256Hex)
	}
	if manifest.ArtifactID != manifest.ObjectID {
		t.Fatalf("ArtifactID = %q, want deterministic object id %q", manifest.ArtifactID, manifest.ObjectID)
	}
	if manifest.ContentType != "application/json" {
		t.Fatalf("ContentType = %q, want application/json", manifest.ContentType)
	}
	if len(manifest.Chunks) != 3 {
		t.Fatalf("Chunks length = %d, want 3", len(manifest.Chunks))
	}
	if last := manifest.Chunks[len(manifest.Chunks)-1]; last.SizeBytes >= manifest.ChunkSizeBytes {
		t.Fatalf("last chunk size = %d, want smaller than chunk size %d", last.SizeBytes, manifest.ChunkSizeBytes)
	}
}

func TestSameBytesSameObjectHashAndChangedBytesChangesObjectHash(t *testing.T) {
	options := ArtifactManifestOptions{
		Kind:            ArtifactKindText,
		ChunkSizeBytes:  4,
		CreatedAtUnixMs: 1700000000000,
	}

	first, err := BuildArtifactManifestFromBytes([]byte("same bytes"), options)
	if err != nil {
		t.Fatalf("first BuildArtifactManifestFromBytes() error = %v", err)
	}
	second, err := BuildArtifactManifestFromBytes([]byte("same bytes"), options)
	if err != nil {
		t.Fatalf("second BuildArtifactManifestFromBytes() error = %v", err)
	}
	changed, err := BuildArtifactManifestFromBytes([]byte("changed bytes"), options)
	if err != nil {
		t.Fatalf("changed BuildArtifactManifestFromBytes() error = %v", err)
	}

	if first.ObjectID != second.ObjectID || first.SHA256Hex != second.SHA256Hex {
		t.Fatalf("same bytes produced different identity: %+v vs %+v", first, second)
	}
	if first.ObjectID == changed.ObjectID || first.SHA256Hex == changed.SHA256Hex {
		t.Fatalf("changed bytes produced same identity: %q", first.ObjectID)
	}
}

func TestChunkPlanDeterministic(t *testing.T) {
	options := ArtifactManifestOptions{
		Kind:            ArtifactKindGenericBlob,
		ChunkSizeBytes:  5,
		CreatedAtUnixMs: 1700000000000,
	}
	data := []byte("deterministic chunk plan")

	first, err := BuildArtifactManifestFromBytes(data, options)
	if err != nil {
		t.Fatalf("first BuildArtifactManifestFromBytes() error = %v", err)
	}
	second, err := BuildArtifactManifestFromBytes(data, options)
	if err != nil {
		t.Fatalf("second BuildArtifactManifestFromBytes() error = %v", err)
	}

	if !reflect.DeepEqual(first.Chunks, second.Chunks) {
		t.Fatalf("chunk plan differs:\nfirst=%+v\nsecond=%+v", first.Chunks, second.Chunks)
	}
	for i, chunk := range first.Chunks {
		if chunk.Index != i {
			t.Fatalf("chunk[%d].Index = %d, want %d", i, chunk.Index, i)
		}
		if chunk.OffsetBytes != int64(i)*options.ChunkSizeBytes {
			t.Fatalf("chunk[%d].OffsetBytes = %d, want %d", i, chunk.OffsetBytes, int64(i)*options.ChunkSizeBytes)
		}
	}
}

func TestManifestDoesNotContainRawData(t *testing.T) {
	raw := "raw customer secret payload sk-test-value"
	manifest, err := BuildArtifactManifestFromBytes([]byte(raw), ArtifactManifestOptions{
		Kind:            ArtifactKindRunnerLog,
		ChunkSizeBytes:  16,
		CreatedAtUnixMs: 1700000000000,
	})
	if err != nil {
		t.Fatalf("BuildArtifactManifestFromBytes() error = %v", err)
	}

	encoded, err := json.Marshal(manifest)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	if strings.Contains(string(encoded), raw) || strings.Contains(string(encoded), "sk-test-value") {
		t.Fatalf("manifest JSON contains raw artifact data: %s", encoded)
	}
}

func TestValidateArtifactManifestRejectsInvalidHash(t *testing.T) {
	manifest := validTestManifest(t)
	manifest.SHA256Hex = strings.Repeat("A", sha256HexLength)

	if err := ValidateArtifactManifest(manifest); err == nil {
		t.Fatalf("ValidateArtifactManifest() error = nil, want invalid hash error")
	}
}

func TestValidateArtifactManifestRejectsTotalMismatch(t *testing.T) {
	manifest := validTestManifest(t)
	manifest.SizeBytes++

	if err := ValidateArtifactManifest(manifest); err == nil {
		t.Fatalf("ValidateArtifactManifest() error = nil, want total mismatch error")
	}
}

func validTestManifest(t *testing.T) ArtifactManifest {
	t.Helper()

	manifest, err := BuildArtifactManifestFromBytes([]byte("valid manifest"), ArtifactManifestOptions{
		Kind:            ArtifactKindText,
		ChunkSizeBytes:  4,
		CreatedAtUnixMs: 1700000000000,
	})
	if err != nil {
		t.Fatalf("BuildArtifactManifestFromBytes() error = %v", err)
	}
	return manifest
}
