package artifact

import (
	"os"
	"path/filepath"
	"testing"
)

func TestBuildArtifactManifestFromFileValid(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "result.json")
	data := []byte(`{"file":true,"status":"ok"}`)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("os.WriteFile() error = %v", err)
	}

	fileManifest, err := BuildArtifactManifestFromFile(path, ArtifactManifestOptions{
		ArtifactID:      "artifact-file-1",
		Kind:            ArtifactKindResultJSON,
		ChunkSizeBytes:  7,
		CreatedAtUnixMs: 1700000000000,
	})
	if err != nil {
		t.Fatalf("BuildArtifactManifestFromFile() error = %v", err)
	}
	if err := ValidateArtifactManifest(fileManifest); err != nil {
		t.Fatalf("ValidateArtifactManifest(file) error = %v", err)
	}

	bytesManifest, err := BuildArtifactManifestFromBytes(data, ArtifactManifestOptions{
		Kind:            ArtifactKindResultJSON,
		ChunkSizeBytes:  7,
		CreatedAtUnixMs: 1700000000000,
	})
	if err != nil {
		t.Fatalf("BuildArtifactManifestFromBytes() error = %v", err)
	}
	if fileManifest.ArtifactID != "artifact-file-1" {
		t.Fatalf("ArtifactID = %q, want caller-provided artifact-file-1", fileManifest.ArtifactID)
	}
	if fileManifest.ObjectID != bytesManifest.ObjectID {
		t.Fatalf("file ObjectID = %q, want bytes ObjectID %q", fileManifest.ObjectID, bytesManifest.ObjectID)
	}
	if fileManifest.SHA256Hex != bytesManifest.SHA256Hex {
		t.Fatalf("file SHA256Hex = %q, want bytes SHA256Hex %q", fileManifest.SHA256Hex, bytesManifest.SHA256Hex)
	}
	if len(fileManifest.Chunks) != len(bytesManifest.Chunks) {
		t.Fatalf("file chunk count = %d, want %d", len(fileManifest.Chunks), len(bytesManifest.Chunks))
	}
	for i := range fileManifest.Chunks {
		if fileManifest.Chunks[i] != bytesManifest.Chunks[i] {
			t.Fatalf("file chunk[%d] = %+v, want %+v", i, fileManifest.Chunks[i], bytesManifest.Chunks[i])
		}
	}
}
