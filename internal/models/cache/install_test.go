package modelcache

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestInstallDownloadedModelMovesArtifactAtomically(t *testing.T) {
	t.Parallel()

	cacheDir := t.TempDir()
	source := filepath.Join(cacheDir, ".prepare-test.tmp")
	if err := os.WriteFile(source, []byte("gguf-model"), 0o600); err != nil {
		t.Fatalf("write source: %v", err)
	}

	result, err := InstallDownloadedModel(InstallOptions{
		CacheDir:     cacheDir,
		ModelID:      "TinyLlama.Q4_K_M.gguf",
		ArtifactURI:  "https://models.example/TinyLlama.Q4_K_M.gguf",
		SourcePath:   source,
		HashVerified: true,
		Now: func() time.Time {
			return time.Unix(1_800_000_000, 0)
		},
	})
	if err != nil {
		t.Fatalf("InstallDownloadedModel() error = %v", err)
	}
	if _, err := os.Stat(source); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("source still exists after install, stat err = %v", err)
	}
	wantPath, err := ModelPath(cacheDir, "TinyLlama.Q4_K_M.gguf", "")
	if err != nil {
		t.Fatalf("ModelPath() error = %v", err)
	}
	if result.DestinationPath != wantPath || result.Model.Path != wantPath {
		t.Fatalf("destination = %q/%q, want %q", result.DestinationPath, result.Model.Path, wantPath)
	}
	body, err := os.ReadFile(wantPath)
	if err != nil {
		t.Fatalf("read installed model: %v", err)
	}
	if string(body) != "gguf-model" {
		t.Fatalf("installed body = %q", body)
	}
	if !result.Model.HashVerified || !result.HashVerified || !result.Model.Installed {
		t.Fatalf("installed model flags = %+v result=%+v", result.Model, result)
	}
}

func TestInstallDownloadedModelDoesNotOverwriteExistingModel(t *testing.T) {
	t.Parallel()

	cacheDir := t.TempDir()
	destination, err := ModelPath(cacheDir, "TinyLlama.Q4_K_M.gguf", "")
	if err != nil {
		t.Fatalf("ModelPath() error = %v", err)
	}
	if err := os.WriteFile(destination, []byte("existing"), 0o644); err != nil {
		t.Fatalf("write existing: %v", err)
	}
	source := filepath.Join(cacheDir, ".prepare-test.tmp")
	if err := os.WriteFile(source, []byte("new"), 0o600); err != nil {
		t.Fatalf("write source: %v", err)
	}
	_, err = InstallDownloadedModel(InstallOptions{
		CacheDir:   cacheDir,
		ModelID:    "TinyLlama.Q4_K_M.gguf",
		SourcePath: source,
	})
	if !errors.Is(err, ErrModelExists) {
		t.Fatalf("InstallDownloadedModel() error = %v, want ErrModelExists", err)
	}
	body, err := os.ReadFile(destination)
	if err != nil {
		t.Fatalf("read existing: %v", err)
	}
	if string(body) != "existing" {
		t.Fatalf("existing model overwritten: %q", body)
	}
}
