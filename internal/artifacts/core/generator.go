package artifact

import (
	"bytes"
	"fmt"
	"os"
	"strings"
	"time"
)

func BuildArtifactManifestFromBytes(data []byte, options ArtifactManifestOptions) (ArtifactManifest, error) {
	normalized, err := normalizeArtifactManifestOptions(options)
	if err != nil {
		return ArtifactManifest{}, err
	}

	sha256Hex, chunks, err := buildArtifactChunks(bytes.NewReader(data), int64(len(data)), normalized.ChunkSizeBytes)
	if err != nil {
		return ArtifactManifest{}, err
	}
	return buildArtifactManifest(sha256Hex, int64(len(data)), chunks, normalized)
}

func BuildArtifactManifestFromFile(path string, options ArtifactManifestOptions) (ArtifactManifest, error) {
	if strings.TrimSpace(path) == "" {
		return ArtifactManifest{}, fmt.Errorf("%w: path required", ErrInvalidArtifactOptions)
	}

	file, err := os.Open(path)
	if err != nil {
		return ArtifactManifest{}, err
	}
	defer file.Close()

	info, err := file.Stat()
	if err != nil {
		return ArtifactManifest{}, err
	}
	if info.IsDir() {
		return ArtifactManifest{}, fmt.Errorf("%w: path is a directory", ErrInvalidArtifactOptions)
	}

	normalized, err := normalizeArtifactManifestOptions(options)
	if err != nil {
		return ArtifactManifest{}, err
	}

	sha256Hex, chunks, err := buildArtifactChunks(file, info.Size(), normalized.ChunkSizeBytes)
	if err != nil {
		return ArtifactManifest{}, err
	}
	return buildArtifactManifest(sha256Hex, info.Size(), chunks, normalized)
}

func normalizeArtifactManifestOptions(options ArtifactManifestOptions) (ArtifactManifestOptions, error) {
	options.ArtifactID = strings.TrimSpace(options.ArtifactID)
	options.ContentType = strings.TrimSpace(options.ContentType)
	options.RuntimeID = strings.TrimSpace(options.RuntimeID)
	options.ModelID = strings.TrimSpace(options.ModelID)

	if options.Kind == "" {
		options.Kind = ArtifactKindGenericBlob
	}
	if !validArtifactKind(options.Kind) {
		return ArtifactManifestOptions{}, fmt.Errorf("%w: invalid kind %q", ErrInvalidArtifactOptions, options.Kind)
	}
	if options.ChunkSizeBytes == 0 {
		options.ChunkSizeBytes = DefaultArtifactChunkSizeBytes
	}
	if options.ChunkSizeBytes < 0 {
		return ArtifactManifestOptions{}, fmt.Errorf("%w: chunk_size_bytes must be non-negative", ErrInvalidArtifactOptions)
	}
	if options.CreatedAtUnixMs < 0 {
		return ArtifactManifestOptions{}, fmt.Errorf("%w: created_at_unix_ms must be non-negative", ErrInvalidArtifactOptions)
	}
	if options.CreatedAtUnixMs == 0 {
		options.CreatedAtUnixMs = time.Now().UnixNano() / int64(time.Millisecond)
	}
	if options.ContentType == "" {
		options.ContentType = defaultContentType(options.Kind)
	}
	return options, nil
}

func buildArtifactManifest(sha256Hex string, sizeBytes int64, chunks []ArtifactChunk, options ArtifactManifestOptions) (ArtifactManifest, error) {
	objectID := objectIDFromHashHex(sha256Hex)
	artifactID := options.ArtifactID
	if artifactID == "" {
		artifactID = objectID
	}

	manifest := ArtifactManifest{
		ArtifactID:      artifactID,
		Kind:            options.Kind,
		ObjectID:        objectID,
		SizeBytes:       sizeBytes,
		SHA256Hex:       sha256Hex,
		ContentType:     options.ContentType,
		ChunkSizeBytes:  options.ChunkSizeBytes,
		Chunks:          cloneArtifactChunks(chunks),
		CreatedAtUnixMs: options.CreatedAtUnixMs,
		RuntimeID:       options.RuntimeID,
		ModelID:         options.ModelID,
	}
	if err := ValidateArtifactManifest(manifest); err != nil {
		return ArtifactManifest{}, err
	}
	return manifest, nil
}

func defaultContentType(kind ArtifactKind) string {
	switch kind {
	case ArtifactKindResultJSON:
		return "application/json"
	case ArtifactKindText, ArtifactKindRunnerLog:
		return "text/plain; charset=utf-8"
	default:
		return "application/octet-stream"
	}
}

func cloneArtifactChunks(chunks []ArtifactChunk) []ArtifactChunk {
	if chunks == nil {
		return nil
	}
	cloned := make([]ArtifactChunk, len(chunks))
	copy(cloned, chunks)
	return cloned
}
