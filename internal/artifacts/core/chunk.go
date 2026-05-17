package artifact

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
)

type ArtifactChunk struct {
	Index       int    `json:"index"`
	OffsetBytes int64  `json:"offset_bytes"`
	SizeBytes   int64  `json:"size_bytes"`
	SHA256Hex   string `json:"sha256_hex"`
}

func buildArtifactChunks(reader io.Reader, sizeBytes int64, chunkSizeBytes int64) (string, []ArtifactChunk, error) {
	if sizeBytes < 0 {
		return "", nil, fmt.Errorf("%w: size_bytes must be non-negative", ErrInvalidArtifactOptions)
	}
	if chunkSizeBytes <= 0 {
		return "", nil, fmt.Errorf("%w: chunk_size_bytes must be greater than zero", ErrInvalidArtifactOptions)
	}
	if chunkSizeBytes > maxInt() {
		return "", nil, fmt.Errorf("%w: chunk_size_bytes too large", ErrInvalidArtifactOptions)
	}

	fullHash := sha256.New()
	if sizeBytes == 0 {
		return hashHex(fullHash), nil, nil
	}

	buffer := make([]byte, int(chunkSizeBytes))
	var chunks []ArtifactChunk
	for index, offset := 0, int64(0); offset < sizeBytes; index++ {
		size := chunkSizeBytes
		if remaining := sizeBytes - offset; remaining < size {
			size = remaining
		}

		chunkBytes := buffer[:int(size)]
		if _, err := io.ReadFull(reader, chunkBytes); err != nil {
			return "", nil, fmt.Errorf("artifact: read chunk %d: %w", index, err)
		}
		if _, err := fullHash.Write(chunkBytes); err != nil {
			return "", nil, fmt.Errorf("artifact: hash chunk %d: %w", index, err)
		}
		chunkHash := sha256.Sum256(chunkBytes)
		chunks = append(chunks, ArtifactChunk{
			Index:       index,
			OffsetBytes: offset,
			SizeBytes:   size,
			SHA256Hex:   hex.EncodeToString(chunkHash[:]),
		})
		offset += size
	}
	return hashHex(fullHash), chunks, nil
}

func validateChunks(manifest ArtifactManifest) error {
	var total int64
	var errs []error
	for expectedIndex, chunk := range manifest.Chunks {
		if chunk.Index != expectedIndex {
			errs = append(errs, fmt.Errorf("%w: chunk index %d out of order", ErrInvalidArtifactManifest, chunk.Index))
		}
		if chunk.OffsetBytes != total {
			errs = append(errs, fmt.Errorf("%w: chunk %d offset mismatch", ErrInvalidArtifactManifest, chunk.Index))
		}
		if chunk.SizeBytes <= 0 {
			errs = append(errs, fmt.Errorf("%w: chunk %d size must be greater than zero", ErrInvalidArtifactManifest, chunk.Index))
		}
		if manifest.ChunkSizeBytes > 0 && chunk.SizeBytes > manifest.ChunkSizeBytes {
			errs = append(errs, fmt.Errorf("%w: chunk %d exceeds chunk_size_bytes", ErrInvalidArtifactManifest, chunk.Index))
		}
		if expectedIndex < len(manifest.Chunks)-1 && manifest.ChunkSizeBytes > 0 && chunk.SizeBytes != manifest.ChunkSizeBytes {
			errs = append(errs, fmt.Errorf("%w: chunk %d must equal chunk_size_bytes unless it is final", ErrInvalidArtifactManifest, chunk.Index))
		}
		if err := validateSHA256Hex(chunk.SHA256Hex, fmt.Sprintf("chunks[%d].sha256_hex", expectedIndex)); err != nil {
			errs = append(errs, err)
		}
		if chunk.SizeBytes > 0 {
			total += chunk.SizeBytes
		}
	}
	if total != manifest.SizeBytes {
		errs = append(errs, fmt.Errorf("%w: chunk total %d does not match size_bytes %d", ErrInvalidArtifactManifest, total, manifest.SizeBytes))
	}
	if manifest.SizeBytes > 0 && len(manifest.Chunks) == 0 {
		errs = append(errs, fmt.Errorf("%w: chunks required for non-empty artifact", ErrInvalidArtifactManifest))
	}
	return errors.Join(errs...)
}

func hashHex(hasher hash.Hash) string {
	return hex.EncodeToString(hasher.Sum(nil))
}

func objectIDFromHashHex(hashHex string) string {
	return objectIDPrefix + hashHex
}

func maxInt() int64 {
	return int64(int(^uint(0) >> 1))
}
