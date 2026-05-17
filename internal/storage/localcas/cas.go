package localcas

import (
	"context"
	"errors"
	"fmt"
	"time"
)

var (
	ErrObjectNotFound  = errors.New("localcas: object not found")
	ErrInvalidMetadata = errors.New("localcas: invalid metadata")
)

type ObjectMetadata struct {
	ID              ObjectID `json:"id"`
	Kind            string   `json:"kind,omitempty"`
	SizeBytes       int64    `json:"size_bytes"`
	CreatedAtUnixMs int64    `json:"created_at_unix_ms,omitempty"`
	Source          string   `json:"source,omitempty"`
	Labels          []string `json:"labels,omitempty"`
}

type CAS interface {
	PutBytes(ctx context.Context, data []byte, metadata ObjectMetadata) (ObjectID, error)
	GetBytes(ctx context.Context, id ObjectID) ([]byte, ObjectMetadata, error)
	Exists(ctx context.Context, id ObjectID) bool
	Delete(ctx context.Context, id ObjectID) error
	List(ctx context.Context) ([]ObjectMetadata, error)
}

func normalizeObjectMetadata(id ObjectID, sizeBytes int64, metadata ObjectMetadata, nowUnixMs int64) (ObjectMetadata, error) {
	if err := ValidateObjectID(id); err != nil {
		return ObjectMetadata{}, err
	}
	if sizeBytes < 0 {
		return ObjectMetadata{}, fmt.Errorf("%w: size_bytes must be non-negative", ErrInvalidMetadata)
	}
	if metadata.CreatedAtUnixMs < 0 {
		return ObjectMetadata{}, fmt.Errorf("%w: created_at_unix_ms must be non-negative", ErrInvalidMetadata)
	}

	metadata.ID = id
	metadata.SizeBytes = sizeBytes
	if metadata.CreatedAtUnixMs == 0 {
		metadata.CreatedAtUnixMs = nowUnixMs
	}
	metadata.Labels = cloneStrings(metadata.Labels)
	return metadata, nil
}

func cloneObjectMetadata(metadata ObjectMetadata) ObjectMetadata {
	metadata.Labels = cloneStrings(metadata.Labels)
	return metadata
}

func cloneBytes(data []byte) []byte {
	if data == nil {
		return nil
	}
	cloned := make([]byte, len(data))
	copy(cloned, data)
	return cloned
}

func cloneStrings(values []string) []string {
	if values == nil {
		return nil
	}
	cloned := make([]string, len(values))
	copy(cloned, values)
	return cloned
}

func currentUnixMs() int64 {
	return time.Now().UnixNano() / int64(time.Millisecond)
}

func contextError(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	return ctx.Err()
}
