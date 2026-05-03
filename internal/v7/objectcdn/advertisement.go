package objectcdn

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/Ryvion/node-agent/internal/v7/localcas"
)

var ErrInvalidCacheAdvertisement = errors.New("objectcdn: invalid cache advertisement")

type CacheAdvertisement struct {
	NodeID            string               `json:"node_id"`
	GeneratedAtUnixMs int64                `json:"generated_at_unix_ms"`
	Objects           []CacheObjectSummary `json:"objects"`
	TotalBytes        int64                `json:"total_bytes"`
	Truncated         bool                 `json:"truncated"`
}

type CacheObjectSummary struct {
	ObjectID        string `json:"object_id"`
	Kind            string `json:"kind"`
	SizeBytes       int64  `json:"size_bytes"`
	CreatedAtUnixMs int64  `json:"created_at_unix_ms"`
}

func BuildCacheAdvertisement(nodeID string, store localcas.CAS, limit int) (CacheAdvertisement, error) {
	nodeID = strings.TrimSpace(nodeID)
	if nodeID == "" {
		return CacheAdvertisement{}, fmt.Errorf("%w: node_id required", ErrInvalidCacheAdvertisement)
	}
	if store == nil {
		return CacheAdvertisement{}, fmt.Errorf("%w: cas required", ErrInvalidCacheAdvertisement)
	}
	if limit < 0 {
		return CacheAdvertisement{}, fmt.Errorf("%w: limit must be non-negative", ErrInvalidCacheAdvertisement)
	}

	metadata, err := store.List(context.Background())
	if err != nil {
		return CacheAdvertisement{}, err
	}
	sort.Slice(metadata, func(i, j int) bool {
		return string(metadata[i].ID) < string(metadata[j].ID)
	})

	objects := make([]CacheObjectSummary, 0, advertisedObjectCapacity(len(metadata), limit))
	var totalBytes int64
	truncated := false
	for i, item := range metadata {
		summary := cacheObjectSummaryFromMetadata(item)
		if err := validateCacheObjectSummary(summary); err != nil {
			return CacheAdvertisement{}, fmt.Errorf("object[%d]: %w", i, err)
		}
		if summary.SizeBytes > math.MaxInt64-totalBytes {
			return CacheAdvertisement{}, fmt.Errorf("%w: total_bytes overflow", ErrInvalidCacheAdvertisement)
		}
		totalBytes += summary.SizeBytes

		if limit == 0 || len(objects) >= limit {
			truncated = true
			continue
		}
		objects = append(objects, summary)
	}

	advertisement := CacheAdvertisement{
		NodeID:            nodeID,
		GeneratedAtUnixMs: time.Now().UnixMilli(),
		Objects:           objects,
		TotalBytes:        totalBytes,
		Truncated:         truncated,
	}
	if err := ValidateCacheAdvertisement(advertisement); err != nil {
		return CacheAdvertisement{}, err
	}
	return advertisement, nil
}

func ValidateCacheAdvertisement(advertisement CacheAdvertisement) error {
	var errs []error
	if strings.TrimSpace(advertisement.NodeID) == "" {
		errs = append(errs, fmt.Errorf("%w: node_id required", ErrInvalidCacheAdvertisement))
	}
	if advertisement.GeneratedAtUnixMs < 0 {
		errs = append(errs, fmt.Errorf("%w: generated_at_unix_ms must be non-negative", ErrInvalidCacheAdvertisement))
	}
	if advertisement.TotalBytes < 0 {
		errs = append(errs, fmt.Errorf("%w: total_bytes must be non-negative", ErrInvalidCacheAdvertisement))
	}

	var previousObjectID string
	var advertisedBytes int64
	for i, object := range advertisement.Objects {
		if err := validateCacheObjectSummary(object); err != nil {
			errs = append(errs, fmt.Errorf("object[%d]: %w", i, err))
			continue
		}

		objectID := strings.TrimSpace(object.ObjectID)
		if previousObjectID != "" && objectID <= previousObjectID {
			errs = append(errs, fmt.Errorf("%w: objects must be sorted by unique object_id", ErrInvalidCacheAdvertisement))
		}
		previousObjectID = objectID

		if object.SizeBytes > math.MaxInt64-advertisedBytes {
			errs = append(errs, fmt.Errorf("%w: advertised object bytes overflow", ErrInvalidCacheAdvertisement))
			continue
		}
		advertisedBytes += object.SizeBytes
	}

	if advertisement.Truncated {
		if advertisement.TotalBytes < advertisedBytes {
			errs = append(errs, fmt.Errorf("%w: total_bytes must cover advertised objects", ErrInvalidCacheAdvertisement))
		}
	} else if advertisement.TotalBytes != advertisedBytes {
		errs = append(errs, fmt.Errorf("%w: total_bytes must equal advertised object bytes when not truncated", ErrInvalidCacheAdvertisement))
	}

	return errors.Join(errs...)
}

func cacheObjectSummaryFromMetadata(metadata localcas.ObjectMetadata) CacheObjectSummary {
	return CacheObjectSummary{
		ObjectID:        strings.TrimSpace(string(metadata.ID)),
		Kind:            strings.TrimSpace(metadata.Kind),
		SizeBytes:       metadata.SizeBytes,
		CreatedAtUnixMs: metadata.CreatedAtUnixMs,
	}
}

func validateCacheObjectSummary(summary CacheObjectSummary) error {
	var errs []error
	objectID := strings.TrimSpace(summary.ObjectID)
	if err := localcas.ValidateObjectID(localcas.ObjectID(objectID)); err != nil {
		errs = append(errs, fmt.Errorf("%w: object_id must be sha256:<64 lowercase hex>", ErrInvalidCacheAdvertisement))
	}
	if summary.SizeBytes < 0 {
		errs = append(errs, fmt.Errorf("%w: size_bytes must be non-negative", ErrInvalidCacheAdvertisement))
	}
	if summary.CreatedAtUnixMs < 0 {
		errs = append(errs, fmt.Errorf("%w: created_at_unix_ms must be non-negative", ErrInvalidCacheAdvertisement))
	}
	return errors.Join(errs...)
}

func advertisedObjectCapacity(objectCount int, limit int) int {
	if limit == 0 {
		return 0
	}
	if limit > objectCount {
		return objectCount
	}
	return limit
}
