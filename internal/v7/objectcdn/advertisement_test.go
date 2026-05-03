package objectcdn

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/Ryvion/node-agent/internal/v7/localcas"
)

func TestBuildCacheAdvertisementFromCAS(t *testing.T) {
	ctx := context.Background()
	store := localcas.NewInMemoryCAS()

	firstID := putCacheObject(t, ctx, store, []byte("artifact-alpha"), "artifact", 1_000)
	secondID := putCacheObject(t, ctx, store, []byte("runner-layer-beta"), "runner-layer", 2_000)

	advertisement, err := BuildCacheAdvertisement(" node-1 ", store, 10)
	if err != nil {
		t.Fatalf("BuildCacheAdvertisement() error = %v", err)
	}
	if err := ValidateCacheAdvertisement(advertisement); err != nil {
		t.Fatalf("ValidateCacheAdvertisement() error = %v", err)
	}
	if advertisement.NodeID != "node-1" {
		t.Fatalf("NodeID = %q, want node-1", advertisement.NodeID)
	}
	if advertisement.GeneratedAtUnixMs <= 0 {
		t.Fatalf("GeneratedAtUnixMs = %d, want positive timestamp", advertisement.GeneratedAtUnixMs)
	}
	if advertisement.Truncated {
		t.Fatalf("Truncated = true, want false")
	}
	wantTotal := int64(len("artifact-alpha") + len("runner-layer-beta"))
	if advertisement.TotalBytes != wantTotal {
		t.Fatalf("TotalBytes = %d, want %d", advertisement.TotalBytes, wantTotal)
	}
	if len(advertisement.Objects) != 2 {
		t.Fatalf("Objects length = %d, want 2", len(advertisement.Objects))
	}

	wantIDs := sortedObjectIDStrings(firstID, secondID)
	gotIDs := cacheAdvertisementObjectIDs(advertisement)
	if !reflect.DeepEqual(gotIDs, wantIDs) {
		t.Fatalf("object ids = %v, want %v", gotIDs, wantIDs)
	}
	for _, object := range advertisement.Objects {
		switch object.ObjectID {
		case string(firstID):
			if object.Kind != "artifact" || object.SizeBytes != int64(len("artifact-alpha")) || object.CreatedAtUnixMs != 1_000 {
				t.Fatalf("first object summary = %+v, want artifact metadata", object)
			}
		case string(secondID):
			if object.Kind != "runner-layer" || object.SizeBytes != int64(len("runner-layer-beta")) || object.CreatedAtUnixMs != 2_000 {
				t.Fatalf("second object summary = %+v, want runner-layer metadata", object)
			}
		default:
			t.Fatalf("unexpected object summary id %q", object.ObjectID)
		}
	}
}

func TestBuildCacheAdvertisementLimitTruncates(t *testing.T) {
	objects := []localcas.ObjectMetadata{
		cacheMetadata("charlie", "artifact", 10, 3_000),
		cacheMetadata("alpha", "artifact", 20, 1_000),
		cacheMetadata("bravo", "runner-layer", 30, 2_000),
	}
	store := &metadataOnlyCAS{metadata: objects}

	advertisement, err := BuildCacheAdvertisement("node-1", store, 2)
	if err != nil {
		t.Fatalf("BuildCacheAdvertisement() error = %v", err)
	}
	if !advertisement.Truncated {
		t.Fatalf("Truncated = false, want true")
	}
	if len(advertisement.Objects) != 2 {
		t.Fatalf("Objects length = %d, want 2", len(advertisement.Objects))
	}
	if advertisement.TotalBytes != 60 {
		t.Fatalf("TotalBytes = %d, want full CAS total 60", advertisement.TotalBytes)
	}

	allIDs := sortedObjectIDStrings(objects[0].ID, objects[1].ID, objects[2].ID)
	gotIDs := cacheAdvertisementObjectIDs(advertisement)
	if !reflect.DeepEqual(gotIDs, allIDs[:2]) {
		t.Fatalf("limited object ids = %v, want first two sorted ids %v", gotIDs, allIDs[:2])
	}
}

func TestCacheAdvertisementInvalidNodeRejected(t *testing.T) {
	if _, err := BuildCacheAdvertisement(" \t", localcas.NewInMemoryCAS(), 10); err == nil {
		t.Fatalf("BuildCacheAdvertisement(blank node) error = nil, want error")
	}

	err := ValidateCacheAdvertisement(CacheAdvertisement{
		NodeID:            " ",
		GeneratedAtUnixMs: 1,
		Objects:           []CacheObjectSummary{},
		TotalBytes:        0,
	})
	if err == nil || !strings.Contains(err.Error(), "node_id required") {
		t.Fatalf("ValidateCacheAdvertisement(blank node) error = %v, want node_id required", err)
	}
}

func TestBuildCacheAdvertisementDoesNotExposeRawData(t *testing.T) {
	raw := []byte("raw-secret-object-data")
	store := &metadataOnlyCAS{
		metadata: []localcas.ObjectMetadata{{
			ID:              localcas.HashBytes(raw),
			Kind:            "artifact",
			SizeBytes:       int64(len(raw)),
			CreatedAtUnixMs: 1_000,
		}},
	}

	advertisement, err := BuildCacheAdvertisement("node-1", store, 10)
	if err != nil {
		t.Fatalf("BuildCacheAdvertisement() error = %v", err)
	}
	if store.getCalled {
		t.Fatalf("BuildCacheAdvertisement() called GetBytes, want metadata-only advertisement")
	}

	encoded, err := json.Marshal(advertisement)
	if err != nil {
		t.Fatalf("Marshal(advertisement) error = %v", err)
	}
	if bytes.Contains(encoded, raw) {
		t.Fatalf("advertisement JSON contains raw object data %q", raw)
	}
}

func TestBuildCacheAdvertisementDeterministicOrdering(t *testing.T) {
	metadata := []localcas.ObjectMetadata{
		cacheMetadata("delta", "artifact", 4, 4_000),
		cacheMetadata("alpha", "artifact", 1, 1_000),
		cacheMetadata("charlie", "artifact", 3, 3_000),
		cacheMetadata("bravo", "artifact", 2, 2_000),
	}
	store := &metadataOnlyCAS{metadata: metadata}

	first, err := BuildCacheAdvertisement("node-1", store, 10)
	if err != nil {
		t.Fatalf("BuildCacheAdvertisement(first) error = %v", err)
	}
	reversed := append([]localcas.ObjectMetadata(nil), metadata...)
	for i, j := 0, len(reversed)-1; i < j; i, j = i+1, j-1 {
		reversed[i], reversed[j] = reversed[j], reversed[i]
	}
	store.metadata = reversed

	second, err := BuildCacheAdvertisement("node-1", store, 10)
	if err != nil {
		t.Fatalf("BuildCacheAdvertisement(second) error = %v", err)
	}

	wantIDs := sortedObjectIDStrings(metadata[0].ID, metadata[1].ID, metadata[2].ID, metadata[3].ID)
	if gotIDs := cacheAdvertisementObjectIDs(first); !reflect.DeepEqual(gotIDs, wantIDs) {
		t.Fatalf("first object ids = %v, want %v", gotIDs, wantIDs)
	}
	if gotIDs := cacheAdvertisementObjectIDs(second); !reflect.DeepEqual(gotIDs, wantIDs) {
		t.Fatalf("second object ids = %v, want %v", gotIDs, wantIDs)
	}
}

func putCacheObject(t *testing.T, ctx context.Context, store localcas.CAS, data []byte, kind string, createdAtUnixMs int64) localcas.ObjectID {
	t.Helper()

	id, err := store.PutBytes(ctx, data, localcas.ObjectMetadata{
		Kind:            kind,
		CreatedAtUnixMs: createdAtUnixMs,
	})
	if err != nil {
		t.Fatalf("PutBytes(%q) error = %v", data, err)
	}
	return id
}

func cacheMetadata(seed string, kind string, sizeBytes int64, createdAtUnixMs int64) localcas.ObjectMetadata {
	return localcas.ObjectMetadata{
		ID:              localcas.HashBytes([]byte(seed)),
		Kind:            kind,
		SizeBytes:       sizeBytes,
		CreatedAtUnixMs: createdAtUnixMs,
	}
}

func cacheAdvertisementObjectIDs(advertisement CacheAdvertisement) []string {
	ids := make([]string, 0, len(advertisement.Objects))
	for _, object := range advertisement.Objects {
		ids = append(ids, object.ObjectID)
	}
	return ids
}

func sortedObjectIDStrings(ids ...localcas.ObjectID) []string {
	values := make([]string, 0, len(ids))
	for _, id := range ids {
		values = append(values, string(id))
	}
	sort.Strings(values)
	return values
}

type metadataOnlyCAS struct {
	metadata  []localcas.ObjectMetadata
	getCalled bool
}

func (store *metadataOnlyCAS) PutBytes(context.Context, []byte, localcas.ObjectMetadata) (localcas.ObjectID, error) {
	return "", errors.New("unexpected PutBytes call")
}

func (store *metadataOnlyCAS) GetBytes(context.Context, localcas.ObjectID) ([]byte, localcas.ObjectMetadata, error) {
	store.getCalled = true
	return nil, localcas.ObjectMetadata{}, errors.New("unexpected GetBytes call")
}

func (store *metadataOnlyCAS) Exists(context.Context, localcas.ObjectID) bool {
	return false
}

func (store *metadataOnlyCAS) Delete(context.Context, localcas.ObjectID) error {
	return errors.New("unexpected Delete call")
}

func (store *metadataOnlyCAS) List(context.Context) ([]localcas.ObjectMetadata, error) {
	return append([]localcas.ObjectMetadata(nil), store.metadata...), nil
}
