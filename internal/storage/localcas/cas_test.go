package localcas

import (
	"bytes"
	"context"
	"errors"
	"sort"
	"strings"
	"testing"
)

func TestInMemoryCASPutGetRoundTrip(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryCAS()
	data := []byte("local artifact bytes")
	metadata := ObjectMetadata{
		Kind:   "artifact",
		Source: "unit-test",
		Labels: []string{"alpha", "beta"},
	}

	id, err := store.PutBytes(ctx, data, metadata)
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	if err := ValidateObjectID(id); err != nil {
		t.Fatalf("PutBytes() id invalid: %v", err)
	}
	if !store.Exists(ctx, id) {
		t.Fatalf("Exists(%q) = false, want true", id)
	}

	got, gotMetadata, err := store.GetBytes(ctx, id)
	if err != nil {
		t.Fatalf("GetBytes() error = %v", err)
	}
	if !bytes.Equal(got, data) {
		t.Fatalf("GetBytes() data = %q, want %q", got, data)
	}
	if gotMetadata.ID != id {
		t.Fatalf("metadata id = %q, want %q", gotMetadata.ID, id)
	}
	if gotMetadata.SizeBytes != int64(len(data)) {
		t.Fatalf("metadata size = %d, want %d", gotMetadata.SizeBytes, len(data))
	}
	if gotMetadata.Kind != metadata.Kind || gotMetadata.Source != metadata.Source {
		t.Fatalf("metadata = %+v, want kind/source from input", gotMetadata)
	}
	if !sameStrings(gotMetadata.Labels, metadata.Labels) {
		t.Fatalf("metadata labels = %v, want %v", gotMetadata.Labels, metadata.Labels)
	}

	got[0] = 'X'
	gotMetadata.Labels[0] = "mutated"
	gotAgain, gotMetadataAgain, err := store.GetBytes(ctx, id)
	if err != nil {
		t.Fatalf("GetBytes() after mutation error = %v", err)
	}
	if !bytes.Equal(gotAgain, data) {
		t.Fatalf("stored data mutated to %q, want %q", gotAgain, data)
	}
	if gotMetadataAgain.Labels[0] != "alpha" {
		t.Fatalf("stored metadata label mutated to %q, want alpha", gotMetadataAgain.Labels[0])
	}
}

func TestHashBytesStableAndContentAddressed(t *testing.T) {
	first := HashBytes([]byte("same bytes"))
	second := HashBytes([]byte("same bytes"))
	changed := HashBytes([]byte("changed bytes"))

	if first != second {
		t.Fatalf("HashBytes(same) = %q and %q, want same id", first, second)
	}
	if first == changed {
		t.Fatalf("HashBytes(changed) = %q, want different from %q", changed, first)
	}
}

func TestValidateObjectIDRejectsInvalidIDs(t *testing.T) {
	valid := HashBytes([]byte("valid"))
	if err := ValidateObjectID(valid); err != nil {
		t.Fatalf("ValidateObjectID(valid) error = %v", err)
	}

	invalidIDs := []ObjectID{
		"",
		"sha256:",
		ObjectID("sha256:" + strings.Repeat("0", 63)),
		ObjectID("sha256:" + strings.Repeat("0", 65)),
		ObjectID("sha256:" + strings.Repeat("A", 64)),
		ObjectID("sha256:" + strings.Repeat("g", 64)),
		ObjectID("md5:" + strings.Repeat("0", 64)),
		ObjectID("sha256:" + strings.Repeat("0", 30) + "../" + strings.Repeat("0", 31)),
	}
	for _, id := range invalidIDs {
		if err := ValidateObjectID(id); err == nil {
			t.Fatalf("ValidateObjectID(%q) error = nil, want error", id)
		}
	}
}

func TestInMemoryCASRejectsInvalidID(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryCAS()
	invalid := ObjectID("sha256:" + strings.Repeat("0", 63))

	if _, _, err := store.GetBytes(ctx, invalid); err == nil {
		t.Fatalf("GetBytes(invalid) error = nil, want error")
	}
	if err := store.Delete(ctx, invalid); err == nil {
		t.Fatalf("Delete(invalid) error = nil, want error")
	}
	if store.Exists(ctx, invalid) {
		t.Fatalf("Exists(invalid) = true, want false")
	}
}

func TestInMemoryCASListDeterministic(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryCAS()
	ids := putTestObjects(t, ctx, store, [][]byte{
		[]byte("charlie"),
		[]byte("alpha"),
		[]byte("bravo"),
	})

	got, err := store.List(ctx)
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	wantIDs := sortedObjectIDs(ids)
	if len(got) != len(wantIDs) {
		t.Fatalf("List() length = %d, want %d", len(got), len(wantIDs))
	}
	for i, metadata := range got {
		if metadata.ID != wantIDs[i] {
			t.Fatalf("List()[%d].ID = %q, want %q", i, metadata.ID, wantIDs[i])
		}
	}
}

func TestInMemoryCASDeleteWorks(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryCAS()
	id, err := store.PutBytes(ctx, []byte("delete me"), ObjectMetadata{})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}

	if err := store.Delete(ctx, id); err != nil {
		t.Fatalf("Delete() error = %v", err)
	}
	if store.Exists(ctx, id) {
		t.Fatalf("Exists() after delete = true, want false")
	}
	if _, _, err := store.GetBytes(ctx, id); !errors.Is(err, ErrObjectNotFound) {
		t.Fatalf("GetBytes() after delete error = %v, want ErrObjectNotFound", err)
	}
}

func putTestObjects(t *testing.T, ctx context.Context, store CAS, objects [][]byte) []ObjectID {
	t.Helper()

	ids := make([]ObjectID, 0, len(objects))
	for _, object := range objects {
		id, err := store.PutBytes(ctx, object, ObjectMetadata{})
		if err != nil {
			t.Fatalf("PutBytes(%q) error = %v", object, err)
		}
		ids = append(ids, id)
	}
	return ids
}

func sortedObjectIDs(ids []ObjectID) []ObjectID {
	sorted := append([]ObjectID(nil), ids...)
	sort.Slice(sorted, func(i, j int) bool {
		return string(sorted[i]) < string(sorted[j])
	})
	return sorted
}

func sameStrings(first []string, second []string) bool {
	if len(first) != len(second) {
		return false
	}
	for i := range first {
		if first[i] != second[i] {
			return false
		}
	}
	return true
}
