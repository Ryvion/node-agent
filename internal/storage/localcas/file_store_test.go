package localcas

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFileCASPutGetRoundTrip(t *testing.T) {
	ctx := context.Background()
	store := newTestFileCAS(t)
	data := []byte("file cas artifact")
	metadata := ObjectMetadata{
		Kind:   "runner-layer",
		Source: "unit-test",
		Labels: []string{"cache", "node"},
	}

	id, err := store.PutBytes(ctx, data, metadata)
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
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
}

func TestFileCASSameBytesSameIDAndChangedBytesDifferentID(t *testing.T) {
	ctx := context.Background()
	store := newTestFileCAS(t)

	first, err := store.PutBytes(ctx, []byte("same"), ObjectMetadata{})
	if err != nil {
		t.Fatalf("PutBytes(first) error = %v", err)
	}
	second, err := store.PutBytes(ctx, []byte("same"), ObjectMetadata{Kind: "ignored-for-id"})
	if err != nil {
		t.Fatalf("PutBytes(second) error = %v", err)
	}
	changed, err := store.PutBytes(ctx, []byte("changed"), ObjectMetadata{})
	if err != nil {
		t.Fatalf("PutBytes(changed) error = %v", err)
	}

	if first != second {
		t.Fatalf("same bytes ids = %q and %q, want same", first, second)
	}
	if first == changed {
		t.Fatalf("changed bytes id = %q, want different from %q", changed, first)
	}
}

func TestFileCASRejectsInvalidID(t *testing.T) {
	ctx := context.Background()
	store := newTestFileCAS(t)
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

func TestFileCASPathTraversalImpossible(t *testing.T) {
	ctx := context.Background()
	parent := t.TempDir()
	root := filepath.Join(parent, "cas")
	outside := filepath.Join(parent, "outside.txt")
	if err := os.WriteFile(outside, []byte("keep"), 0o600); err != nil {
		t.Fatalf("WriteFile(outside) error = %v", err)
	}

	store, err := NewFileCAS(root)
	if err != nil {
		t.Fatalf("NewFileCAS() error = %v", err)
	}
	malicious := ObjectID("sha256:" + strings.Repeat("0", 30) + "../" + strings.Repeat("0", 31))
	if err := store.Delete(ctx, malicious); err == nil {
		t.Fatalf("Delete(malicious) error = nil, want error")
	}
	if _, _, err := store.GetBytes(ctx, malicious); err == nil {
		t.Fatalf("GetBytes(malicious) error = nil, want error")
	}
	gotOutside, err := os.ReadFile(outside)
	if err != nil {
		t.Fatalf("ReadFile(outside) error = %v", err)
	}
	if string(gotOutside) != "keep" {
		t.Fatalf("outside file = %q, want keep", gotOutside)
	}

	id, err := store.PutBytes(ctx, []byte("inside only"), ObjectMetadata{})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	objectPath, err := store.objectPath(id)
	if err != nil {
		t.Fatalf("objectPath() error = %v", err)
	}
	relative, err := filepath.Rel(root, objectPath)
	if err != nil {
		t.Fatalf("Rel(root, objectPath) error = %v", err)
	}
	if relative == ".." || strings.HasPrefix(relative, ".."+string(os.PathSeparator)) || filepath.IsAbs(relative) {
		t.Fatalf("object path %q escapes root %q", objectPath, root)
	}
}

func TestFileCASListDeterministic(t *testing.T) {
	ctx := context.Background()
	store := newTestFileCAS(t)
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

func TestFileCASDeleteWorks(t *testing.T) {
	ctx := context.Background()
	store := newTestFileCAS(t)
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

func TestFileCASPersistsAcrossInstances(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	firstStore, err := NewFileCAS(root)
	if err != nil {
		t.Fatalf("NewFileCAS(first) error = %v", err)
	}
	data := []byte("persistent artifact")
	id, err := firstStore.PutBytes(ctx, data, ObjectMetadata{Kind: "artifact"})
	if err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}

	secondStore, err := NewFileCAS(root)
	if err != nil {
		t.Fatalf("NewFileCAS(second) error = %v", err)
	}
	got, metadata, err := secondStore.GetBytes(ctx, id)
	if err != nil {
		t.Fatalf("GetBytes() from second store error = %v", err)
	}
	if !bytes.Equal(got, data) {
		t.Fatalf("persisted data = %q, want %q", got, data)
	}
	if metadata.ID != id || metadata.Kind != "artifact" {
		t.Fatalf("persisted metadata = %+v, want id %q kind artifact", metadata, id)
	}
}

func newTestFileCAS(t *testing.T) *FileCAS {
	t.Helper()

	store, err := NewFileCAS(t.TempDir())
	if err != nil {
		t.Fatalf("NewFileCAS() error = %v", err)
	}
	return store
}
