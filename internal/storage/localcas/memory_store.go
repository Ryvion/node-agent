package localcas

import (
	"context"
	"sort"
	"sync"
)

type InMemoryCAS struct {
	mu      sync.RWMutex
	objects map[ObjectID]memoryObject
}

type memoryObject struct {
	data     []byte
	metadata ObjectMetadata
}

func NewInMemoryCAS() *InMemoryCAS {
	return &InMemoryCAS{
		objects: make(map[ObjectID]memoryObject),
	}
}

func (store *InMemoryCAS) PutBytes(ctx context.Context, data []byte, metadata ObjectMetadata) (ObjectID, error) {
	if err := contextError(ctx); err != nil {
		return "", err
	}

	id := HashBytes(data)
	normalized, err := normalizeObjectMetadata(id, int64(len(data)), metadata, currentUnixMs())
	if err != nil {
		return "", err
	}

	object := memoryObject{
		data:     cloneBytes(data),
		metadata: normalized,
	}
	if HashBytes(object.data) != id {
		return "", ErrInvalidObjectID
	}

	store.mu.Lock()
	defer store.mu.Unlock()

	if store.objects == nil {
		store.objects = make(map[ObjectID]memoryObject)
	}
	if _, exists := store.objects[id]; !exists {
		store.objects[id] = object
	}
	return id, nil
}

func (store *InMemoryCAS) GetBytes(ctx context.Context, id ObjectID) ([]byte, ObjectMetadata, error) {
	if err := contextError(ctx); err != nil {
		return nil, ObjectMetadata{}, err
	}
	if err := ValidateObjectID(id); err != nil {
		return nil, ObjectMetadata{}, err
	}

	store.mu.RLock()
	defer store.mu.RUnlock()

	object, ok := store.objects[id]
	if !ok {
		return nil, ObjectMetadata{}, ErrObjectNotFound
	}
	return cloneBytes(object.data), cloneObjectMetadata(object.metadata), nil
}

func (store *InMemoryCAS) Exists(ctx context.Context, id ObjectID) bool {
	if err := contextError(ctx); err != nil {
		return false
	}
	if err := ValidateObjectID(id); err != nil {
		return false
	}

	store.mu.RLock()
	defer store.mu.RUnlock()

	_, ok := store.objects[id]
	return ok
}

func (store *InMemoryCAS) Delete(ctx context.Context, id ObjectID) error {
	if err := contextError(ctx); err != nil {
		return err
	}
	if err := ValidateObjectID(id); err != nil {
		return err
	}

	store.mu.Lock()
	defer store.mu.Unlock()

	delete(store.objects, id)
	return nil
}

func (store *InMemoryCAS) List(ctx context.Context) ([]ObjectMetadata, error) {
	if err := contextError(ctx); err != nil {
		return nil, err
	}

	store.mu.RLock()
	defer store.mu.RUnlock()

	ids := make([]string, 0, len(store.objects))
	for id := range store.objects {
		ids = append(ids, string(id))
	}
	sort.Strings(ids)

	metadata := make([]ObjectMetadata, 0, len(ids))
	for _, id := range ids {
		metadata = append(metadata, cloneObjectMetadata(store.objects[ObjectID(id)].metadata))
	}
	return metadata, nil
}
