package localcas

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type FileCAS struct {
	root string
}

func NewFileCAS(root string) (*FileCAS, error) {
	if strings.TrimSpace(root) == "" {
		return nil, fmt.Errorf("localcas: root directory required")
	}

	absoluteRoot, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	store := &FileCAS{root: filepath.Clean(absoluteRoot)}
	if err := os.MkdirAll(store.objectsRoot(), 0o700); err != nil {
		return nil, err
	}
	return store, nil
}

func (store *FileCAS) PutBytes(ctx context.Context, data []byte, metadata ObjectMetadata) (ObjectID, error) {
	if err := contextError(ctx); err != nil {
		return "", err
	}

	id := HashBytes(data)
	normalized, err := normalizeObjectMetadata(id, int64(len(data)), metadata, currentUnixMs())
	if err != nil {
		return "", err
	}

	objectPath, err := store.objectPath(id)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(objectPath), 0o700); err != nil {
		return "", err
	}

	if err := store.putObjectFile(objectPath, data, id); err != nil {
		return "", err
	}
	if err := store.writeMetadataIfMissing(id, normalized); err != nil {
		return "", err
	}
	return id, nil
}

func (store *FileCAS) GetBytes(ctx context.Context, id ObjectID) ([]byte, ObjectMetadata, error) {
	if err := contextError(ctx); err != nil {
		return nil, ObjectMetadata{}, err
	}
	if err := ValidateObjectID(id); err != nil {
		return nil, ObjectMetadata{}, err
	}

	objectPath, err := store.objectPath(id)
	if err != nil {
		return nil, ObjectMetadata{}, err
	}
	data, err := os.ReadFile(objectPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, ObjectMetadata{}, ErrObjectNotFound
		}
		return nil, ObjectMetadata{}, err
	}
	if HashBytes(data) != id {
		return nil, ObjectMetadata{}, fmt.Errorf("localcas: stored bytes do not match object id %s", id)
	}

	metadata, err := store.readMetadata(id, int64(len(data)))
	if err != nil {
		return nil, ObjectMetadata{}, err
	}
	return data, metadata, nil
}

func (store *FileCAS) Exists(ctx context.Context, id ObjectID) bool {
	if err := contextError(ctx); err != nil {
		return false
	}
	if err := ValidateObjectID(id); err != nil {
		return false
	}

	objectPath, err := store.objectPath(id)
	if err != nil {
		return false
	}
	info, err := os.Stat(objectPath)
	return err == nil && info.Mode().IsRegular()
}

func (store *FileCAS) Delete(ctx context.Context, id ObjectID) error {
	if err := contextError(ctx); err != nil {
		return err
	}
	if err := ValidateObjectID(id); err != nil {
		return err
	}

	metadataPath, err := store.metadataPath(id)
	if err != nil {
		return err
	}
	objectPath, err := store.objectPath(id)
	if err != nil {
		return err
	}

	if err := removeIfExists(metadataPath); err != nil {
		return err
	}
	if err := removeIfExists(objectPath); err != nil {
		return err
	}
	_ = os.Remove(filepath.Dir(objectPath))
	return nil
}

func (store *FileCAS) List(ctx context.Context) ([]ObjectMetadata, error) {
	if err := contextError(ctx); err != nil {
		return nil, err
	}

	var ids []ObjectID
	err := filepath.WalkDir(store.objectsRoot(), func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if err := contextError(ctx); err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		if !entry.Type().IsRegular() {
			return nil
		}
		if strings.HasSuffix(entry.Name(), ".json") || strings.HasPrefix(entry.Name(), ".tmp-") {
			return nil
		}

		id, ok := store.idFromObjectPath(path)
		if ok {
			ids = append(ids, id)
		}
		return nil
	})
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return []ObjectMetadata{}, nil
		}
		return nil, err
	}

	sort.Slice(ids, func(i, j int) bool {
		return string(ids[i]) < string(ids[j])
	})

	metadata := make([]ObjectMetadata, 0, len(ids))
	for _, id := range ids {
		if err := contextError(ctx); err != nil {
			return nil, err
		}
		objectPath, err := store.objectPath(id)
		if err != nil {
			return nil, err
		}
		info, err := os.Stat(objectPath)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				continue
			}
			return nil, err
		}
		item, err := store.readMetadata(id, info.Size())
		if err != nil {
			return nil, err
		}
		metadata = append(metadata, item)
	}
	return metadata, nil
}

func (store *FileCAS) putObjectFile(path string, data []byte, id ObjectID) error {
	if _, err := os.Stat(path); err == nil {
		return verifyFileHash(path, id)
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}

	if err := writeFileAtomic(path, data, 0o600); err != nil {
		if _, statErr := os.Stat(path); statErr == nil {
			return verifyFileHash(path, id)
		}
		return err
	}
	return verifyFileHash(path, id)
}

func (store *FileCAS) writeMetadataIfMissing(id ObjectID, metadata ObjectMetadata) error {
	metadataPath, err := store.metadataPath(id)
	if err != nil {
		return err
	}
	if _, err := os.Stat(metadataPath); err == nil {
		return nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}

	data, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	if err := writeFileAtomic(metadataPath, data, 0o600); err != nil {
		if _, statErr := os.Stat(metadataPath); statErr == nil {
			return nil
		}
		return err
	}
	return nil
}

func (store *FileCAS) readMetadata(id ObjectID, sizeBytes int64) (ObjectMetadata, error) {
	metadataPath, err := store.metadataPath(id)
	if err != nil {
		return ObjectMetadata{}, err
	}

	metadata := ObjectMetadata{
		ID:        id,
		SizeBytes: sizeBytes,
	}
	data, err := os.ReadFile(metadataPath)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return metadata, nil
		}
		return ObjectMetadata{}, err
	}
	if err := json.Unmarshal(data, &metadata); err != nil {
		return ObjectMetadata{}, err
	}
	if metadata.ID != "" && metadata.ID != id {
		return ObjectMetadata{}, fmt.Errorf("localcas: metadata id %s does not match object id %s", metadata.ID, id)
	}
	metadata.ID = id
	metadata.SizeBytes = sizeBytes
	metadata.Labels = cloneStrings(metadata.Labels)
	if metadata.CreatedAtUnixMs < 0 {
		return ObjectMetadata{}, fmt.Errorf("%w: created_at_unix_ms must be non-negative", ErrInvalidMetadata)
	}
	return metadata, nil
}

func (store *FileCAS) objectPath(id ObjectID) (string, error) {
	hashHex, err := objectIDHashHex(id)
	if err != nil {
		return "", err
	}
	path := filepath.Join(store.objectsRoot(), hashHex[:2], hashHex)
	if err := store.requirePathWithinRoot(path); err != nil {
		return "", err
	}
	return path, nil
}

func (store *FileCAS) metadataPath(id ObjectID) (string, error) {
	objectPath, err := store.objectPath(id)
	if err != nil {
		return "", err
	}
	return objectPath + ".json", nil
}

func (store *FileCAS) objectsRoot() string {
	return filepath.Join(store.root, "sha256")
}

func (store *FileCAS) requirePathWithinRoot(path string) error {
	relative, err := filepath.Rel(store.root, path)
	if err != nil {
		return err
	}
	if relative == "." || strings.HasPrefix(relative, ".."+string(os.PathSeparator)) || relative == ".." || filepath.IsAbs(relative) {
		return fmt.Errorf("localcas: object path escapes root")
	}
	return nil
}

func (store *FileCAS) idFromObjectPath(path string) (ObjectID, bool) {
	relative, err := filepath.Rel(store.objectsRoot(), path)
	if err != nil {
		return "", false
	}
	parts := strings.Split(filepath.ToSlash(relative), "/")
	if len(parts) != 2 {
		return "", false
	}
	if len(parts[0]) != 2 || len(parts[1]) != objectIDHexLength || parts[0] != parts[1][:2] {
		return "", false
	}

	id := ObjectID(objectIDPrefix + parts[1])
	if ValidateObjectID(id) != nil {
		return "", false
	}
	return id, true
}

func verifyFileHash(path string, id ObjectID) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if HashBytes(data) != id {
		return fmt.Errorf("localcas: stored bytes do not match object id %s", id)
	}
	return nil
}

func writeFileAtomic(path string, data []byte, mode os.FileMode) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}

	tmp, err := os.CreateTemp(dir, ".tmp-"+filepath.Base(path)+"-")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)

	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Chmod(mode); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpName, path); err != nil {
		return err
	}
	return nil
}

func removeIfExists(path string) error {
	if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}
