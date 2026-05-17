package artifact

import (
	"errors"
	"fmt"
	"strings"
)

const (
	objectIDPrefix                = "sha256:"
	sha256HexLength               = 64
	DefaultArtifactChunkSizeBytes = int64(4 * 1024 * 1024)
)

var (
	ErrInvalidArtifactManifest = errors.New("artifact: invalid manifest")
	ErrInvalidArtifactOptions  = errors.New("artifact: invalid options")
)

type ArtifactKind string

const (
	ArtifactKindResultJSON     ArtifactKind = "result_json"
	ArtifactKindImage          ArtifactKind = "image"
	ArtifactKindText           ArtifactKind = "text"
	ArtifactKindModelDelta     ArtifactKind = "model_delta"
	ArtifactKindEvidenceBundle ArtifactKind = "evidence_bundle"
	ArtifactKindRunnerLog      ArtifactKind = "runner_log"
	ArtifactKindGenericBlob    ArtifactKind = "generic_blob"
)

type ArtifactManifest struct {
	ArtifactID      string          `json:"artifact_id"`
	Kind            ArtifactKind    `json:"kind"`
	ObjectID        string          `json:"object_id"`
	SizeBytes       int64           `json:"size_bytes"`
	SHA256Hex       string          `json:"sha256_hex"`
	ContentType     string          `json:"content_type"`
	ChunkSizeBytes  int64           `json:"chunk_size_bytes"`
	Chunks          []ArtifactChunk `json:"chunks"`
	CreatedAtUnixMs int64           `json:"created_at_unix_ms"`
	RuntimeID       string          `json:"runtime_id,omitempty"`
	ModelID         string          `json:"model_id,omitempty"`
}

type ArtifactManifestOptions struct {
	ArtifactID      string
	Kind            ArtifactKind
	ContentType     string
	ChunkSizeBytes  int64
	CreatedAtUnixMs int64
	RuntimeID       string
	ModelID         string
}

func ValidateArtifactManifest(manifest ArtifactManifest) error {
	var errs []error
	if strings.TrimSpace(manifest.ArtifactID) == "" {
		errs = append(errs, fmt.Errorf("%w: artifact_id required", ErrInvalidArtifactManifest))
	}
	if !validArtifactKind(manifest.Kind) {
		errs = append(errs, fmt.Errorf("%w: invalid kind %q", ErrInvalidArtifactManifest, manifest.Kind))
	}
	if err := validateObjectID(manifest.ObjectID); err != nil {
		errs = append(errs, err)
	}
	if manifest.SizeBytes < 0 {
		errs = append(errs, fmt.Errorf("%w: size_bytes must be non-negative", ErrInvalidArtifactManifest))
	}
	if err := validateSHA256Hex(manifest.SHA256Hex, "sha256_hex"); err != nil {
		errs = append(errs, err)
	}
	if manifest.ObjectID != "" && manifest.SHA256Hex != "" && manifest.ObjectID != objectIDFromHashHex(manifest.SHA256Hex) {
		errs = append(errs, fmt.Errorf("%w: object_id must match sha256_hex", ErrInvalidArtifactManifest))
	}
	if strings.TrimSpace(manifest.ContentType) == "" {
		errs = append(errs, fmt.Errorf("%w: content_type required", ErrInvalidArtifactManifest))
	}
	if manifest.ChunkSizeBytes <= 0 {
		errs = append(errs, fmt.Errorf("%w: chunk_size_bytes must be greater than zero", ErrInvalidArtifactManifest))
	}
	if manifest.CreatedAtUnixMs < 0 {
		errs = append(errs, fmt.Errorf("%w: created_at_unix_ms must be non-negative", ErrInvalidArtifactManifest))
	}
	if err := validateChunks(manifest); err != nil {
		errs = append(errs, err)
	}
	if err := validateNoSecretLikeManifestValues(manifest); err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

func validArtifactKind(kind ArtifactKind) bool {
	switch kind {
	case ArtifactKindResultJSON,
		ArtifactKindImage,
		ArtifactKindText,
		ArtifactKindModelDelta,
		ArtifactKindEvidenceBundle,
		ArtifactKindRunnerLog,
		ArtifactKindGenericBlob:
		return true
	default:
		return false
	}
}

func validateObjectID(objectID string) error {
	if len(objectID) != len(objectIDPrefix)+sha256HexLength {
		return fmt.Errorf("%w: object_id must be sha256:<64 lowercase hex>", ErrInvalidArtifactManifest)
	}
	if !strings.HasPrefix(objectID, objectIDPrefix) {
		return fmt.Errorf("%w: object_id missing sha256 prefix", ErrInvalidArtifactManifest)
	}
	return validateSHA256Hex(objectID[len(objectIDPrefix):], "object_id")
}

func validateSHA256Hex(value string, field string) error {
	if len(value) != sha256HexLength {
		return fmt.Errorf("%w: %s must be 64 lowercase hex characters", ErrInvalidArtifactManifest, field)
	}
	for _, ch := range value {
		if (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') {
			continue
		}
		return fmt.Errorf("%w: %s must be 64 lowercase hex characters", ErrInvalidArtifactManifest, field)
	}
	return nil
}

func validateNoSecretLikeManifestValues(manifest ArtifactManifest) error {
	fields := []struct {
		name  string
		value string
	}{
		{name: "artifact_id", value: manifest.ArtifactID},
		{name: "object_id", value: manifest.ObjectID},
		{name: "content_type", value: manifest.ContentType},
		{name: "runtime_id", value: manifest.RuntimeID},
		{name: "model_id", value: manifest.ModelID},
	}
	var found []string
	for _, field := range fields {
		if looksLikeSecret(field.value) {
			found = append(found, field.name)
		}
	}
	if len(found) > 0 {
		return fmt.Errorf("%w: secret-like value in %s", ErrInvalidArtifactManifest, strings.Join(found, ", "))
	}
	return nil
}

func looksLikeSecret(value string) bool {
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "" {
		return false
	}
	needles := []string{
		"-----begin private key",
		"authorization:",
		"bearer ",
		"api_key",
		"apikey",
		"access_token",
		"refresh_token",
		"private_key",
		"client_secret",
		"password=",
		"passwd=",
		"secret=",
	}
	for _, needle := range needles {
		if strings.Contains(value, needle) {
			return true
		}
	}
	return strings.HasPrefix(value, "sk-")
}
