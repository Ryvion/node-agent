package modelcache

import (
	"errors"
	"fmt"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"
	"unicode"
)

var (
	ErrInvalidInstall = errors.New("modelcache: invalid install")
	ErrModelExists    = errors.New("modelcache: model already exists")
)

type InstallOptions struct {
	CacheDir       string
	ModelID        string
	ArtifactURI    string
	SourcePath     string
	SHA256         string
	SizeBytes      int64
	Now            func() time.Time
	GOOS           string
	HashVerified   bool
	FamilyHint     string
	Format         string
	QuantizationID string
}

type InstallResult struct {
	Model           Model  `json:"model"`
	DestinationPath string `json:"destination_path"`
	SkippedExisting bool   `json:"skipped_existing"`
	HashVerified    bool   `json:"hash_verified"`
}

func ModelPath(cacheDir, modelID, artifactURI string) (string, error) {
	cacheDir = cleanCachePath("", cacheDir)
	if cacheDir == "" || unsafeRootPath("", cacheDir) {
		return "", fmt.Errorf("%w: cache_dir invalid", ErrInvalidInstall)
	}
	filename := modelFilename(modelID, artifactURI)
	if filename == "" {
		return "", fmt.Errorf("%w: model_id invalid", ErrInvalidInstall)
	}
	return filepath.Join(cacheDir, filename), nil
}

func ExistingModel(cacheDir, modelID, artifactURI string) (Model, bool, error) {
	destination, err := ModelPath(cacheDir, modelID, artifactURI)
	if err != nil {
		return Model{}, false, err
	}
	info, err := os.Lstat(destination)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return Model{}, false, nil
		}
		return Model{}, false, err
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return Model{}, false, os.ErrPermission
	}
	if info.IsDir() {
		return Model{}, false, fmt.Errorf("%w: destination is a directory", ErrInvalidInstall)
	}
	return installedModel(destination, modelID, info, InstallOptions{Format: DefaultFormat}), true, nil
}

func InstallDownloadedModel(options InstallOptions) (InstallResult, error) {
	options = normalizeInstallOptions(options)
	if options.CacheDir == "" || options.SourcePath == "" || options.ModelID == "" {
		return InstallResult{}, fmt.Errorf("%w: source_path, cache_dir, and model_id required", ErrInvalidInstall)
	}
	destination, err := ModelPath(options.CacheDir, options.ModelID, options.ArtifactURI)
	if err != nil {
		return InstallResult{}, err
	}
	if err := os.MkdirAll(options.CacheDir, 0o755); err != nil {
		return InstallResult{}, err
	}
	if _, err := os.Lstat(destination); err == nil {
		return InstallResult{}, ErrModelExists
	} else if !errors.Is(err, os.ErrNotExist) {
		return InstallResult{}, err
	}
	if err := os.Rename(options.SourcePath, destination); err != nil {
		return InstallResult{}, err
	}
	info, err := os.Lstat(destination)
	if err != nil {
		return InstallResult{}, err
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return InstallResult{}, os.ErrPermission
	}
	model := installedModel(destination, options.ModelID, info, options)
	return InstallResult{
		Model:           model,
		DestinationPath: destination,
		HashVerified:    model.HashVerified,
	}, nil
}

func BuildInstalledModel(pathValue, modelID string, sizeBytes int64, hashVerified bool, now time.Time) Model {
	filename := filepath.Base(pathValue)
	if filename == "." || filename == string(filepath.Separator) {
		filename = modelFilename(modelID, "")
	}
	info := installFileInfo{name: filename, size: sizeBytes, modTime: now}
	return installedModel(pathValue, modelID, info, InstallOptions{
		HashVerified: hashVerified,
		Format:       DefaultFormat,
		Now: func() time.Time {
			return now
		},
	})
}

func normalizeInstallOptions(options InstallOptions) InstallOptions {
	options.CacheDir = cleanCachePath(options.GOOS, options.CacheDir)
	options.ModelID = cleanCacheText(options.ModelID, maxCacheTextLen)
	options.ArtifactURI = cleanCacheText(options.ArtifactURI, maxCachePathLen)
	options.SourcePath = cleanCachePath(options.GOOS, options.SourcePath)
	options.SHA256 = cleanCacheText(strings.ToLower(options.SHA256), maxCacheTextLen)
	options.FamilyHint = normalizeFamily(options.FamilyHint)
	options.Format = cleanCacheText(strings.ToLower(options.Format), maxCacheCompactLen)
	if options.Format == "" {
		options.Format = DefaultFormat
	}
	options.QuantizationID = cleanCacheText(strings.ToUpper(options.QuantizationID), maxCacheCompactLen)
	if options.Now == nil {
		options.Now = time.Now
	}
	return options
}

func installedModel(pathValue, modelID string, info FileInfo, options InstallOptions) Model {
	options = normalizeInstallOptions(options)
	size := info.Size()
	if size < 0 {
		size = 0
	}
	modTime := info.ModTime()
	if modTime.IsZero() {
		modTime = options.Now()
	}
	filename := cleanCacheText(filepath.Base(pathValue), maxCacheTextLen)
	if filename == "" || filename == "." || filename == string(filepath.Separator) {
		filename = cleanCacheText(info.Name(), maxCacheTextLen)
	}
	family := options.FamilyHint
	if family == "" || family == "unknown" {
		family = inferFamily(filename)
	}
	quant := options.QuantizationID
	if quant == "" {
		quant = inferQuantization(filename)
	}
	modelID = cleanCacheText(firstNonEmptyCache(modelID, filename), maxCacheTextLen)
	return Model{
		ModelID:          modelID,
		Filename:         filename,
		Path:             cleanCachePath("", pathValue),
		SizeBytes:        size,
		FamilyHint:       family,
		QuantizationHint: quant,
		Format:           firstNonEmptyCache(options.Format, DefaultFormat),
		Installed:        true,
		HashVerified:     options.HashVerified,
		LastSeenAt:       modTime.UTC(),
	}
}

func modelFilename(modelID, artifactURI string) string {
	modelID = strings.TrimSpace(strings.ReplaceAll(modelID, `\`, "/"))
	base := path.Base(modelID)
	if base == "." || base == "/" {
		base = modelID
	}
	base = sanitizeFilename(base)
	ext := strings.ToLower(filepath.Ext(base))
	if ext == "" {
		if uriExt := strings.ToLower(filepath.Ext(artifactPathBase(artifactURI))); uriExt != "" {
			ext = uriExt
		} else {
			ext = ".gguf"
		}
		base += ext
	}
	return base
}

func artifactPathBase(artifactURI string) string {
	artifactURI = strings.TrimSpace(artifactURI)
	if artifactURI == "" {
		return ""
	}
	parsed, err := url.Parse(artifactURI)
	if err == nil && parsed.Path != "" {
		return path.Base(parsed.Path)
	}
	return path.Base(strings.ReplaceAll(artifactURI, `\`, "/"))
}

func sanitizeFilename(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	var b strings.Builder
	b.Grow(len(value))
	lastUnderscore := false
	for _, r := range value {
		if b.Len() >= maxCacheTextLen {
			break
		}
		switch {
		case unicode.IsLetter(r), unicode.IsDigit(r), r == '.', r == '-', r == '_':
			b.WriteRune(r)
			lastUnderscore = false
		default:
			if !lastUnderscore {
				b.WriteByte('_')
				lastUnderscore = true
			}
		}
	}
	out := strings.Trim(b.String(), "._- ")
	if out == "" || out == "." || out == ".." {
		return ""
	}
	return out
}

func firstNonEmptyCache(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

type installFileInfo struct {
	name    string
	size    int64
	modTime time.Time
}

func (i installFileInfo) Name() string       { return i.name }
func (i installFileInfo) Size() int64        { return i.size }
func (i installFileInfo) ModTime() time.Time { return i.modTime }
func (i installFileInfo) IsDir() bool        { return false }
