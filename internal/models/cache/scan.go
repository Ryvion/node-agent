package modelcache

import (
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"
)

const (
	maxCachePathLen    = 512
	maxCacheTextLen    = 128
	maxCacheCompactLen = 64
	defaultMaxDepth    = 3
	defaultMaxEntries  = 4096
	dirReadLimit       = 256
)

var ggufQuantizationPattern = regexp.MustCompile(`(?i)(?:^|[._\-\s])((?:IQ|Q)[0-9](?:_[A-Z0-9]+){0,3}|BF16|F16|F32)(?:[._\-\s]|$)`)
var parameterCountPattern = regexp.MustCompile(`(?i)(?:^|[._\-\s])([0-9]+(?:\.[0-9]+)?)\s*b(?:[._\-\s]|$)`)

func Scan(cacheDir string) Status {
	return ScanWithOptions(ScanOptions{CacheDir: cacheDir})
}

func ScanWithOptions(options ScanOptions) Status {
	options = normalizeScanOptions(options)
	cacheDir := cleanCachePath(options.GOOS, options.CacheDir)
	status := Status{
		CacheDir: cacheDir,
		Models:   []Model{},
	}
	if cacheDir == "" || unsafeRootPath(options.GOOS, cacheDir) {
		return status
	}

	rootInfo, err := options.Stat(cacheDir)
	if err != nil || rootInfo == nil || !rootInfo.IsDir() {
		return status
	}

	type dirState struct {
		path  string
		depth int
	}
	queue := []dirState{{path: cacheDir}}
	seenDirs := map[string]struct{}{cacheDir: {}}
	seenModels := map[string]struct{}{}
	entriesSeen := 0

	for len(queue) > 0 && len(status.Models) < options.MaxModels && entriesSeen < options.MaxEntries {
		current := queue[0]
		queue = queue[1:]
		names, err := options.ReadDirNames(current.path, dirReadLimit)
		if err != nil {
			continue
		}
		sort.Strings(names)
		for _, name := range names {
			if len(status.Models) >= options.MaxModels || entriesSeen >= options.MaxEntries {
				break
			}
			name = cleanCacheText(name, maxCacheTextLen)
			if name == "" {
				continue
			}
			entriesSeen++
			path := joinCachePath(options.GOOS, current.path, name)
			if path == "" {
				continue
			}
			info, err := options.Stat(path)
			if err != nil || info == nil {
				continue
			}
			if info.IsDir() {
				if current.depth >= options.MaxDepth {
					continue
				}
				path = cleanCachePath(options.GOOS, path)
				if path == "" {
					continue
				}
				if _, ok := seenDirs[path]; ok {
					continue
				}
				seenDirs[path] = struct{}{}
				queue = append(queue, dirState{path: path, depth: current.depth + 1})
				continue
			}
			if !strings.EqualFold(filepath.Ext(name), ".gguf") {
				continue
			}
			if _, ok := seenModels[path]; ok {
				continue
			}
			seenModels[path] = struct{}{}
			model := buildModel(path, name, info, options.Now())
			status.TotalBytes += model.SizeBytes
			status.Models = append(status.Models, model)
		}
	}

	return NormalizeStatus(status)
}

func normalizeScanOptions(options ScanOptions) ScanOptions {
	if options.MaxModels <= 0 || options.MaxModels > DefaultMaxModels {
		options.MaxModels = DefaultMaxModels
	}
	if options.MaxDepth <= 0 || options.MaxDepth > defaultMaxDepth {
		options.MaxDepth = defaultMaxDepth
	}
	if options.MaxEntries <= 0 || options.MaxEntries > defaultMaxEntries {
		options.MaxEntries = defaultMaxEntries
	}
	if options.ReadDirNames == nil {
		options.ReadDirNames = readDirNames
	}
	if options.Stat == nil {
		options.Stat = stat
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	return options
}

func buildModel(path string, filename string, info FileInfo, now time.Time) Model {
	size := info.Size()
	if size < 0 {
		size = 0
	}
	lastSeen := info.ModTime()
	if lastSeen.IsZero() {
		lastSeen = now
	}
	return Model{
		ModelID:                filename,
		Filename:               filename,
		Path:                   path,
		SizeBytes:              size,
		FamilyHint:             inferFamily(filename),
		QuantizationHint:       inferQuantization(filename),
		ParameterCountBillions: InferParameterCountBillions(filename),
		Format:                 DefaultFormat,
		Installed:              true,
		Resident:               true,
		BlockedReasons:         []string{},
		HashVerified:           false,
		LastSeenAt:             lastSeen.UTC(),
	}
}

func readDirNames(dir string, limit int) ([]string, error) {
	if limit <= 0 {
		limit = dirReadLimit
	}
	f, err := os.Open(dir)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	names, err := f.Readdirnames(limit)
	if err != nil && err != io.EOF {
		return names, err
	}
	return names, nil
}

func stat(path string) (FileInfo, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return nil, os.ErrPermission
	}
	return info, nil
}

func inferFamily(filename string) string {
	lower := strings.ToLower(filename)
	switch {
	case strings.Contains(lower, "llama"):
		return "llama"
	case strings.Contains(lower, "phi"):
		return "phi"
	case strings.Contains(lower, "qwen"):
		return "qwen"
	case strings.Contains(lower, "gemma"):
		return "gemma"
	default:
		return "unknown"
	}
}

func inferQuantization(filename string) string {
	matches := ggufQuantizationPattern.FindStringSubmatch(filename)
	if len(matches) < 2 {
		return "unknown"
	}
	return strings.ToUpper(matches[1])
}

func InferParameterCountBillions(filename string) float64 {
	lower := strings.ToLower(strings.TrimSpace(filename))
	if strings.Contains(lower, "phi-4") || strings.Contains(lower, "phi4") {
		return 14
	}
	matches := parameterCountPattern.FindStringSubmatch(filename)
	if len(matches) < 2 {
		return 0
	}
	value, err := strconv.ParseFloat(matches[1], 64)
	if err != nil || value < 0 {
		return 0
	}
	return value
}

func cleanCachePath(goos, value string) string {
	value = cleanCacheText(value, maxCachePathLen)
	if value == "" {
		return ""
	}
	if isWindowsPath(goos, value) {
		return strings.TrimRight(value, `\/`)
	}
	cleaned := filepath.Clean(value)
	if cleaned == "." {
		return ""
	}
	return cleaned
}

func cleanCacheText(value string, maxRunes int) string {
	value = strings.TrimSpace(value)
	if value == "" || maxRunes <= 0 {
		return ""
	}
	var b strings.Builder
	b.Grow(len(value))
	written := 0
	for _, r := range value {
		if unicode.IsControl(r) {
			continue
		}
		if written >= maxRunes {
			break
		}
		b.WriteRune(r)
		written++
	}
	return strings.TrimSpace(b.String())
}

func joinCachePath(goos, dir string, elems ...string) string {
	dir = strings.TrimSpace(dir)
	if dir == "" {
		return ""
	}
	if isWindowsPath(goos, dir) {
		path := strings.TrimRight(dir, `\/`)
		for _, elem := range elems {
			elem = strings.Trim(strings.TrimSpace(elem), `\/`)
			if elem == "" {
				continue
			}
			path += `\` + elem
		}
		return path
	}
	parts := append([]string{dir}, elems...)
	return filepath.Join(parts...)
}

func isWindowsPath(goos, value string) bool {
	return strings.EqualFold(strings.TrimSpace(goos), "windows") || strings.Contains(value, `\`)
}

func unsafeRootPath(goos, value string) bool {
	if value == "" {
		return true
	}
	if isWindowsPath(goos, value) {
		trimmed := strings.TrimSpace(value)
		if trimmed == `\` || trimmed == `/` {
			return true
		}
		if len(trimmed) == 2 && trimmed[1] == ':' {
			return true
		}
		if len(trimmed) == 3 && trimmed[1] == ':' && (trimmed[2] == '\\' || trimmed[2] == '/') {
			return true
		}
		return false
	}
	return value == string(filepath.Separator)
}
