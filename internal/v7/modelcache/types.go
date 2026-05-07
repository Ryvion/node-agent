package modelcache

import "time"

const (
	DefaultMaxModels = 100
	DefaultFormat    = "gguf"
)

type Model struct {
	ModelID          string    `json:"model_id"`
	Filename         string    `json:"filename"`
	Path             string    `json:"path"`
	SizeBytes        int64     `json:"size_bytes"`
	FamilyHint       string    `json:"family_hint"`
	QuantizationHint string    `json:"quantization_hint"`
	Format           string    `json:"format"`
	Installed        bool      `json:"installed"`
	HashVerified     bool      `json:"hash_verified"`
	LastSeenAt       time.Time `json:"last_seen_at"`
}

type Status struct {
	CacheDir   string  `json:"cache_dir"`
	TotalBytes int64   `json:"total_bytes"`
	Models     []Model `json:"models"`
}

type ScanOptions struct {
	CacheDir     string
	MaxModels    int
	MaxDepth     int
	MaxEntries   int
	ReadDirNames func(string, int) ([]string, error)
	Stat         func(string) (FileInfo, error)
	Now          func() time.Time
	GOOS         string
}

type FileInfo interface {
	Name() string
	Size() int64
	ModTime() time.Time
	IsDir() bool
}
