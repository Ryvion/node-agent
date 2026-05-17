package modelcache

import (
	"time"

	"github.com/Ryvion/ryvion-node/internal/v7/modelpolicy"
)

const (
	DefaultMaxModels = 100
	DefaultFormat    = "gguf"
)

type Model struct {
	ModelID                string    `json:"model_id"`
	Filename               string    `json:"filename"`
	Path                   string    `json:"path"`
	SizeBytes              int64     `json:"size_bytes"`
	FamilyHint             string    `json:"family_hint"`
	QuantizationHint       string    `json:"quantization_hint"`
	ParameterCountBillions float64   `json:"parameter_count_billions,omitempty"`
	Format                 string    `json:"format"`
	Installed              bool      `json:"installed"`
	Resident               bool      `json:"resident"`
	Runnable               bool      `json:"runnable"`
	BlockedReasons         []string  `json:"blocked_reasons"`
	HashVerified           bool      `json:"hash_verified"`
	LastSeenAt             time.Time `json:"last_seen_at"`
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

type RuntimeAnnotationInput struct {
	Status                         Status
	Policy                         modelpolicy.Policy
	HardwareCapacityAvailable      bool
	BackendTextGenerationAvailable bool
	V7InferenceEnabled             bool
}

type FileInfo interface {
	Name() string
	Size() int64
	ModTime() time.Time
	IsDir() bool
}
