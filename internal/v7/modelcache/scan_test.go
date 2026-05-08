package modelcache

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Ryvion/node-agent/internal/v7/modelpolicy"
)

func TestScanDetectsGGUFModels(t *testing.T) {
	t.Parallel()

	cacheDir := t.TempDir()
	nested := filepath.Join(cacheDir, "nested")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatalf("create nested dir: %v", err)
	}
	modelA := filepath.Join(cacheDir, "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
	modelB := filepath.Join(nested, "phi-4-Q5_K_M.gguf")
	if err := os.WriteFile(modelA, []byte("gguf-a"), 0o644); err != nil {
		t.Fatalf("write model A: %v", err)
	}
	if err := os.WriteFile(modelB, []byte("gguf-bb"), 0o644); err != nil {
		t.Fatalf("write model B: %v", err)
	}
	if err := os.WriteFile(filepath.Join(cacheDir, "README.txt"), []byte("not a model"), 0o644); err != nil {
		t.Fatalf("write readme: %v", err)
	}

	status := Scan(cacheDir)
	if status.CacheDir != cacheDir {
		t.Fatalf("cache_dir = %q, want %q", status.CacheDir, cacheDir)
	}
	if len(status.Models) != 2 {
		t.Fatalf("models = %+v, want two gguf models", status.Models)
	}
	if status.TotalBytes != int64(len("gguf-a")+len("gguf-bb")) {
		t.Fatalf("total_bytes = %d", status.TotalBytes)
	}
	byName := map[string]Model{}
	for _, model := range status.Models {
		byName[model.Filename] = model
		if model.Format != DefaultFormat || !model.Installed || model.HashVerified {
			t.Fatalf("model status flags = %+v", model)
		}
		if model.LastSeenAt.IsZero() {
			t.Fatalf("last_seen_at is zero: %+v", model)
		}
	}
	if got := byName["Llama-3.2-3B-Instruct-Q4_K_M.gguf"]; got.FamilyHint != "llama" || got.QuantizationHint != "Q4_K_M" {
		t.Fatalf("llama hints = %+v", got)
	}
	if got := byName["phi-4-Q5_K_M.gguf"]; got.FamilyHint != "phi" || got.QuantizationHint != "Q5_K_M" || got.ParameterCountBillions != 14 {
		t.Fatalf("phi hints = %+v", got)
	}
}

func TestScanAppliesModelCapAndDoesNotLeaveConfiguredDir(t *testing.T) {
	t.Parallel()

	cacheDir := "/cache"
	names := make([]string, 0, 140)
	for i := 0; i < 140; i++ {
		names = append(names, "llama-"+string(rune('a'+(i%26)))+"-"+string(rune('a'+((i/26)%26)))+".Q4_K_M.gguf")
	}
	status := ScanWithOptions(ScanOptions{
		CacheDir: cacheDir,
		Stat: func(path string) (FileInfo, error) {
			if path == cacheDir {
				return fakeFileInfo{name: "cache", dir: true}, nil
			}
			if !strings.HasPrefix(path, cacheDir+"/") {
				t.Fatalf("stat path outside cache dir: %q", path)
			}
			return fakeFileInfo{name: filepath.Base(path), size: 4096}, nil
		},
		ReadDirNames: func(dir string, limit int) ([]string, error) {
			if dir != cacheDir {
				t.Fatalf("read dir outside configured cache dir: %q", dir)
			}
			return names, nil
		},
		Now: func() time.Time {
			return time.Unix(100, 0)
		},
	})

	if len(status.Models) != DefaultMaxModels {
		t.Fatalf("models len = %d, want cap %d", len(status.Models), DefaultMaxModels)
	}
	if status.TotalBytes != int64(DefaultMaxModels*4096) {
		t.Fatalf("total_bytes = %d", status.TotalBytes)
	}
}

func TestScanRejectsUnsafeRootCacheDir(t *testing.T) {
	t.Parallel()

	called := false
	status := ScanWithOptions(ScanOptions{
		CacheDir: "/",
		Stat: func(path string) (FileInfo, error) {
			called = true
			return fakeFileInfo{dir: true}, nil
		},
		ReadDirNames: func(dir string, limit int) ([]string, error) {
			t.Fatalf("ReadDirNames should not be called for root")
			return nil, nil
		},
	})

	if called {
		t.Fatalf("Stat should not be called for unsafe root cache dir")
	}
	if status.CacheDir != "/" || len(status.Models) != 0 || status.TotalBytes != 0 {
		t.Fatalf("root status = %+v, want empty status", status)
	}
}

func TestScanWindowsPathHandling(t *testing.T) {
	t.Parallel()

	cacheDir := `C:\Ryvion\models`
	modelPath := `C:\Ryvion\models\qwen.Q8_0.gguf`
	status := ScanWithOptions(ScanOptions{
		CacheDir: cacheDir,
		GOOS:     "windows",
		Stat: func(path string) (FileInfo, error) {
			switch path {
			case cacheDir:
				return fakeFileInfo{name: "models", dir: true}, nil
			case modelPath:
				return fakeFileInfo{name: "qwen.Q8_0.gguf", size: 2048, modTime: time.Unix(200, 0)}, nil
			default:
				return nil, errors.New("not found")
			}
		},
		ReadDirNames: func(dir string, limit int) ([]string, error) {
			if dir != cacheDir {
				t.Fatalf("dir = %q, want %q", dir, cacheDir)
			}
			return []string{"qwen.Q8_0.gguf"}, nil
		},
		Now: func() time.Time {
			return time.Unix(100, 0)
		},
	})

	if len(status.Models) != 1 {
		t.Fatalf("models = %+v, want one", status.Models)
	}
	model := status.Models[0]
	if model.Path != modelPath || model.FamilyHint != "qwen" || model.QuantizationHint != "Q8_0" {
		t.Fatalf("windows model = %+v", model)
	}
}

func TestAnnotateRuntimeStatusSetsRunnableAndBlockedReasons(t *testing.T) {
	t.Parallel()

	status := NormalizeStatus(Status{
		CacheDir: "/cache",
		Models: []Model{
			{
				ModelID:                "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
				Filename:               "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
				Path:                   "/cache/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
				SizeBytes:              3 << 30,
				FamilyHint:             "llama",
				QuantizationHint:       "Q4_K_M",
				ParameterCountBillions: 3,
				Format:                 "gguf",
				Installed:              true,
			},
			{
				ModelID:                "phi-4-Q5_K_M.gguf",
				Filename:               "phi-4-Q5_K_M.gguf",
				Path:                   "/cache/phi-4-Q5_K_M.gguf",
				SizeBytes:              10 << 30,
				FamilyHint:             "phi",
				QuantizationHint:       "Q5_K_M",
				ParameterCountBillions: 14,
				Format:                 "gguf",
				Installed:              true,
			},
		},
	})
	policy := modelpolicy.FromConfigSource(modelpolicy.ConfigSource{
		Getenv:      func(string) string { return "" },
		UserHomeDir: func() (string, error) { return "/home/operator", nil },
		GOOS:        "linux",
	})
	annotated := AnnotateRuntimeStatus(RuntimeAnnotationInput{
		Status:                         status,
		Policy:                         policy,
		HardwareCapacityAvailable:      true,
		BackendTextGenerationAvailable: true,
		V7InferenceEnabled:             true,
	})

	if !annotated.Models[0].Runnable || len(annotated.Models[0].BlockedReasons) != 0 {
		t.Fatalf("llama annotation = %+v, want runnable", annotated.Models[0])
	}
	if annotated.Models[1].Runnable || len(annotated.Models[1].BlockedReasons) == 0 {
		t.Fatalf("phi annotation = %+v, want blocked reasons", annotated.Models[1])
	}
}

func TestCacheStatusJSONHasNoRawTensorPromptOutputOrSecrets(t *testing.T) {
	t.Parallel()

	status := NormalizeStatus(Status{
		CacheDir: "/cache",
		Models: []Model{{
			ModelID:          "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
			Filename:         "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
			Path:             "/cache/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
			SizeBytes:        2048,
			FamilyHint:       "llama",
			QuantizationHint: "Q4_K_M",
			Format:           "gguf",
			Installed:        true,
			LastSeenAt:       time.Unix(100, 0),
		}},
	})
	raw, err := json.Marshal(status)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	text := strings.ToLower(string(raw))
	for _, forbidden := range []string{"raw_prompt", "prompt_text", "model_output", "output_text", "generated_text", "key_data", "value_data", "query_vector", "tensor_bytes", "raw_tensor", "secret", "token"} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("cache JSON contains forbidden marker %q: %s", forbidden, raw)
		}
	}
}

type fakeFileInfo struct {
	name    string
	size    int64
	dir     bool
	modTime time.Time
}

func (f fakeFileInfo) Name() string { return f.name }
func (f fakeFileInfo) Size() int64  { return f.size }
func (f fakeFileInfo) ModTime() time.Time {
	return f.modTime
}
func (f fakeFileInfo) IsDir() bool { return f.dir }
