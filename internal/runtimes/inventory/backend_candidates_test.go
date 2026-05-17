package runtimeinventory

import (
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
	"time"
)

func TestDetectBackendCandidatesFindsLlamaCPPBinariesFromPATH(t *testing.T) {
	t.Parallel()

	versionPath := ""
	candidates := DetectBackendCandidates(CandidateBackendDetector{
		LookPath: func(name string) (string, error) {
			switch name {
			case "llama-cli", "llama-server", "llama-bench":
				return "/usr/local/bin/" + name, nil
			default:
				return "", errors.New("not found")
			}
		},
		VersionCommand: func(path string, timeout time.Duration) (string, error) {
			versionPath = path
			return "llama.cpp build 123\n", nil
		},
		ReadDirNames: func(dir string, limit int) ([]string, error) {
			return nil, errors.New("not configured")
		},
		Getenv: func(name string) string {
			return ""
		},
		UserHomeDir: func() (string, error) {
			return "", errors.New("not configured")
		},
	})

	llama := backendCandidateByName(t, candidates, BackendCandidateLlamaCPP)
	if !llama.Detected {
		t.Fatalf("llama.cpp candidate = %+v, want detected", llama)
	}
	if llama.BinaryPath != "/usr/local/bin/llama-cli" {
		t.Fatalf("binary_path = %q, want llama-cli", llama.BinaryPath)
	}
	if llama.ServerBinaryPath != "/usr/local/bin/llama-server" {
		t.Fatalf("server_binary_path = %q, want llama-server", llama.ServerBinaryPath)
	}
	if llama.BenchBinaryPath != "/usr/local/bin/llama-bench" {
		t.Fatalf("bench_binary_path = %q, want llama-bench", llama.BenchBinaryPath)
	}
	if versionPath != "/usr/local/bin/llama-server" || llama.Version != "llama.cpp build 123" {
		t.Fatalf("version path/version = %q/%q, want server/version output", versionPath, llama.Version)
	}
	if !llama.SupportsTextGeneration ||
		!llama.SupportsStreaming ||
		!llama.SupportsOpenAICompatibleServer ||
		!llama.CandidateForRealTensorAccess ||
		!llama.CandidateForFastTextRuntime {
		t.Fatalf("llama.cpp support flags = %+v, want text/server candidate", llama)
	}
	if llama.SupportsKVAccess || llama.SupportsTensorHooks || llama.SupportsSpeculativeDecode {
		t.Fatalf("llama.cpp should not claim real KV/tensor/speculative support: %+v", llama)
	}
}

func TestDetectBackendCandidatesPreferConfiguredLlamaCPPDirOverPATH(t *testing.T) {
	t.Parallel()

	binDir := "/opt/ryvion/runtime/bin"
	serverPath := binDir + "/llama-server"
	versionPath := ""
	candidates := DetectBackendCandidates(CandidateBackendDetector{
		LookPath: func(name string) (string, error) {
			switch name {
			case "llama-cli", "llama-server", "llama-bench":
				return "/usr/local/bin/" + name, nil
			default:
				return "", errors.New("not found")
			}
		},
		Stat: func(path string) (os.FileInfo, error) {
			if strings.HasPrefix(path, binDir+"/llama-") {
				return fakeFileInfo{name: path, size: 1024}, nil
			}
			return nil, errors.New("not found")
		},
		VersionCommand: func(path string, timeout time.Duration) (string, error) {
			versionPath = path
			return "llama.cpp configured build\n", nil
		},
		ReadDirNames: func(dir string, limit int) ([]string, error) {
			return nil, errors.New("not configured")
		},
		Getenv: func(name string) string {
			return ""
		},
		UserHomeDir: func() (string, error) {
			return "", errors.New("not configured")
		},
		ConfiguredBinaryDirs: []string{binDir},
	})

	llama := backendCandidateByName(t, candidates, BackendCandidateLlamaCPP)
	if llama.ServerBinaryPath != serverPath {
		t.Fatalf("server_binary_path = %q, want configured runtime path %q", llama.ServerBinaryPath, serverPath)
	}
	if versionPath != serverPath {
		t.Fatalf("version path = %q, want configured runtime server path %q", versionPath, serverPath)
	}
}

func TestDetectBackendCandidatesMissingBinariesReturnSafeFalseValues(t *testing.T) {
	t.Parallel()

	inventory := BuildInventory(RuntimeStatus{}, CandidateBackendDetector{
		LookPath: func(name string) (string, error) {
			return "", errors.New("not found")
		},
		Stat: func(path string) (os.FileInfo, error) {
			return nil, errors.New("not found")
		},
		ReadDirNames: func(dir string, limit int) ([]string, error) {
			return nil, errors.New("not configured")
		},
		VersionCommand: func(path string, timeout time.Duration) (string, error) {
			t.Fatalf("version command should not run when no backend binary is detected")
			return "", nil
		},
		Getenv: func(name string) string {
			return ""
		},
		UserHomeDir: func() (string, error) {
			return "", errors.New("not configured")
		},
	})

	if inventory.CandidateBackends != (CandidateBackends{}) {
		t.Fatalf("candidate_backends = %+v, want safe false booleans", inventory.CandidateBackends)
	}
	if len(inventory.BackendCandidates) == 0 {
		t.Fatalf("backend_candidates should include known backend rows with detected=false")
	}
	for _, candidate := range inventory.BackendCandidates {
		if candidate.Detected ||
			candidate.SupportsKVAccess ||
			candidate.SupportsTensorHooks ||
			candidate.SupportsSpeculativeDecode {
			t.Fatalf("candidate should be safe false with missing binaries: %+v", candidate)
		}
		if candidate.Version != unknownVersion || candidate.Reason == "" {
			t.Fatalf("candidate missing safe version/reason: %+v", candidate)
		}
	}
	if len(inventory.GGUFModels) != 0 {
		t.Fatalf("gguf_models = %+v, want empty", inventory.GGUFModels)
	}
}

func TestDetectGGUFModelsScansConfiguredDirsAndCapsResults(t *testing.T) {
	t.Parallel()

	dirA := "/tmp/ryvion-models-a"
	dirB := "/tmp/ryvion-models-b"
	calledDirs := []string{}
	models := DetectGGUFModels(CandidateBackendDetector{
		LookPath: func(name string) (string, error) {
			return "", errors.New("not found")
		},
		Stat: func(path string) (os.FileInfo, error) {
			return fakeFileInfo{name: path, size: 4096}, nil
		},
		ReadDirNames: func(dir string, limit int) ([]string, error) {
			calledDirs = append(calledDirs, dir)
			switch dir {
			case dirA:
				names := []string{"README.txt"}
				for i := 0; i < 12; i++ {
					names = append(names, "llama-"+string(rune('a'+i))+".Q4_K_M.gguf")
				}
				return names, nil
			case dirB:
				names := []string{}
				for i := 0; i < 15; i++ {
					names = append(names, "gemma-"+string(rune('a'+i))+".Q8_0.gguf")
				}
				return names, nil
			default:
				t.Fatalf("unexpected model dir read: %q", dir)
				return nil, nil
			}
		},
		Getenv: func(name string) string {
			return ""
		},
		UserHomeDir: func() (string, error) {
			return "", errors.New("not configured")
		},
		ConfiguredModelDirs: []string{dirA, dirB},
	})

	if len(models) != maxGGUFModels {
		t.Fatalf("gguf model count = %d, want cap %d: %+v", len(models), maxGGUFModels, models)
	}
	if len(calledDirs) != 2 || calledDirs[0] != dirA || calledDirs[1] != dirB {
		t.Fatalf("called dirs = %+v, want configured dirs only", calledDirs)
	}
	for _, model := range models {
		if !strings.HasPrefix(model.Path, dirA) && !strings.HasPrefix(model.Path, dirB) {
			t.Fatalf("model path outside configured dirs: %+v", model)
		}
		if !strings.HasSuffix(model.Filename, ".gguf") || model.SizeBytes != 4096 {
			t.Fatalf("bad gguf metadata: %+v", model)
		}
		if model.ModelFamilyHint != ggufModelFamilyLlama && model.ModelFamilyHint != ggufModelFamilyGemma {
			t.Fatalf("model family hint = %q, want llama or gemma", model.ModelFamilyHint)
		}
		if model.QuantizationHint != "Q4_K_M" && model.QuantizationHint != "Q8_0" {
			t.Fatalf("quantization hint = %q, want Q4_K_M or Q8_0", model.QuantizationHint)
		}
	}
}

func TestBackendCandidateInventoryWindowsPaths(t *testing.T) {
	t.Parallel()

	binDir := `C:\Ryvion\bin`
	modelDir := `C:\Ryvion\models`
	serverPath := `C:\Ryvion\bin\llama-server.exe`
	modelPath := `C:\Ryvion\models\qwen.Q8_0.gguf`

	inventory := BuildInventory(RuntimeStatus{}, CandidateBackendDetector{
		GOOS: "windows",
		LookPath: func(name string) (string, error) {
			return "", errors.New("not found")
		},
		Stat: func(path string) (os.FileInfo, error) {
			switch path {
			case serverPath:
				return fakeFileInfo{name: "llama-server.exe", size: 1024}, nil
			case modelPath:
				return fakeFileInfo{name: "qwen.Q8_0.gguf", size: 2048}, nil
			default:
				return nil, errors.New("not found")
			}
		},
		ReadDirNames: func(dir string, limit int) ([]string, error) {
			if dir != modelDir {
				t.Fatalf("unexpected model dir read: %q", dir)
			}
			return []string{"qwen.Q8_0.gguf"}, nil
		},
		VersionCommand: func(path string, timeout time.Duration) (string, error) {
			return "llama.cpp windows", nil
		},
		Getenv: func(name string) string {
			return ""
		},
		UserHomeDir: func() (string, error) {
			return "", errors.New("not configured")
		},
		ConfiguredBinaryDirs: []string{binDir},
		ConfiguredModelDirs:  []string{modelDir},
	})

	llama := backendCandidateByName(t, inventory.BackendCandidates, BackendCandidateLlamaCPP)
	if !llama.Detected || llama.ServerBinaryPath != serverPath {
		t.Fatalf("windows llama.cpp candidate = %+v, want server exe path", llama)
	}
	if len(inventory.GGUFModels) != 1 {
		t.Fatalf("gguf_models = %+v, want one model", inventory.GGUFModels)
	}
	model := inventory.GGUFModels[0]
	if model.Path != modelPath || model.ModelFamilyHint != ggufModelFamilyQwen || model.QuantizationHint != "Q8_0" {
		t.Fatalf("windows gguf model = %+v, want qwen Q8_0 path", model)
	}
}

func TestBackendCandidateInventoryJSONHasNoRawPromptOutputOrTensorFields(t *testing.T) {
	t.Parallel()

	inventory := BuildInventory(RuntimeStatus{}, CandidateBackendDetector{
		LookPath: func(name string) (string, error) {
			switch name {
			case "llama-server":
				return "/usr/local/bin/llama-server", nil
			case "python3":
				return "/usr/bin/python3", nil
			default:
				return "", errors.New("not found")
			}
		},
		ReadDirNames: func(dir string, limit int) ([]string, error) {
			return []string{"Llama-3.2-3B-Instruct-Q4_K_M.gguf"}, nil
		},
		Stat: func(path string) (os.FileInfo, error) {
			return fakeFileInfo{name: path, size: 1234}, nil
		},
		VersionCommand: func(path string, timeout time.Duration) (string, error) {
			return "llama.cpp test", nil
		},
		Getenv: func(name string) string {
			return ""
		},
		UserHomeDir: func() (string, error) {
			return "/tmp", nil
		},
	})
	raw, err := json.Marshal(inventory)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	body := strings.ToLower(string(raw))
	for _, forbidden := range []string{"raw_prompt", "prompt_text", "model_output", "output_text", "generated_text", "key_data", "value_data", "query_vector", "tensor_bytes", "raw_tensor", "weighted_value"} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("inventory JSON contains forbidden marker %q: %s", forbidden, raw)
		}
	}
}

func backendCandidateByName(t *testing.T, candidates []BackendCandidate, backend string) BackendCandidate {
	t.Helper()
	for _, candidate := range candidates {
		if candidate.Backend == backend {
			return candidate
		}
	}
	t.Fatalf("candidate %q not found in %+v", backend, candidates)
	return BackendCandidate{}
}

type fakeFileInfo struct {
	name string
	size int64
	dir  bool
}

func (f fakeFileInfo) Name() string       { return f.name }
func (f fakeFileInfo) Size() int64        { return f.size }
func (f fakeFileInfo) Mode() os.FileMode  { return 0o644 }
func (f fakeFileInfo) ModTime() time.Time { return time.Unix(0, 0) }
func (f fakeFileInfo) IsDir() bool        { return f.dir }
func (f fakeFileInfo) Sys() any           { return nil }
