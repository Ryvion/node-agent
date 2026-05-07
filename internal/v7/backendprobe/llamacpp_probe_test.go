package backendprobe

import (
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
	"time"
)

func TestProbeLlamaCPPDetectsMockedBinaries(t *testing.T) {
	t.Parallel()

	modelProbeCalls := 0
	versionPath := ""
	probe := ProbeLlamaCPP(Detector{
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
			return "llama.cpp build 456\n", nil
		},
		ReadDirNames: func(dir string, limit int) ([]string, error) {
			return []string{"tinyllama.Q4_K_M.gguf"}, nil
		},
		Stat: func(path string) (os.FileInfo, error) {
			return fakeFileInfo{name: path, size: 1024}, nil
		},
		Getenv: func(name string) string {
			return ""
		},
		UserHomeDir: func() (string, error) {
			return "/tmp", nil
		},
		ModelProbeCommand: func(binary string, modelPath string, timeout time.Duration) error {
			modelProbeCalls++
			return nil
		},
	})

	if !probe.Available {
		t.Fatalf("available = false, want true: %+v", probe)
	}
	if probe.BinaryPath != "/usr/local/bin/llama-cli" ||
		probe.ServerBinaryPath != "/usr/local/bin/llama-server" ||
		probe.BenchBinaryPath != "/usr/local/bin/llama-bench" {
		t.Fatalf("binary paths = %+v", probe)
	}
	if versionPath != "/usr/local/bin/llama-server" || probe.Version != "llama.cpp build 456" {
		t.Fatalf("version path/version = %q/%q", versionPath, probe.Version)
	}
	if !probe.GGUFModelsDetected ||
		!probe.SupportsTextGeneration ||
		!probe.SupportsStreaming ||
		!probe.SupportsOpenAICompatibleServer ||
		!probe.CandidateForFastTextRuntime ||
		!probe.CandidateForRealTensorAccess {
		t.Fatalf("support flags = %+v, want llama.cpp candidate", probe)
	}
	if probe.SupportsKVAccess || probe.SupportsTensorHooks {
		t.Fatalf("probe should not advertise KV/tensor hooks: %+v", probe)
	}
	if probe.ProbeModelConfigured {
		t.Fatalf("probe_model_configured = true without env: %+v", probe)
	}
	if modelProbeCalls != 0 {
		t.Fatalf("model probe calls = %d, want 0 without explicit probe model", modelProbeCalls)
	}
	if probe.Reason != "llama.cpp detected; real KV/tensor hooks require adapter implementation" {
		t.Fatalf("reason = %q", probe.Reason)
	}
}

func TestProbeLlamaCPPMissingBinaries(t *testing.T) {
	t.Parallel()

	probe := ProbeLlamaCPP(Detector{
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
			t.Fatalf("version command should not run without binaries")
			return "", nil
		},
		Getenv: func(name string) string {
			return ""
		},
		UserHomeDir: func() (string, error) {
			return "", errors.New("not configured")
		},
		ModelProbeCommand: func(binary string, modelPath string, timeout time.Duration) error {
			t.Fatalf("model probe should not run without configured model")
			return nil
		},
	})

	if probe.Available ||
		probe.GGUFModelsDetected ||
		probe.ProbeModelConfigured ||
		probe.SupportsTextGeneration ||
		probe.SupportsStreaming ||
		probe.SupportsOpenAICompatibleServer ||
		probe.SupportsKVAccess ||
		probe.SupportsTensorHooks ||
		probe.CandidateForFastTextRuntime ||
		probe.CandidateForRealTensorAccess {
		t.Fatalf("probe = %+v, want safe false values", probe)
	}
	if probe.Version != unknownVersion || probe.Reason != "llama.cpp binary not detected" {
		t.Fatalf("version/reason = %q/%q", probe.Version, probe.Reason)
	}
}

func TestProbeLlamaCPPConfiguredGGUFModelRunsOnlyExplicitProbe(t *testing.T) {
	t.Parallel()

	modelPath := "/tmp/ryvion-models/tiny.Q4_K_M.gguf"
	var calledBinary, calledModel string
	probe := ProbeLlamaCPP(Detector{
		LookPath: func(name string) (string, error) {
			if name == "llama-cli" {
				return "/usr/local/bin/llama-cli", nil
			}
			return "", errors.New("not found")
		},
		VersionCommand: func(path string, timeout time.Duration) (string, error) {
			return "llama.cpp test", nil
		},
		ReadDirNames: func(dir string, limit int) ([]string, error) {
			return nil, errors.New("not configured")
		},
		Stat: func(path string) (os.FileInfo, error) {
			if path != modelPath {
				return nil, errors.New("not found")
			}
			return fakeFileInfo{name: "tiny.Q4_K_M.gguf", size: 64 * 1024}, nil
		},
		Getenv: func(name string) string {
			if name == EnvLlamaCPPProbeModel {
				return modelPath
			}
			return ""
		},
		UserHomeDir: func() (string, error) {
			return "", errors.New("not configured")
		},
		ModelProbeCommand: func(binary string, modelPath string, timeout time.Duration) error {
			calledBinary = binary
			calledModel = modelPath
			return nil
		},
	})

	if !probe.Available || !probe.ProbeModelConfigured || !probe.GGUFModelsDetected {
		t.Fatalf("probe = %+v, want available configured GGUF probe", probe)
	}
	if calledBinary != "/usr/local/bin/llama-cli" || calledModel != modelPath {
		t.Fatalf("model probe call = %q/%q", calledBinary, calledModel)
	}
	if probe.SupportsKVAccess || probe.SupportsTensorHooks {
		t.Fatalf("probe should remain metadata-only: %+v", probe)
	}
}

func TestProbeLlamaCPPJSONContainsNoPromptOutputOrTensorFields(t *testing.T) {
	t.Parallel()

	probe := ProbeAll(Detector{
		LookPath: func(name string) (string, error) {
			if name == "llama-server" {
				return "/usr/local/bin/llama-server", nil
			}
			return "", errors.New("not found")
		},
		VersionCommand: func(path string, timeout time.Duration) (string, error) {
			return "llama.cpp test", nil
		},
		ReadDirNames: func(dir string, limit int) ([]string, error) {
			return []string{"model.Q4_K_M.gguf"}, nil
		},
		Stat: func(path string) (os.FileInfo, error) {
			return fakeFileInfo{name: path, size: 1024}, nil
		},
		Getenv: func(name string) string {
			return ""
		},
		UserHomeDir: func() (string, error) {
			return "/tmp", nil
		},
	})
	raw, err := json.Marshal(probe)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	body := strings.ToLower(string(raw))
	for _, forbidden := range []string{"raw_prompt", "prompt_text", "model_output", "output_text", "generated_text", "key_data", "value_data", "query_vector", "tensor_bytes", "raw_tensor", "weighted_value"} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("probe JSON contains forbidden marker %q: %s", forbidden, raw)
		}
	}
}

func TestNormalizeProbesSanitizesCapsAndDropsNoisyVersionLogs(t *testing.T) {
	t.Parallel()

	longPath := "/tmp/" + strings.Repeat("p", 600)
	probes := NormalizeProbes(Probes{
		LlamaCPP: LlamaCPPProbe{
			Available:                      true,
			BinaryPath:                     longPath + "\n",
			ServerBinaryPath:               longPath + "\t",
			BenchBinaryPath:                longPath + "\r",
			Version:                        "ggml_metal_device_init: tensor API disabled\nllama.cpp build 789\n",
			SupportsTextGeneration:         true,
			SupportsStreaming:              true,
			SupportsOpenAICompatibleServer: true,
			SupportsKVAccess:               true,
			SupportsTensorHooks:            true,
			CandidateForFastTextRuntime:    true,
			CandidateForRealTensorAccess:   true,
			Reason:                         strings.Repeat("r", 300) + "\n",
		},
	})

	probe := probes.LlamaCPP
	if len(probe.BinaryPath) != maxProbePathLen ||
		len(probe.ServerBinaryPath) != maxProbePathLen ||
		len(probe.BenchBinaryPath) != maxProbePathLen {
		t.Fatalf("path lengths = %d/%d/%d, want cap %d", len(probe.BinaryPath), len(probe.ServerBinaryPath), len(probe.BenchBinaryPath), maxProbePathLen)
	}
	if probe.Version != "llama.cpp build 789" {
		t.Fatalf("version = %q, want first clean version-looking line", probe.Version)
	}
	if len(probe.Reason) != maxProbeReasonLen {
		t.Fatalf("reason length = %d, want %d", len(probe.Reason), maxProbeReasonLen)
	}
	if strings.ContainsAny(probe.BinaryPath+probe.ServerBinaryPath+probe.BenchBinaryPath+probe.Reason, "\t\n\r") {
		t.Fatalf("probe text still contains control whitespace: %+v", probe)
	}
	if probe.SupportsKVAccess || probe.SupportsTensorHooks {
		t.Fatalf("normalized probe should not advertise KV/tensor hooks: %+v", probe)
	}
}

func TestNormalizeProbesSetsUnknownForOnlyInitializationLogs(t *testing.T) {
	t.Parallel()

	probes := NormalizeProbes(Probes{
		LlamaCPP: LlamaCPPProbe{
			Available: true,
			Version:   "ggml_metal_device_init: tensor API disabled for this runtime\n",
		},
	})

	if probes.LlamaCPP.Version != unknownVersion {
		t.Fatalf("version = %q, want unknown for initialization-only logs", probes.LlamaCPP.Version)
	}
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
