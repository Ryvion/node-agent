package runtimeinventory

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/Ryvion/node-agent/internal/v7/tensoraccess"
)

func TestBuildInventoryFromNativeRuntimeStatus(t *testing.T) {
	t.Parallel()

	inventory := BuildInventory(RuntimeStatus{
		RuntimeKind:             RuntimeKindNative,
		Backend:                 BackendNative,
		Provider:                ProviderNoop,
		ProcessMode:             ProcessModeSidecar,
		NativeInferenceReady:    true,
		NativeModel:             "ryvion-llama-3.2-3b",
		ModelLoaded:             true,
		SupportsTextGeneration:  true,
		SupportsStreaming:       true,
		SupportsTensorPlaneDemo: true,
		Reason:                  tensoraccess.ReasonTextGenerationOnly,
	}, CandidateBackendDetector{
		LookPath: func(name string) (string, error) {
			return "", errors.New("not found")
		},
		ReadDirNames: func(dir string, limit int) ([]string, error) {
			return nil, errors.New("not configured")
		},
		UserHomeDir: func() (string, error) {
			return "", errors.New("not configured")
		},
	})

	if inventory.RuntimeKind != RuntimeKindNative || inventory.Backend != BackendNative || inventory.Provider != ProviderNoop {
		t.Fatalf("runtime identity = %+v", inventory)
	}
	if inventory.ProcessMode != ProcessModeSidecar {
		t.Fatalf("process_mode = %q, want sidecar", inventory.ProcessMode)
	}
	if !inventory.NativeInferenceReady || inventory.NativeModel != "ryvion-llama-3.2-3b" {
		t.Fatalf("native status = %+v", inventory)
	}
	if len(inventory.LoadedModels) != 1 {
		t.Fatalf("loaded models length = %d, want 1", len(inventory.LoadedModels))
	}
	model := inventory.LoadedModels[0]
	if model.ModelID != "ryvion-llama-3.2-3b" || !model.Loaded || !model.Warm {
		t.Fatalf("model residency = %+v", model)
	}
	if !model.SupportsTextGeneration || !model.SupportsStreaming || !model.SupportsTensorPlaneDemo {
		t.Fatalf("model text/demo support = %+v", model)
	}
	if model.SupportsKVAccess || model.SupportsTensorPlane {
		t.Fatalf("inventory should not claim real KV/TensorPlane support: %+v", model)
	}
	if model.Reason != tensoraccess.ReasonTextGenerationOnly {
		t.Fatalf("reason = %q, want %q", model.Reason, tensoraccess.ReasonTextGenerationOnly)
	}
}

func TestBuildInventoryUnknownRuntimeUsesSafeDefaults(t *testing.T) {
	t.Parallel()

	inventory := BuildInventory(RuntimeStatus{}, CandidateBackendDetector{
		LookPath: func(name string) (string, error) {
			return "", errors.New("not found")
		},
		ReadDirNames: func(dir string, limit int) ([]string, error) {
			return nil, errors.New("not configured")
		},
		UserHomeDir: func() (string, error) {
			return "", errors.New("not configured")
		},
	})

	if inventory.RuntimeKind != RuntimeKindUnknown {
		t.Fatalf("runtime_kind = %q, want unknown", inventory.RuntimeKind)
	}
	if inventory.Backend != BackendUnknown {
		t.Fatalf("backend = %q, want unknown", inventory.Backend)
	}
	if inventory.Provider != ProviderNoop {
		t.Fatalf("provider = %q, want noop", inventory.Provider)
	}
	if inventory.ProcessMode != ProcessModeUnknown {
		t.Fatalf("process_mode = %q, want unknown", inventory.ProcessMode)
	}
	if inventory.NativeInferenceReady || inventory.NativeModel != "" {
		t.Fatalf("native status should be unavailable by default: %+v", inventory)
	}
	if len(inventory.LoadedModels) != 0 {
		t.Fatalf("loaded_models = %+v, want empty", inventory.LoadedModels)
	}
	if inventory.CandidateBackends != (CandidateBackends{}) {
		t.Fatalf("candidate backends = %+v, want zero", inventory.CandidateBackends)
	}
}

func TestNormalizeInventorySanitizesAndCapsHeartbeatFields(t *testing.T) {
	t.Parallel()

	models := make([]ModelResidencySnapshot, 0, 35)
	for i := 0; i < 35; i++ {
		models = append(models, ModelResidencySnapshot{
			ModelID:     strings.Repeat("m", 150),
			RuntimeKind: " native\n",
			Backend:     " native\t",
			Loaded:      true,
			Warm:        true,
			Reason:      strings.Repeat("r", 300),
		})
	}
	inventory := NormalizeInventory(Inventory{
		RuntimeKind:          " native\n",
		Backend:              " native\t",
		Provider:             " noop\n",
		ProcessMode:          " sidecar\t",
		NativeInferenceReady: true,
		NativeModel:          strings.Repeat("n", 150),
		LoadedModels:         models,
		CandidateBackends: CandidateBackends{
			LlamaCPPDetected: true,
		},
	})

	if inventory.RuntimeKind != RuntimeKindNative ||
		inventory.Backend != BackendNative ||
		inventory.Provider != ProviderNoop ||
		inventory.ProcessMode != ProcessModeSidecar {
		t.Fatalf("compact fields were not normalized: %+v", inventory)
	}
	if len(inventory.NativeModel) != maxInventoryModelIDLen {
		t.Fatalf("native_model length = %d, want %d", len(inventory.NativeModel), maxInventoryModelIDLen)
	}
	if len(inventory.LoadedModels) != maxInventoryLoadedModels {
		t.Fatalf("loaded_models length = %d, want %d", len(inventory.LoadedModels), maxInventoryLoadedModels)
	}
	for _, model := range inventory.LoadedModels {
		if len(model.ModelID) != maxInventoryModelIDLen {
			t.Fatalf("model_id length = %d, want %d", len(model.ModelID), maxInventoryModelIDLen)
		}
		if len(model.Reason) != maxInventoryReasonLen {
			t.Fatalf("reason length = %d, want %d", len(model.Reason), maxInventoryReasonLen)
		}
		if strings.ContainsAny(model.ModelID+model.Reason, "\n\t") {
			t.Fatalf("loaded model text contains control whitespace: %+v", model)
		}
	}
	if !inventory.CandidateBackends.LlamaCPPDetected {
		t.Fatalf("candidate_backends were not preserved: %+v", inventory.CandidateBackends)
	}
}

func TestDetectCandidateBackendsCanBeMocked(t *testing.T) {
	t.Parallel()

	seenNames := map[string]bool{}
	detected := map[string]string{
		"llama-server": "/usr/local/bin/llama-server",
		"ollama":       "/usr/local/bin/ollama",
		"python3":      "/usr/bin/python3",
	}
	candidates := DetectCandidateBackends(CandidateBackendDetector{
		LookPath: func(name string) (string, error) {
			seenNames[name] = true
			if path := detected[name]; path != "" {
				return path, nil
			}
			return "", errors.New("not found")
		},
		ReadDirNames: func(dir string, limit int) ([]string, error) {
			if dir != "/tmp/ryvion-models" {
				t.Fatalf("unexpected model dir read: %q", dir)
			}
			return []string{"readme.txt", "local.Q4_K_M.gguf"}, nil
		},
		ConfiguredModelDirs: []string{"/tmp/ryvion-models"},
		UserHomeDir: func() (string, error) {
			return "", errors.New("not configured")
		},
	})

	if !candidates.LlamaCPPDetected || !candidates.OllamaDetected || !candidates.PythonTransformersDetected || !candidates.GGUFModelsDetected {
		t.Fatalf("candidate detection = %+v", candidates)
	}
	if candidates.VLLMDetected {
		t.Fatalf("vllm_detected = true, want false")
	}
	for _, want := range []string{"llama-server", "ollama", "vllm", "python", "python3"} {
		if !seenNames[want] {
			t.Fatalf("LookPath was not called for %q", want)
		}
	}
}

func TestInventoryJSONHasNoRawTensorPromptOrOutputFields(t *testing.T) {
	t.Parallel()

	inventory := BuildInventory(RuntimeStatus{
		RuntimeKind:             RuntimeKindNative,
		Backend:                 BackendNative,
		Provider:                ProviderNoop,
		ProcessMode:             ProcessModeSidecar,
		NativeInferenceReady:    true,
		NativeModel:             "ryvion-llama-3.2-3b",
		ModelLoaded:             true,
		SupportsTextGeneration:  true,
		SupportsStreaming:       true,
		SupportsTensorPlaneDemo: true,
	}, CandidateBackendDetector{
		LookPath: func(name string) (string, error) {
			return "", errors.New("not found")
		},
		ReadDirNames: func(dir string, limit int) ([]string, error) {
			return nil, errors.New("not configured")
		},
		UserHomeDir: func() (string, error) {
			return "", errors.New("not configured")
		},
	})
	encoded, err := json.Marshal(inventory)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	text := strings.ToLower(string(encoded))
	if !strings.Contains(text, `"loaded_models"`) || !strings.Contains(text, `"candidate_backends"`) {
		t.Fatalf("inventory JSON missing expected blocks: %s", encoded)
	}
	for _, forbidden := range []string{"raw_prompt", "prompt_text", "model_output", "output_text", "generated_text", "key_data", "value_data", "query_vector", "tensor_bytes", "raw_tensor", "weighted_value"} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("inventory JSON contains forbidden marker %q: %s", forbidden, encoded)
		}
	}
}
