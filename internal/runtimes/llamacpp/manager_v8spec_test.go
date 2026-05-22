package llamacpp

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// V8 Phase 1: backend-local speculative decoding (V7.2 Level 0).
// llama-server takes the draft model as a single-process speculative pair.
// These tests cover env loading, build-arg emission, normalization, and
// status surfacing for the new --model-draft path.

func TestBuildServerArgsOmitsDraftFlagsWhenDraftAbsent(t *testing.T) {
	t.Parallel()
	cfg := LlamaCppSidecarConfig{
		ServerPath:  "/usr/local/bin/llama-server",
		ModelPath:   "/models/llama-3.2-3b.gguf",
		Host:        DefaultHost,
		Port:        45910,
		ContextSize: 4096,
	}
	args := buildServerArgs(cfg)
	joined := strings.Join(args, " ")
	if strings.Contains(joined, "--model-draft") {
		t.Fatalf("expected no --model-draft flag, got: %q", joined)
	}
	if strings.Contains(joined, "--spec-draft-n-max") {
		t.Fatalf("expected no --spec-draft-n-max flag, got: %q", joined)
	}
}

func TestBuildServerArgsEmitsDraftFlagsWhenDraftReadable(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	draftPath := filepath.Join(dir, "tinyllama-1.1b.gguf")
	if err := os.WriteFile(draftPath, []byte("not-a-real-gguf-but-non-empty"), 0o644); err != nil {
		t.Fatalf("write draft model: %v", err)
	}
	cfg := LlamaCppSidecarConfig{
		ServerPath:     "/usr/local/bin/llama-server",
		ModelPath:      "/models/llama-3.2-3b.gguf",
		Host:           DefaultHost,
		Port:           45910,
		ContextSize:    4096,
		DraftModelPath: draftPath,
		DraftMaxTokens: 8,
		DraftMinTokens: 2,
		DraftPMin:      0.6,
		DraftGPULayers: 24,
	}
	args := buildServerArgs(cfg)
	joined := strings.Join(args, " ")

	for _, want := range []string{
		"--model-draft " + draftPath,
		"--spec-draft-n-max 8",
		"--spec-draft-n-min 2",
		"--draft-p-min 0.600",
		"--n-gpu-layers-draft 24",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("buildServerArgs missing %q\nfull: %s", want, joined)
		}
	}
}

func TestBuildServerArgsSilentlyDropsMissingDraftFile(t *testing.T) {
	t.Parallel()
	// Pointing at a nonexistent path must NOT cause llama-server to start
	// with --model-draft (which would crash the sidecar). Instead the
	// flag is silently omitted and llama-server runs without speculation.
	cfg := LlamaCppSidecarConfig{
		ServerPath:     "/usr/local/bin/llama-server",
		ModelPath:      "/models/llama-3.2-3b.gguf",
		Host:           DefaultHost,
		Port:           45910,
		ContextSize:    4096,
		DraftModelPath: "/this/path/does/not/exist.gguf",
		DraftMaxTokens: 16,
	}
	args := buildServerArgs(cfg)
	joined := strings.Join(args, " ")
	if strings.Contains(joined, "--model-draft") {
		t.Fatalf("expected --model-draft to be omitted for missing draft file, got: %q", joined)
	}
}

func TestBuildServerArgsEmitsNativeMTPFlagsWhenEnabled(t *testing.T) {
	t.Parallel()
	cfg := LlamaCppSidecarConfig{
		ServerPath:     "/usr/local/bin/llama-server",
		ModelPath:      "/models/Qwen3.6-27B-MTP-Q5_K_M.gguf",
		Host:           DefaultHost,
		Port:           45910,
		ContextSize:    8192,
		NativeMTP:      true,
		DraftMaxTokens: 3,
	}
	args := buildServerArgs(cfg)
	joined := strings.Join(args, " ")

	for _, want := range []string{
		"--spec-type draft-mtp",
		"--spec-draft-n-max 3",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("buildServerArgs missing %q\nfull: %s", want, joined)
		}
	}
	if strings.Contains(joined, "--model-draft") {
		t.Fatalf("native MTP must use the target model heads, not --model-draft: %s", joined)
	}
}

func TestBuildServerArgsSkipsNativeMTPForPlainModel(t *testing.T) {
	t.Parallel()
	cfg := LlamaCppSidecarConfig{
		ServerPath:     "/usr/local/bin/llama-server",
		ModelPath:      "/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		Host:           DefaultHost,
		Port:           45910,
		ContextSize:    8192,
		NativeMTP:      true,
		DraftMaxTokens: 3,
	}
	args := buildServerArgs(cfg)
	joined := strings.Join(args, " ")
	if strings.Contains(joined, "--spec-type") || strings.Contains(joined, "draft-mtp") {
		t.Fatalf("plain model args = %q, should not enable native MTP without an MTP-head model", joined)
	}
}

func TestBuildServerArgsFallsBackToDraftModelWhenNativeMTPRequestedForPlainModel(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	draftPath := filepath.Join(dir, "tinyllama-1.1b.gguf")
	if err := os.WriteFile(draftPath, []byte("draft"), 0o644); err != nil {
		t.Fatalf("write draft model: %v", err)
	}
	cfg := LlamaCppSidecarConfig{
		ServerPath:     "/usr/local/bin/llama-server",
		ModelPath:      "/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		Host:           DefaultHost,
		Port:           45910,
		ContextSize:    8192,
		NativeMTP:      true,
		DraftModelPath: draftPath,
		DraftMaxTokens: 8,
		DraftMinTokens: 2,
		DraftGPULayers: 12,
	}
	args := buildServerArgs(cfg)
	joined := strings.Join(args, " ")
	if strings.Contains(joined, "--spec-type draft-mtp") {
		t.Fatalf("plain model args = %q, should not force native MTP", joined)
	}
	if !strings.Contains(joined, "--model-draft "+draftPath) ||
		!strings.Contains(joined, "--spec-draft-n-max 8") ||
		!strings.Contains(joined, "--n-gpu-layers-draft 12") {
		t.Fatalf("plain model args = %q, want readable draft model fallback", joined)
	}
}

func TestBuildServerArgsNativeMTPPreservesExplicitSpecTypeExtraArg(t *testing.T) {
	t.Parallel()
	cfg := LlamaCppSidecarConfig{
		ServerPath:  "/usr/local/bin/llama-server",
		ModelPath:   "/models/Qwen3.6-27B-MTP-Q5_K_M.gguf",
		Host:        DefaultHost,
		Port:        45910,
		ContextSize: 8192,
		NativeMTP:   true,
		ExtraArgs:   []string{"--spec-type", "draft-mtp,ngram-mod"},
	}
	args := buildServerArgs(cfg)
	count := 0
	for _, arg := range args {
		if arg == "--spec-type" {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("--spec-type count = %d, want 1 in args: %v", count, args)
	}
}

func TestNormalizeConfigClampsDraftParameters(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	draft := filepath.Join(dir, "drafter.gguf")
	if err := os.WriteFile(draft, []byte("x"), 0o644); err != nil {
		t.Fatalf("write draft: %v", err)
	}

	tests := []struct {
		name string
		in   LlamaCppSidecarConfig
		want LlamaCppSidecarConfig
	}{
		{
			name: "default-max-when-draft-set",
			in: LlamaCppSidecarConfig{
				DraftModelPath: draft,
			},
			want: LlamaCppSidecarConfig{
				DraftModelPath: draft,
				DraftMaxTokens: DefaultDraftMaxTokens,
			},
		},
		{
			name: "out-of-range-max-cleared",
			in: LlamaCppSidecarConfig{
				DraftModelPath: draft,
				DraftMaxTokens: 1024,
			},
			want: LlamaCppSidecarConfig{
				DraftModelPath: draft,
				DraftMaxTokens: DefaultDraftMaxTokens,
			},
		},
		{
			name: "min-clamped-to-max",
			in: LlamaCppSidecarConfig{
				DraftModelPath: draft,
				DraftMaxTokens: 4,
				DraftMinTokens: 16,
			},
			want: LlamaCppSidecarConfig{
				DraftModelPath: draft,
				DraftMaxTokens: 4,
				DraftMinTokens: 4,
			},
		},
		{
			name: "p-min-must-be-strict-prob",
			in: LlamaCppSidecarConfig{
				DraftModelPath: draft,
				DraftPMin:      1.5,
			},
			want: LlamaCppSidecarConfig{
				DraftModelPath: draft,
				DraftMaxTokens: DefaultDraftMaxTokens,
				DraftPMin:      0,
			},
		},
		{
			name: "negative-gpu-layers-floored",
			in: LlamaCppSidecarConfig{
				DraftModelPath: draft,
				DraftGPULayers: -1,
			},
			want: LlamaCppSidecarConfig{
				DraftModelPath: draft,
				DraftMaxTokens: DefaultDraftMaxTokens,
				DraftGPULayers: 0,
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := normalizeConfig(tc.in)
			if got.DraftModelPath != tc.want.DraftModelPath {
				t.Errorf("DraftModelPath = %q, want %q", got.DraftModelPath, tc.want.DraftModelPath)
			}
			if got.DraftMaxTokens != tc.want.DraftMaxTokens {
				t.Errorf("DraftMaxTokens = %d, want %d", got.DraftMaxTokens, tc.want.DraftMaxTokens)
			}
			if got.DraftMinTokens != tc.want.DraftMinTokens {
				t.Errorf("DraftMinTokens = %d, want %d", got.DraftMinTokens, tc.want.DraftMinTokens)
			}
			if got.DraftPMin != tc.want.DraftPMin {
				t.Errorf("DraftPMin = %v, want %v", got.DraftPMin, tc.want.DraftPMin)
			}
			if got.DraftGPULayers != tc.want.DraftGPULayers {
				t.Errorf("DraftGPULayers = %d, want %d", got.DraftGPULayers, tc.want.DraftGPULayers)
			}
		})
	}
}

func TestNormalizeConfigDefaultsNativeMTPDepth(t *testing.T) {
	t.Parallel()
	cfg := normalizeConfig(LlamaCppSidecarConfig{NativeMTP: true})
	if cfg.DraftMaxTokens != DefaultNativeMTPMaxTokens {
		t.Fatalf("Native MTP draft max = %d, want %d", cfg.DraftMaxTokens, DefaultNativeMTPMaxTokens)
	}
}

func TestConfigFromEnvLoadsDraftFields(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	model := filepath.Join(dir, "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
	draft := filepath.Join(dir, "tinyllama-1.1b-Q4_K_M.gguf")
	for _, p := range []string{model, draft} {
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatalf("write %s: %v", p, err)
		}
	}
	server := filepath.Join(dir, "llama-server")
	if err := os.WriteFile(server, []byte("server"), 0o755); err != nil {
		t.Fatalf("write server: %v", err)
	}

	env := map[string]string{
		EnvEnabled:        "1",
		EnvServer:         server,
		EnvModel:          model,
		EnvDraftModel:     draft,
		EnvDraftMaxTokens: "12",
		EnvDraftMinTokens: "3",
		EnvDraftPMin:      "0.65",
		EnvDraftGPULayers: "20",
	}
	cfg := ConfigFromEnvWith(ConfigSource{
		Getenv: func(name string) string { return env[name] },
	})
	if cfg.DraftModelPath != draft {
		t.Errorf("DraftModelPath = %q, want %q", cfg.DraftModelPath, draft)
	}
	if cfg.DraftMaxTokens != 12 {
		t.Errorf("DraftMaxTokens = %d, want 12", cfg.DraftMaxTokens)
	}
	if cfg.DraftMinTokens != 3 {
		t.Errorf("DraftMinTokens = %d, want 3", cfg.DraftMinTokens)
	}
	if cfg.DraftPMin != 0.65 {
		t.Errorf("DraftPMin = %v, want 0.65", cfg.DraftPMin)
	}
	if cfg.DraftGPULayers != 20 {
		t.Errorf("DraftGPULayers = %d, want 20", cfg.DraftGPULayers)
	}
}

func TestConfigFromEnvLoadsNativeMTP(t *testing.T) {
	t.Parallel()
	cfg := ConfigFromEnvWith(ConfigSource{
		Getenv: func(name string) string {
			switch name {
			case EnvNativeMTP:
				return "1"
			case EnvDraftMaxTokens:
				return "2"
			default:
				return ""
			}
		},
	})
	if !cfg.NativeMTP {
		t.Fatal("NativeMTP = false, want true")
	}
	if cfg.DraftMaxTokens != 2 {
		t.Fatalf("DraftMaxTokens = %d, want 2", cfg.DraftMaxTokens)
	}
}

func TestConfigFromEnvDoesNotAutoDiscoverDraftByDefault(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	model := filepath.Join(dir, "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
	draft := filepath.Join(dir, "tinyllama-1.1b-Q4_K_M.gguf")
	for _, p := range []string{model, draft} {
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatalf("write %s: %v", p, err)
		}
	}

	cfg := ConfigFromEnvWith(ConfigSource{
		Getenv: func(name string) string {
			if name == EnvModel {
				return model
			}
			return ""
		},
		ConfiguredModelDirs: []string{dir},
	})
	if cfg.DraftModelPath != "" {
		t.Fatalf("draft auto-discovered by default: %q", cfg.DraftModelPath)
	}
}

func TestConfigFromEnvAutoDiscoversDraftWhenEnabled(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	model := filepath.Join(dir, "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
	draft := filepath.Join(dir, "tinyllama-1.1b-Q4_K_M.gguf")
	for _, p := range []string{model, draft} {
		if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
			t.Fatalf("write %s: %v", p, err)
		}
	}

	cfg := ConfigFromEnvWith(ConfigSource{
		Getenv: func(name string) string {
			switch name {
			case EnvModel:
				return model
			case EnvDraftAuto:
				return "1"
			default:
				return ""
			}
		},
		ConfiguredModelDirs: []string{dir},
	})
	if cfg.DraftModelPath != draft {
		t.Fatalf("auto draft = %q, want %q", cfg.DraftModelPath, draft)
	}
	if cfg.DraftMaxTokens != DefaultDraftMaxTokens {
		t.Fatalf("draft max tokens = %d, want %d", cfg.DraftMaxTokens, DefaultDraftMaxTokens)
	}
}

func TestStatusReportsNativeMTPConfiguration(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	model := filepath.Join(dir, "Qwen3.6-27B-MTP-Q5_K_M.gguf")
	server := filepath.Join(dir, "llama-server")
	for _, p := range []struct {
		path  string
		body  []byte
		perms os.FileMode
	}{
		{model, []byte("model"), 0o644},
		{server, []byte("server"), 0o755},
	} {
		if err := os.WriteFile(p.path, p.body, p.perms); err != nil {
			t.Fatalf("write %s: %v", p.path, err)
		}
	}

	m := NewManager(LlamaCppSidecarConfig{
		Enabled:        true,
		ServerPath:     server,
		ModelPath:      model,
		Host:           DefaultHost,
		Port:           45910,
		NativeMTP:      true,
		DraftMaxTokens: 3,
	})
	st := m.statusLocked()
	if st.SpeculativeMethod != SpeculativeMethodNativeMTP {
		t.Fatalf("SpeculativeMethod = %q, want %q", st.SpeculativeMethod, SpeculativeMethodNativeMTP)
	}
	if !st.NativeMTP {
		t.Fatal("NativeMTP = false, want true")
	}
	if st.DraftModelPath != "" || st.DraftModelFilename != "" {
		t.Fatalf("native MTP status should not expose draft model fields: %+v", st)
	}
	if st.SpeculativeEnabled {
		t.Fatal("SpeculativeEnabled true on uninitialised manager")
	}
}

func TestStatusDoesNotReportNativeMTPForPlainModel(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	model := filepath.Join(dir, "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
	server := filepath.Join(dir, "llama-server")
	for _, p := range []struct {
		path  string
		body  []byte
		perms os.FileMode
	}{
		{model, []byte("model"), 0o644},
		{server, []byte("server"), 0o755},
	} {
		if err := os.WriteFile(p.path, p.body, p.perms); err != nil {
			t.Fatalf("write %s: %v", p.path, err)
		}
	}

	m := NewManager(LlamaCppSidecarConfig{
		Enabled:        true,
		ServerPath:     server,
		ModelPath:      model,
		Host:           DefaultHost,
		Port:           45910,
		NativeMTP:      true,
		DraftMaxTokens: 3,
	})
	st := m.statusLocked()
	if st.SpeculativeMethod == SpeculativeMethodNativeMTP || st.NativeMTP {
		t.Fatalf("plain model status = %+v, should not advertise native_mtp", st)
	}
	caps := BuildBackendRuntimes(st).LlamaCPP.OptimizationCapabilities
	if len(caps) != 0 {
		t.Fatalf("plain model optimization caps = %+v, want none", caps)
	}
}

func TestBuildBackendRuntimesSurfacesNativeMTPCapability(t *testing.T) {
	t.Parallel()
	runtimes := BuildBackendRuntimes(LlamaCppSidecarStatus{
		Enabled:                true,
		Available:              true,
		Running:                true,
		Healthy:                true,
		Backend:                BackendName,
		BaseURL:                "http://127.0.0.1:45910",
		ModelPath:              "/models/Qwen3.6-27B-MTP-Q5_K_M.gguf",
		ModelFilename:          "Qwen3.6-27B-MTP-Q5_K_M.gguf",
		OpenAICompatible:       true,
		SupportsTextGeneration: true,
		SupportsStreaming:      true,
		SpeculativeEnabled:     true,
		SpeculativeMethod:      SpeculativeMethodNativeMTP,
		NativeMTP:              true,
	})
	caps := runtimes.LlamaCPP.OptimizationCapabilities
	if len(caps) != 1 || caps[0].Name != SpeculativeMethodNativeMTP || !caps[0].Supported || !caps[0].Enabled {
		t.Fatalf("OptimizationCapabilities = %+v, want enabled native_mtp", caps)
	}
}

func TestStatusReportsSpeculativeReadiness(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	model := filepath.Join(dir, "llama-3.2-3b.gguf")
	draft := filepath.Join(dir, "tinyllama-1.1b.gguf")
	server := filepath.Join(dir, "llama-server")
	for _, p := range []struct {
		path  string
		body  []byte
		perms os.FileMode
	}{
		{model, []byte("model"), 0o644},
		{draft, []byte("draft"), 0o644},
		{server, []byte("server"), 0o755},
	} {
		if err := os.WriteFile(p.path, p.body, p.perms); err != nil {
			t.Fatalf("write %s: %v", p.path, err)
		}
	}

	m := NewManager(LlamaCppSidecarConfig{
		Enabled:        true,
		ServerPath:     server,
		ModelPath:      model,
		Host:           DefaultHost,
		Port:           45910,
		DraftModelPath: draft,
		DraftMaxTokens: 8,
	})
	// Manager's status surfaces draft fields independently of process
	// state - the SpeculativeEnabled flag also requires running+healthy,
	// which is exercised in higher-level integration tests.
	st := m.statusLocked()
	if st.DraftModelPath != draft {
		t.Errorf("status.DraftModelPath = %q, want %q", st.DraftModelPath, draft)
	}
	if st.DraftModelFilename != "tinyllama-1.1b.gguf" {
		t.Errorf("status.DraftModelFilename = %q", st.DraftModelFilename)
	}
	if st.DraftMaxTokens != 8 {
		t.Errorf("status.DraftMaxTokens = %d, want 8", st.DraftMaxTokens)
	}
	if st.DraftModelFamilyHint != "llama" {
		t.Errorf("status.DraftModelFamilyHint = %q, want llama", st.DraftModelFamilyHint)
	}
	// SpeculativeEnabled requires running+healthy, which a freshly
	// constructed manager does not satisfy until Start succeeds.
	if st.SpeculativeEnabled {
		t.Errorf("SpeculativeEnabled true on uninitialised manager")
	}
}
