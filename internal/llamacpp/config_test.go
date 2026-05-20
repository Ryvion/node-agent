package llamacpp

import (
	"reflect"
	"testing"
)

func TestResolveConfigAcceptsLegacyManagedServerEnv(t *testing.T) {
	env := map[string]string{
		EnvEnabled:    "1",
		EnvKeepWarm:   "1",
		EnvServerPath: "/opt/ryvion/bin/llama-server",
		EnvModelPath:  "/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		EnvHost:       "127.0.0.1",
		EnvPort:       "45910",
		EnvCtxSize:    "4096",
	}

	cfg := ResolveConfig(func(key string) string { return env[key] })
	if cfg.ServerURL != "http://127.0.0.1:45910" {
		t.Fatalf("ServerURL = %q, want legacy host/port URL", cfg.ServerURL)
	}
	if cfg.Model != "Llama-3.2-3B-Instruct-Q4_K_M" {
		t.Fatalf("Model = %q, want model id from model path", cfg.Model)
	}
	if !cfg.ManagedServerEnabled() {
		t.Fatal("ManagedServerEnabled() = false, want true")
	}
	wantArgs := []string{
		"--model", "/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
		"--host", "127.0.0.1",
		"--port", "45910",
		"--ctx-size", "4096",
	}
	if got := BuildManagedServerArgs(cfg); !reflect.DeepEqual(got, wantArgs) {
		t.Fatalf("BuildManagedServerArgs() = %#v, want %#v", got, wantArgs)
	}
}

func TestResolveConfigRejectsUnsafeLegacyHost(t *testing.T) {
	env := map[string]string{
		EnvEnabled:    "1",
		EnvServerPath: "/opt/ryvion/bin/llama-server",
		EnvModelPath:  "/models/local.gguf",
		EnvHost:       "10.0.0.4",
		EnvPort:       "45910",
	}

	cfg := ResolveConfig(func(key string) string { return env[key] })
	if cfg.ServerURL != "" {
		t.Fatalf("ServerURL = %q, want no URL for unsafe host", cfg.ServerURL)
	}
	if err := cfg.Validate(); err == nil {
		t.Fatal("Validate() error = nil, want unsafe host config rejected")
	}
}
