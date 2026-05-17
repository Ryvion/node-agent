package sglang

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	v7hardware "github.com/Ryvion/ryvion-node/internal/v7/hardware"
	v7llamacpp "github.com/Ryvion/ryvion-node/internal/v7/llamacpp"
	"github.com/Ryvion/ryvion-node/internal/v7/runtimeinventory"
)

func TestConfigFromEnvBuildsLocalPythonLaunch(t *testing.T) {
	t.Parallel()

	binDir := t.TempDir()
	launcherPath := filepath.Join(binDir, "python3")
	if err := os.WriteFile(launcherPath, []byte("python"), 0o755); err != nil {
		t.Fatalf("write launcher fixture: %v", err)
	}
	modelPath := filepath.Join(t.TempDir(), "qwen2.5-7b-instruct")
	if err := os.MkdirAll(modelPath, 0o755); err != nil {
		t.Fatalf("write model fixture: %v", err)
	}
	env := map[string]string{
		EnvEnabled:           "1",
		EnvServer:            launcherPath,
		EnvModel:             modelPath,
		EnvHost:              "0.0.0.0",
		EnvPort:              "45921",
		EnvContextLength:     "8192",
		EnvTPSize:            "2",
		EnvMemFractionStatic: "0.75",
		EnvExtraArgs:         "--host 0.0.0.0 --disable-cuda-graph --tp-size 9 --chunked-prefill-size 4096 --model-path /tmp/other",
	}

	cfg := ConfigFromEnvWith(ConfigSource{
		Getenv: func(name string) string {
			return env[name]
		},
		LookPath: func(string) (string, error) {
			return "", errors.New("path lookup disabled")
		},
		Stat: os.Stat,
		GOOS: "linux",
	})

	if !cfg.Enabled || cfg.ServerPath != launcherPath || cfg.ModelPath != modelPath {
		t.Fatalf("config = %+v, want enabled explicit launcher/model", cfg)
	}
	if cfg.Host != DefaultHost {
		t.Fatalf("host = %q, want localhost default", cfg.Host)
	}
	if cfg.Port != 45921 || cfg.ContextLength != 8192 || cfg.TPSize != 2 || cfg.MemFractionStatic != 0.75 {
		t.Fatalf("numeric config = %+v, want port/context/tp/mem fraction from env", cfg)
	}

	args := buildServerArgs(cfg)
	wantArgs := []string{
		"-m", "sglang.launch_server",
		"--model-path", modelPath,
		"--host", DefaultHost,
		"--port", "45921",
		"--context-length", "8192",
		"--tp-size", "2",
		"--mem-fraction-static", "0.750",
		"--disable-cuda-graph",
		"--chunked-prefill-size", "4096",
	}
	if !reflect.DeepEqual(args, wantArgs) {
		t.Fatalf("launch args = %#v, want %#v", args, wantArgs)
	}
}

func TestManagerStartsManagedPythonModuleSidecar(t *testing.T) {
	t.Parallel()

	launcherPath := filepath.Join(t.TempDir(), "python3")
	if err := os.WriteFile(launcherPath, []byte("python"), 0o755); err != nil {
		t.Fatalf("write launcher fixture: %v", err)
	}
	modelPath := filepath.Join(t.TempDir(), "qwen2.5-7b-instruct")
	if err := os.MkdirAll(modelPath, 0o755); err != nil {
		t.Fatalf("write model fixture: %v", err)
	}

	client := &sequenceHealthClient{responses: []healthClientResponse{
		{err: errors.New("connection refused")},
		{err: errors.New("connection refused")},
		{statusCode: http.StatusOK, body: `{"status":"ok"}`},
	}}
	var gotBinary string
	var gotArgs []string
	manager := NewManager(SGLangSidecarConfig{
		Enabled:       true,
		ServerPath:    launcherPath,
		ModelPath:     modelPath,
		Host:          DefaultHost,
		Port:          45921,
		ContextLength: 4096,
		TPSize:        1,
	}, WithProcessStarter(func(ctx context.Context, binary string, args []string, output io.Writer) (managedProcess, error) {
		gotBinary = binary
		gotArgs = append([]string(nil), args...)
		return newFakeManagedProcess(2468), nil
	}), WithHealthClient(client), WithHealthTimeout(time.Millisecond))
	t.Cleanup(func() {
		_ = manager.Stop(context.Background())
	})

	status := manager.Start(context.Background())
	if gotBinary != launcherPath {
		t.Fatalf("started binary = %q, want %q", gotBinary, launcherPath)
	}
	if len(gotArgs) < 8 || gotArgs[0] != "-m" || gotArgs[1] != "sglang.launch_server" {
		t.Fatalf("started args = %#v, want python module launch", gotArgs)
	}
	if !status.Enabled || !status.Available || !status.Running || !status.Healthy || status.PID != 2468 {
		t.Fatalf("status = %+v, want managed healthy SGLang sidecar", status)
	}
	if status.Backend != BackendName || !status.OpenAICompatible || !status.SupportsTextGeneration || !status.SupportsStreaming {
		t.Fatalf("capability flags = %+v", status)
	}
	if status.SupportsKVAccess || status.SupportsTensorHooks {
		t.Fatalf("SGLang sidecar should not advertise raw KV/tensor access by default: %+v", status)
	}
}

func TestStatusJSONExcludesPromptOutputAuthAndTensorData(t *testing.T) {
	t.Parallel()

	launcherPath := filepath.Join(t.TempDir(), "python3")
	if err := os.WriteFile(launcherPath, []byte("python"), 0o755); err != nil {
		t.Fatalf("write launcher fixture: %v", err)
	}
	modelPath := filepath.Join(t.TempDir(), "qwen2.5-7b-instruct")
	if err := os.MkdirAll(modelPath, 0o755); err != nil {
		t.Fatalf("write model fixture: %v", err)
	}
	manager := NewManager(SGLangSidecarConfig{
		Enabled:    true,
		ServerPath: launcherPath,
		ModelPath:  modelPath,
		Host:       DefaultHost,
		Port:       45921,
	}, WithHealthClient(errorHealthClient{err: errors.New("raw_prompt output_text auth_token tensor_bytes")}), WithHealthTimeout(time.Millisecond))

	status := manager.Status(context.Background())
	raw, err := json.Marshal(status)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	body := strings.ToLower(string(raw))
	for _, forbidden := range []string{"raw_prompt", "prompt_text", "model_output", "output_text", "generated_text", "key_data", "value_data", "query_vector", "tensor_bytes", "raw_tensor", "auth_token", "bind_token"} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("status JSON contains forbidden marker %q: %s", forbidden, raw)
		}
	}
	if status.LastError != "sglang sidecar status redacted" {
		t.Fatalf("last_error = %q, want redacted status", status.LastError)
	}
}

func TestBuildBackendRuntimeMarksHealthySidecarLoadedWarm(t *testing.T) {
	t.Parallel()

	lastHealth := time.Unix(123, 456000000)
	status := SGLangSidecarStatus{
		Enabled:                  true,
		Available:                true,
		Running:                  true,
		Healthy:                  true,
		BaseURL:                  "http://127.0.0.1:45921",
		ServerPath:               "/opt/ryvion/bin/python3",
		ModelPath:                "/models/qwen2.5-7b-instruct",
		ModelID:                  "qwen2.5-7b-instruct",
		ContextLength:            8192,
		StartedAt:                time.Unix(100, 0),
		LastHealthAt:             lastHealth,
		Backend:                  BackendName,
		Launch:                   &LaunchConfig{Mode: "managed", Managed: true, ServerPath: "/opt/ryvion/bin/python3", ServerFilename: "python3", TPSize: 2, MemFractionStatic: 0.75},
		Acceleration:             []string{"cuda"},
		AccelerationReason:       "configured_cuda_unconfirmed",
		OpenAICompatible:         true,
		SupportsTextGeneration:   true,
		SupportsStreaming:        true,
		SupportsStatefulSessions: true,
	}
	runtimeStatus := BuildBackendRuntime(status, v7hardware.CapacityInventory{
		GPUDetected:       true,
		GPUVendor:         v7hardware.GPUVendorNVIDIA,
		CUDAAvailable:     true,
		ComputeCapability: "8.9",
	})

	if runtimeStatus.Backend != runtimeinventory.BackendCandidateSGLang ||
		!runtimeStatus.Enabled || !runtimeStatus.Available || !runtimeStatus.Running ||
		!runtimeStatus.Healthy || !runtimeStatus.Loaded || !runtimeStatus.Warm {
		t.Fatalf("backend runtime = %+v, want active loaded warm SGLang", runtimeStatus)
	}
	if runtimeStatus.ModelID != "qwen2.5-7b-instruct" ||
		runtimeStatus.ModelPath != "/models/qwen2.5-7b-instruct" ||
		runtimeStatus.MaxContextTokens != 8192 ||
		runtimeStatus.LastHealthAtUnixMs != lastHealth.UnixMilli() {
		t.Fatalf("runtime model/health metadata = %+v", runtimeStatus)
	}
	if !runtimeStatus.OpenAICompatible || !runtimeStatus.SupportsTextGeneration || !runtimeStatus.SupportsStreaming || !runtimeStatus.SupportsStatefulSessions {
		t.Fatalf("runtime capability flags = %+v", runtimeStatus)
	}
	if runtimeStatus.SupportsKVAccess || runtimeStatus.SupportsTensorHooks {
		t.Fatalf("runtime should not expose raw KV/tensor hooks by default: %+v", runtimeStatus)
	}
}

type fakeManagedProcess struct {
	pid  int
	done chan struct{}
}

func newFakeManagedProcess(pid int) *fakeManagedProcess {
	return &fakeManagedProcess{pid: pid, done: make(chan struct{})}
}

func (p *fakeManagedProcess) PID() int {
	return p.pid
}

func (p *fakeManagedProcess) Wait() error {
	<-p.done
	return nil
}

func (p *fakeManagedProcess) Kill() error {
	select {
	case <-p.done:
	default:
		close(p.done)
	}
	return nil
}

type healthClientResponse struct {
	statusCode int
	body       string
	err        error
}

type sequenceHealthClient struct {
	responses []healthClientResponse
}

func (c *sequenceHealthClient) Do(*http.Request) (*http.Response, error) {
	if len(c.responses) == 0 {
		return &http.Response{StatusCode: http.StatusOK, Status: "200 OK", Body: io.NopCloser(strings.NewReader(`{"ok":true}`))}, nil
	}
	next := c.responses[0]
	c.responses = c.responses[1:]
	if next.err != nil {
		return nil, next.err
	}
	if next.statusCode == 0 {
		next.statusCode = http.StatusOK
	}
	return &http.Response{
		StatusCode: next.statusCode,
		Status:     http.StatusText(next.statusCode),
		Body:       io.NopCloser(strings.NewReader(next.body)),
	}, nil
}

type errorHealthClient struct {
	err error
}

func (c errorHealthClient) Do(*http.Request) (*http.Response, error) {
	return nil, c.err
}

var _ = v7llamacpp.BackendRuntimeStatus{}
