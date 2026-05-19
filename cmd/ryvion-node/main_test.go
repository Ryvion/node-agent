package main

import (
	"context"
	"crypto/ed25519"
	"encoding/json"
	"flag"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/Ryvion/ryvion-node/internal/hub"
	"github.com/Ryvion/ryvion-node/internal/hw"
	"github.com/Ryvion/ryvion-node/internal/llamacpp"
)

func TestParseConfigUsesLegacyDeviceTypeEnv(t *testing.T) {
	withArgs(t, "ryvion-node")
	t.Setenv("RYV_DEVICE", "")
	t.Setenv("RYV_DEVICE_TYPE", "gpu")

	cfg := parseConfig()
	if cfg.Device != "gpu" {
		t.Fatalf("Device = %q, want legacy RYV_DEVICE_TYPE fallback", cfg.Device)
	}
}

func TestParseConfigPrefersDeviceEnv(t *testing.T) {
	withArgs(t, "ryvion-node")
	t.Setenv("RYV_DEVICE", "cpu")
	t.Setenv("RYV_DEVICE_TYPE", "gpu")

	cfg := parseConfig()
	if cfg.Device != "cpu" {
		t.Fatalf("Device = %q, want RYV_DEVICE to override RYV_DEVICE_TYPE", cfg.Device)
	}
}

func TestParseConfigDeviceFlagOverridesEnv(t *testing.T) {
	withArgs(t, "ryvion-node", "-device", "gpu")
	t.Setenv("RYV_DEVICE", "cpu")
	t.Setenv("RYV_DEVICE_TYPE", "")

	cfg := parseConfig()
	if cfg.Device != "gpu" {
		t.Fatalf("Device = %q, want -device flag", cfg.Device)
	}
}

func TestParseConfigTypeFlagAlias(t *testing.T) {
	withArgs(t, "ryvion-node", "-type", "gpu")
	t.Setenv("RYV_DEVICE", "")
	t.Setenv("RYV_DEVICE_TYPE", "")

	cfg := parseConfig()
	if cfg.Device != "gpu" {
		t.Fatalf("Device = %q, want deprecated -type alias", cfg.Device)
	}
}

func TestBuildHeartbeatPayloadDoesNotAdvertiseMissingOCI(t *testing.T) {
	payload, err := buildNodeHeartbeatPayload("node-test", zeroCaps(), "cpu", "", runtimeHealthSnapshot{Health: "missing"})
	if err != nil {
		t.Fatalf("buildNodeHeartbeatPayload() error = %v", err)
	}
	runtimeProfile := payload.CapabilityPassport.RuntimeProfile
	if runtimeProfile.OCIAvailable {
		t.Fatal("OCIAvailable = true, want false for missing runtime")
	}
	if len(runtimeProfile.SupportedRunnerKinds) != 0 {
		t.Fatalf("SupportedRunnerKinds = %v, want empty", runtimeProfile.SupportedRunnerKinds)
	}
}

func TestRunWorkUsesRuntimeRequirementsForLlamaCPP(t *testing.T) {
	work := &hub.WorkAssignment{
		JobID: "job_runtime_requirement",
		RuntimeRequirements: hub.RuntimeRequirements{
			NeedsLlamaCPP: true,
		},
		Image: "ghcr.io/ryvion/runner:old",
	}
	got := failureExecutor(work)
	if got != llamacpp.ExecutorKind {
		t.Fatalf("failureExecutor() = %q, want %q", got, llamacpp.ExecutorKind)
	}
}

func TestRunLlamaCPPWorkUploadFailureSubmitsFailureReceipt(t *testing.T) {
	llamaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("llama path = %q", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "local-model",
			"choices": []map[string]any{{
				"message":       map[string]any{"content": "answer"},
				"finish_reason": "stop",
			}},
			"usage": map[string]any{
				"completion_tokens": 4,
				"total_tokens":      7,
			},
		})
	}))
	defer llamaServer.Close()

	var receipts []map[string]any
	hubServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v1/node/upload/prepare":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"ok":      true,
				"put_url": "/api/v1/blob/job_upload_failure",
				"key":     "artifacts/job_upload_failure/output.json",
			})
		case "/api/v1/blob/job_upload_failure":
			http.Error(w, "upload unavailable", http.StatusInternalServerError)
		case "/api/v1/node/receipt":
			var body map[string]any
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Fatalf("decode receipt: %v", err)
			}
			receipts = append(receipts, body)
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": true})
		default:
			t.Fatalf("hub path = %q", r.URL.Path)
		}
	}))
	defer hubServer.Close()

	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	client := hub.New(hubServer.URL, pub, priv, hub.WithGRPCTransport("", "http", false))
	result, err := runLlamaCPPWork(context.Background(), client, &hub.WorkAssignment{
		JobID:    "job_upload_failure",
		Kind:     llamacpp.Task,
		SpecJSON: `{"task":"llama_cpp_inference","prompt":"hi"}`,
	}, llamacpp.Config{ServerURL: llamaServer.URL, Model: "local-model"})
	if err == nil || !strings.Contains(err.Error(), "artifact PUT failed") {
		t.Fatalf("runLlamaCPPWork() error = %v, want artifact upload failure", err)
	}
	if result == nil || result.MeteringUnits != 0 {
		t.Fatalf("result = %+v, want zero-metered failure snapshot", result)
	}
	if len(receipts) != 1 {
		t.Fatalf("receipts len = %d, want 1", len(receipts))
	}
	receipt := receipts[0]
	if got := receipt["metering_units"]; got != float64(0) {
		t.Fatalf("metering_units = %#v, want 0", got)
	}
	metadata, ok := receipt["metadata"].(map[string]any)
	if !ok {
		t.Fatalf("metadata = %#v", receipt["metadata"])
	}
	if metadata["executor"] != llamacpp.ExecutorKind {
		t.Fatalf("metadata.executor = %#v, want %q", metadata["executor"], llamacpp.ExecutorKind)
	}
	if metadata["reason"] != "artifact_upload_failed" {
		t.Fatalf("metadata.reason = %#v, want artifact_upload_failed", metadata["reason"])
	}
}

func zeroCaps() hw.CapSet { return hw.CapSet{CPUCores: 1, RAMBytes: 1 << 30} }

func withArgs(t *testing.T, args ...string) {
	t.Helper()

	oldArgs := os.Args
	oldCommandLine := flag.CommandLine
	fs := flag.NewFlagSet(args[0], flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	flag.CommandLine = fs
	os.Args = args

	t.Cleanup(func() {
		os.Args = oldArgs
		flag.CommandLine = oldCommandLine
	})
}
