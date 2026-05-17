package llamacpp

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestResidencyKeeperDisabledDoesNotAutoStart(t *testing.T) {
	t.Parallel()

	sidecar := newFakeResidencySidecar(LlamaCppSidecarStatus{
		Enabled:   true,
		Available: true,
		Running:   false,
		Healthy:   false,
	})
	keeper := NewResidencyKeeper(sidecar, ResidencyKeeperConfig{
		Enabled:            false,
		HealthInterval:     time.Second,
		RestartBackoff:     0,
		MaxRestartsPerHour: 10,
	})

	status := keeper.CheckOnce(context.Background())
	if sidecar.startCount() != 0 || sidecar.restartCount() != 0 {
		t.Fatalf("starts/restarts = %d/%d, want 0/0", sidecar.startCount(), sidecar.restartCount())
	}
	if status.Running || status.Healthy {
		t.Fatalf("status = %+v, want unchanged stopped sidecar", status)
	}
	if keeper.Snapshot().Enabled {
		t.Fatalf("keeper status enabled = true, want false")
	}
}

func TestResidencyKeeperEnabledStartsSidecarOnStartupCheck(t *testing.T) {
	t.Parallel()

	sidecar := newFakeResidencySidecar(LlamaCppSidecarStatus{
		Enabled:   true,
		Available: true,
		Running:   false,
		Healthy:   false,
	})
	sidecar.nextStart = LlamaCppSidecarStatus{
		Enabled:   true,
		Available: true,
		Running:   true,
		Healthy:   true,
		Reason:    "managed llama.cpp sidecar healthy",
	}
	keeper := NewResidencyKeeper(sidecar, ResidencyKeeperConfig{
		Enabled:            true,
		HealthInterval:     time.Second,
		RestartBackoff:     0,
		MaxRestartsPerHour: 10,
	})

	status := keeper.CheckOnce(context.Background())
	if sidecar.startCount() != 1 || sidecar.restartCount() != 0 {
		t.Fatalf("starts/restarts = %d/%d, want 1/0", sidecar.startCount(), sidecar.restartCount())
	}
	if !status.Running || !status.Healthy {
		t.Fatalf("status = %+v, want healthy running sidecar", status)
	}
	snapshot := keeper.Snapshot()
	if !snapshot.Enabled || snapshot.RestartCount != 1 || snapshot.LastRestartAt.IsZero() || snapshot.LastError != "" {
		t.Fatalf("keeper snapshot = %+v, want one clean keeper start", snapshot)
	}
}

func TestResidencyKeeperUnhealthySidecarTriggersRestart(t *testing.T) {
	t.Parallel()

	sidecar := newFakeResidencySidecar(LlamaCppSidecarStatus{
		Enabled:   true,
		Available: true,
		Running:   true,
		Healthy:   false,
		LastError: "connection refused",
	})
	sidecar.nextRestart = LlamaCppSidecarStatus{
		Enabled:   true,
		Available: true,
		Running:   true,
		Healthy:   true,
		Reason:    "managed llama.cpp sidecar healthy",
	}
	keeper := NewResidencyKeeper(sidecar, ResidencyKeeperConfig{
		Enabled:            true,
		HealthInterval:     time.Second,
		RestartBackoff:     0,
		MaxRestartsPerHour: 10,
	})

	status := keeper.CheckOnce(context.Background())
	if sidecar.restartCount() != 1 {
		t.Fatalf("restarts = %d, want 1", sidecar.restartCount())
	}
	if sidecar.startCount() != 0 {
		t.Fatalf("starts = %d, want 0 for unhealthy managed process", sidecar.startCount())
	}
	if !status.Running || !status.Healthy {
		t.Fatalf("status = %+v, want healthy after restart", status)
	}
}

func TestResidencyKeeperRestartStormCapWorks(t *testing.T) {
	t.Parallel()

	now := time.Unix(1_800_000_000, 0)
	sidecar := newFakeResidencySidecar(LlamaCppSidecarStatus{
		Enabled:   true,
		Available: true,
		Running:   false,
		Healthy:   false,
	})
	sidecar.nextStart = LlamaCppSidecarStatus{
		Enabled:   true,
		Available: true,
		Running:   false,
		Healthy:   false,
		Reason:    "connection refused",
	}
	keeper := NewResidencyKeeper(sidecar, ResidencyKeeperConfig{
		Enabled:            true,
		HealthInterval:     time.Second,
		RestartBackoff:     0,
		MaxRestartsPerHour: 2,
	}, WithResidencyKeeperNow(func() time.Time {
		return now
	}))

	_ = keeper.CheckOnce(context.Background())
	now = now.Add(time.Minute)
	_ = keeper.CheckOnce(context.Background())
	now = now.Add(time.Minute)
	_ = keeper.CheckOnce(context.Background())

	if sidecar.startCount() != 2 {
		t.Fatalf("starts = %d, want capped at 2", sidecar.startCount())
	}
	snapshot := keeper.Snapshot()
	if snapshot.RestartCount != 2 {
		t.Fatalf("restart_count = %d, want 2", snapshot.RestartCount)
	}
	if !strings.Contains(snapshot.LastError, "restart limit reached") {
		t.Fatalf("last error = %q, want restart limit", snapshot.LastError)
	}
}

func TestLlamaCppSidecarStatusViewKeepsWarmFieldsSafe(t *testing.T) {
	t.Parallel()

	view := BuildLlamaCppSidecarStatusView(LlamaCppSidecarStatus{
		Enabled:   true,
		Available: true,
		Running:   false,
		Healthy:   false,
		LastError: "raw_prompt output_text auth_token tensor_bytes",
		Reason:    "llama.cpp sidecar stopped",
	}, ResidencyKeeperStatus{
		Enabled:               true,
		RestartCount:          3,
		LastRestartAt:         time.Unix(123, 0),
		LastError:             "prompt_text generated_text secret raw_tensor",
		RestartBackoffSeconds: 10,
	})
	if !view.KeepWarmEnabled || view.RestartCount != 3 || view.RestartBackoffSeconds != 10 {
		t.Fatalf("view keepwarm fields = %+v", view)
	}
	if !strings.Contains(view.Reason, "keep_warm will restart") {
		t.Fatalf("reason = %q, want manual stop keep_warm behavior documented", view.Reason)
	}
	raw, err := json.Marshal(view)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	body := strings.ToLower(string(raw))
	for _, want := range []string{"keep_warm_enabled", "restart_count", "last_restart_at", "last_keepwarm_error", "restart_backoff_seconds"} {
		if !strings.Contains(body, want) {
			t.Fatalf("status JSON missing %q: %s", want, raw)
		}
	}
	for _, forbidden := range []string{"raw_prompt", "prompt_text", "model_output", "output_text", "generated_text", "tensor_bytes", "raw_tensor", "auth_token", "secret"} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("status JSON contains forbidden marker %q: %s", forbidden, raw)
		}
	}
}

func TestResidencyKeeperConfigFromEnvDefaults(t *testing.T) {
	t.Parallel()

	env := map[string]string{
		EnvKeepWarm: "1",
	}
	cfg := ResidencyKeeperConfigFromEnvWith(ConfigSource{
		Getenv: func(name string) string {
			return env[name]
		},
	})
	if !cfg.Enabled {
		t.Fatalf("enabled = false, want true")
	}
	if cfg.HealthInterval != DefaultHealthInterval ||
		cfg.RestartBackoff != DefaultRestartBackoff ||
		cfg.MaxRestartsPerHour != DefaultMaxRestartsPerHour {
		t.Fatalf("defaults = %+v", cfg)
	}
}

func TestResidencyKeeperConfigDefaultsToIdleNoWarmResidency(t *testing.T) {
	t.Parallel()

	cfg := ResidencyKeeperConfigFromEnvWith(ConfigSource{
		Getenv: func(string) string { return "" },
	})
	if cfg.Enabled {
		t.Fatal("enabled = true, want idle-safe warm residency off by default")
	}
}

func TestResidencyKeeperConfigHonorsModelWarmDisable(t *testing.T) {
	t.Parallel()

	cfg := ResidencyKeeperConfigFromEnvWith(ConfigSource{
		Getenv: func(name string) string {
			if name == EnvDisableModelWarm {
				return "1"
			}
			return ""
		},
	})
	if cfg.Enabled {
		t.Fatal("enabled = true, want disabled when model warm is disabled")
	}
}

type fakeResidencySidecar struct {
	mu          sync.Mutex
	cfg         LlamaCppSidecarConfig
	status      LlamaCppSidecarStatus
	nextStart   LlamaCppSidecarStatus
	nextRestart LlamaCppSidecarStatus
	starts      int
	restarts    int
}

func newFakeResidencySidecar(status LlamaCppSidecarStatus) *fakeResidencySidecar {
	return &fakeResidencySidecar{
		cfg: LlamaCppSidecarConfig{
			Enabled: status.Enabled,
			Host:    DefaultHost,
			Port:    DefaultPort,
		},
		status: status,
	}
}

func (f *fakeResidencySidecar) Config() LlamaCppSidecarConfig {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.cfg
}

func (f *fakeResidencySidecar) Status(context.Context) LlamaCppSidecarStatus {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.status
}

func (f *fakeResidencySidecar) Start(context.Context) LlamaCppSidecarStatus {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.starts++
	if f.nextStart.Enabled || f.nextStart.Available || f.nextStart.Running || f.nextStart.Healthy || f.nextStart.Reason != "" || f.nextStart.LastError != "" {
		f.status = f.nextStart
	}
	return f.status
}

func (f *fakeResidencySidecar) Restart(context.Context) LlamaCppSidecarStatus {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.restarts++
	if f.nextRestart.Enabled || f.nextRestart.Available || f.nextRestart.Running || f.nextRestart.Healthy || f.nextRestart.Reason != "" || f.nextRestart.LastError != "" {
		f.status = f.nextRestart
	}
	return f.status
}

func (f *fakeResidencySidecar) startCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.starts
}

func (f *fakeResidencySidecar) restartCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.restarts
}
