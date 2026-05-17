package llamacpp

import (
	"context"
	"strings"
	"sync"
	"time"
)

type ResidencySidecar interface {
	Config() LlamaCppSidecarConfig
	Status(context.Context) LlamaCppSidecarStatus
	Start(context.Context) LlamaCppSidecarStatus
	Restart(context.Context) LlamaCppSidecarStatus
}

type ResidencyKeeperOption func(*ResidencyKeeper)

type ResidencyKeeper struct {
	mu           sync.Mutex
	manager      ResidencySidecar
	cfg          ResidencyKeeperConfig
	now          func() time.Time
	started      bool
	cancel       context.CancelFunc
	restarts     []time.Time
	restartCount int
	lastRestart  time.Time
	lastError    string
}

type ResidencyKeeperStatus struct {
	Enabled               bool
	RestartCount          int
	LastRestartAt         time.Time
	LastError             string
	RestartBackoff        time.Duration
	RestartBackoffSeconds int
}

type LlamaCppSidecarStatusView struct {
	LlamaCppSidecarStatus
	KeepWarmEnabled       bool      `json:"keep_warm_enabled"`
	RestartCount          int       `json:"restart_count"`
	LastRestartAt         time.Time `json:"last_restart_at,omitempty"`
	LastKeepWarmError     string    `json:"last_keepwarm_error"`
	RestartBackoffSeconds int       `json:"restart_backoff_seconds"`
	Reason                string    `json:"reason"`
}

func NewResidencyKeeperFromEnv(manager ResidencySidecar, opts ...ResidencyKeeperOption) *ResidencyKeeper {
	return NewResidencyKeeper(manager, ResidencyKeeperConfigFromEnv(), opts...)
}

func NewResidencyKeeper(manager ResidencySidecar, cfg ResidencyKeeperConfig, opts ...ResidencyKeeperOption) *ResidencyKeeper {
	k := &ResidencyKeeper{
		manager: manager,
		cfg:     normalizeResidencyKeeperConfig(cfg),
		now:     time.Now,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(k)
		}
	}
	if k.now == nil {
		k.now = time.Now
	}
	k.cfg = normalizeResidencyKeeperConfig(k.cfg)
	return k
}

func WithResidencyKeeperNow(now func() time.Time) ResidencyKeeperOption {
	return func(k *ResidencyKeeper) {
		k.now = now
	}
}

func (k *ResidencyKeeper) Start(ctx context.Context) {
	if k == nil {
		return
	}
	k.mu.Lock()
	if k.started || !k.cfg.Enabled || k.manager == nil {
		k.mu.Unlock()
		return
	}
	loopCtx, cancel := context.WithCancel(ctx)
	k.started = true
	k.cancel = cancel
	interval := k.cfg.HealthInterval
	k.mu.Unlock()

	go k.loop(loopCtx, interval)
}

func (k *ResidencyKeeper) Stop() {
	if k == nil {
		return
	}
	k.mu.Lock()
	cancel := k.cancel
	k.cancel = nil
	k.started = false
	k.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

func (k *ResidencyKeeper) CheckOnce(ctx context.Context) LlamaCppSidecarStatus {
	if k == nil || k.manager == nil {
		return LlamaCppSidecarStatus{}
	}
	k.mu.Lock()
	enabled := k.cfg.Enabled
	k.mu.Unlock()
	if !enabled {
		return k.manager.Status(ctx)
	}

	status := k.manager.Status(ctx)
	if !status.Enabled {
		k.recordKeepWarmError("")
		return status
	}
	if !status.Available {
		k.recordKeepWarmError(status.Reason)
		return status
	}
	if status.Running && status.Healthy {
		k.recordKeepWarmError("")
		return status
	}
	if !k.reserveRestartAttempt() {
		return status
	}
	if status.Running && !status.Attached {
		status = k.manager.Restart(ctx)
	} else {
		status = k.manager.Start(ctx)
	}
	k.recordPostRestartStatus(status)
	return status
}

func (k *ResidencyKeeper) Snapshot() ResidencyKeeperStatus {
	if k == nil {
		return ResidencyKeeperStatus{}
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	cfg := normalizeResidencyKeeperConfig(k.cfg)
	return ResidencyKeeperStatus{
		Enabled:               cfg.Enabled,
		RestartCount:          k.restartCount,
		LastRestartAt:         k.lastRestart,
		LastError:             cleanStatusText(k.lastError, maxStatusReasonLen),
		RestartBackoff:        cfg.RestartBackoff,
		RestartBackoffSeconds: durationSeconds(cfg.RestartBackoff),
	}
}

func BuildLlamaCppSidecarStatusView(status LlamaCppSidecarStatus, keeper ResidencyKeeperStatus) LlamaCppSidecarStatusView {
	status.LastError = cleanStatusText(status.LastError, maxStatusReasonLen)
	status.Reason = keepWarmReason(status, keeper)
	return LlamaCppSidecarStatusView{
		LlamaCppSidecarStatus: status,
		KeepWarmEnabled:       keeper.Enabled,
		RestartCount:          keeper.RestartCount,
		LastRestartAt:         keeper.LastRestartAt,
		LastKeepWarmError:     cleanStatusText(keeper.LastError, maxStatusReasonLen),
		RestartBackoffSeconds: keeper.RestartBackoffSeconds,
		Reason:                status.Reason,
	}
}

func (k *ResidencyKeeper) loop(ctx context.Context, interval time.Duration) {
	if interval <= 0 {
		interval = DefaultHealthInterval
	}
	_ = k.CheckOnce(ctx)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			_ = k.CheckOnce(ctx)
		}
	}
}

func (k *ResidencyKeeper) reserveRestartAttempt() bool {
	k.mu.Lock()
	defer k.mu.Unlock()
	cfg := normalizeResidencyKeeperConfig(k.cfg)
	now := k.currentTimeLocked()
	if !k.lastRestart.IsZero() && cfg.RestartBackoff > 0 && now.Sub(k.lastRestart) < cfg.RestartBackoff {
		k.lastError = "llama.cpp keep_warm restart backoff active"
		return false
	}
	k.pruneRestartWindowLocked(now)
	if cfg.MaxRestartsPerHour > 0 && len(k.restarts) >= cfg.MaxRestartsPerHour {
		k.lastError = "llama.cpp keep_warm restart limit reached"
		return false
	}
	k.restarts = append(k.restarts, now)
	k.restartCount++
	k.lastRestart = now
	k.lastError = ""
	return true
}

func (k *ResidencyKeeper) recordPostRestartStatus(status LlamaCppSidecarStatus) {
	if status.Running && status.Healthy {
		k.recordKeepWarmError("")
		return
	}
	reason := firstNonEmptyRuntimeText(status.LastError, status.Reason)
	if strings.TrimSpace(reason) == "" {
		reason = "llama.cpp keep_warm restart did not become healthy"
	}
	k.recordKeepWarmError(reason)
}

func (k *ResidencyKeeper) recordKeepWarmError(value string) {
	k.mu.Lock()
	defer k.mu.Unlock()
	k.lastError = cleanStatusText(value, maxStatusReasonLen)
}

func (k *ResidencyKeeper) currentTimeLocked() time.Time {
	if k.now == nil {
		return time.Now()
	}
	return k.now()
}

func (k *ResidencyKeeper) pruneRestartWindowLocked(now time.Time) {
	cutoff := now.Add(-time.Hour)
	keep := k.restarts[:0]
	for _, ts := range k.restarts {
		if ts.After(cutoff) || ts.Equal(cutoff) {
			keep = append(keep, ts)
		}
	}
	k.restarts = keep
}

func keepWarmReason(status LlamaCppSidecarStatus, keeper ResidencyKeeperStatus) string {
	reason := cleanStatusText(status.Reason, maxStatusReasonLen)
	if !keeper.Enabled {
		return reason
	}
	if status.Enabled && status.Available && !status.Running {
		return "llama.cpp sidecar stopped; keep_warm will restart unless node-agent is stopped"
	}
	if strings.TrimSpace(keeper.LastError) != "" && !status.Healthy {
		return cleanStatusText(firstNonEmptyRuntimeText(reason, keeper.LastError), maxStatusReasonLen)
	}
	return reason
}

func durationSeconds(value time.Duration) int {
	if value <= 0 {
		return 0
	}
	seconds := int(value / time.Second)
	if value%time.Second != 0 {
		seconds++
	}
	return seconds
}
