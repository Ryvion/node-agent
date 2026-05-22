package llamacpp

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	capshardware "github.com/Ryvion/ryvion-node/internal/capabilities/hardware"
	"github.com/Ryvion/ryvion-node/internal/runtimes/inventory"
)

const (
	defaultHealthTimeout = 750 * time.Millisecond
	maxStatusReasonLen   = 256
	maxLogBufferBytes    = 64 * 1024
	maxRuntimePathLen    = 512

	fallbackPartialGPULayers = 35
)

var ggufQuantizationPattern = regexp.MustCompile(`(?i)(?:^|[._\-\s])((?:IQ|Q)[0-9](?:_[A-Z0-9]+){0,3}|BF16|F16|F32)(?:[._\-\s]|$)`)
var nativeMTPModelPattern = regexp.MustCompile(`(?i)(?:^|[._\-\s])mtp(?:[._\-\s]|$)`)

type managedProcess interface {
	PID() int
	Wait() error
	Kill() error
}

type processStarter func(ctx context.Context, binary string, args []string, output io.Writer) (managedProcess, error)

type ManagerOption func(*Manager)

type Manager struct {
	mu            sync.Mutex
	cfg           LlamaCppSidecarConfig
	client        healthHTTPClient
	starter       processStarter
	logs          *boundedLogBuffer
	process       managedProcess
	processCancel context.CancelFunc
	startedAt     time.Time
	lastHealthAt  time.Time
	lastError     string
	serverProps   *LlamaCppServerProperties
	attached      bool
	stopping      bool
	now           func() time.Time
	healthTimeout time.Duration
}

func NewManager(cfg LlamaCppSidecarConfig, opts ...ManagerOption) *Manager {
	m := &Manager{
		cfg:           normalizeConfig(cfg),
		client:        defaultHealthHTTPClient(defaultHealthTimeout),
		starter:       defaultProcessStarter,
		logs:          newBoundedLogBuffer(maxLogBufferBytes),
		now:           time.Now,
		healthTimeout: defaultHealthTimeout,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(m)
		}
	}
	if m.client == nil {
		m.client = defaultHealthHTTPClient(defaultHealthTimeout)
	}
	if m.starter == nil {
		m.starter = defaultProcessStarter
	}
	if m.logs == nil {
		m.logs = newBoundedLogBuffer(maxLogBufferBytes)
	}
	if m.now == nil {
		m.now = time.Now
	}
	if m.healthTimeout <= 0 {
		m.healthTimeout = defaultHealthTimeout
	}
	return m
}

func NewManagerFromEnv() *Manager {
	return NewManager(ConfigFromEnv())
}

func WithProcessStarter(starter processStarter) ManagerOption {
	return func(m *Manager) {
		m.starter = starter
	}
}

func WithHealthClient(client healthHTTPClient) ManagerOption {
	return func(m *Manager) {
		m.client = client
	}
}

func WithHealthTimeout(timeout time.Duration) ManagerOption {
	return func(m *Manager) {
		m.healthTimeout = timeout
	}
}

func (m *Manager) Config() LlamaCppSidecarConfig {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.cfg
}

func (m *Manager) SetEnabled(enabled bool) LlamaCppSidecarConfig {
	if m == nil {
		return NewManager(LlamaCppSidecarConfig{}).SetEnabled(enabled)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.cfg.Enabled = enabled
	m.cfg = normalizeConfig(m.cfg)
	return m.cfg
}

func (m *Manager) SetModelPath(modelPath string) LlamaCppSidecarConfig {
	if m == nil {
		return NewManager(LlamaCppSidecarConfig{}).SetModelPath(modelPath)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.cfg.ModelPath = cleanConfigText(modelPath, maxConfigTextLen)
	// V8 Phase 1.6: keep an operator-pinned speculative drafter when the
	// target model changes at runtime. Auto-discovery is opt-in because
	// draft/target compatibility is backend-build sensitive.
	m.cfg.DraftModelPath = redriveDraftModelPath(m.cfg.DraftModelPath, m.cfg.ModelPath)
	redriveFastCUDAProfile(&m.cfg, false)
	m.cfg = normalizeConfig(m.cfg)
	return m.cfg
}

// redriveDraftModelPath returns the draft companion for a freshly switched
// target model. Operator pins always win. Auto-discovery is disabled unless
// EnvDraftAuto is set, avoiding accidental draft/target pairs that can make
// llama-server fail to start on a specific platform build.
func redriveDraftModelPath(currentDraft, newTarget string) string {
	envDraft := strings.TrimSpace(os.Getenv(EnvDraftModel))
	if envDraft != "" {
		// Operator pin wins.
		return envDraft
	}
	if newTarget == "" {
		return ""
	}
	if !envBool(os.Getenv(EnvDraftAuto)) {
		return ""
	}
	source := normalizeConfigSource(ConfigSource{})
	discovered := discoverDraftModelPath(source, newTarget)
	if discovered != "" {
		return discovered
	}
	// Auto-discovery turned up nothing for the new target. Drop the
	// stale draft path rather than running the previous target's
	// drafter against a different model.
	return ""
}

func (m *Manager) Status(ctx context.Context) LlamaCppSidecarStatus {
	if m == nil {
		return NewManager(LlamaCppSidecarConfig{}).Status(ctx)
	}
	m.mu.Lock()
	status := m.statusLocked()
	shouldHealthCheck := status.Enabled && status.Available
	m.mu.Unlock()

	if shouldHealthCheck {
		health := CheckHealth(ctx, status.BaseURL, m.client, m.healthTimeout)
		props := LlamaCppServerProperties{}
		propsOK := false
		if health.Healthy {
			props, propsOK = FetchServerProperties(ctx, status.BaseURL, m.client, m.healthTimeout)
		}
		m.mu.Lock()
		m.applyHealthLocked(health)
		if propsOK {
			m.serverProps = normalizeServerProperties(&props)
		} else if !health.Healthy {
			m.serverProps = nil
		}
		status = m.statusLocked()
		m.mu.Unlock()
	}
	return status
}

func (m *Manager) Start(ctx context.Context) LlamaCppSidecarStatus {
	if m == nil {
		return NewManager(LlamaCppSidecarConfig{}).Start(ctx)
	}
	m.ensureBundledServer(ctx)
	m.mu.Lock()
	status := m.statusLocked()
	if !status.Enabled {
		m.lastError = ""
		m.mu.Unlock()
		return status
	}
	if !status.Available {
		m.lastError = status.Reason
		status = m.statusLocked()
		m.mu.Unlock()
		return status
	}
	if m.process != nil || m.attached {
		m.mu.Unlock()
		return m.Status(ctx)
	}
	baseURL := status.BaseURL
	m.mu.Unlock()

	if health := CheckHealth(ctx, baseURL, m.client, m.healthTimeout); health.Healthy {
		m.mu.Lock()
		m.applyHealthLocked(health)
		status = m.statusLocked()
		m.mu.Unlock()
		return status
	}

	m.mu.Lock()
	if m.process != nil || m.attached {
		m.mu.Unlock()
		return m.Status(ctx)
	}
	processCtx, cancel := context.WithCancel(context.Background())
	args := buildServerArgs(m.cfg)
	process, err := m.starter(processCtx, m.cfg.ServerPath, args, m.logs)
	if err != nil {
		cancel()
		m.lastError = cleanStatusText(err.Error(), maxStatusReasonLen)
		status = m.statusLocked()
		m.mu.Unlock()
		return status
	}
	m.process = process
	m.processCancel = cancel
	m.startedAt = m.now()
	m.attached = false
	m.stopping = false
	m.lastError = ""
	go m.waitForManagedProcess(process)
	m.mu.Unlock()

	return m.Status(ctx)
}

func (m *Manager) ensureBundledServer(ctx context.Context) {
	if m == nil {
		return
	}
	m.mu.Lock()
	cfg := normalizeConfig(m.cfg)
	shouldEnsure := shouldEnsureBundledServer(cfg)
	serverPath := cfg.ServerPath
	m.mu.Unlock()
	if !shouldEnsure {
		return
	}
	installed, reason, err := ensureWindowsGPUServerBundle(ctx, serverPath)
	m.mu.Lock()
	defer m.mu.Unlock()
	if err != nil {
		m.lastError = cleanStatusText("llama-server bundle repair failed: "+err.Error(), maxStatusReasonLen)
		return
	}
	if installed {
		m.lastError = ""
		m.serverProps = nil
		if m.logs != nil {
			_, _ = m.logs.Write([]byte("llama-server bundle refreshed: " + reason + "\n"))
		}
	}
}

func shouldEnsureBundledServer(cfg LlamaCppSidecarConfig) bool {
	cfg = normalizeConfig(cfg)
	if !cfg.Enabled || cfg.ServerPathExplicit || strings.TrimSpace(cfg.ServerPath) == "" {
		return false
	}
	if runtime.GOOS != "windows" || runtime.GOARCH != "amd64" {
		return false
	}
	return strings.EqualFold(filepath.Base(cfg.ServerPath), "llama-server.exe")
}

func (m *Manager) Stop(ctx context.Context) LlamaCppSidecarStatus {
	if m == nil {
		return NewManager(LlamaCppSidecarConfig{}).Stop(ctx)
	}
	m.mu.Lock()
	process := m.process
	cancel := m.processCancel
	if process == nil {
		m.attached = false
		m.lastError = ""
		status := m.statusLocked()
		m.mu.Unlock()
		return status
	}
	m.stopping = true
	m.process = nil
	m.processCancel = nil
	m.attached = false
	m.startedAt = time.Time{}
	m.mu.Unlock()

	if cancel != nil {
		cancel()
	}
	if err := process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
		m.mu.Lock()
		m.lastError = cleanStatusText(err.Error(), maxStatusReasonLen)
		status := m.statusLocked()
		m.mu.Unlock()
		return status
	}

	m.mu.Lock()
	m.lastError = ""
	status := m.statusLocked()
	m.mu.Unlock()
	return status
}

func (m *Manager) Restart(ctx context.Context) LlamaCppSidecarStatus {
	if m == nil {
		return NewManager(LlamaCppSidecarConfig{}).Restart(ctx)
	}
	_ = m.Stop(ctx)
	return m.Start(ctx)
}

func (m *Manager) RestartWithModel(ctx context.Context, modelPath string) LlamaCppSidecarStatus {
	if m == nil {
		return NewManager(LlamaCppSidecarConfig{}).RestartWithModel(ctx, modelPath)
	}
	_ = m.SetModelPath(modelPath)
	m.rehomeExternalServerForManagedRestart()
	return m.Restart(ctx)
}

func (m *Manager) RestartWithModelFastCUDA(ctx context.Context, modelPath string) LlamaCppSidecarStatus {
	return m.restartWithModelLaunchFallback(ctx, modelPath, func(cfg *LlamaCppSidecarConfig) {
		redriveFastCUDAProfile(cfg, true)
	})
}

func (m *Manager) RestartWithModelSafeCUDA(ctx context.Context, modelPath string) LlamaCppSidecarStatus {
	return m.restartWithModelLaunchFallback(ctx, modelPath, func(cfg *LlamaCppSidecarConfig) {
		cfg.FastDefaults = false
		cfg.DraftModelPath = ""
		cfg.LaunchProfile = LaunchProfileCUDASafe
	})
}

func (m *Manager) RestartWithModelPartialGPU(ctx context.Context, modelPath string) LlamaCppSidecarStatus {
	return m.restartWithModelLaunchFallback(ctx, modelPath, func(cfg *LlamaCppSidecarConfig) {
		cfg.FastDefaults = false
		cfg.DraftModelPath = ""
		if cfg.GPULayers > fallbackPartialGPULayers {
			cfg.GPULayers = fallbackPartialGPULayers
		}
		cfg.LaunchProfile = LaunchProfileCUDAPartial
	})
}

func (m *Manager) restartWithModelLaunchFallback(ctx context.Context, modelPath string, mutate func(*LlamaCppSidecarConfig)) LlamaCppSidecarStatus {
	if m == nil {
		return NewManager(LlamaCppSidecarConfig{}).restartWithModelLaunchFallback(ctx, modelPath, mutate)
	}
	modelPath = cleanConfigText(modelPath, maxConfigTextLen)
	m.mu.Lock()
	if modelPath != "" {
		m.cfg.ModelPath = modelPath
		m.cfg.DraftModelPath = redriveDraftModelPath(m.cfg.DraftModelPath, m.cfg.ModelPath)
	}
	if mutate != nil {
		mutate(&m.cfg)
	}
	m.cfg = normalizeConfig(m.cfg)
	m.mu.Unlock()
	m.rehomeExternalServerForManagedRestart()
	return m.Restart(ctx)
}

func redriveFastCUDAProfile(cfg *LlamaCppSidecarConfig, force bool) {
	if cfg == nil {
		return
	}
	if !cfg.GPULayersExplicit && cfg.GPULayers > 0 && cfg.GPULayers < DefaultGPULayers {
		cfg.GPULayers = DefaultGPULayers
	}
	if !cfg.FastDefaultsExplicit && cfg.GPULayers > 0 && (force || cudaLaunchDefaultsSupportedGOOS(runtime.GOOS)) {
		cfg.FastDefaults = true
	}
	cfg.LaunchProfile = ""
}

func cudaLaunchDefaultsSupportedGOOS(goos string) bool {
	switch strings.ToLower(strings.TrimSpace(goos)) {
	case "windows", "linux":
		return true
	default:
		return false
	}
}

func (m *Manager) rehomeExternalServerForManagedRestart() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.process != nil {
		return
	}
	port, ok := freeLocalPortExcept(m.cfg.Host, m.cfg.Port)
	if !ok {
		return
	}
	m.cfg.Port = port
	m.cfg = normalizeConfig(m.cfg)
	m.attached = false
	m.lastError = ""
}

func (m *Manager) waitForManagedProcess(process managedProcess) {
	err := process.Wait()
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.process != process {
		m.stopping = false
		return
	}
	m.process = nil
	m.processCancel = nil
	m.attached = false
	m.startedAt = time.Time{}
	if !m.stopping && err != nil {
		m.lastError = cleanStatusText(processExitStatus(err, m.logs), maxStatusReasonLen)
	}
	m.stopping = false
}

func processExitStatus(err error, logs *boundedLogBuffer) string {
	if err == nil {
		return ""
	}
	message := err.Error()
	if logs != nil {
		if tail := logs.TailString(512); strings.TrimSpace(tail) != "" {
			message += ": " + tail
		}
	}
	return message
}

func (m *Manager) applyHealthLocked(health HealthResult) {
	if health.CheckedAt.IsZero() {
		health.CheckedAt = m.now()
	}
	m.lastHealthAt = health.CheckedAt
	if health.Healthy {
		m.lastError = ""
		if m.process == nil {
			m.attached = true
		}
		return
	}
	if m.process == nil {
		m.attached = false
		if strings.TrimSpace(m.lastError) != "" {
			return
		}
		// A stopped on-demand sidecar is expected when keep-warm is disabled.
		// Preserve real process/startup errors, but do not surface a routine
		// connection refusal from probing an intentionally idle llama-server.
		return
	}
	m.lastError = cleanStatusText(health.Error, maxStatusReasonLen)
}

func (m *Manager) statusLocked() LlamaCppSidecarStatus {
	cfg := normalizeConfig(m.cfg)
	m.cfg = cfg
	meta := modelMetadata(cfg.ModelPath)
	serverOK := serverPathAvailable(cfg.ServerPath)
	modelOK := meta.readable
	running := m.process != nil || m.attached
	healthy := running && m.lastError == "" && !m.lastHealthAt.IsZero()
	pid := 0
	if m.process != nil {
		pid = m.process.PID()
	}
	launch := buildLaunchConfig(cfg, m.process != nil, m.attached && m.process == nil)

	reason := "llama.cpp sidecar ready"
	if !cfg.Enabled {
		reason = "llama.cpp sidecar disabled"
	} else if !serverOK {
		reason = "llama-server binary not detected"
	} else if cfg.ModelPath == "" {
		reason = "GGUF model not configured or detected"
	} else if !modelOK {
		reason = "GGUF model path is not readable"
	} else if running && healthy {
		if m.attached && m.process == nil {
			reason = "attached to healthy llama.cpp server"
		} else {
			reason = "managed llama.cpp sidecar healthy"
		}
	} else if running {
		reason = "llama.cpp sidecar running; health not confirmed"
	} else if m.lastError != "" {
		reason = m.lastError
	} else {
		reason = "llama.cpp sidecar stopped; starts on inference demand"
	}

	nativeMTP := nativeMTPUsable(cfg)
	draftMeta := modelMetadata(cfg.DraftModelPath)
	draftReady := cfg.DraftModelPath != "" && draftMeta.readable && !nativeMTP
	speculativeMethod := speculativeMethodFromConfig(cfg, draftReady)
	speculativeConfigured := speculativeMethod != "" && cfg.Enabled && serverOK && modelOK
	speculativeActive := speculativeConfigured && running && healthy
	acceleration, accelerationReason := sidecarAccelerationStatus(cfg, m.serverProps)
	if running && healthy {
		if suffix := sidecarAccelerationReasonSuffix(accelerationReason); suffix != "" {
			reason += "; " + suffix
		}
	}

	return LlamaCppSidecarStatus{
		Enabled:                cfg.Enabled,
		Available:              serverOK && modelOK,
		Running:                running,
		Healthy:                healthy,
		Attached:               m.attached && m.process == nil,
		PID:                    pid,
		BaseURL:                baseURL(cfg.Host, cfg.Port),
		ServerPath:             cfg.ServerPath,
		ModelPath:              cfg.ModelPath,
		ContextSize:            cfg.ContextSize,
		ModelFilename:          meta.filename,
		ModelSizeBytes:         meta.sizeBytes,
		ModelFamilyHint:        meta.familyHint,
		QuantizationHint:       meta.quantizationHint,
		StartedAt:              m.startedAt,
		LastHealthAt:           m.lastHealthAt,
		LastError:              cleanStatusText(m.lastError, maxStatusReasonLen),
		Backend:                BackendName,
		Launch:                 launch,
		ServerProperties:       cloneServerProperties(m.serverProps),
		Acceleration:           acceleration,
		AccelerationReason:     accelerationReason,
		OpenAICompatible:       true,
		SupportsTextGeneration: true,
		SupportsStreaming:      true,
		SupportsKVAccess:       false,
		SupportsTensorHooks:    false,
		Reason:                 cleanStatusText(reason, maxStatusReasonLen),

		// V8 speculative decoding (Level 0).
		SpeculativeEnabled:   speculativeConfigured,
		SpeculativeActive:    speculativeActive,
		SpeculativeMethod:    speculativeMethod,
		NativeMTP:            nativeMTP,
		DraftModelPath:       draftPathForStatus(cfg, draftReady),
		DraftModelFilename:   draftFilenameForStatus(draftMeta, draftReady),
		DraftModelSizeBytes:  draftSizeForStatus(draftMeta, draftReady),
		DraftModelFamilyHint: draftFamilyForStatus(draftMeta, draftReady),
		DraftMaxTokens:       cfg.DraftMaxTokens,
		DraftMinTokens:       cfg.DraftMinTokens,
	}
}

func BuildBackendRuntimes(status LlamaCppSidecarStatus) BackendRuntimes {
	return NormalizeBackendRuntimes(BackendRuntimes{
		LlamaCPP: BackendRuntimeStatus{
			Enabled:                  status.Enabled,
			Available:                status.Available,
			Running:                  status.Running,
			Healthy:                  status.Healthy,
			Backend:                  firstNonEmptyRuntimeText(status.Backend, BackendName),
			BaseURL:                  status.BaseURL,
			ModelID:                  status.ModelFilename,
			ModelPath:                status.ModelPath,
			MaxContextTokens:         status.ContextSize,
			ModelFilename:            status.ModelFilename,
			ModelSizeBytes:           status.ModelSizeBytes,
			ModelFamilyHint:          status.ModelFamilyHint,
			QuantizationHint:         status.QuantizationHint,
			OpenAICompatible:         status.OpenAICompatible,
			SupportsTextGeneration:   status.SupportsTextGeneration,
			SupportsStreaming:        status.SupportsStreaming,
			Acceleration:             statusAccelerationOrDefault(status),
			AccelerationReason:       statusAccelerationReasonOrDefault(status),
			GPUArchitecture:          gpuArchitectureFromHardware(capshardware.CapacityInventory{}),
			SupportsKVAccess:         status.SupportsKVAccess,
			SupportsTensorHooks:      status.SupportsTensorHooks,
			LastHealthAtUnixMs:       unixMilliOrZero(status.LastHealthAt),
			LastError:                status.LastError,
			Launch:                   cloneLaunchConfig(status.Launch),
			ServerProperties:         cloneServerProperties(status.ServerProperties),
			OptimizationCapabilities: statusOptimizationCapabilities(status),
		},
	})
}

func buildLaunchConfig(cfg LlamaCppSidecarConfig, managed bool, attached bool) *LlamaCppLaunchConfig {
	cfg = normalizeConfig(cfg)
	mode := "stopped"
	switch {
	case !cfg.Enabled:
		mode = "disabled"
	case managed:
		mode = "managed"
	case attached:
		mode = "attached"
	}
	serverFilename := ""
	if strings.TrimSpace(cfg.ServerPath) != "" {
		serverFilename = filepath.Base(cfg.ServerPath)
	}
	return normalizeLaunchConfig(&LlamaCppLaunchConfig{
		Mode:                     mode,
		Managed:                  managed,
		Attached:                 attached,
		ServerPath:               cfg.ServerPath,
		ServerFilename:           serverFilename,
		ConfiguredGPULayers:      cfg.GPULayers,
		FastDefaultsEnabled:      cfg.FastDefaults,
		ConfiguredDraftGPULayers: cfg.DraftGPULayers,
		Profile:                  cfg.LaunchProfile,
	})
}

func speculativeMethodFromConfig(cfg LlamaCppSidecarConfig, draftReady bool) string {
	cfg = normalizeConfig(cfg)
	if nativeMTPUsable(cfg) {
		return SpeculativeMethodNativeMTP
	}
	if draftReady {
		return SpeculativeMethodBackendLocalDraft
	}
	if draftlessSpecType(cfg.SpecType) {
		return cfg.SpecType
	}
	return ""
}

func nativeMTPConfigured(cfg LlamaCppSidecarConfig) bool {
	cfg.ExtraArgs = sanitizeExtraArgs(strings.Join(cfg.ExtraArgs, " "))
	return cfg.NativeMTP ||
		(cfg.NativeMTPAuto && modelSupportsNativeMTP(cfg.ModelPath)) ||
		specTypeExtraArgContains(cfg.ExtraArgs, "draft-mtp")
}

func nativeMTPUsable(cfg LlamaCppSidecarConfig) bool {
	return nativeMTPConfigured(cfg) && modelSupportsNativeMTP(cfg.ModelPath)
}

func modelSupportsNativeMTP(modelPath string) bool {
	modelPath = strings.TrimSpace(modelPath)
	if modelPath == "" {
		return false
	}
	return nativeMTPModelPattern.MatchString(filepath.Base(modelPath))
}

func draftlessSpecType(specType string) bool {
	switch normalizeSpecType(specType) {
	case SpeculativeMethodNGramSimple, SpeculativeMethodNGramMapK, SpeculativeMethodNGramMapK4V, SpeculativeMethodNGramMod, SpeculativeMethodNGramCache:
		return true
	default:
		return false
	}
}

func specTypeExtraArgContains(args []string, want string) bool {
	want = strings.ToLower(strings.TrimSpace(want))
	if want == "" {
		return false
	}
	for i, arg := range args {
		if strings.TrimSpace(arg) != "--spec-type" || i+1 >= len(args) {
			continue
		}
		for _, part := range strings.Split(args[i+1], ",") {
			if strings.EqualFold(strings.TrimSpace(part), want) {
				return true
			}
		}
	}
	return false
}

func draftPathForStatus(cfg LlamaCppSidecarConfig, draftReady bool) string {
	if !draftReady {
		return ""
	}
	return cfg.DraftModelPath
}

func draftFilenameForStatus(meta modelMeta, draftReady bool) string {
	if !draftReady {
		return ""
	}
	return meta.filename
}

func draftSizeForStatus(meta modelMeta, draftReady bool) int64 {
	if !draftReady {
		return 0
	}
	return meta.sizeBytes
}

func draftFamilyForStatus(meta modelMeta, draftReady bool) string {
	if !draftReady {
		return ""
	}
	return meta.familyHint
}

func statusOptimizationCapabilities(status LlamaCppSidecarStatus) []OptimizationCapability {
	if !status.NativeMTP && status.SpeculativeMethod != SpeculativeMethodNativeMTP {
		return []OptimizationCapability{}
	}
	return normalizeOptimizationCapabilities([]OptimizationCapability{{
		Name:      SpeculativeMethodNativeMTP,
		Supported: true,
		Enabled:   status.SpeculativeEnabled && status.SpeculativeMethod == SpeculativeMethodNativeMTP,
		Backend:   BackendName,
		Notes:     "Uses llama.cpp --spec-type draft-mtp with native model MTP heads.",
	}}, BackendName, "")
}

func BuildBackendRuntimesWithInventory(status LlamaCppSidecarStatus, inventory runtimeinventory.Inventory, hardware capshardware.CapacityInventory) BackendRuntimes {
	return EnrichBackendRuntimes(BuildBackendRuntimes(status), inventory, hardware)
}

func EnrichBackendRuntimes(runtimes BackendRuntimes, inventory runtimeinventory.Inventory, hardware capshardware.CapacityInventory) BackendRuntimes {
	inventory = runtimeinventory.NormalizeInventory(inventory)
	hardware = capshardware.NormalizeInventory(hardware)
	runtimes = NormalizeBackendRuntimes(runtimes)
	if !runtimes.LlamaCPP.Loaded {
		runtimes.LlamaCPP.Acceleration = mergeAcceleration(runtimes.LlamaCPP.Acceleration, accelerationFromHardware(hardware))
	}
	runtimes.LlamaCPP.GPUArchitecture = firstNonEmptyRuntimeText(runtimes.LlamaCPP.GPUArchitecture, gpuArchitectureFromHardware(hardware))
	runtimes.LlamaCPP.GPUComputeCapability = firstNonEmptyRuntimeText(runtimes.LlamaCPP.GPUComputeCapability, hardware.ComputeCapability)
	for _, candidate := range inventory.BackendCandidates {
		runtime := runtimeFromBackendCandidate(candidate, hardware)
		switch candidate.Backend {
		case runtimeinventory.BackendCandidateLlamaCPP:
			if !runtimes.LlamaCPP.Available && runtime.Available {
				runtimes.LlamaCPP = mergeDetectedRuntime(runtimes.LlamaCPP, runtime)
			}
		case runtimeinventory.BackendCandidateTensorRTLLM:
			runtimes.TensorRTLLM = runtime
		case runtimeinventory.BackendCandidateVLLM:
			runtimes.VLLM = runtime
		case runtimeinventory.BackendCandidateSGLang:
			runtimes.SGLang = mergeDetectedRuntime(runtimes.SGLang, runtime)
		case runtimeinventory.BackendCandidateOllama:
			if runtime.Available {
				runtimes.Other = append(runtimes.Other, runtime)
			}
		}
	}
	return NormalizeBackendRuntimes(runtimes)
}

func NormalizeBackendRuntimes(runtimes BackendRuntimes) BackendRuntimes {
	llama := normalizeBackendRuntimeStatus(runtimes.LlamaCPP, BackendName)
	tensorRTLLM := normalizeBackendRuntimeStatus(runtimes.TensorRTLLM, runtimeinventory.BackendCandidateTensorRTLLM)
	vllm := normalizeBackendRuntimeStatus(runtimes.VLLM, runtimeinventory.BackendCandidateVLLM)
	sglang := normalizeBackendRuntimeStatus(runtimes.SGLang, runtimeinventory.BackendCandidateSGLang)
	other := normalizeOtherBackendRuntimes(runtimes.Other)
	return BackendRuntimes{LlamaCPP: llama, TensorRTLLM: tensorRTLLM, VLLM: vllm, SGLang: sglang, Other: other}
}

func normalizeBackendRuntimeStatus(llama BackendRuntimeStatus, defaultBackend string) BackendRuntimeStatus {
	llama.Backend = cleanRuntimeCompactText(firstNonEmptyRuntimeText(llama.Backend, BackendName), 64)
	if strings.TrimSpace(defaultBackend) != "" && llama.Backend == BackendName && defaultBackend != BackendName {
		llama.Backend = cleanRuntimeCompactText(defaultBackend, 64)
	}
	if llama.Backend == "" {
		llama.Backend = cleanRuntimeCompactText(defaultBackend, 64)
	}
	llama.BaseURL = cleanRuntimeBaseURL(llama.BaseURL)
	llama.ModelPath = cleanRuntimePath(llama.ModelPath)
	llama.ModelFilename = cleanRuntimePath(llama.ModelFilename)
	llama.ModelID = cleanRuntimePath(firstNonEmptyRuntimeText(llama.ModelID, llama.ModelFilename))
	llama.WarmModelID = cleanRuntimePath(llama.WarmModelID)
	llama.ModelFamilyHint = cleanRuntimeCompactText(llama.ModelFamilyHint, 64)
	llama.QuantizationHint = cleanRuntimeCompactText(llama.QuantizationHint, 64)
	llama.GPUArchitecture = cleanRuntimeCompactText(llama.GPUArchitecture, 64)
	llama.GPUComputeCapability = cleanRuntimeCompactText(llama.GPUComputeCapability, 64)
	llama.AccelerationReason = cleanRuntimeCompactText(llama.AccelerationReason, maxStatusReasonLen)
	llama.OptimizationCapabilities = normalizeOptimizationCapabilities(llama.OptimizationCapabilities, llama.Backend, llama.GPUArchitecture)
	llama.LastError = cleanStatusText(llama.LastError, maxStatusReasonLen)
	llama.Launch = normalizeLaunchConfig(llama.Launch)
	llama.ServerProperties = normalizeServerProperties(llama.ServerProperties)
	llama.Acceleration = normalizeAcceleration(llama.Acceleration)
	if llama.MaxContextTokens < 0 {
		llama.MaxContextTokens = 0
	}
	if llama.ModelSizeBytes < 0 {
		llama.ModelSizeBytes = 0
	}
	if !llama.Enabled {
		llama.Available = false
		llama.Running = false
		llama.Healthy = false
	}
	if !llama.Available {
		llama.Healthy = false
	}
	if !llama.Running {
		llama.Healthy = false
	}
	if !llama.Healthy {
		llama.LastHealthAtUnixMs = 0
	}
	llama.SupportsKVAccess = false
	llama.SupportsKVHooks = false
	llama.SupportsTensorHooks = false
	llama.SupportsDistributedKV = false
	loaded := llama.Enabled && llama.Running && llama.Healthy && llama.ModelPath != "" && llama.ModelFilename != ""
	llama.Loaded = loaded
	llama.Warm = loaded
	if !loaded {
		llama.Warm = false
		llama.WarmModelID = ""
	} else if llama.WarmModelID == "" {
		llama.WarmModelID = llama.ModelID
	}
	llama.Health = normalizeRuntimeHealth(llama)
	return llama
}

func normalizeLaunchConfig(launch *LlamaCppLaunchConfig) *LlamaCppLaunchConfig {
	if launch == nil {
		return nil
	}
	out := *launch
	out.Mode = cleanRuntimeCompactText(strings.ToLower(strings.TrimSpace(out.Mode)), 32)
	switch out.Mode {
	case "disabled", "managed", "attached", "stopped":
	default:
		out.Mode = "stopped"
	}
	out.ServerPath = cleanRuntimePath(out.ServerPath)
	out.ServerFilename = cleanRuntimePath(out.ServerFilename)
	if out.ServerFilename == "" && out.ServerPath != "" {
		out.ServerFilename = cleanRuntimePath(filepath.Base(out.ServerPath))
	}
	if out.ConfiguredGPULayers < 0 {
		out.ConfiguredGPULayers = 0
	}
	if out.ConfiguredDraftGPULayers < 0 {
		out.ConfiguredDraftGPULayers = 0
	}
	out.Profile = normalizeLaunchProfile(out.Profile)
	if out.Profile == "" {
		out.Profile = LaunchProfileDefault
	}
	if out.Mode == "managed" {
		out.Managed = true
		out.Attached = false
	} else if out.Mode == "attached" {
		out.Managed = false
		out.Attached = true
	} else {
		out.Managed = false
		out.Attached = false
	}
	return &out
}

func cloneLaunchConfig(launch *LlamaCppLaunchConfig) *LlamaCppLaunchConfig {
	return normalizeLaunchConfig(launch)
}

func normalizeServerProperties(props *LlamaCppServerProperties) *LlamaCppServerProperties {
	if props == nil {
		return nil
	}
	out := *props
	out.BuildInfo = cleanRuntimeCompactText(out.BuildInfo, 256)
	out.SystemInfo = cleanRuntimeCompactText(out.SystemInfo, 512)
	if out.ReportedGPULayers < 0 {
		out.ReportedGPULayers = 0
	}
	out.ReportedAcceleration = normalizeAcceleration(out.ReportedAcceleration)
	if out.BuildInfo == "" && out.SystemInfo == "" && out.ReportedGPULayers == 0 && len(out.ReportedAcceleration) == 0 {
		return nil
	}
	return &out
}

func cloneServerProperties(props *LlamaCppServerProperties) *LlamaCppServerProperties {
	return normalizeServerProperties(props)
}

func gpuOffloadRequestedButServerReportsCPUOnly(cfg LlamaCppSidecarConfig, props *LlamaCppServerProperties) bool {
	if cfg.GPULayers <= 0 || props == nil {
		return false
	}
	normalized := normalizeServerProperties(props)
	if normalized == nil || len(normalized.ReportedAcceleration) != 1 {
		return false
	}
	return normalized.ReportedAcceleration[0] == "cpu"
}

func sidecarAccelerationStatus(cfg LlamaCppSidecarConfig, props *LlamaCppServerProperties) ([]string, string) {
	cfg = normalizeConfig(cfg)
	normalizedProps := normalizeServerProperties(props)
	if normalizedProps != nil && len(normalizedProps.ReportedAcceleration) > 0 {
		acceleration := normalizedProps.ReportedAcceleration
		switch {
		case accelerationContains(acceleration, "cuda"):
			return acceleration, "server_reported_cuda"
		case accelerationContains(acceleration, "vulkan"):
			return acceleration, "server_reported_vulkan"
		case len(acceleration) == 1 && acceleration[0] == "cpu" && cfg.GPULayers > 0:
			return acceleration, "server_reported_cpu_after_gpu_offload_requested"
		case len(acceleration) == 1 && acceleration[0] == "cpu":
			return acceleration, "server_reported_cpu"
		default:
			return acceleration, "server_reported_acceleration"
		}
	}
	if cfg.GPULayers > 0 && accelerationContains(cfg.AccelerationHints, "cuda") {
		return []string{"cuda"}, "configured_cuda_unconfirmed"
	}
	if cfg.GPULayers > 0 && accelerationContains(cfg.AccelerationHints, "vulkan") {
		return []string{"vulkan"}, "configured_vulkan_unconfirmed"
	}
	return []string{"cpu"}, "server_acceleration_unreported"
}

func statusAccelerationOrDefault(status LlamaCppSidecarStatus) []string {
	acceleration := normalizeAcceleration(status.Acceleration)
	if len(acceleration) > 0 {
		return acceleration
	}
	if status.ServerProperties != nil && len(status.ServerProperties.ReportedAcceleration) > 0 {
		return status.ServerProperties.ReportedAcceleration
	}
	return []string{"cpu"}
}

func statusAccelerationReasonOrDefault(status LlamaCppSidecarStatus) string {
	if strings.TrimSpace(status.AccelerationReason) != "" {
		return status.AccelerationReason
	}
	cfg := LlamaCppSidecarConfig{}
	if status.Launch != nil {
		cfg.GPULayers = status.Launch.ConfiguredGPULayers
	}
	cfg.AccelerationHints = statusAccelerationOrDefault(status)
	_, reason := sidecarAccelerationStatus(cfg, status.ServerProperties)
	return reason
}

func accelerationContains(values []string, want string) bool {
	want = strings.ToLower(strings.TrimSpace(want))
	if want == "" {
		return false
	}
	for _, value := range normalizeAcceleration(values) {
		if value == want {
			return true
		}
	}
	return false
}

func sidecarAccelerationReasonSuffix(reason string) string {
	switch strings.TrimSpace(reason) {
	case "server_reported_cuda":
		return "CUDA acceleration reported active"
	case "server_reported_vulkan":
		return "Vulkan acceleration reported active"
	case "server_reported_cpu_after_gpu_offload_requested":
		return "GPU offload requested but server reports CPU-only acceleration"
	case "server_reported_cpu":
		return "CPU acceleration reported active"
	case "configured_cuda_unconfirmed":
		return "CUDA acceleration requested; server report not available"
	case "configured_vulkan_unconfirmed":
		return "Vulkan acceleration requested; server report not available"
	default:
		return ""
	}
}

func normalizeOtherBackendRuntimes(runtimes []BackendRuntimeStatus) []BackendRuntimeStatus {
	if len(runtimes) == 0 {
		return []BackendRuntimeStatus{}
	}
	out := make([]BackendRuntimeStatus, 0, min(len(runtimes), 8))
	for _, runtime := range runtimes {
		if len(out) >= 8 {
			break
		}
		runtime = normalizeBackendRuntimeStatus(runtime, runtime.Backend)
		if runtime.Backend == "" || runtime.Backend == BackendName ||
			runtime.Backend == runtimeinventory.BackendCandidateTensorRTLLM ||
			runtime.Backend == runtimeinventory.BackendCandidateVLLM ||
			runtime.Backend == runtimeinventory.BackendCandidateSGLang {
			continue
		}
		if !runtime.Available {
			continue
		}
		out = append(out, runtime)
	}
	if out == nil {
		return []BackendRuntimeStatus{}
	}
	return out
}

func runtimeFromBackendCandidate(candidate runtimeinventory.BackendCandidate, hardware capshardware.CapacityInventory) BackendRuntimeStatus {
	acceleration := []string{}
	if candidate.Detected {
		acceleration = accelerationFromBackend(candidate.Backend, hardware)
	}
	return BackendRuntimeStatus{
		Enabled:                  candidate.Detected,
		Available:                candidate.Detected,
		Running:                  false,
		Healthy:                  false,
		Backend:                  candidate.Backend,
		BaseURL:                  "",
		Acceleration:             acceleration,
		GPUArchitecture:          gpuArchitectureFromHardware(hardware),
		GPUComputeCapability:     hardware.ComputeCapability,
		OptimizationCapabilities: optimizationCapabilitiesForBackend(candidate.Backend, hardware),
		OpenAICompatible:         candidate.SupportsOpenAICompatibleServer,
		SupportsTextGeneration:   candidate.SupportsTextGeneration,
		SupportsStreaming:        candidate.SupportsStreaming,
		SupportsStatefulSessions: false,
		SupportsKVAccess:         false,
		SupportsKVHooks:          false,
		SupportsTensorHooks:      false,
		SupportsDistributedKV:    false,
		LastError:                "",
	}
}

func mergeDetectedRuntime(active BackendRuntimeStatus, detected BackendRuntimeStatus) BackendRuntimeStatus {
	if active.Backend == "" {
		active.Backend = detected.Backend
	}
	active.Enabled = active.Enabled || detected.Enabled
	active.Available = active.Available || detected.Available
	active.Acceleration = mergeAcceleration(active.Acceleration, detected.Acceleration)
	active.GPUArchitecture = firstNonEmptyRuntimeText(active.GPUArchitecture, detected.GPUArchitecture)
	active.GPUComputeCapability = firstNonEmptyRuntimeText(active.GPUComputeCapability, detected.GPUComputeCapability)
	active.OptimizationCapabilities = normalizeOptimizationCapabilities(append(active.OptimizationCapabilities, detected.OptimizationCapabilities...), active.Backend, active.GPUArchitecture)
	active.OpenAICompatible = active.OpenAICompatible || detected.OpenAICompatible
	active.SupportsTextGeneration = active.SupportsTextGeneration || detected.SupportsTextGeneration
	active.SupportsStreaming = active.SupportsStreaming || detected.SupportsStreaming
	return active
}

func accelerationFromBackend(backend string, hardware capshardware.CapacityInventory) []string {
	switch backend {
	case runtimeinventory.BackendCandidateTensorRTLLM, runtimeinventory.BackendCandidateVLLM, runtimeinventory.BackendCandidateSGLang:
		acceleration := accelerationFromHardware(hardware)
		if len(acceleration) > 0 {
			return acceleration
		}
		return []string{"cpu"}
	default:
		return accelerationFromHardware(hardware)
	}
}

func accelerationFromHardware(hardware capshardware.CapacityInventory) []string {
	hardware = capshardware.NormalizeInventory(hardware)
	if len(hardware.AccelerationHints) > 0 {
		return hardware.AccelerationHints
	}
	acceleration := []string{}
	if hardware.CPULogicalCores > 0 {
		acceleration = append(acceleration, "cpu")
	}
	if hardware.CUDAAvailable {
		acceleration = append(acceleration, "cuda")
	}
	if hardware.VulkanAvailable {
		acceleration = append(acceleration, "vulkan")
	}
	if hardware.DirectMLAvailable {
		acceleration = append(acceleration, "directml")
	}
	if hardware.MetalAvailable {
		acceleration = append(acceleration, "metal")
	}
	if len(acceleration) == 0 {
		return []string{}
	}
	return acceleration
}

func mergeAcceleration(left, right []string) []string {
	return normalizeAcceleration(append(cloneStrings(left), right...))
}

func normalizeAcceleration(values []string) []string {
	if len(values) == 0 {
		return []string{}
	}
	seen := map[string]struct{}{}
	out := make([]string, 0, min(len(values), 8))
	for _, value := range values {
		if len(out) >= 8 {
			break
		}
		value = cleanRuntimeCompactText(strings.ToLower(strings.TrimSpace(value)), 32)
		switch value {
		case "cpu", "cuda", "vulkan", "directml", "metal", "rocm", "other":
		default:
			value = "other"
		}
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	if out == nil {
		return []string{}
	}
	return out
}

func normalizeOptimizationCapabilities(capabilities []OptimizationCapability, backend string, gpuArchitecture string) []OptimizationCapability {
	if len(capabilities) == 0 {
		return []OptimizationCapability{}
	}
	backend = cleanRuntimeCompactText(firstNonEmptyRuntimeText(backend, BackendName), 64)
	gpuArchitecture = cleanRuntimeCompactText(gpuArchitecture, 64)
	seen := map[string]struct{}{}
	out := make([]OptimizationCapability, 0, min(len(capabilities), 16))
	for _, capability := range capabilities {
		capability.Name = normalizeOptimizationName(capability.Name)
		if capability.Name == "" {
			continue
		}
		capability.Backend = cleanRuntimeCompactText(firstNonEmptyRuntimeText(capability.Backend, backend), 64)
		capability.RequiresAttention = cleanRuntimeCompactText(capability.RequiresAttention, 64)
		capability.RequiresGPUArch = cleanRuntimeCompactText(capability.RequiresGPUArch, 64)
		capability.Notes = cleanRuntimeCompactText(capability.Notes, 256)
		if capability.ContextMinTokens < 0 {
			capability.ContextMinTokens = 0
		}
		if capability.Name == "gvr_topk" && !gpuArchitectureSatisfiesRequirement(gpuArchitecture, capability.RequiresGPUArch) {
			capability.Supported = false
			capability.Enabled = false
		}
		key := capability.Name + "\x00" + capability.Backend + "\x00" + capability.RequiresAttention + "\x00" + capability.RequiresGPUArch
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, capability)
		if len(out) >= 16 {
			break
		}
	}
	if out == nil {
		return []OptimizationCapability{}
	}
	return out
}

func optimizationCapabilitiesForBackend(backend string, hardware capshardware.CapacityInventory) []OptimizationCapability {
	hardware = capshardware.NormalizeInventory(hardware)
	gpuArchitecture := gpuArchitectureFromHardware(hardware)
	switch backend {
	case runtimeinventory.BackendCandidateTensorRTLLM:
		supported := gpuArchitectureSatisfiesRequirement(gpuArchitecture, "blackwell_sm100_plus")
		return normalizeOptimizationCapabilities([]OptimizationCapability{{
			Name:              "gvr_topk",
			Supported:         supported,
			Enabled:           supported,
			Backend:           runtimeinventory.BackendCandidateTensorRTLLM,
			RequiresAttention: "deepseek_sparse_attention",
			RequiresGPUArch:   "blackwell_sm100_plus",
			ContextMinTokens:  16384,
			Notes:             "Optional optimization; not required for general inference.",
		}}, backend, gpuArchitecture)
	default:
		return []OptimizationCapability{}
	}
}

func normalizeOptimizationName(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	value = strings.ReplaceAll(value, "-", "_")
	value = strings.ReplaceAll(value, " ", "_")
	switch value {
	case "gvr_topk",
		"speculative_decode",
		"ngram",
		"draft_model",
		"draft_target",
		"native_mtp",
		"eagle",
		"eagle3",
		"medusa",
		"redrafter",
		"flash_attention",
		"paged_attention",
		"sparse_attention",
		"kv_cache_quantization":
		return value
	default:
		return ""
	}
}

func gpuArchitectureFromHardware(hardware capshardware.CapacityInventory) string {
	hardware = capshardware.NormalizeInventory(hardware)
	switch hardware.GPUVendor {
	case capshardware.GPUVendorApple:
		if hardware.MetalAvailable {
			return "apple_metal"
		}
	case capshardware.GPUVendorNVIDIA:
		return nvidiaArchitectureFromComputeCapability(hardware.ComputeCapability)
	case capshardware.GPUVendorAMD:
		if hardware.VulkanAvailable {
			return "amd_vulkan"
		}
	}
	return ""
}

func nvidiaArchitectureFromComputeCapability(computeCapability string) string {
	computeCapability = strings.TrimSpace(computeCapability)
	major, minor := parseComputeCapability(computeCapability)
	switch {
	case major >= 10:
		return "blackwell_sm100_plus"
	case major == 9:
		return "hopper_sm90"
	case major == 8 && minor >= 9:
		return "ada_sm89"
	case major == 8:
		return "ampere_sm80"
	case major == 7:
		return "turing_sm75"
	default:
		return ""
	}
}

func gpuArchitectureSatisfiesRequirement(gpuArchitecture string, required string) bool {
	gpuArchitecture = strings.ToLower(strings.TrimSpace(gpuArchitecture))
	required = strings.ToLower(strings.TrimSpace(required))
	if required == "" {
		return true
	}
	if gpuArchitecture == required {
		return true
	}
	return required == "blackwell_sm100_plus" && gpuArchitecture == "blackwell_sm100_plus"
}

func parseComputeCapability(computeCapability string) (int, int) {
	computeCapability = strings.TrimSpace(computeCapability)
	if computeCapability == "" {
		return 0, 0
	}
	parts := strings.SplitN(computeCapability, ".", 3)
	major, _ := strconv.Atoi(parts[0])
	minor := 0
	if len(parts) > 1 {
		minor, _ = strconv.Atoi(parts[1])
	}
	return major, minor
}

func normalizeRuntimeHealth(runtime BackendRuntimeStatus) string {
	switch {
	case !runtime.Enabled:
		return "disabled"
	case runtime.Running && runtime.Healthy:
		return "healthy"
	case runtime.Running:
		return "degraded"
	case runtime.Available:
		return "available"
	default:
		return "unavailable"
	}
}

func cloneStrings(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	out := make([]string, len(values))
	copy(out, values)
	return out
}

func buildServerArgs(cfg LlamaCppSidecarConfig) []string {
	cfg = normalizeConfig(cfg)
	args := []string{
		"--host", cfg.Host,
		"--port", strconv.Itoa(cfg.Port),
		"--model", cfg.ModelPath,
		"--ctx-size", strconv.Itoa(cfg.ContextSize),
	}
	if cfg.Threads > 0 {
		args = append(args, "--threads", strconv.Itoa(cfg.Threads))
	}
	if cfg.GPULayers > 0 {
		args = append(args, "--n-gpu-layers", strconv.Itoa(cfg.GPULayers))
	}
	extraArgs := cfg.ExtraArgs
	if cfg.FastDefaults && cfg.GPULayers > 0 {
		args = appendGPUFastDefaults(args, extraArgs)
	}
	if jinjaTemplateRequiredForModel(cfg.ModelPath) && !hasArgFlag(extraArgs, "--jinja") {
		args = append(args, "--jinja")
	}
	if reasoningFormat := reasoningFormatForModel(cfg.ModelPath); reasoningFormat != "" {
		if !hasArgFlag(extraArgs, "--reasoning-format") {
			args = append(args, "--reasoning-format", reasoningFormat)
		}
	}
	// V8 speculative decoding (Level 0).
	// Native MTP uses the target model's own MTP heads. The older
	// --model-draft path remains the default when only DraftModelPath is set.
	if nativeMTPUsable(cfg) {
		if !hasArgFlag(extraArgs, "--spec-type") {
			args = append(args, "--spec-type", "draft-mtp")
		}
		if cfg.DraftMaxTokens > 0 && !hasArgFlag(extraArgs, "--spec-draft-n-max") {
			args = append(args, "--spec-draft-n-max", strconv.Itoa(cfg.DraftMaxTokens))
		}
		if cfg.DraftMinTokens > 0 && !hasArgFlag(extraArgs, "--spec-draft-n-min") {
			args = append(args, "--spec-draft-n-min", strconv.Itoa(cfg.DraftMinTokens))
		}
	} else if cfg.DraftModelPath != "" && draftModelReadable(cfg.DraftModelPath) {
		args = append(args, "--model-draft", cfg.DraftModelPath)
		if cfg.DraftMaxTokens > 0 {
			args = append(args, "--spec-draft-n-max", strconv.Itoa(cfg.DraftMaxTokens))
		}
		if cfg.DraftMinTokens > 0 {
			args = append(args, "--spec-draft-n-min", strconv.Itoa(cfg.DraftMinTokens))
		}
		if cfg.DraftPMin > 0 {
			args = append(args, "--draft-p-min", strconv.FormatFloat(cfg.DraftPMin, 'f', 3, 64))
		}
		if cfg.DraftGPULayers > 0 {
			args = append(args, "--n-gpu-layers-draft", strconv.Itoa(cfg.DraftGPULayers))
		}
	} else if draftlessSpecType(cfg.SpecType) && !hasArgFlag(extraArgs, "--spec-type") {
		args = append(args, "--spec-type", cfg.SpecType)
		if cfg.DraftMaxTokens > 0 {
			args = append(args, "--spec-draft-n-max", strconv.Itoa(cfg.DraftMaxTokens))
		}
		if cfg.DraftMinTokens > 0 {
			args = append(args, "--spec-draft-n-min", strconv.Itoa(cfg.DraftMinTokens))
		}
	}
	args = append(args, extraArgs...)
	return args
}

func jinjaTemplateRequiredForModel(modelPath string) bool {
	meta := modelMetadata(modelPath)
	switch meta.familyHint {
	case "qwen", "deepseek", "gpt-oss":
		return true
	default:
		return false
	}
}

func reasoningFormatForModel(modelPath string) string {
	meta := modelMetadata(modelPath)
	switch meta.familyHint {
	case "qwen", "deepseek":
		return "deepseek"
	default:
		return ""
	}
}

func appendGPUFastDefaults(args []string, extraArgs []string) []string {
	return appendGPUFastDefaultsForGOOS(args, extraArgs, runtime.GOOS)
}

func appendGPUFastDefaultsForGOOS(args []string, extraArgs []string, goos string) []string {
	defaults := [][]string{
		{"--batch-size", "512"},
		{"--ubatch-size", "512"},
	}
	if !windowsGOOS(goos) {
		defaults = append([][]string{{"--flash-attn"}}, defaults...)
		defaults = append(defaults,
			[]string{"--cache-type-k", "q8_0"},
			[]string{"--cache-type-v", "q8_0"},
		)
	}
	for _, flag := range defaults {
		if len(flag) == 0 || hasArgFlag(extraArgs, flag[0]) {
			continue
		}
		args = append(args, flag...)
	}
	return args
}

func windowsGOOS(goos string) bool {
	return strings.EqualFold(strings.TrimSpace(goos), "windows")
}

func hasArgFlag(args []string, flag string) bool {
	flag = strings.TrimSpace(flag)
	if flag == "" {
		return false
	}
	for _, arg := range args {
		if strings.TrimSpace(arg) == flag {
			return true
		}
	}
	return false
}

// draftModelReadable verifies the draft GGUF exists before we hand the
// path to llama-server. A missing draft file would crash the whole
// sidecar instead of degrading to non-speculative mode.
func draftModelReadable(path string) bool {
	if strings.TrimSpace(path) == "" {
		return false
	}
	info, err := os.Stat(path)
	if err != nil || info.IsDir() {
		return false
	}
	return info.Size() > 0
}

func serverPathAvailable(path string) bool {
	path = strings.TrimSpace(path)
	if path == "" {
		return false
	}
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func freeLocalPort(host string) (int, bool) {
	return freeLocalPortExcept(host, 0)
}

func freeLocalPortExcept(host string, excludedPort int) (int, bool) {
	host = normalizeHost(host)
	for i := 0; i < 8; i++ {
		listener, err := net.Listen("tcp", net.JoinHostPort(host, "0"))
		if err != nil {
			return 0, false
		}
		addr, ok := listener.Addr().(*net.TCPAddr)
		port := 0
		if ok {
			port = addr.Port
		}
		_ = listener.Close()
		if port <= 0 || port > 65535 {
			continue
		}
		if excludedPort > 0 && port == excludedPort {
			continue
		}
		return port, true
	}
	return 0, false
}

type modelMeta struct {
	filename         string
	sizeBytes        int64
	familyHint       string
	quantizationHint string
	readable         bool
}

func modelMetadata(path string) modelMeta {
	path = strings.TrimSpace(path)
	filename := filepath.Base(path)
	if path == "" || filename == "." || filename == string(filepath.Separator) {
		return modelMeta{}
	}
	meta := modelMeta{
		filename:         filename,
		familyHint:       inferModelFamily(filename),
		quantizationHint: inferQuantization(filename),
	}
	if !strings.EqualFold(filepath.Ext(filename), ".gguf") {
		return meta
	}
	info, err := os.Stat(path)
	if err != nil || info.IsDir() {
		return meta
	}
	meta.readable = true
	meta.sizeBytes = info.Size()
	if meta.sizeBytes < 0 {
		meta.sizeBytes = 0
	}
	return meta
}

func inferModelFamily(filename string) string {
	lower := strings.ToLower(filename)
	switch {
	case strings.Contains(lower, "llama"):
		return "llama"
	case strings.Contains(lower, "phi"):
		return "phi"
	case strings.Contains(lower, "qwen"):
		return "qwen"
	case strings.Contains(lower, "gpt-oss"), strings.Contains(lower, "gptoss"):
		return "gpt-oss"
	case strings.Contains(lower, "gemma"):
		return "gemma"
	default:
		return "unknown"
	}
}

func inferQuantization(filename string) string {
	matches := ggufQuantizationPattern.FindStringSubmatch(filename)
	if len(matches) < 2 {
		return "unknown"
	}
	return strings.ToUpper(matches[1])
}

func baseURL(host string, port int) string {
	host = normalizeHost(host)
	if port <= 0 || port > 65535 {
		port = DefaultPort
	}
	if strings.Contains(host, ":") && !strings.HasPrefix(host, "[") {
		host = "[" + host + "]"
	}
	return fmt.Sprintf("http://%s:%d", host, port)
}

func isLocalBaseURL(raw string) bool {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil || parsed.Scheme != "http" {
		return false
	}
	host := parsed.Hostname()
	if host == "" {
		return false
	}
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func cleanStatusText(value string, maxRunes int) string {
	value = strings.Join(strings.Fields(value), " ")
	value = cleanConfigText(value, maxRunes)
	if value == "" {
		return ""
	}
	lower := strings.ToLower(value)
	for _, marker := range []string{"raw_prompt", "prompt_text", "generated_text", "output_text", "model_output", "tensor_bytes", "raw_tensor", "auth_token", "bind_token", "secret"} {
		if strings.Contains(lower, marker) {
			return "llama.cpp sidecar status redacted"
		}
	}
	return value
}

func cleanRuntimePath(value string) string {
	value = cleanConfigText(value, maxRuntimePathLen)
	if sensitiveStatusText(value) {
		return ""
	}
	return value
}

func cleanRuntimeBaseURL(value string) string {
	value = cleanConfigText(value, maxRuntimePathLen)
	if value == "" || sensitiveStatusText(value) || !isLocalBaseURL(value) {
		return ""
	}
	return value
}

func cleanRuntimeCompactText(value string, maxRunes int) string {
	value = cleanConfigText(value, maxRunes)
	if sensitiveStatusText(value) {
		return ""
	}
	return value
}

func firstNonEmptyRuntimeText(values ...string) string {
	for _, value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			return trimmed
		}
	}
	return ""
}

func unixMilliOrZero(t time.Time) int64 {
	if t.IsZero() {
		return 0
	}
	return t.UnixMilli()
}

func sensitiveStatusText(value string) bool {
	lower := strings.ToLower(value)
	for _, marker := range []string{"raw_prompt", "prompt_text", "generated_text", "output_text", "model_output", "tensor_bytes", "raw_tensor", "auth_token", "bind_token", "secret"} {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

type osManagedProcess struct {
	cmd *exec.Cmd
}

func (p *osManagedProcess) PID() int {
	if p == nil || p.cmd == nil || p.cmd.Process == nil {
		return 0
	}
	return p.cmd.Process.Pid
}

func (p *osManagedProcess) Wait() error {
	if p == nil || p.cmd == nil {
		return nil
	}
	return p.cmd.Wait()
}

func (p *osManagedProcess) Kill() error {
	if p == nil || p.cmd == nil || p.cmd.Process == nil {
		return nil
	}
	return p.cmd.Process.Kill()
}

func defaultProcessStarter(ctx context.Context, binary string, args []string, output io.Writer) (managedProcess, error) {
	binary = strings.TrimSpace(binary)
	if binary == "" {
		return nil, os.ErrNotExist
	}
	cmd := exec.CommandContext(ctx, binary, args...)
	configureProcessCommand(cmd, binary)
	if output != nil {
		cmd.Stdout = output
		cmd.Stderr = output
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return &osManagedProcess{cmd: cmd}, nil
}

func configureProcessCommand(cmd *exec.Cmd, binary string) {
	if cmd == nil {
		return
	}
	binDir := filepath.Dir(strings.TrimSpace(binary))
	if binDir == "" || binDir == "." {
		return
	}
	cmd.Dir = binDir
	env := os.Environ()
	separator := string(os.PathListSeparator)
	env = prependEnvSearchPath(env, "PATH", binDir, separator)
	if runtime.GOOS != "windows" {
		env = prependEnvSearchPath(env, "LD_LIBRARY_PATH", binDir, separator)
		env = prependEnvSearchPath(env, "DYLD_LIBRARY_PATH", binDir, separator)
	}
	cmd.Env = env
}

func prependEnvSearchPath(env []string, key string, pathValue string, separator string) []string {
	key = strings.TrimSpace(key)
	pathValue = strings.TrimSpace(pathValue)
	if key == "" || pathValue == "" {
		return append([]string(nil), env...)
	}
	if separator == "" {
		separator = string(os.PathListSeparator)
	}
	out := make([]string, 0, len(env)+1)
	found := false
	for _, item := range env {
		name, value, ok := strings.Cut(item, "=")
		if ok && strings.EqualFold(name, key) {
			found = true
			if envSearchPathContains(value, pathValue, separator) {
				out = append(out, item)
			} else if strings.TrimSpace(value) == "" {
				out = append(out, name+"="+pathValue)
			} else {
				out = append(out, name+"="+pathValue+separator+value)
			}
			continue
		}
		out = append(out, item)
	}
	if !found {
		out = append(out, key+"="+pathValue)
	}
	return out
}

func envSearchPathContains(value string, pathValue string, separator string) bool {
	for _, part := range strings.Split(value, separator) {
		if strings.EqualFold(strings.TrimSpace(part), pathValue) {
			return true
		}
	}
	return false
}

type boundedLogBuffer struct {
	mu    sync.Mutex
	limit int
	data  []byte
}

func newBoundedLogBuffer(limit int) *boundedLogBuffer {
	if limit <= 0 {
		limit = maxLogBufferBytes
	}
	return &boundedLogBuffer{limit: limit}
}

func (b *boundedLogBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.data = append(b.data, p...)
	if len(b.data) > b.limit {
		b.data = append([]byte(nil), b.data[len(b.data)-b.limit:]...)
	}
	return len(p), nil
}

func (b *boundedLogBuffer) TailString(limit int) string {
	if b == nil {
		return ""
	}
	if limit <= 0 {
		limit = 512
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if len(b.data) == 0 {
		return ""
	}
	start := 0
	if len(b.data) > limit {
		start = len(b.data) - limit
	}
	return string(append([]byte(nil), b.data[start:]...))
}
