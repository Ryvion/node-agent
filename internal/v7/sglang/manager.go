package sglang

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
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	v7hardware "github.com/Ryvion/ryvion-node/internal/v7/hardware"
	v7llamacpp "github.com/Ryvion/ryvion-node/internal/v7/llamacpp"
)

const (
	maxStatusReasonLen = 256
	maxLogBufferBytes  = 64 * 1024
	maxRuntimePathLen  = 512
)

type managedProcess interface {
	PID() int
	Wait() error
	Kill() error
}

type processStarter func(ctx context.Context, binary string, args []string, output io.Writer) (managedProcess, error)

type ManagerOption func(*Manager)

type Manager struct {
	mu            sync.Mutex
	cfg           SGLangSidecarConfig
	client        healthHTTPClient
	starter       processStarter
	logs          *boundedLogBuffer
	process       managedProcess
	processCancel context.CancelFunc
	startedAt     time.Time
	lastHealthAt  time.Time
	lastError     string
	attached      bool
	stopping      bool
	now           func() time.Time
	healthTimeout time.Duration
}

func NewManager(cfg SGLangSidecarConfig, opts ...ManagerOption) *Manager {
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

func (m *Manager) Config() SGLangSidecarConfig {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.cfg
}

func (m *Manager) Status(ctx context.Context) SGLangSidecarStatus {
	if m == nil {
		return NewManager(SGLangSidecarConfig{}).Status(ctx)
	}
	m.mu.Lock()
	status := m.statusLocked()
	shouldHealthCheck := status.Enabled && status.Available
	m.mu.Unlock()

	if shouldHealthCheck {
		health := CheckHealth(ctx, status.BaseURL, m.client, m.healthTimeout)
		m.mu.Lock()
		m.applyHealthLocked(health)
		status = m.statusLocked()
		m.mu.Unlock()
	}
	return status
}

func (m *Manager) Start(ctx context.Context) SGLangSidecarStatus {
	if m == nil {
		return NewManager(SGLangSidecarConfig{}).Start(ctx)
	}
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

func (m *Manager) Stop(ctx context.Context) SGLangSidecarStatus {
	if m == nil {
		return NewManager(SGLangSidecarConfig{}).Stop(ctx)
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
	_ = ctx
	return status
}

func (m *Manager) Restart(ctx context.Context) SGLangSidecarStatus {
	if m == nil {
		return NewManager(SGLangSidecarConfig{}).Restart(ctx)
	}
	_ = m.Stop(ctx)
	return m.Start(ctx)
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
	}
	m.lastError = cleanStatusText(health.Error, maxStatusReasonLen)
}

func (m *Manager) statusLocked() SGLangSidecarStatus {
	cfg := normalizeConfig(m.cfg)
	m.cfg = cfg
	meta := modelMetadata(cfg.ModelPath, cfg.ModelID)
	serverOK := serverPathAvailable(cfg.ServerPath)
	modelOK := meta.readable
	running := m.process != nil || m.attached
	healthy := running && m.lastError == "" && !m.lastHealthAt.IsZero()
	pid := 0
	if m.process != nil {
		pid = m.process.PID()
	}
	launch := buildLaunchConfig(cfg, m.process != nil, m.attached && m.process == nil)

	reason := "SGLang sidecar ready"
	if !cfg.Enabled {
		reason = "SGLang sidecar disabled"
	} else if !serverOK {
		reason = "SGLang launcher not detected"
	} else if cfg.ModelPath == "" {
		reason = "SGLang model path not configured"
	} else if !modelOK {
		reason = "SGLang model path is not readable"
	} else if running && healthy {
		if m.attached && m.process == nil {
			reason = "attached to healthy SGLang server"
		} else {
			reason = "managed SGLang sidecar healthy"
		}
	} else if running {
		reason = "SGLang sidecar running; health not confirmed"
	} else if m.lastError != "" {
		reason = m.lastError
	}
	acceleration, accelerationReason := sidecarAccelerationStatus(cfg)
	return SGLangSidecarStatus{
		Enabled:                  cfg.Enabled,
		Available:                serverOK && modelOK,
		Running:                  running,
		Healthy:                  healthy,
		Attached:                 m.attached && m.process == nil,
		PID:                      pid,
		BaseURL:                  baseURL(cfg.Host, cfg.Port),
		ServerPath:               cfg.ServerPath,
		ModelPath:                cfg.ModelPath,
		ModelID:                  meta.modelID,
		ContextLength:            cfg.ContextLength,
		TPSize:                   cfg.TPSize,
		MemFractionStatic:        cfg.MemFractionStatic,
		ModelFilename:            meta.filename,
		ModelSizeBytes:           meta.sizeBytes,
		StartedAt:                m.startedAt,
		LastHealthAt:             m.lastHealthAt,
		LastError:                cleanStatusText(m.lastError, maxStatusReasonLen),
		Backend:                  BackendName,
		Launch:                   launch,
		Acceleration:             acceleration,
		AccelerationReason:       accelerationReason,
		OpenAICompatible:         true,
		SupportsTextGeneration:   true,
		SupportsStreaming:        true,
		SupportsStatefulSessions: true,
		SupportsKVAccess:         false,
		SupportsTensorHooks:      false,
		Reason:                   cleanStatusText(reason, maxStatusReasonLen),
	}
}

func BuildBackendRuntime(status SGLangSidecarStatus, hardware v7hardware.CapacityInventory) v7llamacpp.BackendRuntimeStatus {
	hardware = v7hardware.NormalizeInventory(hardware)
	status.Backend = firstNonEmptyRuntimeText(status.Backend, BackendName)
	modelFilename := firstNonEmptyRuntimeText(status.ModelFilename, modelIDFromPath(status.ModelPath))
	runtimeStatus := v7llamacpp.BackendRuntimeStatus{
		Enabled:                  status.Enabled,
		Available:                status.Available,
		Running:                  status.Running,
		Healthy:                  status.Healthy,
		Backend:                  BackendName,
		BaseURL:                  status.BaseURL,
		ModelID:                  firstNonEmptyRuntimeText(status.ModelID, modelFilename),
		ModelPath:                status.ModelPath,
		ModelFilename:            modelFilename,
		ModelSizeBytes:           status.ModelSizeBytes,
		OpenAICompatible:         status.OpenAICompatible,
		SupportsTextGeneration:   status.SupportsTextGeneration,
		SupportsStreaming:        status.SupportsStreaming,
		SupportsStatefulSessions: status.SupportsStatefulSessions,
		SupportsKVAccess:         false,
		SupportsKVHooks:          false,
		SupportsTensorHooks:      false,
		SupportsDistributedKV:    false,
		MaxContextTokens:         status.ContextLength,
		LastHealthAtUnixMs:       unixMilliOrZero(status.LastHealthAt),
		LastError:                status.LastError,
		Acceleration:             mergeAcceleration(status.Acceleration, accelerationFromHardware(hardware)),
		AccelerationReason:       firstNonEmptyRuntimeText(status.AccelerationReason, "configured_acceleration_unconfirmed"),
		GPUArchitecture:          gpuArchitectureFromHardware(hardware),
		GPUComputeCapability:     hardware.ComputeCapability,
		Launch:                   mapLaunchConfig(status.Launch),
	}
	return v7llamacpp.NormalizeBackendRuntimes(v7llamacpp.BackendRuntimes{SGLang: runtimeStatus}).SGLang
}

func buildServerArgs(cfg SGLangSidecarConfig) []string {
	cfg = normalizeConfig(cfg)
	args := []string{}
	if launcherKind(cfg.ServerPath) == "python_module" {
		args = append(args, "-m", "sglang.launch_server")
	} else if launcherKind(cfg.ServerPath) == "sglang_binary" {
		args = append(args, "launch_server")
	}
	args = append(args,
		"--model-path", cfg.ModelPath,
		"--host", cfg.Host,
		"--port", strconv.Itoa(cfg.Port),
	)
	if cfg.ContextLength > 0 {
		args = append(args, "--context-length", strconv.Itoa(cfg.ContextLength))
	}
	if cfg.TPSize > 0 {
		args = append(args, "--tp-size", strconv.Itoa(cfg.TPSize))
	}
	if cfg.MemFractionStatic > 0 {
		args = append(args, "--mem-fraction-static", strconv.FormatFloat(cfg.MemFractionStatic, 'f', 3, 64))
	}
	args = append(args, cfg.ExtraArgs...)
	return args
}

func buildLaunchConfig(cfg SGLangSidecarConfig, managed bool, attached bool) *LaunchConfig {
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
	return normalizeLaunchConfig(&LaunchConfig{
		Mode:              mode,
		Managed:           managed,
		Attached:          attached,
		ServerPath:        cfg.ServerPath,
		ServerFilename:    serverFilename,
		Launcher:          launcherKind(cfg.ServerPath),
		ContextLength:     cfg.ContextLength,
		TPSize:            cfg.TPSize,
		MemFractionStatic: cfg.MemFractionStatic,
		Profile:           cfg.LaunchProfile,
	})
}

func normalizeLaunchConfig(launch *LaunchConfig) *LaunchConfig {
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
	out.Launcher = cleanRuntimeCompactText(out.Launcher, 64)
	out.Profile = cleanRuntimeCompactText(out.Profile, 64)
	if out.ContextLength < 0 {
		out.ContextLength = 0
	}
	if out.TPSize < 0 {
		out.TPSize = 0
	}
	if out.MemFractionStatic < 0 || out.MemFractionStatic >= 1 {
		out.MemFractionStatic = 0
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

func mapLaunchConfig(launch *LaunchConfig) *v7llamacpp.LlamaCppLaunchConfig {
	normalized := normalizeLaunchConfig(launch)
	if normalized == nil {
		return nil
	}
	return &v7llamacpp.LlamaCppLaunchConfig{
		Mode:           normalized.Mode,
		Managed:        normalized.Managed,
		Attached:       normalized.Attached,
		ServerPath:     normalized.ServerPath,
		ServerFilename: normalized.ServerFilename,
		Profile:        normalized.Profile,
	}
}

func launcherKind(path string) string {
	base := strings.ToLower(strings.TrimSuffix(filepath.Base(strings.TrimSpace(path)), ".exe"))
	switch base {
	case "python", "python3":
		return "python_module"
	case "sglang":
		return "sglang_binary"
	default:
		if strings.Contains(base, "python") {
			return "python_module"
		}
		return "python_module"
	}
}

func serverPathAvailable(path string) bool {
	path = strings.TrimSpace(path)
	if path == "" {
		return false
	}
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

type modelMeta struct {
	modelID   string
	filename  string
	sizeBytes int64
	readable  bool
}

func modelMetadata(path string, configuredID string) modelMeta {
	path = strings.TrimSpace(path)
	configuredID = cleanConfigText(configuredID, maxConfigTextLen)
	filename := filepath.Base(path)
	if path == "" || filename == "." || filename == string(filepath.Separator) {
		return modelMeta{modelID: configuredID}
	}
	meta := modelMeta{
		modelID:  firstNonEmptyRuntimeText(configuredID, modelIDFromPath(path)),
		filename: filename,
	}
	info, err := os.Stat(path)
	if err != nil {
		return meta
	}
	meta.readable = true
	if !info.IsDir() {
		meta.sizeBytes = info.Size()
		if meta.sizeBytes < 0 {
			meta.sizeBytes = 0
		}
	}
	return meta
}

func modelIDFromPath(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}
	base := filepath.Base(path)
	if base == "." || base == string(filepath.Separator) {
		return ""
	}
	return cleanConfigText(base, maxConfigTextLen)
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

func sidecarAccelerationStatus(cfg SGLangSidecarConfig) ([]string, string) {
	acceleration := normalizeAcceleration(cfg.AccelerationHints)
	if len(acceleration) == 0 {
		return []string{"cpu"}, "configured_acceleration_unconfirmed"
	}
	if accelerationContains(acceleration, "cuda") {
		return acceleration, "configured_cuda_unconfirmed"
	}
	return acceleration, "configured_acceleration_unconfirmed"
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

func accelerationFromHardware(hardware v7hardware.CapacityInventory) []string {
	hardware = v7hardware.NormalizeInventory(hardware)
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

func gpuArchitectureFromHardware(hardware v7hardware.CapacityInventory) string {
	hardware = v7hardware.NormalizeInventory(hardware)
	switch hardware.GPUVendor {
	case v7hardware.GPUVendorApple:
		if hardware.MetalAvailable {
			return "apple_metal"
		}
	case v7hardware.GPUVendorNVIDIA:
		return nvidiaArchitectureFromComputeCapability(hardware.ComputeCapability)
	case v7hardware.GPUVendorAMD:
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

func parseComputeCapability(value string) (int, int) {
	value = strings.TrimSpace(value)
	if value == "" {
		return 0, 0
	}
	parts := strings.SplitN(value, ".", 3)
	major, _ := strconv.Atoi(parts[0])
	minor := 0
	if len(parts) > 1 {
		minor, _ = strconv.Atoi(parts[1])
	}
	return major, minor
}

func cleanStatusText(value string, maxRunes int) string {
	value = strings.Join(strings.Fields(value), " ")
	value = cleanConfigText(value, maxRunes)
	if value == "" {
		return ""
	}
	if strings.Contains(strings.ToLower(value), "sglang sidecar status redacted") {
		return "sglang sidecar status redacted"
	}
	if sensitiveStatusText(value) {
		return "sglang sidecar status redacted"
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

func sensitiveStatusText(value string) bool {
	lower := strings.ToLower(value)
	for _, marker := range []string{"raw_prompt", "prompt_text", "generated_text", "output_text", "model_output", "tensor_bytes", "raw_tensor", "auth_token", "bind_token", "secret"} {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

func unixMilliOrZero(t time.Time) int64 {
	if t.IsZero() {
		return 0
	}
	return t.UnixMilli()
}

func cloneStrings(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	out := make([]string, len(values))
	copy(out, values)
	return out
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
	return cleanStatusText(string(b.data[start:]), maxStatusReasonLen)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
