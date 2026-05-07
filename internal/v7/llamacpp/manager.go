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
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	defaultHealthTimeout = 750 * time.Millisecond
	maxStatusReasonLen   = 256
	maxLogBufferBytes    = 64 * 1024
	maxRuntimePathLen    = 512
)

var ggufQuantizationPattern = regexp.MustCompile(`(?i)(?:^|[._\-\s])((?:IQ|Q)[0-9](?:_[A-Z0-9]+){0,3}|BF16|F16|F32)(?:[._\-\s]|$)`)

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
		m.mu.Lock()
		m.applyHealthLocked(health)
		status = m.statusLocked()
		m.mu.Unlock()
	}
	return status
}

func (m *Manager) Start(ctx context.Context) LlamaCppSidecarStatus {
	if m == nil {
		return NewManager(LlamaCppSidecarConfig{}).Start(ctx)
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
		m.lastError = cleanStatusText(err.Error(), maxStatusReasonLen)
	}
	m.stopping = false
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
		ModelFilename:          meta.filename,
		ModelSizeBytes:         meta.sizeBytes,
		ModelFamilyHint:        meta.familyHint,
		QuantizationHint:       meta.quantizationHint,
		StartedAt:              m.startedAt,
		LastHealthAt:           m.lastHealthAt,
		LastError:              cleanStatusText(m.lastError, maxStatusReasonLen),
		Backend:                BackendName,
		OpenAICompatible:       true,
		SupportsTextGeneration: true,
		SupportsStreaming:      true,
		SupportsKVAccess:       false,
		SupportsTensorHooks:    false,
		Reason:                 cleanStatusText(reason, maxStatusReasonLen),
	}
}

func BuildBackendRuntimes(status LlamaCppSidecarStatus) BackendRuntimes {
	return NormalizeBackendRuntimes(BackendRuntimes{
		LlamaCPP: BackendRuntimeStatus{
			Enabled:                status.Enabled,
			Available:              status.Available,
			Running:                status.Running,
			Healthy:                status.Healthy,
			Backend:                firstNonEmptyRuntimeText(status.Backend, BackendName),
			BaseURL:                status.BaseURL,
			ModelID:                status.ModelFilename,
			ModelPath:              status.ModelPath,
			ModelFilename:          status.ModelFilename,
			ModelSizeBytes:         status.ModelSizeBytes,
			ModelFamilyHint:        status.ModelFamilyHint,
			QuantizationHint:       status.QuantizationHint,
			OpenAICompatible:       status.OpenAICompatible,
			SupportsTextGeneration: status.SupportsTextGeneration,
			SupportsStreaming:      status.SupportsStreaming,
			SupportsKVAccess:       status.SupportsKVAccess,
			SupportsTensorHooks:    status.SupportsTensorHooks,
			LastHealthAtUnixMs:     unixMilliOrZero(status.LastHealthAt),
			LastError:              status.LastError,
		},
	})
}

func NormalizeBackendRuntimes(runtimes BackendRuntimes) BackendRuntimes {
	llama := runtimes.LlamaCPP
	llama.Backend = cleanRuntimeCompactText(firstNonEmptyRuntimeText(llama.Backend, BackendName), 64)
	if llama.Backend == "" {
		llama.Backend = BackendName
	}
	llama.BaseURL = cleanRuntimeBaseURL(llama.BaseURL)
	llama.ModelPath = cleanRuntimePath(llama.ModelPath)
	llama.ModelFilename = cleanRuntimePath(llama.ModelFilename)
	llama.ModelID = cleanRuntimePath(firstNonEmptyRuntimeText(llama.ModelID, llama.ModelFilename))
	llama.ModelFamilyHint = cleanRuntimeCompactText(llama.ModelFamilyHint, 64)
	llama.QuantizationHint = cleanRuntimeCompactText(llama.QuantizationHint, 64)
	llama.LastError = cleanStatusText(llama.LastError, maxStatusReasonLen)
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
	llama.SupportsTensorHooks = false
	loaded := llama.Enabled && llama.Running && llama.Healthy && llama.ModelPath != "" && llama.ModelFilename != ""
	llama.Loaded = loaded
	llama.Warm = loaded
	if !loaded {
		llama.Warm = false
	}
	return BackendRuntimes{LlamaCPP: llama}
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
	args = append(args, cfg.ExtraArgs...)
	return args
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
	if output != nil {
		cmd.Stdout = output
		cmd.Stderr = output
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return &osManagedProcess{cmd: cmd}, nil
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
