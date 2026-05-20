package modelwarm

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/Ryvion/ryvion-node/internal/models/cache"
	"github.com/Ryvion/ryvion-node/internal/models/policy"
	"github.com/Ryvion/ryvion-node/internal/runtimes/llamacpp"
)

var (
	ErrWarmDisabled  = errors.New("modelwarm: feature disabled")
	ErrModelNotFound = errors.New("modelwarm: model not found in local cache")
	ErrWarmFailed    = errors.New("modelwarm: llama.cpp warm failed")
)

type LlamaCppManager interface {
	Status(context.Context) llamacpp.LlamaCppSidecarStatus
	RestartWithModel(context.Context, string) llamacpp.LlamaCppSidecarStatus
}

type LlamaCppEnabler interface {
	SetEnabled(bool) llamacpp.LlamaCppSidecarConfig
}

type BenchmarkRunner interface {
	Run(context.Context, llamacpp.BenchmarkConfig) llamacpp.BenchmarkStatusSnapshot
}

type ExecuteOptions struct {
	Getenv          func(string) string
	Policy          modelpolicy.Policy
	ModelCache      *modelcache.Status
	LlamaCppManager LlamaCppManager
	BenchmarkRunner BenchmarkRunner
}

type WarmExecutionResult struct {
	Spec           WarmSpec
	ModelPath      string
	ModelSizeBytes int64
	Warm           bool
	Benchmark      *llamacpp.BenchmarkStatusSnapshot
}

type LocalStatusCounters struct {
	Seen             uint64 `json:"seen"`
	Executed         uint64 `json:"executed"`
	Rejected         uint64 `json:"rejected"`
	ReceiptSubmitted uint64 `json:"receipt_submitted"`
	ReceiptFailed    uint64 `json:"receipt_failed"`
}

type LocalStatusSnapshot struct {
	LastWarmID  string              `json:"last_warm_id"`
	LastModelID string              `json:"last_model_id"`
	LastError   string              `json:"last_error"`
	Counters    LocalStatusCounters `json:"counters"`
}

type LocalStatus struct {
	mu       sync.RWMutex
	snapshot LocalStatusSnapshot
}

func NewLocalStatus() *LocalStatus {
	return &LocalStatus{}
}

func ExecuteWarmAssignment(ctx context.Context, specJSON string, opts ExecuteOptions) (WarmReceipt, bool, error) {
	identity, isWarm := WarmAssignmentIdentityFromJSON(specJSON)
	if !isWarm {
		return WarmReceipt{}, false, nil
	}
	if !WarmEnabledFromEnv(opts.getenv()) {
		err := codedError{code: "model_warm_disabled", err: ErrWarmDisabled}
		return BuildWarmRejectionReceiptFromIdentity(identity, err), true, err
	}
	spec, err := DecodeWarmSpec(specJSON)
	if err != nil {
		return BuildWarmRejectionReceiptFromIdentity(identity, err), true, err
	}
	receipt, err := ExecuteWarmSpec(ctx, spec, opts)
	if err != nil && strings.TrimSpace(receipt.ResultHashHex) == "" {
		receipt = BuildWarmRejectionReceipt(spec, err)
	}
	return receipt, true, err
}

func ExecuteWarmSpec(ctx context.Context, spec WarmSpec, opts ExecuteOptions) (WarmReceipt, error) {
	spec = NormalizeWarmSpec(spec)
	if err := ValidateWarmSpec(spec); err != nil {
		return WarmReceipt{}, err
	}
	if !WarmEnabledFromEnv(opts.getenv()) {
		return WarmReceipt{}, codedError{code: "model_warm_disabled", err: ErrWarmDisabled}
	}
	if ctx == nil {
		ctx = context.Background()
	}
	timeout := time.Duration(spec.TimeoutMs) * time.Millisecond
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	policy := opts.policy()
	if err := modelpolicy.ValidatePolicy(policy); err != nil {
		return WarmReceipt{}, codedError{code: "policy_invalid", err: err}
	}
	cacheStatus := opts.cacheStatus(policy.CacheDir)
	cachedModel, ok := resolveWarmModel(cacheStatus, spec)
	if !ok {
		err := codedError{code: "model_not_found", err: ErrModelNotFound}
		return BuildWarmRejectionReceipt(spec, err), err
	}

	manager := opts.manager()
	if enabler, ok := manager.(LlamaCppEnabler); ok {
		enabler.SetEnabled(true)
	}

	status := manager.Status(runCtx)
	if status.Attached {
		status = manager.RestartWithModel(runCtx, cachedModel.Path)
		if !modelStatusWarm(status, cachedModel.Path) {
			status = waitForWarmModel(runCtx, manager, cachedModel.Path, status)
		}
	} else if !modelStatusWarm(status, cachedModel.Path) {
		if sameWarmPath(status.ModelPath, cachedModel.Path) && status.Running {
			status = waitForWarmModel(runCtx, manager, cachedModel.Path, status)
		} else {
			status = manager.RestartWithModel(runCtx, cachedModel.Path)
			if !modelStatusWarm(status, cachedModel.Path) {
				status = waitForWarmModel(runCtx, manager, cachedModel.Path, status)
			}
		}
	}
	warm := modelStatusWarm(status, cachedModel.Path)
	if !warm {
		err := codedError{code: "llamacpp_warm_failed", err: ErrWarmFailed}
		return buildWarmFailureReceipt(spec, cachedModel, err), err
	}

	result := WarmExecutionResult{
		Spec:           spec,
		ModelPath:      cachedModel.Path,
		ModelSizeBytes: cachedModel.SizeBytes,
		Warm:           true,
	}
	if spec.RunBenchmarkAfterWarm {
		runner := opts.benchmarkRunner(manager)
		snapshot := runner.Run(runCtx, llamacpp.BenchmarkConfig{
			ModelID:      spec.ModelID,
			MaxTokens:    16,
			Temperature:  0,
			TimeoutMs:    minWarmInt64(spec.TimeoutMs, llamacpp.DefaultBenchmarkTimeoutMs),
			Streaming:    true,
			MeasuredRuns: 1,
			WarmupRuns:   1,
		})
		result.Benchmark = &snapshot
	}
	return BuildWarmReceipt(result)
}

func resolveWarmModel(status modelcache.Status, spec WarmSpec) (modelcache.Model, bool) {
	if model, ok := findCachedModel(status, spec.ModelID); ok {
		return model, true
	}
	if strings.TrimSpace(spec.ModelPath) == "" || !warmModelPathMatches(spec.ModelPath, spec.ModelID) {
		return modelcache.Model{}, false
	}
	info, err := os.Stat(spec.ModelPath)
	if err != nil || info.IsDir() {
		return modelcache.Model{}, false
	}
	return modelcache.Model{
		ModelID:   spec.ModelID,
		Filename:  filepath.Base(spec.ModelPath),
		Path:      spec.ModelPath,
		SizeBytes: info.Size(),
		Format:    "gguf",
		Installed: true,
		Resident:  true,
	}, true
}

func buildWarmFailureReceipt(spec WarmSpec, cachedModel modelcache.Model, runErr error) WarmReceipt {
	return buildWarmRejectionReceipt(spec.WarmID, spec.RequestID, spec.JobID, spec.ModelID, spec.Backend, cachedModel.Path, cachedModel.SizeBytes, runErr)
}

func (opts ExecuteOptions) getenv() func(string) string {
	if opts.Getenv != nil {
		return opts.Getenv
	}
	return os.Getenv
}

func (opts ExecuteOptions) policy() modelpolicy.Policy {
	if strings.TrimSpace(opts.Policy.CacheDir) == "" {
		return modelpolicy.FromEnv()
	}
	return modelpolicy.NormalizePolicy(opts.Policy)
}

func (opts ExecuteOptions) cacheStatus(cacheDir string) modelcache.Status {
	if opts.ModelCache != nil {
		status := *opts.ModelCache
		return modelcache.NormalizeStatus(status)
	}
	return modelcache.BuildStatus(cacheDir)
}

func (opts ExecuteOptions) manager() LlamaCppManager {
	if opts.LlamaCppManager != nil {
		return opts.LlamaCppManager
	}
	cfg := llamacpp.ConfigFromEnv()
	cfg.Enabled = true
	return llamacpp.NewManager(cfg)
}

func (opts ExecuteOptions) benchmarkRunner(manager LlamaCppManager) BenchmarkRunner {
	if opts.BenchmarkRunner != nil {
		return opts.BenchmarkRunner
	}
	return llamacpp.BenchmarkRunner{
		Sidecar: benchmarkSidecarAdapter{manager: manager},
		Client:  llamacpp.OpenAIClient{},
	}
}

type benchmarkSidecarAdapter struct {
	manager LlamaCppManager
}

func (a benchmarkSidecarAdapter) Start(ctx context.Context) llamacpp.LlamaCppSidecarStatus {
	return a.manager.Status(ctx)
}

func (a benchmarkSidecarAdapter) Status(ctx context.Context) llamacpp.LlamaCppSidecarStatus {
	return a.manager.Status(ctx)
}

func findCachedModel(status modelcache.Status, modelID string) (modelcache.Model, bool) {
	status = modelcache.NormalizeStatus(status)
	modelID = cleanWarmText(modelID, maxWarmTextLen)
	if modelID == "" {
		return modelcache.Model{}, false
	}
	for _, model := range status.Models {
		if !model.Installed || strings.TrimSpace(model.Path) == "" {
			continue
		}
		if warmModelMatch(model, modelID) {
			return model, true
		}
	}
	return modelcache.Model{}, false
}

func warmModelMatch(model modelcache.Model, modelID string) bool {
	return modelcache.ModelMatches(model, modelID)
}

func warmModelPathMatches(modelPath string, modelID string) bool {
	return modelcache.ModelIDMatches(modelPath, modelID)
}

func modelStatusWarm(status llamacpp.LlamaCppSidecarStatus, modelPath string) bool {
	return status.Enabled &&
		status.Available &&
		status.Running &&
		status.Healthy &&
		sameWarmPath(status.ModelPath, modelPath)
}

func waitForWarmModel(ctx context.Context, manager LlamaCppManager, modelPath string, last llamacpp.LlamaCppSidecarStatus) llamacpp.LlamaCppSidecarStatus {
	if modelStatusWarm(last, modelPath) {
		return last
	}
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return last
		case <-ticker.C:
			last = manager.Status(ctx)
			if modelStatusWarm(last, modelPath) {
				return last
			}
		}
	}
}

func sameWarmPath(a, b string) bool {
	a = strings.TrimSpace(a)
	b = strings.TrimSpace(b)
	if a == "" || b == "" {
		return false
	}
	if a == b {
		return true
	}
	return filepath.Clean(a) == filepath.Clean(b)
}

func minWarmInt64(a, b int64) int64 {
	if a <= 0 {
		return b
	}
	if b <= 0 || a < b {
		return a
	}
	return b
}

func (s *LocalStatus) RecordSeen(warmID, modelID string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastWarmID = cleanWarmText(warmID, maxWarmTextLen)
	s.snapshot.LastModelID = cleanWarmText(modelID, maxWarmTextLen)
	s.snapshot.LastError = ""
	s.snapshot.Counters.Seen++
}

func (s *LocalStatus) RecordExecuted(warmID, modelID string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastWarmID = cleanWarmText(warmID, maxWarmTextLen)
	s.snapshot.LastModelID = cleanWarmText(modelID, maxWarmTextLen)
	s.snapshot.LastError = ""
	s.snapshot.Counters.Executed++
}

func (s *LocalStatus) RecordRejected(warmID, modelID string, err error) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastWarmID = cleanWarmText(warmID, maxWarmTextLen)
	s.snapshot.LastModelID = cleanWarmText(modelID, maxWarmTextLen)
	s.snapshot.LastError = ErrorCode(err)
	s.snapshot.Counters.Rejected++
}

func (s *LocalStatus) RecordReceiptSubmitted(warmID, modelID string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastWarmID = cleanWarmText(warmID, maxWarmTextLen)
	s.snapshot.LastModelID = cleanWarmText(modelID, maxWarmTextLen)
	s.snapshot.LastError = ""
	s.snapshot.Counters.ReceiptSubmitted++
}

func (s *LocalStatus) RecordReceiptFailed(warmID, modelID string, err error) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastWarmID = cleanWarmText(warmID, maxWarmTextLen)
	s.snapshot.LastModelID = cleanWarmText(modelID, maxWarmTextLen)
	s.snapshot.LastError = ErrorCode(err)
	s.snapshot.Counters.ReceiptFailed++
}

func (s *LocalStatus) Snapshot() LocalStatusSnapshot {
	if s == nil {
		return LocalStatusSnapshot{}
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.snapshot
}

type codedError struct {
	code string
	err  error
}

func (e codedError) Error() string {
	code := cleanWarmText(e.code, maxWarmTextLen)
	if code == "" {
		code = "model_warm_failed"
	}
	return code
}

func (e codedError) Unwrap() error {
	return e.err
}

func ErrorCode(err error) string {
	if err == nil {
		return ""
	}
	var coded codedError
	if errors.As(err, &coded) {
		return cleanWarmText(coded.code, maxWarmTextLen)
	}
	message := strings.ToLower(strings.TrimSpace(err.Error()))
	switch {
	case errors.Is(err, ErrWarmDisabled), strings.Contains(message, "feature disabled"):
		return "model_warm_disabled"
	case errors.Is(err, ErrModelNotFound):
		return "model_not_found"
	case errors.Is(err, ErrWarmFailed):
		return "llamacpp_warm_failed"
	case strings.Contains(message, "context deadline"):
		return "timeout"
	default:
		return "model_warm_failed"
	}
}
