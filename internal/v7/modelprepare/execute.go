package modelprepare

import (
	"context"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/Ryvion/ryvion-node/internal/v7/llamacpp"
	"github.com/Ryvion/ryvion-node/internal/v7/modelcache"
	"github.com/Ryvion/ryvion-node/internal/v7/modelpolicy"
)

var (
	ErrPrepareDisabled = errors.New("modelprepare: feature disabled")
	ErrPolicyBlocked   = errors.New("modelprepare: policy blocked prepare")
	ErrHashMismatch    = errors.New("modelprepare: artifact sha256 mismatch")
	ErrExistingModel   = errors.New("modelprepare: existing model is not safely reusable")
	ErrWarmFailed      = errors.New("modelprepare: keep_warm failed")
)

type LlamaCppManager interface {
	Start(context.Context) llamacpp.LlamaCppSidecarStatus
	Status(context.Context) llamacpp.LlamaCppSidecarStatus
	RestartWithModel(context.Context, string) llamacpp.LlamaCppSidecarStatus
}

type BenchmarkRunner interface {
	Run(context.Context, llamacpp.BenchmarkConfig) llamacpp.BenchmarkStatusSnapshot
}

type ExecuteOptions struct {
	Getenv            func(string) string
	Policy            modelpolicy.Policy
	HTTPClient        *http.Client
	LlamaCppManager   LlamaCppManager
	BenchmarkRunner   BenchmarkRunner
	Now               func() time.Time
	AllowFileURI      bool
	AllowInsecureHTTP bool
}

type PrepareExecutionResult struct {
	Spec           PrepareSpec
	Downloaded     bool
	HashVerified   bool
	Installed      bool
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
	LastPrepareID string              `json:"last_prepare_id"`
	LastModelID   string              `json:"last_model_id"`
	LastError     string              `json:"last_error"`
	Counters      LocalStatusCounters `json:"counters"`
}

type LocalStatus struct {
	mu       sync.RWMutex
	snapshot LocalStatusSnapshot
}

func NewLocalStatus() *LocalStatus {
	return &LocalStatus{}
}

func ExecutePrepareAssignment(ctx context.Context, specJSON string, opts ExecuteOptions) (PrepareReceipt, bool, error) {
	identity, isPrepare := PrepareAssignmentIdentityFromJSON(specJSON)
	if !isPrepare {
		return PrepareReceipt{}, false, nil
	}
	if !PrepareEnabledFromEnv(opts.getenv()) {
		err := codedError{code: "model_prepare_disabled", err: ErrPrepareDisabled}
		return BuildPrepareRejectionReceiptFromIdentity(identity, err), true, err
	}
	spec, err := DecodePrepareSpec(specJSON)
	if err != nil {
		return BuildPrepareRejectionReceiptFromIdentity(identity, err), true, err
	}
	receipt, err := ExecutePrepareSpec(ctx, spec, opts)
	if err != nil && strings.TrimSpace(receipt.ResultHashHex) == "" {
		receipt = BuildPrepareRejectionReceipt(spec, err)
	}
	return receipt, true, err
}

func ExecutePrepareSpec(ctx context.Context, spec PrepareSpec, opts ExecuteOptions) (PrepareReceipt, error) {
	spec = NormalizePrepareSpec(spec)
	if err := ValidatePrepareSpec(spec); err != nil {
		return PrepareReceipt{}, err
	}
	if !PrepareEnabledFromEnv(opts.getenv()) {
		return PrepareReceipt{}, codedError{code: "model_prepare_disabled", err: ErrPrepareDisabled}
	}
	if ctx == nil {
		ctx = context.Background()
	}
	timeout := time.Duration(spec.TimeoutMs) * time.Millisecond
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	policy := modelpolicy.NormalizePolicy(opts.Policy)
	if err := modelpolicy.ValidatePolicy(policy); err != nil {
		return PrepareReceipt{}, codedError{code: "policy_invalid", err: err}
	}
	size := uint64(spec.ArtifactSizeBytes)
	preflight := modelpolicy.EvaluatePrepareRequest(policy, modelpolicy.PrepareRequest{
		ModelID:        spec.ModelID,
		ModelSizeBytes: size,
		Family:         spec.ModelFamily,
		Format:         spec.ArtifactFormat,
	})
	if !preflight.Allowed {
		return BuildPrepareRejectionReceipt(spec, codedError{code: preflight.Reason, err: ErrPolicyBlocked}), codedError{code: preflight.Reason, err: ErrPolicyBlocked}
	}

	existing, exists, err := modelcache.ExistingModel(policy.CacheDir, spec.ModelID, spec.ArtifactURI)
	if err != nil {
		return PrepareReceipt{}, codedError{code: "cache_existing_check_failed", err: err}
	}
	if exists {
		reused, err := reusableExistingModel(existing.Path, spec.ArtifactSHA256)
		if err != nil {
			return BuildPrepareRejectionReceipt(spec, err), err
		}
		if reused {
			result := PrepareExecutionResult{
				Spec:           spec,
				Downloaded:     false,
				HashVerified:   spec.ArtifactSHA256 != "",
				Installed:      true,
				ModelPath:      existing.Path,
				ModelSizeBytes: existing.SizeBytes,
			}
			return finalizePrepareResult(runCtx, result, opts)
		}
	}

	cacheStatus := modelcache.Scan(policy.CacheDir)
	capacity := modelpolicy.EvaluatePrepareRequest(policy, modelpolicy.PrepareRequest{
		ModelID:        spec.ModelID,
		ModelSizeBytes: size,
		CacheUsedBytes: uint64NonNegative(cacheStatus.TotalBytes),
		Family:         spec.ModelFamily,
		Format:         spec.ArtifactFormat,
	})
	if !capacity.Allowed {
		return BuildPrepareRejectionReceipt(spec, codedError{code: capacity.Reason, err: ErrPolicyBlocked}), codedError{code: capacity.Reason, err: ErrPolicyBlocked}
	}
	if err := os.MkdirAll(policy.CacheDir, 0o755); err != nil {
		return PrepareReceipt{}, codedError{code: "cache_create_failed", err: err}
	}
	tmp, err := os.CreateTemp(policy.CacheDir, ".prepare-*.tmp")
	if err != nil {
		return PrepareReceipt{}, codedError{code: "cache_temp_create_failed", err: err}
	}
	tmpPath := tmp.Name()
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return PrepareReceipt{}, codedError{code: "cache_temp_close_failed", err: err}
	}
	_ = os.Remove(tmpPath)
	removeTmp := true
	defer func() {
		if removeTmp {
			_ = os.Remove(tmpPath)
		}
	}()

	download, err := DownloadArtifact(runCtx, spec.ArtifactURI, tmpPath, DownloadOptions{
		HTTPClient:        opts.HTTPClient,
		MaxBytes:          size,
		ExpectedSizeBytes: spec.ArtifactSizeBytes,
		AllowFileURI:      opts.AllowFileURI || FileURIAllowedFromEnv(opts.getenv()),
		AllowInsecureHTTP: opts.AllowInsecureHTTP,
	})
	if err != nil {
		return BuildPrepareRejectionReceipt(spec, codedError{code: "download_failed", err: err}), codedError{code: "download_failed", err: err}
	}
	hashVerified := false
	if spec.ArtifactSHA256 != "" {
		ok, _, err := VerifyFileSHA256(download.Path, spec.ArtifactSHA256)
		if err != nil {
			return BuildPrepareRejectionReceipt(spec, codedError{code: "hash_verify_failed", err: err}), codedError{code: "hash_verify_failed", err: err}
		}
		if !ok {
			return BuildPrepareRejectionReceipt(spec, codedError{code: "hash_mismatch", err: ErrHashMismatch}), codedError{code: "hash_mismatch", err: ErrHashMismatch}
		}
		hashVerified = true
	}
	installed, err := modelcache.InstallDownloadedModel(modelcache.InstallOptions{
		CacheDir:       policy.CacheDir,
		ModelID:        spec.ModelID,
		ArtifactURI:    spec.ArtifactURI,
		SourcePath:     download.Path,
		HashVerified:   hashVerified,
		FamilyHint:     spec.ModelFamily,
		Format:         firstNonEmptyPrepare(spec.ArtifactFormat, modelcache.DefaultFormat),
		SizeBytes:      download.Bytes,
		Now:            opts.now,
		QuantizationID: "",
	})
	if err != nil {
		return BuildPrepareRejectionReceipt(spec, codedError{code: "cache_install_failed", err: err}), codedError{code: "cache_install_failed", err: err}
	}
	removeTmp = false

	result := PrepareExecutionResult{
		Spec:           spec,
		Downloaded:     true,
		HashVerified:   hashVerified,
		Installed:      true,
		ModelPath:      installed.DestinationPath,
		ModelSizeBytes: installed.Model.SizeBytes,
	}
	return finalizePrepareResult(runCtx, result, opts)
}

func finalizePrepareResult(ctx context.Context, result PrepareExecutionResult, opts ExecuteOptions) (PrepareReceipt, error) {
	manager := opts.LlamaCppManager
	if manager == nil && (result.Spec.KeepWarm || result.Spec.RunBenchmarkAfterPrepare) {
		manager = llamacpp.NewManagerFromEnv()
	}
	if manager != nil && (result.Spec.KeepWarm || result.Spec.RunBenchmarkAfterPrepare) {
		status := manager.RestartWithModel(ctx, result.ModelPath)
		result.Warm = status.Healthy && strings.TrimSpace(status.ModelPath) == strings.TrimSpace(result.ModelPath)
		if result.Spec.KeepWarm && !result.Warm {
			return BuildPrepareRejectionReceipt(result.Spec, codedError{code: "keep_warm_failed", err: ErrWarmFailed}), codedError{code: "keep_warm_failed", err: ErrWarmFailed}
		}
	}
	if result.Spec.RunBenchmarkAfterPrepare {
		runner := opts.BenchmarkRunner
		if runner == nil && manager != nil {
			runner = llamacpp.BenchmarkRunner{
				Sidecar: manager,
				Client:  llamacpp.OpenAIClient{},
			}
		}
		if runner != nil {
			snapshot := runner.Run(ctx, llamacpp.BenchmarkConfig{
				ModelID:      result.Spec.ModelID,
				MaxTokens:    16,
				Temperature:  0,
				TimeoutMs:    minInt64(result.Spec.TimeoutMs, llamacpp.DefaultBenchmarkTimeoutMs),
				Streaming:    true,
				MeasuredRuns: 1,
				WarmupRuns:   1,
			})
			result.Benchmark = &snapshot
		}
	}
	return BuildPrepareReceipt(result)
}

func reusableExistingModel(path string, expectedSHA string) (bool, error) {
	if strings.TrimSpace(expectedSHA) == "" {
		return false, codedError{code: "existing_model_unverified", err: ErrExistingModel}
	}
	ok, _, err := VerifyFileSHA256(path, expectedSHA)
	if err != nil {
		return false, codedError{code: "existing_model_hash_failed", err: err}
	}
	if !ok {
		return false, codedError{code: "existing_model_hash_mismatch", err: ErrExistingModel}
	}
	return true, nil
}

func (opts ExecuteOptions) getenv() func(string) string {
	if opts.Getenv != nil {
		return opts.Getenv
	}
	return os.Getenv
}

func (opts ExecuteOptions) now() time.Time {
	if opts.Now != nil {
		return opts.Now()
	}
	return time.Now()
}

func uint64NonNegative(value int64) uint64 {
	if value <= 0 {
		return 0
	}
	return uint64(value)
}

func minInt64(a, b int64) int64 {
	if a <= 0 {
		return b
	}
	if b <= 0 || a < b {
		return a
	}
	return b
}

func (s *LocalStatus) RecordSeen(prepareID, modelID string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastPrepareID = cleanPrepareText(prepareID, maxPrepareTextLen)
	s.snapshot.LastModelID = cleanPrepareText(modelID, maxPrepareTextLen)
	s.snapshot.LastError = ""
	s.snapshot.Counters.Seen++
}

func (s *LocalStatus) RecordExecuted(prepareID, modelID string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastPrepareID = cleanPrepareText(prepareID, maxPrepareTextLen)
	s.snapshot.LastModelID = cleanPrepareText(modelID, maxPrepareTextLen)
	s.snapshot.LastError = ""
	s.snapshot.Counters.Executed++
}

func (s *LocalStatus) RecordRejected(prepareID, modelID string, err error) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastPrepareID = cleanPrepareText(prepareID, maxPrepareTextLen)
	s.snapshot.LastModelID = cleanPrepareText(modelID, maxPrepareTextLen)
	s.snapshot.LastError = ErrorCode(err)
	s.snapshot.Counters.Rejected++
}

func (s *LocalStatus) RecordReceiptSubmitted(prepareID, modelID string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastPrepareID = cleanPrepareText(prepareID, maxPrepareTextLen)
	s.snapshot.LastModelID = cleanPrepareText(modelID, maxPrepareTextLen)
	s.snapshot.LastError = ""
	s.snapshot.Counters.ReceiptSubmitted++
}

func (s *LocalStatus) RecordReceiptFailed(prepareID, modelID string, err error) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastPrepareID = cleanPrepareText(prepareID, maxPrepareTextLen)
	s.snapshot.LastModelID = cleanPrepareText(modelID, maxPrepareTextLen)
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
	code := cleanPrepareText(e.code, maxPrepareTextLen)
	if code == "" {
		code = "model_prepare_failed"
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
		return cleanPrepareText(coded.code, maxPrepareTextLen)
	}
	message := strings.ToLower(strings.TrimSpace(err.Error()))
	switch {
	case errors.Is(err, ErrPrepareDisabled), strings.Contains(message, "feature disabled"):
		return "model_prepare_disabled"
	case errors.Is(err, ErrPolicyBlocked):
		return "policy_prepare_blocked"
	case errors.Is(err, ErrHashMismatch):
		return "hash_mismatch"
	case errors.Is(err, ErrExistingModel):
		return "existing_model_unsafe"
	case errors.Is(err, ErrWarmFailed):
		return "keep_warm_failed"
	case errors.Is(err, ErrInvalidArtifactURI):
		return "invalid_artifact_uri"
	case errors.Is(err, ErrDownloadTooLarge):
		return "download_too_large"
	case errors.Is(err, ErrDownloadSize):
		return "download_size_mismatch"
	case errors.Is(err, modelcache.ErrModelExists):
		return "cache_model_exists"
	case strings.Contains(message, "context deadline"):
		return "timeout"
	default:
		return "model_prepare_failed"
	}
}

func safeModelPath(pathValue string) string {
	return cleanPrepareText(filepath.Clean(strings.TrimSpace(pathValue)), maxPreparePathLen)
}
