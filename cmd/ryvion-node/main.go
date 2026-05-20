package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"log/slog"
	"math"
	"os"
	"os/signal"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/Ryvion/ryvion-node/internal/blob"
	capability "github.com/Ryvion/ryvion-node/internal/capabilities/passport"
	"github.com/Ryvion/ryvion-node/internal/hub"
	heartbeat "github.com/Ryvion/ryvion-node/internal/hub/heartbeat"
	"github.com/Ryvion/ryvion-node/internal/hw"
	"github.com/Ryvion/ryvion-node/internal/llamacpp"
	"github.com/Ryvion/ryvion-node/internal/nodekey"
	"github.com/Ryvion/ryvion-node/internal/runner"
	"github.com/Ryvion/ryvion-node/internal/runtimeexec"
	"github.com/Ryvion/ryvion-node/internal/sandbox"
)

var version = "dev"

var cachedGPUUtil atomic.Uint64

type runnerResultSnapshot struct {
	DurationMs    int64
	ResultHashHex string
	ExitCode      int
	MeteringUnits uint64
	BlobURL       string
	ObjectKey     string
	Metadata      map[string]any
}

type nodeConfig struct {
	HubURL       string
	Device       string
	Country      string
	Referral     string
	GPUs         string
	KeyPath      string
	MaxGPUUtil   float64
	BindToken    string
	UIPort       int
	HeartbeatDur time.Duration
	LlamaCPP     llamacpp.Config
}

type runtimeHealthSnapshot struct {
	OCIAvailable         bool
	LlamaCPPAvailable    bool
	GPUReady             bool
	ManagedOCIGPUReady   bool
	SupportedRunnerKinds []string
	EngineKind           string
	LlamaCPPModel        string
	Health               string
	Message              string
}

func main() {
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "claim":
			runClaim()
			return
		case "identity":
			runIdentity()
			return
		}
	}
	if isWindowsService() {
		runAsWindowsService()
		return
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	runNode(ctx)
}

func runNode(ctx context.Context) {
	cfg := parseConfig()
	pub, priv, err := nodekey.LoadOrCreate(cfg.KeyPath)
	if err != nil {
		slog.Error("failed to load node key", "error", err)
		os.Exit(1)
	}

	client := hub.New(
		cfg.HubURL,
		pub,
		priv,
		hub.WithBindToken(cfg.BindToken),
		hub.WithUserAgent("ryvion-node/"+version),
	)

	caps := hw.DetectCaps(cfg.Device)
	deviceType := resolveDeviceType(cfg.Device, caps)
	if err := registerWithRetry(ctx, client, caps, deviceType, cfg); err != nil {
		slog.Error("registration stopped", "error", err)
		return
	}

	if err := client.SolveChallenge(ctx); err != nil {
		slog.Debug("challenge solve failed", "error", err)
	}

	go heartbeatLoop(ctx, client, caps, deviceType, cfg)
	workLoop(ctx, client, caps, cfg)
}

func parseConfig() nodeConfig {
	cfg := nodeConfig{
		HubURL:       firstNonEmpty(os.Getenv("RYV_HUB_URL"), "https://api.ryvion.ai"),
		Device:       firstNonEmpty(os.Getenv("RYV_DEVICE"), os.Getenv("RYV_DEVICE_TYPE")),
		Country:      os.Getenv("RYV_COUNTRY"),
		Referral:     os.Getenv("RYV_REFERRAL"),
		GPUs:         firstNonEmpty(os.Getenv("RYV_GPUS"), "auto"),
		KeyPath:      os.Getenv("RYV_KEY_PATH"),
		BindToken:    os.Getenv("RYV_BIND_TOKEN"),
		HeartbeatDur: 30 * time.Second,
	}
	flag.StringVar(&cfg.HubURL, "hub", cfg.HubURL, "Ryvion hub URL")
	flag.StringVar(&cfg.Device, "device", cfg.Device, "operator device class hint")
	flag.StringVar(&cfg.Device, "type", cfg.Device, "deprecated alias for -device")
	flag.StringVar(&cfg.Country, "country", cfg.Country, "declared operator country")
	flag.StringVar(&cfg.Referral, "referral", cfg.Referral, "operator referral code")
	flag.StringVar(&cfg.GPUs, "gpus", cfg.GPUs, "OCI GPU flag value, usually auto or all")
	flag.StringVar(&cfg.KeyPath, "key", cfg.KeyPath, "node identity key path")
	flag.Float64Var(&cfg.MaxGPUUtil, "max-gpu-util", envFloat("RYV_MAX_GPU_UTIL"), "skip work while GPU util is above this percentage")
	flag.IntVar(&cfg.UIPort, "ui-port", envInt("RYV_UI_PORT"), "legacy local status UI port; accepted for compatibility")
	flag.DurationVar(&cfg.HeartbeatDur, "heartbeat-interval", cfg.HeartbeatDur, "heartbeat interval")
	flag.Parse()
	cfg.LlamaCPP = llamacpp.ResolveConfig(os.Getenv)
	cfg.HubURL = strings.TrimRight(strings.TrimSpace(cfg.HubURL), "/")
	if cfg.HubURL == "" {
		cfg.HubURL = "https://api.ryvion.ai"
	}
	if cfg.HeartbeatDur <= 0 {
		cfg.HeartbeatDur = 30 * time.Second
	}
	return cfg
}

func registerWithRetry(ctx context.Context, client *hub.Client, caps hw.CapSet, deviceType string, cfg nodeConfig) error {
	backoff := 5 * time.Second
	for {
		err := client.Register(ctx, hubCapabilitiesFromCaps(caps), deviceType, cfg.Referral, cfg.Country)
		if err == nil {
			slog.Info("registered node", "hub", cfg.HubURL, "device_type", deviceType, "gpu", caps.GPUModel, "pubkey", client.PublicKeyHex())
			return nil
		}
		slog.Warn("register failed", "error", err, "retry_in", backoff)
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(backoff):
		}
		if backoff < 2*time.Minute {
			backoff = time.Duration(float64(backoff) * 1.5)
		}
	}
}

func heartbeatLoop(ctx context.Context, client *hub.Client, caps hw.CapSet, deviceType string, cfg nodeConfig) {
	send := func(reason string) {
		metrics := hw.SampleMetrics()
		runtimeHealth := detectRuntimeHealth(ctx, caps, cfg)
		cachedGPUUtil.Store(math.Float64bits(metrics.GPUUtil))
		throttled := cfg.MaxGPUUtil > 0 && cfg.MaxGPUUtil < 100 && metrics.GPUUtil > cfg.MaxGPUUtil

		var capsPayload *heartbeat.NodeHeartbeatPayload
		if heartbeat.NodeCapsEnabledFromEnv() {
			payload, err := buildNodeHeartbeatPayload(client.PublicKeyHex(), caps, deviceType, cfg.Country, runtimeHealth)
			if err != nil {
				slog.Warn("capability payload skipped", "error", err)
			} else {
				capsPayload = &payload
			}
		}

		if err := client.SendHealthReport(ctx, hub.HealthReport{
			GPUReady:   runtimeHealth.GPUReady,
			RuntimeGPU: runtimeHealth.ManagedOCIGPUReady,
			Message:    runtimeHealth.Message,
		}); err != nil {
			slog.Debug("health report failed", "reason", reason, "error", err)
		}

		_, err := client.Heartbeat(ctx, hub.Metrics{
			TimestampMs:    time.Now().UnixMilli(),
			CPUUtil:        metrics.CPUUtil,
			MemUtil:        metrics.MemUtil,
			GPUUtil:        metrics.GPUUtil,
			PowerWatts:     metrics.PowerWatts,
			GPUThrottled:   throttled,
			NodeCapability: capsPayload,
		})
		if err != nil {
			slog.Warn("heartbeat failed", "reason", reason, "error", err)
			return
		}
		slog.Debug("heartbeat sent", "reason", reason)
	}

	send("startup")
	ticker := time.NewTicker(cfg.HeartbeatDur)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			send("periodic")
		}
	}
}

func buildNodeHeartbeatPayload(nodePublicKey string, caps hw.CapSet, deviceType string, country string, runtimeHealth runtimeHealthSnapshot) (heartbeat.NodeHeartbeatPayload, error) {
	policy := sandbox.DefaultSandboxPolicy()
	return heartbeat.BuildNodeHeartbeatPayload(heartbeat.BuildNodeHeartbeatPayloadInput{
		AgentVersion:         version,
		NodePublicKey:        nodePublicKey,
		OS:                   runtime.GOOS,
		Arch:                 runtime.GOARCH,
		DeviceType:           deviceType,
		DeclaredCountry:      country,
		HardwareCapabilities: caps,
		RuntimeProfile: capability.RuntimeProfile{
			OCIAvailable:         runtimeHealth.OCIAvailable,
			LlamaCPPAvailable:    runtimeHealth.LlamaCPPAvailable,
			LlamaCPPModel:        runtimeHealth.LlamaCPPModel,
			SupportedRunnerKinds: runtimeHealth.SupportedRunnerKinds,
		},
		WorkCapabilitySummary: capability.WorkCapabilitySummary{
			SupportsManagedOCI:     runtimeHealth.OCIAvailable,
			SupportsLlamaCPP:       runtimeHealth.LlamaCPPAvailable,
			SupportsArtifactUpload: true,
		},
		SandboxCapabilitySummary: capability.SandboxCapabilitySummary{
			RejectsUntrustedCustomRunners: true,
			RunnerAllowlistEnabled:        true,
			NetworkIsolationSupported:     true,
			FilesystemIsolationPlanned:    true,
		},
		SandboxPolicy: &policy,
		EvidenceCapabilitySummary: capability.EvidenceCapabilitySummary{
			SupportsArtifactManifest:    true,
			SupportsRYV3EvidencePayload: true,
			SupportsRuntimeHash:         false,
		},
	})
}

func detectRuntimeHealth(ctx context.Context, caps hw.CapSet, cfg nodeConfig) runtimeHealthSnapshot {
	gpuReady := strings.TrimSpace(caps.GPUModel) != "" || caps.VRAMBytes > 0
	snapshot := runtimeHealthSnapshot{
		GPUReady: gpuReady,
		Health:   "missing",
	}
	llamaHealth := llamacpp.Probe(ctx, cfg.LlamaCPP, nil)
	if llamaHealth.Available {
		snapshot.LlamaCPPAvailable = true
		snapshot.LlamaCPPModel = llamaHealth.Model
		snapshot.Health = "ready"
		snapshot.SupportedRunnerKinds = append(snapshot.SupportedRunnerKinds, "llama_cpp")
	}
	executor, err := runtimeexec.ResolveExecutor(runtime.GOOS, os.Getenv)
	if err == nil {
		snapshot.OCIAvailable = true
		snapshot.EngineKind = runtimeexec.EngineKind(executor.BinaryPath)
		if snapshot.EngineKind == "" {
			snapshot.EngineKind = runtimeexec.EngineKind(executor.Command)
		}
		snapshot.Health = "ready"
		snapshot.SupportedRunnerKinds = append(snapshot.SupportedRunnerKinds, "managed_oci")
	}
	snapshot.ManagedOCIGPUReady = snapshot.OCIAvailable && gpuReady
	snapshot.Message = runtimeHealthMessage(snapshot)
	return snapshot
}

func runtimeHealthMessage(snapshot runtimeHealthSnapshot) string {
	tokens := []string{
		"runtime-ready:" + boolToken(snapshot.OCIAvailable || snapshot.LlamaCPPAvailable),
		"runtime-health:" + firstNonEmpty(snapshot.Health, "missing"),
		"cap:llama_cpp:" + boolToken(snapshot.LlamaCPPAvailable),
		"cap:managed_oci_cpu:" + boolToken(snapshot.OCIAvailable),
		"cap:managed_oci_gpu:" + boolToken(snapshot.ManagedOCIGPUReady),
	}
	if snapshot.LlamaCPPAvailable {
		model := strings.TrimSpace(snapshot.LlamaCPPModel)
		if model == "" {
			model = "default"
		}
		tokens = append(tokens, "runtime:llama_cpp:"+sanitizeStatusToken(model), "tool:llama_cpp")
	}
	if strings.TrimSpace(snapshot.EngineKind) != "" {
		tokens = append(tokens, "runtime-source:"+strings.TrimSpace(snapshot.EngineKind))
	}
	return strings.Join(tokens, ",")
}

func boolToken(value bool) string {
	if value {
		return "1"
	}
	return "0"
}

func workLoop(ctx context.Context, client *hub.Client, caps hw.CapSet, cfg nodeConfig) {
	backoff := 5 * time.Second
	for {
		if cfg.MaxGPUUtil > 0 && cfg.MaxGPUUtil < 100 {
			gpuUtil := math.Float64frombits(cachedGPUUtil.Load())
			if gpuUtil > cfg.MaxGPUUtil {
				slog.Debug("GPU busy; delaying work fetch", "gpu_util", gpuUtil, "max_gpu_util", cfg.MaxGPUUtil)
				if !sleepOrDone(ctx, 5*time.Second) {
					return
				}
				continue
			}
		}

		work, err := client.FetchWork(ctx)
		if err != nil {
			slog.Warn("fetch work failed", "error", err, "retry_in", backoff)
			if !sleepOrDone(ctx, backoff) {
				return
			}
			if backoff < 2*time.Minute {
				backoff = time.Duration(float64(backoff) * 1.5)
			}
			continue
		}
		backoff = 5 * time.Second
		if work == nil {
			continue
		}
		processWork(ctx, client, work, caps, cfg)
	}
}

func processWork(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, caps hw.CapSet, cfg nodeConfig) {
	timeout := jobTimeout()
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	stopAbortMonitor := startAbortMonitor(runCtx, client, work, cancel)
	defer stopAbortMonitor()

	start := time.Now()
	result, runErr := runWork(runCtx, client, work, caps, cfg)
	if runErr != nil {
		slog.Warn("job failed", "job_id", work.JobID, "duration_ms", time.Since(start).Milliseconds(), "error", runErr)
		return
	}
	if result != nil {
		slog.Info("job completed", "job_id", work.JobID, "hash", result.ResultHashHex, "units", result.MeteringUnits)
	}
}

func runWork(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, caps hw.CapSet, cfg nodeConfig) (*runnerResultSnapshot, error) {
	wantsOCI := strings.TrimSpace(work.Image) != "" || strings.EqualFold(strings.TrimSpace(work.ExecutorKind), "managed_oci")
	if work.RuntimeRequirements.NeedsLlamaCPP && wantsOCI {
		err := fmt.Errorf("assignment requires llama.cpp but also declares managed OCI execution")
		return submitFailureReceiptForExecutor(ctx, client, work, err, "contradictory_runtime_requirements", llamacpp.ExecutorKind)
	}
	if work.RuntimeRequirements.NeedsLlamaCPP || (llamacpp.ShouldHandle(work.Kind, work.ExecutorKind, work.SpecJSON) && !wantsOCI) {
		return runLlamaCPPWork(ctx, client, work, cfg.LlamaCPP)
	}
	if wantsOCI || work.RuntimeRequirements.NeedsManagedOCI || work.RuntimeRequirements.NeedsManagedOCIGPU {
		return runOCIWork(ctx, client, work, cfg.GPUs, caps)
	}
	return submitFailureReceipt(ctx, client, work, fmt.Errorf("unsupported executor %q", work.ExecutorKind), "unsupported_executor")
}

func runLlamaCPPWork(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, cfg llamacpp.Config) (*runnerResultSnapshot, error) {
	if err := cfg.Validate(); err != nil {
		return submitFailureReceipt(ctx, client, work, err, "llama_cpp_unavailable")
	}
	exec, runErr := llamacpp.Execute(ctx, work.SpecJSON, cfg, llamacpp.Client{}, receiptMetadataBase(work))
	if runErr != nil {
		return submitFailureReceipt(ctx, client, work, runErr, "llama_cpp_failed")
	}
	resultHash := exec.ResultHashHex
	metadata := exec.Metadata
	if strings.TrimSpace(exec.OutputPath) != "" {
		uploadRes, uploadErr := blob.Upload(ctx, client, work.JobID, exec.OutputPath)
		if uploadErr != nil {
			metadata["upload_error"] = uploadErr.Error()
			_ = os.Remove(exec.OutputPath)
			return submitFailureReceiptForExecutor(ctx, client, work, uploadErr, "artifact_upload_failed", llamacpp.ExecutorKind)
		}
		if uploadRes == nil {
			uploadErr = fmt.Errorf("artifact upload returned no result")
			metadata["upload_error"] = uploadErr.Error()
			_ = os.Remove(exec.OutputPath)
			return submitFailureReceiptForExecutor(ctx, client, work, uploadErr, "artifact_upload_failed", llamacpp.ExecutorKind)
		}
		if uploadRes != nil {
			metadata["blob_url"] = uploadRes.URL
			metadata["object_key"] = uploadRes.Key
			if strings.TrimSpace(uploadRes.Hash) != "" {
				metadata["artifact_sha256"] = uploadRes.Hash
			}
		}
		_ = os.Remove(exec.OutputPath)
	}
	receipt := hub.Receipt{
		JobID:         work.JobID,
		ResultHashHex: resultHash,
		MeteringUnits: exec.MeteringUnits,
		Metadata:      llamacpp.SanitizeMetadata(metadata),
	}
	if err := submitReceiptWithRetry(ctx, client, receipt); err != nil {
		return &runnerResultSnapshot{ResultHashHex: resultHash, MeteringUnits: exec.MeteringUnits, Metadata: metadata}, err
	}
	return &runnerResultSnapshot{ResultHashHex: resultHash, MeteringUnits: exec.MeteringUnits, Metadata: metadata}, nil
}

func runOCIWork(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, gpus string, caps hw.CapSet) (*runnerResultSnapshot, error) {
	if strings.TrimSpace(work.Image) == "" || strings.TrimSpace(work.SpecJSON) == "" {
		err := fmt.Errorf("missing container image or spec")
		return submitFailureReceiptForExecutor(ctx, client, work, err, "missing_spec", "oci")
	}

	result, runErr := runner.Run(ctx, work.Image, work.SpecJSON, gpus)
	if result == nil {
		return submitFailureReceiptForExecutor(ctx, client, work, runErr, "runner_failed", "oci")
	}

	resultHash := result.Hash
	metadata := receiptMetadataBase(work, map[string]any{
		"executor":    "oci",
		"duration_ms": result.Duration.Milliseconds(),
		"exit_code":   result.ExitCode,
		"stderr_tail": result.Logs,
		"metrics":     result.Metrics,
		"gpu_model":   caps.GPUModel,
	})
	if ctx.Err() != nil {
		metadata["status"] = "aborted"
		metadata["abort_reason"] = abortReason(ctx, runErr)
	}
	if strings.TrimSpace(result.OutputPath) != "" {
		uploadRes, uploadErr := blob.Upload(ctx, client, work.JobID, result.OutputPath)
		if uploadErr != nil {
			metadata["upload_error"] = uploadErr.Error()
			_ = os.Remove(result.OutputPath)
			return submitFailureReceiptForExecutor(ctx, client, work, uploadErr, "artifact_upload_failed", "oci")
		}
		if uploadRes == nil {
			uploadErr = fmt.Errorf("artifact upload returned no result")
			metadata["upload_error"] = uploadErr.Error()
			_ = os.Remove(result.OutputPath)
			return submitFailureReceiptForExecutor(ctx, client, work, uploadErr, "artifact_upload_failed", "oci")
		}
		if uploadRes != nil {
			metadata["blob_url"] = uploadRes.URL
			metadata["object_key"] = uploadRes.Key
			if strings.TrimSpace(uploadRes.Hash) != "" {
				metadata["artifact_sha256"] = uploadRes.Hash
				resultHash = uploadRes.Hash
			}
		}
		_ = os.Remove(result.OutputPath)
	}

	units := uint64(work.Units)
	if runErr != nil || result.ExitCode != 0 {
		units = 0
	}
	receipt := hub.Receipt{
		JobID:         work.JobID,
		ResultHashHex: resultHash,
		MeteringUnits: units,
		Metadata:      metadata,
	}
	if err := submitReceiptWithRetry(ctx, client, receipt); err != nil {
		return snapshotFromResult(result, resultHash, units, metadata), err
	}
	return snapshotFromResult(result, resultHash, units, metadata), runErr
}

func submitFailureReceipt(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, cause error, reason string) (*runnerResultSnapshot, error) {
	return submitFailureReceiptForExecutor(ctx, client, work, cause, reason, failureExecutor(work))
}

func submitFailureReceiptForExecutor(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, cause error, reason string, executor string) (*runnerResultSnapshot, error) {
	executor = strings.TrimSpace(executor)
	if executor == "" {
		executor = "unknown"
	}
	hash := sha256.Sum256([]byte(work.JobID + ":" + reason))
	metadata := receiptMetadataBase(work, map[string]any{
		"executor":  executor,
		"exit_code": 1,
		"status":    "failed",
		"reason":    reason,
	})
	if cause != nil {
		metadata["error"] = cause.Error()
	}
	receipt := hub.Receipt{
		JobID:         work.JobID,
		ResultHashHex: hex.EncodeToString(hash[:]),
		MeteringUnits: 0,
		Metadata:      llamacpp.SanitizeMetadata(metadata),
	}
	submitCtx := ctx
	var cancel context.CancelFunc
	if submitCtx == nil || submitCtx.Err() != nil {
		submitCtx, cancel = context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
	}
	if err := submitReceiptWithRetry(submitCtx, client, receipt); err != nil {
		return nil, err
	}
	return &runnerResultSnapshot{
		ResultHashHex: receipt.ResultHashHex,
		ExitCode:      1,
		MeteringUnits: 0,
		Metadata:      receipt.Metadata,
	}, cause
}

func failureExecutor(work *hub.WorkAssignment) string {
	if work == nil {
		return "unknown"
	}
	if work.RuntimeRequirements.NeedsLlamaCPP || llamacpp.ShouldHandle(work.Kind, work.ExecutorKind, work.SpecJSON) {
		return llamacpp.ExecutorKind
	}
	if strings.TrimSpace(work.Image) != "" ||
		strings.EqualFold(strings.TrimSpace(work.ExecutorKind), "managed_oci") ||
		work.RuntimeRequirements.NeedsManagedOCI ||
		work.RuntimeRequirements.NeedsManagedOCIGPU {
		return "oci"
	}
	if strings.TrimSpace(work.ExecutorKind) != "" {
		return strings.TrimSpace(work.ExecutorKind)
	}
	return "unknown"
}

func startAbortMonitor(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, cancel context.CancelFunc) func() {
	abortScopeID := strings.TrimSpace(work.WorkScopeID)
	if abortScopeID == "" || client == nil || cancel == nil {
		return func() {}
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		ticker := time.NewTicker(2 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				abort, err := client.FetchAbortSignal(ctx, abortScopeID)
				if err != nil {
					slog.Debug("abort signal check failed", "abort_scope_id", abortScopeID, "error", err)
					continue
				}
				if abort != nil {
					slog.Warn("abort signal received", "abort_scope_id", abortScopeID, "reason", abort.Reason)
					cancel()
					return
				}
			}
		}
	}()
	return func() {
		select {
		case <-done:
		case <-time.After(100 * time.Millisecond):
		}
	}
}

func submitReceiptWithRetry(ctx context.Context, client *hub.Client, receipt hub.Receipt) error {
	if client == nil {
		return fmt.Errorf("hub client required")
	}
	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		if err := client.SubmitReceipt(ctx, receipt); err == nil {
			return nil
		} else {
			lastErr = err
		}
		if !sleepOrDone(ctx, time.Duration(attempt+1)*time.Second) {
			break
		}
	}
	return lastErr
}

func receiptMetadataBase(work *hub.WorkAssignment, extras ...map[string]any) map[string]any {
	metadata := map[string]any{
		"node_agent_version": version,
	}
	if work != nil {
		metadata["job_id"] = work.JobID
		metadata["abort_scope_id"] = strings.TrimSpace(work.WorkScopeID)
		metadata["work_kind"] = strings.TrimSpace(work.Kind)
		metadata["assurance_class"] = strings.TrimSpace(work.AssuranceClass)
	}
	for _, extra := range extras {
		for key, value := range extra {
			if strings.TrimSpace(key) != "" && value != nil {
				metadata[key] = value
			}
		}
	}
	return metadata
}

func snapshotFromResult(result *runner.Result, hash string, units uint64, metadata map[string]any) *runnerResultSnapshot {
	if result == nil {
		return nil
	}
	return &runnerResultSnapshot{
		DurationMs:    result.Duration.Milliseconds(),
		ResultHashHex: hash,
		ExitCode:      result.ExitCode,
		MeteringUnits: units,
		BlobURL:       stringValue(metadata["blob_url"]),
		ObjectKey:     stringValue(metadata["object_key"]),
		Metadata:      metadata,
	}
}

func hubCapabilitiesFromCaps(caps hw.CapSet) hub.Capabilities {
	return hub.Capabilities{
		GPUModel:          caps.GPUModel,
		CPUModel:          caps.CPUModel,
		CPUCores:          caps.CPUCores,
		RAMBytes:          caps.RAMBytes,
		VRAMBytes:         caps.VRAMBytes,
		Sensors:           caps.Sensors,
		BandwidthMbps:     caps.BandwidthMbps,
		GeohashBucket:     caps.GeohashBucket,
		AttestationMethod: caps.Attestation,
		TEESupported:      caps.TEESupported,
		TEEType:           caps.TEEType,
	}
}

func resolveDeviceType(flagValue string, caps hw.CapSet) string {
	if v := strings.TrimSpace(flagValue); v != "" {
		return v
	}
	if strings.TrimSpace(caps.GPUModel) != "" || caps.VRAMBytes > 0 {
		return "gpu"
	}
	return "cpu"
}

func jobTimeout() time.Duration {
	if raw := strings.TrimSpace(os.Getenv("RYV_JOB_TIMEOUT")); raw != "" {
		if d, err := time.ParseDuration(raw); err == nil && d > 0 {
			return d
		}
	}
	return 2 * time.Hour
}

func abortReason(ctx context.Context, runErr error) string {
	if ctx != nil {
		if ctx.Err() != nil {
			return ctx.Err().Error()
		}
	}
	if runErr != nil {
		return runErr.Error()
	}
	return "abort_requested"
}

func sleepOrDone(ctx context.Context, d time.Duration) bool {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

func envFloat(name string) float64 {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return 0
	}
	out, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return 0
	}
	return out
}

func envInt(name string) int {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return 0
	}
	out, err := strconv.Atoi(value)
	if err != nil {
		return 0
	}
	return out
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			return value
		}
	}
	return ""
}

func sanitizeStatusToken(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	value = strings.ReplaceAll(value, ",", "_")
	value = strings.ReplaceAll(value, ":", "_")
	value = strings.ReplaceAll(value, " ", "_")
	if value == "" {
		return "default"
	}
	return value
}

func stringValue(value any) string {
	switch typed := value.(type) {
	case string:
		return typed
	default:
		return ""
	}
}
