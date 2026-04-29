package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"math"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/Ryvion/node-agent/internal/hub"
	"github.com/Ryvion/node-agent/internal/hw"
	"github.com/Ryvion/node-agent/internal/inference"
	"github.com/Ryvion/node-agent/internal/nodekey"
	"github.com/Ryvion/node-agent/internal/runtimeexec"
	"github.com/Ryvion/node-agent/internal/update"
)

// Set via -ldflags at build time.
var version = "dev"

// Package-level flags so runNode() and service handler can access them.
var (
	flagHub        string
	flagDevice     string
	flagCountry    string
	flagReferral   string
	flagGPUs       string
	flagUIPort     string
	flagMaxGPUUtil float64
)

// cachedGPUUtil stores the latest GPU utilization from heartbeat sampling.
// Used by the work loop to skip fetching work when GPU is busy.
var cachedGPUUtil atomic.Uint64 // stores float64 bits via math.Float64bits

// jobActive is set to 1 while a job is being processed.
// The update check reads this to avoid restarting during active work.
var jobActive atomic.Int32

// latestHubVersion stores the most recent agent version advertised by the hub.
// Written by heartbeat goroutine, read by work loop for auto-update checks.
var latestHubVersion atomic.Value // string

func main() {
	// Subcommand: ryvion-node claim <CODE>
	if len(os.Args) > 1 && os.Args[1] == "claim" {
		runClaim()
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "identity" {
		runIdentity()
		return
	}

	versionFlag := flag.Bool("version", false, "Print version and exit")
	flag.StringVar(&flagHub, "hub", "https://api.ryvion.ai", "Hub orchestrator base URL")
	flag.StringVar(&flagDevice, "type", "", "Node device type (gpu|cpu|mobile|iot)")
	flag.StringVar(&flagCountry, "country", "", "Declared ISO 3166-1 alpha-2 country code for sovereign routing")
	flag.StringVar(&flagReferral, "referral", "", "Optional referral code")
	flag.StringVar(&flagGPUs, "gpus", "auto", "Managed OCI GPU selection value (auto|all|none|device list)")
	flag.StringVar(&flagUIPort, "ui-port", defaultOperatorAPIPort, "Local operator API port (set 0 to disable)")
	flag.Float64Var(&flagMaxGPUUtil, "max-gpu-util", 90, "Skip jobs when GPU utilization exceeds this % (0=disabled)")
	flag.Parse()

	// Allow env override for max GPU util
	if v := strings.TrimSpace(os.Getenv("RYV_MAX_GPU_UTIL")); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			flagMaxGPUUtil = f
		}
	}

	if *versionFlag {
		fmt.Println("ryvion-node", version)
		os.Exit(0)
	}

	initLogger()

	// On Windows, if running as a service (session 0), use proper SCM integration.
	// This ensures the SCM can query status and send stop/shutdown commands.
	if isWindowsService() {
		slog.Info("starting as Windows service")
		runAsWindowsService()
		return
	}

	// Console mode — signal-based context.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	runNode(ctx)
}

// runNode contains all node logic. Called from console mode directly
// or from the Windows service handler with a cancellable context.
func runNode(ctx context.Context) {
	ensureServiceRecovery()
	cleanupOrphanedContainers()

	hubURL := strings.TrimSpace(flagHub)
	if envHub := strings.TrimSpace(os.Getenv("RYV_HUB_URL")); envHub != "" {
		hubURL = envHub
	}
	if hubURL == "" {
		slog.Error("hub URL is required")
		return
	}

	pub, priv, err := nodekey.LoadOrCreate(strings.TrimSpace(os.Getenv("RYV_KEY_PATH")))
	if err != nil {
		slog.Error("failed to load node key", "error", err)
		return
	}

	client := hub.New(
		hubURL,
		pub,
		priv,
		hub.WithBindToken(os.Getenv("RYV_BIND_TOKEN")),
		hub.WithWallet(os.Getenv("RYV_WALLET")),
		hub.WithAdminKey(os.Getenv("RYV_ADMIN_KEY")),
		hub.WithUserAgent("ryvion-node/"+version),
	)
	runtimeContract, err := resolveRuntimeContractMetadata(version)
	if err != nil {
		slog.Warn("failed to load runtime contract metadata, falling back to local defaults", "error", err)
	}
	runtimeMgr := newRuntimeManager(version, runtimeContract)
	if err := syncManagedRuntimeFromHub(ctx, hubURL, runtimeMgr); err != nil {
		slog.Warn("managed runtime auto-sync failed; continuing with current runtime", "error", err)
	}
	if err := ensureUserImageRuntimeHelper(); err != nil {
		slog.Warn("image runtime helper bootstrap failed; local image jobs may stay unavailable", "error", err)
	}

	caps := hw.DetectCaps(flagDevice)
	deviceType := resolveDeviceType(flagDevice, caps)
	declaredCountry, err := resolveInitialDeclaredCountry(flagCountry)
	if err != nil {
		slog.Warn("failed to load declared country preference, defaulting to flag/env value", "error", err)
		declaredCountry = strings.TrimSpace(flagCountry)
		if envCountry := strings.TrimSpace(os.Getenv("RYV_DECLARED_COUNTRY")); envCountry != "" {
			declaredCountry = envCountry
		}
	}
	publicAIOptIn, err := resolveInitialPublicAIOptIn()
	if err != nil {
		slog.Warn("failed to load operator preferences, defaulting public participation to off", "error", err)
	}

	operatorRuntimeState = newOperatorRuntime(version, hubURL, deviceType, declaredCountry, publicAIOptIn, caps, client)
	operatorRuntimeState.setRuntimeManager(runtimeMgr)
	startOperatorAPIServer(ctx, operatorRuntimeState, operatorAPIPort(flagUIPort))

	// Retry registration with backoff — on Windows the service starts before
	// Windows service startup can race the managed runtime, WSL2, and network.
	// Keep
	// the process alive so SCM doesn't exhaust its restart budget.
	regBackoff := 5 * time.Second
	for {
		if err := client.Register(ctx, hub.Capabilities{
			GPUModel:          caps.GPUModel,
			CPUCores:          caps.CPUCores,
			RAMBytes:          caps.RAMBytes,
			VRAMBytes:         caps.VRAMBytes,
			Sensors:           caps.Sensors,
			BandwidthMbps:     caps.BandwidthMbps,
			GeohashBucket:     caps.GeohashBucket,
			AttestationMethod: caps.Attestation,
			TEESupported:      caps.TEESupported,
			TEEType:           caps.TEEType,
		}, deviceType, strings.TrimSpace(flagReferral), declaredCountry); err != nil {
			if operatorRuntimeState != nil {
				operatorRuntimeState.setRegistered(false, err)
			}
			slog.Warn("register failed, retrying", "error", err, "retry_in", regBackoff)
			select {
			case <-ctx.Done():
				return
			case <-time.After(regBackoff):
			}
			if regBackoff < 2*time.Minute {
				regBackoff = time.Duration(float64(regBackoff) * 1.5)
			}
			continue
		}
		if operatorRuntimeState != nil {
			operatorRuntimeState.setRegistered(true, nil)
		}
		break
	}
	bindToken := strings.TrimSpace(os.Getenv("RYV_BIND_TOKEN"))
	slog.Info("register succeeded", "hub", hubURL, "device_type", deviceType, "pubkey", client.PublicKeyHex(),
		"bind_token", redact(bindToken))
	if flagMaxGPUUtil > 0 && flagMaxGPUUtil < 100 {
		slog.Info("GPU utilization cap enabled", "max_gpu_util", flagMaxGPUUtil)
	}

	if caps.TEESupported {
		slog.Info("attempting TEE attestation", "tee_type", caps.TEEType)
		if err := client.Attest(ctx, caps); err != nil {
			slog.Warn("TEE attestation failed", "error", err)
		} else {
			slog.Info("TEE attestation verified", "tee_type", caps.TEEType)
		}
	}

	if err := client.SolveChallenge(ctx); err != nil {
		slog.Warn("challenge solve failed", "error", err)
	}

	// Start persistent inference manager
	dataDir := strings.TrimSpace(os.Getenv("RYV_DATA_DIR"))
	infMgr := inference.New(dataDir)
	infMgr.SetHubAuth(hubURL, client.NodeAuthToken)
	go func() {
		defer func() {
			if r := recover(); r != nil {
				slog.Error("inference manager panic", "error", r)
			}
		}()
		if err := infMgr.Start(ctx); err != nil && ctx.Err() == nil {
			slog.Error("inference manager stopped", "error", err)
		}
	}()
	defer infMgr.Stop()
	if operatorRuntimeState != nil {
		operatorRuntimeState.setInferenceManager(infMgr)
	}
	startUserImageRuntimePrewarm(ctx, caps, detectAvailableDiskGB(), strings.TrimSpace(caps.GPUModel) != "")

	// Health report loop keeps scheduler-facing capability flags up to date
	// (for example native inference readiness).
	go func() {
		defer func() {
			if r := recover(); r != nil {
				slog.Error("health report goroutine panic", "error", r)
			}
		}()
		healthReportLoop(ctx, client, caps, infMgr, runtimeMgr)
	}()

	// Independent heartbeat goroutine — keeps node "online" regardless of
	// what the work loop is doing (long-poll, job execution, etc.).
	go func() {
		defer func() {
			if r := recover(); r != nil {
				slog.Error("heartbeat goroutine panic", "error", r)
			}
		}()
		heartbeatLoop(ctx, client)
	}()

	// Work loop — fetch and process jobs.
	workLoop(ctx, client, flagGPUs, hubURL, version, infMgr, runtimeMgr, strings.TrimSpace(caps.GPUModel) != "")
}

// heartbeatLoop sends heartbeats on a fixed interval, completely independent
// of the work loop. Implements a circuit breaker: after 30 consecutive failures
// (~5 min at 10s), the interval increases to 60s with a warning. Resets on success.
func heartbeatLoop(ctx context.Context, client *hub.Client) {
	const (
		normalInterval    = 30 * time.Second
		backoffInterval   = 60 * time.Second
		circuitBreakerMax = 30
	)

	ticker := time.NewTicker(normalInterval)
	defer ticker.Stop()

	var consecutiveFailures int

	// Send first heartbeat immediately.
	if sendHeartbeat(ctx, client) {
		consecutiveFailures = 0
	} else {
		consecutiveFailures++
	}

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if sendHeartbeat(ctx, client) {
				if consecutiveFailures >= circuitBreakerMax {
					slog.Info("hub heartbeat recovered after circuit breaker", "prev_failures", consecutiveFailures)
					ticker.Reset(normalInterval)
				}
				consecutiveFailures = 0
			} else {
				consecutiveFailures++
				if consecutiveFailures == circuitBreakerMax {
					slog.Warn("hub heartbeat circuit breaker tripped — backing off to 60s interval",
						"consecutive_failures", consecutiveFailures)
					ticker.Reset(backoffInterval)
				}
			}
		}
	}
}

func sendHeartbeat(ctx context.Context, client *hub.Client) bool {
	metrics := hw.SampleMetrics()

	// Cache GPU utilization for the work loop's throttle check.
	cachedGPUUtil.Store(math.Float64bits(metrics.GPUUtil))

	// Report whether the node is self-throttling due to operator GPU usage.
	throttled := flagMaxGPUUtil > 0 && flagMaxGPUUtil < 100 && metrics.GPUUtil > flagMaxGPUUtil

	heartbeat, err := client.Heartbeat(ctx, hub.Metrics{
		TimestampMs:  time.Now().UnixMilli(),
		CPUUtil:      metrics.CPUUtil,
		MemUtil:      metrics.MemUtil,
		GPUUtil:      metrics.GPUUtil,
		PowerWatts:   metrics.PowerWatts,
		GPUThrottled: throttled,
	})
	if err != nil {
		if operatorRuntimeState != nil {
			operatorRuntimeState.recordHeartbeat(metrics, hub.HeartbeatResponse{}, err)
		}
		slog.Warn("heartbeat failed", "error", err)
		return false
	}
	if operatorRuntimeState != nil {
		operatorRuntimeState.recordHeartbeat(metrics, heartbeat, nil)
	}
	// Store latest version for work loop update checks.
	if heartbeat.LatestVersion != "" {
		latestHubVersion.Store(heartbeat.LatestVersion)
	}
	return true
}

func healthReportLoop(ctx context.Context, client *hub.Client, caps hw.CapSet, infMgr *inference.Manager, runtimeMgr *runtimeManager) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	send := func() {
		report := buildHealthReport(caps, infMgr, runtimeMgr)
		if operatorRuntimeState != nil {
			operatorRuntimeState.recordHealthReport(report)
		}
		if err := client.SendHealthReport(ctx, report); err != nil {
			slog.Warn("health report failed", "error", err)
		}
	}

	// Initial report at startup.
	send()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			send()
		}
	}
}

// workLoop fetches and processes jobs. Heartbeats are handled separately.
func workLoop(ctx context.Context, client *hub.Client, gpus, hubURL, currentVersion string, infMgr *inference.Manager, runtimeMgr *runtimeManager, gpuDetected bool) {
	var lastUpdateAttempt time.Time
	backoff := 5 * time.Second
	maxBackoff := 2 * time.Minute

	for {
		// Check for version updates (read from atomic, never missed).
		if v, ok := latestHubVersion.Load().(string); ok && v != "" {
			if update.NeedsUpdate(currentVersion, v) && time.Since(lastUpdateAttempt) > 5*time.Minute {
				if jobActive.Load() != 0 {
					slog.Info("update available but job in progress, deferring", "latest", v)
				} else {
					lastUpdateAttempt = time.Now()
					slog.Info("update available", "current", currentVersion, "latest", v)
					if err := update.Apply(ctx, hubURL); err != nil {
						slog.Warn("auto-update failed", "error", err)
					} else {
						slog.Info("update applied, restarting")
						if restartErr := update.Restart(); restartErr != nil {
							slog.Warn("restart failed — update will take effect on next manual restart", "error", restartErr)
						}
					}
				}
			}
		}

		// GPU-aware scheduling: skip work fetch when operator's GPU is busy.
		if flagMaxGPUUtil > 0 && flagMaxGPUUtil < 100 {
			gpuUtil := math.Float64frombits(cachedGPUUtil.Load())
			if gpuUtil > flagMaxGPUUtil {
				slog.Debug("GPU busy, skipping work fetch", "gpu_util", gpuUtil, "max_gpu_util", flagMaxGPUUtil)
				select {
				case <-ctx.Done():
					return
				case <-time.After(5 * time.Second):
				}
				continue
			}
		}

		work, err := client.FetchWork(ctx)
		if err != nil {
			slog.Warn("fetch work failed", "error", err)
			select {
			case <-ctx.Done():
				return
			case <-time.After(backoff):
			}
			backoff = time.Duration(float64(backoff) * 1.5)
			if backoff > maxBackoff {
				backoff = maxBackoff
			}
			continue
		}

		// Successful fetch — reset backoff.
		backoff = 5 * time.Second

		if work == nil {
			continue
		}

		jobActive.Store(1)
		processWork(ctx, client, work, gpus, infMgr, runtimeMgr, gpuDetected)
		jobActive.Store(0)
	}
}

func processWork(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, gpus string, infMgr *inference.Manager, runtimeMgr *runtimeManager, gpuDetected bool) {
	if operatorRuntimeState != nil {
		operatorRuntimeState.startJob(work)
	}

	// Determine job timeout based on type or explicit env var
	executorKind := executorKindForAssignment(work)
	isStreaming := executorKind == executorKindNativeStreaming
	isNativeReport := executorKind == executorKindNativeReport
	isRyvionRuntime := executorKind == executorKindRyvionRuntime
	isTraining := work.Kind == "training"
	isAgentHosting := executorKind == executorKindAgentHosting
	jobTimeout := 10 * time.Minute
	if isStreaming {
		jobTimeout = 30 * time.Minute // Streaming inference often takes much longer context generation
	}
	if isNativeReport {
		jobTimeout = 15 * time.Minute
	}
	if isRyvionRuntime {
		jobTimeout = 2 * time.Hour
	}
	if isTraining {
		jobTimeout = 4 * time.Hour // Training/fine-tuning jobs can take hours
	}
	if isAgentHosting {
		jobTimeout = 720 * time.Hour // 30 days max — agents run until stopped by hub
	}
	if v := strings.TrimSpace(os.Getenv("RYV_JOB_TIMEOUT")); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d > 0 {
			jobTimeout = d
		}
	}

	runCtx, cancel := context.WithTimeout(ctx, jobTimeout)
	defer cancel()

	// Pre-job VRAM check — reject if GPU is too busy
	if isStreaming {
		freeVRAM := hw.GetFreeVRAM()
		if freeVRAM > 0 && freeVRAM < 2*1024*1024*1024 { // Less than 2GB free
			slog.Warn("insufficient free VRAM, rejecting job", "free_vram_mb", freeVRAM/(1024*1024), "job_id", work.JobID)
			relayStreamingFailure(runCtx, client, work.JobID, fmt.Errorf("insufficient VRAM: %d MB free", freeVRAM/(1024*1024)))
			if operatorRuntimeState != nil {
				operatorRuntimeState.finishJob(work, nil, fmt.Errorf("insufficient VRAM"))
			}
			return
		}
	}

	engine := selectExecutionEngine(work)
	slog.Info("dispatching work", "job_id", work.JobID, "executor_kind", engine.Kind(), "assurance_class", assuranceClassForAssignment(work))
	result, runErr := engine.Execute(runCtx, work, executionContext{
		client:         client,
		gpus:           gpus,
		infMgr:         infMgr,
		runtimeManager: runtimeMgr,
		gpuDetected:    gpuDetected,
	})
	if runErr != nil {
		if engine.Kind() == executorKindManagedOCI && result != nil && managedOCIRuntimeUnavailableError(runErr, stringValue(result.Metadata["stderr_tail"])) {
			reportManagedOCIRuntimeDegraded(client, infMgr, runtimeMgr)
		}
		slog.Warn("job execution failed", "job_id", work.JobID, "executor_kind", engine.Kind(), "error", runErr)
		if operatorRuntimeState != nil {
			operatorRuntimeState.finishJob(work, result, runErr)
		}
		return
	}
	if result != nil {
		slog.Info("job completed", "job_id", work.JobID, "executor_kind", engine.Kind(), "hash", result.ResultHashHex, "units", result.MeteringUnits)
	}
	if operatorRuntimeState != nil {
		operatorRuntimeState.finishJob(work, result, nil)
	}
}

// relayStreamingFailure sends a terminal SSE error chunk to hub-orch so the
// buyer stream exits quickly instead of hanging until server timeout.
func relayStreamingFailure(ctx context.Context, client *hub.Client, jobID string, runErr error) {
	if client == nil {
		return
	}
	jobID = strings.TrimSpace(jobID)
	if jobID == "" {
		return
	}

	msg := "streaming inference failed"
	if runErr != nil {
		if s := strings.TrimSpace(runErr.Error()); s != "" {
			msg = s
		}
	}
	if len(msg) > 512 {
		msg = msg[:512]
	}

	payloadJSON, err := json.Marshal(map[string]any{
		"error": map[string]string{
			"message": msg,
			"type":    "node_error",
		},
	})
	if err != nil {
		slog.Warn("failed to encode streaming error payload", "job_id", jobID, "error", err)
		payloadJSON = []byte(`{"error":{"message":"streaming inference failed","type":"node_error"}}`)
	}

	relayCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	payload := "data: " + string(payloadJSON) + "\n\n" + "data: [DONE]\n\n"
	if err := client.StreamInference(relayCtx, jobID, strings.NewReader(payload)); err != nil {
		slog.Warn("failed to relay streaming failure to hub", "job_id", jobID, "error", err)
	}
}

// submitReceiptWithRetry attempts receipt submission with exponential backoff.
// Receipts represent completed work — losing one means the operator doesn't get paid.
func submitReceiptWithRetry(ctx context.Context, client *hub.Client, receipt hub.Receipt) error {
	const maxAttempts = 5
	delay := 2 * time.Second
	var lastErr error
	for i := 0; i < maxAttempts; i++ {
		if err := client.SubmitReceipt(ctx, receipt); err != nil {
			lastErr = err
			slog.Warn("receipt submission attempt failed", "job_id", receipt.JobID, "attempt", i+1, "error", err)
			select {
			case <-ctx.Done():
				return fmt.Errorf("context cancelled during receipt retry: %w", lastErr)
			case <-time.After(delay):
			}
			delay = time.Duration(float64(delay) * 2)
			if delay > 30*time.Second {
				delay = 30 * time.Second
			}
			continue
		}
		return nil
	}
	return fmt.Errorf("receipt submission failed after %d attempts: %w", maxAttempts, lastErr)
}

func isAgentHostingTask(specJSON string) bool {
	var spec struct {
		Task string `json:"task"`
	}
	if json.Unmarshal([]byte(specJSON), &spec) != nil {
		return false
	}
	return spec.Task == "agent_hosting"
}

func isWorkCapsuleTask(specJSON string) bool {
	var spec struct {
		Task     string `json:"task"`
		WorkType string `json:"work_type"`
	}
	if json.Unmarshal([]byte(specJSON), &spec) != nil {
		return false
	}
	return spec.Task == executorKindWorkCapsule || spec.WorkType == "certified_change"
}

func isRyvionRuntimeTask(specJSON string) bool {
	var spec struct {
		ExecutorKind string `json:"executor_kind"`
		RuntimeTask  string `json:"runtime_task"`
	}
	if json.Unmarshal([]byte(specJSON), &spec) != nil {
		return false
	}
	return strings.EqualFold(strings.TrimSpace(spec.ExecutorKind), executorKindRyvionRuntime) ||
		strings.TrimSpace(spec.RuntimeTask) != ""
}

func extractDeploymentID(specJSON string) string {
	var spec struct {
		DeploymentID string `json:"deployment_id"`
	}
	if json.Unmarshal([]byte(specJSON), &spec) != nil {
		return ""
	}
	return spec.DeploymentID
}

func cleanupOrphanedContainers() {
	executor, err := runtimeexec.ResolveExecutor(runtime.GOOS, os.Getenv)
	if err != nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()
	psArgs := append([]string{}, executor.PrefixArgs...)
	psArgs = append(psArgs, "ps", "-q", "--filter", "name=ryv_")
	out, err := exec.CommandContext(ctx, executor.Command, psArgs...).CombinedOutput()
	if err != nil {
		return
	}
	ids := strings.TrimSpace(string(out))
	if ids == "" {
		return
	}
	for _, id := range strings.Split(ids, "\n") {
		id = strings.TrimSpace(id)
		if id != "" {
			slog.Info("killing orphaned container", "id", id)
			killArgs := append([]string{}, executor.PrefixArgs...)
			killArgs = append(killArgs, "kill", id)
			exec.Command(executor.Command, killArgs...).Run()
			rmArgs := append([]string{}, executor.PrefixArgs...)
			rmArgs = append(rmArgs, "rm", "-f", id)
			exec.Command(executor.Command, rmArgs...).Run()
		}
	}
}

func buildHealthReport(caps hw.CapSet, infMgr *inference.Manager, runtimeMgr *runtimeManager) hub.HealthReport {
	gpuReady := strings.TrimSpace(caps.GPUModel) != ""
	parts := []string{}
	nativeSupported := inference.NativeRuntimeAvailable()
	nativeReady := nativeSupported && infMgr != nil && infMgr.Healthy()
	publicAIReady := publicAIOptInEnabled()
	publicInferenceReady := publicAIReady && nativeReady
	diskGB := detectAvailableDiskGB()
	ffmpegOK := commandExists("ffmpeg")
	pdalOK := commandExists("pdal")
	open3dOK := commandExists("open3d") || pythonModuleAvailable("open3d")
	gitOK := commandExists("git")
	nodeOK := commandExists("node")
	playwrightOK := commandExists("playwright") || commandExists("npx")
	codexOK := commandExists("codex")
	claudeOK := commandExists("claude") || commandExists("claude-code")
	geminiOK := commandExists("gemini") || commandExists("gemini-cli")
	runtimeTokens := runtimeMgr.StatusTokens(gpuReady)
	runtimeSnap := runtimeMgr.Snapshot(gpuReady)
	localFluxReady := publicAIReady && localFlux2KleinReady(caps, diskGB, gpuReady)
	localFluxFastReady := localFluxReady && localFlux2KleinFastGPUEligible(caps, gpuReady)
	localFluxPreparing := publicAIReady && localFlux2KleinFastGPUEligible(caps, gpuReady) && localFlux2KleinPreparing(caps, diskGB, gpuReady)
	localFluxPrepareEligible := publicAIReady && localFlux2KleinPrepareEligible(caps, diskGB, gpuReady)

	if gpuReady {
		parts = append(parts, "gpu-detect:ok")
	} else {
		parts = append(parts, "gpu-detect:missing")
	}

	parts = append(parts, runtimeTokens...)

	if gpuReady {
		parts = append(parts, "gpu_model:"+caps.GPUModel)
	}
	if caps.GfxVersion != "" {
		parts = append(parts, "gfx_version:"+caps.GfxVersion)
	}
	parts = append(parts, "disk_gb:"+strconv.FormatUint(diskGB, 10))
	if ffmpegOK {
		parts = append(parts, "tool:ffmpeg")
	}
	if pdalOK {
		parts = append(parts, "tool:pdal")
	}
	if open3dOK {
		parts = append(parts, "tool:open3d")
	}
	if gitOK {
		parts = append(parts, "tool:git")
	}
	if nodeOK {
		parts = append(parts, "tool:node")
	}
	if playwrightOK {
		parts = append(parts, "tool:playwright")
	}
	if codexOK {
		parts = append(parts, "tool:codex")
	}
	if claudeOK {
		parts = append(parts, "tool:claude-code")
	}
	if geminiOK {
		parts = append(parts, "tool:gemini-cli")
	}
	spatialReady := ffmpegOK && (pdalOK || open3dOK) && diskGB >= 50 && (gpuReady || caps.CPUCores >= 8)
	if spatialReady {
		parts = append(parts, "spatial-ready:1")
	} else {
		parts = append(parts, "spatial-ready:0")
	}
	if nativeSupported {
		parts = append(parts, "native-inference:supported")
	} else {
		parts = append(parts, "native-inference:unsupported")
	}
	parts = append(parts, boolStatusToken("cap:native_streaming", nativeReady))
	parts = append(parts, "cap:native_report:1")
	if publicAIReady {
		parts = append(parts, "public-ai-ready:1")
	} else {
		parts = append(parts, "public-ai-ready:0")
	}
	parts = append(parts, boolStatusToken("cap:image_gen", publicAIReady && (runtimeSnap.GPUReady || localFluxFastReady)))
	parts = append(parts, boolStatusToken("cap:ryvion_runtime", localFluxFastReady))
	if localFluxPreparing {
		parts = append(parts, "runtime:image:"+flux2Klein4BLocalModel+":preparing:1")
	}
	if localFluxFastReady {
		parts = append(parts, "runtime:image:"+flux2Klein4BLocalModel)
		parts = append(parts, "model:"+flux2Klein4BLocalModel)
		parts = append(parts, fmt.Sprintf("runtime:image:%s:min_vram_mb:%d", flux2Klein4BLocalModel, flux2Klein4BMinVRAMMB))
	} else if localFluxPrepareEligible {
		parts = append(parts, "runtime:image:"+flux2Klein4BLocalModel+":eligible:1")
		parts = append(parts, fmt.Sprintf("runtime:image:%s:min_vram_mb:%d", flux2Klein4BLocalModel, flux2Klein4BMinVRAMMB))
	} else if localFluxReady {
		parts = append(parts, "runtime:image:"+flux2Klein4BLocalModel+":mode:cpu-preview")
		parts = append(parts, fmt.Sprintf("runtime:image:%s:min_ram_gb:%d", flux2Klein4BLocalModel, flux2Klein4BMinRAMGB))
		parts = append(parts, fmt.Sprintf("runtime:image:%s:min_cpu_cores:%d", flux2Klein4BLocalModel, flux2Klein4BMinCPUCores))
	}
	if nativeReady {
		parts = append(parts, "native-inference-ready:1")
		parts = append(parts, "native-model:"+infMgr.ModelName())
	} else {
		parts = append(parts, "native-inference-ready:0")
	}
	if publicInferenceReady {
		parts = append(parts, "public-inference-ready:1")
		for _, modelID := range inference.SupportedNativeChatModels(caps.VRAMBytes) {
			parts = append(parts, "model:"+modelID)
		}
	} else {
		parts = append(parts, "public-inference-ready:0")
	}

	return hub.HealthReport{
		TimestampMs: time.Now().UnixMilli(),
		GPUReady:    gpuReady,
		RuntimeGPU:  runtimeSnap.GPUReady,
		Message:     strings.Join(parts, ","),
	}
}

func managedOCIRuntimeUnavailableError(runErr error, logs string) bool {
	text := strings.ToLower(strings.TrimSpace(logs))
	if runErr != nil {
		text = text + "\n" + strings.ToLower(runErr.Error())
	}
	for _, needle := range []string{
		"failed to connect to the docker api",
		"cannot connect to the docker daemon",
		"docker daemon is not running",
		"error during connect",
		"pipe/docker_engine",
	} {
		if strings.Contains(text, needle) {
			return true
		}
	}
	return false
}

func reportManagedOCIRuntimeDegraded(client *hub.Client, infMgr *inference.Manager, runtimeMgr *runtimeManager) {
	if client == nil {
		return
	}
	caps := hw.CapSet{}
	if operatorRuntimeState != nil {
		caps = operatorRuntimeState.caps
	}
	report := buildHealthReport(caps, infMgr, runtimeMgr)
	if !strings.Contains(strings.ToLower(report.Message), "runtime-ready:0") {
		return
	}
	if operatorRuntimeState != nil {
		operatorRuntimeState.recordHealthReport(report)
	}
	reportCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := client.SendHealthReport(reportCtx, report); err != nil {
		slog.Warn("immediate managed OCI runtime downgrade failed", "error", err)
		return
	}
	slog.Warn("reported degraded managed OCI runtime health")
}

func publicAIOptInEnabled() bool {
	if operatorRuntimeState != nil {
		return operatorRuntimeState.publicAIOptInEnabled()
	}
	enabled, err := resolveInitialPublicAIOptIn()
	if err != nil {
		return false
	}
	return enabled
}

func detectManagedOCIBackendWithProbes(gpuDetected bool, resolve func() string, readyCheck func(string) bool, gpuCheck func(string) bool) (bool, bool, bool) {
	backendBin := strings.TrimSpace(resolve())
	if backendBin == "" {
		return false, false, false
	}

	runtimeReady := readyCheck(backendBin)
	if !runtimeReady {
		return true, false, false
	}

	if !gpuDetected {
		return true, true, false
	}

	return true, true, gpuCheck(backendBin)
}

func detectAvailableDiskGB() uint64 {
	if runtime.GOOS == "windows" {
		wmic := resolveWindowsSystemTool("wmic")
		if wmic == "" {
			return 0
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		out, err := exec.CommandContext(ctx, wmic, "logicaldisk", "where", "DeviceID='C:'", "get", "FreeSpace", "/value").CombinedOutput()
		if err != nil {
			return 0
		}
		for _, line := range strings.Split(string(out), "\n") {
			line = strings.TrimSpace(line)
			if !strings.HasPrefix(strings.ToLower(line), "freespace=") {
				continue
			}
			v := strings.TrimSpace(strings.TrimPrefix(line, "FreeSpace="))
			if bytes, err := strconv.ParseUint(v, 10, 64); err == nil {
				return bytes / (1024 * 1024 * 1024)
			}
		}
		return 0
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "sh", "-lc", "df -k . | tail -1 | awk '{print $4}'").CombinedOutput()
	if err != nil {
		return 0
	}
	kbRaw := strings.TrimSpace(string(out))
	kb, err := strconv.ParseUint(kbRaw, 10, 64)
	if err != nil {
		return 0
	}
	return kb / (1024 * 1024)
}

func pythonModuleAvailable(module string) bool {
	module = strings.TrimSpace(module)
	if module == "" {
		return false
	}
	py := "python3"
	if runtime.GOOS == "windows" {
		py = "python"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, py, "-c", "import "+module)
	return cmd.Run() == nil
}

func testOCIBackendReady(backendBin string) bool {
	if strings.TrimSpace(backendBin) == "" {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, backendBin, "version", "--format", "{{.Server.Version}}").CombinedOutput()
	if err != nil {
		slog.Debug("managed OCI backend health check failed", "error", err, "output", strings.TrimSpace(string(out)))
		return false
	}
	return strings.TrimSpace(string(out)) != ""
}

// testManagedOCIGPU checks if the current OCI backend can access the GPU by
// running a minimal container. It tries NVIDIA first, then ROCm for AMD.
func testManagedOCIGPU(backendBin string) bool {
	if strings.TrimSpace(backendBin) == "" {
		return false
	}
	if testManagedOCIGPUNvidia(backendBin) {
		return true
	}
	return testManagedOCIGPURocm(backendBin)
}

func testManagedOCIGPUNvidia(backendBin string) bool {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, backendBin, "run", "--rm", "--gpus", "all",
		"nvidia/cuda:12.4.1-base-ubuntu22.04", "nvidia-smi", "--query-gpu=name", "--format=csv,noheader").CombinedOutput()
	if err != nil {
		slog.Debug("managed OCI NVIDIA GPU test failed", "error", err, "output", strings.TrimSpace(string(out)))
		return false
	}
	result := strings.TrimSpace(string(out))
	slog.Info("managed OCI NVIDIA GPU test passed", "gpu", result)
	return result != ""
}

func testManagedOCIGPURocm(backendBin string) bool {
	// Check if ROCm devices exist before pulling a container image.
	if _, err := os.Stat("/dev/kfd"); err != nil {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, backendBin, "run", "--rm",
		"--device=/dev/kfd", "--device=/dev/dri",
		"rocm/rocm-terminal:latest", "rocm-smi", "--showproductname").CombinedOutput()
	if err != nil {
		slog.Debug("managed OCI ROCm GPU test failed", "error", err, "output", strings.TrimSpace(string(out)))
		return false
	}
	result := strings.TrimSpace(string(out))
	slog.Info("managed OCI ROCm GPU test passed", "output", result)
	return result != ""
}

func resolveOCIBackendCLI() string {
	backend, err := runtimeexec.ResolveBackendCommand(runtime.GOOS, os.Getenv)
	if err != nil {
		return ""
	}
	return backend
}

func resolveWindowsSystemTool(name string) string {
	if p, err := exec.LookPath(name); err == nil {
		return p
	}
	if runtime.GOOS != "windows" {
		return ""
	}
	candidates := []string{
		filepath.Join(os.Getenv("SystemRoot"), "System32", name+".exe"),
		filepath.Join(os.Getenv("WINDIR"), "System32", name+".exe"),
	}
	for _, candidate := range candidates {
		if candidate == "" {
			continue
		}
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
	}
	return ""
}

func resolveDeviceType(raw string, caps hw.CapSet) string {
	if v := strings.TrimSpace(raw); v != "" {
		return strings.ToLower(v)
	}
	if strings.TrimSpace(caps.GPUModel) != "" {
		return "gpu"
	}
	if _, err := os.Stat("/system/build.prop"); err == nil {
		return "mobile"
	}
	return "cpu"
}

func commandExists(name string) bool {
	_, err := exec.LookPath(name)
	return err == nil
}

// ensureServiceRecovery configures Windows SCM failure-recovery so the
// service auto-restarts after crashes and auto-updates. This is idempotent
// and fixes nodes installed with older installers that lack the config.
func ensureServiceRecovery() {
	if runtime.GOOS != "windows" {
		return
	}
	_ = exec.Command("sc.exe", "failure", "RyvionNode", "reset=", "86400",
		"actions=", "restart/5000/restart/10000/restart/30000").Run()
	_ = exec.Command("sc.exe", "failureflag", "RyvionNode", "1").Run()
}

func initLogger() {
	level := slog.LevelInfo
	switch strings.ToLower(strings.TrimSpace(os.Getenv("RYV_LOG_LEVEL"))) {
	case "debug":
		level = slog.LevelDebug
	case "warn", "warning":
		level = slog.LevelWarn
	case "error":
		level = slog.LevelError
	}
	writer := io.Writer(os.Stdout)
	if operatorLogBuffer != nil {
		writer = io.MultiWriter(os.Stdout, operatorLogBuffer)
	}
	logger := slog.New(slog.NewJSONHandler(writer, &slog.HandlerOptions{Level: level}))
	slog.SetDefault(logger)
}

func redact(s string) string {
	if len(s) <= 8 {
		return "***"
	}
	return s[:4] + "..." + s[len(s)-4:]
}

func stringValue(v any) string {
	s, _ := v.(string)
	return strings.TrimSpace(s)
}
