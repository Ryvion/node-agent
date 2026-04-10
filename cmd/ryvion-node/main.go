package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
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

	"github.com/Ryvion/node-agent/internal/blob"
	"github.com/Ryvion/node-agent/internal/hub"
	"github.com/Ryvion/node-agent/internal/hw"
	"github.com/Ryvion/node-agent/internal/inference"
	"github.com/Ryvion/node-agent/internal/nodekey"
	"github.com/Ryvion/node-agent/internal/runner"
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

	versionFlag := flag.Bool("version", false, "Print version and exit")
	flag.StringVar(&flagHub, "hub", "https://api.ryvion.ai", "Hub orchestrator base URL")
	flag.StringVar(&flagDevice, "type", "", "Node device type (gpu|cpu|mobile|iot)")
	flag.StringVar(&flagCountry, "country", "", "Declared ISO 3166-1 alpha-2 country code for sovereign routing")
	flag.StringVar(&flagReferral, "referral", "", "Optional referral code")
	flag.StringVar(&flagGPUs, "gpus", "auto", "Docker --gpus value (auto|all|none|device list)")
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
	ensureDockerGPURuntime()

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

	caps := hw.DetectCaps(flagDevice)
	deviceType := resolveDeviceType(flagDevice, caps)
	declaredCountry := strings.TrimSpace(flagCountry)
	if envCountry := strings.TrimSpace(os.Getenv("RYV_DECLARED_COUNTRY")); envCountry != "" {
		declaredCountry = envCountry
	}

	operatorRuntimeState = newOperatorRuntime(version, hubURL, deviceType, declaredCountry, caps, client)
	startOperatorAPIServer(ctx, operatorRuntimeState, operatorAPIPort(flagUIPort))

	// Retry registration with backoff — on Windows the service starts before
	// Docker/WSL2/network are ready, so the first attempts will fail.  Keep
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

	// Health report loop keeps scheduler-facing capability flags up to date
	// (for example native inference readiness).
	go func() {
		defer func() {
			if r := recover(); r != nil {
				slog.Error("health report goroutine panic", "error", r)
			}
		}()
		healthReportLoop(ctx, client, caps, infMgr)
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
	workLoop(ctx, client, flagGPUs, hubURL, version, infMgr)
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

	latest, err := client.Heartbeat(ctx, hub.Metrics{
		TimestampMs:  time.Now().UnixMilli(),
		CPUUtil:      metrics.CPUUtil,
		MemUtil:      metrics.MemUtil,
		GPUUtil:      metrics.GPUUtil,
		PowerWatts:   metrics.PowerWatts,
		GPUThrottled: throttled,
	})
	if err != nil {
		if operatorRuntimeState != nil {
			operatorRuntimeState.recordHeartbeat(metrics, "", err)
		}
		slog.Warn("heartbeat failed", "error", err)
		return false
	}
	if operatorRuntimeState != nil {
		operatorRuntimeState.recordHeartbeat(metrics, latest, nil)
	}
	// Store latest version for work loop update checks.
	if latest != "" {
		latestHubVersion.Store(latest)
	}
	return true
}

func healthReportLoop(ctx context.Context, client *hub.Client, caps hw.CapSet, infMgr *inference.Manager) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	send := func() {
		report := buildHealthReport(caps, infMgr)
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
func workLoop(ctx context.Context, client *hub.Client, gpus, hubURL, currentVersion string, infMgr *inference.Manager) {
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
		processWork(ctx, client, work, gpus, infMgr)
		jobActive.Store(0)
	}
}

func processWork(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, gpus string, infMgr *inference.Manager) {
	if operatorRuntimeState != nil {
		operatorRuntimeState.startJob(work)
	}

	// Determine job timeout based on type or explicit env var
	isStreaming := work.Kind == "inference" && work.Image == "streaming"
	isTraining := work.Kind == "training"
	isAgentHosting := work.Kind == "agent_hosting" || isAgentHostingTask(work.SpecJSON)
	jobTimeout := 10 * time.Minute
	if isStreaming {
		jobTimeout = 30 * time.Minute // Streaming inference often takes much longer context generation
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

	// Agent hosting: long-running container with health monitoring
	if isAgentHosting {
		slog.Info("starting agent hosting job", "job_id", work.JobID, "image", work.Image)

		healthFn := func(uptimeSeconds int) bool {
			resp, err := client.ReportAgentHealth(runCtx, extractDeploymentID(work.SpecJSON), uptimeSeconds)
			if err != nil {
				slog.Warn("agent health report failed", "job_id", work.JobID, "error", err)
				return false
			}
			if resp.ShouldStop {
				slog.Info("hub requested agent stop", "job_id", work.JobID, "deployment_id", extractDeploymentID(work.SpecJSON), "status", resp.Status, "job_status", resp.JobStatus)
				return true
			}
			return false
		}

		result, runErr := runner.RunAgent(runCtx, work.Image, work.SpecJSON, gpus, healthFn)

		// Submit final receipt with total uptime
		uptimeSeconds := 0
		if result != nil {
			uptimeSeconds = result.UptimeSeconds
		}

		hash := sha256.Sum256([]byte(work.JobID + fmt.Sprintf("%d", uptimeSeconds)))
		receipt := hub.Receipt{
			JobID:         work.JobID,
			ResultHashHex: hex.EncodeToString(hash[:]),
			MeteringUnits: uint64(uptimeSeconds),
			Metadata: map[string]any{
				"executor":       "agent_hosting",
				"uptime_seconds": uptimeSeconds,
				"exit_code":      0,
			},
		}
		if result != nil {
			receipt.Metadata["exit_code"] = result.ExitCode
		}
		if runErr != nil {
			receipt.Metadata["error"] = runErr.Error()
		}

		if err := submitReceiptWithRetry(ctx, client, receipt); err != nil {
			slog.Error("agent receipt submission failed", "job_id", work.JobID, "error", err)
		} else {
			slog.Info("agent job completed", "job_id", work.JobID, "uptime_seconds", uptimeSeconds)
		}
		if operatorRuntimeState != nil {
			operatorRuntimeState.finishJob(work, &runnerResultSnapshot{
				ResultHashHex: hex.EncodeToString(hash[:]),
				MeteringUnits: uint64(uptimeSeconds),
				Metadata:      receipt.Metadata,
			}, runErr)
		}
		return
	}

	// Route streaming inference jobs to persistent llama-server.
	// "streaming" is a pseudo-image, not a real container — never fall through to OCI runner.
	if isStreaming {
		if !infMgr.Healthy() {
			err := fmt.Errorf("inference manager is not healthy")
			slog.Warn("streaming job received but inference manager not healthy", "job_id", work.JobID)
			relayStreamingFailure(runCtx, client, work.JobID, err)
			if operatorRuntimeState != nil {
				operatorRuntimeState.finishJob(work, nil, err)
			}
			return
		}
		slog.Info("routing to inference manager", "job_id", work.JobID)
		if err := infMgr.RunStreamingJob(runCtx, client, work.JobID, work.SpecJSON); err != nil {
			slog.Warn("streaming inference failed", "job_id", work.JobID, "error", err)
			relayStreamingFailure(runCtx, client, work.JobID, err)
			if operatorRuntimeState != nil {
				operatorRuntimeState.finishJob(work, nil, err)
			}
			return
		}
		if operatorRuntimeState != nil {
			operatorRuntimeState.finishJob(work, nil, nil)
		}
		return
	}

	if strings.TrimSpace(work.Image) == "" || strings.TrimSpace(work.SpecJSON) == "" {
		slog.Warn("received work assignment without container spec, fast-rejecting", "job_id", work.JobID)
		rejectHash := sha256.Sum256([]byte(work.JobID + ":missing_spec"))
		rejectReceipt := hub.Receipt{
			JobID:         work.JobID,
			ResultHashHex: hex.EncodeToString(rejectHash[:]),
			MeteringUnits: 0,
			Metadata: map[string]any{
				"executor":  "node_agent",
				"exit_code": 1,
				"error":     "missing container image or spec",
			},
		}
		if err := submitReceiptWithRetry(runCtx, client, rejectReceipt); err != nil {
			slog.Error("fast-reject receipt submission failed", "job_id", work.JobID, "error", err)
		}
		if operatorRuntimeState != nil {
			operatorRuntimeState.finishJob(work, nil, fmt.Errorf("missing container image or spec"))
		}
		return
	}

	result, runErr := runner.Run(runCtx, work.Image, work.SpecJSON, gpus)
	if result == nil {
		slog.Warn("runner failed", "job_id", work.JobID, "error", runErr)
		if operatorRuntimeState != nil {
			operatorRuntimeState.finishJob(work, nil, runErr)
		}
		return
	}
	if runErr != nil {
		slog.Warn("container exited with error", "job_id", work.JobID, "exit_code", result.ExitCode, "error", runErr)
		if dockerRuntimeUnavailableError(runErr, result.Logs) {
			reportDockerRuntimeDegraded(client, infMgr)
		}
	}

	resultHash := result.Hash
	metadata := map[string]any{
		"executor":    "oci",
		"duration_ms": result.Duration.Milliseconds(),
		"exit_code":   result.ExitCode,
		"stderr_tail": result.Logs,
		"metrics":     result.Metrics,
	}

	if strings.TrimSpace(result.OutputPath) != "" {
		uploadRes, uploadErr := blob.Upload(runCtx, client, work.JobID, result.OutputPath)
		if uploadErr != nil {
			slog.Warn("artifact upload failed", "job_id", work.JobID, "error", uploadErr)
		} else {
			metadata["blob_url"] = uploadRes.URL
			metadata["object_key"] = uploadRes.Key
			if strings.TrimSpace(uploadRes.Key) != "" {
				metadata["manifest_key"] = uploadRes.Key + ".manifest.json"
			}
			if strings.TrimSpace(uploadRes.Hash) != "" {
				metadata["artifact_sha256"] = uploadRes.Hash
				resultHash = uploadRes.Hash
			}
		}
		_ = os.Remove(result.OutputPath)
	}

	units := uint64(work.Units)
	if units == 0 {
		units = 1
	}
	receipt := hub.Receipt{
		JobID:         work.JobID,
		ResultHashHex: resultHash,
		MeteringUnits: units,
		Metadata:      metadata,
	}
	if err := submitReceiptWithRetry(ctx, client, receipt); err != nil {
		slog.Error("submit receipt failed after retries", "job_id", work.JobID, "error", err)
		if operatorRuntimeState != nil {
			operatorRuntimeState.finishJob(work, &runnerResultSnapshot{
				DurationMs:    result.Duration.Milliseconds(),
				ResultHashHex: resultHash,
				ExitCode:      result.ExitCode,
				MeteringUnits: units,
				BlobURL:       stringValue(metadata["blob_url"]),
				ObjectKey:     stringValue(metadata["object_key"]),
				Metadata:      metadata,
			}, err)
		}
		return
	}
	slog.Info("job completed", "job_id", work.JobID, "hash", resultHash, "units", units)
	if operatorRuntimeState != nil {
		operatorRuntimeState.finishJob(work, &runnerResultSnapshot{
			DurationMs:    result.Duration.Milliseconds(),
			ResultHashHex: resultHash,
			ExitCode:      result.ExitCode,
			MeteringUnits: units,
			BlobURL:       stringValue(metadata["blob_url"]),
			ObjectKey:     stringValue(metadata["object_key"]),
			Metadata:      metadata,
		}, runErr)
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
	dockerBin := resolveDockerCLI()
	if dockerBin == "" {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, dockerBin, "ps", "-q", "--filter", "name=ryv_").CombinedOutput()
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
			exec.Command(dockerBin, "kill", id).Run()
			exec.Command(dockerBin, "rm", "-f", id).Run()
		}
	}
}

func buildHealthReport(caps hw.CapSet, infMgr *inference.Manager) hub.HealthReport {
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
	dockerCLI, dockerReady, dockerGPU, dockerParts := detectDockerRuntimeWithProbes(gpuReady, resolveDockerCLI, testDockerDaemon, testDockerGPU)

	if gpuReady {
		parts = append(parts, "gpu-detect:ok")
	} else {
		parts = append(parts, "gpu-detect:missing")
	}

	parts = append(parts, dockerParts...)

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
	if publicAIReady {
		parts = append(parts, "public-ai-ready:1")
	} else {
		parts = append(parts, "public-ai-ready:0")
	}
	if nativeReady {
		parts = append(parts, "native-inference-ready:1")
		parts = append(parts, "native-model:"+infMgr.ModelName())
	} else {
		parts = append(parts, "native-inference-ready:0")
	}
	if publicInferenceReady {
		parts = append(parts, "public-inference-ready:1")
	} else {
		parts = append(parts, "public-inference-ready:0")
	}

	return hub.HealthReport{
		TimestampMs: time.Now().UnixMilli(),
		GPUReady:    gpuReady,
		DockerGPU:   dockerCLI && dockerReady && dockerGPU,
		Message:     strings.Join(parts, ","),
	}
}

func dockerRuntimeUnavailableError(runErr error, logs string) bool {
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

func reportDockerRuntimeDegraded(client *hub.Client, infMgr *inference.Manager) {
	if client == nil {
		return
	}
	caps := hw.CapSet{}
	if operatorRuntimeState != nil {
		caps = operatorRuntimeState.caps
	}
	report := buildHealthReport(caps, infMgr)
	if !strings.Contains(strings.ToLower(report.Message), "docker-ready:0") {
		return
	}
	if operatorRuntimeState != nil {
		operatorRuntimeState.recordHealthReport(report)
	}
	reportCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := client.SendHealthReport(reportCtx, report); err != nil {
		slog.Warn("immediate docker health downgrade failed", "error", err)
		return
	}
	slog.Warn("reported degraded docker runtime health")
}

func publicAIOptInEnabled() bool {
	raw := strings.TrimSpace(os.Getenv("RYV_PUBLIC_AI"))
	if raw == "" {
		return true // default ON — operators opt out with RYV_PUBLIC_AI=0
	}
	switch strings.ToLower(raw) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func detectDockerRuntimeWithProbes(gpuReady bool, resolve func() string, daemonCheck func(string) bool, gpuCheck func(string) bool) (bool, bool, bool, []string) {
	dockerBin := strings.TrimSpace(resolve())
	if dockerBin == "" {
		parts := []string{"docker-cli:missing", "docker-ready:0", "docker:missing"}
		if gpuReady {
			parts = append(parts, "docker-gpu:missing")
		}
		return false, false, false, parts
	}

	daemonReady := daemonCheck(dockerBin)
	parts := []string{"docker-cli:present"}
	if daemonReady {
		parts = append(parts, "docker-ready:1", "docker:ok")
	} else {
		parts = append(parts, "docker-ready:0", "docker:unavailable")
		if gpuReady {
			parts = append(parts, "docker-gpu:unavailable")
		}
		return true, false, false, parts
	}

	if !gpuReady {
		return true, true, false, parts
	}

	dockerGPU := gpuCheck(dockerBin)
	if dockerGPU {
		parts = append(parts, "docker-gpu:ok")
	} else {
		parts = append(parts, "docker-gpu:failed")
	}
	return true, true, dockerGPU, parts
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

// ensureDockerGPURuntime auto-installs nvidia-container-toolkit on Linux if an
// NVIDIA GPU is detected and Docker is available but GPU passthrough fails.
// This runs once at startup so operators don't need to know about the toolkit.
func ensureDockerGPURuntime() {
	if runtime.GOOS != "linux" {
		return
	}
	// Only needed for NVIDIA GPUs.
	if _, err := exec.LookPath("nvidia-smi"); err != nil {
		return
	}
	dockerBin := resolveDockerCLI()
	if dockerBin == "" {
		return
	}
	if !testDockerDaemon(dockerBin) {
		// Try starting Docker daemon.
		slog.Info("Docker not running, attempting to start")
		if out, err := exec.Command("sudo", "systemctl", "start", "docker").CombinedOutput(); err != nil {
			slog.Warn("failed to start Docker", "error", err, "output", strings.TrimSpace(string(out)))
			return
		}
		time.Sleep(2 * time.Second)
		if !testDockerDaemon(dockerBin) {
			return
		}
	}
	// Docker is running — check if GPU passthrough already works.
	if testDockerGPUNvidia(dockerBin) {
		slog.Info("Docker GPU passthrough already configured")
		return
	}
	slog.Info("NVIDIA GPU detected but Docker GPU passthrough failed, installing nvidia-container-toolkit")

	// Check if nvidia-ctk is already installed (just not configured).
	if _, err := exec.LookPath("nvidia-ctk"); err == nil {
		slog.Info("nvidia-ctk found, configuring runtime")
		if configureNvidiaDockerRuntime(dockerBin) {
			return
		}
	}

	// Install nvidia-container-toolkit via package manager.
	if installNvidiaContainerToolkit() {
		configureNvidiaDockerRuntime(dockerBin)
	}
}

func installNvidiaContainerToolkit() bool {
	// Detect package manager and install.
	if _, err := exec.LookPath("apt-get"); err == nil {
		return installNvidiaCTKApt()
	}
	if _, err := exec.LookPath("dnf"); err == nil {
		return installNvidiaCTKDnf()
	}
	if _, err := exec.LookPath("yum"); err == nil {
		return installNvidiaCTKYum()
	}
	slog.Warn("no supported package manager found for nvidia-container-toolkit install")
	return false
}

func installNvidiaCTKApt() bool {
	steps := []struct {
		name string
		cmd  []string
	}{
		{"add NVIDIA GPG key", []string{"bash", "-c",
			`curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null`}},
		{"add NVIDIA repo", []string{"bash", "-c",
			`curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list`}},
		{"apt update", []string{"apt-get", "update", "-qq"}},
		{"install toolkit", []string{"apt-get", "install", "-y", "-qq", "nvidia-container-toolkit"}},
	}
	for _, s := range steps {
		slog.Info("nvidia-container-toolkit setup", "step", s.name)
		ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
		out, err := exec.CommandContext(ctx, "sudo", s.cmd...).CombinedOutput()
		cancel()
		if err != nil {
			slog.Warn("nvidia-container-toolkit install failed", "step", s.name, "error", err, "output", strings.TrimSpace(string(out)))
			return false
		}
	}
	slog.Info("nvidia-container-toolkit installed via apt")
	return true
}

func installNvidiaCTKDnf() bool {
	steps := []struct {
		name string
		cmd  []string
	}{
		{"add NVIDIA repo", []string{"bash", "-c",
			`curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | tee /etc/yum.repos.d/nvidia-container-toolkit.repo`}},
		{"install toolkit", []string{"dnf", "install", "-y", "nvidia-container-toolkit"}},
	}
	for _, s := range steps {
		slog.Info("nvidia-container-toolkit setup", "step", s.name)
		ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
		out, err := exec.CommandContext(ctx, "sudo", s.cmd...).CombinedOutput()
		cancel()
		if err != nil {
			slog.Warn("nvidia-container-toolkit install failed", "step", s.name, "error", err, "output", strings.TrimSpace(string(out)))
			return false
		}
	}
	slog.Info("nvidia-container-toolkit installed via dnf")
	return true
}

func installNvidiaCTKYum() bool {
	steps := []struct {
		name string
		cmd  []string
	}{
		{"add NVIDIA repo", []string{"bash", "-c",
			`curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | tee /etc/yum.repos.d/nvidia-container-toolkit.repo`}},
		{"install toolkit", []string{"yum", "install", "-y", "nvidia-container-toolkit"}},
	}
	for _, s := range steps {
		slog.Info("nvidia-container-toolkit setup", "step", s.name)
		ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
		out, err := exec.CommandContext(ctx, "sudo", s.cmd...).CombinedOutput()
		cancel()
		if err != nil {
			slog.Warn("nvidia-container-toolkit install failed", "step", s.name, "error", err, "output", strings.TrimSpace(string(out)))
			return false
		}
	}
	slog.Info("nvidia-container-toolkit installed via yum")
	return true
}

func configureNvidiaDockerRuntime(dockerBin string) bool {
	slog.Info("configuring NVIDIA Docker runtime")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	out, err := exec.CommandContext(ctx, "sudo", "nvidia-ctk", "runtime", "configure", "--runtime=docker").CombinedOutput()
	cancel()
	if err != nil {
		slog.Warn("nvidia-ctk configure failed", "error", err, "output", strings.TrimSpace(string(out)))
		return false
	}
	// Restart Docker to pick up new runtime config.
	slog.Info("restarting Docker to apply NVIDIA runtime")
	ctx2, cancel2 := context.WithTimeout(context.Background(), 30*time.Second)
	out2, err2 := exec.CommandContext(ctx2, "sudo", "systemctl", "restart", "docker").CombinedOutput()
	cancel2()
	if err2 != nil {
		slog.Warn("Docker restart failed", "error", err2, "output", strings.TrimSpace(string(out2)))
		return false
	}
	time.Sleep(3 * time.Second)
	// Verify it worked.
	if testDockerGPUNvidia(dockerBin) {
		slog.Info("Docker GPU passthrough configured successfully")
		return true
	}
	slog.Warn("Docker GPU passthrough still failing after toolkit install")
	return false
}

func testDockerDaemon(dockerBin string) bool {
	if strings.TrimSpace(dockerBin) == "" {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, dockerBin, "version", "--format", "{{.Server.Version}}").CombinedOutput()
	if err != nil {
		slog.Debug("docker daemon check failed", "error", err, "output", strings.TrimSpace(string(out)))
		return false
	}
	return strings.TrimSpace(string(out)) != ""
}

// testDockerGPU checks if Docker can access the GPU by running a minimal container.
// Tries NVIDIA (--gpus all) first, then ROCm (--device=/dev/kfd + /dev/dri) for AMD.
func testDockerGPU(dockerBin string) bool {
	if strings.TrimSpace(dockerBin) == "" {
		return false
	}
	if testDockerGPUNvidia(dockerBin) {
		return true
	}
	return testDockerGPURocm(dockerBin)
}

func testDockerGPUNvidia(dockerBin string) bool {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, dockerBin, "run", "--rm", "--gpus", "all",
		"nvidia/cuda:12.4.1-base-ubuntu22.04", "nvidia-smi", "--query-gpu=name", "--format=csv,noheader").CombinedOutput()
	if err != nil {
		slog.Debug("docker NVIDIA GPU test failed", "error", err, "output", strings.TrimSpace(string(out)))
		return false
	}
	result := strings.TrimSpace(string(out))
	slog.Info("docker NVIDIA GPU test passed", "gpu", result)
	return result != ""
}

func testDockerGPURocm(dockerBin string) bool {
	// Check if ROCm devices exist before pulling a container image.
	if _, err := os.Stat("/dev/kfd"); err != nil {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, dockerBin, "run", "--rm",
		"--device=/dev/kfd", "--device=/dev/dri",
		"rocm/rocm-terminal:latest", "rocm-smi", "--showproductname").CombinedOutput()
	if err != nil {
		slog.Debug("docker ROCm GPU test failed", "error", err, "output", strings.TrimSpace(string(out)))
		return false
	}
	result := strings.TrimSpace(string(out))
	slog.Info("docker ROCm GPU test passed", "output", result)
	return result != ""
}

func resolveDockerCLI() string {
	if p, err := exec.LookPath("docker"); err == nil {
		return p
	}
	if runtime.GOOS == "windows" {
		candidates := []string{
			filepath.Join(os.Getenv("ProgramFiles"), "Docker", "Docker", "resources", "bin", "docker.exe"),
			filepath.Join(os.Getenv("ProgramW6432"), "Docker", "Docker", "resources", "bin", "docker.exe"),
		}
		for _, candidate := range candidates {
			if candidate == "" {
				continue
			}
			if _, err := os.Stat(candidate); err == nil {
				return candidate
			}
		}
	}
	return ""
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
