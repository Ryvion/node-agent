package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"math"
	"os"
	"os/exec"
	"os/signal"
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
	flagReferral   string
	flagGPUs       string
	flagMaxGPUUtil float64
)

// cachedGPUUtil stores the latest GPU utilization from heartbeat sampling.
// Used by the work loop to skip fetching work when GPU is busy.
var cachedGPUUtil atomic.Uint64 // stores float64 bits via math.Float64bits

func main() {
	// Subcommand: ryvion-node claim <CODE>
	if len(os.Args) > 1 && os.Args[1] == "claim" {
		runClaim()
		return
	}

	versionFlag := flag.Bool("version", false, "Print version and exit")
	flag.StringVar(&flagHub, "hub", "https://ryvion-hub.fly.dev", "Hub orchestrator base URL")
	flag.StringVar(&flagDevice, "type", "", "Node device type (gpu|cpu|mobile|iot)")
	flag.StringVar(&flagReferral, "referral", "", "Optional referral code")
	flag.StringVar(&flagGPUs, "gpus", "auto", "Docker --gpus value (auto|all|none|device list)")
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

	caps := hw.DetectCaps(flagDevice)
	deviceType := resolveDeviceType(flagDevice, caps)

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
	}, deviceType, strings.TrimSpace(flagReferral)); err != nil {
		slog.Error("register failed", "error", err)
		return
	}
	slog.Info("register succeeded", "hub", hubURL, "device_type", deviceType, "pubkey", client.PublicKeyHex())
	if flagMaxGPUUtil > 0 && flagMaxGPUUtil < 100 {
		slog.Info("GPU utilization cap enabled", "max_gpu_util", flagMaxGPUUtil)
	}

	if err := client.SolveChallenge(ctx); err != nil {
		slog.Warn("challenge solve failed", "error", err)
	}
	if err := client.SendHealthReport(ctx, buildHealthReport(caps)); err != nil {
		slog.Warn("health report failed", "error", err)
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

	// Independent heartbeat goroutine — keeps node "online" regardless of
	// what the work loop is doing (long-poll, job execution, etc.).
	latestVersion := make(chan string, 1)
	go heartbeatLoop(ctx, client, latestVersion)

	// Work loop — fetch and process jobs.
	workLoop(ctx, client, flagGPUs, hubURL, version, infMgr, latestVersion)
}

// heartbeatLoop sends heartbeats on a fixed 30-second interval, completely
// independent of the work loop. This ensures the node never goes stale
// even if a job or long-poll is in progress.
func heartbeatLoop(ctx context.Context, client *hub.Client, versionCh chan<- string) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	// Send first heartbeat immediately.
	sendHeartbeat(ctx, client, versionCh)

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			sendHeartbeat(ctx, client, versionCh)
		}
	}
}

func sendHeartbeat(ctx context.Context, client *hub.Client, versionCh chan<- string) {
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
		slog.Warn("heartbeat failed", "error", err)
		return
	}
	// Non-blocking send of latest version to work loop for update checks.
	if latest != "" {
		select {
		case versionCh <- latest:
		default:
		}
	}
}

// workLoop fetches and processes jobs. Heartbeats are handled separately.
func workLoop(ctx context.Context, client *hub.Client, gpus, hubURL, currentVersion string, infMgr *inference.Manager, versionCh <-chan string) {
	var lastUpdateAttempt time.Time
	backoff := 5 * time.Second
	maxBackoff := 2 * time.Minute

	for {
		// Check for version updates (non-blocking read from heartbeat goroutine).
		select {
		case latest := <-versionCh:
			if update.NeedsUpdate(currentVersion, latest) && time.Since(lastUpdateAttempt) > 30*time.Minute {
				lastUpdateAttempt = time.Now()
				slog.Info("update available", "current", currentVersion, "latest", latest)
				if err := update.Apply(ctx, hubURL); err != nil {
					slog.Warn("auto-update failed", "error", err)
				} else {
					slog.Info("update applied, restarting")
					if restartErr := update.Restart(); restartErr != nil {
						slog.Warn("restart failed — update will take effect on next manual restart", "error", restartErr)
					}
				}
			}
		default:
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

		processWork(ctx, client, work, gpus, infMgr)
	}
}

func processWork(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, gpus string, infMgr *inference.Manager) {
	// Route inference jobs to persistent llama-server if available
	if work.Kind == "inference" && infMgr.Healthy() {
		slog.Info("routing to inference manager", "job_id", work.JobID)
		jobTimeout := 5 * time.Minute
		runCtx, cancel := context.WithTimeout(ctx, jobTimeout)
		defer cancel()
		if err := infMgr.RunStreamingJob(runCtx, client, work.JobID, work.SpecJSON); err != nil {
			slog.Warn("streaming inference failed", "job_id", work.JobID, "error", err)
		}
		return
	}

	if strings.TrimSpace(work.Image) == "" || strings.TrimSpace(work.SpecJSON) == "" {
		slog.Warn("received work assignment without container spec", "job_id", work.JobID)
		return
	}

	jobTimeout := 10 * time.Minute
	if v := strings.TrimSpace(os.Getenv("RYV_JOB_TIMEOUT")); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d > 0 {
			jobTimeout = d
		}
	}
	runCtx, cancel := context.WithTimeout(ctx, jobTimeout)
	defer cancel()

	result, runErr := runner.Run(runCtx, work.Image, work.SpecJSON, gpus)
	if result == nil {
		slog.Warn("runner failed", "job_id", work.JobID, "error", runErr)
		return
	}
	if runErr != nil {
		slog.Warn("container exited with error", "job_id", work.JobID, "exit_code", result.ExitCode, "error", runErr)
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
	if err := client.SubmitReceipt(ctx, hub.Receipt{
		JobID:         work.JobID,
		ResultHashHex: resultHash,
		MeteringUnits: units,
		Metadata:      metadata,
	}); err != nil {
		slog.Warn("submit receipt failed", "job_id", work.JobID, "error", err)
		return
	}
	slog.Info("job completed", "job_id", work.JobID, "hash", resultHash, "units", units)
}

func cleanupOrphanedContainers() {
	out, err := exec.Command("docker", "ps", "-q", "--filter", "name=ryv_").CombinedOutput()
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
			exec.Command("docker", "kill", id).Run()
			exec.Command("docker", "rm", "-f", id).Run()
		}
	}
}

func buildHealthReport(caps hw.CapSet) hub.HealthReport {
	gpuReady := strings.TrimSpace(caps.GPUModel) != ""
	hasDocker := commandExists("docker")
	dockerGPU := false
	parts := []string{}

	if gpuReady {
		parts = append(parts, "nvidia-smi:ok")
	} else {
		parts = append(parts, "nvidia-smi:missing")
	}

	if hasDocker {
		if gpuReady {
			// Actually test Docker GPU passthrough with a quick container
			dockerGPU = testDockerGPU()
			if dockerGPU {
				parts = append(parts, "docker-gpu:ok")
			} else {
				parts = append(parts, "docker-gpu:failed")
			}
		} else {
			parts = append(parts, "docker:ok")
		}
	} else {
		parts = append(parts, "docker:missing")
	}

	if gpuReady {
		parts = append(parts, "gpu_model:"+caps.GPUModel)
	}

	return hub.HealthReport{
		TimestampMs: time.Now().UnixMilli(),
		GPUReady:    gpuReady,
		DockerGPU:   dockerGPU,
		Message:     strings.Join(parts, ","),
	}
}

// testDockerGPU checks if Docker can access the GPU by running a minimal container.
func testDockerGPU() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	// Use hello-world-sized image to test --gpus flag; nvidia-smi is baked into
	// the NVIDIA base images and also available on Windows hosts.
	out, err := exec.CommandContext(ctx, "docker", "run", "--rm", "--gpus", "all",
		"nvidia/cuda:12.4.1-base-ubuntu22.04", "nvidia-smi", "--query-gpu=name", "--format=csv,noheader").CombinedOutput()
	if err != nil {
		slog.Debug("docker GPU test failed", "error", err, "output", strings.TrimSpace(string(out)))
		return false
	}
	result := strings.TrimSpace(string(out))
	slog.Info("docker GPU test passed", "gpu", result)
	return result != ""
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
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: level}))
	slog.SetDefault(logger)
}
