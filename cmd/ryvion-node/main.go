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

func main() {
	// Subcommand: ryvion-node claim <CODE>
	if len(os.Args) > 1 && os.Args[1] == "claim" {
		runClaim()
		return
	}

	versionFlag := flag.Bool("version", false, "Print version and exit")
	flag.StringVar(&flagHub, "hub", "https://ryvion-hub.fly.dev", "Hub orchestrator base URL")
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
	slog.Info("register succeeded", "hub", hubURL, "device_type", deviceType, "pubkey", client.PublicKeyHex())
	if flagMaxGPUUtil > 0 && flagMaxGPUUtil < 100 {
		slog.Info("GPU utilization cap enabled", "max_gpu_util", flagMaxGPUUtil)
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
	latestVersion := make(chan string, 1)
	go func() {
		defer func() {
			if r := recover(); r != nil {
				slog.Error("heartbeat goroutine panic", "error", r)
			}
		}()
		heartbeatLoop(ctx, client, latestVersion)
	}()

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
		if operatorRuntimeState != nil {
			operatorRuntimeState.recordHeartbeat(metrics, "", err)
		}
		slog.Warn("heartbeat failed", "error", err)
		return
	}
	if operatorRuntimeState != nil {
		operatorRuntimeState.recordHeartbeat(metrics, latest, nil)
	}
	// Non-blocking send of latest version to work loop for update checks.
	if latest != "" {
		select {
		case versionCh <- latest:
		default:
		}
	}
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
func workLoop(ctx context.Context, client *hub.Client, gpus, hubURL, currentVersion string, infMgr *inference.Manager, versionCh <-chan string) {
	var lastUpdateAttempt time.Time
	backoff := 5 * time.Second
	maxBackoff := 2 * time.Minute

	for {
		// Check for version updates (non-blocking read from heartbeat goroutine).
		select {
		case latest := <-versionCh:
			if update.NeedsUpdate(currentVersion, latest) && time.Since(lastUpdateAttempt) > 5*time.Minute {
				if jobActive.Load() != 0 {
					slog.Info("update available but job in progress, deferring", "latest", latest)
				} else {
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
	jobTimeout := 10 * time.Minute
	if isStreaming {
		jobTimeout = 30 * time.Minute // Streaming inference often takes much longer context generation
	}
	if v := strings.TrimSpace(os.Getenv("RYV_JOB_TIMEOUT")); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d > 0 {
			jobTimeout = d
		}
	}

	runCtx, cancel := context.WithTimeout(ctx, jobTimeout)
	defer cancel()

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
		slog.Warn("received work assignment without container spec", "job_id", work.JobID)
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
	if nativeReady {
		parts = append(parts, "native-inference-ready:1")
		parts = append(parts, "native-model:"+infMgr.ModelName())
	} else {
		parts = append(parts, "native-inference-ready:0")
	}

	return hub.HealthReport{
		TimestampMs: time.Now().UnixMilli(),
		GPUReady:    gpuReady,
		DockerGPU:   dockerCLI && dockerReady && dockerGPU,
		Message:     strings.Join(parts, ","),
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

func stringValue(v any) string {
	s, _ := v.(string)
	return strings.TrimSpace(s)
}
