package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/Ryvion/node-agent/internal/blob"
	"github.com/Ryvion/node-agent/internal/hub"
	"github.com/Ryvion/node-agent/internal/hw"
	"github.com/Ryvion/node-agent/internal/nodekey"
	"github.com/Ryvion/node-agent/internal/runner"
)

// Set via -ldflags at build time.
var version = "dev"

func main() {
	versionFlag := flag.Bool("version", false, "Print version and exit")
	hubFlag := flag.String("hub", "https://ryvion-hub.fly.dev", "Hub orchestrator base URL")
	deviceTypeFlag := flag.String("type", "", "Node device type (gpu|cpu|mobile|iot)")
	referralFlag := flag.String("referral", "", "Optional referral code")
	gpusFlag := flag.String("gpus", "auto", "Docker --gpus value (auto|all|none|device list)")
	flag.Parse()

	if *versionFlag {
		fmt.Println("ryvion-node", version)
		os.Exit(0)
	}

	initLogger()
	cleanupOrphanedContainers()

	hubURL := strings.TrimSpace(*hubFlag)
	if envHub := strings.TrimSpace(os.Getenv("RYV_HUB_URL")); envHub != "" {
		hubURL = envHub
	}
	if hubURL == "" {
		slog.Error("hub URL is required")
		os.Exit(1)
	}

	pub, priv, err := nodekey.LoadOrCreate(strings.TrimSpace(os.Getenv("RYV_KEY_PATH")))
	if err != nil {
		slog.Error("failed to load node key", "error", err)
		os.Exit(1)
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

	caps := hw.DetectCaps(*deviceTypeFlag)
	deviceType := resolveDeviceType(*deviceTypeFlag, caps)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := client.Register(ctx, hub.Capabilities{
		GPUModel:          caps.GPUModel,
		CPUCores:          caps.CPUCores,
		RAMBytes:          caps.RAMBytes,
		VRAMBytes:         caps.VRAMBytes,
		Sensors:           caps.Sensors,
		BandwidthMbps:     caps.BandwidthMbps,
		GeohashBucket:     caps.GeohashBucket,
		AttestationMethod: caps.Attestation,
	}, deviceType, strings.TrimSpace(*referralFlag)); err != nil {
		slog.Error("register failed", "error", err)
		os.Exit(1)
	}
	slog.Info("register succeeded", "hub", hubURL, "device_type", deviceType, "pubkey", client.PublicKeyHex())

	if err := client.SolveChallenge(ctx); err != nil {
		slog.Warn("challenge solve failed", "error", err)
	}
	if err := client.SendHealthReport(ctx, buildHealthReport(caps)); err != nil {
		slog.Warn("health report failed", "error", err)
	}

	backoff := 10 * time.Second
	maxBackoff := 5 * time.Minute
	consecutiveFailures := 0

	for {
		err := runCycle(ctx, client, *gpusFlag)
		if err != nil {
			consecutiveFailures++
			backoff = time.Duration(float64(backoff) * 1.5)
			if backoff > maxBackoff {
				backoff = maxBackoff
			}
			slog.Warn("cycle failed, backing off", "error", err, "backoff", backoff, "failures", consecutiveFailures)
		} else {
			if consecutiveFailures > 0 {
				slog.Info("hub connection restored", "previous_failures", consecutiveFailures)
			}
			consecutiveFailures = 0
			backoff = 10 * time.Second
		}

		select {
		case <-ctx.Done():
			slog.Info("shutting down")
			return
		case <-time.After(backoff):
		}
	}
}

func runCycle(ctx context.Context, client *hub.Client, gpus string) error {
	metrics := hw.SampleMetrics()
	if err := client.Heartbeat(ctx, hub.Metrics{
		TimestampMs: time.Now().UnixMilli(),
		CPUUtil:     metrics.CPUUtil,
		MemUtil:     metrics.MemUtil,
		GPUUtil:     metrics.GPUUtil,
		PowerWatts:  metrics.PowerWatts,
	}); err != nil {
		return fmt.Errorf("heartbeat failed: %w", err)
	}

	work, err := client.FetchWork(ctx)
	if err != nil {
		return fmt.Errorf("fetch work failed: %w", err)
	}
	if work == nil {
		return nil
	}
	if strings.TrimSpace(work.Image) == "" || strings.TrimSpace(work.SpecJSON) == "" {
		slog.Warn("received work assignment without container spec", "job_id", work.JobID)
		return nil
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
		return nil
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
		return nil
	}
	slog.Info("job completed", "job_id", work.JobID, "hash", resultHash, "units", units)
	return nil
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
	dockerGPU := gpuReady && commandExists("docker")
	parts := []string{}
	if gpuReady {
		parts = append(parts, "nvidia-smi:ok")
	} else {
		parts = append(parts, "nvidia-smi:missing")
	}
	if dockerGPU {
		parts = append(parts, "docker:ok")
	} else if commandExists("docker") {
		parts = append(parts, "docker:no-gpu")
	} else {
		parts = append(parts, "docker:missing")
	}
	if strings.TrimSpace(caps.GPUModel) != "" {
		parts = append(parts, "gpu_model:"+caps.GPUModel)
	}
	return hub.HealthReport{
		TimestampMs: time.Now().UnixMilli(),
		GPUReady:    gpuReady,
		DockerGPU:   dockerGPU,
		Message:     strings.Join(parts, ","),
	}
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
