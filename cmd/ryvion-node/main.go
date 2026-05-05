package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
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

	"github.com/Ryvion/node-agent/internal/diagnostics"
	"github.com/Ryvion/node-agent/internal/hub"
	"github.com/Ryvion/node-agent/internal/hw"
	"github.com/Ryvion/node-agent/internal/inference"
	"github.com/Ryvion/node-agent/internal/nodekey"
	"github.com/Ryvion/node-agent/internal/runtimeexec"
	"github.com/Ryvion/node-agent/internal/update"
	v7artifact "github.com/Ryvion/node-agent/internal/v7/artifact"
	v7capability "github.com/Ryvion/node-agent/internal/v7/capability"
	v7heartbeat "github.com/Ryvion/node-agent/internal/v7/heartbeat"
	v7memorybench "github.com/Ryvion/node-agent/internal/v7/memorybench"
	v7modelbench "github.com/Ryvion/node-agent/internal/v7/modelbench"
	v7onboarding "github.com/Ryvion/node-agent/internal/v7/onboarding"
	v7proofrunner "github.com/Ryvion/node-agent/internal/v7/proofrunner"
	v7sandbox "github.com/Ryvion/node-agent/internal/v7/sandbox"
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

const (
	v7ProofFlagEnv                  = "RYV_NODE_V7_PROOF"
	v7ProofMetadataKey              = "v7_proof"
	v7ProofOutputBytesMetadataKey   = "_v7_proof_output_bytes"
	v7ProofArtifactBytesMetadataKey = "_v7_proof_artifact_bytes"
)

var (
	workLoopDiagnostics       = diagnostics.NewWorkLoopDiagnostics()
	v7ModelBenchmarkStatus    = v7modelbench.NewLocalStatus()
	newV7ModelBenchmarkRunner = func(infMgr *inference.Manager, gpuDetected bool) v7modelbench.ModelBenchmarkRunner {
		return v7modelbench.NativeInferenceModelBenchmarkRunner{
			Native:           infMgr,
			AgentVersion:     version,
			OS:               runtime.GOOS,
			Arch:             runtime.GOARCH,
			GPUDetected:      gpuDetected,
			RuntimeAvailable: inference.NativeRuntimeAvailable,
		}
	}
)

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
	if len(os.Args) > 1 && os.Args[1] == "doctor" {
		runDoctor(os.Args[2:])
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "memorybench-selftest" {
		runMemoryBenchSelfTest(os.Args[2:])
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "modelbench-selftest" {
		runModelBenchSelfTest(os.Args[2:])
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

func runDoctor(args []string) {
	fs := flag.NewFlagSet("doctor", flag.ExitOnError)
	hubURL := fs.String("hub", firstNonEmpty(strings.TrimSpace(os.Getenv("RYV_HUB_URL")), "https://api.ryvion.ai"), "Hub orchestrator base URL")
	dataDir := fs.String("data-dir", strings.TrimSpace(os.Getenv("RYV_DATA_DIR")), "Node data directory to check")
	deviceType := fs.String("type", "", "Node device type (gpu|cpu|mobile|iot)")
	_ = fs.Parse(args)

	report := v7onboarding.RunBasicOnboardingChecksWithOptions(v7onboarding.CheckOptions{
		AgentVersion: version,
		HubURL:       *hubURL,
		DataDir:      *dataDir,
		DeviceType:   *deviceType,
	})
	fmt.Println(v7onboarding.FormatOnboardingReport(report))
	if report.HasHardErrors() {
		os.Exit(1)
	}
	os.Exit(0)
}

func runMemoryBenchSelfTest(args []string) {
	defaults := v7memorybench.DefaultMemoryBenchSelfTestConfig()
	fs := flag.NewFlagSet("memorybench-selftest", flag.ExitOnError)
	tokenCount := fs.Int("tokens", defaults.TokenCount, "Synthetic token count")
	valueDim := fs.Int("dim", defaults.ValueDim, "Synthetic attention value dimension")
	seed := fs.Int64("seed", defaults.Seed, "Synthetic request seed")
	jsonOutput := fs.Bool("json", false, "Print JSON output")
	_ = fs.Parse(args)

	result, err := v7memorybench.RunMemoryBenchSelfTest(v7memorybench.MemoryBenchSelfTestConfig{
		Seed:       *seed,
		TokenCount: *tokenCount,
		ValueDim:   *valueDim,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println(v7memorybench.FormatMemoryBenchSelfTestResult(result, *jsonOutput))
	os.Exit(0)
}

func runModelBenchSelfTest(args []string) {
	config, jsonOutput, err := parseModelBenchSelfTestFlags(args)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(config.TimeoutMs)*time.Millisecond)
	defer cancel()

	result, err := runModelBenchSelfTestViaOperatorAPI(ctx, config, operatorAPIPort(defaultOperatorAPIPort))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println(v7modelbench.FormatModelBenchmarkSelfTestResult(result, jsonOutput))
	os.Exit(0)
}

func parseModelBenchSelfTestFlags(args []string) (v7modelbench.ModelBenchmarkSelfTestConfig, bool, error) {
	defaults := v7modelbench.DefaultModelBenchmarkSelfTestConfig()
	fs := flag.NewFlagSet("modelbench-selftest", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	modelID := fs.String("model", defaults.ModelID, "Native model ID")
	maxTokens := fs.Int("tokens", defaults.MaxTokens, "Maximum generated tokens")
	timeoutRaw := fs.String("timeout", fmt.Sprintf("%dms", defaults.TimeoutMs), "Benchmark timeout as duration or milliseconds")
	jsonOutput := fs.Bool("json", false, "Print JSON output")
	if err := fs.Parse(args); err != nil {
		return v7modelbench.ModelBenchmarkSelfTestConfig{}, false, err
	}
	timeoutMs, err := parseModelBenchTimeoutMs(*timeoutRaw)
	if err != nil {
		return v7modelbench.ModelBenchmarkSelfTestConfig{}, false, err
	}
	return v7modelbench.ModelBenchmarkSelfTestConfig{
		ModelID:   *modelID,
		MaxTokens: *maxTokens,
		TimeoutMs: timeoutMs,
	}, *jsonOutput, nil
}

func parseModelBenchTimeoutMs(raw string) (int64, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return v7modelbench.DefaultModelBenchmarkSelfTestConfig().TimeoutMs, nil
	}
	if ms, err := strconv.ParseInt(raw, 10, 64); err == nil {
		if ms <= 0 {
			return 0, fmt.Errorf("timeout must be greater than zero")
		}
		return ms, nil
	}
	duration, err := time.ParseDuration(raw)
	if err != nil {
		return 0, fmt.Errorf("invalid timeout %q", raw)
	}
	if duration <= 0 {
		return 0, fmt.Errorf("timeout must be greater than zero")
	}
	return duration.Milliseconds(), nil
}

func runModelBenchSelfTestViaOperatorAPI(ctx context.Context, config v7modelbench.ModelBenchmarkSelfTestConfig, port string) (v7modelbench.ModelBenchmarkResult, error) {
	body, err := json.Marshal(struct {
		ModelID   string `json:"model_id"`
		MaxTokens int    `json:"max_tokens"`
		TimeoutMs int64  `json:"timeout_ms"`
	}{
		ModelID:   config.ModelID,
		MaxTokens: config.MaxTokens,
		TimeoutMs: config.TimeoutMs,
	})
	if err != nil {
		return v7modelbench.ModelBenchmarkResult{}, err
	}

	url := fmt.Sprintf("http://127.0.0.1:%s/api/v1/operator/v7/model-benchmark/run", strings.TrimSpace(port))
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err == nil {
		req.Header.Set("Content-Type", "application/json")
		if resp, doErr := http.DefaultClient.Do(req); doErr == nil {
			defer resp.Body.Close()
			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				var result v7modelbench.ModelBenchmarkResult
				if decodeErr := json.NewDecoder(resp.Body).Decode(&result); decodeErr != nil {
					return v7modelbench.ModelBenchmarkResult{}, decodeErr
				}
				if validationErr := v7modelbench.ValidateModelBenchmarkResult(result); validationErr != nil {
					return result, validationErr
				}
				return result, nil
			}
			_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
		}
	}

	runner := v7modelbench.NativeInferenceModelBenchmarkRunner{
		AgentVersion:     version,
		RuntimeAvailable: inference.NativeRuntimeAvailable,
	}
	return v7modelbench.RunModelBenchmarkSelfTest(ctx, runner, config)
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
		heartbeatLoop(ctx, client, caps, deviceType, declaredCountry, infMgr, runtimeMgr)
	}()

	// Work loop — fetch and process jobs.
	workLoop(ctx, client, flagGPUs, hubURL, version, infMgr, runtimeMgr, strings.TrimSpace(caps.GPUModel) != "")
}

// heartbeatLoop sends heartbeats on a fixed interval, completely independent
// of the work loop. Implements a circuit breaker: after 30 consecutive failures
// (~5 min at 10s), the interval increases to 60s with a warning. Resets on success.
func heartbeatLoop(ctx context.Context, client *hub.Client, caps hw.CapSet, deviceType string, declaredCountry string, infMgr *inference.Manager, runtimeMgr *runtimeManager) {
	const (
		normalInterval    = 30 * time.Second
		backoffInterval   = 60 * time.Second
		circuitBreakerMax = 30
	)

	ticker := time.NewTicker(normalInterval)
	defer ticker.Stop()

	var consecutiveFailures int

	// Send first heartbeat immediately.
	if sendHeartbeat(ctx, client, caps, deviceType, declaredCountry, infMgr, runtimeMgr) {
		consecutiveFailures = 0
	} else {
		consecutiveFailures++
	}

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if sendHeartbeat(ctx, client, caps, deviceType, declaredCountry, infMgr, runtimeMgr) {
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

func sendHeartbeat(ctx context.Context, client *hub.Client, caps hw.CapSet, deviceType string, declaredCountry string, infMgr *inference.Manager, runtimeMgr *runtimeManager) bool {
	metrics := hw.SampleMetrics()

	// Cache GPU utilization for the work loop's throttle check.
	cachedGPUUtil.Store(math.Float64bits(metrics.GPUUtil))

	// Report whether the node is self-throttling due to operator GPU usage.
	throttled := flagMaxGPUUtil > 0 && flagMaxGPUUtil < 100 && metrics.GPUUtil > flagMaxGPUUtil

	heartbeatMetrics := hub.Metrics{
		TimestampMs:  time.Now().UnixMilli(),
		CPUUtil:      metrics.CPUUtil,
		MemUtil:      metrics.MemUtil,
		GPUUtil:      metrics.GPUUtil,
		PowerWatts:   metrics.PowerWatts,
		GPUThrottled: throttled,
		V7Heartbeat:  buildOptionalV7HeartbeatPayload(client.PublicKeyHex(), caps, deviceType, declaredCountry, infMgr, runtimeMgr),
	}

	heartbeat, err := client.Heartbeat(ctx, heartbeatMetrics)
	if err != nil && heartbeatMetrics.V7Heartbeat != nil {
		heartbeatMetrics.V7Heartbeat = nil
		if retryHeartbeat, retryErr := client.Heartbeat(ctx, heartbeatMetrics); retryErr == nil {
			slog.Warn("heartbeat with V7 payload failed; retried without V7 payload", "error", err)
			heartbeat = retryHeartbeat
			err = nil
		}
	}
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

func buildOptionalV7HeartbeatPayload(nodePublicKey string, caps hw.CapSet, deviceType string, declaredCountry string, infMgr *inference.Manager, runtimeMgr *runtimeManager) *v7heartbeat.V7HeartbeatPayload {
	if !v7heartbeat.V7HeartbeatEnabledFromEnv() {
		return nil
	}
	payload, err := buildV7HeartbeatPayloadForNode(nodePublicKey, caps, deviceType, declaredCountry, infMgr, runtimeMgr)
	if err != nil {
		slog.Warn("failed to build V7 heartbeat payload; sending legacy heartbeat", "error", err)
		return nil
	}
	return payload
}

func buildV7HeartbeatPayloadForNode(nodePublicKey string, caps hw.CapSet, deviceType string, declaredCountry string, infMgr *inference.Manager, runtimeMgr *runtimeManager) (*v7heartbeat.V7HeartbeatPayload, error) {
	gpuDetected := strings.TrimSpace(caps.GPUModel) != "" || caps.VRAMBytes > 0
	nativeSupported := inference.NativeRuntimeAvailable()
	nativeReady := nativeSupported && infMgr != nil && infMgr.Healthy()

	runtimeProfile := v7capability.RuntimeProfile{
		NativeInferenceSupported: nativeSupported,
		OCIAvailable:             false,
		LlamaServerAvailable:     nativeSupported,
		ImageRuntimeAvailable:    false,
		SupportedRunnerKinds:     v7SupportedRunnerKinds(nativeSupported, false, false),
	}

	residentModelIDs := []string{}
	if nativeReady {
		residentModelIDs = append(residentModelIDs, infMgr.ModelName())
	}

	evidenceSummary := v7capability.EvidenceCapabilitySummary{
		SupportsArtifactManifest:    true,
		SupportsRYV3EvidencePayload: true,
		SupportsRuntimeHash:         v7RuntimeManifestHash(runtimeMgr) != "",
	}
	sandboxSummary := v7capability.SandboxCapabilitySummary{
		RejectsUnsafePickle:        true,
		RunnerAllowlistEnabled:     true,
		FilesystemIsolationPlanned: true,
		NetworkIsolationSupported:  false,
	}
	sandboxPolicy := v7sandbox.DefaultSandboxPolicy()

	payload, err := v7heartbeat.BuildV7HeartbeatPayload(v7heartbeat.BuildV7HeartbeatPayloadInput{
		AgentVersion:         version,
		NodePublicKey:        nodePublicKey,
		OS:                   runtime.GOOS,
		Arch:                 runtime.GOARCH,
		DeviceType:           deviceType,
		DeclaredCountry:      declaredCountry,
		HardwareCapabilities: caps,
		RuntimeProfile:       runtimeProfile,
		ModelCapabilitySummary: v7capability.ModelCapabilitySummary{
			ResidentModelIDs:      residentModelIDs,
			MaxResidentModelBytes: caps.VRAMBytes,
			SupportsModelLease:    gpuDetected && caps.VRAMBytes > 0 && nativeSupported,
		},
		SandboxCapabilitySummary:  sandboxSummary,
		SandboxPolicy:             &sandboxPolicy,
		EvidenceCapabilitySummary: evidenceSummary,
	})
	if err != nil {
		return nil, err
	}
	return &payload, nil
}

func v7RuntimeManifestHash(runtimeMgr *runtimeManager) string {
	if runtimeMgr == nil {
		return ""
	}
	hash := strings.TrimSpace(runtimeMgr.contract.ManifestHash)
	if hash != "" {
		return hash
	}
	meta := runtimeMgr.contract
	if strings.TrimSpace(meta.Channel) == "" &&
		strings.TrimSpace(meta.Version) == "" &&
		strings.TrimSpace(meta.Provider) == "" &&
		strings.TrimSpace(meta.Mode) == "" &&
		strings.TrimSpace(meta.Source) == "" &&
		strings.TrimSpace(meta.Artifact) == "" &&
		strings.TrimSpace(meta.Binary) == "" &&
		strings.TrimSpace(meta.Backend) == "" {
		return ""
	}
	return computeRuntimeManifestHash(meta)
}

func v7SupportedRunnerKinds(nativeSupported bool, ociAvailable bool, ryvionRuntimeAvailable bool) []string {
	kinds := []string{executorKindNativeReport}
	if nativeSupported {
		kinds = append(kinds, executorKindNativeStreaming)
	}
	if ociAvailable {
		kinds = append(kinds, executorKindManagedOCI, executorKindAgentHosting)
		if commandExists("git") {
			kinds = append(kinds, executorKindWorkCapsule)
		}
	}
	if ryvionRuntimeAvailable {
		kinds = append(kinds, executorKindRyvionRuntime)
	}
	return kinds
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
				workLoopDiagnostics.RecordPollStart()
				select {
				case <-ctx.Done():
					workLoopDiagnostics.RecordPollEnd(ctx.Err())
					return
				case <-time.After(5 * time.Second):
				}
				workLoopDiagnostics.RecordPollEnd(nil)
				continue
			}
		}

		workLoopDiagnostics.RecordPollStart()
		work, err := client.FetchWork(ctx)
		if err != nil {
			slog.Warn("fetch work failed", "error", err)
			select {
			case <-ctx.Done():
				workLoopDiagnostics.RecordPollEnd(err)
				return
			case <-time.After(backoff):
			}
			workLoopDiagnostics.RecordPollEnd(err)
			backoff = time.Duration(float64(backoff) * 1.5)
			if backoff > maxBackoff {
				backoff = maxBackoff
			}
			continue
		}
		workLoopDiagnostics.RecordPollEnd(nil)

		// Successful fetch — reset backoff.
		backoff = 5 * time.Second

		if work == nil {
			continue
		}

		decodeStarted := time.Now()
		specTask := diagnostics.WorkSpecTaskFromJSON(work.SpecJSON)
		workLoopDiagnostics.RecordWorkDecode(time.Since(decodeStarted))
		workLoopDiagnostics.RecordWorkSeen(work.JobID, work.Kind, specTask)

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

	if handled, result, runErr := processOptionalV7MemoryBenchmark(runCtx, client, work, runtimeMgr, gpuDetected); handled {
		if runErr != nil {
			slog.Warn("V7 memory benchmark execution failed", "job_id", work.JobID, "error", runErr)
		} else if result != nil {
			slog.Info("V7 memory benchmark completed", "job_id", work.JobID, "hash", result.ResultHashHex, "units", result.MeteringUnits)
		}
		if operatorRuntimeState != nil {
			operatorRuntimeState.finishJob(work, result, runErr)
		}
		return
	}

	if handled, result, runErr := processOptionalV7ModelBenchmark(runCtx, client, work, infMgr, runtimeMgr, gpuDetected); handled {
		if runErr != nil {
			slog.Warn("V7 model benchmark execution failed", "job_id", work.JobID, "error", runErr)
		} else if result != nil {
			slog.Info("V7 model benchmark completed", "job_id", work.JobID, "hash", result.ResultHashHex, "units", result.MeteringUnits)
		}
		if operatorRuntimeState != nil {
			operatorRuntimeState.finishJob(work, result, runErr)
		}
		return
	}

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
	executionStarted := time.Now()
	workLoopDiagnostics.RecordExecutionStart(work.JobID)
	result, runErr := engine.Execute(runCtx, work, executionContext{
		client:         client,
		gpus:           gpus,
		infMgr:         infMgr,
		runtimeManager: runtimeMgr,
		gpuDetected:    gpuDetected,
	})
	workLoopDiagnostics.RecordExecutionEnd(time.Since(executionStarted), runErr)
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

func processOptionalV7MemoryBenchmark(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, runtimeMgr *runtimeManager, gpuDetected bool) (bool, *runnerResultSnapshot, error) {
	identity, isBenchmark := v7memorybench.BenchmarkAssignmentIdentityFromJSON(work.SpecJSON)
	statusJobID := firstNonEmptyString(work.JobID, identity.JobID)
	if isBenchmark && operatorRuntimeState != nil {
		operatorRuntimeState.recordV7MemoryBenchmarkSeen(statusJobID, identity.RequestID)
	}

	executionStarted := time.Now()
	if isBenchmark && v7memorybench.BenchmarkEnabledFromEnv(os.Getenv) {
		workLoopDiagnostics.RecordExecutionStart(statusJobID)
	}
	receipt, receiptBuildTimings, handled, err := v7memorybench.ExecuteBenchmarkAssignmentWithReceiptTimings(ctx, work.SpecJSON, v7memorybench.ExecuteOptions{
		Getenv: os.Getenv,
	})
	if !handled {
		return false, nil, nil
	}
	workLoopDiagnostics.RecordExecutionEnd(time.Since(executionStarted), err)
	if err != nil && operatorRuntimeState != nil {
		operatorRuntimeState.recordV7MemoryBenchmarkRejected(statusJobID, err)
	}
	if err == nil && operatorRuntimeState != nil {
		operatorRuntimeState.recordV7MemoryBenchmarkExecuted(firstNonEmptyString(receipt.JobID, statusJobID))
	}

	metadataBuildStarted := time.Now()
	runtimeMeta := map[string]any{}
	if runtimeMgr != nil {
		runtimeMeta = runtimeMgr.ReceiptMetadata(gpuDetected)
	}
	extra := map[string]any{
		"executor":      v7memorybench.BenchmarkTask,
		"executor_kind": v7memorybench.BenchmarkTask,
		"task":          v7memorybench.BenchmarkTask,
	}
	if err != nil {
		receipt = v7memorybench.BuildBenchmarkRejectionReceipt(work.JobID, err)
		extra["exit_code"] = 1
		extra["error"] = "v7 memory benchmark rejected"
	} else {
		extra["exit_code"] = 0
	}
	exitCode, _ := extra["exit_code"].(int)
	metadata := receiptMetadataBase(work, runtimeMeta, receipt.Metadata, extra)
	metadataBuildDuration := time.Since(metadataBuildStarted)

	envelopeBuildStarted := time.Now()
	hubReceipt := hub.Receipt{
		JobID:         firstNonEmptyString(receipt.JobID, work.JobID),
		ResultHashHex: receipt.ResultHashHex,
		MeteringUnits: receipt.MeteringUnits,
		Metadata:      metadata,
	}
	snapshot := &runnerResultSnapshot{
		ResultHashHex: hubReceipt.ResultHashHex,
		MeteringUnits: hubReceipt.MeteringUnits,
		ExitCode:      exitCode,
		Metadata:      metadata,
	}
	envelopeBuildDuration := time.Since(envelopeBuildStarted)
	workLoopDiagnostics.RecordReceiptBuildTimings(workLoopReceiptTimingsFromMemorybench(receiptBuildTimings, metadataBuildDuration, envelopeBuildDuration))
	if submitErr := submitReceiptWithRetry(ctx, client, hubReceipt); submitErr != nil {
		if operatorRuntimeState != nil {
			operatorRuntimeState.recordV7MemoryBenchmarkReceiptFailed(hubReceipt.JobID, submitErr)
		}
		if err != nil {
			return true, snapshot, fmt.Errorf("%v; receipt submit failed: %w", err, submitErr)
		}
		return true, snapshot, submitErr
	}
	if operatorRuntimeState != nil {
		operatorRuntimeState.recordV7MemoryBenchmarkReceiptSubmitted(hubReceipt.JobID)
	}
	return true, snapshot, err
}

func workLoopReceiptTimingsFromMemorybench(receiptTimings v7memorybench.ReceiptBuildTimings, metadataBuild, envelopeBuild time.Duration) diagnostics.ReceiptBuildTimings {
	metadataBuildExtraUs := durationMicrosecondsForWorkLoop(metadataBuild)
	metadataBuildUs := receiptTimings.MetadataBuildUs + metadataBuildExtraUs
	envelopeBuildUs := durationMicrosecondsForWorkLoop(envelopeBuild)
	totalBuildUs := receiptTimings.TotalBuildUs + metadataBuildExtraUs + envelopeBuildUs
	minTotalUs := metadataBuildUs + receiptTimings.HashUs + receiptTimings.JSONMeasureUs + envelopeBuildUs
	if totalBuildUs < minTotalUs {
		totalBuildUs = minTotalUs
	}
	return diagnostics.ReceiptBuildTimingsFromMicroseconds(
		metadataBuildUs,
		receiptTimings.HashUs,
		receiptTimings.JSONMeasureUs,
		envelopeBuildUs,
		totalBuildUs,
	)
}

func durationMicrosecondsForWorkLoop(duration time.Duration) int64 {
	if duration <= 0 {
		return 0
	}
	return duration.Microseconds()
}

func processOptionalV7ModelBenchmark(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, infMgr *inference.Manager, runtimeMgr *runtimeManager, gpuDetected bool) (bool, *runnerResultSnapshot, error) {
	identity, isBenchmark := v7modelbench.ModelBenchmarkAssignmentIdentityFromJSON(work.SpecJSON)
	taskName := v7modelbench.ModelBenchmarkTask
	if !isBenchmark {
		if seriesIdentity, isSeriesBenchmark := v7modelbench.ModelBenchmarkSeriesAssignmentIdentityFromJSON(work.SpecJSON); isSeriesBenchmark {
			identity = seriesIdentity
			isBenchmark = true
			taskName = v7modelbench.ModelBenchmarkSeriesTask
		}
	}
	statusJobID := firstNonEmptyString(work.JobID, identity.JobID)
	if isBenchmark && v7modelbench.ModelBenchmarkEnabledFromEnv(os.Getenv) && v7ModelBenchmarkStatus != nil {
		v7ModelBenchmarkStatus.RecordSeen(statusJobID, identity.RequestID)
	}

	runner := newV7ModelBenchmarkRunner(infMgr, gpuDetected)
	executionStarted := time.Now()
	if isBenchmark && v7modelbench.ModelBenchmarkEnabledFromEnv(os.Getenv) {
		workLoopDiagnostics.RecordExecutionStart(statusJobID)
	}
	receipt, handled, err := v7modelbench.ExecuteModelBenchmarkAssignment(ctx, work.SpecJSON, runner, os.Getenv)
	if !handled {
		receipt, handled, err = v7modelbench.ExecuteModelBenchmarkSeriesAssignment(ctx, work.SpecJSON, runner, os.Getenv)
	}
	if !handled {
		return false, nil, nil
	}
	workLoopDiagnostics.RecordExecutionEnd(time.Since(executionStarted), err)
	if err != nil && v7ModelBenchmarkStatus != nil {
		v7ModelBenchmarkStatus.RecordError(err)
	}

	receiptBuildStarted := time.Now()
	runtimeMeta := map[string]any{}
	if runtimeMgr != nil {
		runtimeMeta = runtimeMgr.ReceiptMetadata(gpuDetected)
	}
	extra := map[string]any{
		"executor":      taskName,
		"executor_kind": taskName,
		"task":          taskName,
	}
	if strings.TrimSpace(receipt.ResultHashHex) == "" {
		if taskName == v7modelbench.ModelBenchmarkSeriesTask {
			receipt = v7modelbench.BuildModelBenchmarkSeriesRejectionReceipt(work.JobID, err)
		} else {
			receipt = v7modelbench.BuildModelBenchmarkRejectionReceipt(work.JobID, err)
		}
	}
	exitCode := 0
	if err != nil {
		exitCode = 1
		extra["exit_code"] = 1
		if taskName == v7modelbench.ModelBenchmarkSeriesTask {
			extra["error"] = "v7 model benchmark series failed"
		} else {
			extra["error"] = "v7 model benchmark failed"
		}
	} else {
		extra["exit_code"] = 0
	}
	metadata := receiptMetadataBase(work, runtimeMeta, receipt.Metadata, extra)
	hubReceipt := hub.Receipt{
		JobID:         firstNonEmptyString(receipt.JobID, work.JobID),
		ResultHashHex: receipt.ResultHashHex,
		MeteringUnits: receipt.MeteringUnits,
		Metadata:      metadata,
	}
	snapshot := &runnerResultSnapshot{
		ResultHashHex: hubReceipt.ResultHashHex,
		MeteringUnits: hubReceipt.MeteringUnits,
		ExitCode:      exitCode,
		Metadata:      metadata,
	}
	workLoopDiagnostics.RecordReceiptBuild(time.Since(receiptBuildStarted))
	if err == nil && v7ModelBenchmarkStatus != nil {
		v7ModelBenchmarkStatus.RecordExecuted(hubReceipt.JobID)
	}
	if client == nil {
		submitErr := fmt.Errorf("hub client unavailable")
		workLoopDiagnostics.RecordReceiptSubmitStart(hubReceipt.JobID, 1)
		workLoopDiagnostics.RecordReceiptSubmitEnd(0, submitErr)
		if v7ModelBenchmarkStatus != nil {
			v7ModelBenchmarkStatus.RecordReceiptFailed(hubReceipt.JobID, submitErr)
		}
		if err != nil {
			return true, snapshot, fmt.Errorf("%v; receipt submit failed: %w", err, submitErr)
		}
		return true, snapshot, submitErr
	}
	submitStarted := time.Now()
	workLoopDiagnostics.RecordReceiptSubmitStart(hubReceipt.JobID, 1)
	submitErr := client.SubmitReceipt(ctx, hubReceipt)
	workLoopDiagnostics.RecordReceiptSubmitEnd(time.Since(submitStarted), submitErr)
	if submitErr != nil {
		if v7ModelBenchmarkStatus != nil {
			v7ModelBenchmarkStatus.RecordReceiptFailed(hubReceipt.JobID, submitErr)
		}
		if err != nil {
			return true, snapshot, fmt.Errorf("%v; receipt submit failed: %w", err, submitErr)
		}
		return true, snapshot, submitErr
	}
	if v7ModelBenchmarkStatus != nil {
		v7ModelBenchmarkStatus.RecordReceiptSubmitted(hubReceipt.JobID)
	}
	return true, snapshot, err
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
	receipt = prepareReceiptForSubmission(receipt)
	const maxAttempts = 5
	delay := 2 * time.Second
	var lastErr error
	submitStarted := time.Now()
	for i := 0; i < maxAttempts; i++ {
		workLoopDiagnostics.RecordReceiptSubmitStart(receipt.JobID, i+1)
		if err := client.SubmitReceipt(ctx, receipt); err != nil {
			lastErr = err
			slog.Warn("receipt submission attempt failed", "job_id", receipt.JobID, "attempt", i+1, "error", err)
			select {
			case <-ctx.Done():
				finalErr := fmt.Errorf("context cancelled during receipt retry: %w", lastErr)
				workLoopDiagnostics.RecordReceiptSubmitEnd(time.Since(submitStarted), finalErr)
				return finalErr
			case <-time.After(delay):
			}
			delay = time.Duration(float64(delay) * 2)
			if delay > 30*time.Second {
				delay = 30 * time.Second
			}
			continue
		}
		workLoopDiagnostics.RecordReceiptSubmitEnd(time.Since(submitStarted), nil)
		return nil
	}
	finalErr := fmt.Errorf("receipt submission failed after %d attempts: %w", maxAttempts, lastErr)
	workLoopDiagnostics.RecordReceiptSubmitEnd(time.Since(submitStarted), finalErr)
	return finalErr
}

func prepareReceiptForSubmission(receipt hub.Receipt) hub.Receipt {
	if strings.TrimSpace(os.Getenv(v7ProofFlagEnv)) != "1" {
		return receipt
	}

	metadata, outputBytes, hasOutputBytes := cloneReceiptMetadataWithoutV7ProofSources(receipt.Metadata)
	if metadata == nil {
		metadata = map[string]any{}
	}

	runnerKind := firstNonEmptyString(
		metadataString(metadata, "runner_kind"),
		metadataString(metadata, "executor"),
		metadataString(metadata, "executor_kind"),
		"node_agent",
	)

	if hasOutputBytes {
		now := time.Now().UnixMilli()
		startedAt := metadataInt64OrDefault(metadata, now, "started_at_unix_ms", "execution_started_at_unix_ms")
		finishedAt := metadataInt64OrDefault(metadata, now, "finished_at_unix_ms", "execution_finished_at_unix_ms")
		if finishedAt < startedAt {
			finishedAt = startedAt
		}

		proof, err := v7proofrunner.BuildProofCarryingOutput(v7proofrunner.RunnerResultInput{
			JobID: firstNonEmptyString(
				strings.TrimSpace(receipt.JobID),
				metadataString(metadata, "job_id"),
			),
			AssignmentID: firstNonEmptyString(
				metadataString(metadata, "assignment_id"),
				strings.TrimSpace(receipt.JobID),
			),
			NodeID: firstNonEmptyString(
				metadataString(metadata, "node_id"),
				metadataString(metadata, "node_public_key"),
				strings.TrimSpace(os.Getenv("RYV_NODE_ID")),
				"node-agent",
			),
			RunnerKind: runnerKind,
			AgentVersion: firstNonEmptyString(
				metadataString(metadata, "agent_version"),
				version,
				"dev",
			),
			ModelID: firstNonEmptyString(
				metadataString(metadata, "model_id"),
				metadataString(metadata, "model"),
				metadataString(metadata, "runtime_model_id"),
				"unknown-model.gguf",
			),
			ModelRevision: firstNonEmptyString(
				metadataString(metadata, "model_revision"),
				metadataString(metadata, "revision"),
				"unknown",
			),
			QuantizationID: firstNonEmptyString(
				metadataString(metadata, "quantization_id"),
				metadataString(metadata, "quantization"),
				"unknown",
			),
			OutputBytes:      outputBytes,
			MeteringUnits:    int64FromUint64(receipt.MeteringUnits),
			ArtifactKind:     v7ProofArtifactKind(metadata),
			StartedAtUnixMs:  startedAt,
			FinishedAtUnixMs: finishedAt,
		})
		if err == nil {
			metadata[v7ProofMetadataKey] = fullV7ProofMetadata(proof)
			receipt.Metadata = metadata
			return receipt
		}
		slog.Warn("failed to build V7 proof metadata; submitting legacy receipt metadata", "job_id", receipt.JobID, "error", err)
		metadata[v7ProofMetadataKey] = partialV7ProofMetadata(receipt, runnerKind, "build_failed")
		receipt.Metadata = metadata
		return receipt
	}

	metadata[v7ProofMetadataKey] = partialV7ProofMetadata(receipt, runnerKind, "unavailable_no_output_bytes")
	receipt.Metadata = metadata
	return receipt
}

func cloneReceiptMetadataWithoutV7ProofSources(metadata map[string]any) (map[string]any, []byte, bool) {
	if metadata == nil {
		return nil, nil, false
	}
	out := make(map[string]any, len(metadata))
	var outputBytes []byte
	hasOutputBytes := false
	for key, value := range metadata {
		switch key {
		case v7ProofOutputBytesMetadataKey, v7ProofArtifactBytesMetadataKey:
			if !hasOutputBytes {
				if b, ok := metadataBytes(value); ok {
					outputBytes = b
					hasOutputBytes = true
				}
			}
			continue
		default:
			out[key] = value
		}
	}
	return out, outputBytes, hasOutputBytes
}

func metadataBytes(value any) ([]byte, bool) {
	switch v := value.(type) {
	case []byte:
		return append([]byte(nil), v...), true
	case json.RawMessage:
		return append([]byte(nil), v...), true
	case string:
		return []byte(v), true
	default:
		return nil, false
	}
}

func fullV7ProofMetadata(proof v7proofrunner.ProofCarryingRunnerOutput) map[string]any {
	return map[string]any{
		"output_hash":           proof.OutputHash,
		"output_bytes":          proof.OutputBytes,
		"artifact_object_id":    proof.ArtifactObjectID,
		"artifact_manifest":     proof.ArtifactManifest,
		"evidence_payload":      proof.EvidencePayload,
		"cas_object_references": proof.CASObjectReferences,
		"proof_status":          "complete",
	}
}

func partialV7ProofMetadata(receipt hub.Receipt, runnerKind string, proofStatus string) map[string]any {
	objectID := v7ProofObjectIDFromResultHash(receipt.ResultHashHex)
	casReferences := []string{}
	if objectID != "" {
		casReferences = append(casReferences, objectID)
	}
	return map[string]any{
		"output_hash":           objectID,
		"output_bytes":          int64(0),
		"artifact_object_id":    objectID,
		"artifact_manifest":     nil,
		"evidence_payload":      nil,
		"cas_object_references": casReferences,
		"proof_status":          proofStatus,
		"result_hash":           strings.TrimSpace(receipt.ResultHashHex),
		"metering_units":        receipt.MeteringUnits,
		"runner_kind":           strings.TrimSpace(runnerKind),
		"job_id":                strings.TrimSpace(receipt.JobID),
	}
}

func v7ProofObjectIDFromResultHash(resultHash string) string {
	hash := strings.TrimSpace(resultHash)
	if strings.HasPrefix(hash, "sha256:") {
		if validLowerSHA256Hex(strings.TrimPrefix(hash, "sha256:")) {
			return hash
		}
		return ""
	}
	if validLowerSHA256Hex(hash) {
		return "sha256:" + hash
	}
	return ""
}

func validLowerSHA256Hex(value string) bool {
	if len(value) != 64 {
		return false
	}
	for _, ch := range value {
		if (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') {
			continue
		}
		return false
	}
	return true
}

func metadataString(metadata map[string]any, key string) string {
	if metadata == nil {
		return ""
	}
	switch v := metadata[key].(type) {
	case string:
		return strings.TrimSpace(v)
	case fmt.Stringer:
		return strings.TrimSpace(v.String())
	default:
		return ""
	}
}

func metadataInt64OrDefault(metadata map[string]any, fallback int64, keys ...string) int64 {
	for _, key := range keys {
		value, ok := metadataInt64(metadata, key)
		if ok {
			return value
		}
	}
	return fallback
}

func metadataInt64(metadata map[string]any, key string) (int64, bool) {
	if metadata == nil {
		return 0, false
	}
	switch v := metadata[key].(type) {
	case int:
		return int64(v), true
	case int8:
		return int64(v), true
	case int16:
		return int64(v), true
	case int32:
		return int64(v), true
	case int64:
		return v, true
	case uint:
		if uint64(v) > uint64(maxInt64) {
			return maxInt64, true
		}
		return int64(v), true
	case uint8:
		return int64(v), true
	case uint16:
		return int64(v), true
	case uint32:
		return int64(v), true
	case uint64:
		return int64FromUint64(v), true
	case float64:
		if v >= 0 && math.Trunc(v) == v && v <= float64(maxInt64) {
			return int64(v), true
		}
	case json.Number:
		if n, err := v.Int64(); err == nil {
			return n, true
		}
	case string:
		if n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64); err == nil {
			return n, true
		}
	}
	return 0, false
}

const maxInt64 = int64(1<<63 - 1)

func int64FromUint64(value uint64) int64 {
	if value > uint64(maxInt64) {
		return maxInt64
	}
	return int64(value)
}

func firstNonEmptyString(values ...string) string {
	for _, value := range values {
		if s := strings.TrimSpace(value); s != "" {
			return s
		}
	}
	return ""
}

func v7ProofArtifactKind(metadata map[string]any) v7artifact.ArtifactKind {
	kind := strings.ToLower(firstNonEmptyString(
		metadataString(metadata, "artifact_kind"),
		metadataString(metadata, "artifact_mime"),
		metadataString(metadata, "mime_type"),
	))
	switch kind {
	case "result_json", "json", "application/json":
		return v7artifact.ArtifactKindResultJSON
	case "image", "image/png", "image/jpeg", "image/webp":
		return v7artifact.ArtifactKindImage
	case "text", "text/plain":
		return v7artifact.ArtifactKindText
	case "model_delta":
		return v7artifact.ArtifactKindModelDelta
	case "evidence_bundle":
		return v7artifact.ArtifactKindEvidenceBundle
	case "runner_log":
		return v7artifact.ArtifactKindRunnerLog
	case "", "generic_blob", "application/octet-stream":
		return v7artifact.ArtifactKindGenericBlob
	default:
		return v7artifact.ArtifactKind(kind)
	}
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
	localFluxPreparing := publicAIReady && localFlux2KleinHardwareEligible(caps, gpuReady) && localFlux2KleinPreparing(caps, diskGB, gpuReady)
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
	parts = append(parts, boolStatusToken("cap:image_gen", publicAIReady && (runtimeSnap.GPUReady || localFluxReady)))
	parts = append(parts, boolStatusToken("cap:ryvion_runtime", publicAIReady && (localFluxReady || localFluxPreparing || localFluxPrepareEligible)))
	if localFluxPreparing {
		parts = append(parts, "runtime:image:"+flux2Klein4BLocalModel+":preparing:1")
	}
	if localFluxFastReady {
		parts = append(parts, "runtime:image:"+flux2Klein4BLocalModel)
		parts = append(parts, "model:"+flux2Klein4BLocalModel)
		parts = append(parts, fmt.Sprintf("runtime:image:%s:min_vram_mb:%d", flux2Klein4BLocalModel, flux2Klein4BMinVRAMMB))
	} else if localFluxPrepareEligible {
		parts = append(parts, "runtime:image:"+flux2Klein4BLocalModel+":eligible:1")
		if localFlux2KleinFastGPUEligible(caps, gpuReady) {
			parts = append(parts, fmt.Sprintf("runtime:image:%s:min_vram_mb:%d", flux2Klein4BLocalModel, flux2Klein4BMinVRAMMB))
		} else {
			parts = append(parts, "runtime:image:"+flux2Klein4BLocalModel+":mode:cpu-preview")
			parts = append(parts, fmt.Sprintf("runtime:image:%s:min_ram_gb:%d", flux2Klein4BLocalModel, flux2Klein4BMinRAMGB))
			parts = append(parts, fmt.Sprintf("runtime:image:%s:min_cpu_cores:%d", flux2Klein4BLocalModel, flux2Klein4BMinCPUCores))
		}
	} else if localFluxReady {
		parts = append(parts, "runtime:image:"+flux2Klein4BLocalModel)
		parts = append(parts, "model:"+flux2Klein4BLocalModel)
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
