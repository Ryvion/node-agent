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
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	artifact "github.com/Ryvion/ryvion-node/internal/artifacts/core"
	memorybench "github.com/Ryvion/ryvion-node/internal/benchmarks/memory"
	tensorbench "github.com/Ryvion/ryvion-node/internal/benchmarks/tensor"
	capshardware "github.com/Ryvion/ryvion-node/internal/capabilities/hardware"
	capability "github.com/Ryvion/ryvion-node/internal/capabilities/passport"
	"github.com/Ryvion/ryvion-node/internal/diagnostics"
	"github.com/Ryvion/ryvion-node/internal/hub"
	heartbeat "github.com/Ryvion/ryvion-node/internal/hub/heartbeat"
	"github.com/Ryvion/ryvion-node/internal/hw"
	"github.com/Ryvion/ryvion-node/internal/inference"
	inferencebench "github.com/Ryvion/ryvion-node/internal/inference/benchmark"
	routedinference "github.com/Ryvion/ryvion-node/internal/inference/routed"
	modelbench "github.com/Ryvion/ryvion-node/internal/models/benchmark"
	modelpolicy "github.com/Ryvion/ryvion-node/internal/models/policy"
	modelprepare "github.com/Ryvion/ryvion-node/internal/models/prepare"
	modelwarm "github.com/Ryvion/ryvion-node/internal/models/warm"
	netprofile "github.com/Ryvion/ryvion-node/internal/network/profile"
	"github.com/Ryvion/ryvion-node/internal/nodekey"
	onboarding "github.com/Ryvion/ryvion-node/internal/operator/onboarding"
	proofrunner "github.com/Ryvion/ryvion-node/internal/receipts/proofrunner"
	"github.com/Ryvion/ryvion-node/internal/runtimeexec"
	llamacpp "github.com/Ryvion/ryvion-node/internal/runtimes/llamacpp"
	backendprobe "github.com/Ryvion/ryvion-node/internal/runtimes/probe"
	_ "github.com/Ryvion/ryvion-node/internal/runtimes/tensoraccess"
	sandbox "github.com/Ryvion/ryvion-node/internal/sandbox"
	"github.com/Ryvion/ryvion-node/internal/update"
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

var hubNetworkProfile = netprofile.NewRollingHubProfile(8, "")

// jobActive is set to 1 while a job is being processed.
// The update check reads this to avoid restarting during active work.
var jobActive atomic.Int32

// latestHubVersion stores the most recent agent version advertised by the hub.
// Written by heartbeat goroutine, read by heartbeat/work-loop auto-update checks.
var latestHubVersion atomic.Value // string

// v7CapabilityHeartbeatRequests wakes the heartbeat loop when local inventory
// or hub connectivity changes before the next periodic tick.
var v7CapabilityHeartbeatRequests = make(chan string, 1)

type heartbeatSendResult struct {
	hubOK               bool
	capabilitySubmitted bool
	latestVersion       string
	payloadSummary      operatorHeartbeatPayloadSummary
}

const autoUpdateRetryInterval = 5 * time.Minute

var (
	autoUpdateInProgress    atomic.Int32
	lastAutoUpdateAttemptMs atomic.Int64
	applyAutoUpdate         = update.Apply
	restartAfterAutoUpdate  = update.Restart
	autoUpdateNow           = time.Now
)

type capabilityState struct {
	mu         sync.RWMutex
	caps       hw.CapSet
	deviceType string
}

func newCapabilityState(caps hw.CapSet, deviceType string) *capabilityState {
	return &capabilityState{
		caps:       caps,
		deviceType: strings.TrimSpace(deviceType),
	}
}

func (s *capabilityState) Snapshot() (hw.CapSet, string) {
	if s == nil {
		return hw.CapSet{}, ""
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.caps, s.deviceType
}

func (s *capabilityState) Update(caps hw.CapSet, deviceType string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.caps = caps
	s.deviceType = strings.TrimSpace(deviceType)
}

func capSetGPUDetected(caps hw.CapSet) bool {
	return strings.TrimSpace(caps.GPUModel) != "" || caps.VRAMBytes > 0
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

func improvedHardwareCaps(current hw.CapSet, detected hw.CapSet) (hw.CapSet, bool) {
	if !capSetGPUDetected(detected) {
		return current, false
	}
	if !capSetGPUDetected(current) {
		return detected, true
	}
	currentModel := strings.TrimSpace(current.GPUModel)
	detectedModel := strings.TrimSpace(detected.GPUModel)
	if currentModel == "" && detectedModel != "" {
		return detected, true
	}
	if detected.VRAMBytes > current.VRAMBytes {
		return detected, true
	}
	sameModel := currentModel != "" && detectedModel == currentModel
	if sameModel && strings.TrimSpace(current.Sensors) == "" && strings.TrimSpace(detected.Sensors) != "" {
		return detected, true
	}
	if sameModel && strings.TrimSpace(current.GfxVersion) == "" && strings.TrimSpace(detected.GfxVersion) != "" {
		return detected, true
	}
	return current, false
}

func detectImprovedHardwareCaps(deviceFlag string, current hw.CapSet, detect func(string) hw.CapSet) (hw.CapSet, string, bool) {
	if detect == nil {
		detect = hw.DetectCaps
	}
	detected := detect(deviceFlag)
	updated, ok := improvedHardwareCaps(current, detected)
	if !ok {
		return current, resolveDeviceType(deviceFlag, current), false
	}
	return updated, resolveDeviceType(deviceFlag, updated), true
}

func refreshCapabilityStateFromDetection(deviceFlag string, state *capabilityState, detect func(string) hw.CapSet) (hw.CapSet, string, bool) {
	current, currentDeviceType := state.Snapshot()
	updated, deviceType, changed := detectImprovedHardwareCaps(deviceFlag, current, detect)
	if !changed {
		if strings.TrimSpace(currentDeviceType) != "" {
			deviceType = currentDeviceType
		}
		return current, deviceType, false
	}
	state.Update(updated, deviceType)
	return updated, deviceType, true
}

func capabilityRegistrationKey(caps hw.CapSet, deviceType string) string {
	return strings.Join([]string{
		strings.TrimSpace(deviceType),
		strings.TrimSpace(caps.GPUModel),
		strconv.FormatUint(caps.VRAMBytes, 10),
		strings.TrimSpace(caps.Sensors),
		strings.TrimSpace(caps.GfxVersion),
	}, "\x00")
}

func maybeApplyAutoUpdate(ctx context.Context, hubURL string, currentVersion string, latestVersion string, reason string) bool {
	latestVersion = strings.TrimSpace(latestVersion)
	if !update.NeedsUpdate(currentVersion, latestVersion) {
		return false
	}
	if jobActive.Load() != 0 {
		slog.Info("update available but job in progress, deferring", "latest", latestVersion, "reason", reason)
		return false
	}
	if !autoUpdateInProgress.CompareAndSwap(0, 1) {
		return false
	}
	defer autoUpdateInProgress.Store(0)

	nowMs := autoUpdateNow().UnixMilli()
	for {
		lastMs := lastAutoUpdateAttemptMs.Load()
		if lastMs > 0 && time.Duration(nowMs-lastMs)*time.Millisecond < autoUpdateRetryInterval {
			return false
		}
		if lastAutoUpdateAttemptMs.CompareAndSwap(lastMs, nowMs) {
			break
		}
	}

	slog.Info("update available", "current", currentVersion, "latest", latestVersion, "reason", reason)
	if err := applyAutoUpdate(ctx, latestVersion); err != nil {
		slog.Warn("auto-update failed", "error", err, "reason", reason)
		return false
	}
	slog.Info("update applied, restarting", "reason", reason)
	if restartErr := restartAfterAutoUpdate(); restartErr != nil {
		slog.Warn("restart failed — update will take effect on next manual restart", "error", restartErr)
	}
	return true
}

const (
	v7ProofFlagEnv                  = "RYV_NODE_V7_PROOF"
	v7ProofMetadataKey              = "v7_proof"
	v7ProofOutputBytesMetadataKey   = "_v7_proof_output_bytes"
	v7ProofArtifactBytesMetadataKey = "_v7_proof_artifact_bytes"
	legacyNativeInferenceFlagEnv    = "RYV_NODE_LEGACY_NATIVE_INFERENCE"
)

var (
	workLoopDiagnostics          = diagnostics.NewWorkLoopDiagnostics()
	v7ModelBenchmarkStatus       = modelbench.NewLocalStatus()
	v7TensorPlaneBenchmarkStatus = tensorbench.NewLocalStatus()
	v7BackendBenchmarkStatus     = llamacpp.NewBackendBenchmarkLocalStatus()
	v7InferenceBenchmarkStatus   = inferencebench.NewLocalStatus()
	v7DashboardInferenceStatus   = routedinference.NewLocalStatus()
	v7ModelPrepareStatus         = modelprepare.NewLocalStatus()
	v7ModelWarmStatus            = modelwarm.NewLocalStatus()
	newV7ModelBenchmarkRunner    = func(infMgr *inference.Manager, gpuDetected bool) modelbench.ModelBenchmarkRunner {
		return modelbench.NativeInferenceModelBenchmarkRunner{
			Native:           infMgr,
			AgentVersion:     version,
			OS:               runtime.GOOS,
			Arch:             runtime.GOARCH,
			GPUDetected:      gpuDetected,
			RuntimeAvailable: inference.NativeRuntimeAvailable,
		}
	}
	newV7LlamaCppBackendBenchmarkRunner = func() llamacpp.BackendBenchmarkRunner {
		sidecar := llamacpp.BenchmarkSidecar(llamacpp.NewManagerFromEnv())
		if operatorRuntimeState != nil {
			sidecar = operatorRuntimeState.llamaCppManager()
		}
		return llamacpp.BenchmarkRunner{
			Sidecar: sidecar,
			Client:  llamacpp.OpenAIClient{},
		}
	}
	newV7BackendInferenceBenchmarkRunner = func() inferencebench.BenchmarkRunner {
		var sidecar inferencebench.LlamaCppSidecar = llamacpp.NewManagerFromEnv()
		var keepWarm inferencebench.KeepWarmChecker
		if operatorRuntimeState != nil {
			sidecar = operatorRuntimeState.llamaCppManager()
			keepWarm = operatorRuntimeState.llamaCppResidencyKeeper()
		}
		return inferencebench.LlamaCppBenchmarkRunner{
			Sidecar:  sidecar,
			KeepWarm: keepWarm,
			Client:   llamacpp.OpenAIClient{},
			Getenv:   os.Getenv,
		}
	}
	newV7DashboardInferenceRunner = func() routedinference.Runner {
		var sidecar routedinference.LlamaCppSidecar = llamacpp.NewManagerFromEnv()
		var keepWarm routedinference.KeepWarmChecker
		if operatorRuntimeState != nil {
			sidecar = operatorRuntimeState.llamaCppManager()
			keepWarm = operatorRuntimeState.llamaCppResidencyKeeper()
		}
		return routedinference.LlamaCppRunner{
			Sidecar:  sidecar,
			KeepWarm: keepWarm,
			Client:   llamacpp.OpenAIClient{},
			Getenv:   os.Getenv,
			Policy:   buildDashboardInferencePolicy(os.Getenv, buildHardwareCapacityStatus),
		}
	}
)

func buildDashboardInferencePolicy(getenv func(string) string, detectHardware func(string) capshardware.CapacityInventory) modelpolicy.Policy {
	if getenv == nil {
		getenv = os.Getenv
	}
	basePolicy := modelpolicy.FromConfigSource(modelpolicy.ConfigSource{Getenv: getenv})
	if detectHardware == nil {
		detectHardware = buildHardwareCapacityStatus
	}
	return buildDerivedModelPolicyStatus(basePolicy, detectHardware(basePolicy.CacheDir))
}

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
	if len(os.Args) > 1 && os.Args[1] == "heartbeat-preview" {
		runHeartbeatPreview(os.Args[2:])
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "backend-probe" {
		runBackendProbe(os.Args[2:])
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "llamacpp-bench" {
		runLlamaCppBench(os.Args[2:])
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
	if len(os.Args) > 1 && os.Args[1] == "tensorplane-selftest" {
		runTensorPlaneSelfTest(os.Args[2:])
		return
	}
	if len(os.Args) > 1 && os.Args[1] == "tensorplane-provider-selftest" {
		runTensorPlaneProviderSelfTest(os.Args[2:])
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

func runHeartbeatPreview(args []string) {
	fs := flag.NewFlagSet("heartbeat-preview", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	jsonOutput := fs.Bool("json", false, "Emit compact JSON")
	if err := fs.Parse(args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "usage: ryvion-node heartbeat-preview --json")
		os.Exit(2)
	}

	nodeID, err := nodekey.PublicKeyHex(strings.TrimSpace(os.Getenv("RYV_KEY_PATH")))
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load node identity: %v\n", err)
		os.Exit(1)
	}
	caps := hw.DetectCaps("")
	deviceType := resolveDeviceType("", caps)
	declaredCountry, err := resolveInitialDeclaredCountry("")
	if err != nil {
		declaredCountry = strings.TrimSpace(os.Getenv("RYV_DECLARED_COUNTRY"))
	}
	backendRuntimes := buildStandaloneBackendRuntimes(context.Background())
	payload, err := buildV7HeartbeatPayloadForNodeWithBackendRuntimes(nodeID, caps, deviceType, declaredCountry, nil, nil, backendRuntimes)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to build V7 heartbeat preview: %v\n", err)
		os.Exit(1)
	}
	preview := buildV7HeartbeatPreviewResponse(nodeID, *payload)
	if *jsonOutput {
		if err := json.NewEncoder(os.Stdout).Encode(preview); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
	fmt.Printf("v7 heartbeat preview node_id=%s backend_candidates=%d gguf_models=%d llama_cpp_probe=%t\n",
		preview.NodeID,
		preview.FieldPresence.BackendCandidatesLen,
		preview.FieldPresence.GGUFModelsLen,
		preview.FieldPresence.LlamaCPPProbePresent,
	)
	os.Exit(0)
}

func runBackendProbe(args []string) {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: ryvion-node backend-probe llamacpp --json")
		os.Exit(2)
	}
	switch args[0] {
	case "llamacpp":
		runLlamaCPPBackendProbe(args[1:])
	default:
		fmt.Fprintf(os.Stderr, "unknown backend probe %q\n", args[0])
		os.Exit(2)
	}
}

func runLlamaCPPBackendProbe(args []string) {
	fs := flag.NewFlagSet("backend-probe llamacpp", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	jsonOutput := fs.Bool("json", false, "Emit compact JSON")
	if err := fs.Parse(args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if fs.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "usage: ryvion-node backend-probe llamacpp --json")
		os.Exit(2)
	}

	probe := backendprobe.ProbeLlamaCPP(backendprobe.Detector{})
	if *jsonOutput {
		if err := json.NewEncoder(os.Stdout).Encode(probe); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
	fmt.Printf("llama.cpp available=%t reason=%s\n", probe.Available, probe.Reason)
	os.Exit(0)
}

func runLlamaCppBench(args []string) {
	config, jsonOutput, err := parseLlamaCppBenchFlags(args)
	if err != nil {
		if err == flag.ErrHelp {
			os.Exit(0)
		}
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if !llamacpp.BenchmarkEnabledFromEnv(os.Getenv) {
		fmt.Fprintf(os.Stderr, "%s=1 required for llama.cpp benchmark\n", llamacpp.EnvBenchmark)
		os.Exit(2)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(config.TimeoutMs)*time.Millisecond)
	defer cancel()
	runner := llamacpp.BenchmarkRunner{
		Sidecar: llamacpp.NewManagerFromEnv(),
		Client:  llamacpp.OpenAIClient{},
	}
	snapshot := runner.Run(ctx, config)
	fmt.Println(llamacpp.FormatBenchmarkStatus(snapshot, jsonOutput))
	if snapshot.Status != llamacpp.BenchmarkStatusCompleted {
		os.Exit(1)
	}
	os.Exit(0)
}

func runDoctor(args []string) {
	fs := flag.NewFlagSet("doctor", flag.ExitOnError)
	hubURL := fs.String("hub", firstNonEmpty(strings.TrimSpace(os.Getenv("RYV_HUB_URL")), "https://api.ryvion.ai"), "Hub orchestrator base URL")
	dataDir := fs.String("data-dir", strings.TrimSpace(os.Getenv("RYV_DATA_DIR")), "Node data directory to check")
	deviceType := fs.String("type", "", "Node device type (gpu|cpu|mobile|iot)")
	_ = fs.Parse(args)

	report := onboarding.RunBasicOnboardingChecksWithOptions(onboarding.CheckOptions{
		AgentVersion: version,
		HubURL:       *hubURL,
		DataDir:      *dataDir,
		DeviceType:   *deviceType,
	})
	fmt.Println(onboarding.FormatOnboardingReport(report))
	if report.HasHardErrors() {
		os.Exit(1)
	}
	os.Exit(0)
}

func runMemoryBenchSelfTest(args []string) {
	defaults := memorybench.DefaultMemoryBenchSelfTestConfig()
	fs := flag.NewFlagSet("memorybench-selftest", flag.ExitOnError)
	tokenCount := fs.Int("tokens", defaults.TokenCount, "Synthetic token count")
	valueDim := fs.Int("dim", defaults.ValueDim, "Synthetic attention value dimension")
	seed := fs.Int64("seed", defaults.Seed, "Synthetic request seed")
	jsonOutput := fs.Bool("json", false, "Print JSON output")
	_ = fs.Parse(args)

	result, err := memorybench.RunMemoryBenchSelfTest(memorybench.MemoryBenchSelfTestConfig{
		Seed:       *seed,
		TokenCount: *tokenCount,
		ValueDim:   *valueDim,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println(memorybench.FormatMemoryBenchSelfTestResult(result, *jsonOutput))
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
	fmt.Println(modelbench.FormatModelBenchmarkSelfTestResult(result, jsonOutput))
	os.Exit(0)
}

func runTensorPlaneSelfTest(args []string) {
	config, jsonOutput, err := parseTensorPlaneSelfTestFlags(args)
	if err != nil {
		if err == flag.ErrHelp {
			os.Exit(0)
		}
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	result, err := tensorbench.RunTensorPlaneProbe(config)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println(tensorbench.FormatTensorPlaneProbeResult(result, jsonOutput))
	os.Exit(0)
}

func runTensorPlaneProviderSelfTest(args []string) {
	req, jsonOutput, err := parseTensorPlaneProviderSelfTestFlags(args)
	if err != nil {
		if err == flag.ErrHelp {
			os.Exit(0)
		}
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	result, err := tensorbench.RunProviderBackedTensorPlaneProbe(context.Background(), req)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println(tensorbench.FormatProviderBackedTensorPlaneProbeResult(result, jsonOutput))
	os.Exit(0)
}

func parseModelBenchSelfTestFlags(args []string) (modelbench.ModelBenchmarkSelfTestConfig, bool, error) {
	defaults := modelbench.DefaultModelBenchmarkSelfTestConfig()
	fs := flag.NewFlagSet("modelbench-selftest", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	modelID := fs.String("model", defaults.ModelID, "Native model ID")
	maxTokens := fs.Int("tokens", defaults.MaxTokens, "Maximum generated tokens")
	timeoutRaw := fs.String("timeout", fmt.Sprintf("%dms", defaults.TimeoutMs), "Benchmark timeout as duration or milliseconds")
	jsonOutput := fs.Bool("json", false, "Print JSON output")
	if err := fs.Parse(args); err != nil {
		return modelbench.ModelBenchmarkSelfTestConfig{}, false, err
	}
	timeoutMs, err := parseModelBenchTimeoutMs(*timeoutRaw)
	if err != nil {
		return modelbench.ModelBenchmarkSelfTestConfig{}, false, err
	}
	return modelbench.ModelBenchmarkSelfTestConfig{
		ModelID:   *modelID,
		MaxTokens: *maxTokens,
		TimeoutMs: timeoutMs,
	}, *jsonOutput, nil
}

func parseLlamaCppBenchFlags(args []string) (llamacpp.BenchmarkConfig, bool, error) {
	defaults := llamacpp.DefaultBenchmarkConfig()
	fs := flag.NewFlagSet("llamacpp-bench", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	modelID := fs.String("model", defaults.ModelID, "llama.cpp model ID or model filename")
	maxTokens := fs.Int("max-tokens", defaults.MaxTokens, "Maximum generated tokens")
	runs := fs.Int("runs", defaults.MeasuredRuns, "Measured benchmark runs")
	warmups := fs.Int("warmup-runs", defaults.WarmupRuns, "Warmup runs before measurement")
	timeoutRaw := fs.String("timeout", fmt.Sprintf("%dms", defaults.TimeoutMs), "Benchmark timeout as duration or milliseconds")
	jsonOutput := fs.Bool("json", false, "Print JSON output")
	if err := fs.Parse(args); err != nil {
		return llamacpp.BenchmarkConfig{}, false, err
	}
	if fs.NArg() != 0 {
		return llamacpp.BenchmarkConfig{}, false, fmt.Errorf("usage: ryvion-node llamacpp-bench --json --max-tokens 32 --runs 3")
	}
	timeoutMs, err := parseLlamaCppBenchTimeoutMs(*timeoutRaw)
	if err != nil {
		return llamacpp.BenchmarkConfig{}, false, err
	}
	config := llamacpp.BenchmarkConfig{
		ModelID:      *modelID,
		MaxTokens:    *maxTokens,
		Temperature:  0,
		TimeoutMs:    timeoutMs,
		Streaming:    true,
		MeasuredRuns: *runs,
		WarmupRuns:   *warmups,
	}
	if err := llamacpp.ValidateBenchmarkConfig(config); err != nil {
		return llamacpp.BenchmarkConfig{}, false, err
	}
	return llamacpp.NormalizeBenchmarkConfig(config), *jsonOutput, nil
}

func parseLlamaCppBenchTimeoutMs(raw string) (int64, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return llamacpp.DefaultBenchmarkConfig().TimeoutMs, nil
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

func parseTensorPlaneSelfTestFlags(args []string) (tensorbench.TensorPlaneProbeConfig, bool, error) {
	defaults := tensorbench.DefaultTensorPlaneProbeConfig()
	fs := flag.NewFlagSet("tensorplane-selftest", flag.ContinueOnError)
	tokens := fs.Int("tokens", defaults.Tokens, "Tensor page token count")
	headDim := fs.Int("head-dim", defaults.HeadDim, "Tensor page key/query head dimension")
	valueDim := fs.Int("value-dim", defaults.ValueDim, "Tensor page value dimension")
	dtype := fs.String("dtype", string(defaults.DType), "Tensor dtype (float32|float16)")
	seed := fs.Int64("seed", defaults.Seed, "Deterministic fixture seed")
	jsonOutput := fs.Bool("json", false, "Print JSON output")
	writeFixture := fs.String("write-fixture", "", "Write deterministic fixture JSON to path")
	readFixture := fs.String("read-fixture", "", "Read fixture JSON from path")
	if err := fs.Parse(args); err != nil {
		return tensorbench.TensorPlaneProbeConfig{}, false, err
	}
	return tensorbench.TensorPlaneProbeConfig{
		Tokens:           *tokens,
		HeadDim:          *headDim,
		ValueDim:         *valueDim,
		DType:            tensorbench.TensorDType(*dtype),
		Seed:             *seed,
		WriteFixturePath: *writeFixture,
		ReadFixturePath:  *readFixture,
	}, *jsonOutput, nil
}

func parseTensorPlaneProviderSelfTestFlags(args []string) (tensorbench.ProviderBackedProbeRequest, bool, error) {
	defaults := tensorbench.DefaultProviderBackedProbeRequest()
	fs := flag.NewFlagSet("tensorplane-provider-selftest", flag.ContinueOnError)
	provider := fs.String("provider", defaults.Provider, "Tensor access provider (noop|tensorplane_demo)")
	modelID := fs.String("model", defaults.ModelID, "Tensor access model ID")
	layerIndex := fs.Int("layer", defaults.LayerIndex, "Tensor layer index")
	tokens := fs.Int("tokens", defaults.Tokens, "Tensor page token count")
	headDim := fs.Int("head-dim", defaults.HeadDim, "Tensor page key/query head dimension")
	valueDim := fs.Int("value-dim", defaults.ValueDim, "Tensor page value dimension")
	dtype := fs.String("dtype", string(defaults.DType), "Tensor dtype (float32|float16)")
	seed := fs.Int64("seed", defaults.Seed, "Deterministic provider seed")
	jsonOutput := fs.Bool("json", false, "Print JSON output")
	if err := fs.Parse(args); err != nil {
		return tensorbench.ProviderBackedProbeRequest{}, false, err
	}
	return tensorbench.ProviderBackedProbeRequest{
		Provider:   *provider,
		ModelID:    *modelID,
		LayerIndex: *layerIndex,
		DType:      tensorbench.TensorDType(*dtype),
		Tokens:     *tokens,
		HeadDim:    *headDim,
		ValueDim:   *valueDim,
		Seed:       *seed,
	}, *jsonOutput, nil
}

func parseModelBenchTimeoutMs(raw string) (int64, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return modelbench.DefaultModelBenchmarkSelfTestConfig().TimeoutMs, nil
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

func runModelBenchSelfTestViaOperatorAPI(ctx context.Context, config modelbench.ModelBenchmarkSelfTestConfig, port string) (modelbench.ModelBenchmarkResult, error) {
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
		return modelbench.ModelBenchmarkResult{}, err
	}

	url := fmt.Sprintf("http://127.0.0.1:%s/api/v1/operator/v7/model-benchmark/run", strings.TrimSpace(port))
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err == nil {
		req.Header.Set("Content-Type", "application/json")
		if resp, doErr := http.DefaultClient.Do(req); doErr == nil {
			defer resp.Body.Close()
			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				var result modelbench.ModelBenchmarkResult
				if decodeErr := json.NewDecoder(resp.Body).Decode(&result); decodeErr != nil {
					return modelbench.ModelBenchmarkResult{}, decodeErr
				}
				if validationErr := modelbench.ValidateModelBenchmarkResult(result); validationErr != nil {
					return result, validationErr
				}
				return result, nil
			}
			_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
		}
	}

	runner := modelbench.NativeInferenceModelBenchmarkRunner{
		AgentVersion:     version,
		RuntimeAvailable: inference.NativeRuntimeAvailable,
	}
	return modelbench.RunModelBenchmarkSelfTest(ctx, runner, config)
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
	if err := update.ReconcileWindowsCanonicalBinary(); err != nil {
		slog.Warn("Windows canonical binary reconciliation failed", "error", err)
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
	defer func() {
		if err := client.Close(); err != nil {
			slog.Warn("failed to close hub transport", "error", err)
		}
	}()
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
	capState := newCapabilityState(caps, deviceType)
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
	defer operatorRuntimeState.stopLlamaCppSidecar(context.Background())
	defer operatorRuntimeState.stopSGLangSidecar(context.Background())
	startOperatorAPIServer(ctx, operatorRuntimeState, operatorAPIPort(flagUIPort))
	operatorRuntimeState.startLlamaCppResidencyKeeper(ctx)

	// Retry registration with backoff — on Windows the service starts before
	// Windows service startup can race the managed runtime, WSL2, and network.
	// Keep
	// the process alive so SCM doesn't exhaust its restart budget.
	regBackoff := 5 * time.Second
	for {
		if currentCaps, _ := capState.Snapshot(); !capSetGPUDetected(currentCaps) {
			if refreshedCaps, refreshedDeviceType, refreshed := refreshCapabilityStateFromDetection(flagDevice, capState, hw.DetectCaps); refreshed {
				if operatorRuntimeState != nil {
					operatorRuntimeState.setCapabilities(refreshedCaps, refreshedDeviceType)
				}
				slog.Info("GPU capabilities detected during registration retry",
					"device_type", refreshedDeviceType,
					"gpu_model", strings.TrimSpace(refreshedCaps.GPUModel),
					"vram_bytes", refreshedCaps.VRAMBytes)
			}
		}
		caps, deviceType = capState.Snapshot()
		if err := client.Register(ctx, hubCapabilitiesFromCaps(caps), deviceType, strings.TrimSpace(flagReferral), declaredCountry); err != nil {
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
	caps, deviceType = capState.Snapshot()
	registeredCapabilitiesKey := capabilityRegistrationKey(caps, deviceType)
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
	if legacyNativeInferenceManagerEnabled(os.Getenv) {
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
	} else {
		slog.Info("legacy native inference manager disabled; V7 llama.cpp sidecar owns model execution",
			"enable_env", legacyNativeInferenceFlagEnv)
	}
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
		healthReportLoop(ctx, client, capState, infMgr, runtimeMgr)
	}()

	go func() {
		defer func() {
			if r := recover(); r != nil {
				slog.Error("capability refresh goroutine panic", "error", r)
			}
		}()
		capabilityRefreshLoop(ctx, client, capState, flagDevice, strings.TrimSpace(flagReferral), declaredCountry, registeredCapabilitiesKey)
	}()

	// Independent heartbeat goroutine — keeps node "online" regardless of
	// what the work loop is doing (long-poll, job execution, etc.).
	go func() {
		defer func() {
			if r := recover(); r != nil {
				slog.Error("heartbeat goroutine panic", "error", r)
			}
		}()
		heartbeatLoop(ctx, client, capState, declaredCountry, infMgr, runtimeMgr, hubURL, version)
	}()

	// Work loop — fetch and process jobs.
	workLoop(ctx, client, flagGPUs, hubURL, version, infMgr, runtimeMgr, capState)
}

func capabilityRefreshLoop(ctx context.Context, client *hub.Client, capState *capabilityState, deviceFlag string, referral string, declaredCountry string, registeredKey string) {
	if client == nil || capState == nil {
		return
	}

	const refreshInterval = 30 * time.Second
	initialDelay := time.NewTimer(10 * time.Second)
	defer initialDelay.Stop()
	ticker := time.NewTicker(refreshInterval)
	defer ticker.Stop()

	tryRefresh := func(reason string) {
		currentCaps, currentDeviceType := capState.Snapshot()
		if !capSetGPUDetected(currentCaps) {
			refreshedCaps, refreshedDeviceType, refreshed := refreshCapabilityStateFromDetection(deviceFlag, capState, hw.DetectCaps)
			if refreshed {
				currentCaps = refreshedCaps
				currentDeviceType = refreshedDeviceType
				if operatorRuntimeState != nil {
					operatorRuntimeState.setCapabilities(refreshedCaps, refreshedDeviceType)
				}
				slog.Info("GPU capabilities refreshed after startup",
					"reason", reason,
					"device_type", refreshedDeviceType,
					"gpu_model", strings.TrimSpace(refreshedCaps.GPUModel),
					"vram_bytes", refreshedCaps.VRAMBytes)
			}
		}

		desiredKey := capabilityRegistrationKey(currentCaps, currentDeviceType)
		if desiredKey == registeredKey {
			return
		}
		if err := client.Register(ctx, hubCapabilitiesFromCaps(currentCaps), currentDeviceType, referral, declaredCountry); err != nil {
			if operatorRuntimeState != nil {
				operatorRuntimeState.setRegistered(false, err)
			}
			slog.Warn("capability refresh register failed", "error", err, "retry_in", refreshInterval)
			return
		}
		registeredKey = desiredKey
		if operatorRuntimeState != nil {
			operatorRuntimeState.setRegistered(true, nil)
		}
		requestV7CapabilityHeartbeat("hardware_capabilities_refreshed")
		slog.Info("capability refresh register succeeded",
			"device_type", currentDeviceType,
			"gpu_model", strings.TrimSpace(currentCaps.GPUModel),
			"vram_bytes", currentCaps.VRAMBytes)
	}

	for {
		select {
		case <-ctx.Done():
			return
		case <-initialDelay.C:
			tryRefresh("startup_delay")
		case <-ticker.C:
			tryRefresh("periodic")
		}
	}
}

// heartbeatLoop sends heartbeats independently of the work loop. It also accepts
// explicit wakeups when hub connectivity or local capability inventory changes.
func heartbeatLoop(ctx context.Context, client *hub.Client, capState *capabilityState, declaredCountry string, infMgr *inference.Manager, runtimeMgr *runtimeManager, hubURL string, currentVersion string) {
	const (
		normalInterval         = 30 * time.Second
		backoffInterval        = 60 * time.Second
		inventoryCheckInterval = 10 * time.Second
		circuitBreakerMax      = 30
	)

	ticker := time.NewTicker(normalInterval)
	defer ticker.Stop()
	inventoryTicker := time.NewTicker(inventoryCheckInterval)
	defer inventoryTicker.Stop()

	var consecutiveFailures int
	var lastSubmittedSummary operatorHeartbeatPayloadSummary
	var haveSubmittedSummary bool

	sendNow := func(reason string) {
		caps, deviceType := capState.Snapshot()
		result := sendHeartbeatDetailed(ctx, client, caps, deviceType, declaredCountry, infMgr, runtimeMgr)
		if result.hubOK {
			if consecutiveFailures >= circuitBreakerMax {
				slog.Info("hub heartbeat recovered after circuit breaker", "prev_failures", consecutiveFailures)
				ticker.Reset(normalInterval)
			} else if consecutiveFailures > 0 {
				slog.Info("hub heartbeat recovered", "prev_failures", consecutiveFailures)
			}
			consecutiveFailures = 0
			if result.capabilitySubmitted {
				lastSubmittedSummary = result.payloadSummary
				haveSubmittedSummary = true
			}
			if result.latestVersion != "" {
				updateReason := "heartbeat"
				if strings.TrimSpace(reason) != "" {
					updateReason = "heartbeat_" + strings.TrimSpace(reason)
				}
				maybeApplyAutoUpdate(ctx, hubURL, currentVersion, result.latestVersion, updateReason)
			}
			return
		}
		consecutiveFailures++
		if consecutiveFailures == circuitBreakerMax {
			slog.Warn("hub heartbeat circuit breaker tripped — backing off to 60s interval",
				"consecutive_failures", consecutiveFailures)
			ticker.Reset(backoffInterval)
		}
		if reason != "" {
			slog.Debug("capability heartbeat attempt failed", "reason", reason, "consecutive_failures", consecutiveFailures)
		}
	}

	// Send first heartbeat immediately.
	sendNow("startup")

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			sendNow("periodic")
		case reason := <-v7CapabilityHeartbeatRequests:
			sendNow(reason)
		case <-inventoryTicker.C:
			if consecutiveFailures >= circuitBreakerMax {
				continue
			}
			caps, deviceType := capState.Snapshot()
			summary, ok := currentV7HeartbeatPayloadSummary(ctx, client, caps, deviceType, declaredCountry, infMgr, runtimeMgr)
			if ok && (!haveSubmittedSummary || summary != lastSubmittedSummary) {
				sendNow("inventory_changed")
			}
		}
	}
}

func sendHeartbeat(ctx context.Context, client *hub.Client, caps hw.CapSet, deviceType string, declaredCountry string, infMgr *inference.Manager, runtimeMgr *runtimeManager) bool {
	return sendHeartbeatDetailed(ctx, client, caps, deviceType, declaredCountry, infMgr, runtimeMgr).hubOK
}

func sendHeartbeatDetailed(ctx context.Context, client *hub.Client, caps hw.CapSet, deviceType string, declaredCountry string, infMgr *inference.Manager, runtimeMgr *runtimeManager) heartbeatSendResult {
	started := time.Now()
	recordHubRTTProbe(ctx, client)
	networkProfile := currentHubNetworkProfile()
	metrics := hw.SampleMetrics()

	// Cache GPU utilization for the work loop's throttle check.
	cachedGPUUtil.Store(math.Float64bits(metrics.GPUUtil))

	// Report whether the node is self-throttling due to operator GPU usage.
	throttled := flagMaxGPUUtil > 0 && flagMaxGPUUtil < 100 && metrics.GPUUtil > flagMaxGPUUtil

	var v7Payload *heartbeat.V7HeartbeatPayload
	var payloadSummary operatorHeartbeatPayloadSummary
	if heartbeat.V7HeartbeatEnabledFromEnv() {
		payload, err := buildV7HeartbeatPayloadForNodeWithBackendRuntimes(client.PublicKeyHex(), caps, deviceType, declaredCountry, infMgr, runtimeMgr, buildCurrentBackendRuntimes(ctx))
		if err != nil {
			slog.Warn("failed to build V7 heartbeat payload; sending legacy heartbeat", "error", err)
			if operatorRuntimeState != nil {
				operatorRuntimeState.recordCapabilityHeartbeat(started, time.Now(), payloadSummary, operatorHeartbeatResponseSummary{}, err)
			}
		} else {
			v7Payload = payload
			payloadSummary = summarizeV7HeartbeatPayload(payload)
		}
	}

	heartbeatMetrics := hub.Metrics{
		TimestampMs:    time.Now().UnixMilli(),
		CPUUtil:        metrics.CPUUtil,
		MemUtil:        metrics.MemUtil,
		GPUUtil:        metrics.GPUUtil,
		PowerWatts:     metrics.PowerWatts,
		GPUThrottled:   throttled,
		NetworkProfile: networkProfile,
		V7Heartbeat:    v7Payload,
	}

	heartbeat, err := client.Heartbeat(ctx, heartbeatMetrics)
	if err != nil && heartbeatMetrics.V7Heartbeat != nil {
		if operatorRuntimeState != nil {
			operatorRuntimeState.recordCapabilityHeartbeat(started, time.Now(), payloadSummary, operatorHeartbeatResponseSummary{}, err)
		}
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
		return heartbeatSendResult{hubOK: false, payloadSummary: payloadSummary}
	}
	if operatorRuntimeState != nil {
		operatorRuntimeState.recordHeartbeat(metrics, heartbeat, nil)
		if v7Payload != nil && heartbeatMetrics.V7Heartbeat != nil {
			responseSummary := summarizeHeartbeatResponse(heartbeat)
			snapshotErr := heartbeatV7SnapshotConfirmationError(responseSummary)
			if snapshotErr != nil {
				responseSummary.Warning = snapshotErr.Error()
			}
			operatorRuntimeState.recordCapabilityHeartbeat(started, time.Now(), payloadSummary, responseSummary, snapshotErr)
		}
	}
	// Store latest version for work loop update checks.
	if heartbeat.LatestVersion != "" {
		latestHubVersion.Store(heartbeat.LatestVersion)
	}
	return heartbeatSendResult{
		hubOK:               true,
		capabilitySubmitted: v7Payload != nil && heartbeatMetrics.V7Heartbeat != nil && heartbeat.V7SnapshotUpserted != nil && *heartbeat.V7SnapshotUpserted,
		latestVersion:       strings.TrimSpace(heartbeat.LatestVersion),
		payloadSummary:      payloadSummary,
	}
}

func recordHubRTTProbe(ctx context.Context, client *hub.Client) {
	if client == nil || hubNetworkProfile == nil {
		return
	}
	hubNetworkProfile.SetTarget(client.HeartbeatProbeTarget())
	rtt, err := client.ProbeHubRTT(ctx)
	if err != nil {
		slog.Debug("hub RTT probe failed", "error", err)
		return
	}
	hubNetworkProfile.RecordRTT(rtt, time.Now())
}

func requestV7CapabilityHeartbeat(reason string) {
	reason = strings.TrimSpace(reason)
	if reason == "" {
		reason = "requested"
	}
	select {
	case v7CapabilityHeartbeatRequests <- reason:
	default:
	}
}

func currentV7HeartbeatPayloadSummary(ctx context.Context, client *hub.Client, caps hw.CapSet, deviceType string, declaredCountry string, infMgr *inference.Manager, runtimeMgr *runtimeManager) (operatorHeartbeatPayloadSummary, bool) {
	if !heartbeat.V7HeartbeatEnabledFromEnv() || client == nil {
		return operatorHeartbeatPayloadSummary{}, false
	}
	payload, err := buildV7HeartbeatPayloadForNodeWithBackendRuntimes(client.PublicKeyHex(), caps, deviceType, declaredCountry, infMgr, runtimeMgr, buildCurrentBackendRuntimes(ctx))
	if err != nil {
		slog.Debug("failed to build V7 heartbeat inventory summary", "error", err)
		return operatorHeartbeatPayloadSummary{}, false
	}
	return summarizeV7HeartbeatPayload(payload), true
}

func summarizeV7HeartbeatPayload(payload *heartbeat.V7HeartbeatPayload) operatorHeartbeatPayloadSummary {
	if payload == nil {
		return operatorHeartbeatPayloadSummary{}
	}
	models := make(map[string]struct{})
	addModel := func(values ...string) {
		for _, value := range values {
			value = strings.TrimSpace(value)
			if value == "" {
				continue
			}
			models[value] = struct{}{}
			return
		}
	}
	for _, model := range payload.ModelCache.Models {
		addModel(model.ModelID, model.Filename)
	}
	for _, model := range payload.CapabilityProfile.Models {
		addModel(model.ModelID)
	}

	backends := make(map[string]struct{})
	addBackend := func(name string, available bool) {
		name = strings.TrimSpace(name)
		if name == "" || !available {
			return
		}
		backends[name] = struct{}{}
	}
	addBackend(payload.BackendRuntimes.LlamaCPP.Backend, heartbeatRuntimeAvailable(payload.BackendRuntimes.LlamaCPP))
	addBackend(payload.BackendRuntimes.TensorRTLLM.Backend, heartbeatRuntimeAvailable(payload.BackendRuntimes.TensorRTLLM))
	addBackend(payload.BackendRuntimes.VLLM.Backend, heartbeatRuntimeAvailable(payload.BackendRuntimes.VLLM))
	addBackend(payload.BackendRuntimes.SGLang.Backend, heartbeatRuntimeAvailable(payload.BackendRuntimes.SGLang))
	for _, runtime := range payload.BackendRuntimes.Other {
		addBackend(runtime.Backend, heartbeatRuntimeAvailable(runtime))
	}
	for _, candidate := range payload.RuntimeInventory.BackendCandidates {
		addBackend(candidate.Backend, candidate.Detected)
	}
	if payload.RuntimeInventory.CandidateBackends.LlamaCPPDetected || payload.BackendProbes.LlamaCPP.Available {
		addBackend(llamacpp.BackendName, true)
	}
	if payload.RuntimeInventory.CandidateBackends.TensorRTLLMDetected {
		addBackend("tensorrt_llm", true)
	}
	if payload.RuntimeInventory.CandidateBackends.VLLMDetected {
		addBackend("vllm", true)
	}
	if payload.RuntimeInventory.CandidateBackends.SGLangDetected {
		addBackend("sglang", true)
	}
	if payload.RuntimeInventory.CandidateBackends.OllamaDetected {
		addBackend("ollama", true)
	}
	if payload.RuntimeInventory.CandidateBackends.PythonTransformersDetected {
		addBackend("python_transformers", true)
	}

	return operatorHeartbeatPayloadSummary{
		ModelCount:           len(models),
		BackendCount:         len(backends),
		HasCapabilityProfile: payload.CapabilityProfile.SchemaVersion != "",
		TextOutput:           payload.CapabilityProfile.TextOutput,
		Streaming:            payload.CapabilityProfile.Streaming,
		WarmModelID:          firstNonEmptyString(payload.BackendRuntimes.LlamaCPP.WarmModelID, payload.CapabilityProfile.WarmModel.ModelID),
	}
}

func summarizeHeartbeatResponse(heartbeat hub.HeartbeatResponse) operatorHeartbeatResponseSummary {
	return operatorHeartbeatResponseSummary{
		V7SnapshotUpserted:   heartbeat.V7SnapshotUpserted,
		SnapshotModelCount:   heartbeat.SnapshotModelCount,
		SnapshotBackendCount: heartbeat.SnapshotBackendCount,
		HasCapabilityProfile: heartbeat.HasCapabilityProfile,
		HubInstanceID:        strings.TrimSpace(heartbeat.HubInstanceID),
	}
}

func heartbeatV7SnapshotConfirmationError(summary operatorHeartbeatResponseSummary) error {
	if summary.V7SnapshotUpserted == nil {
		return fmt.Errorf("v7_snapshot_upserted_missing")
	}
	if !*summary.V7SnapshotUpserted {
		return fmt.Errorf("v7_snapshot_upserted_false")
	}
	return nil
}

func heartbeatRuntimeAvailable(runtime llamacpp.BackendRuntimeStatus) bool {
	return runtime.Available ||
		runtime.Running ||
		runtime.Healthy ||
		runtime.Loaded ||
		runtime.Warm
}

func buildOptionalV7HeartbeatPayload(nodePublicKey string, caps hw.CapSet, deviceType string, declaredCountry string, infMgr *inference.Manager, runtimeMgr *runtimeManager) *heartbeat.V7HeartbeatPayload {
	return buildOptionalV7HeartbeatPayloadWithBackendRuntimes(nodePublicKey, caps, deviceType, declaredCountry, infMgr, runtimeMgr, llamacpp.NormalizeBackendRuntimes(llamacpp.BackendRuntimes{}))
}

func buildOptionalV7HeartbeatPayloadWithBackendRuntimes(nodePublicKey string, caps hw.CapSet, deviceType string, declaredCountry string, infMgr *inference.Manager, runtimeMgr *runtimeManager, backendRuntimes llamacpp.BackendRuntimes) *heartbeat.V7HeartbeatPayload {
	if !heartbeat.V7HeartbeatEnabledFromEnv() {
		return nil
	}
	payload, err := buildV7HeartbeatPayloadForNodeWithBackendRuntimes(nodePublicKey, caps, deviceType, declaredCountry, infMgr, runtimeMgr, backendRuntimes)
	if err != nil {
		slog.Warn("failed to build V7 heartbeat payload; sending legacy heartbeat", "error", err)
		return nil
	}
	return payload
}

func buildV7HeartbeatPayloadForNode(nodePublicKey string, caps hw.CapSet, deviceType string, declaredCountry string, infMgr *inference.Manager, runtimeMgr *runtimeManager) (*heartbeat.V7HeartbeatPayload, error) {
	return buildV7HeartbeatPayloadForNodeWithBackendRuntimes(nodePublicKey, caps, deviceType, declaredCountry, infMgr, runtimeMgr, llamacpp.NormalizeBackendRuntimes(llamacpp.BackendRuntimes{}))
}

func buildV7HeartbeatPayloadForNodeWithBackendRuntimes(nodePublicKey string, caps hw.CapSet, deviceType string, declaredCountry string, infMgr *inference.Manager, runtimeMgr *runtimeManager, backendRuntimes llamacpp.BackendRuntimes) (*heartbeat.V7HeartbeatPayload, error) {
	gpuDetected := strings.TrimSpace(caps.GPUModel) != "" || caps.VRAMBytes > 0
	nativeSupported := inference.NativeRuntimeAvailable()
	nativeReady := nativeSupported && infMgr != nil && infMgr.Healthy()

	runtimeProfile := capability.RuntimeProfile{
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
	kvCapability := buildNativeTensorAccessCapability(infMgr)
	tensorAccess := buildRuntimeTensorAccessStatus(infMgr)
	runtimeInfo := operatorRuntimeInfo{
		NativeInferenceReady: nativeReady,
	}
	if infMgr != nil {
		runtimeInfo.NativeModel = infMgr.ModelName()
	}
	runtimeInventory := buildRuntimeInventoryStatus(runtimeInfo, tensorAccess, infMgr)
	baseModelPolicy := buildModelPolicyStatus()
	hardwareCapacity := buildHardwareCapacityStatus(baseModelPolicy.CacheDir)
	modelPolicy := buildDerivedModelPolicyStatus(baseModelPolicy, hardwareCapacity)
	backendProbes := buildBackendProbeStatus()
	backendRuntimes = llamacpp.EnrichBackendRuntimes(backendRuntimes, runtimeInventory, hardwareCapacity)
	modelCache := buildModelCacheRuntimeStatus(buildModelCacheStatus(modelPolicy), modelPolicy, hardwareCapacity, backendProbes, backendRuntimes)
	speculativeReport := buildSpeculativeReportStatus(hardwareCapacity, modelPolicy, modelCache, backendProbes, backendRuntimes, runtimeInventory)
	capabilityProfile := buildCapabilityProfileStatus(hardwareCapacity, modelPolicy, modelCache, backendProbes, backendRuntimes, runtimeInventory, &speculativeReport.SpeculativeDecoding, &kvCapability, tensorAccess)

	evidenceSummary := capability.EvidenceCapabilitySummary{
		SupportsArtifactManifest:    true,
		SupportsRYV3EvidencePayload: true,
		SupportsRuntimeHash:         v7RuntimeManifestHash(runtimeMgr) != "",
	}
	sandboxSummary := capability.SandboxCapabilitySummary{
		RejectsUnsafePickle:        true,
		RunnerAllowlistEnabled:     true,
		FilesystemIsolationPlanned: true,
		NetworkIsolationSupported:  false,
	}
	sandboxPolicy := sandbox.DefaultSandboxPolicy()

	payload, err := heartbeat.BuildV7HeartbeatPayload(heartbeat.BuildV7HeartbeatPayloadInput{
		AgentVersion:         version,
		NodePublicKey:        nodePublicKey,
		OS:                   runtime.GOOS,
		Arch:                 runtime.GOARCH,
		DeviceType:           deviceType,
		DeclaredCountry:      declaredCountry,
		HardwareCapabilities: caps,
		NetworkProfile:       currentHubNetworkProfile(),
		RuntimeProfile:       runtimeProfile,
		ModelCapabilitySummary: capability.ModelCapabilitySummary{
			ResidentModelIDs:      residentModelIDs,
			MaxResidentModelBytes: caps.VRAMBytes,
			SupportsModelLease:    gpuDetected && caps.VRAMBytes > 0 && nativeSupported,
		},
		KVCapability:              &kvCapability,
		TensorAccess:              &tensorAccess,
		RuntimeInventory:          &runtimeInventory,
		HardwareCapacity:          &hardwareCapacity,
		ModelPolicy:               &modelPolicy,
		ModelCache:                &modelCache,
		BackendProbes:             &backendProbes,
		BackendRuntimes:           &backendRuntimes,
		CapabilityProfile:         &capabilityProfile,
		SpeculativeProfiles:       speculativeReport.SpeculativeProfiles,
		SandboxCapabilitySummary:  sandboxSummary,
		SandboxPolicy:             &sandboxPolicy,
		EvidenceCapabilitySummary: evidenceSummary,
	})
	if err != nil {
		return nil, err
	}
	return &payload, nil
}

func currentHubNetworkProfile() *netprofile.NetworkProfile {
	if hubNetworkProfile == nil {
		return nil
	}
	profile, ok := hubNetworkProfile.Snapshot()
	if !ok || profile.SampleCount == 0 {
		return nil
	}
	return &profile
}

func buildCurrentBackendRuntimes(ctx context.Context) llamacpp.BackendRuntimes {
	if operatorRuntimeState != nil {
		return operatorRuntimeState.backendRuntimesStatus(ctx)
	}
	return buildStandaloneBackendRuntimes(ctx)
}

func buildStandaloneBackendRuntimes(ctx context.Context) llamacpp.BackendRuntimes {
	_ = ctx
	return llamacpp.NormalizeBackendRuntimes(llamacpp.BackendRuntimes{})
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
		if commandExists("git") && workCapsuleEnabled() {
			kinds = append(kinds, executorKindWorkCapsule)
		}
	}
	if ryvionRuntimeAvailable {
		kinds = append(kinds, executorKindRyvionRuntime)
	}
	return kinds
}

func healthReportLoop(ctx context.Context, client *hub.Client, capState *capabilityState, infMgr *inference.Manager, runtimeMgr *runtimeManager) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	send := func() {
		caps, _ := capState.Snapshot()
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
func workLoop(ctx context.Context, client *hub.Client, gpus, hubURL, currentVersion string, infMgr *inference.Manager, runtimeMgr *runtimeManager, capState *capabilityState) {
	backoff := 5 * time.Second
	maxBackoff := 2 * time.Minute
	hadPollFailure := false

	for {
		// Check for version updates (read from atomic, never missed).
		if v, ok := latestHubVersion.Load().(string); ok && v != "" {
			maybeApplyAutoUpdate(ctx, hubURL, currentVersion, v, "work_loop")
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
			hadPollFailure = true
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
		if hadPollFailure {
			requestV7CapabilityHeartbeat("work_poll_recovered")
			hadPollFailure = false
		}

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
		caps, _ := capState.Snapshot()
		gpuDetected := capSetGPUDetected(caps)
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
	stopAbortMonitor := startWorkGraphAbortMonitor(runCtx, client, work, cancel)
	defer stopAbortMonitor()

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

	if handled, result, runErr := processOptionalV7TensorPlaneBenchmark(runCtx, client, work, runtimeMgr, gpuDetected); handled {
		if runErr != nil {
			slog.Warn("V7 TensorPlane benchmark execution failed", "job_id", work.JobID, "error", runErr)
		} else if result != nil {
			slog.Info("V7 TensorPlane benchmark completed", "job_id", work.JobID, "hash", result.ResultHashHex, "units", result.MeteringUnits)
		}
		if operatorRuntimeState != nil {
			operatorRuntimeState.finishJob(work, result, runErr)
		}
		return
	}

	if handled, result, runErr := processOptionalV7ModelWarm(runCtx, client, work, runtimeMgr, gpuDetected); handled {
		if runErr != nil {
			slog.Warn("V7 model warm execution failed", "job_id", work.JobID, "error", runErr)
		} else if result != nil {
			slog.Info("V7 model warm completed", "job_id", work.JobID, "hash", result.ResultHashHex, "units", result.MeteringUnits)
		}
		if operatorRuntimeState != nil {
			operatorRuntimeState.finishJob(work, result, runErr)
		}
		return
	}

	if handled, result, runErr := processOptionalV7LlamaCppBackendBenchmark(runCtx, client, work, runtimeMgr, gpuDetected); handled {
		if runErr != nil {
			slog.Warn("V7 llama.cpp backend benchmark execution failed", "job_id", work.JobID, "error", runErr)
		} else if result != nil {
			slog.Info("V7 llama.cpp backend benchmark completed", "job_id", work.JobID, "hash", result.ResultHashHex, "units", result.MeteringUnits)
		}
		if operatorRuntimeState != nil {
			operatorRuntimeState.finishJob(work, result, runErr)
		}
		return
	}

	if handled, result, runErr := processOptionalSpeculativeNativeDraftHotSession(runCtx, client, work, runtimeMgr, gpuDetected); handled {
		if runErr != nil {
			slog.Warn("Speculative native hot draft session failed", "job_id", work.JobID, "error", runErr)
		} else if result != nil {
			slog.Info("Speculative native hot draft session completed", "job_id", work.JobID, "hash", result.ResultHashHex, "units", result.MeteringUnits)
		}
		if operatorRuntimeState != nil {
			operatorRuntimeState.finishJob(work, result, runErr)
		}
		return
	}

	if handled, result, runErr := processOptionalSpeculativeNativeVerifierHotSession(runCtx, client, work, runtimeMgr, gpuDetected); handled {
		if runErr != nil {
			slog.Warn("Speculative native hot verifier session failed", "job_id", work.JobID, "error", runErr)
		} else if result != nil {
			slog.Info("Speculative native hot verifier session completed", "job_id", work.JobID, "hash", result.ResultHashHex, "units", result.MeteringUnits)
		}
		if operatorRuntimeState != nil {
			operatorRuntimeState.finishJob(work, result, runErr)
		}
		return
	}

	if handled, result, runErr := processOptionalSpeculativeNativeDraft(runCtx, client, work, runtimeMgr, gpuDetected); handled {
		if runErr != nil {
			slog.Warn("Speculative native draft execution failed", "job_id", work.JobID, "error", runErr)
		} else if result != nil {
			slog.Info("Speculative native draft completed", "job_id", work.JobID, "hash", result.ResultHashHex, "units", result.MeteringUnits)
		}
		if operatorRuntimeState != nil {
			operatorRuntimeState.finishJob(work, result, runErr)
		}
		return
	}

	if handled, result, runErr := processOptionalSpeculativeNativeVerifier(runCtx, client, work, runtimeMgr, gpuDetected); handled {
		if runErr != nil {
			slog.Warn("Speculative native verifier execution failed", "job_id", work.JobID, "error", runErr)
		} else if result != nil {
			slog.Info("Speculative native verifier completed", "job_id", work.JobID, "hash", result.ResultHashHex, "units", result.MeteringUnits)
		}
		if operatorRuntimeState != nil {
			operatorRuntimeState.finishJob(work, result, runErr)
		}
		return
	}

	if handled, result, runErr := processOptionalV7DashboardInference(runCtx, client, work, runtimeMgr, gpuDetected); handled {
		if runErr != nil {
			slog.Warn("V7 dashboard inference execution failed", "job_id", work.JobID, "error", runErr)
		} else if result != nil {
			slog.Info("V7 dashboard inference completed", "job_id", work.JobID, "hash", result.ResultHashHex, "units", result.MeteringUnits)
		}
		if operatorRuntimeState != nil {
			operatorRuntimeState.finishJob(work, result, runErr)
		}
		return
	}

	if handled, result, runErr := processOptionalV7BackendInferenceBenchmark(runCtx, client, work, runtimeMgr, gpuDetected); handled {
		if runErr != nil {
			slog.Warn("V7 backend inference benchmark execution failed", "job_id", work.JobID, "error", runErr)
		} else if result != nil {
			slog.Info("V7 backend inference benchmark completed", "job_id", work.JobID, "hash", result.ResultHashHex, "units", result.MeteringUnits)
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

	if handled, result, runErr := processOptionalV7ModelPrepare(runCtx, client, work, runtimeMgr, gpuDetected); handled {
		if runErr != nil {
			slog.Warn("V7 model prepare execution failed", "job_id", work.JobID, "error", runErr)
		} else if result != nil {
			slog.Info("V7 model prepare completed", "job_id", work.JobID, "hash", result.ResultHashHex, "units", result.MeteringUnits)
		}
		if operatorRuntimeState != nil {
			operatorRuntimeState.finishJob(work, result, runErr)
		}
		return
	}

	workLoopDiagnostics.RecordEvent("generic_path_entered", work.JobID, work.Kind, mainWorkLoopSpecContext(diagnostics.WorkSpecTaskFromJSON(work.SpecJSON)))

	// Low free VRAM is expected when llama-server already keeps a model
	// resident. Let the native streaming manager execute or hot-swap the model
	// instead of rejecting valid warm jobs before llama.cpp can run them.
	if isStreaming {
		freeVRAM := hw.GetFreeVRAM()
		if freeVRAM > 0 && freeVRAM < 2*1024*1024*1024 { // Less than 2GB free
			slog.Info("low free VRAM before native streaming; proceeding with llama.cpp residency manager", "free_vram_mb", freeVRAM/(1024*1024), "job_id", work.JobID)
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
	identity, isBenchmark := memorybench.BenchmarkAssignmentIdentityFromJSON(work.SpecJSON)
	statusJobID := firstNonEmptyString(work.JobID, identity.JobID)
	if isBenchmark && operatorRuntimeState != nil {
		operatorRuntimeState.recordV7MemoryBenchmarkSeen(statusJobID, identity.RequestID)
	}

	benchmarkEnabled := isBenchmark && memorybench.BenchmarkEnabledFromEnv(os.Getenv)
	executionStarted := time.Now()
	if benchmarkEnabled {
		workLoopDiagnostics.RecordExecutionStart(statusJobID)
		workLoopDiagnostics.RecordEvent("v7_fast_path_start", statusJobID, work.Kind, v7MemoryBenchmarkWorkLoopEventContextFromSpec(work.SpecJSON))
	}
	restoreReceiptRecorder := func() {}
	if benchmarkEnabled {
		restoreReceiptRecorder = memorybench.SetReceiptSubstepEventRecorder(workLoopDiagnostics)
	}
	receipt, receiptBuildTimings, handled, err := memorybench.ExecuteBenchmarkAssignmentWithReceiptTimings(ctx, work.SpecJSON, memorybench.ExecuteOptions{
		Getenv: os.Getenv,
	})
	restoreReceiptRecorder()
	if !handled {
		return false, nil, nil
	}
	receiptTimings := workLoopReceiptTimingsFromMemorybench(receiptBuildTimings)
	workLoopDiagnostics.RecordReceiptBuildTimings(receiptTimings)
	workLoopDiagnostics.RecordExecutionEnd(time.Since(executionStarted), err)

	workLoopDiagnostics.RecordEvent("pre_submit_block_start", firstNonEmptyString(receipt.JobID, statusJobID), work.Kind, v7MemoryBenchmarkWorkLoopEventContextFromReceipt(work.SpecJSON, receipt, receiptTimings))
	runtimeMeta := v7BenchmarkFastPathRuntimeMetadata(runtimeMgr, gpuDetected)
	extra := map[string]any{
		"executor":      memorybench.BenchmarkTask,
		"executor_kind": memorybench.BenchmarkTask,
		"task":          memorybench.BenchmarkTask,
	}
	if err != nil {
		receipt = memorybench.BuildBenchmarkRejectionReceipt(work.JobID, err)
		extra["exit_code"] = 1
		extra["error"] = "v7 memory benchmark rejected"
	} else {
		extra["exit_code"] = 0
	}
	exitCode, _ := extra["exit_code"].(int)
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
	receiptContext := v7MemoryBenchmarkWorkLoopEventContextFromReceipt(work.SpecJSON, receipt, receiptTimings)
	workLoopDiagnostics.RecordEvent("pre_submit_block_end", hubReceipt.JobID, work.Kind, receiptContext)
	receiptReadyAt := time.Now()
	workLoopDiagnostics.RecordEvent("v7_fast_path_receipt_ready", hubReceipt.JobID, work.Kind, receiptContext)
	workLoopDiagnostics.RecordReceiptReady(hubReceipt.JobID, work.Kind, receiptReadyAt, receiptContext)

	workLoopDiagnostics.RecordEvent("v7_fast_path_submit_start", hubReceipt.JobID, work.Kind, receiptContext)
	if submitErr := submitReceiptWithRetry(ctx, client, hubReceipt); submitErr != nil {
		workLoopDiagnostics.RecordEvent("v7_fast_path_submit_end", hubReceipt.JobID, work.Kind, receiptContext)
		if operatorRuntimeState != nil {
			if err != nil {
				operatorRuntimeState.recordV7MemoryBenchmarkRejected(statusJobID, err)
			} else {
				operatorRuntimeState.recordV7MemoryBenchmarkExecuted(firstNonEmptyString(receipt.JobID, statusJobID))
			}
			operatorRuntimeState.recordV7MemoryBenchmarkReceiptFailed(hubReceipt.JobID, submitErr)
		}
		if err != nil {
			return true, snapshot, fmt.Errorf("%v; receipt submit failed: %w", err, submitErr)
		}
		return true, snapshot, submitErr
	}
	workLoopDiagnostics.RecordEvent("v7_fast_path_submit_end", hubReceipt.JobID, work.Kind, receiptContext)
	if operatorRuntimeState != nil {
		if err != nil {
			operatorRuntimeState.recordV7MemoryBenchmarkRejected(statusJobID, err)
		} else {
			operatorRuntimeState.recordV7MemoryBenchmarkExecuted(firstNonEmptyString(receipt.JobID, statusJobID))
		}
		operatorRuntimeState.recordV7MemoryBenchmarkReceiptSubmitted(hubReceipt.JobID)
	}
	return true, snapshot, err
}

func v7BenchmarkFastPathRuntimeMetadata(runtimeMgr *runtimeManager, gpuDetected bool) map[string]any {
	_ = gpuDetected
	if runtimeMgr == nil {
		return map[string]any{}
	}
	// Internal benchmark latency measurement requires no artificial pre-submit delay.
	// Keep the receipt metadata key shape, but do not probe the managed runtime before SubmitReceipt.
	return nonProbingRuntimeReceiptMetadata(runtimeMgr)
}

func nonProbingRuntimeReceiptMetadata(runtimeMgr *runtimeManager) map[string]any {
	if runtimeMgr == nil {
		return map[string]any{}
	}
	version := sanitizeStatusValue(runtimeMgr.contract.Version)
	if version == "" {
		version = sanitizeStatusValue(runtimeMgr.version)
	}
	if version == "" {
		version = "dev"
	}
	manifestHash := sanitizeStatusValue(runtimeMgr.contract.ManifestHash)
	if manifestHash == "" {
		manifestHash = computeRuntimeManifestHash(runtimeMgr.contract)
	}
	mode := sanitizeStatusValue(runtimeMgr.contract.Mode)
	source := sanitizeStatusValue(runtimeMgr.contract.Source)
	health := ""
	if ociLaneDisabled() {
		health = "disabled"
		mode = "native_only"
		source = "operator_opt_out"
	}
	engine := sanitizeStatusValue(runtimeMgr.contract.Engine)
	engineKind := sanitizeStatusValue(firstNonEmpty(runtimeMgr.contract.EngineKind, runtimeexec.EngineKind(engine)))
	return map[string]any{
		"runtime_version":       version,
		"runtime_manifest_hash": manifestHash,
		"runtime_health":        health,
		"runtime_warming":       false,
		"runtime_channel":       sanitizeStatusValue(runtimeMgr.contract.Channel),
		"runtime_provider":      sanitizeStatusValue(runtimeMgr.contract.Provider),
		"runtime_mode":          mode,
		"runtime_source":        source,
		"runtime_artifact":      sanitizeStatusValue(runtimeMgr.contract.Artifact),
		"runtime_binary":        sanitizeStatusValue(runtimeMgr.contract.Binary),
		"runtime_backend":       sanitizeStatusValue(runtimeMgr.contract.Backend),
		"runtime_engine":        engine,
		"runtime_engine_kind":   engineKind,
	}
}

func workLoopReceiptTimingsFromMemorybench(receiptTimings memorybench.ReceiptBuildTimings) diagnostics.ReceiptBuildTimings {
	return diagnostics.ReceiptBuildTimings{
		MetadataBuildUs:     receiptTimings.MetadataBuildUs,
		MetadataStructUs:    receiptTimings.MetadataStructUs,
		WeightedValueCopyUs: receiptTimings.WeightedValueCopyUs,
		MetadataDefaultsUs:  receiptTimings.MetadataDefaultsUs,
		MetadataValidateUs:  receiptTimings.MetadataValidateUs,
		MetadataGapUs:       receiptTimings.MetadataGapUs,
		MetadataTotalUs:     receiptTimings.MetadataTotalUs,
		HashUs:              receiptTimings.HashUs,
		JSONMeasureUs:       receiptTimings.JSONMeasureUs,
		EnvelopeBuildUs:     receiptTimings.EnvelopeBuildUs,
		TotalBuildUs:        receiptTimings.TotalBuildUs,
	}
}

func v7MemoryBenchmarkWorkLoopEventContextFromSpec(specJSON string) map[string]string {
	context := map[string]string{
		"spec_task": memorybench.BenchmarkTask,
	}
	var spec struct {
		Task       string `json:"task"`
		TokenCount int    `json:"token_count"`
		ValueDim   int    `json:"value_dim"`
	}
	if err := json.Unmarshal([]byte(specJSON), &spec); err != nil {
		return context
	}
	if strings.TrimSpace(spec.Task) == memorybench.BenchmarkTask {
		context["spec_task"] = memorybench.BenchmarkTask
	}
	if spec.TokenCount > 0 {
		context["token_count"] = strconv.Itoa(spec.TokenCount)
	}
	if spec.ValueDim > 0 {
		context["value_dim"] = strconv.Itoa(spec.ValueDim)
	}
	return context
}

func v7MemoryBenchmarkWorkLoopEventContextFromReceipt(specJSON string, receipt memorybench.BenchmarkReceipt, timings diagnostics.ReceiptBuildTimings) map[string]string {
	context := v7MemoryBenchmarkWorkLoopEventContextFromSpec(specJSON)
	putWorkLoopInt64Context(context, "metadata_total_us", timings.MetadataTotalUs)
	putWorkLoopInt64Context(context, "metadata_gap_us", timings.MetadataGapUs)
	if bodyBytes := v7MemoryBenchmarkReceiptBodyBytes(receipt.Metadata); bodyBytes > 0 {
		putWorkLoopInt64Context(context, "receipt_body_bytes", bodyBytes)
	}
	if taskMetadata, ok := receipt.Metadata[memorybench.BenchmarkTask].(map[string]any); ok {
		putWorkLoopAnyIntContext(context, "token_count", taskMetadata["token_count"])
		putWorkLoopAnyIntContext(context, "value_dim", taskMetadata["value_dim"])
		if weightedValue, ok := taskMetadata["weighted_value"].([]float64); ok {
			context["weighted_value_len"] = strconv.Itoa(len(weightedValue))
		}
	}
	return context
}

func v7MemoryBenchmarkReceiptBodyBytes(metadata map[string]any) int64 {
	taskMetadata, ok := metadata[memorybench.BenchmarkTask].(map[string]any)
	if !ok {
		return 0
	}
	return workLoopAnyInt64(taskMetadata["receipt_envelope_json_bytes"])
}

func putWorkLoopAnyIntContext(context map[string]string, key string, value any) {
	if n := workLoopAnyInt64(value); n > 0 {
		context[key] = strconv.FormatInt(n, 10)
	}
}

func putWorkLoopInt64Context(context map[string]string, key string, value int64) {
	if value > 0 {
		context[key] = strconv.FormatInt(value, 10)
	}
}

func workLoopAnyInt64(value any) int64 {
	switch v := value.(type) {
	case int:
		return int64(v)
	case int64:
		return v
	case int32:
		return int64(v)
	case uint64:
		const maxInt64Uint64 = uint64(1<<63 - 1)
		if v <= maxInt64Uint64 {
			return int64(v)
		}
	case uint32:
		return int64(v)
	case float64:
		const maxInt64Float64 = float64(1<<63 - 1)
		if v > 0 && v <= maxInt64Float64 {
			return int64(v)
		}
	}
	return 0
}

func processOptionalV7TensorPlaneBenchmark(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, runtimeMgr *runtimeManager, gpuDetected bool) (bool, *runnerResultSnapshot, error) {
	identity, isBenchmark := tensorbench.BenchmarkAssignmentIdentityFromJSON(work.SpecJSON)
	statusJobID := firstNonEmptyString(work.JobID, identity.JobID)
	benchmarkEnabled := isBenchmark && tensorbench.BenchmarkEnabledFromEnv(os.Getenv)
	if benchmarkEnabled && v7TensorPlaneBenchmarkStatus != nil {
		v7TensorPlaneBenchmarkStatus.RecordSeen(statusJobID)
	}

	executionStarted := time.Now()
	if benchmarkEnabled {
		workLoopDiagnostics.RecordExecutionStart(statusJobID)
		workLoopDiagnostics.RecordEvent("v7_fast_path_start", statusJobID, work.Kind, v7TensorPlaneBenchmarkWorkLoopEventContextFromSpec(work.SpecJSON))
	}
	receipt, handled, err := tensorbench.ExecuteBenchmarkAssignment(ctx, work.SpecJSON, tensorbench.ExecuteOptions{
		Getenv: os.Getenv,
	})
	if !handled {
		return false, nil, nil
	}
	workLoopDiagnostics.RecordExecutionEnd(time.Since(executionStarted), err)
	if err != nil && v7TensorPlaneBenchmarkStatus != nil {
		v7TensorPlaneBenchmarkStatus.RecordError(statusJobID, err)
	}

	receiptBuildStarted := time.Now()
	workLoopDiagnostics.RecordEvent("pre_submit_block_start", firstNonEmptyString(receipt.JobID, statusJobID), work.Kind, v7TensorPlaneBenchmarkWorkLoopEventContextFromSpec(work.SpecJSON))
	runtimeMeta := v7BenchmarkFastPathRuntimeMetadata(runtimeMgr, gpuDetected)
	extra := map[string]any{
		"executor":      tensorbench.BenchmarkTask,
		"executor_kind": tensorbench.BenchmarkTask,
		"task":          tensorbench.BenchmarkTask,
	}
	if strings.TrimSpace(receipt.ResultHashHex) == "" {
		receipt = tensorbench.BuildBenchmarkRejectionReceipt(work.JobID, err)
	}
	exitCode := 0
	if err != nil {
		exitCode = 1
		extra["exit_code"] = 1
		extra["error"] = "v7 tensorplane benchmark failed"
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
	receiptContext := v7TensorPlaneBenchmarkWorkLoopEventContextFromReceipt(work.SpecJSON, receipt)
	workLoopDiagnostics.RecordEvent("pre_submit_block_end", hubReceipt.JobID, work.Kind, receiptContext)
	workLoopDiagnostics.RecordEvent("v7_fast_path_receipt_ready", hubReceipt.JobID, work.Kind, receiptContext)
	workLoopDiagnostics.RecordReceiptReady(hubReceipt.JobID, work.Kind, time.Now(), receiptContext)
	workLoopDiagnostics.RecordEvent("v7_fast_path_submit_start", hubReceipt.JobID, work.Kind, receiptContext)
	if client == nil {
		submitErr := fmt.Errorf("hub client unavailable")
		workLoopDiagnostics.RecordReceiptSubmitStart(hubReceipt.JobID, 1)
		workLoopDiagnostics.RecordReceiptSubmitEnd(0, submitErr)
		workLoopDiagnostics.RecordEvent("v7_fast_path_submit_end", hubReceipt.JobID, work.Kind, receiptContext)
		if v7TensorPlaneBenchmarkStatus != nil {
			if err == nil {
				v7TensorPlaneBenchmarkStatus.RecordExecuted(hubReceipt.JobID)
			}
			v7TensorPlaneBenchmarkStatus.RecordReceiptFailed(hubReceipt.JobID, submitErr)
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
	workLoopDiagnostics.RecordEvent("v7_fast_path_submit_end", hubReceipt.JobID, work.Kind, receiptContext)
	if submitErr != nil {
		if v7TensorPlaneBenchmarkStatus != nil {
			if err == nil {
				v7TensorPlaneBenchmarkStatus.RecordExecuted(hubReceipt.JobID)
			}
			v7TensorPlaneBenchmarkStatus.RecordReceiptFailed(hubReceipt.JobID, submitErr)
		}
		if err != nil {
			return true, snapshot, fmt.Errorf("%v; receipt submit failed: %w", err, submitErr)
		}
		return true, snapshot, submitErr
	}
	if v7TensorPlaneBenchmarkStatus != nil {
		if err == nil {
			v7TensorPlaneBenchmarkStatus.RecordExecuted(hubReceipt.JobID)
		}
		v7TensorPlaneBenchmarkStatus.RecordReceiptSubmitted(hubReceipt.JobID)
	}
	return true, snapshot, err
}

func v7TensorPlaneBenchmarkWorkLoopEventContextFromSpec(specJSON string) map[string]string {
	context := map[string]string{
		"spec_task": tensorbench.BenchmarkTask,
	}
	var spec struct {
		Task     string `json:"task"`
		Tokens   int    `json:"tokens"`
		HeadDim  int    `json:"head_dim"`
		ValueDim int    `json:"value_dim"`
	}
	if err := json.Unmarshal([]byte(specJSON), &spec); err != nil {
		return context
	}
	if strings.TrimSpace(spec.Task) == tensorbench.BenchmarkTask {
		context["spec_task"] = tensorbench.BenchmarkTask
	}
	if spec.Tokens > 0 {
		context["tokens"] = strconv.Itoa(spec.Tokens)
	}
	if spec.HeadDim > 0 {
		context["head_dim"] = strconv.Itoa(spec.HeadDim)
	}
	if spec.ValueDim > 0 {
		context["value_dim"] = strconv.Itoa(spec.ValueDim)
	}
	return context
}

func v7TensorPlaneBenchmarkWorkLoopEventContextFromReceipt(specJSON string, receipt tensorbench.BenchmarkReceipt) map[string]string {
	context := v7TensorPlaneBenchmarkWorkLoopEventContextFromSpec(specJSON)
	if taskMetadata, ok := receipt.Metadata[tensorbench.BenchmarkTask].(map[string]any); ok {
		putWorkLoopAnyIntContext(context, "tokens", taskMetadata["tokens"])
		putWorkLoopAnyIntContext(context, "head_dim", taskMetadata["head_dim"])
		putWorkLoopAnyIntContext(context, "value_dim", taskMetadata["value_dim"])
		putWorkLoopAnyIntContext(context, "payload_bytes_estimate", taskMetadata["payload_bytes_estimate"])
		if weightedValue, ok := taskMetadata["weighted_value"].([]float64); ok {
			context["weighted_value_len"] = strconv.Itoa(len(weightedValue))
		}
		if status, ok := taskMetadata["correctness_status"].(string); ok && strings.TrimSpace(status) != "" {
			context["correctness_status"] = strings.TrimSpace(status)
		}
	}
	return context
}

func currentV7ModelWarmStatus() *modelwarm.LocalStatus {
	if operatorRuntimeState != nil {
		return operatorRuntimeState.modelWarmStatus()
	}
	return v7ModelWarmStatus
}

func processOptionalV7ModelWarm(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, runtimeMgr *runtimeManager, gpuDetected bool) (bool, *runnerResultSnapshot, error) {
	identity, isWarm := modelwarm.WarmAssignmentIdentityFromJSON(work.SpecJSON)
	if !isWarm {
		return false, nil, nil
	}
	status := currentV7ModelWarmStatus()
	if status != nil {
		status.RecordSeen(identity.WarmID, identity.ModelID)
	}
	statusJobID := firstNonEmptyString(work.JobID, identity.JobID)
	executionStarted := time.Now()
	if modelwarm.WarmEnabledFromEnv(os.Getenv) {
		workLoopDiagnostics.RecordExecutionStart(statusJobID)
		workLoopDiagnostics.RecordEvent("v7_fast_path_start", statusJobID, work.Kind, v7ModelWarmWorkLoopEventContextFromSpec(work.SpecJSON))
	}

	var manager modelwarm.LlamaCppManager
	if operatorRuntimeState != nil {
		manager = operatorRuntimeState.llamaCppManager()
	}
	warmOptions := modelwarm.ExecuteOptions{
		Getenv:          os.Getenv,
		Policy:          modelpolicy.FromEnv(),
		LlamaCppManager: manager,
	}
	if manager != nil {
		warmOptions.BenchmarkRunner = llamacpp.BenchmarkRunner{
			Sidecar: benchmarkSidecarFromWarmManager(manager),
			Client:  llamacpp.OpenAIClient{},
		}
	}
	receipt, handled, err := modelwarm.ExecuteWarmAssignment(ctx, work.SpecJSON, warmOptions)
	if !handled {
		return false, nil, nil
	}
	workLoopDiagnostics.RecordExecutionEnd(time.Since(executionStarted), err)
	if strings.TrimSpace(receipt.ResultHashHex) == "" {
		receipt = modelwarm.BuildWarmRejectionReceiptFromIdentity(identity, err)
	}

	receiptBuildStarted := time.Now()
	workLoopDiagnostics.RecordEvent("pre_submit_block_start", firstNonEmptyString(receipt.JobID, statusJobID), work.Kind, v7ModelWarmWorkLoopEventContextFromSpec(work.SpecJSON))
	extra := map[string]any{
		"executor":      modelwarm.WarmTask,
		"executor_kind": modelwarm.WarmTask,
		"task":          modelwarm.WarmTask,
	}
	warmed := v7ModelWarmReceiptWarmed(receipt)
	recordErr := err
	if !warmed && recordErr == nil {
		recordErr = v7ModelWarmRejectionError(receipt)
	}
	exitCode := 0
	if recordErr != nil || !warmed {
		exitCode = 1
		extra["exit_code"] = 1
		extra["error"] = "v7 model warm failed"
	} else {
		extra["exit_code"] = 0
	}
	metadata := receiptMetadataBase(work, v7BenchmarkFastPathRuntimeMetadata(runtimeMgr, gpuDetected), receipt.Metadata, extra)
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
	receiptContext := v7ModelWarmWorkLoopEventContextFromReceipt(work.SpecJSON, receipt)
	workLoopDiagnostics.RecordEvent("pre_submit_block_end", hubReceipt.JobID, work.Kind, receiptContext)
	workLoopDiagnostics.RecordEvent("v7_fast_path_receipt_ready", hubReceipt.JobID, work.Kind, receiptContext)
	workLoopDiagnostics.RecordReceiptReady(hubReceipt.JobID, work.Kind, time.Now(), receiptContext)
	workLoopDiagnostics.RecordEvent("v7_fast_path_submit_start", hubReceipt.JobID, work.Kind, receiptContext)
	if client == nil {
		submitErr := fmt.Errorf("hub client unavailable")
		workLoopDiagnostics.RecordReceiptSubmitStart(hubReceipt.JobID, 1)
		workLoopDiagnostics.RecordReceiptSubmitEnd(0, submitErr)
		workLoopDiagnostics.RecordEvent("v7_fast_path_submit_end", hubReceipt.JobID, work.Kind, receiptContext)
		if status != nil {
			if warmed {
				status.RecordExecuted(identity.WarmID, identity.ModelID)
			} else {
				status.RecordRejected(identity.WarmID, identity.ModelID, recordErr)
			}
			status.RecordReceiptFailed(identity.WarmID, identity.ModelID, submitErr)
		}
		if recordErr != nil {
			return true, snapshot, fmt.Errorf("%v; receipt submit failed: %w", recordErr, submitErr)
		}
		return true, snapshot, submitErr
	}
	submitStarted := time.Now()
	workLoopDiagnostics.RecordReceiptSubmitStart(hubReceipt.JobID, 1)
	submitErr := client.SubmitReceipt(ctx, hubReceipt)
	workLoopDiagnostics.RecordReceiptSubmitEnd(time.Since(submitStarted), submitErr)
	workLoopDiagnostics.RecordEvent("v7_fast_path_submit_end", hubReceipt.JobID, work.Kind, receiptContext)
	if submitErr != nil {
		if status != nil {
			if warmed {
				status.RecordExecuted(identity.WarmID, identity.ModelID)
			} else {
				status.RecordRejected(identity.WarmID, identity.ModelID, recordErr)
			}
			status.RecordReceiptFailed(identity.WarmID, identity.ModelID, submitErr)
		}
		if recordErr != nil {
			return true, snapshot, fmt.Errorf("%v; receipt submit failed: %w", recordErr, submitErr)
		}
		return true, snapshot, submitErr
	}
	if status != nil {
		if warmed {
			status.RecordExecuted(identity.WarmID, identity.ModelID)
		} else {
			status.RecordRejected(identity.WarmID, identity.ModelID, recordErr)
		}
		status.RecordReceiptSubmitted(identity.WarmID, identity.ModelID)
	}
	if warmed {
		requestV7CapabilityHeartbeat("v7_model_warm_changed")
	}
	return true, snapshot, recordErr
}

func benchmarkSidecarFromWarmManager(manager modelwarm.LlamaCppManager) llamacpp.BenchmarkSidecar {
	if sidecar, ok := manager.(llamacpp.BenchmarkSidecar); ok && sidecar != nil {
		return sidecar
	}
	return llamacpp.NewManagerFromEnv()
}

func v7ModelWarmReceiptWarmed(receipt modelwarm.WarmReceipt) bool {
	if taskMetadata, ok := receipt.Metadata[modelwarm.WarmTask].(map[string]any); ok {
		proofStatus, _ := taskMetadata["proof_status"].(string)
		warm, _ := taskMetadata["warm"].(bool)
		return warm && strings.TrimSpace(proofStatus) == modelwarm.ProofStatusModelWarmed
	}
	return false
}

func v7ModelWarmRejectionError(receipt modelwarm.WarmReceipt) error {
	code := "model_warm_failed"
	if taskMetadata, ok := receipt.Metadata[modelwarm.WarmTask].(map[string]any); ok {
		if value, ok := taskMetadata["error_code"].(string); ok && strings.TrimSpace(value) != "" {
			code = strings.TrimSpace(value)
		}
	}
	return fmt.Errorf("%s", code)
}

func v7ModelWarmWorkLoopEventContextFromSpec(specJSON string) map[string]string {
	context := map[string]string{
		"spec_task": modelwarm.WarmTask,
	}
	var spec struct {
		Task                  string `json:"task"`
		WarmID                string `json:"warm_id"`
		Backend               string `json:"backend"`
		ModelID               string `json:"model_id"`
		RunBenchmarkAfterWarm bool   `json:"run_benchmark_after_warm"`
	}
	if err := json.Unmarshal([]byte(specJSON), &spec); err != nil {
		return context
	}
	if strings.TrimSpace(spec.Task) == modelwarm.WarmTask {
		context["spec_task"] = modelwarm.WarmTask
	}
	if warmID := strings.TrimSpace(spec.WarmID); warmID != "" {
		context["warm_id"] = warmID
	}
	if backend := strings.TrimSpace(spec.Backend); backend != "" {
		context["backend"] = backend
	}
	if modelID := strings.TrimSpace(spec.ModelID); modelID != "" {
		context["model_id"] = modelID
	}
	context["run_benchmark_after_warm"] = strconv.FormatBool(spec.RunBenchmarkAfterWarm)
	return context
}

func v7ModelWarmWorkLoopEventContextFromReceipt(specJSON string, receipt modelwarm.WarmReceipt) map[string]string {
	context := v7ModelWarmWorkLoopEventContextFromSpec(specJSON)
	if taskMetadata, ok := receipt.Metadata[modelwarm.WarmTask].(map[string]any); ok {
		if warm, ok := taskMetadata["warm"].(bool); ok {
			context["warm"] = strconv.FormatBool(warm)
		}
		if status, ok := taskMetadata["proof_status"].(string); ok && strings.TrimSpace(status) != "" {
			context["proof_status"] = strings.TrimSpace(status)
		}
	}
	return context
}

func currentV7BackendBenchmarkStatus() *llamacpp.BackendBenchmarkLocalStatus {
	if operatorRuntimeState != nil {
		return operatorRuntimeState.backendBenchmarkStatus()
	}
	return v7BackendBenchmarkStatus
}

func currentV7LlamaCppBackendBenchmarkProfile(ctx context.Context, client *hub.Client) llamacpp.BackendBenchmarkProfile {
	cfg := llamacpp.ConfigFromEnv()
	if operatorRuntimeState != nil {
		cfg = operatorRuntimeState.llamaCppManager().Config()
	}
	basePolicy := buildModelPolicyStatus()
	hardware := buildHardwareCapacityStatus(basePolicy.CacheDir)
	runtimes := buildCurrentBackendRuntimes(ctx)
	nodeID := ""
	if client != nil {
		nodeID = client.PublicKeyHex()
	}
	return llamacpp.BackendBenchmarkProfile{
		NodeID:              nodeID,
		Acceleration:        llamaCppBenchmarkAccelerationMode(cfg, hardware),
		Warm:                runtimes.LlamaCPP.Warm,
		ContextLengthTokens: cfg.ContextSize,
		StreamingSupported:  runtimes.LlamaCPP.SupportsStreaming,
	}
}

func llamaCppBenchmarkAccelerationMode(cfg llamacpp.LlamaCppSidecarConfig, hardware capshardware.CapacityInventory) string {
	if !llamaCppConfigUsesGPU(cfg) {
		return "cpu"
	}
	hardware = capshardware.NormalizeInventory(hardware)
	switch {
	case hardware.CUDAAvailable:
		return "cuda"
	case hardware.MetalAvailable:
		return "metal"
	case hardware.VulkanAvailable:
		return "vulkan"
	case hardware.DirectMLAvailable:
		return "directml"
	default:
		return "other"
	}
}

func llamaCppConfigUsesGPU(cfg llamacpp.LlamaCppSidecarConfig) bool {
	if cfg.GPULayers > 0 {
		return true
	}
	for i, arg := range cfg.ExtraArgs {
		arg = strings.TrimSpace(arg)
		switch {
		case strings.HasPrefix(arg, "--n-gpu-layers="):
			value := strings.TrimSpace(strings.TrimPrefix(arg, "--n-gpu-layers="))
			n, err := strconv.Atoi(value)
			return err == nil && n != 0
		case arg == "--n-gpu-layers" || arg == "-ngl":
			if i+1 >= len(cfg.ExtraArgs) {
				return false
			}
			n, err := strconv.Atoi(strings.TrimSpace(cfg.ExtraArgs[i+1]))
			return err == nil && n != 0
		}
	}
	return false
}

func processOptionalV7LlamaCppBackendBenchmark(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, runtimeMgr *runtimeManager, gpuDetected bool) (bool, *runnerResultSnapshot, error) {
	identity, isBenchmark := llamacpp.BackendBenchmarkAssignmentIdentityFromJSON(work.SpecJSON)
	statusJobID := firstNonEmptyString(work.JobID, identity.JobID)
	benchmarkEnabled := isBenchmark && llamacpp.BackendBenchmarkEnabledFromEnv(os.Getenv)
	status := currentV7BackendBenchmarkStatus()
	if benchmarkEnabled && status != nil {
		status.RecordSeen(statusJobID)
	}

	runner := newV7LlamaCppBackendBenchmarkRunner()
	executionStarted := time.Now()
	if benchmarkEnabled {
		workLoopDiagnostics.RecordExecutionStart(statusJobID)
		workLoopDiagnostics.RecordEvent("v7_fast_path_start", statusJobID, work.Kind, v7LlamaCppBackendBenchmarkWorkLoopEventContextFromSpec(work.SpecJSON))
	}
	receipt, handled, err := llamacpp.ExecuteBackendBenchmarkAssignment(ctx, work.SpecJSON, llamacpp.ExecuteBackendBenchmarkOptions{
		Getenv:  os.Getenv,
		Runner:  runner,
		Profile: currentV7LlamaCppBackendBenchmarkProfile(ctx, client),
	})
	if !handled {
		return false, nil, nil
	}
	workLoopDiagnostics.RecordExecutionEnd(time.Since(executionStarted), err)
	if err != nil && status != nil {
		status.RecordError(statusJobID, err)
	}

	receiptBuildStarted := time.Now()
	workLoopDiagnostics.RecordEvent("pre_submit_block_start", firstNonEmptyString(receipt.JobID, statusJobID), work.Kind, v7LlamaCppBackendBenchmarkWorkLoopEventContextFromSpec(work.SpecJSON))
	runtimeMeta := v7BenchmarkFastPathRuntimeMetadata(runtimeMgr, gpuDetected)
	extra := map[string]any{
		"executor":      llamacpp.BackendBenchmarkTask,
		"executor_kind": llamacpp.BackendBenchmarkTask,
		"task":          llamacpp.BackendBenchmarkTask,
	}
	if strings.TrimSpace(receipt.ResultHashHex) == "" {
		receipt = llamacpp.BuildBackendBenchmarkRejectionReceipt(work.JobID, err)
	}
	exitCode := 0
	if err != nil {
		exitCode = 1
		extra["exit_code"] = 1
		extra["error"] = "v7 llama.cpp backend benchmark failed"
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
	receiptContext := v7LlamaCppBackendBenchmarkWorkLoopEventContextFromReceipt(work.SpecJSON, receipt)
	workLoopDiagnostics.RecordEvent("pre_submit_block_end", hubReceipt.JobID, work.Kind, receiptContext)
	workLoopDiagnostics.RecordEvent("v7_fast_path_receipt_ready", hubReceipt.JobID, work.Kind, receiptContext)
	workLoopDiagnostics.RecordReceiptReady(hubReceipt.JobID, work.Kind, time.Now(), receiptContext)
	workLoopDiagnostics.RecordEvent("v7_fast_path_submit_start", hubReceipt.JobID, work.Kind, receiptContext)
	if client == nil {
		submitErr := fmt.Errorf("hub client unavailable")
		workLoopDiagnostics.RecordReceiptSubmitStart(hubReceipt.JobID, 1)
		workLoopDiagnostics.RecordReceiptSubmitEnd(0, submitErr)
		workLoopDiagnostics.RecordEvent("v7_fast_path_submit_end", hubReceipt.JobID, work.Kind, receiptContext)
		if status != nil {
			if err == nil {
				status.RecordExecuted(hubReceipt.JobID)
			}
			status.RecordReceiptFailed(hubReceipt.JobID, submitErr)
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
	workLoopDiagnostics.RecordEvent("v7_fast_path_submit_end", hubReceipt.JobID, work.Kind, receiptContext)
	if submitErr != nil {
		if status != nil {
			if err == nil {
				status.RecordExecuted(hubReceipt.JobID)
			}
			status.RecordReceiptFailed(hubReceipt.JobID, submitErr)
		}
		if err != nil {
			return true, snapshot, fmt.Errorf("%v; receipt submit failed: %w", err, submitErr)
		}
		return true, snapshot, submitErr
	}
	if status != nil {
		if err == nil {
			status.RecordExecuted(hubReceipt.JobID)
		}
		status.RecordReceiptSubmitted(hubReceipt.JobID)
	}
	return true, snapshot, err
}

func v7LlamaCppBackendBenchmarkWorkLoopEventContextFromSpec(specJSON string) map[string]string {
	context := map[string]string{
		"spec_task": llamacpp.BackendBenchmarkTask,
	}
	var spec struct {
		Task         string `json:"task"`
		Backend      string `json:"backend"`
		ModelID      string `json:"model_id"`
		MeasuredRuns int    `json:"measured_runs"`
	}
	if err := json.Unmarshal([]byte(specJSON), &spec); err != nil {
		return context
	}
	if strings.TrimSpace(spec.Task) == llamacpp.BackendBenchmarkTask {
		context["spec_task"] = llamacpp.BackendBenchmarkTask
	}
	if backend := strings.TrimSpace(spec.Backend); backend != "" {
		context["backend"] = backend
	}
	if modelID := strings.TrimSpace(spec.ModelID); modelID != "" {
		context["model_id"] = modelID
	}
	if spec.MeasuredRuns > 0 {
		context["measured_runs"] = strconv.Itoa(spec.MeasuredRuns)
	}
	return context
}

func v7LlamaCppBackendBenchmarkWorkLoopEventContextFromReceipt(specJSON string, receipt llamacpp.BackendBenchmarkReceipt) map[string]string {
	context := v7LlamaCppBackendBenchmarkWorkLoopEventContextFromSpec(specJSON)
	if taskMetadata, ok := receipt.Metadata[llamacpp.BackendBenchmarkTask].(map[string]any); ok {
		putWorkLoopAnyIntContext(context, "warmup_runs", taskMetadata["warmup_runs"])
		putWorkLoopAnyIntContext(context, "measured_runs", taskMetadata["measured_runs"])
		putWorkLoopAnyIntContext(context, "p50_ttft_ms", taskMetadata["p50_ttft_ms"])
		if status, ok := taskMetadata["proof_status"].(string); ok && strings.TrimSpace(status) != "" {
			context["proof_status"] = strings.TrimSpace(status)
		}
	}
	return context
}

func currentV7DashboardInferenceStatus() *routedinference.LocalStatus {
	if operatorRuntimeState != nil {
		return operatorRuntimeState.dashboardInferenceStatus()
	}
	return v7DashboardInferenceStatus
}

func processOptionalV7DashboardInference(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, runtimeMgr *runtimeManager, gpuDetected bool) (bool, *runnerResultSnapshot, error) {
	identity, isDashboardInference := routedinference.AssignmentIdentityFromJSON(work.SpecJSON)
	statusJobID := firstNonEmptyString(work.JobID, identity.JobID)
	statusRunID := identity.RunID
	status := currentV7DashboardInferenceStatus()
	if isDashboardInference && status != nil {
		status.RecordSeen(statusRunID, statusJobID)
	}

	runner := newV7DashboardInferenceRunner()
	var progress routedinference.ProgressSender
	if client != nil {
		progress = client
	}
	executionStarted := time.Now()
	if isDashboardInference {
		workLoopDiagnostics.RecordExecutionStart(statusJobID)
		workLoopDiagnostics.RecordEvent("v7_fast_path_start", statusJobID, work.Kind, v7DashboardInferenceWorkLoopEventContextFromSpec(work.SpecJSON))
	}
	receipt, handled, err := routedinference.ExecuteAssignment(ctx, work.SpecJSON, routedinference.ExecuteOptions{
		Getenv:   os.Getenv,
		Runner:   runner,
		Progress: progress,
	})
	if !handled {
		return false, nil, nil
	}
	workLoopDiagnostics.RecordExecutionEnd(time.Since(executionStarted), err)
	if err != nil && status != nil {
		status.RecordError(statusRunID, statusJobID, err)
	}

	receiptBuildStarted := time.Now()
	workLoopDiagnostics.RecordEvent("pre_submit_block_start", firstNonEmptyString(receipt.JobID, statusJobID), work.Kind, v7DashboardInferenceWorkLoopEventContextFromSpec(work.SpecJSON))
	runtimeMeta := v7BenchmarkFastPathRuntimeMetadata(runtimeMgr, gpuDetected)
	extra := map[string]any{
		"executor":      routedinference.Task,
		"executor_kind": routedinference.Task,
		"task":          routedinference.Task,
	}
	if strings.TrimSpace(receipt.ResultHashHex) == "" {
		receipt = routedinference.BuildRejectionReceiptFromIdentity(identity, err)
	}
	proofStatus := routedinference.ReceiptProofStatus(receipt)
	measured := proofStatus == routedinference.ProofStatusMeasured
	recordErr := err
	if !measured && recordErr == nil {
		recordErr = v7DashboardInferenceRejectionError(receipt)
	}
	exitCode := 0
	if recordErr != nil || !measured {
		exitCode = 1
		extra["exit_code"] = 1
		extra["error"] = "v7 dashboard inference failed"
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
	receiptContext := v7DashboardInferenceWorkLoopEventContextFromReceipt(work.SpecJSON, receipt)
	workLoopDiagnostics.RecordEvent("pre_submit_block_end", hubReceipt.JobID, work.Kind, receiptContext)
	workLoopDiagnostics.RecordEvent("v7_fast_path_receipt_ready", hubReceipt.JobID, work.Kind, receiptContext)
	workLoopDiagnostics.RecordReceiptReady(hubReceipt.JobID, work.Kind, time.Now(), receiptContext)
	workLoopDiagnostics.RecordEvent("v7_fast_path_submit_start", hubReceipt.JobID, work.Kind, receiptContext)
	if client == nil {
		submitErr := fmt.Errorf("hub client unavailable")
		workLoopDiagnostics.RecordReceiptSubmitStart(hubReceipt.JobID, 1)
		workLoopDiagnostics.RecordReceiptSubmitEnd(0, submitErr)
		workLoopDiagnostics.RecordEvent("v7_fast_path_submit_end", hubReceipt.JobID, work.Kind, receiptContext)
		if status != nil {
			if measured {
				status.RecordExecuted(statusRunID, hubReceipt.JobID)
			} else {
				status.RecordRejected(statusRunID, hubReceipt.JobID, recordErr)
			}
			status.RecordReceiptFailed(statusRunID, hubReceipt.JobID, submitErr)
		}
		if recordErr != nil {
			return true, snapshot, fmt.Errorf("%v; receipt submit failed: %w", recordErr, submitErr)
		}
		return true, snapshot, submitErr
	}
	submitStarted := time.Now()
	workLoopDiagnostics.RecordReceiptSubmitStart(hubReceipt.JobID, 1)
	submitErr := client.SubmitReceipt(ctx, hubReceipt)
	workLoopDiagnostics.RecordReceiptSubmitEnd(time.Since(submitStarted), submitErr)
	workLoopDiagnostics.RecordEvent("v7_fast_path_submit_end", hubReceipt.JobID, work.Kind, receiptContext)
	if submitErr != nil {
		if status != nil {
			if measured {
				status.RecordExecuted(statusRunID, hubReceipt.JobID)
			} else {
				status.RecordRejected(statusRunID, hubReceipt.JobID, recordErr)
			}
			status.RecordReceiptFailed(statusRunID, hubReceipt.JobID, submitErr)
		}
		if recordErr != nil {
			return true, snapshot, fmt.Errorf("%v; receipt submit failed: %w", recordErr, submitErr)
		}
		return true, snapshot, submitErr
	}
	if status != nil {
		if measured {
			status.RecordExecuted(statusRunID, hubReceipt.JobID)
		} else {
			status.RecordRejected(statusRunID, hubReceipt.JobID, recordErr)
		}
		status.RecordReceiptSubmitted(statusRunID, hubReceipt.JobID)
	}
	if measured {
		requestV7CapabilityHeartbeat("v7_dashboard_inference_warm_changed")
	}
	return true, snapshot, recordErr
}

func v7DashboardInferenceWorkLoopEventContextFromSpec(specJSON string) map[string]string {
	context := map[string]string{
		"spec_task": routedinference.Task,
	}
	var spec struct {
		Task      string `json:"task"`
		RunID     string `json:"run_id"`
		Backend   string `json:"backend"`
		ModelID   string `json:"model_id"`
		MaxTokens int    `json:"max_tokens"`
	}
	if err := json.Unmarshal([]byte(specJSON), &spec); err != nil {
		return context
	}
	if strings.TrimSpace(spec.Task) == routedinference.Task {
		context["spec_task"] = routedinference.Task
	}
	if runID := strings.TrimSpace(spec.RunID); runID != "" {
		context["run_id"] = runID
	}
	if backend := strings.TrimSpace(spec.Backend); backend != "" {
		context["backend"] = backend
	}
	if modelID := strings.TrimSpace(spec.ModelID); modelID != "" {
		context["model_id"] = modelID
	}
	if spec.MaxTokens > 0 {
		context["max_tokens"] = strconv.Itoa(spec.MaxTokens)
	}
	return context
}

func v7DashboardInferenceWorkLoopEventContextFromReceipt(specJSON string, receipt routedinference.Receipt) map[string]string {
	context := v7DashboardInferenceWorkLoopEventContextFromSpec(specJSON)
	if taskMetadata, ok := receipt.Metadata[routedinference.Task].(map[string]any); ok {
		putWorkLoopAnyIntContext(context, "tokens_generated", taskMetadata["tokens_generated"])
		putWorkLoopAnyIntContext(context, "ttft_ms", taskMetadata["ttft_ms"])
		putWorkLoopAnyIntContext(context, "total_time_ms", taskMetadata["total_time_ms"])
		if status, ok := taskMetadata["proof_status"].(string); ok && strings.TrimSpace(status) != "" {
			context["proof_status"] = strings.TrimSpace(status)
		}
	}
	return context
}

func v7DashboardInferenceRejectionError(receipt routedinference.Receipt) error {
	code := "dashboard_inference_rejected"
	if taskMetadata, ok := receipt.Metadata[routedinference.Task].(map[string]any); ok {
		if value, ok := taskMetadata["error_code"].(string); ok && strings.TrimSpace(value) != "" {
			code = strings.TrimSpace(value)
		}
	}
	return fmt.Errorf("%s", code)
}

func currentV7BackendInferenceBenchmarkStatus() *inferencebench.LocalStatus {
	if operatorRuntimeState != nil {
		return operatorRuntimeState.inferenceBenchmarkStatus()
	}
	return v7InferenceBenchmarkStatus
}

func processOptionalV7BackendInferenceBenchmark(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, runtimeMgr *runtimeManager, gpuDetected bool) (bool, *runnerResultSnapshot, error) {
	identity, isBenchmark := inferencebench.BenchmarkAssignmentIdentityFromJSON(work.SpecJSON)
	statusJobID := firstNonEmptyString(work.JobID, identity.JobID)
	benchmarkEnabled := isBenchmark && inferencebench.BenchmarkEnabledFromEnv(os.Getenv)
	status := currentV7BackendInferenceBenchmarkStatus()
	if benchmarkEnabled && status != nil {
		status.RecordSeen(statusJobID)
	}

	runner := newV7BackendInferenceBenchmarkRunner()
	executionStarted := time.Now()
	if benchmarkEnabled {
		workLoopDiagnostics.RecordExecutionStart(statusJobID)
		workLoopDiagnostics.RecordEvent("v7_fast_path_start", statusJobID, work.Kind, v7BackendInferenceBenchmarkWorkLoopEventContextFromSpec(work.SpecJSON))
	}
	receipt, handled, err := inferencebench.ExecuteBenchmarkAssignment(ctx, work.SpecJSON, inferencebench.ExecuteOptions{
		Getenv: os.Getenv,
		Runner: runner,
	})
	if !handled {
		return false, nil, nil
	}
	workLoopDiagnostics.RecordExecutionEnd(time.Since(executionStarted), err)
	if err != nil && status != nil {
		status.RecordError(statusJobID, err)
	}

	receiptBuildStarted := time.Now()
	workLoopDiagnostics.RecordEvent("pre_submit_block_start", firstNonEmptyString(receipt.JobID, statusJobID), work.Kind, v7BackendInferenceBenchmarkWorkLoopEventContextFromSpec(work.SpecJSON))
	runtimeMeta := v7BenchmarkFastPathRuntimeMetadata(runtimeMgr, gpuDetected)
	extra := map[string]any{
		"executor":      inferencebench.BenchmarkTask,
		"executor_kind": inferencebench.BenchmarkTask,
		"task":          inferencebench.BenchmarkTask,
	}
	if strings.TrimSpace(receipt.ResultHashHex) == "" {
		receipt = inferencebench.BuildBenchmarkRejectionReceipt(statusJobID, err)
	}
	exitCode := 0
	if err != nil {
		exitCode = 1
		extra["exit_code"] = 1
		extra["error"] = "v7 backend inference benchmark failed"
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
	receiptContext := v7BackendInferenceBenchmarkWorkLoopEventContextFromReceipt(work.SpecJSON, receipt)
	workLoopDiagnostics.RecordEvent("pre_submit_block_end", hubReceipt.JobID, work.Kind, receiptContext)
	workLoopDiagnostics.RecordEvent("v7_fast_path_receipt_ready", hubReceipt.JobID, work.Kind, receiptContext)
	workLoopDiagnostics.RecordReceiptReady(hubReceipt.JobID, work.Kind, time.Now(), receiptContext)
	workLoopDiagnostics.RecordEvent("v7_fast_path_submit_start", hubReceipt.JobID, work.Kind, receiptContext)
	if client == nil {
		submitErr := fmt.Errorf("hub client unavailable")
		workLoopDiagnostics.RecordReceiptSubmitStart(hubReceipt.JobID, 1)
		workLoopDiagnostics.RecordReceiptSubmitEnd(0, submitErr)
		workLoopDiagnostics.RecordEvent("v7_fast_path_submit_end", hubReceipt.JobID, work.Kind, receiptContext)
		if status != nil {
			if err == nil {
				status.RecordExecuted(hubReceipt.JobID)
			}
			status.RecordReceiptFailed(hubReceipt.JobID, submitErr)
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
	workLoopDiagnostics.RecordEvent("v7_fast_path_submit_end", hubReceipt.JobID, work.Kind, receiptContext)
	if submitErr != nil {
		if status != nil {
			if err == nil {
				status.RecordExecuted(hubReceipt.JobID)
			}
			status.RecordReceiptFailed(hubReceipt.JobID, submitErr)
		}
		if err != nil {
			return true, snapshot, fmt.Errorf("%v; receipt submit failed: %w", err, submitErr)
		}
		return true, snapshot, submitErr
	}
	if status != nil {
		if err == nil {
			status.RecordExecuted(firstNonEmptyString(receipt.JobID, statusJobID))
		}
		status.RecordReceiptSubmitted(hubReceipt.JobID)
	}
	return true, snapshot, err
}

func v7BackendInferenceBenchmarkWorkLoopEventContextFromSpec(specJSON string) map[string]string {
	context := map[string]string{
		"spec_task": inferencebench.BenchmarkTask,
	}
	var spec struct {
		Task      string `json:"task"`
		Backend   string `json:"backend"`
		ModelID   string `json:"model_id"`
		MaxTokens int    `json:"max_tokens"`
	}
	if err := json.Unmarshal([]byte(specJSON), &spec); err != nil {
		return context
	}
	if strings.TrimSpace(spec.Task) == inferencebench.BenchmarkTask {
		context["spec_task"] = inferencebench.BenchmarkTask
	}
	if backend := strings.TrimSpace(spec.Backend); backend != "" {
		context["backend"] = backend
	}
	if modelID := strings.TrimSpace(spec.ModelID); modelID != "" {
		context["model_id"] = modelID
	}
	if spec.MaxTokens > 0 {
		context["max_tokens"] = strconv.Itoa(spec.MaxTokens)
	}
	return context
}

func v7BackendInferenceBenchmarkWorkLoopEventContextFromReceipt(specJSON string, receipt inferencebench.BenchmarkReceipt) map[string]string {
	context := v7BackendInferenceBenchmarkWorkLoopEventContextFromSpec(specJSON)
	if taskMetadata, ok := receipt.Metadata[inferencebench.BenchmarkTask].(map[string]any); ok {
		putWorkLoopAnyIntContext(context, "tokens_generated", taskMetadata["tokens_generated"])
		putWorkLoopAnyIntContext(context, "p50_ttft_ms", taskMetadata["p50_ttft_ms"])
		putWorkLoopAnyIntContext(context, "p95_ttft_ms", taskMetadata["p95_ttft_ms"])
		if status, ok := taskMetadata["proof_status"].(string); ok && strings.TrimSpace(status) != "" {
			context["proof_status"] = strings.TrimSpace(status)
		}
	}
	return context
}

func processOptionalV7ModelBenchmark(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, infMgr *inference.Manager, runtimeMgr *runtimeManager, gpuDetected bool) (bool, *runnerResultSnapshot, error) {
	identity, isBenchmark := modelbench.ModelBenchmarkAssignmentIdentityFromJSON(work.SpecJSON)
	taskName := modelbench.ModelBenchmarkTask
	if !isBenchmark {
		if seriesIdentity, isSeriesBenchmark := modelbench.ModelBenchmarkSeriesAssignmentIdentityFromJSON(work.SpecJSON); isSeriesBenchmark {
			identity = seriesIdentity
			isBenchmark = true
			taskName = modelbench.ModelBenchmarkSeriesTask
		}
	}
	statusJobID := firstNonEmptyString(work.JobID, identity.JobID)
	if isBenchmark && modelbench.ModelBenchmarkEnabledFromEnv(os.Getenv) && v7ModelBenchmarkStatus != nil {
		v7ModelBenchmarkStatus.RecordSeen(statusJobID, identity.RequestID)
	}

	runner := newV7ModelBenchmarkRunner(infMgr, gpuDetected)
	executionStarted := time.Now()
	if isBenchmark && modelbench.ModelBenchmarkEnabledFromEnv(os.Getenv) {
		workLoopDiagnostics.RecordExecutionStart(statusJobID)
		workLoopDiagnostics.RecordEvent("v7_fast_path_start", statusJobID, work.Kind, mainWorkLoopSpecContext(taskName))
	}
	receipt, handled, err := modelbench.ExecuteModelBenchmarkAssignment(ctx, work.SpecJSON, runner, os.Getenv)
	if !handled {
		receipt, handled, err = modelbench.ExecuteModelBenchmarkSeriesAssignment(ctx, work.SpecJSON, runner, os.Getenv)
	}
	if !handled {
		return false, nil, nil
	}
	workLoopDiagnostics.RecordExecutionEnd(time.Since(executionStarted), err)
	if err != nil && v7ModelBenchmarkStatus != nil {
		v7ModelBenchmarkStatus.RecordError(err)
	}

	receiptBuildStarted := time.Now()
	workLoopDiagnostics.RecordEvent("pre_submit_block_start", firstNonEmptyString(receipt.JobID, statusJobID), work.Kind, mainWorkLoopSpecContext(taskName))
	runtimeMeta := v7BenchmarkFastPathRuntimeMetadata(runtimeMgr, gpuDetected)
	extra := map[string]any{
		"executor":      taskName,
		"executor_kind": taskName,
		"task":          taskName,
	}
	if strings.TrimSpace(receipt.ResultHashHex) == "" {
		if taskName == modelbench.ModelBenchmarkSeriesTask {
			receipt = modelbench.BuildModelBenchmarkSeriesRejectionReceipt(work.JobID, err)
		} else {
			receipt = modelbench.BuildModelBenchmarkRejectionReceipt(work.JobID, err)
		}
	}
	exitCode := 0
	if err != nil {
		exitCode = 1
		extra["exit_code"] = 1
		if taskName == modelbench.ModelBenchmarkSeriesTask {
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
	receiptContext := mainWorkLoopSpecContext(taskName)
	workLoopDiagnostics.RecordEvent("pre_submit_block_end", hubReceipt.JobID, work.Kind, receiptContext)
	workLoopDiagnostics.RecordEvent("v7_fast_path_receipt_ready", hubReceipt.JobID, work.Kind, receiptContext)
	workLoopDiagnostics.RecordReceiptReady(hubReceipt.JobID, work.Kind, time.Now(), receiptContext)
	workLoopDiagnostics.RecordEvent("v7_fast_path_submit_start", hubReceipt.JobID, work.Kind, receiptContext)
	if client == nil {
		submitErr := fmt.Errorf("hub client unavailable")
		workLoopDiagnostics.RecordReceiptSubmitStart(hubReceipt.JobID, 1)
		workLoopDiagnostics.RecordReceiptSubmitEnd(0, submitErr)
		workLoopDiagnostics.RecordEvent("v7_fast_path_submit_end", hubReceipt.JobID, work.Kind, receiptContext)
		if v7ModelBenchmarkStatus != nil {
			if err == nil {
				v7ModelBenchmarkStatus.RecordExecuted(hubReceipt.JobID)
			}
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
	workLoopDiagnostics.RecordEvent("v7_fast_path_submit_end", hubReceipt.JobID, work.Kind, receiptContext)
	if submitErr != nil {
		if v7ModelBenchmarkStatus != nil {
			if err == nil {
				v7ModelBenchmarkStatus.RecordExecuted(hubReceipt.JobID)
			}
			v7ModelBenchmarkStatus.RecordReceiptFailed(hubReceipt.JobID, submitErr)
		}
		if err != nil {
			return true, snapshot, fmt.Errorf("%v; receipt submit failed: %w", err, submitErr)
		}
		return true, snapshot, submitErr
	}
	if v7ModelBenchmarkStatus != nil {
		if err == nil {
			v7ModelBenchmarkStatus.RecordExecuted(hubReceipt.JobID)
		}
		v7ModelBenchmarkStatus.RecordReceiptSubmitted(hubReceipt.JobID)
	}
	return true, snapshot, err
}

func currentV7ModelPrepareStatus() *modelprepare.LocalStatus {
	if operatorRuntimeState != nil {
		return operatorRuntimeState.modelPrepareStatus()
	}
	return v7ModelPrepareStatus
}

func processOptionalV7ModelPrepare(ctx context.Context, client *hub.Client, work *hub.WorkAssignment, runtimeMgr *runtimeManager, gpuDetected bool) (bool, *runnerResultSnapshot, error) {
	identity, isPrepare := modelprepare.PrepareAssignmentIdentityFromJSON(work.SpecJSON)
	if !isPrepare {
		return false, nil, nil
	}
	status := currentV7ModelPrepareStatus()
	if status != nil {
		status.RecordSeen(identity.PrepareID, identity.ModelID)
	}
	statusJobID := firstNonEmptyString(work.JobID, identity.JobID)
	executionStarted := time.Now()
	workLoopDiagnostics.RecordExecutionStart(statusJobID)
	workLoopDiagnostics.RecordEvent("v7_fast_path_start", statusJobID, work.Kind, mainWorkLoopSpecContext(modelprepare.PrepareTask))

	var manager modelprepare.LlamaCppManager
	if operatorRuntimeState != nil {
		manager = operatorRuntimeState.llamaCppManager()
	} else {
		manager = llamacpp.NewManagerFromEnv()
	}
	receipt, handled, err := modelprepare.ExecutePrepareAssignment(ctx, work.SpecJSON, modelprepare.ExecuteOptions{
		Getenv:          os.Getenv,
		Policy:          modelpolicy.FromEnv(),
		LlamaCppManager: manager,
		BenchmarkRunner: llamacpp.BenchmarkRunner{
			Sidecar: manager,
			Client:  llamacpp.OpenAIClient{},
		},
	})
	if !handled {
		return false, nil, nil
	}
	workLoopDiagnostics.RecordExecutionEnd(time.Since(executionStarted), err)
	if strings.TrimSpace(receipt.ResultHashHex) == "" {
		receipt = modelprepare.BuildPrepareRejectionReceiptFromIdentity(identity, err)
	}
	receiptBuildStarted := time.Now()
	workLoopDiagnostics.RecordEvent("pre_submit_block_start", firstNonEmptyString(receipt.JobID, statusJobID), work.Kind, mainWorkLoopSpecContext(modelprepare.PrepareTask))
	extra := map[string]any{
		"executor":      modelprepare.PrepareTask,
		"executor_kind": modelprepare.PrepareTask,
		"task":          modelprepare.PrepareTask,
	}
	exitCode := 0
	if err != nil {
		exitCode = 1
		extra["exit_code"] = 1
		extra["error"] = "v7 model prepare failed"
	} else {
		extra["exit_code"] = 0
	}
	metadata := receiptMetadataBase(work, v7BenchmarkFastPathRuntimeMetadata(runtimeMgr, gpuDetected), receipt.Metadata, extra)
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
	receiptContext := mainWorkLoopSpecContext(modelprepare.PrepareTask)
	workLoopDiagnostics.RecordEvent("pre_submit_block_end", hubReceipt.JobID, work.Kind, receiptContext)
	workLoopDiagnostics.RecordEvent("v7_fast_path_receipt_ready", hubReceipt.JobID, work.Kind, receiptContext)
	workLoopDiagnostics.RecordReceiptReady(hubReceipt.JobID, work.Kind, time.Now(), receiptContext)
	workLoopDiagnostics.RecordEvent("v7_fast_path_submit_start", hubReceipt.JobID, work.Kind, receiptContext)
	if client == nil {
		submitErr := fmt.Errorf("hub client unavailable")
		workLoopDiagnostics.RecordReceiptSubmitStart(hubReceipt.JobID, 1)
		workLoopDiagnostics.RecordReceiptSubmitEnd(0, submitErr)
		workLoopDiagnostics.RecordEvent("v7_fast_path_submit_end", hubReceipt.JobID, work.Kind, receiptContext)
		if status != nil {
			if err == nil {
				status.RecordExecuted(identity.PrepareID, identity.ModelID)
			} else {
				status.RecordRejected(identity.PrepareID, identity.ModelID, err)
			}
			status.RecordReceiptFailed(identity.PrepareID, identity.ModelID, submitErr)
		}
		if err != nil {
			return true, snapshot, fmt.Errorf("%v; receipt submit failed: %w", err, submitErr)
		}
		return true, snapshot, submitErr
	}
	if submitErr := submitReceiptWithRetry(ctx, client, hubReceipt); submitErr != nil {
		workLoopDiagnostics.RecordEvent("v7_fast_path_submit_end", hubReceipt.JobID, work.Kind, receiptContext)
		if status != nil {
			if err == nil {
				status.RecordExecuted(identity.PrepareID, identity.ModelID)
			} else {
				status.RecordRejected(identity.PrepareID, identity.ModelID, err)
			}
			status.RecordReceiptFailed(identity.PrepareID, identity.ModelID, submitErr)
		}
		if err != nil {
			return true, snapshot, fmt.Errorf("%v; receipt submit failed: %w", err, submitErr)
		}
		return true, snapshot, submitErr
	}
	workLoopDiagnostics.RecordEvent("v7_fast_path_submit_end", hubReceipt.JobID, work.Kind, receiptContext)
	if status != nil {
		if err == nil {
			status.RecordExecuted(identity.PrepareID, identity.ModelID)
		} else {
			status.RecordRejected(identity.PrepareID, identity.ModelID, err)
		}
		status.RecordReceiptSubmitted(identity.PrepareID, identity.ModelID)
	}
	if err == nil {
		requestV7CapabilityHeartbeat("v7_model_prepare_changed")
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

		proof, err := proofrunner.BuildProofCarryingOutput(proofrunner.RunnerResultInput{
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

func fullV7ProofMetadata(proof proofrunner.ProofCarryingRunnerOutput) map[string]any {
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

func legacyNativeInferenceManagerEnabled(getenv func(string) string) bool {
	if getenv == nil {
		getenv = os.Getenv
	}
	switch strings.ToLower(strings.TrimSpace(getenv(legacyNativeInferenceFlagEnv))) {
	case "1", "true", "yes", "on", "enabled":
		return true
	case "0", "false", "no", "off", "disabled":
		return false
	}
	// Idle-safe default: do not launch llama-server at OS startup. Consumer
	// operator machines must stay usable until an explicit warm setting or
	// an assigned inference job starts the model runtime.
	return false
}

func mainWorkLoopSpecContext(specTask string) map[string]string {
	specTask = strings.TrimSpace(specTask)
	if specTask == "" {
		return nil
	}
	return map[string]string{"spec_task": specTask}
}

func v7ProofArtifactKind(metadata map[string]any) artifact.ArtifactKind {
	kind := strings.ToLower(firstNonEmptyString(
		metadataString(metadata, "artifact_kind"),
		metadataString(metadata, "artifact_mime"),
		metadataString(metadata, "mime_type"),
	))
	switch kind {
	case "result_json", "json", "application/json":
		return artifact.ArtifactKindResultJSON
	case "image", "image/png", "image/jpeg", "image/webp":
		return artifact.ArtifactKindImage
	case "text", "text/plain":
		return artifact.ArtifactKindText
	case "model_delta":
		return artifact.ArtifactKindModelDelta
	case "evidence_bundle":
		return artifact.ArtifactKindEvidenceBundle
	case "runner_log":
		return artifact.ArtifactKindRunnerLog
	case "", "generic_blob", "application/octet-stream":
		return artifact.ArtifactKindGenericBlob
	default:
		return artifact.ArtifactKind(kind)
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
		// Surface a structured blocker so the hub dashboard can show the
		// operator a concrete reason (binary-missing, health-check-timeout,
		// model-download-failed, process-failed, disk-space, starting,
		// not-started, platform-unsupported, not-installed). Without this
		// token, operators see "degraded" but cannot tell whether to wait
		// for a cold load or to investigate a missing GPU bundle.
		parts = append(parts, "native-inference-blocker:"+nativeInferenceBlockerToken(nativeSupported, infMgr))
	}
	if publicInferenceReady {
		parts = append(parts, "public-inference-ready:1")
		for _, modelID := range inference.SupportedNativeChatModels(caps.VRAMBytes) {
			parts = append(parts, "model:"+modelID)
		}
	} else {
		parts = append(parts, "public-inference-ready:0")
	}
	parts = append(parts, speculativeRoleStatusTokens(caps, diskGB, publicInferenceReady)...)
	parts = append(parts, smartShardingStatusTokens(caps, diskGB, publicInferenceReady)...)
	parts = append(parts, worldGridRoleStatusTokens(caps, ffmpegOK, spatialReady)...)

	return hub.HealthReport{
		TimestampMs: time.Now().UnixMilli(),
		GPUReady:    gpuReady,
		RuntimeGPU:  runtimeSnap.GPUReady,
		Message:     strings.Join(parts, ","),
	}
}

func speculativeRoleStatusTokens(caps hw.CapSet, diskGB uint64, publicInferenceReady bool) []string {
	tokens := []string{}
	residueReady := caps.CPUCores >= 2 || strings.TrimSpace(caps.CPUModel) != ""
	if residueReady {
		tokens = append(tokens,
			"role:residue_swarm_builder",
			"cap:residue_swarm_builder:1",
			"speculative:proposal_window:ready:1",
		)
	} else {
		tokens = append(tokens, "cap:residue_swarm_builder:0", "speculative:proposal_window:ready:0")
	}
	if publicInferenceReady {
		tokens = append(tokens, "role:draft_worker", "cap:draft_worker:1")
	} else {
		tokens = append(tokens, "cap:draft_worker:0")
	}
	if diskGB >= 10 {
		tokens = append(tokens, "role:cache_provider", "cap:cache_provider:1")
	} else {
		tokens = append(tokens, "cap:cache_provider:0")
	}
	return tokens
}

func smartShardingStatusTokens(caps hw.CapSet, diskGB uint64, publicInferenceReady bool) []string {
	tokens := []string{
		"hmm_router_shadow:1",
		"smart_sharding:mode_aware:1",
		"scheduler:auto_apply:0",
		"pricing:auto_apply:0",
		"cap:smart_sharding_token_critical_wan:0",
	}
	observationReady := caps.CPUCores >= 2 || strings.TrimSpace(caps.CPUModel) != "" || strings.TrimSpace(caps.GPUModel) != ""
	if observationReady {
		tokens = append(tokens, "cap:hmm_router_observation:1", "cap:smart_sharding_scheduler:1")
	} else {
		tokens = append(tokens, "cap:hmm_router_observation:0", "cap:smart_sharding_scheduler:0")
	}
	tokens = append(tokens, boolStatusToken("cap:smart_sharding_draft_signal", publicInferenceReady))
	tokens = append(tokens, boolStatusToken("cap:smart_sharding_cache_signal", diskGB >= 10))
	return tokens
}

func worldGridRoleStatusTokens(caps hw.CapSet, ffmpegOK bool, spatialReady bool) []string {
	tokens := []string{}
	asyncReady := caps.CPUCores >= 4 || strings.TrimSpace(caps.CPUModel) != "" || strings.TrimSpace(caps.GPUModel) != ""
	frameReady := strings.TrimSpace(caps.GPUModel) != "" || caps.VRAMBytes >= 4*1024*1024*1024
	npcReady := asyncReady
	if asyncReady {
		tokens = append(tokens, "worldgrid_async:1", "cap:worldgrid_async_worker:1")
	} else {
		tokens = append(tokens, "worldgrid_async:0", "cap:worldgrid_async_worker:0")
	}
	if frameReady {
		tokens = append(tokens, "role:frame_worker", "cap:worldgrid_frame_worker:1")
	} else {
		tokens = append(tokens, "cap:worldgrid_frame_worker:0")
	}
	if ffmpegOK {
		tokens = append(tokens, "role:stream_segment_worker", "cap:worldgrid_stream_segment_worker:1")
	} else {
		tokens = append(tokens, "cap:worldgrid_stream_segment_worker:0")
	}
	if npcReady {
		tokens = append(tokens, "role:future_branch_proposer", "cap:worldgrid_npc_speculative:1")
	} else {
		tokens = append(tokens, "cap:worldgrid_npc_speculative:0")
	}
	if spatialReady {
		tokens = append(tokens, "role:stage_worker", "cap:worldgrid_spatial_worker:1")
	} else {
		tokens = append(tokens, "cap:worldgrid_spatial_worker:0")
	}
	tokens = append(tokens, "cap:worldgrid_realtime_cell:0")
	return tokens
}

// nativeInferenceBlockerToken returns the dashboard-safe blocker token used
// by the native-inference-blocker:<reason> heartbeat field. Caller must
// have already established that the native path is NOT ready.
func nativeInferenceBlockerToken(nativeSupported bool, infMgr *inference.Manager) string {
	if !nativeSupported {
		return string(inference.BlockerPlatformUnsupported)
	}
	if infMgr == nil {
		return string(inference.BlockerNotStarted)
	}
	reason := infMgr.BlockerReason()
	if reason == inference.BlockerNone {
		return string(inference.BlockerNotStarted)
	}
	return string(reason)
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
