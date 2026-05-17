package onboarding

import (
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	goruntime "runtime"
	"strings"
	"time"

	"github.com/Ryvion/ryvion-node/internal/hw"
	"github.com/Ryvion/ryvion-node/internal/inference"
	"github.com/Ryvion/ryvion-node/internal/runtimeexec"
	"github.com/Ryvion/ryvion-node/internal/sandbox"
	"github.com/Ryvion/ryvion-node/internal/v7/capability"
)

const reportSchemaVersion = "ryvion.node.onboarding_report.v1"

type CheckStatus string

const (
	CheckStatusOK        CheckStatus = "ok"
	CheckStatusWarning   CheckStatus = "warning"
	CheckStatusHardError CheckStatus = "hard_error"
)

type OnboardingCheck struct {
	ID      string      `json:"id"`
	Title   string      `json:"title"`
	Status  CheckStatus `json:"status"`
	Message string      `json:"message"`
}

type OnboardingReport struct {
	SchemaVersion string            `json:"schema_version"`
	OS            string            `json:"os"`
	Arch          string            `json:"arch"`
	GeneratedAt   time.Time         `json:"generated_at"`
	Checks        []OnboardingCheck `json:"checks"`
}

type CheckOptions struct {
	AgentVersion string
	HubURL       string
	DataDir      string
	DeviceType   string
	OS           string
	Arch         string

	Getenv                 func(string) string
	DetectCaps             func(deviceType string) hw.CapSet
	NativeRuntimeAvailable func() bool
	ResolveOCIEnginePath   func(goos string, getenv func(string) string) string
	Now                    func() time.Time
}

func RunBasicOnboardingChecks() OnboardingReport {
	return RunBasicOnboardingChecksWithOptions(CheckOptions{})
}

func RunBasicOnboardingChecksWithOptions(options CheckOptions) OnboardingReport {
	options = normalizeCheckOptions(options)

	caps := options.DetectCaps(options.DeviceType)
	gpuDetected := strings.TrimSpace(caps.GPUModel) != "" || caps.VRAMBytes > 0
	nativeAvailable := options.NativeRuntimeAvailable()
	ociPath := options.ResolveOCIEnginePath(options.OS, options.Getenv)
	ociAvailable := strings.TrimSpace(ociPath) != ""

	checks := []OnboardingCheck{
		checkSupportedPlatform(options.OS, options.Arch),
		checkDataDir(options.DataDir),
		checkNativeInference(options.OS, options.Arch, nativeAvailable),
		checkOptionalOCI(ociPath),
		checkGPUOrCPUFallback(caps, gpuDetected),
		checkHubURL(options.HubURL),
		checkCapabilityPassport(options, caps, gpuDetected, nativeAvailable, ociAvailable),
		checkSandboxPolicy(),
	}

	return OnboardingReport{
		SchemaVersion: reportSchemaVersion,
		OS:            options.OS,
		Arch:          options.Arch,
		GeneratedAt:   options.Now().UTC(),
		Checks:        checks,
	}
}

func (r OnboardingReport) HardErrorCount() int {
	count := 0
	for _, check := range r.Checks {
		if check.Status == CheckStatusHardError {
			count++
		}
	}
	return count
}

func (r OnboardingReport) WarningCount() int {
	count := 0
	for _, check := range r.Checks {
		if check.Status == CheckStatusWarning {
			count++
		}
	}
	return count
}

func (r OnboardingReport) HasHardErrors() bool {
	return r.HardErrorCount() > 0
}

func normalizeCheckOptions(options CheckOptions) CheckOptions {
	if options.Getenv == nil {
		options.Getenv = os.Getenv
	}
	if options.DetectCaps == nil {
		options.DetectCaps = hw.DetectCaps
	}
	if options.NativeRuntimeAvailable == nil {
		options.NativeRuntimeAvailable = inference.NativeRuntimeAvailable
	}
	if options.ResolveOCIEnginePath == nil {
		options.ResolveOCIEnginePath = runtimeexec.ResolveEnginePath
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	options.OS = firstNonEmpty(strings.TrimSpace(options.OS), goruntime.GOOS)
	options.Arch = firstNonEmpty(strings.TrimSpace(options.Arch), goruntime.GOARCH)
	options.HubURL = firstNonEmpty(strings.TrimSpace(options.HubURL), strings.TrimSpace(options.Getenv("RYV_HUB_URL")), "https://api.ryvion.ai")
	options.DataDir = firstNonEmpty(strings.TrimSpace(options.DataDir), strings.TrimSpace(options.Getenv("RYV_DATA_DIR")))
	options.AgentVersion = firstNonEmpty(strings.TrimSpace(options.AgentVersion), "dev")
	return options
}

func checkSupportedPlatform(goos, goarch string) OnboardingCheck {
	if supportedPlatform(goos, goarch) {
		return okCheck("platform", "Platform", fmt.Sprintf("%s/%s is supported", goos, goarch))
	}
	return hardErrorCheck("platform", "Platform", fmt.Sprintf("%s/%s is not currently supported", cleanToken(goos), cleanToken(goarch)))
}

func supportedPlatform(goos, goarch string) bool {
	switch goos {
	case "darwin", "linux", "windows":
	default:
		return false
	}
	switch goarch {
	case "amd64", "arm64":
		return true
	default:
		return false
	}
}

func checkDataDir(dataDir string) OnboardingCheck {
	dataDir = strings.TrimSpace(dataDir)
	if dataDir == "" {
		return okCheck("data_dir", "Data directory", "default data directory will be used")
	}
	if err := dataDirWritable(dataDir); err != nil {
		return hardErrorCheck("data_dir", "Data directory", "configured data directory is not writable: "+err.Error())
	}
	return okCheck("data_dir", "Data directory", "configured data directory is writable")
}

func dataDirWritable(dataDir string) error {
	info, err := os.Stat(dataDir)
	if err == nil {
		if !info.IsDir() {
			return fmt.Errorf("path is not a directory")
		}
		return writeProbe(dataDir)
	}
	if !os.IsNotExist(err) {
		return err
	}
	parent := filepath.Dir(dataDir)
	if parent == "" || parent == "." {
		parent = "."
	}
	parentInfo, parentErr := os.Stat(parent)
	if parentErr != nil {
		return fmt.Errorf("parent directory is unavailable")
	}
	if !parentInfo.IsDir() {
		return fmt.Errorf("parent path is not a directory")
	}
	if err := writeProbe(parent); err != nil {
		return fmt.Errorf("parent directory is not writable")
	}
	return nil
}

func writeProbe(dir string) error {
	file, err := os.CreateTemp(dir, ".ryvion-onboarding-*")
	if err != nil {
		return err
	}
	name := file.Name()
	closeErr := file.Close()
	removeErr := os.Remove(name)
	if closeErr != nil {
		return closeErr
	}
	return removeErr
}

func checkNativeInference(goos, goarch string, available bool) OnboardingCheck {
	if available {
		return okCheck("native_inference", "Native inference", "native llama-server path is available")
	}
	return warningCheck("native_inference", "Native inference", fmt.Sprintf("native llama-server path was not detected for %s/%s", cleanToken(goos), cleanToken(goarch)))
}

func checkOptionalOCI(enginePath string) OnboardingCheck {
	enginePath = strings.TrimSpace(enginePath)
	if enginePath == "" {
		return warningCheck("oci_runtime", "Optional OCI runtime", "Docker/Podman runtime was not detected; native onboarding can continue")
	}
	engineKind := runtimeexec.EngineKind(enginePath)
	if engineKind == "" {
		engineKind = "oci"
	}
	return okCheck("oci_runtime", "Optional OCI runtime", fmt.Sprintf("%s runtime detected", engineKind))
}

func checkGPUOrCPUFallback(caps hw.CapSet, gpuDetected bool) OnboardingCheck {
	if gpuDetected {
		model := strings.TrimSpace(caps.GPUModel)
		if model == "" {
			model = "GPU"
		}
		return okCheck("compute_fallback", "Compute", "GPU detected: "+cleanText(model))
	}
	return okCheck("compute_fallback", "Compute", "no GPU detected; CPU fallback is available")
}

func checkHubURL(rawHubURL string) OnboardingCheck {
	parsed, err := parseHubURL(rawHubURL)
	if err != nil {
		return hardErrorCheck("hub_url", "Hub URL", "invalid hub URL; set RYV_HUB_URL to an http(s) URL")
	}
	return okCheck("hub_url", "Hub URL", "hub URL parsed: "+safeURLForReport(parsed))
}

func parseHubURL(rawHubURL string) (*url.URL, error) {
	rawHubURL = strings.TrimSpace(rawHubURL)
	if rawHubURL == "" {
		return nil, fmt.Errorf("empty URL")
	}
	parsed, err := url.Parse(rawHubURL)
	if err != nil {
		return nil, err
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return nil, fmt.Errorf("unsupported scheme")
	}
	if strings.TrimSpace(parsed.Host) == "" {
		return nil, fmt.Errorf("host required")
	}
	return parsed, nil
}

func checkCapabilityPassport(options CheckOptions, caps hw.CapSet, gpuDetected, nativeAvailable, ociAvailable bool) OnboardingCheck {
	_, err := capability.BuildCapabilityPassport(capability.BuildPassportInput{
		AgentVersion:         options.AgentVersion,
		OS:                   options.OS,
		Arch:                 options.Arch,
		DeviceType:           inferredDeviceType(options.DeviceType, gpuDetected),
		HardwareCapabilities: caps,
		RuntimeProfile: capability.RuntimeProfile{
			NativeInferenceSupported: nativeAvailable,
			OCIAvailable:             ociAvailable,
			LlamaServerAvailable:     nativeAvailable,
			SupportedRunnerKinds:     supportedRunnerKinds(nativeAvailable, ociAvailable),
		},
		ModelCapabilitySummary: capability.ModelCapabilitySummary{
			MaxResidentModelBytes: caps.VRAMBytes,
			SupportsModelLease:    gpuDetected && nativeAvailable,
		},
		SandboxCapabilitySummary: capability.SandboxCapabilitySummary{
			RejectsUnsafePickle:        true,
			RunnerAllowlistEnabled:     true,
			FilesystemIsolationPlanned: true,
			NetworkIsolationSupported:  ociAvailable,
		},
		EvidenceCapabilitySummary: capability.EvidenceCapabilitySummary{
			SupportsArtifactManifest:    true,
			SupportsRYV3EvidencePayload: true,
		},
	})
	if err != nil {
		return hardErrorCheck("capability_passport", "Capability passport", "capability passport is not buildable: "+cleanText(err.Error()))
	}
	return okCheck("capability_passport", "Capability passport", "capability passport is buildable")
}

func checkSandboxPolicy() OnboardingCheck {
	policy := sandbox.DefaultSandboxPolicy()
	if policy.PythonSourceDecision == "" {
		return hardErrorCheck("sandbox_policy", "Sandbox policy", "default sandbox policy is missing")
	}
	result := sandbox.EvaluateSandbox(policy, sandbox.SandboxRequest{
		ModelPath:           "model.pkl",
		RunnerKind:          sandbox.RunnerKindNativeLlama,
		IsAllowlistedRunner: true,
		IsTrustedSource:     true,
	})
	if result.Decision != sandbox.SandboxDecisionReject {
		return hardErrorCheck("sandbox_policy", "Sandbox policy", "unsafe pickle model was not rejected")
	}
	return okCheck("sandbox_policy", "Sandbox policy", "default sandbox policy is present")
}

func supportedRunnerKinds(nativeAvailable, ociAvailable bool) []string {
	kinds := []string{"native_report"}
	if nativeAvailable {
		kinds = append(kinds, "native_streaming")
	}
	if ociAvailable {
		kinds = append(kinds, "managed_oci", "agent_hosting")
	}
	return kinds
}

func inferredDeviceType(raw string, gpuDetected bool) string {
	raw = strings.ToLower(strings.TrimSpace(raw))
	if raw != "" {
		return raw
	}
	if gpuDetected {
		return "gpu"
	}
	return "cpu"
}

func okCheck(id, title, message string) OnboardingCheck {
	return OnboardingCheck{ID: id, Title: title, Status: CheckStatusOK, Message: cleanText(message)}
}

func warningCheck(id, title, message string) OnboardingCheck {
	return OnboardingCheck{ID: id, Title: title, Status: CheckStatusWarning, Message: cleanText(message)}
}

func hardErrorCheck(id, title, message string) OnboardingCheck {
	return OnboardingCheck{ID: id, Title: title, Status: CheckStatusHardError, Message: cleanText(message)}
}

func safeURLForReport(parsed *url.URL) string {
	out := &url.URL{
		Scheme: parsed.Scheme,
		Host:   parsed.Host,
		Path:   strings.TrimRight(parsed.Path, "/"),
	}
	return out.String()
}

func cleanText(value string) string {
	value = strings.TrimSpace(value)
	value = strings.ReplaceAll(value, "\r", " ")
	value = strings.ReplaceAll(value, "\n", " ")
	value = strings.Join(strings.Fields(value), " ")
	value = redactSecretQueryValues(value)
	return value
}

func cleanToken(value string) string {
	value = cleanText(value)
	if value == "" {
		return "unknown"
	}
	return value
}

func redactSecretQueryValues(value string) string {
	for _, key := range []string{"token=", "secret=", "password=", "passwd=", "api_key=", "apikey=", "access_token=", "refresh_token=", "client_secret="} {
		searchFrom := 0
		for {
			lower := strings.ToLower(value)
			if searchFrom >= len(lower) {
				break
			}
			idx := strings.Index(lower[searchFrom:], key)
			if idx < 0 {
				break
			}
			idx += searchFrom
			start := idx + len(key)
			end := start
			for end < len(value) && value[end] != '&' && value[end] != ' ' {
				end++
			}
			value = value[:start] + "<redacted>" + value[end:]
			searchFrom = start + len("<redacted>")
		}
	}
	return value
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}
