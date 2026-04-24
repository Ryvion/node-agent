package main

import (
	"context"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/Ryvion/node-agent/internal/runtimeexec"
)

const (
	executorKindNativeStreaming = "native_streaming"
	executorKindNativeReport    = "native_report"
	executorKindManagedOCI      = "managed_oci"
	executorKindAgentHosting    = "agent_hosting"

	assuranceClassVerifiedGateway    = "verified_gateway"
	assuranceClassSovereignExecution = "sovereign_execution"
)

type runtimeSnapshot struct {
	CLIInstalled bool
	Ready        bool
	GPUReady     bool
	Warming      bool
	Health       string
	Version      string
	Channel      string
	Provider     string
	Mode         string
	Source       string
	Artifact     string
	Binary       string
	Backend      string
	Engine       string
	EngineKind   string
	ManifestHash string
}

type runtimeManager struct {
	version  string
	contract runtimeContractMetadata
}

var probeManagedRuntimeStatus = runtimeexec.ProbeStatus

// ociLaneDisabled reports whether the operator has opted out of the managed
// OCI (container) lane entirely via RYV_DISABLE_OCI=1. When disabled, the
// runtime manager never probes Podman/Docker and reports the node as
// native-only. Phase 1 of the Docker-free native path.
func ociLaneDisabled() bool {
	raw := strings.ToLower(strings.TrimSpace(os.Getenv("RYV_DISABLE_OCI")))
	switch raw {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

func newRuntimeManager(version string, contract runtimeContractMetadata) *runtimeManager {
	return &runtimeManager{version: strings.TrimSpace(version), contract: contract}
}

func (m *runtimeManager) Snapshot(gpuDetected bool) runtimeSnapshot {
	if ociLaneDisabled() {
		version := sanitizeStatusValue(m.contract.Version)
		if version == "" {
			version = sanitizeStatusValue(m.version)
		}
		if version == "" {
			version = "dev"
		}
		return runtimeSnapshot{
			CLIInstalled: false,
			Ready:        false,
			GPUReady:     false,
			Warming:      false,
			Health:       "disabled",
			Version:      version,
			Channel:      sanitizeStatusValue(m.contract.Channel),
			Provider:     sanitizeStatusValue(m.contract.Provider),
			Mode:         "native_only",
			Source:       "operator_opt_out",
			ManifestHash: sanitizeStatusValue(m.contract.ManifestHash),
		}
	}
	if snap, ok := m.snapshotFromManagedRuntimeWrapper(gpuDetected); ok {
		return snap
	}

	backendCLI, runtimeReady, runtimeGPUReady := detectManagedOCIBackendWithProbes(gpuDetected, resolveOCIBackendCLI, testOCIBackendReady, testManagedOCIGPU)
	health := "missing"
	switch {
	case runtimeReady:
		health = "ready"
	case backendCLI:
		health = "degraded"
	}
	version := sanitizeStatusValue(m.contract.Version)
	if version == "" {
		version = sanitizeStatusValue(m.version)
	}
	if version == "" {
		version = "dev"
	}
	manifestHash := sanitizeStatusValue(m.contract.ManifestHash)
	if manifestHash == "" {
		manifestHash = computeRuntimeManifestHash(m.contract)
	}
	engine := sanitizeStatusValue(firstNonEmpty(m.contract.Engine, runtimeexec.ResolveEnginePath(runtime.GOOS, os.Getenv)))
	engineKind := sanitizeStatusValue(firstNonEmpty(m.contract.EngineKind, runtimeexec.EngineKind(engine)))
	warming := runtimeWarmingHeuristic(runtime.GOOS, backendCLI, runtimeReady, health, m.contract.Backend, engineKind)
	if warming {
		health = "warming"
	}
	return runtimeSnapshot{
		CLIInstalled: backendCLI,
		Ready:        runtimeReady,
		GPUReady:     runtimeGPUReady,
		Warming:      warming,
		Health:       health,
		Version:      version,
		Channel:      sanitizeStatusValue(m.contract.Channel),
		Provider:     sanitizeStatusValue(m.contract.Provider),
		Mode:         sanitizeStatusValue(m.contract.Mode),
		Source:       sanitizeStatusValue(m.contract.Source),
		Artifact:     sanitizeStatusValue(m.contract.Artifact),
		Binary:       sanitizeStatusValue(m.contract.Binary),
		Backend:      sanitizeStatusValue(m.contract.Backend),
		Engine:       engine,
		EngineKind:   engineKind,
		ManifestHash: manifestHash,
	}
}

func (m *runtimeManager) snapshotFromManagedRuntimeWrapper(gpuDetected bool) (runtimeSnapshot, bool) {
	binary := strings.TrimSpace(m.contract.Binary)
	if binary == "" {
		binary = runtimeexec.ResolveBinaryPath(runtime.GOOS, os.Getenv)
	}
	if binary == "" {
		return runtimeSnapshot{}, false
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	status, ok := probeManagedRuntimeStatus(ctx, runtime.GOOS, os.Getenv, binary)
	if !ok {
		return runtimeSnapshot{}, false
	}

	version := sanitizeStatusValue(m.contract.Version)
	if version == "" {
		version = sanitizeStatusValue(m.version)
	}
	if version == "" {
		version = "dev"
	}
	manifestHash := sanitizeStatusValue(m.contract.ManifestHash)
	if manifestHash == "" {
		manifestHash = computeRuntimeManifestHash(m.contract)
	}
	health := sanitizeStatusValue(status.Health)
	engineKind := sanitizeStatusValue(firstNonEmpty(status.EngineKind, runtimeexec.EngineKind(status.EnginePath), m.contract.EngineKind))
	backend := sanitizeStatusValue(firstNonEmpty(status.BackendPath, m.contract.Backend))
	warming := runtimeWarmingHeuristic(runtime.GOOS, status.CLIInstalled, status.Ready, health, backend, engineKind)
	if warming {
		health = "warming"
	}

	return runtimeSnapshot{
		CLIInstalled: status.CLIInstalled,
		Ready:        status.Ready,
		GPUReady:     status.GPUReady,
		Warming:      warming,
		Health:       health,
		Version:      version,
		Channel:      sanitizeStatusValue(m.contract.Channel),
		Provider:     sanitizeStatusValue(m.contract.Provider),
		Mode:         sanitizeStatusValue(m.contract.Mode),
		Source:       sanitizeStatusValue(m.contract.Source),
		Artifact:     sanitizeStatusValue(m.contract.Artifact),
		Binary:       sanitizeStatusValue(status.BinaryPath),
		Backend:      backend,
		Engine:       sanitizeStatusValue(firstNonEmpty(status.EnginePath, m.contract.Engine)),
		EngineKind:   engineKind,
		ManifestHash: manifestHash,
	}, true
}

func (m *runtimeManager) StatusTokens(gpuDetected bool) []string {
	snap := m.Snapshot(gpuDetected)
	tokens := []string{
		boolStatusToken("runtime-ready", snap.Ready),
		boolStatusToken("runtime-gpu-ready", snap.GPUReady),
		boolStatusToken("runtime-warming", snap.Warming),
		"runtime-health:" + sanitizeStatusValue(snap.Health),
		"runtime-version:" + sanitizeStatusValue(snap.Version),
		"runtime-manifest-hash:" + sanitizeStatusValue(snap.ManifestHash),
		"runtime-channel:" + sanitizeStatusValue(snap.Channel),
		"runtime-provider:" + sanitizeStatusValue(snap.Provider),
		"runtime-mode:" + sanitizeStatusValue(snap.Mode),
		"runtime-source:" + sanitizeStatusValue(snap.Source),
		"runtime-artifact:" + sanitizeStatusValue(snap.Artifact),
		"runtime-binary:" + sanitizeStatusValue(snap.Binary),
		"runtime-backend:" + sanitizeStatusValue(snap.Backend),
		"runtime-engine:" + sanitizeStatusValue(snap.Engine),
		"runtime-engine-kind:" + sanitizeStatusValue(snap.EngineKind),
		boolStatusToken("cap:managed_oci_cpu", snap.Ready),
		boolStatusToken("cap:managed_oci_gpu", snap.GPUReady),
		boolStatusToken("cap:agent_hosting", snap.Ready),
	}
	if ociLaneDisabled() {
		tokens = append(tokens, "oci-lane:disabled")
	}
	return tokens
}

func (m *runtimeManager) ReceiptMetadata(gpuDetected bool) map[string]any {
	snap := m.Snapshot(gpuDetected)
	return map[string]any{
		"runtime_version":       snap.Version,
		"runtime_manifest_hash": snap.ManifestHash,
		"runtime_health":        snap.Health,
		"runtime_warming":       snap.Warming,
		"runtime_channel":       snap.Channel,
		"runtime_provider":      snap.Provider,
		"runtime_mode":          snap.Mode,
		"runtime_source":        snap.Source,
		"runtime_artifact":      snap.Artifact,
		"runtime_binary":        snap.Binary,
		"runtime_backend":       snap.Backend,
		"runtime_engine":        snap.Engine,
		"runtime_engine_kind":   snap.EngineKind,
	}
}

func runtimeWarmingHeuristic(goos string, cliInstalled bool, ready bool, health string, backend string, engineKind string) bool {
	if ready {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(health)) {
	case "warming":
		return true
	case "degraded", "missing", "ready":
		return false
	}
	if goos != "windows" || !cliInstalled {
		return false
	}
	if strings.TrimSpace(backend) == "" {
		return false
	}
	return strings.EqualFold(strings.TrimSpace(engineKind), "podman")
}

func boolStatusToken(prefix string, ready bool) string {
	if ready {
		return prefix + ":1"
	}
	return prefix + ":0"
}

func sanitizeStatusValue(v string) string {
	v = strings.TrimSpace(v)
	if v == "" {
		return ""
	}
	replacer := strings.NewReplacer(",", "_", " ", "_")
	return replacer.Replace(v)
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}
