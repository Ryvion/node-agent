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
	executorKindManagedOCI      = "managed_oci"
	executorKindAgentHosting    = "agent_hosting"

	assuranceClassVerifiedGateway    = "verified_gateway"
	assuranceClassSovereignExecution = "sovereign_execution"
)

type runtimeSnapshot struct {
	CLIInstalled bool
	Ready        bool
	GPUReady     bool
	Health       string
	Version      string
	Channel      string
	Provider     string
	Mode         string
	Source       string
	Artifact     string
	Binary       string
	Backend      string
	ManifestHash string
}

type runtimeManager struct {
	version  string
	contract runtimeContractMetadata
}

var probeManagedRuntimeStatus = runtimeexec.ProbeStatus

func newRuntimeManager(version string, contract runtimeContractMetadata) *runtimeManager {
	return &runtimeManager{version: strings.TrimSpace(version), contract: contract}
}

func (m *runtimeManager) Snapshot(gpuDetected bool) runtimeSnapshot {
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
	return runtimeSnapshot{
		CLIInstalled: backendCLI,
		Ready:        runtimeReady,
		GPUReady:     runtimeGPUReady,
		Health:       health,
		Version:      version,
		Channel:      sanitizeStatusValue(m.contract.Channel),
		Provider:     sanitizeStatusValue(m.contract.Provider),
		Mode:         sanitizeStatusValue(m.contract.Mode),
		Source:       sanitizeStatusValue(m.contract.Source),
		Artifact:     sanitizeStatusValue(m.contract.Artifact),
		Binary:       sanitizeStatusValue(m.contract.Binary),
		Backend:      sanitizeStatusValue(m.contract.Backend),
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

	return runtimeSnapshot{
		CLIInstalled: status.CLIInstalled,
		Ready:        status.Ready,
		GPUReady:     status.GPUReady,
		Health:       sanitizeStatusValue(status.Health),
		Version:      version,
		Channel:      sanitizeStatusValue(m.contract.Channel),
		Provider:     sanitizeStatusValue(m.contract.Provider),
		Mode:         sanitizeStatusValue(m.contract.Mode),
		Source:       sanitizeStatusValue(m.contract.Source),
		Artifact:     sanitizeStatusValue(m.contract.Artifact),
		Binary:       sanitizeStatusValue(status.BinaryPath),
		Backend:      sanitizeStatusValue(firstNonEmpty(status.BackendPath, m.contract.Backend)),
		ManifestHash: manifestHash,
	}, true
}

func (m *runtimeManager) StatusTokens(gpuDetected bool) []string {
	snap := m.Snapshot(gpuDetected)
	return []string{
		boolStatusToken("runtime-ready", snap.Ready),
		boolStatusToken("runtime-gpu-ready", snap.GPUReady),
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
		boolStatusToken("cap:managed_oci_cpu", snap.Ready),
		boolStatusToken("cap:managed_oci_gpu", snap.GPUReady),
		boolStatusToken("cap:agent_hosting", snap.Ready),
	}
}

func (m *runtimeManager) ReceiptMetadata(gpuDetected bool) map[string]any {
	snap := m.Snapshot(gpuDetected)
	return map[string]any{
		"runtime_version":       snap.Version,
		"runtime_manifest_hash": snap.ManifestHash,
		"runtime_health":        snap.Health,
		"runtime_channel":       snap.Channel,
		"runtime_provider":      snap.Provider,
		"runtime_mode":          snap.Mode,
		"runtime_source":        snap.Source,
		"runtime_artifact":      snap.Artifact,
		"runtime_binary":        snap.Binary,
		"runtime_backend":       snap.Backend,
	}
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
