package heartbeat

import (
	"context"
	"os"
	goruntime "runtime"
	"sort"
	"strings"
	"time"

	"github.com/Ryvion/node-agent/internal/hw"
	"github.com/Ryvion/node-agent/internal/v7/backendprobe"
	"github.com/Ryvion/node-agent/internal/v7/capability"
	"github.com/Ryvion/node-agent/internal/v7/kvprobe"
	"github.com/Ryvion/node-agent/internal/v7/modellease"
	"github.com/Ryvion/node-agent/internal/v7/netprofile"
	"github.com/Ryvion/node-agent/internal/v7/runtimeinventory"
	"github.com/Ryvion/node-agent/internal/v7/sandbox"
	"github.com/Ryvion/node-agent/internal/v7/tensoraccess"
)

const (
	SchemaVersionV1 = "v7.heartbeat-payload.v1"
	EnvV7Caps       = "RYV_NODE_V7_CAPS"
)

type V7HeartbeatPayload struct {
	SchemaVersion             string                                `json:"schema_version"`
	CapabilityPassport        capability.CapabilityPassport         `json:"capability_passport"`
	NetworkProfile            *netprofile.NetworkProfile            `json:"network_profile,omitempty"`
	ModelLeaseSummary         *ModelLeaseSummary                    `json:"model_lease_summary,omitempty"`
	KVCapability              *kvprobe.Capability                   `json:"kv_capability,omitempty"`
	TensorAccess              tensoraccess.TensorAccessCapability   `json:"tensor_access"`
	RuntimeInventory          runtimeinventory.Inventory            `json:"runtime_inventory"`
	BackendProbes             backendprobe.Probes                   `json:"backend_probes"`
	CASSummary                *CASSummary                           `json:"cas_summary,omitempty"`
	SandboxPolicySummary      *SandboxPolicySummary                 `json:"sandbox_policy_summary,omitempty"`
	EvidenceCapabilitySummary *capability.EvidenceCapabilitySummary `json:"evidence_capability_summary,omitempty"`
	CreatedAtUnixMs           int64                                 `json:"created_at_unix_ms"`
}

type BuildV7HeartbeatPayloadInput struct {
	SchemaVersion   string
	AgentVersion    string
	NodePublicKey   string
	OS              string
	Arch            string
	DeviceType      string
	DeclaredCountry string

	HardwareCapabilities hw.CapSet
	HardwareProfile      capability.HardwareProfile
	RuntimeProfile       capability.RuntimeProfile

	NetworkCapabilitySummary capability.NetworkCapabilitySummary
	NetworkProfile           *netprofile.NetworkProfile

	ModelCapabilitySummary capability.ModelCapabilitySummary
	ModelLeases            []modellease.ModelLease
	ModelLeaseSummary      *ModelLeaseSummary
	KVCapability           *kvprobe.Capability
	TensorAccess           *tensoraccess.TensorAccessCapability
	RuntimeInventory       *runtimeinventory.Inventory
	BackendProbes          *backendprobe.Probes

	CASCapabilitySummary capability.CASCapabilitySummary
	CASSummary           *CASSummary

	SandboxCapabilitySummary capability.SandboxCapabilitySummary
	SandboxPolicy            *sandbox.SandboxPolicy
	SandboxPolicySummary     *SandboxPolicySummary

	EvidenceCapabilitySummary capability.EvidenceCapabilitySummary
	CreatedAtUnixMs           int64
}

type ModelLeaseSummary struct {
	SupportsModelLease bool     `json:"supports_model_lease"`
	TotalLeases        int      `json:"total_leases"`
	ResidentLeases     int      `json:"resident_leases"`
	LoadingLeases      int      `json:"loading_leases"`
	DrainingLeases     int      `json:"draining_leases"`
	FailedLeases       int      `json:"failed_leases"`
	ResidentModelIDs   []string `json:"resident_model_ids,omitempty"`
	VRAMReservedBytes  uint64   `json:"vram_reserved_bytes"`
	UpdatedAtUnixMs    int64    `json:"updated_at_unix_ms,omitempty"`
}

type CASSummary struct {
	Enabled         bool   `json:"enabled"`
	ObjectCount     int    `json:"object_count"`
	TotalBytes      uint64 `json:"total_bytes"`
	MaxBytes        uint64 `json:"max_bytes,omitempty"`
	UpdatedAtUnixMs int64  `json:"updated_at_unix_ms,omitempty"`
}

type SandboxPolicySummary struct {
	RejectsUnsafePickle                   bool     `json:"rejects_unsafe_pickle"`
	RunnerAllowlistEnabled                bool     `json:"runner_allowlist_enabled"`
	NetworkIsolationSupported             bool     `json:"network_isolation_supported"`
	FilesystemIsolationPlanned            bool     `json:"filesystem_isolation_planned"`
	PythonSourceDecision                  string   `json:"python_source_decision,omitempty"`
	NetworkAllowedRunnerKinds             []string `json:"network_allowed_runner_kinds,omitempty"`
	FilesystemWriteAllowedRunnerKinds     []string `json:"filesystem_write_allowed_runner_kinds,omitempty"`
	AllowsUnknownTrustedAllowlisted       bool     `json:"allows_unknown_trusted_allowlisted"`
	AllowsPyTorchPickleTrustedAllowlisted bool     `json:"allows_pytorch_pickle_trusted_allowlisted"`
}

func V7HeartbeatEnabledFromEnv() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(EnvV7Caps))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func BuildV7HeartbeatPayload(input BuildV7HeartbeatPayloadInput) (V7HeartbeatPayload, error) {
	createdAtUnixMs := input.CreatedAtUnixMs
	if createdAtUnixMs == 0 {
		createdAtUnixMs = time.Now().UnixMilli()
	}

	schemaVersion := strings.TrimSpace(input.SchemaVersion)
	if schemaVersion == "" {
		schemaVersion = SchemaVersionV1
	}

	networkSummary := input.NetworkCapabilitySummary
	networkProfile := cloneNetworkProfile(input.NetworkProfile)
	if networkProfile != nil {
		if err := netprofile.ValidateNetworkProfile(*networkProfile); err != nil {
			return V7HeartbeatPayload{}, err
		}
		networkSummary = networkCapabilitySummaryFromProfile(*networkProfile)
	}

	modelSummary := input.ModelCapabilitySummary
	modelLeaseSummary := cloneModelLeaseSummary(input.ModelLeaseSummary)
	if modelLeaseSummary == nil && len(input.ModelLeases) > 0 {
		summary := summarizeModelLeases(input.ModelLeases, modelSummary.SupportsModelLease)
		modelLeaseSummary = &summary
	}
	if modelLeaseSummary != nil {
		modelSummary.ResidentModelIDs = mergeStrings(modelSummary.ResidentModelIDs, modelLeaseSummary.ResidentModelIDs)
		if modelSummary.MaxResidentModelBytes == 0 {
			modelSummary.MaxResidentModelBytes = modelLeaseSummary.VRAMReservedBytes
		}
	}
	kvCapability := cloneKVCapability(input.KVCapability)
	tensorAccess := cloneTensorAccessCapability(input.TensorAccess)
	runtimeInventory := cloneRuntimeInventory(input.RuntimeInventory)
	backendProbes := cloneBackendProbes(input.BackendProbes)

	casSummary := cloneCASSummary(input.CASSummary)
	if casSummary == nil && input.CASCapabilitySummary.Enabled {
		casSummary = &CASSummary{
			Enabled:     true,
			ObjectCount: 0,
			TotalBytes:  0,
			MaxBytes:    input.CASCapabilitySummary.MaxBytes,
		}
	}

	sandboxSummary := input.SandboxCapabilitySummary
	sandboxPolicySummary := cloneSandboxPolicySummary(input.SandboxPolicySummary)
	if sandboxPolicySummary == nil && input.SandboxPolicy != nil {
		summary := summarizeSandboxPolicy(*input.SandboxPolicy, sandboxSummary)
		sandboxPolicySummary = &summary
	}

	passport, err := capability.BuildCapabilityPassport(capability.BuildPassportInput{
		SchemaVersion:             capability.SchemaVersionV1,
		AgentVersion:              input.AgentVersion,
		NodePublicKey:             input.NodePublicKey,
		OS:                        firstNonEmpty(input.OS, goruntime.GOOS),
		Arch:                      firstNonEmpty(input.Arch, goruntime.GOARCH),
		DeviceType:                input.DeviceType,
		DeclaredCountry:           input.DeclaredCountry,
		HardwareCapabilities:      input.HardwareCapabilities,
		HardwareProfile:           input.HardwareProfile,
		RuntimeProfile:            input.RuntimeProfile,
		NetworkCapabilitySummary:  networkSummary,
		ModelCapabilitySummary:    modelSummary,
		SandboxCapabilitySummary:  sandboxSummary,
		CASCapabilitySummary:      input.CASCapabilitySummary,
		EvidenceCapabilitySummary: input.EvidenceCapabilitySummary,
		CreatedAtUnixMs:           createdAtUnixMs,
	})
	if err != nil {
		return V7HeartbeatPayload{}, err
	}

	payload := V7HeartbeatPayload{
		SchemaVersion:        schemaVersion,
		CapabilityPassport:   passport,
		NetworkProfile:       networkProfile,
		ModelLeaseSummary:    modelLeaseSummary,
		KVCapability:         kvCapability,
		TensorAccess:         tensorAccess,
		RuntimeInventory:     runtimeInventory,
		BackendProbes:        backendProbes,
		CASSummary:           casSummary,
		SandboxPolicySummary: sandboxPolicySummary,
		CreatedAtUnixMs:      createdAtUnixMs,
	}
	if hasEvidenceCapability(input.EvidenceCapabilitySummary) {
		summary := input.EvidenceCapabilitySummary
		payload.EvidenceCapabilitySummary = &summary
	}
	return payload, nil
}

func networkCapabilitySummaryFromProfile(profile netprofile.NetworkProfile) capability.NetworkCapabilitySummary {
	return capability.NetworkCapabilitySummary{
		UploadMbpsP50:   profile.UploadMbpsP50,
		UploadMbpsP95:   profile.UploadMbpsP95,
		DownloadMbpsP50: profile.DownloadMbpsP50,
		DownloadMbpsP95: profile.DownloadMbpsP95,
		RTTMsP50:        profile.RTTMsP50,
		RTTMsP95:        profile.RTTMsP95,
		JitterMsP95:     profile.JitterMsP95,
		LossRateP95:     profile.LossRateP95,
	}
}

func summarizeModelLeases(leases []modellease.ModelLease, supportsModelLease bool) ModelLeaseSummary {
	summary := ModelLeaseSummary{
		SupportsModelLease: supportsModelLease,
		TotalLeases:        len(leases),
	}
	residentIDs := map[string]struct{}{}
	for _, lease := range leases {
		switch lease.State {
		case modellease.ModelLeaseStateResident:
			summary.ResidentLeases++
			if modelID := strings.TrimSpace(lease.ModelID); modelID != "" {
				residentIDs[modelID] = struct{}{}
			}
		case modellease.ModelLeaseStateLoading, modellease.ModelLeaseStateWarmup:
			summary.LoadingLeases++
		case modellease.ModelLeaseStateDraining, modellease.ModelLeaseStateEvicting:
			summary.DrainingLeases++
		case modellease.ModelLeaseStateFailed:
			summary.FailedLeases++
		}
		summary.VRAMReservedBytes += lease.VRAMReservedBytes
		if lease.UpdatedAtUnixMs > summary.UpdatedAtUnixMs {
			summary.UpdatedAtUnixMs = lease.UpdatedAtUnixMs
		}
	}
	summary.ResidentModelIDs = sortedKeys(residentIDs)
	return summary
}

func summarizeSandboxPolicy(policy sandbox.SandboxPolicy, capabilitySummary capability.SandboxCapabilitySummary) SandboxPolicySummary {
	return SandboxPolicySummary{
		RejectsUnsafePickle:                   true,
		RunnerAllowlistEnabled:                capabilitySummary.RunnerAllowlistEnabled,
		NetworkIsolationSupported:             capabilitySummary.NetworkIsolationSupported,
		FilesystemIsolationPlanned:            capabilitySummary.FilesystemIsolationPlanned,
		PythonSourceDecision:                  string(policy.PythonSourceDecision),
		NetworkAllowedRunnerKinds:             runnerKindStrings(policy.NetworkAllowedRunnerKinds),
		FilesystemWriteAllowedRunnerKinds:     runnerKindStrings(policy.FilesystemWriteAllowedRunnerKinds),
		AllowsUnknownTrustedAllowlisted:       policy.AllowUnknownTrustedAllowlisted,
		AllowsPyTorchPickleTrustedAllowlisted: policy.AllowPyTorchPickleTrustedAllowlisted,
	}
}

func cloneNetworkProfile(profile *netprofile.NetworkProfile) *netprofile.NetworkProfile {
	if profile == nil {
		return nil
	}
	cloned := *profile
	return &cloned
}

func cloneModelLeaseSummary(summary *ModelLeaseSummary) *ModelLeaseSummary {
	if summary == nil {
		return nil
	}
	cloned := *summary
	cloned.ResidentModelIDs = cloneStrings(summary.ResidentModelIDs)
	return &cloned
}

func cloneKVCapability(capability *kvprobe.Capability) *kvprobe.Capability {
	if capability == nil {
		return nil
	}
	cloned := kvprobe.NormalizeCapability(*capability)
	return &cloned
}

func cloneTensorAccessCapability(capability *tensoraccess.TensorAccessCapability) tensoraccess.TensorAccessCapability {
	if capability == nil {
		return tensoraccess.NewNoopProvider(tensoraccess.NoopProviderConfig{}).Capability(context.Background())
	}
	return tensoraccess.NormalizeCapability(*capability)
}

func cloneRuntimeInventory(inventory *runtimeinventory.Inventory) runtimeinventory.Inventory {
	if inventory == nil {
		return runtimeinventory.NormalizeInventory(runtimeinventory.Inventory{})
	}
	return runtimeinventory.NormalizeInventory(*inventory)
}

func cloneBackendProbes(probes *backendprobe.Probes) backendprobe.Probes {
	if probes == nil {
		return backendprobe.NormalizeProbes(backendprobe.Probes{})
	}
	return backendprobe.NormalizeProbes(*probes)
}

func cloneCASSummary(summary *CASSummary) *CASSummary {
	if summary == nil {
		return nil
	}
	cloned := *summary
	return &cloned
}

func cloneSandboxPolicySummary(summary *SandboxPolicySummary) *SandboxPolicySummary {
	if summary == nil {
		return nil
	}
	cloned := *summary
	cloned.NetworkAllowedRunnerKinds = cloneStrings(summary.NetworkAllowedRunnerKinds)
	cloned.FilesystemWriteAllowedRunnerKinds = cloneStrings(summary.FilesystemWriteAllowedRunnerKinds)
	return &cloned
}

func runnerKindStrings(kinds []sandbox.RunnerKind) []string {
	if len(kinds) == 0 {
		return nil
	}
	out := make([]string, 0, len(kinds))
	for _, kind := range kinds {
		value := strings.TrimSpace(string(kind))
		if value != "" {
			out = append(out, value)
		}
	}
	return out
}

func mergeStrings(left, right []string) []string {
	values := map[string]struct{}{}
	for _, value := range append(cloneStrings(left), right...) {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		values[value] = struct{}{}
	}
	return sortedKeys(values)
}

func sortedKeys(values map[string]struct{}) []string {
	if len(values) == 0 {
		return nil
	}
	out := make([]string, 0, len(values))
	for value := range values {
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func cloneStrings(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	out := make([]string, len(values))
	copy(out, values)
	return out
}

func hasEvidenceCapability(summary capability.EvidenceCapabilitySummary) bool {
	return summary.SupportsArtifactManifest ||
		summary.SupportsRYV3EvidencePayload ||
		summary.SupportsRuntimeHash
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			return value
		}
	}
	return ""
}
