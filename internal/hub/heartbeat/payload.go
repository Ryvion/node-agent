package heartbeat

import (
	"os"
	goruntime "runtime"
	"strings"
	"time"

	capshardware "github.com/Ryvion/ryvion-node/internal/capabilities/hardware"
	capability "github.com/Ryvion/ryvion-node/internal/capabilities/passport"
	"github.com/Ryvion/ryvion-node/internal/hw"
	netprofile "github.com/Ryvion/ryvion-node/internal/network/profile"
	"github.com/Ryvion/ryvion-node/internal/sandbox"
)

const (
	SchemaVersionV1 = "render.node-heartbeat.v1"
	EnvNodeCaps     = "RYV_NODE_CAPS"
	// EnvLegacyV7Caps is accepted only as a deprecated compatibility alias.
	EnvLegacyV7Caps = "RYV_NODE_V7_CAPS"
)

type NodeHeartbeatPayload struct {
	SchemaVersion             string                                `json:"schema_version"`
	NodeID                    string                                `json:"node_id"`
	CapabilityPassport        capability.CapabilityPassport         `json:"capability_passport"`
	NetworkProfile            *netprofile.NetworkProfile            `json:"network_profile,omitempty"`
	HardwareCapacity          capshardware.CapacityInventory        `json:"hardware_capacity"`
	CASSummary                *CASSummary                           `json:"cas_summary,omitempty"`
	SandboxPolicySummary      *SandboxPolicySummary                 `json:"sandbox_policy_summary,omitempty"`
	EvidenceCapabilitySummary *capability.EvidenceCapabilitySummary `json:"evidence_capability_summary,omitempty"`
	CreatedAtUnixMs           int64                                 `json:"created_at_unix_ms"`
}

type BuildNodeHeartbeatPayloadInput struct {
	SchemaVersion   string
	AgentVersion    string
	NodeID          string
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

	HardwareCapacity *capshardware.CapacityInventory

	RenderCapabilitySummary capability.RenderCapabilitySummary

	CASCapabilitySummary capability.CASCapabilitySummary
	CASSummary           *CASSummary

	SandboxCapabilitySummary capability.SandboxCapabilitySummary
	SandboxPolicy            *sandbox.SandboxPolicy
	SandboxPolicySummary     *SandboxPolicySummary

	EvidenceCapabilitySummary capability.EvidenceCapabilitySummary
	CreatedAtUnixMs           int64
}

type CASSummary struct {
	Enabled         bool   `json:"enabled"`
	ObjectCount     int    `json:"object_count"`
	TotalBytes      uint64 `json:"total_bytes"`
	MaxBytes        uint64 `json:"max_bytes,omitempty"`
	UpdatedAtUnixMs int64  `json:"updated_at_unix_ms,omitempty"`
}

type SandboxPolicySummary struct {
	RejectsUntrustedCustomRunners     bool     `json:"rejects_untrusted_custom_runners"`
	RunnerAllowlistEnabled            bool     `json:"runner_allowlist_enabled"`
	NetworkIsolationSupported         bool     `json:"network_isolation_supported"`
	FilesystemIsolationPlanned        bool     `json:"filesystem_isolation_planned"`
	NetworkAllowedRunnerKinds         []string `json:"network_allowed_runner_kinds,omitempty"`
	FilesystemWriteAllowedRunnerKinds []string `json:"filesystem_write_allowed_runner_kinds,omitempty"`
	AllowsTrustedCustomRunners        bool     `json:"allows_trusted_custom_runners"`
}

func NodeCapsEnabledFromEnv() bool {
	value := strings.TrimSpace(os.Getenv(EnvNodeCaps))
	if value == "" {
		value = strings.TrimSpace(os.Getenv(EnvLegacyV7Caps))
	}
	switch strings.ToLower(value) {
	case "0", "false", "no", "off", "disabled":
		return false
	default:
		return true
	}
}

func BuildNodeHeartbeatPayload(input BuildNodeHeartbeatPayloadInput) (NodeHeartbeatPayload, error) {
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
			return NodeHeartbeatPayload{}, err
		}
		networkSummary = networkCapabilitySummaryFromProfile(*networkProfile)
	}

	runtimeProfile := input.RuntimeProfile
	if len(runtimeProfile.SupportedRunnerKinds) == 0 {
		runtimeProfile.SupportedRunnerKinds = []string{"oci"}
	}

	hardwareCapacity := cloneHardwareCapacity(input.HardwareCapacity, firstNonEmpty(input.OS, goruntime.GOOS), firstNonEmpty(input.Arch, goruntime.GOARCH))

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
		RuntimeProfile:            runtimeProfile,
		NetworkCapabilitySummary:  networkSummary,
		RenderCapabilitySummary:   input.RenderCapabilitySummary,
		SandboxCapabilitySummary:  sandboxSummary,
		CASCapabilitySummary:      input.CASCapabilitySummary,
		EvidenceCapabilitySummary: input.EvidenceCapabilitySummary,
		CreatedAtUnixMs:           createdAtUnixMs,
	})
	if err != nil {
		return NodeHeartbeatPayload{}, err
	}

	payload := NodeHeartbeatPayload{
		SchemaVersion:        schemaVersion,
		NodeID:               firstNonEmpty(input.NodeID, input.NodePublicKey),
		CapabilityPassport:   passport,
		NetworkProfile:       networkProfile,
		HardwareCapacity:     hardwareCapacity,
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

func summarizeSandboxPolicy(policy sandbox.SandboxPolicy, capabilitySummary capability.SandboxCapabilitySummary) SandboxPolicySummary {
	return SandboxPolicySummary{
		RejectsUntrustedCustomRunners:     true,
		RunnerAllowlistEnabled:            capabilitySummary.RunnerAllowlistEnabled,
		NetworkIsolationSupported:         capabilitySummary.NetworkIsolationSupported,
		FilesystemIsolationPlanned:        capabilitySummary.FilesystemIsolationPlanned,
		NetworkAllowedRunnerKinds:         runnerKindStrings(policy.NetworkAllowedRunnerKinds),
		FilesystemWriteAllowedRunnerKinds: runnerKindStrings(policy.FilesystemWriteAllowedRunnerKinds),
		AllowsTrustedCustomRunners:        policy.AllowTrustedCustomRunners,
	}
}

func cloneNetworkProfile(profile *netprofile.NetworkProfile) *netprofile.NetworkProfile {
	if profile == nil {
		return nil
	}
	cloned := *profile
	return &cloned
}

func cloneHardwareCapacity(inventory *capshardware.CapacityInventory, osName, arch string) capshardware.CapacityInventory {
	if inventory == nil {
		return capshardware.NormalizeInventory(capshardware.CapacityInventory{
			OS:   osName,
			Arch: arch,
		})
	}
	return capshardware.NormalizeInventory(*inventory)
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
