package modellease

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
)

var ErrInvalidResidencyReport = errors.New("modellease: invalid residency report")

type ModelResidencySummary struct {
	ModelID              string          `json:"model_id"`
	QuantizationID       string          `json:"quantization_id,omitempty"`
	State                ModelLeaseState `json:"state"`
	VRAMReservedBytes    uint64          `json:"vram_reserved_bytes"`
	CanAcceptHotRequest  bool            `json:"can_accept_hot_request"`
	LeaseExpiresAtUnixMs int64           `json:"lease_expires_at_unix_ms"`
}

type InvalidModelLeaseReport struct {
	Index   int      `json:"index"`
	Reasons []string `json:"reasons"`
}

type ModelResidencyReport struct {
	NodeID        string                    `json:"node_id"`
	Summaries     []ModelResidencySummary   `json:"summaries"`
	InvalidLeases []InvalidModelLeaseReport `json:"invalid_leases,omitempty"`
}

func BuildResidencyReport(nodeID string, leases []ModelLease) (ModelResidencyReport, error) {
	nodeID = strings.TrimSpace(nodeID)
	if nodeID == "" {
		return ModelResidencyReport{}, fmt.Errorf("%w: node_id required", ErrInvalidResidencyReport)
	}

	nowUnixMs := time.Now().UnixMilli()
	report := ModelResidencyReport{
		NodeID:    nodeID,
		Summaries: make([]ModelResidencySummary, 0, len(leases)),
	}

	for index, lease := range leases {
		if reasons := invalidResidencyLeaseReasons(nodeID, lease); len(reasons) > 0 {
			report.InvalidLeases = append(report.InvalidLeases, InvalidModelLeaseReport{
				Index:   index,
				Reasons: reasons,
			})
			continue
		}
		if !isResidencyTelemetryState(lease.State) {
			continue
		}

		report.Summaries = append(report.Summaries, ModelResidencySummary{
			ModelID:              strings.TrimSpace(lease.ModelID),
			QuantizationID:       strings.TrimSpace(lease.QuantizationID),
			State:                lease.State,
			VRAMReservedBytes:    lease.VRAMReservedBytes,
			CanAcceptHotRequest:  CanAcceptHotRequest(lease, nowUnixMs),
			LeaseExpiresAtUnixMs: lease.LeaseExpiresAtUnixMs,
		})
	}

	sort.Slice(report.Summaries, func(i, j int) bool {
		left := report.Summaries[i]
		right := report.Summaries[j]
		if left.ModelID != right.ModelID {
			return left.ModelID < right.ModelID
		}
		if left.QuantizationID != right.QuantizationID {
			return left.QuantizationID < right.QuantizationID
		}
		if left.State != right.State {
			return left.State < right.State
		}
		if left.LeaseExpiresAtUnixMs != right.LeaseExpiresAtUnixMs {
			return left.LeaseExpiresAtUnixMs < right.LeaseExpiresAtUnixMs
		}
		if left.VRAMReservedBytes != right.VRAMReservedBytes {
			return left.VRAMReservedBytes < right.VRAMReservedBytes
		}
		return !left.CanAcceptHotRequest && right.CanAcceptHotRequest
	})

	return report, nil
}

func isResidencyTelemetryState(state ModelLeaseState) bool {
	switch state {
	case ModelLeaseStateLoading,
		ModelLeaseStateWarmup,
		ModelLeaseStateResident:
		return true
	default:
		return false
	}
}

func invalidResidencyLeaseReasons(reportNodeID string, lease ModelLease) []string {
	reasons := make([]string, 0, 4)
	if strings.TrimSpace(lease.LeaseID) == "" {
		reasons = append(reasons, "lease_id_required")
	}
	if strings.TrimSpace(lease.NodeID) == "" {
		reasons = append(reasons, "node_id_required")
	} else if strings.TrimSpace(lease.NodeID) != reportNodeID {
		reasons = append(reasons, "node_id_mismatch")
	}
	if strings.TrimSpace(lease.ModelID) == "" {
		reasons = append(reasons, "model_id_required")
	}
	reasons = appendUnsafeResidencyIdentifierReasons(reasons, "model_id", lease.ModelID)
	reasons = appendUnsafeResidencyIdentifierReasons(reasons, "quantization_id", lease.QuantizationID)
	if !isKnownModelLeaseState(lease.State) {
		reasons = append(reasons, "unknown_state")
	}
	if modelLeaseStateRequiresVRAM(lease.State) && lease.VRAMReservedBytes == 0 {
		reasons = append(reasons, "vram_reserved_bytes_required")
	}
	if lease.MinResidencyUntilUnixMs < 0 {
		reasons = append(reasons, "min_residency_until_unix_ms_negative")
	}
	if lease.LeaseExpiresAtUnixMs < 0 {
		reasons = append(reasons, "lease_expires_at_unix_ms_negative")
	}
	if lease.ReadinessRewardCentsPerMinute < 0 {
		reasons = append(reasons, "readiness_reward_cents_per_minute_negative")
	}
	if lease.UpdatedAtUnixMs < 0 {
		reasons = append(reasons, "updated_at_unix_ms_negative")
	}
	return reasons
}

func appendUnsafeResidencyIdentifierReasons(reasons []string, fieldName string, value string) []string {
	value = strings.TrimSpace(value)
	if value == "" {
		return reasons
	}
	if len(value) > 256 {
		reasons = append(reasons, fieldName+"_too_long")
	}
	if containsControlCharacter(value) {
		reasons = append(reasons, fieldName+"_control_character")
	}
	if looksLikeLocalPath(value) {
		reasons = append(reasons, fieldName+"_local_path")
	}
	if looksLikeSecret(value) {
		reasons = append(reasons, fieldName+"_secret_like")
	}
	return reasons
}

func containsControlCharacter(value string) bool {
	for _, r := range value {
		if r < 0x20 || r == 0x7f {
			return true
		}
	}
	return false
}

func looksLikeLocalPath(value string) bool {
	normalized := strings.ReplaceAll(value, "\\", "/")
	if strings.HasPrefix(normalized, "/") ||
		strings.HasPrefix(normalized, "~/") ||
		strings.HasPrefix(normalized, "./") ||
		strings.HasPrefix(normalized, "../") {
		return true
	}
	return len(value) >= 3 && isASCIIAlpha(value[0]) && value[1] == ':' && (value[2] == '/' || value[2] == '\\')
}

func looksLikeSecret(value string) bool {
	lower := strings.ToLower(value)
	return strings.Contains(lower, "api_key=") ||
		strings.Contains(lower, "authorization:") ||
		strings.Contains(lower, "bearer ") ||
		strings.Contains(lower, "secret=") ||
		strings.Contains(lower, "token=") ||
		strings.HasPrefix(lower, "sk-")
}

func isASCIIAlpha(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}
