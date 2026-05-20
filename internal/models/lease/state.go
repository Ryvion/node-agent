package modellease

import (
	"errors"
	"fmt"
	"strings"
)

type ModelLeaseState string

const (
	ModelLeaseStateUnpinned ModelLeaseState = "unpinned"
	ModelLeaseStateBidding  ModelLeaseState = "bidding"
	ModelLeaseStateLoading  ModelLeaseState = "loading"
	ModelLeaseStateWarmup   ModelLeaseState = "warmup"
	ModelLeaseStateResident ModelLeaseState = "resident"
	ModelLeaseStateDraining ModelLeaseState = "draining"
	ModelLeaseStateEvicting ModelLeaseState = "evicting"
	ModelLeaseStateFailed   ModelLeaseState = "failed"
)

var (
	ErrInvalidModelLease           = errors.New("modellease: invalid model lease")
	ErrInvalidModelLeaseEvent      = errors.New("modellease: invalid model lease event")
	ErrInvalidModelLeaseTransition = errors.New("modellease: invalid model lease transition")
)

type ModelLease struct {
	LeaseID                       string          `json:"lease_id"`
	NodeID                        string          `json:"node_id"`
	ModelID                       string          `json:"model_id"`
	QuantizationID                string          `json:"quantization_id,omitempty"`
	State                         ModelLeaseState `json:"state"`
	VRAMReservedBytes             uint64          `json:"vram_reserved_bytes"`
	MinResidencyUntilUnixMs       int64           `json:"min_residency_until_unix_ms"`
	LeaseExpiresAtUnixMs          int64           `json:"lease_expires_at_unix_ms"`
	ReadinessRewardCentsPerMinute int64           `json:"readiness_reward_cents_per_minute"`
	UpdatedAtUnixMs               int64           `json:"updated_at_unix_ms"`
}

func CanAcceptHotRequest(lease ModelLease, nowUnixMs int64) bool {
	if nowUnixMs < 0 {
		return false
	}
	if err := ValidateModelLease(lease); err != nil {
		return false
	}
	if lease.State != ModelLeaseStateResident {
		return false
	}
	return lease.LeaseExpiresAtUnixMs > nowUnixMs
}

func ValidateModelLease(lease ModelLease) error {
	var errs []error
	if strings.TrimSpace(lease.LeaseID) == "" {
		errs = append(errs, fmt.Errorf("%w: lease_id required", ErrInvalidModelLease))
	}
	if strings.TrimSpace(lease.NodeID) == "" {
		errs = append(errs, fmt.Errorf("%w: node_id required", ErrInvalidModelLease))
	}
	if strings.TrimSpace(lease.ModelID) == "" {
		errs = append(errs, fmt.Errorf("%w: model_id required", ErrInvalidModelLease))
	}
	if !isKnownModelLeaseState(lease.State) {
		errs = append(errs, fmt.Errorf("%w: unknown state %q", ErrInvalidModelLease, lease.State))
	}
	if modelLeaseStateRequiresVRAM(lease.State) && lease.VRAMReservedBytes == 0 {
		errs = append(errs, fmt.Errorf("%w: vram_reserved_bytes required for state %q", ErrInvalidModelLease, lease.State))
	}
	if lease.MinResidencyUntilUnixMs < 0 {
		errs = append(errs, fmt.Errorf("%w: min_residency_until_unix_ms must be non-negative", ErrInvalidModelLease))
	}
	if lease.LeaseExpiresAtUnixMs < 0 {
		errs = append(errs, fmt.Errorf("%w: lease_expires_at_unix_ms must be non-negative", ErrInvalidModelLease))
	}
	if lease.ReadinessRewardCentsPerMinute < 0 {
		errs = append(errs, fmt.Errorf("%w: readiness_reward_cents_per_minute must be non-negative", ErrInvalidModelLease))
	}
	if lease.UpdatedAtUnixMs < 0 {
		errs = append(errs, fmt.Errorf("%w: updated_at_unix_ms must be non-negative", ErrInvalidModelLease))
	}
	return errors.Join(errs...)
}

func modelLeaseStateRequiresVRAM(state ModelLeaseState) bool {
	switch state {
	case ModelLeaseStateLoading,
		ModelLeaseStateWarmup,
		ModelLeaseStateResident:
		return true
	default:
		return false
	}
}

func isKnownModelLeaseState(state ModelLeaseState) bool {
	switch state {
	case ModelLeaseStateUnpinned,
		ModelLeaseStateBidding,
		ModelLeaseStateLoading,
		ModelLeaseStateWarmup,
		ModelLeaseStateResident,
		ModelLeaseStateDraining,
		ModelLeaseStateEvicting,
		ModelLeaseStateFailed:
		return true
	default:
		return false
	}
}

func clearReleasedModelLeaseFields(lease *ModelLease) {
	lease.VRAMReservedBytes = 0
	lease.MinResidencyUntilUnixMs = 0
	lease.LeaseExpiresAtUnixMs = 0
	lease.ReadinessRewardCentsPerMinute = 0
}
