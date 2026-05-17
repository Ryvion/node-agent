package modellease

import (
	"errors"
	"strings"
	"testing"
)

const modelLeaseTestNowUnixMs int64 = 1_800_000_000_000

func TestModelLeaseNormalTransitionPathToResident(t *testing.T) {
	lease := validModelLease(ModelLeaseStateUnpinned)

	lease = applyModelLeaseEventOrFatal(t, lease, ModelLeaseEventBid)
	if lease.State != ModelLeaseStateBidding {
		t.Fatalf("state after bid = %q, want %q", lease.State, ModelLeaseStateBidding)
	}

	lease.VRAMReservedBytes = 80_000_000_000
	lease = applyModelLeaseEventOrFatal(t, lease, ModelLeaseEventStartLoading)
	if lease.State != ModelLeaseStateLoading {
		t.Fatalf("state after start_loading = %q, want %q", lease.State, ModelLeaseStateLoading)
	}

	lease = applyModelLeaseEventOrFatal(t, lease, ModelLeaseEventWarmupComplete)
	if lease.State != ModelLeaseStateWarmup {
		t.Fatalf("state after warmup_complete = %q, want %q", lease.State, ModelLeaseStateWarmup)
	}

	lease = applyModelLeaseEventOrFatal(t, lease, ModelLeaseEventBecomeResident)
	if lease.State != ModelLeaseStateResident {
		t.Fatalf("state after become_resident = %q, want %q", lease.State, ModelLeaseStateResident)
	}
}

func TestModelLeaseInvalidDirectHotSwitchRejected(t *testing.T) {
	lease := validModelLease(ModelLeaseStateResident)

	_, err := ApplyModelLeaseEvent(lease, ModelLeaseEventBecomeResident)
	if !errors.Is(err, ErrInvalidModelLeaseTransition) {
		t.Fatalf("ApplyModelLeaseEvent() error = %v, want ErrInvalidModelLeaseTransition", err)
	}
}

func TestModelLeaseResidentLeaseAcceptsHotRequest(t *testing.T) {
	lease := validModelLease(ModelLeaseStateResident)

	if !CanAcceptHotRequest(lease, modelLeaseTestNowUnixMs) {
		t.Fatalf("CanAcceptHotRequest() = false, want true")
	}
}

func TestModelLeaseExpiredResidentLeaseRejectsHotRequest(t *testing.T) {
	lease := validModelLease(ModelLeaseStateResident)
	lease.LeaseExpiresAtUnixMs = modelLeaseTestNowUnixMs

	if CanAcceptHotRequest(lease, modelLeaseTestNowUnixMs) {
		t.Fatalf("CanAcceptHotRequest() = true, want false")
	}
}

func TestModelLeaseDrainingRejectsHotRequest(t *testing.T) {
	if CanAcceptHotRequest(validModelLease(ModelLeaseStateDraining), modelLeaseTestNowUnixMs) {
		t.Fatalf("CanAcceptHotRequest() = true, want false")
	}
}

func TestModelLeaseFailedResetsToUnpinned(t *testing.T) {
	lease := validModelLease(ModelLeaseStateLoading)

	lease = applyModelLeaseEventOrFatal(t, lease, ModelLeaseEventFail)
	if lease.State != ModelLeaseStateFailed {
		t.Fatalf("state after fail = %q, want %q", lease.State, ModelLeaseStateFailed)
	}

	lease = applyModelLeaseEventOrFatal(t, lease, ModelLeaseEventReset)
	if lease.State != ModelLeaseStateUnpinned {
		t.Fatalf("state after reset = %q, want %q", lease.State, ModelLeaseStateUnpinned)
	}
	if lease.VRAMReservedBytes != 0 {
		t.Fatalf("vram after reset = %d, want 0", lease.VRAMReservedBytes)
	}
}

func TestValidateModelLeaseMissingFieldsRejected(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(*ModelLease)
		want string
	}{
		{name: "lease", edit: func(lease *ModelLease) { lease.LeaseID = "" }, want: "lease_id required"},
		{name: "node", edit: func(lease *ModelLease) { lease.NodeID = "" }, want: "node_id required"},
		{name: "model", edit: func(lease *ModelLease) { lease.ModelID = "" }, want: "model_id required"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			lease := validModelLease(ModelLeaseStateResident)
			tc.edit(&lease)

			err := ValidateModelLease(lease)
			if !errors.Is(err, ErrInvalidModelLease) || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("ValidateModelLease() error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestValidateModelLeaseVRAMRequiredForActiveResidentStates(t *testing.T) {
	for _, state := range []ModelLeaseState{
		ModelLeaseStateLoading,
		ModelLeaseStateWarmup,
		ModelLeaseStateResident,
	} {
		t.Run(string(state), func(t *testing.T) {
			lease := validModelLease(state)
			lease.VRAMReservedBytes = 0

			err := ValidateModelLease(lease)
			if !errors.Is(err, ErrInvalidModelLease) || !strings.Contains(err.Error(), "vram_reserved_bytes required") {
				t.Fatalf("ValidateModelLease() error = %v, want vram error", err)
			}
		})
	}
}

func TestModelLeaseInvalidTransitionRejected(t *testing.T) {
	lease := validModelLease(ModelLeaseStateUnpinned)

	_, err := ApplyModelLeaseEvent(lease, ModelLeaseEventStartLoading)
	if !errors.Is(err, ErrInvalidModelLeaseTransition) {
		t.Fatalf("ApplyModelLeaseEvent() error = %v, want ErrInvalidModelLeaseTransition", err)
	}
}

func TestModelLeaseEvictCompleteClearsReleasedFields(t *testing.T) {
	lease := validModelLease(ModelLeaseStateEvicting)

	lease = applyModelLeaseEventOrFatal(t, lease, ModelLeaseEventEvictComplete)
	if lease.State != ModelLeaseStateUnpinned {
		t.Fatalf("state after evict_complete = %q, want %q", lease.State, ModelLeaseStateUnpinned)
	}
	if lease.VRAMReservedBytes != 0 {
		t.Fatalf("vram after evict_complete = %d, want 0", lease.VRAMReservedBytes)
	}
	if lease.MinResidencyUntilUnixMs != 0 || lease.LeaseExpiresAtUnixMs != 0 || lease.ReadinessRewardCentsPerMinute != 0 {
		t.Fatalf("released fields not cleared: %+v", lease)
	}
}

func applyModelLeaseEventOrFatal(t *testing.T, lease ModelLease, event ModelLeaseEvent) ModelLease {
	t.Helper()

	next, err := ApplyModelLeaseEvent(lease, event)
	if err != nil {
		t.Fatalf("ApplyModelLeaseEvent(%q, %q) error = %v, want nil", lease.State, event, err)
	}
	return next
}

func validModelLease(state ModelLeaseState) ModelLease {
	lease := ModelLease{
		LeaseID:                       "lease-1",
		NodeID:                        "node-1",
		ModelID:                       "llama-3.1-8b",
		QuantizationID:                "q4_k_m",
		State:                         state,
		MinResidencyUntilUnixMs:       modelLeaseTestNowUnixMs + 60_000,
		LeaseExpiresAtUnixMs:          modelLeaseTestNowUnixMs + 600_000,
		ReadinessRewardCentsPerMinute: 3,
		UpdatedAtUnixMs:               modelLeaseTestNowUnixMs,
	}
	if modelLeaseStateRequiresVRAM(state) || state == ModelLeaseStateDraining || state == ModelLeaseStateEvicting {
		lease.VRAMReservedBytes = 80_000_000_000
	}
	return lease
}
