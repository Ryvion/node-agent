package modellease

import "fmt"

type ModelLeaseEvent string

const (
	ModelLeaseEventBid            ModelLeaseEvent = "bid"
	ModelLeaseEventStartLoading   ModelLeaseEvent = "start_loading"
	ModelLeaseEventWarmupComplete ModelLeaseEvent = "warmup_complete"
	ModelLeaseEventBecomeResident ModelLeaseEvent = "become_resident"
	ModelLeaseEventStartDraining  ModelLeaseEvent = "start_draining"
	ModelLeaseEventStartEvicting  ModelLeaseEvent = "start_evicting"
	ModelLeaseEventEvictComplete  ModelLeaseEvent = "evict_complete"
	ModelLeaseEventFail           ModelLeaseEvent = "fail"
	ModelLeaseEventReset          ModelLeaseEvent = "reset"
)

func ApplyModelLeaseEvent(lease ModelLease, event ModelLeaseEvent) (ModelLease, error) {
	if !isKnownModelLeaseEvent(event) {
		return ModelLease{}, fmt.Errorf("%w: unknown event %q", ErrInvalidModelLeaseEvent, event)
	}
	if !isKnownModelLeaseState(lease.State) {
		return ModelLease{}, fmt.Errorf("%w: unknown state %q", ErrInvalidModelLease, lease.State)
	}

	next := lease
	switch event {
	case ModelLeaseEventBid:
		if lease.State != ModelLeaseStateUnpinned {
			return ModelLease{}, invalidModelLeaseTransition(lease.State, event)
		}
		next.State = ModelLeaseStateBidding
	case ModelLeaseEventStartLoading:
		if lease.State != ModelLeaseStateBidding {
			return ModelLease{}, invalidModelLeaseTransition(lease.State, event)
		}
		next.State = ModelLeaseStateLoading
	case ModelLeaseEventWarmupComplete:
		if lease.State != ModelLeaseStateLoading {
			return ModelLease{}, invalidModelLeaseTransition(lease.State, event)
		}
		next.State = ModelLeaseStateWarmup
	case ModelLeaseEventBecomeResident:
		if lease.State != ModelLeaseStateWarmup {
			return ModelLease{}, invalidModelLeaseTransition(lease.State, event)
		}
		next.State = ModelLeaseStateResident
	case ModelLeaseEventStartDraining:
		if lease.State != ModelLeaseStateResident {
			return ModelLease{}, invalidModelLeaseTransition(lease.State, event)
		}
		next.State = ModelLeaseStateDraining
	case ModelLeaseEventStartEvicting:
		if lease.State != ModelLeaseStateDraining {
			return ModelLease{}, invalidModelLeaseTransition(lease.State, event)
		}
		next.State = ModelLeaseStateEvicting
	case ModelLeaseEventEvictComplete:
		if lease.State != ModelLeaseStateEvicting {
			return ModelLease{}, invalidModelLeaseTransition(lease.State, event)
		}
		next.State = ModelLeaseStateUnpinned
		clearReleasedModelLeaseFields(&next)
	case ModelLeaseEventFail:
		next.State = ModelLeaseStateFailed
	case ModelLeaseEventReset:
		if lease.State != ModelLeaseStateFailed {
			return ModelLease{}, invalidModelLeaseTransition(lease.State, event)
		}
		next.State = ModelLeaseStateUnpinned
		clearReleasedModelLeaseFields(&next)
	}

	if err := ValidateModelLease(next); err != nil {
		return ModelLease{}, err
	}
	return next, nil
}

func invalidModelLeaseTransition(state ModelLeaseState, event ModelLeaseEvent) error {
	return fmt.Errorf("%w: cannot apply %q from state %q", ErrInvalidModelLeaseTransition, event, state)
}

func isKnownModelLeaseEvent(event ModelLeaseEvent) bool {
	switch event {
	case ModelLeaseEventBid,
		ModelLeaseEventStartLoading,
		ModelLeaseEventWarmupComplete,
		ModelLeaseEventBecomeResident,
		ModelLeaseEventStartDraining,
		ModelLeaseEventStartEvicting,
		ModelLeaseEventEvictComplete,
		ModelLeaseEventFail,
		ModelLeaseEventReset:
		return true
	default:
		return false
	}
}
