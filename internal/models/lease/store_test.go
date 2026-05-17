package modellease

import "testing"

func TestInMemoryModelLeaseStoreSaveGetListWorks(t *testing.T) {
	store := NewInMemoryModelLeaseStore()
	leaseB := validStoreModelLease("lease-b", ModelLeaseStateBidding)
	leaseA := validStoreModelLease("lease-a", ModelLeaseStateResident)

	if err := store.Save(leaseB); err != nil {
		t.Fatalf("Save(lease-b) error = %v", err)
	}
	if err := store.Save(leaseA); err != nil {
		t.Fatalf("Save(lease-a) error = %v", err)
	}

	got, ok := store.Get("lease-a")
	if !ok {
		t.Fatalf("Get(lease-a) ok = false, want true")
	}
	if got != leaseA {
		t.Fatalf("Get(lease-a) = %+v, want %+v", got, leaseA)
	}

	leases := store.List()
	if len(leases) != 2 {
		t.Fatalf("List() length = %d, want 2", len(leases))
	}
	if leases[0].LeaseID != "lease-a" || leases[1].LeaseID != "lease-b" {
		t.Fatalf("List() order = [%q, %q], want [lease-a, lease-b]", leases[0].LeaseID, leases[1].LeaseID)
	}
}

func TestInMemoryModelLeaseStoreRejectsInvalidLease(t *testing.T) {
	store := NewInMemoryModelLeaseStore()
	lease := validStoreModelLease("lease-1", ModelLeaseStateResident)
	lease.ModelID = ""

	if err := store.Save(lease); err == nil {
		t.Fatalf("Save(invalid) error = nil, want error")
	}
}

func TestInMemoryModelLeaseStoreClonesData(t *testing.T) {
	store := NewInMemoryModelLeaseStore()
	lease := validStoreModelLease("lease-1", ModelLeaseStateResident)

	if err := store.Save(lease); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	lease.ModelID = "mutated-after-save"

	got, ok := store.Get("lease-1")
	if !ok {
		t.Fatalf("Get() ok = false, want true")
	}
	if got.ModelID != "model-lease-1" {
		t.Fatalf("stored model id = %q, want original", got.ModelID)
	}

	got.ModelID = "mutated-after-get"
	gotAgain, ok := store.Get("lease-1")
	if !ok {
		t.Fatalf("Get() second ok = false, want true")
	}
	if gotAgain.ModelID != "model-lease-1" {
		t.Fatalf("stored model id after get mutation = %q, want original", gotAgain.ModelID)
	}

	leases := store.List()
	if len(leases) != 1 {
		t.Fatalf("List() length = %d, want 1", len(leases))
	}
	leases[0].ModelID = "mutated-after-list"
	gotAfterListMutation, ok := store.Get("lease-1")
	if !ok {
		t.Fatalf("Get() after list ok = false, want true")
	}
	if gotAfterListMutation.ModelID != "model-lease-1" {
		t.Fatalf("stored model id after list mutation = %q, want original", gotAfterListMutation.ModelID)
	}
}

func validStoreModelLease(leaseID string, state ModelLeaseState) ModelLease {
	lease := validModelLease(state)
	lease.LeaseID = leaseID
	lease.ModelID = "model-" + leaseID
	return lease
}
