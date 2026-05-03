package modellease

import (
	"sort"
	"sync"
)

type InMemoryModelLeaseStore struct {
	mu     sync.RWMutex
	leases map[string]ModelLease
}

func NewInMemoryModelLeaseStore() *InMemoryModelLeaseStore {
	return &InMemoryModelLeaseStore{
		leases: make(map[string]ModelLease),
	}
}

func (store *InMemoryModelLeaseStore) Save(lease ModelLease) error {
	if err := ValidateModelLease(lease); err != nil {
		return err
	}

	store.mu.Lock()
	defer store.mu.Unlock()

	if store.leases == nil {
		store.leases = make(map[string]ModelLease)
	}
	store.leases[lease.LeaseID] = cloneModelLease(lease)
	return nil
}

func (store *InMemoryModelLeaseStore) Get(leaseID string) (ModelLease, bool) {
	store.mu.RLock()
	defer store.mu.RUnlock()

	lease, ok := store.leases[leaseID]
	if !ok {
		return ModelLease{}, false
	}
	return cloneModelLease(lease), true
}

func (store *InMemoryModelLeaseStore) List() []ModelLease {
	store.mu.RLock()
	defer store.mu.RUnlock()

	leaseIDs := make([]string, 0, len(store.leases))
	for leaseID := range store.leases {
		leaseIDs = append(leaseIDs, leaseID)
	}
	sort.Strings(leaseIDs)

	leases := make([]ModelLease, 0, len(leaseIDs))
	for _, leaseID := range leaseIDs {
		leases = append(leases, cloneModelLease(store.leases[leaseID]))
	}
	return leases
}

func cloneModelLease(lease ModelLease) ModelLease {
	return lease
}
