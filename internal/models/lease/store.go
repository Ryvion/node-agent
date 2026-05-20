package modellease

type ModelLeaseStore interface {
	Save(lease ModelLease) error
	Get(leaseID string) (ModelLease, bool)
	List() []ModelLease
}
