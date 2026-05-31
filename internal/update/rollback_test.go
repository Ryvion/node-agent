package update

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// fixedT is a stable "now" so attempt-window math is deterministic.
var fixedT = time.Date(2026, 5, 31, 12, 0, 0, 0, time.UTC)

// rollbackObserver isolates ledger state to a temp dir, pins time, stubs the
// rollback action, and returns a getter for whether/what was rolled back.
func rollbackObserver(t *testing.T) func() *updateState {
	t.Helper()
	t.Setenv("RYV_DATA_DIR", t.TempDir())
	rollbackNow = func() time.Time { return fixedT }
	t.Cleanup(func() { rollbackNow = time.Now })
	var rolled *updateState
	rollbackFunc = func(s updateState) error { c := s; rolled = &c; return nil }
	t.Cleanup(func() { rollbackFunc = performRollback })
	return func() *updateState { return rolled }
}

func existingPrev(t *testing.T) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "ryvion-node.prev")
	if err := os.WriteFile(p, []byte("prev-binary"), 0o755); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestRecordBoot_FirstBoot(t *testing.T) {
	getRolled := rollbackObserver(t)
	RecordBootAndMaybeRollback("1.0.0")
	if getRolled() != nil {
		t.Fatal("first boot must never roll back")
	}
	s := loadState()
	if s.RunningVersion != "1.0.0" || s.Committed || len(s.BootAttemptsMs) != 1 {
		t.Fatalf("unexpected first-boot state: %+v", s)
	}
}

func TestRecordBoot_DevVersionNoop(t *testing.T) {
	getRolled := rollbackObserver(t)
	RecordBootAndMaybeRollback("dev")
	RecordBootAndMaybeRollback("")
	if getRolled() != nil {
		t.Fatal("dev/unstamped builds are not managed")
	}
	if _, err := os.Stat(stateFilePath()); err == nil {
		t.Fatal("dev build must not write a ledger")
	}
}

func TestRecordBoot_NewVersionResetsKeepsPrev(t *testing.T) {
	getRolled := rollbackObserver(t)
	prev := existingPrev(t)
	saveState(updateState{RunningVersion: "1.0.0", Committed: true, PrevPath: prev, PrevVersion: "1.0.0"})

	RecordBootAndMaybeRollback("2.0.0") // new binary just got installed
	if getRolled() != nil {
		t.Fatal("a newly-installed version must not roll back on its first boot")
	}
	s := loadState()
	if s.RunningVersion != "2.0.0" || s.Committed || len(s.BootAttemptsMs) != 1 {
		t.Fatalf("new version should reset attempts/committed: %+v", s)
	}
	if s.PrevPath != prev || s.PrevVersion != "1.0.0" {
		t.Fatalf("last-known-good prev must be preserved across update: %+v", s)
	}
}

func TestRecordBoot_CommittedNeverRollsBack(t *testing.T) {
	getRolled := rollbackObserver(t)
	prev := existingPrev(t)
	saveState(updateState{
		RunningVersion: "2.0.0", Committed: true, PrevPath: prev, PrevVersion: "1.0.0",
		BootAttemptsMs: []int64{fixedT.Add(-3 * time.Minute).UnixMilli(), fixedT.Add(-2 * time.Minute).UnixMilli(), fixedT.Add(-time.Minute).UnixMilli()},
	})
	RecordBootAndMaybeRollback("2.0.0")
	if getRolled() != nil {
		t.Fatal("a binary that already proved healthy must never be auto-rolled-back")
	}
}

func TestRecordBoot_CrashLoopRollsBack(t *testing.T) {
	getRolled := rollbackObserver(t)
	prev := existingPrev(t)
	saveState(updateState{
		RunningVersion: "2.0.0", Committed: false, PrevPath: prev, PrevVersion: "1.0.0",
		BootAttemptsMs: []int64{fixedT.Add(-2 * time.Minute).UnixMilli(), fixedT.Add(-time.Minute).UnixMilli()},
	})
	RecordBootAndMaybeRollback("2.0.0") // the 3rd uncommitted boot in-window
	r := getRolled()
	if r == nil {
		t.Fatal("3 uncommitted boots with a valid prev must roll back")
	}
	if r.PrevVersion != "1.0.0" {
		t.Fatalf("rollback should target the last-known-good version, got %q", r.PrevVersion)
	}
}

func TestRecordBoot_NoRollbackToSameVersion(t *testing.T) {
	getRolled := rollbackObserver(t)
	prev := existingPrev(t)
	// PrevVersion == current => we ARE the rollback target; must not loop.
	saveState(updateState{
		RunningVersion: "1.0.0", Committed: false, PrevPath: prev, PrevVersion: "1.0.0",
		BootAttemptsMs: []int64{fixedT.Add(-2 * time.Minute).UnixMilli(), fixedT.Add(-time.Minute).UnixMilli()},
	})
	RecordBootAndMaybeRollback("1.0.0")
	if getRolled() != nil {
		t.Fatal("must not roll back to the same version (infinite-loop guard)")
	}
}

func TestRecordBoot_PrunesOldAttempts(t *testing.T) {
	getRolled := rollbackObserver(t)
	prev := existingPrev(t)
	// Two attempts older than the 10-min window => only this boot counts.
	saveState(updateState{
		RunningVersion: "2.0.0", Committed: false, PrevPath: prev, PrevVersion: "1.0.0",
		BootAttemptsMs: []int64{fixedT.Add(-30 * time.Minute).UnixMilli(), fixedT.Add(-20 * time.Minute).UnixMilli()},
	})
	RecordBootAndMaybeRollback("2.0.0")
	if getRolled() != nil {
		t.Fatal("stale attempts must be pruned and not trigger rollback")
	}
	if s := loadState(); len(s.BootAttemptsMs) != 1 {
		t.Fatalf("expected 1 in-window attempt after prune, got %d", len(s.BootAttemptsMs))
	}
}

func TestRecordBoot_NoRollbackIfPrevMissing(t *testing.T) {
	getRolled := rollbackObserver(t)
	saveState(updateState{
		RunningVersion: "2.0.0", Committed: false, PrevPath: "/nonexistent/ryvion-node.prev", PrevVersion: "1.0.0",
		BootAttemptsMs: []int64{fixedT.Add(-2 * time.Minute).UnixMilli(), fixedT.Add(-time.Minute).UnixMilli()},
	})
	RecordBootAndMaybeRollback("2.0.0")
	if getRolled() != nil {
		t.Fatal("must not roll back when the prev binary file is missing")
	}
}

func TestMarkBootHealthy_CommitsAndSnapshots(t *testing.T) {
	t.Setenv("RYV_DATA_DIR", t.TempDir())
	saveState(updateState{RunningVersion: "2.0.0", Committed: false, BootAttemptsMs: []int64{1, 2, 3}})

	MarkBootHealthy("2.0.0")

	s := loadState()
	if !s.Committed || len(s.BootAttemptsMs) != 0 {
		t.Fatalf("healthy boot should commit and clear attempts: %+v", s)
	}
	if s.PrevVersion != "2.0.0" || !fileExists(s.PrevPath) {
		t.Fatalf("healthy boot should snapshot self as prev: %+v", s)
	}
}

func TestMarkBootHealthy_Idempotent(t *testing.T) {
	t.Setenv("RYV_DATA_DIR", t.TempDir())
	MarkBootHealthy("2.0.0")
	first := loadState()
	MarkBootHealthy("2.0.0") // second call is a no-op
	second := loadState()
	if first.PrevPath != second.PrevPath || !second.Committed {
		t.Fatalf("repeat commit must be a stable no-op: %+v -> %+v", first, second)
	}
}
