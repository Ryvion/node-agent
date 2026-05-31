package update

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// Layer-2 crash-loop rollback. Layer-1 (selfCheckBinary) stops a binary that
// can't START from being adopted. Layer-2 covers the residual case: a binary
// that passes --version, gets adopted, but then crashes during real operation —
// the service manager restarts it into the same crash, and (because the
// auto-update check lives in the heartbeat loop) it can never pull a fix. This
// brick is the exact incident this guards against.
//
// Design — SAFE BY DEFAULT. Any uncertainty => boot normally, never roll back:
//   - A binary snapshots ITSELF to <data>/ryvion-node.prev only once it proves
//     healthy (MarkBootHealthy: first successful hub registration, or a grace
//     period of uptime). That snapshot is the rollback target.
//   - Every boot records an attempt. A binary that has NOT yet proved healthy and
//     has restarted >= maxUncommittedBoots times within bootAttemptWindow rolls
//     back to the last-known-good .prev.
//   - A binary that has EVER proved healthy is never auto-rolled-back (its crashes
//     are environmental/job-specific, not a bad-binary brick).
//   - Rolling back sets RunningVersion = prevVersion, so the next boot sees
//     "we are the rollback target" and the PrevVersion != current guard prevents
//     an infinite rollback loop.
//   - All state I/O is best-effort; a read/write error degrades to "no rollback".
//
// It reuses the proven replaceUnix/replaceWindows atomic-swap + self-check, so
// the rollback path is as safe as the forward-update path.

const (
	bootAttemptWindow   = 10 * time.Minute
	maxUncommittedBoots = 3
	prevBinaryBaseName  = "ryvion-node.prev"
)

// rollbackNow is overridable in tests; argless time.Now() is otherwise used.
var rollbackNow = time.Now

// stateMu serializes ledger read-modify-write across the boot check, the
// registration-success commit, and the grace-timer commit.
var stateMu sync.Mutex

// rollbackFunc is the rollback action; a package var so tests can observe the
// decision without performing a real binary swap + restart.
var rollbackFunc = performRollback

// updateState is the persisted boot ledger (<data>/update_state.json).
type updateState struct {
	// RunningVersion is the version the ledger last recorded as on-disk.
	RunningVersion string `json:"running_version"`
	// Committed is true once RunningVersion proved healthy at least once.
	Committed bool `json:"committed"`
	// BootAttemptsMs are unix-ms starts of RunningVersion not yet proven healthy.
	BootAttemptsMs []int64 `json:"boot_attempts_ms,omitempty"`
	// PrevPath is the last-known-good binary snapshot to roll back to.
	PrevPath string `json:"prev_path,omitempty"`
	// PrevVersion is the version of PrevPath.
	PrevVersion string `json:"prev_version,omitempty"`
}

func rollbackStateDir() string {
	if d := strings.TrimSpace(os.Getenv("RYV_DATA_DIR")); d != "" {
		return d
	}
	if home, err := os.UserHomeDir(); err == nil && strings.TrimSpace(home) != "" {
		return filepath.Join(home, ".ryvion")
	}
	return filepath.Join(os.TempDir(), "ryvion")
}

func stateFilePath() string { return filepath.Join(rollbackStateDir(), "update_state.json") }

func prevBinaryPath() string {
	name := prevBinaryBaseName
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	return filepath.Join(rollbackStateDir(), name)
}

// loadState reads the ledger; any error (missing/corrupt) yields the zero value
// so the node boots normally and never rolls back on bad state.
func loadState() updateState {
	var s updateState
	b, err := os.ReadFile(stateFilePath())
	if err != nil {
		return updateState{}
	}
	if json.Unmarshal(b, &s) != nil {
		return updateState{}
	}
	return s
}

// saveState writes the ledger atomically (temp + rename). Best-effort: a failure
// to persist must never crash the node.
func saveState(s updateState) {
	dir := rollbackStateDir()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		slog.Warn("update rollback: could not create state dir", "dir", dir, "error", err)
		return
	}
	b, err := json.Marshal(s)
	if err != nil {
		return
	}
	tmp, err := os.CreateTemp(dir, ".update_state-*")
	if err != nil {
		return
	}
	tmpPath := tmp.Name()
	if _, err := tmp.Write(b); err != nil {
		tmp.Close()
		_ = os.Remove(tmpPath)
		return
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return
	}
	if err := os.Rename(tmpPath, stateFilePath()); err != nil {
		_ = os.Remove(tmpPath)
	}
}

func pruneAttempts(attempts []int64, nowMs int64) []int64 {
	cutoff := nowMs - bootAttemptWindow.Milliseconds()
	kept := attempts[:0:0]
	for _, a := range attempts {
		if a >= cutoff {
			kept = append(kept, a)
		}
	}
	return kept
}

func fileExists(path string) bool {
	if strings.TrimSpace(path) == "" {
		return false
	}
	st, err := os.Stat(path)
	return err == nil && !st.IsDir()
}

// RecordBootAndMaybeRollback runs as early as possible at startup. It records
// this boot and, if the current binary is crash-looping before ever proving
// healthy, rolls back to the last-known-good binary. On a successful rollback it
// restarts and does not return.
func RecordBootAndMaybeRollback(currentVersion string) {
	currentVersion = strings.TrimSpace(currentVersion)
	if currentVersion == "" || currentVersion == "dev" {
		return // unstamped/dev builds are not auto-update managed.
	}

	stateMu.Lock()
	s := loadState()
	nowMs := rollbackNow().UnixMilli()

	switch {
	case s.RunningVersion == "":
		// First tracked boot: start the ledger, nothing to roll back to.
		s = updateState{RunningVersion: currentVersion, BootAttemptsMs: []int64{nowMs}}
		saveState(s)
		stateMu.Unlock()
		return

	case s.RunningVersion != currentVersion:
		// The on-disk binary changed since last boot (forward update, a prior
		// rollback, or a manual install). Reset the attempt counter for the new
		// version but KEEP the last-known-good prev as the rollback target.
		s.RunningVersion = currentVersion
		s.Committed = false
		s.BootAttemptsMs = []int64{nowMs}
		saveState(s)
		stateMu.Unlock()
		return
	}

	// Same version as last boot.
	s.BootAttemptsMs = append(pruneAttempts(s.BootAttemptsMs, nowMs), nowMs)
	shouldRollback := !s.Committed &&
		len(s.BootAttemptsMs) >= maxUncommittedBoots &&
		s.PrevVersion != "" && s.PrevVersion != currentVersion &&
		fileExists(s.PrevPath)
	saveState(s)
	snapshot := s
	stateMu.Unlock()

	if !shouldRollback {
		return
	}
	if err := rollbackFunc(snapshot); err != nil {
		slog.Error("crash-loop rollback failed; staying on current binary", "error", err)
	}
}

// MarkBootHealthy is called once the running binary has proven it works (hub
// registration succeeded, or it survived the grace period). It snapshots the
// running binary as the new last-known-good and commits it so it is never
// auto-rolled-back. Idempotent + best-effort.
func MarkBootHealthy(currentVersion string) {
	currentVersion = strings.TrimSpace(currentVersion)
	if currentVersion == "" || currentVersion == "dev" {
		return
	}
	stateMu.Lock()
	defer stateMu.Unlock()
	s := loadState()
	if s.Committed && s.RunningVersion == currentVersion {
		return // already committed this version; nothing to do.
	}
	if exe, err := resolveSelfExe(); err == nil {
		if data, err := os.ReadFile(exe); err == nil {
			if err := writeBinaryAtomic(prevBinaryPath(), data); err == nil {
				s.PrevPath = prevBinaryPath()
				s.PrevVersion = currentVersion
			} else {
				slog.Warn("update rollback: could not snapshot known-good binary", "error", err)
			}
		}
	}
	s.RunningVersion = currentVersion
	s.Committed = true
	s.BootAttemptsMs = nil
	saveState(s)
}

func resolveSelfExe() (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	if resolved, err := filepath.EvalSymlinks(exe); err == nil && strings.TrimSpace(resolved) != "" {
		return resolved, nil
	}
	return exe, nil
}

func writeBinaryAtomic(target string, data []byte) error {
	dir := filepath.Dir(target)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".ryvion-node-prev-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		_ = os.Remove(tmpPath)
		return err
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	if err := os.Chmod(tmpPath, 0o755); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	if err := os.Rename(tmpPath, target); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	return nil
}

// performRollback reverts the running executable to the recorded last-known-good
// binary and restarts. It reuses the same self-checked atomic-swap path as a
// forward update, then records that we are now the rollback target (so the next
// boot's PrevVersion==current guard prevents an infinite loop).
func performRollback(s updateState) error {
	slog.Warn("crash-loop detected before this version ever became healthy; rolling back",
		"from", s.RunningVersion, "to", s.PrevVersion, "prev_binary", s.PrevPath)

	if err := selfCheckBinary(s.PrevPath); err != nil {
		return fmt.Errorf("rollback target failed self-check: %w", err)
	}
	data, err := os.ReadFile(s.PrevPath)
	if err != nil {
		return fmt.Errorf("read previous binary: %w", err)
	}
	exePath, err := resolveSelfExe()
	if err != nil {
		return fmt.Errorf("resolve executable: %w", err)
	}

	// Record the rollback BEFORE swapping so a crash mid-swap still leaves a
	// coherent ledger (RunningVersion=prev, committed=false, same prev target).
	stateMu.Lock()
	saveState(updateState{
		RunningVersion: s.PrevVersion,
		Committed:      false,
		BootAttemptsMs: nil,
		PrevPath:       s.PrevPath,
		PrevVersion:    s.PrevVersion,
	})
	stateMu.Unlock()

	if runtime.GOOS == "windows" {
		if err := replaceWindows(exePath, data); err != nil {
			return fmt.Errorf("windows rollback swap: %w", err)
		}
	} else {
		if err := replaceUnix(exePath, data); err != nil {
			return fmt.Errorf("rollback swap: %w", err)
		}
	}
	slog.Warn("rolled back to previous binary; restarting", "version", s.PrevVersion)
	return Restart()
}
