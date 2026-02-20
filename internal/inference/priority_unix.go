//go:build !windows

package inference

import (
	"log/slog"
	"syscall"
)

// setLowPriority lowers the process priority (nice +10) so operator
// workloads like games take precedence over inference jobs.
func setLowPriority(pid int) {
	if err := syscall.Setpriority(syscall.PRIO_PROCESS, pid, 10); err != nil {
		slog.Debug("failed to set low priority", "pid", pid, "error", err)
	}
}
