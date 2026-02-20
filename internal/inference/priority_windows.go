//go:build windows

package inference

import (
	"log/slog"
	"syscall"
	"unsafe"
)

var (
	modkernel32         = syscall.NewLazyDLL("kernel32.dll")
	procSetPriorityClass = modkernel32.NewProc("SetPriorityClass")
	procOpenProcess      = modkernel32.NewProc("OpenProcess")
	procCloseHandle      = modkernel32.NewProc("CloseHandle")
)

const (
	processSetInformation    = 0x0200
	belowNormalPriorityClass = 0x00004000
)

// setLowPriority sets the process to BELOW_NORMAL_PRIORITY_CLASS on Windows
// so operator workloads like games take precedence over inference jobs.
func setLowPriority(pid int) {
	handle, _, err := procOpenProcess.Call(processSetInformation, 0, uintptr(pid))
	if handle == 0 {
		slog.Debug("failed to open process for priority", "pid", pid, "error", err)
		return
	}
	defer procCloseHandle.Call(handle)

	ret, _, err := procSetPriorityClass.Call(handle, belowNormalPriorityClass)
	if ret == 0 {
		slog.Debug("failed to set below-normal priority", "pid", pid, "error", err)
	}
}

// Suppress unused import warning — unsafe is needed for syscall interop.
var _ unsafe.Pointer
