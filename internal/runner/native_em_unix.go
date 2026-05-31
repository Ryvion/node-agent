//go:build !windows && !linux

package runner

import (
	"os/exec"
	"syscall"
)

// applyNativeEMResourceLimits (darwin/BSD): no cgroup v2 here, so the hard-cap
// guarantee is the cell-budget cap (RYV_EM_MAX_CELLS / RYV_EM_VRAM_MB) inside
// the bundle plus process-group containment + kill-on-timeout. Returns the
// no-op controller; never nil.
func applyNativeEMResourceLimits(cmd *exec.Cmd, _ emBudget) emResourceController {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true
	return noopEMController{}
}

// killNativeEMProcessGroup sends SIGKILL to the entire process group so any
// child processes the engine spawned are reaped after a timeout/abort.
func killNativeEMProcessGroup(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	pgid, err := syscall.Getpgid(cmd.Process.Pid)
	if err == nil && pgid > 0 {
		_ = syscall.Kill(-pgid, syscall.SIGKILL)
		return
	}
	_ = cmd.Process.Kill()
}
