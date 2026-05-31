//go:build linux

package runner

import (
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
)

// applyNativeEMResourceLimits places the EM process in its own process group so
// the whole tree can be killed on timeout, and builds a dedicated cgroup v2
// slice with memory.max and cpu.max so a runaway engine is hard-capped (and
// OOM-killed) by the kernel exactly like the OCI lane did.
//
// The cgroup is applied to the child at clone time via SysProcAttr.UseCgroupFD
// (race-free: the process is born inside the cgroup), so AfterStart is a no-op
// for the cgroup controller. When the cgroup cannot be created (cgroup v2 not
// mounted, unprivileged host) we fall back to the process-group +
// kill-on-timeout containment and return the no-op controller. Never nil.
func applyNativeEMResourceLimits(cmd *exec.Cmd, budget emBudget) emResourceController {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	// Process-group containment (fallback + straggler reaping) — always set.
	cmd.SysProcAttr.Setpgid = true

	ctrl, err := newLinuxCgroupController(cmd, budget)
	if err != nil {
		slog.Warn("EM cgroup hard cap unavailable; using process-group fallback", "error", err)
		return noopEMController{}
	}
	return ctrl
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

// cgroup2Mount is the conventional cgroup v2 unified hierarchy mount point.
const cgroup2Mount = "/sys/fs/cgroup"

// linuxCgroupController owns a dedicated cgroup v2 directory under the unified
// hierarchy and the open FD used to clone the child directly into it.
type linuxCgroupController struct {
	dir   string
	dirFD int
}

// newLinuxCgroupController creates a leaf cgroup v2 slice, writes memory.max /
// cpu.max derived from the job budget, and wires cmd.SysProcAttr to place the
// child into it at clone time. It also enables memory.oom.group so the kernel
// kills the whole cgroup as a unit on OOM. Any failure unwinds and returns an
// error so the caller degrades to the process-group fallback.
func newLinuxCgroupController(cmd *exec.Cmd, budget emBudget) (*linuxCgroupController, error) {
	if !cgroup2Available() {
		return nil, fmt.Errorf("cgroup v2 not mounted at %s", cgroup2Mount)
	}
	base := strings.TrimSpace(os.Getenv("RYV_EM_CGROUP_PARENT"))
	if base == "" {
		base = filepath.Join(cgroup2Mount, "ryvion-em")
	}
	// Ensure the parent slice exists. On an unprivileged host this MkdirAll
	// fails (EACCES/EROFS) and we degrade — never crash.
	if err := os.MkdirAll(base, 0o755); err != nil {
		return nil, fmt.Errorf("create cgroup parent %s: %w", base, err)
	}
	// Make controllers available to children of the parent slice. Best-effort:
	// if the parent already delegates them this is a no-op; if it fails the
	// leaf writes below will surface the real error.
	enableCgroupControllers(base)

	dir, err := os.MkdirTemp(base, "job-*")
	if err != nil {
		return nil, fmt.Errorf("create leaf cgroup: %w", err)
	}

	// memory.max — hard RAM ceiling. Breach => kernel OOM-kills the cgroup.
	if capBytes := emMemoryCapBytes(budget); capBytes > 0 {
		if err := writeCgroupFile(dir, "memory.max", strconv.FormatUint(capBytes, 10)); err != nil {
			cleanupCgroupDir(dir)
			return nil, fmt.Errorf("set memory.max: %w", err)
		}
		// Kill the entire cgroup as a unit on OOM rather than a single task.
		_ = writeCgroupFile(dir, "memory.oom.group", "1")
		// Disable swap so the RAM cap is a true ceiling (best-effort; may be
		// absent when swap accounting is compiled out).
		_ = writeCgroupFile(dir, "memory.swap.max", "0")
	}

	// cpu.max — "<quota> <period>" in microseconds. quota/period == cores.
	if pct := emCPUQuotaPercent(); pct > 0 {
		const period = 100000 // 100ms
		quota := pct * period / 100
		if err := writeCgroupFile(dir, "cpu.max", fmt.Sprintf("%d %d", quota, period)); err != nil {
			// CPU cap is best-effort: the cpu controller may not be delegated.
			// Don't fail the whole controller over it (memory is the hard one).
			slog.Warn("EM cgroup cpu.max not set; cpu uncapped", "error", err)
		}
	}

	// Open the cgroup dir FD so the runtime can clone the child into it.
	fd, err := syscall.Open(dir, syscall.O_RDONLY|syscall.O_DIRECTORY|syscall.O_CLOEXEC, 0)
	if err != nil {
		cleanupCgroupDir(dir)
		return nil, fmt.Errorf("open cgroup dir fd: %w", err)
	}

	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.UseCgroupFD = true
	cmd.SysProcAttr.CgroupFD = fd

	slog.Info("EM cgroup hard cap armed", "cgroup", dir,
		"memory_max_mb", emMemoryCapBytes(budget)>>20, "cpu_pct", emCPUQuotaPercent())
	return &linuxCgroupController{dir: dir, dirFD: fd}, nil
}

// AfterStart is a no-op: the child was already cloned into the cgroup via
// UseCgroupFD. We close the cloned-into FD now that it has served its purpose.
func (c *linuxCgroupController) AfterStart(*exec.Cmd) error {
	if c.dirFD > 0 {
		_ = syscall.Close(c.dirFD)
		c.dirFD = 0
	}
	return nil
}

// Kill writes "1" to cgroup.kill, atomically SIGKILLing every task in the slice
// (Linux 5.14+). Falls back to listing cgroup.procs and killing each PID on
// older kernels where cgroup.kill is absent.
func (c *linuxCgroupController) Kill() {
	if c == nil || c.dir == "" {
		return
	}
	if err := writeCgroupFile(c.dir, "cgroup.kill", "1"); err == nil {
		return
	}
	// Fallback for kernels without cgroup.kill.
	if data, err := os.ReadFile(filepath.Join(c.dir, "cgroup.procs")); err == nil {
		for _, line := range strings.Fields(string(data)) {
			if pid, perr := strconv.Atoi(line); perr == nil && pid > 0 {
				_ = syscall.Kill(pid, syscall.SIGKILL)
			}
		}
	}
}

// Close releases the cgroup dir FD and removes the now-empty leaf cgroup. rmdir
// only succeeds once the cgroup has no live tasks, so Kill must precede it (the
// caller's defer ordering guarantees Kill runs first).
func (c *linuxCgroupController) Close() {
	if c == nil {
		return
	}
	if c.dirFD > 0 {
		_ = syscall.Close(c.dirFD)
		c.dirFD = 0
	}
	if c.dir != "" {
		cleanupCgroupDir(c.dir)
	}
}

func cgroup2Available() bool {
	// cgroup v2 exposes cgroup.controllers at the unified mount root.
	_, err := os.Stat(filepath.Join(cgroup2Mount, "cgroup.controllers"))
	return err == nil
}

// enableCgroupControllers best-effort delegates memory+cpu to children of the
// parent slice by writing to cgroup.subtree_control. Errors are non-fatal.
func enableCgroupControllers(parent string) {
	for _, ctrl := range []string{"+memory", "+cpu"} {
		_ = writeCgroupFile(parent, "cgroup.subtree_control", ctrl)
	}
}

func writeCgroupFile(dir, name, value string) error {
	return os.WriteFile(filepath.Join(dir, name), []byte(value), 0o644)
}

// cleanupCgroupDir removes a leaf cgroup directory (rmdir; cgroupfs rejects
// unlink of regular interface files, so RemoveAll is wrong here).
func cleanupCgroupDir(dir string) {
	if err := os.Remove(dir); err != nil {
		// Tasks may still be exiting; one retry after a best-effort kill.
		slog.Debug("EM cgroup leaf not yet removable", "dir", dir, "error", err)
	}
}
