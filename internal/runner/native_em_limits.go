package runner

import (
	"os"
	"os/exec"
	"strconv"
	"strings"
)

// emResourceController is the OS-level hard-cap handle for a native EM process.
// It re-creates the containment OCI gave for free (memory.max / cpu.max on Linux
// via a cgroup v2 slice; a Job Object with a hard memory cap + kill-tree on
// Windows). Every method is safe to call on any value (including the no-op
// fallback returned when the OS feature is unavailable or the host is
// unprivileged) so RunNativeEM never has to branch on the platform.
type emResourceController interface {
	// AfterStart is called once the child process has been started. On Windows
	// it assigns the process to the Job Object before the child spawns its own
	// children; on Linux the cgroup is applied at clone time so this is a no-op.
	// A non-nil error means the hard cap is NOT in force and the process-group
	// fallback (kill-on-timeout) is the only containment.
	AfterStart(cmd *exec.Cmd) error
	// Kill terminates the whole controlled process tree (Job Object kill / cgroup
	// kill). Safe to call multiple times and before AfterStart.
	Kill()
	// Close releases any OS resources (cgroup dir, Job Object handle). Always
	// called via defer; safe on the no-op controller.
	Close()
}

// noopEMController is the fallback when no OS-level hard cap could be set up. The
// existing process-group + kill-on-timeout path (applyNativeEMResourceLimits'
// SysProcAttr side effects + killNativeEMProcessGroup) remains the containment.
type noopEMController struct{}

func (noopEMController) AfterStart(*exec.Cmd) error { return nil }
func (noopEMController) Kill()                      {}
func (noopEMController) Close()                     {}

// emMemoryCapBytes derives the hard RAM ceiling for the native EM process from
// the VRAM budget the scheduler sized the job to. FDTD field arrays of the same
// extent are mirrored in host RAM (staging, host-side grids, Python overhead),
// so we bound RAM at the VRAM budget plus headroom rather than at host RAM.
// Returns 0 when no meaningful cap can be derived (caller then skips the cap).
//
// RYV_EM_RAM_CAP_MB, if set (>0), overrides the derived value (operator escape
// hatch / GPU-less CPU engines that need a different ratio).
func emMemoryCapBytes(budget emBudget) uint64 {
	if override := strings.TrimSpace(os.Getenv("RYV_EM_RAM_CAP_MB")); override != "" {
		if mb, err := strconv.Atoi(override); err == nil && mb > 0 {
			return uint64(mb) << 20
		}
	}
	vram := budget.EstVRAMMB
	if vram <= 0 {
		return 0
	}
	// Headroom: 2x the VRAM budget + a 1 GiB floor for interpreter/runtime. The
	// 2x covers the host-side mirror of the device field arrays plus framework
	// overhead; the cell-budget cap (RYV_EM_MAX_CELLS) bounds the rest.
	capMB := vram*2 + 1024
	const minCapMB = 512
	if capMB < minCapMB {
		capMB = minCapMB
	}
	return uint64(capMB) << 20
}

// emCPUQuotaPercent returns the CPU cap as a percentage of a single core (e.g.
// 400 == 4 full cores), or 0 to leave CPU unbounded. Mirrors the OCI lane's
// RYV_CONTAINER_CPUS default of 4 cores; overridable via RYV_EM_CPU_CORES.
func emCPUQuotaPercent() int {
	cores := 4.0
	if override := strings.TrimSpace(os.Getenv("RYV_EM_CPU_CORES")); override != "" {
		if c, err := strconv.ParseFloat(override, 64); err == nil && c > 0 {
			cores = c
		}
	}
	pct := int(cores * 100)
	if pct <= 0 {
		return 0
	}
	return pct
}
