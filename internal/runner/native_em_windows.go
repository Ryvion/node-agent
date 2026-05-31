//go:build windows

package runner

import (
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

// applyNativeEMResourceLimits creates a new process group on Windows so the
// engine and its children can be signalled together, and additionally builds a
// Job Object with a hard process-memory cap (JOB_OBJECT_LIMIT_PROCESS_MEMORY)
// and kill-on-close + active-process bounds. The job is assigned to the child in
// AfterStart (after CreateProcess but before the child spawns its own children),
// so the whole tree is captured and terminated as a unit on breach/timeout —
// re-creating what the OCI --memory + container teardown gave for free.
//
// If the Job Object cannot be created, we degrade to the process-group +
// taskkill /T fallback and return the no-op controller. Never nil.
func applyNativeEMResourceLimits(cmd *exec.Cmd, budget emBudget) emResourceController {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.CreationFlags |= syscall.CREATE_NEW_PROCESS_GROUP

	ctrl, err := newWindowsJobController(budget)
	if err != nil {
		slog.Warn("EM Job Object hard cap unavailable; using process-group fallback", "error", err)
		return noopEMController{}
	}
	return ctrl
}

// killNativeEMProcessGroup terminates the EM process tree. CommandContext kills
// the immediate process when the deadline fires; taskkill /T reaps children that
// escaped the Job Object (or when no Job Object was assigned).
func killNativeEMProcessGroup(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	pid := cmd.Process.Pid
	_ = exec.Command("taskkill", "/F", "/T", "/PID", itoaWin(pid)).Run()
	_ = cmd.Process.Kill()
}

// windowsJobController owns a Job Object handle. Assigning the child to it makes
// the kernel enforce JOB_OBJECT_LIMIT_PROCESS_MEMORY and terminate every process
// in the job when TerminateJobObject is called or the last handle closes
// (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE).
type windowsJobController struct {
	job windows.Handle
}

func newWindowsJobController(budget emBudget) (*windowsJobController, error) {
	job, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		return nil, fmt.Errorf("CreateJobObject: %w", err)
	}

	limits := windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION{}
	limits.BasicLimitInformation.LimitFlags = windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

	// JOB_OBJECT_LIMIT_PROCESS_MEMORY: hard per-process commit cap. Breach makes
	// the offending allocation fail (engine OOM-aborts) without taking the host
	// down — the kill-tree on timeout reaps it.
	if capBytes := emMemoryCapBytes(budget); capBytes > 0 {
		limits.BasicLimitInformation.LimitFlags |= windows.JOB_OBJECT_LIMIT_PROCESS_MEMORY
		limits.ProcessMemoryLimit = uintptr(capBytes)
	}

	// JOB_OBJECT_LIMIT_ACTIVE_PROCESS: bound the number of concurrent processes
	// in the job, mirroring the OCI --pids-limit guard against fork bombs.
	limits.BasicLimitInformation.LimitFlags |= windows.JOB_OBJECT_LIMIT_ACTIVE_PROCESS
	limits.BasicLimitInformation.ActiveProcessLimit = emActiveProcessLimit()

	if _, err := windows.SetInformationJobObject(
		job,
		windows.JobObjectExtendedLimitInformation,
		uintptr(unsafe.Pointer(&limits)),
		uint32(unsafe.Sizeof(limits)),
	); err != nil {
		_ = windows.CloseHandle(job)
		return nil, fmt.Errorf("SetInformationJobObject: %w", err)
	}

	slog.Info("EM Job Object hard cap armed",
		"memory_max_mb", emMemoryCapBytes(budget)>>20,
		"active_process_limit", emActiveProcessLimit())
	return &windowsJobController{job: job}, nil
}

// AfterStart assigns the freshly-started child to the Job Object. Because the
// child has been started but its main thread has barely begun (and certainly
// before it has spawned worker processes), the whole tree is captured.
func (c *windowsJobController) AfterStart(cmd *exec.Cmd) error {
	if c == nil || c.job == 0 {
		return fmt.Errorf("job object not initialized")
	}
	if cmd == nil || cmd.Process == nil {
		return fmt.Errorf("process not started")
	}
	h, err := windows.OpenProcess(
		windows.PROCESS_SET_QUOTA|windows.PROCESS_TERMINATE,
		false,
		uint32(cmd.Process.Pid),
	)
	if err != nil {
		return fmt.Errorf("OpenProcess: %w", err)
	}
	defer windows.CloseHandle(h)
	if err := windows.AssignProcessToJobObject(c.job, h); err != nil {
		return fmt.Errorf("AssignProcessToJobObject: %w", err)
	}
	return nil
}

// Kill terminates every process in the job as a unit.
func (c *windowsJobController) Kill() {
	if c == nil || c.job == 0 {
		return
	}
	_ = windows.TerminateJobObject(c.job, 1)
}

// Close releases the Job Object handle. With KILL_ON_JOB_CLOSE set, closing the
// last handle also terminates any survivors, so Close doubles as a safety net.
func (c *windowsJobController) Close() {
	if c == nil || c.job == 0 {
		return
	}
	_ = windows.CloseHandle(c.job)
	c.job = 0
}

// emActiveProcessLimit bounds concurrent processes in the EM job, mirroring the
// OCI --pids-limit=256 guard. Overridable via RYV_EM_MAX_PROCS.
func emActiveProcessLimit() uint32 {
	const def = 256
	if v := windowsEnvInt("RYV_EM_MAX_PROCS"); v > 0 {
		return uint32(v)
	}
	return def
}

func windowsEnvInt(key string) int {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return 0
	}
	n, err := strconv.Atoi(raw)
	if err != nil {
		return 0
	}
	return n
}

func itoaWin(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
