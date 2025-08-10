package executor

import (
    "bytes"
    "context"
    "crypto/sha256"
    "encoding/hex"
    "os/exec"
    "time"
)

// TryDockerInference attempts to run a dockerized inference job.
// If docker is unavailable or the run fails quickly, returns ok=false and an empty hash.
func TryDockerInference(ctx context.Context, image string, args []string, timeout time.Duration) (hashHex string, exitCode int, logs string, ok bool) {
    if image == "" { return "", 0, "", false }
    if timeout <= 0 { timeout = 30 * time.Second }
    // Check for docker binary
    if _, err := exec.LookPath("docker"); err != nil { return "", 0, "", false }
    // Build command: docker run --rm --gpus all <image> <args>
    allArgs := append([]string{"run", "--rm", "--gpus", "all", image}, args...)
    cmd := exec.CommandContext(ctx, "docker", allArgs...)
    var out bytes.Buffer
    cmd.Stdout = &out
    cmd.Stderr = &out
    if err := cmd.Start(); err != nil { return "", 0, "", false }
    done := make(chan error, 1)
    go func() { done <- cmd.Wait() }()
    select {
    case err := <-done:
        if err != nil {
            if cmd.ProcessState != nil { exitCode = cmd.ProcessState.ExitCode() }
            logs = tailString(out.String(), 2048)
            // still compute hash over logs for determinism/debug
            sum := sha256.Sum256(out.Bytes())
            return hex.EncodeToString(sum[:]), exitCode, logs, false
        }
    case <-time.After(timeout):
        _ = cmd.Process.Kill()
        return "", -1, "timeout", false
    }
    if cmd.ProcessState != nil { exitCode = cmd.ProcessState.ExitCode() }
    logs = tailString(out.String(), 2048)
    sum := sha256.Sum256(out.Bytes())
    return hex.EncodeToString(sum[:]), exitCode, logs, true
}

func tailString(s string, max int) string {
    if max <= 0 { return "" }
    if len(s) <= max { return s }
    return s[len(s)-max:]
}
