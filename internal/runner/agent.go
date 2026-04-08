package runner

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

// AgentResult is returned when the agent container exits.
type AgentResult struct {
	ExitCode      int
	UptimeSeconds int
	Logs          string
}

// RunAgent starts a long-running agent container and monitors it.
// It blocks until the context is cancelled or the container exits.
// The healthFn is called periodically with the current uptime. If it returns
// true, RunAgent stops the container and returns.
func RunAgent(ctx context.Context, image, specJSON, gpus string, healthFn func(uptimeSeconds int) bool) (*AgentResult, error) {
	if err := validateAgentImageRef(image); err != nil {
		return nil, fmt.Errorf("invalid agent image: %w", err)
	}
	if err := verifyAgentImageSignature(ctx, image); err != nil {
		return nil, fmt.Errorf("verify agent image signature: %w", err)
	}

	// 1. Parse agent spec
	var spec struct {
		Task         string            `json:"task"`
		DeploymentID string            `json:"deployment_id"`
		HubURL       string            `json:"hub_url"`
		KBIDs        string            `json:"kb_ids"`
		Model        string            `json:"model"`
		EnvVars      map[string]string `json:"env_vars"`
	}
	if err := json.Unmarshal([]byte(specJSON), &spec); err != nil {
		return nil, fmt.Errorf("parse agent spec: %w", err)
	}

	// 2. Create work directory
	workBase := resolveWorkBase(runtime.GOOS, os.Getenv)
	if workBase != "" {
		if err := os.MkdirAll(workBase, 0o755); err != nil {
			return nil, fmt.Errorf("create work dir: %w", err)
		}
	}
	workDir, err := os.MkdirTemp(workBase, "ryv_agent_*")
	if err != nil {
		return nil, fmt.Errorf("create temp work dir: %w", err)
	}
	defer os.RemoveAll(workDir)

	// Write job spec
	if err := os.WriteFile(filepath.Join(workDir, "job.json"), []byte(specJSON), 0o644); err != nil {
		return nil, fmt.Errorf("write job.json: %w", err)
	}

	// 3. Resolve docker binary
	dockerBin, err := resolveDocker()
	if err != nil {
		return nil, fmt.Errorf("docker not found: %w", err)
	}

	// 4. Pull image
	slog.Info("agent: pulling image", "image", image)
	pullCtx, pullCancel := context.WithTimeout(ctx, 15*time.Minute)
	pullCmd := exec.CommandContext(pullCtx, dockerBin, "pull", image)
	pullCmd.Stdout = io.Discard
	pullCmd.Stderr = io.Discard
	if err := pullCmd.Run(); err != nil {
		slog.Warn("agent: image pull failed, using cached", "image", image, "error", err)
	}
	pullCancel()

	// 5. Build docker run args
	name := "ryv_agent_" + spec.DeploymentID

	memLimit := strings.TrimSpace(os.Getenv("RYV_CONTAINER_MEMORY"))
	if memLimit == "" {
		memLimit = "8g"
	}
	cpuLimit := strings.TrimSpace(os.Getenv("RYV_CONTAINER_CPUS"))
	if cpuLimit == "" {
		cpuLimit = "4"
	}

	args := []string{
		"run", "--name", name, "--rm",
		"-v", workDir + ":/work",
		"--memory", memLimit, "--memory-swap", memLimit,
		"--cpus", cpuLimit, "--pids-limit", "256",
		"--cap-drop=ALL",
		"--security-opt=no-new-privileges:true",
		"--network=bridge", // Agents need hub API access
	}

	// GPU passthrough (reuse existing helpers from oci.go)
	if gpuArg := resolveGPUFlag(gpus); gpuArg != "" {
		args = append(args, "--gpus", gpuArg)
	} else if gpus == "auto" && isROCmAvailable() {
		args = append(args, "--device=/dev/kfd", "--device=/dev/dri", "--group-add=video")
	}

	// Environment variables: inject hub URL, deployment ID, KB IDs, model
	args = append(args, "-e", "RYVION_HUB_URL="+spec.HubURL)
	args = append(args, "-e", "RYVION_DEPLOYMENT_ID="+spec.DeploymentID)
	if spec.KBIDs != "" {
		args = append(args, "-e", "RYVION_KB_IDS="+spec.KBIDs)
	}
	if spec.Model != "" {
		args = append(args, "-e", "RYVION_MODEL="+spec.Model)
	}
	args = append(args, "-e", "RYVION_MCP_URL="+spec.HubURL+"/mcp")

	// Custom env vars from buyer (cannot override RYVION_ prefix)
	for k, v := range spec.EnvVars {
		if !strings.HasPrefix(strings.ToUpper(k), "RYVION_") {
			args = append(args, "-e", k+"="+v)
		}
	}

	args = append(args, image)

	// 6. Start container
	slog.Info("agent: starting container", "name", name, "image", image, "deployment_id", spec.DeploymentID)
	runCtx, cancelRun := context.WithCancel(ctx)
	defer cancelRun()

	cmd := exec.CommandContext(runCtx, dockerBin, args...)

	// Capture logs (reuse cappedBuffer from oci.go)
	var logBuf cappedBuffer
	cmd.Stdout = &logBuf
	cmd.Stderr = &logBuf

	startTime := time.Now()
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start agent container: %w", err)
	}

	// Ensure container is cleaned up on exit
	defer func() {
		killCtx, killCancel := context.WithTimeout(context.Background(), 30*time.Second)
		_ = exec.CommandContext(killCtx, dockerBin, "stop", "--time", "10", name).Run()
		_ = exec.CommandContext(killCtx, dockerBin, "rm", "-f", name).Run()
		killCancel()
	}()

	// 7. Health monitor goroutine
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		if healthFn != nil && healthFn(0) {
			cancelRun()
			return
		}
		ticker := time.NewTicker(agentHealthInterval())
		defer ticker.Stop()
		for {
			select {
			case <-runCtx.Done():
				return
			case <-ticker.C:
				uptime := int(time.Since(startTime).Seconds())
				if healthFn != nil && healthFn(uptime) {
					cancelRun()
					return
				}
			}
		}
	}()

	// 8. Wait for container to exit (or context cancel)
	err = cmd.Wait()
	exitCode := 0
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		}
	}

	wg.Wait()

	uptime := int(time.Since(startTime).Seconds())
	slog.Info("agent: container exited", "name", name, "exit_code", exitCode, "uptime_seconds", uptime)

	return &AgentResult{
		ExitCode:      exitCode,
		UptimeSeconds: uptime,
		Logs:          logBuf.Tail(65536),
	}, nil
}

func agentHealthInterval() time.Duration {
	raw := strings.TrimSpace(os.Getenv("RYV_AGENT_HEALTH_INTERVAL_SECONDS"))
	if raw == "" {
		return 15 * time.Second
	}
	seconds, err := strconv.Atoi(raw)
	if err != nil {
		return 15 * time.Second
	}
	if seconds < 5 {
		seconds = 5
	}
	if seconds > 300 {
		seconds = 300
	}
	return time.Duration(seconds) * time.Second
}
