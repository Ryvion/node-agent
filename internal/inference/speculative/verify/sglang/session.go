package sglang

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

var LookPath = exec.LookPath

type CommandSpec struct {
	Name     string
	Args     []string
	Shell    bool
	Original string
}

func ResolveVerifierCommand() (CommandSpec, bool) {
	if raw := strings.TrimSpace(os.Getenv("RYV_SGLANG_VERIFIER_CMD")); raw != "" {
		return CommandSpec{Shell: true, Original: raw}, true
	}
	if script := strings.TrimSpace(os.Getenv("RYV_SGLANG_VERIFIER_SCRIPT")); script != "" {
		if python, ok := ResolvePythonCommand(); ok {
			return CommandSpec{Name: python, Args: []string{script}, Original: python + " " + script}, true
		}
	}
	for _, name := range []string{
		"ryvion-verifier-sglang",
		"ryvion-sglang-verifier-v8",
		"sglang-verifier-runner-v8",
	} {
		if path, err := LookPath(name); err == nil && strings.TrimSpace(path) != "" {
			return CommandSpec{Name: path, Original: path}, true
		}
	}
	for _, script := range BundledScripts() {
		if _, err := os.Stat(script); err == nil {
			if python, ok := ResolvePythonCommand(); ok {
				return CommandSpec{Name: python, Args: []string{script}, Original: python + " " + script}, true
			}
		}
	}
	return CommandSpec{}, false
}

func BundledScripts() []string {
	var out []string
	if exe, err := os.Executable(); err == nil {
		base := filepath.Dir(exe)
		out = append(out,
			filepath.Join(base, "runtimes", "verifier", "sglang", "run.py"),
			filepath.Join(base, "ryvion-runtimes", "runtimes", "verifier", "sglang", "run.py"),
			filepath.Join(base, "sglang-verifier-runner-v8", "run.py"),
			filepath.Join(base, "runners", "sglang-verifier-runner-v8", "run.py"),
			filepath.Join(base, "resources", "sglang-verifier-runner-v8", "run.py"),
		)
	}
	return out
}

func ResolvePythonCommand() (string, bool) {
	if raw := strings.TrimSpace(os.Getenv("RYV_PYTHON")); raw != "" {
		return raw, true
	}
	for _, name := range []string{"python3", "python"} {
		if path, err := LookPath(name); err == nil && strings.TrimSpace(path) != "" {
			return path, true
		}
	}
	return "", false
}

func Command(ctx context.Context, spec CommandSpec) *exec.Cmd {
	if spec.Shell {
		if runtime.GOOS == "windows" {
			return exec.CommandContext(ctx, "cmd", "/C", spec.Original)
		}
		return exec.CommandContext(ctx, "sh", "-c", spec.Original)
	}
	return exec.CommandContext(ctx, spec.Name, spec.Args...)
}

func WaitForSocket(ctx context.Context, socketPath string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		if _, err := os.Stat(socketPath); err == nil {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("native SGLang verifier socket was not created within %s", timeout)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(100 * time.Millisecond):
		}
	}
}

func RPC(ctx context.Context, socketPath string, method string, params map[string]any) (map[string]any, error) {
	var dialer net.Dialer
	conn, err := dialer.DialContext(ctx, "unix", socketPath)
	if err != nil {
		return nil, err
	}
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(5 * time.Minute))
	req := map[string]any{"jsonrpc": "2.0", "id": shortHash(fmt.Sprintf("%s|%d", method, time.Now().UnixNano())), "method": method, "params": params}
	raw, _ := json.Marshal(req)
	if _, err := conn.Write(append(raw, '\n')); err != nil {
		return nil, err
	}
	line, err := bufio.NewReader(conn).ReadBytes('\n')
	if err != nil {
		return nil, err
	}
	var resp map[string]any
	if err := json.Unmarshal(line, &resp); err != nil {
		return nil, err
	}
	if rpcErr := mapFromAny(resp["error"]); len(rpcErr) > 0 {
		return nil, fmt.Errorf("native SGLang verifier RPC %s failed: %s", method, stringValue(rpcErr["message"]))
	}
	return mapFromAny(resp["result"]), nil
}

func StopCommand(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	done := make(chan struct{})
	go func() {
		_ = cmd.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(500 * time.Millisecond):
		_ = cmd.Process.Kill()
		<-done
	}
}

func shortHash(value string) string {
	sum := 0
	for _, r := range value {
		sum = (sum*31 + int(r)) & 0x7fffffff
	}
	return fmt.Sprintf("%08x", sum)
}

func mapFromAny(value any) map[string]any {
	if typed, ok := value.(map[string]any); ok {
		return typed
	}
	return map[string]any{}
}

func stringValue(value any) string {
	switch typed := value.(type) {
	case string:
		return typed
	default:
		return strings.TrimSpace(fmt.Sprint(typed))
	}
}
