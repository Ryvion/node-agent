# Codex Agent Instructions — node-agent

## Project Context

Run `mempalace search "node agent runner"` for codebase context.

## Architecture

Go 1.24 cross-platform agent. Runs natively on operator machines, polls hub for work.

- `cmd/ryvion-node/main.go` — startup, work loop, heartbeat
- `internal/runner/oci.go` — OCI container execution
- `internal/runner/agent.go` — long-running agent container support
- `internal/inference/manager.go` — native llama-server lifecycle
- `internal/hub/client.go` — hub API client (Ed25519 signed)
- `internal/hw/` — hardware detection

## Key Rules

- Build: `go build ./...` must pass for Linux, macOS, AND Windows
- Cross-compile check: `GOOS=windows go build ./...`
- Zero external dependencies (Go stdlib + x/sys only)
- Windows: ALWAYS use native llama-server (Docker GPU unreliable)
- Container security: --cap-drop=ALL, --network=none (except finetune/agent_hosting → bridge)
- Commits: Keep messages SHORT, no Co-Authored-By
