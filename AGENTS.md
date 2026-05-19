# Codex Agent Instructions — ryvion-node

## Project Context

Run `mempalace search "node agent llama cpp ai work"` for codebase context.

## Architecture

Go 1.24 cross-platform agent. Runs natively on trusted machines and executes assigned AI work.

- `cmd/ryvion-node/main.go` — startup, work loop, heartbeat
- `internal/runner/oci.go` — OCI container execution
- `internal/llamacpp/` — local llama.cpp inference execution
- `internal/hub/client.go` — hub API client (Ed25519 signed)
- `internal/hw/` — hardware detection

## Key Rules

- Build: `go build ./...` must pass for Linux, macOS, AND Windows
- Cross-compile check: `GOOS=windows go build ./...`
- Zero external dependencies (Go stdlib + x/sys only)
- Container security: --cap-drop=ALL, --network=none for managed OCI jobs
- Keep AI inference local and explicit through llama.cpp; do not add V7/V8,
  benchmark-plane, model-warm, speculative, or mesh code to the active node path
- Archive inactive surfaces in `ryvion-archive`; never import archive code back into production repos
- Commits: Keep messages SHORT, no Co-Authored-By
