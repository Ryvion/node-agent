# Node Agent (node-agent)

Go 1.24 cross-platform agent (Linux/macOS/Windows). Runs natively on operator machines, polls hub every 10s. Auto-updates via GitHub Releases. Auto-deploys on push.

## Build & Test

```bash
go build ./...
go test ./...
GOOS=windows go build ./...   # must cross-compile cleanly
GOOS=linux go build ./...
```

## Key Files

- `cmd/ryvion-node/main.go` — startup, registration with retry/backoff, work loop, heartbeat, auto-update
- `internal/runner/oci.go` — OCI container execution (Docker)
- `internal/inference/manager.go` — native llama-server for streaming inference
- `internal/hub/client.go` — hub API client (Ed25519 signed requests)
- `internal/hw/` — hardware detection (GPU, CPU, RAM, VRAM, sensors)
- `internal/nodekey/` — Ed25519 keypair management
- `internal/update/` — auto-update from GitHub Releases
- `internal/blob/` — artifact upload to R2 presigned URLs

## Startup Flow

1. Parse flags (hub URL, device type, country, GPUs, UI port, max GPU util)
2. Load/create Ed25519 keypair from `~/.ryvion/node-key`
3. Detect hardware capabilities
4. Start operator API server (port 45890)
5. Register with hub (retry with exponential backoff — important for Windows where Docker/network aren't ready at boot)
6. Start heartbeat goroutine
7. Enter work loop: poll hub for jobs, execute, submit receipt

## Container Execution (oci.go)

- Security: `--cap-drop=ALL`, `--security-opt=no-new-privileges:true`, `--pids-limit=256`
- Network: `--network=none` by default. `needsNetwork()` allows `--network=bridge` for finetune jobs only (HuggingFace downloads).
- `docker pull` before every run (15min timeout). Falls back to cached image on pull failure.
- Resource limits: 8GB memory (configurable via `RYV_CONTAINER_MEMORY`), 4 CPUs (configurable via `RYV_CONTAINER_CPUS`)
- GPU passthrough: NVIDIA `--gpus` flag or AMD ROCm (`--device=/dev/kfd`, `--device=/dev/dri`)
- Prefetches `payload_url`, `training_data_url`, `audio_url`, `input_url`, `model_url` from job spec into `/work/` before container start
- Upload timeout: 30min for presigned R2 URLs

## Inference Manager (manager.go)

- Downloads llama-server binary + GGUF models to `~/.ryvion/`
- Health check every 5s, auto-restart on crash
- Startup timeout: 120s
- Default context size: 16384 tokens

### Native Models

```
ryvion-llama-3.2-3b  → Llama-3.2-3B-Instruct-Q4_K_M.gguf (bartowski)
phi-4                → phi-4-Q4_K_M.gguf (bartowski)
tinyllama            → tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf (TheBloke)
```

## Auto-Update

Hub advertises latest version via heartbeat response. Node compares semver, downloads from GitHub Release, replaces binary. On Windows: `os.Exit(1)` for SCM restart.

## Operator API

Port 45890 (configurable via `--ui-port`). Endpoints:
- `/api/v1/operator/status` — node status
- `/diagnostics` — system diagnostics
- `/logs` — recent logs
- `/jobs` — job history

## Data Directory

`~/.ryvion/` contains: `bin/` (llama-server), `models/` (GGUF files), `config.json`, `node-key` (Ed25519 keypair)

## Platform-Specific

- `diskcheck_linux.go` uses `syscall.Statfs`; `diskcheck_other.go` is a no-op
- Windows timezone detection via PowerShell
- Windows service: detected via `isWindowsService()`, runs through SCM integration
- Registration retries with backoff — critical for Windows where Docker/WSL2/network start late

## Ed25519 Signing

All API calls to hub are signed with node keypair (public key as hex identifier).

## Common Gotchas

- `needsNetwork()` in oci.go: only finetune (`task == "finetune"`) gets bridge mode
- Windows: containerized inference ALWAYS fails (exit 137 OOM) — use native only
- The `min()` function is defined in oci.go (needed for Go <1.21 compat, though module is now 1.24)
- `flagMaxGPUUtil` (default 90%) — skips jobs when GPU is busy. Env override: `RYV_MAX_GPU_UTIL`
- `cleanupOrphanedContainers()` runs at startup to clean stale `ryv_*` containers
- `ensureDockerGPURuntime()` runs at startup to verify Docker GPU availability
