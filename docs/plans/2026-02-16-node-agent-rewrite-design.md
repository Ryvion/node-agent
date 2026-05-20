# Node-Agent Rewrite Design

## Context

The current node-agent is ~4,500 LOC across 23 Go files with significant dead code (~2,255 lines of unused Docker executors), duplicate implementations (two artifact upload paths), a 595-line monolith (`agent.go`), and scattered concerns. This rewrite reduces it to ~740 LOC across 6 Go files with clean separation of concerns.

## Scope

**Keep:** Register, heartbeat, fetch work, OCI execution, artifact upload, receipt submission.
**Cut:** Tray UI, auto-update, native ffmpeg/embed executors, simulation fallback, `executor/executor/` package, `pkg/storage/`, config file parsing.

## Directory Structure

```
node-agent/
├── cmd/ryvion-node/main.go      # ~80 lines. Flags, slog, signal, run loop.
├── internal/
│   ├── nodekey/nodekey.go        # ~40 lines. Load/generate ed25519 keypair.
│   ├── hub/client.go             # ~250 lines. Typed HTTP client for all hub APIs.
│   ├── hw/hw.go                  # ~150 lines. Hardware detection (CPU, RAM, GPU, power).
│   ├── runner/oci.go             # ~120 lines. Docker container execution.
│   └── blob/upload.go            # ~100 lines. Artifact upload via presigned URLs.
├── deploy/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── digitalocean-setup.sh
├── go.mod
└── go.sum
```

## Package Contracts

### `cmd/ryvion-node/main.go`
- Parses flags (`-hub`, `-type`, `-referral`) and env vars (`RYV_HUB_URL`, `RYV_KEY_PATH`, `RYV_WORK_DIR`, `RYV_BIND_TOKEN`, `RYV_WALLET`)
- Initializes slog JSON handler
- Loads keypair via `nodekey.LoadOrCreate`
- Creates `hub.Client`
- Registers, then loops: heartbeat → fetch → execute → upload → receipt
- Signal handling via `signal.NotifyContext`
- Only place that calls `os.Exit`

### `internal/hub/client.go`
- `Client` struct: baseURL, pub, priv, http.Client, optional bind headers
- One public method per hub endpoint (Register, Heartbeat, FetchWork, SubmitReceipt, SavePayout, SolveChallenge, SendHealthReport, PrepareUpload, PresignManifest)
- Private: `sign(parts ...string) []byte`, `post(ctx, path, body, out) error`, `pubHex() string`
- All request/response types are named structs
- No `map[string]any`

### `internal/hw/hw.go`
- `DetectCaps(deviceType string) Caps` — CPU cores, RAM, GPU model, VRAM, sensors, bandwidth
- `SampleMetrics() Metrics` — CPU/Mem/GPU utilization, power watts
- Returns zero values when detection fails (no random numbers)

### `internal/runner/oci.go`
- `Run(ctx, image, specJSON, gpuMode) (*Result, error)`
- Creates temp workdir, writes job.json, runs `docker run`, reads receipt.json + metrics.json
- Returns `Result{Hash, Duration, ExitCode, Logs, Metrics, OutputPath}`

### `internal/blob/upload.go`
- `Upload(ctx, hub.Client, jobID, filePath) (url, key, hash, error)`
- Gets presigned URL from hub, PUTs file, uploads signed manifest

### `internal/nodekey/nodekey.go`
- `LoadOrCreate(path string) (ed25519.PublicKey, ed25519.PrivateKey, error)`
- Returns error, never panics

## Error Philosophy
- Packages return errors. Only `main()` calls `os.Exit`.
- Registration failure is fatal. Heartbeat/work failures are warnings.
- Zero silent error discards.

## Signing Protocol
All signatures use: `SHA256("AKT1|" + verb + "|" + fields...) → ed25519.Sign(priv, hash)`.
Single `sign(parts ...string)` method replaces 8 separate message-building functions.
