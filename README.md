# Ryvion Node Agent

The Ryvion node agent turns any machine with a GPU into a compute node on the [Ryvion](https://ryvion.com) distributed inference network.

It registers with the hub orchestrator, sends signed heartbeats, polls for jobs, runs OCI container workloads through the managed execution runtime, and submits cryptographically signed receipts.

## Quickstart

```bash
# Download the latest release for your platform
curl -L https://github.com/Ryvion/ryvion-node/releases/latest/download/ryvion-node-linux-amd64 -o ryvion-node
chmod +x ryvion-node

# Start the node (generates an Ed25519 key on first run)
./ryvion-node -hub https://api.ryvion.ai
```

The node will:
1. Generate an Ed25519 keypair (stored in `~/.ryvion/node-key`)
2. Register with the hub and begin sending heartbeats
3. Report detected hardware without installing runtimes or downloading models
4. Poll for explicitly enabled workload classes and submit signed receipts

Fresh nodes are resource-neutral: public AI work, runtime auto-sync, model
downloads, and model prewarming are disabled until the operator enables them.

## Requirements

- Linux (amd64) with the Ryvion managed execution runtime installed
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) when the runtime backend is configured for NVIDIA GPU workloads
- CPU-only mode works without a GPU

## Configuration

All configuration is via flags or environment variables:

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `-hub` | `RYV_HUB_URL` | `https://api.ryvion.ai` | Hub orchestrator URL |
| `-device` | `RYV_DEVICE_TYPE` | auto-detected | Device type: `gpu`, `cpu`, `mobile`, `iot` |
| `-gpus` | `RYV_GPUS` | auto-detected | GPU configuration |
| `-country` | `RYV_DECLARED_COUNTRY` | — | ISO country code for jurisdiction routing |
| — | `RYV_KEY_PATH` | `~/.ryvion/node-key` | Path to Ed25519 node key |
| — | `RYV_DATA_DIR` | `~/.ryvion` | Ryvion-owned binaries and native model data |
| `-bind-token` | `RYV_BIND_TOKEN` | — | Token to bind node to a specific account |
| `-ui-port` | `RYV_UI_PORT` | `0` | Local status UI port (0 = disabled) |
| — | `RYV_CONTAINER_CPUS` | — | CPU limit for containers |
| — | `RYV_CONTAINER_MEMORY` | — | Memory limit for containers |
| — | `RYV_JOB_TIMEOUT` | `10m` | Maximum job execution time |
| — | `RYV_MAX_GPU_UTIL` | — | GPU utilization threshold |
| — | `RYV_LOG_LEVEL` | `info` | Log level: `debug`, `info`, `warn`, `error` |
| — | `RYV_PUBLIC_AI` | `0` | Explicitly allow buyer-facing AI workloads |
| — | `RYV_RUNTIME_AUTO_SYNC` | `0` | Explicitly allow background managed-runtime repair/update after registration |
| — | `RYV_MODEL_AUTO_DOWNLOAD` | `0` | Explicitly allow hub-requested large model preparation |
| — | `RYV_MODEL_PREWARM_MODE` | `off` | Startup model prewarm: `off`, `lean`, or `all` |
| — | `RYV_PREWARM_MODELS` | — | Explicit comma-separated model IDs to prewarm |
| — | `RYV_MODEL_MAX_CACHE_GB` | `50` | Maximum native managed model-cache size |

Native downloads preserve at least 10 GiB of free disk on Linux, macOS, and
Windows. The configured cache limit does not enable hub-requested automatic
downloads; those still require `RYV_MODEL_AUTO_DOWNLOAD=1`.

## Building from source

```bash
go build -o ryvion-node ./cmd/ryvion-node
```

## Architecture

```
ryvion-node/
  cmd/ryvion-node/     Entry point
  internal/
    hub/               Typed API client for hub endpoints
    hw/                Hardware detection + metrics sampling
    runner/            OCI container workload execution
    blob/              Artifact upload flow
    nodekey/           Ed25519 key management
    inference/         Native llama.cpp inference (streaming)
    update/            Signed auto-update (SHA256SUMS + Ed25519 sig)
```

**Job lifecycle:**
1. Node polls hub for assigned jobs
2. Hub returns a job with container image + parameters
3. Node pulls the OCI image and runs it with GPU passthrough
4. Container reads `/work/job.json`, writes `/work/output.json` + `/work/receipt.json`
5. Node uploads artifacts and submits a signed receipt to the hub

## Auto-updates

The node agent supports signed auto-updates. When a new release is published, running nodes will download and verify the update using Ed25519 signatures before applying it.

## License

Business Source License 1.1 — see [LICENSE](LICENSE).

You can use, modify, and run this software freely. You cannot use it to operate a competing distributed GPU compute network. The code converts to Apache 2.0 on April 4, 2030.
