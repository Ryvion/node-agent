# Ryvion Node Agent

The Ryvion node agent turns any machine with a GPU into a compute node on the [Ryvion](https://ryvion.com) distributed inference network.

It registers with the hub orchestrator, sends signed heartbeats, polls for jobs, runs OCI container workloads via Docker, and submits cryptographically signed receipts.

## Quickstart

```bash
# Download the latest release for your platform
curl -L https://github.com/Ryvion/node-agent/releases/latest/download/ryvion-node-linux-amd64 -o ryvion-node
chmod +x ryvion-node

# Start the node (generates an Ed25519 key on first run)
./ryvion-node -hub https://ryvion-hub.fly.dev
```

The node will:
1. Generate an Ed25519 keypair (stored in `~/.ryvion/node.key`)
2. Register with the hub and begin sending heartbeats
3. Poll for jobs and execute them in Docker containers
4. Submit signed receipts for completed work

## Requirements

- Linux (amd64) with Docker installed
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU workloads
- CPU-only mode works without a GPU

## Configuration

All configuration is via flags or environment variables:

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `-hub` | `RYV_HUB_URL` | `https://ryvion-hub.fly.dev` | Hub orchestrator URL |
| `-device` | `RYV_DEVICE_TYPE` | auto-detected | Device type: `gpu`, `cpu`, `mobile`, `iot` |
| `-gpus` | `RYV_GPUS` | auto-detected | GPU configuration |
| `-country` | `RYV_DECLARED_COUNTRY` | — | ISO country code for jurisdiction routing |
| `-key` | `RYV_KEY_PATH` | `~/.ryvion/node.key` | Path to Ed25519 node key |
| `-data` | `RYV_DATA_DIR` | `~/.ryvion/data` | Working directory for job artifacts |
| `-bind-token` | `RYV_BIND_TOKEN` | — | Token to bind node to a specific account |
| `-ui-port` | `RYV_UI_PORT` | `0` | Local status UI port (0 = disabled) |
| — | `RYV_CONTAINER_CPUS` | — | CPU limit for containers |
| — | `RYV_CONTAINER_MEMORY` | — | Memory limit for containers |
| — | `RYV_JOB_TIMEOUT` | `10m` | Maximum job execution time |
| — | `RYV_MAX_GPU_UTIL` | — | GPU utilization threshold |
| — | `RYV_LOG_LEVEL` | `info` | Log level: `debug`, `info`, `warn`, `error` |

## Building from source

```bash
go build -o ryvion-node ./cmd/ryvion-node
```

## Architecture

```
node-agent/
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
