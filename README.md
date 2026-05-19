# Ryvion Node Agent

The Ryvion node agent turns trusted machines into workers for Ryvion local AI work.

It registers with the hub, sends signed heartbeats, receives work, runs local llama.cpp inference or managed OCI workloads, uploads artifacts, and submits signed receipts.

## Quickstart

```bash
# Download the latest release for your platform
curl -L https://github.com/Ryvion/ryvion-node/releases/latest/download/ryvion-node-linux-amd64 -o ryvion-node
chmod +x ryvion-node

# Start the node (generates an Ed25519 key on first run)
./ryvion-node -hub https://api.ryvion.ai
```

The node will:
1. Generate an Ed25519 keypair (stored in `~/.ryvion/node.key`)
2. Register with the hub and begin sending heartbeats
3. Poll for jobs and execute them through local llama.cpp or the managed OCI runtime
4. Submit signed receipts for completed work

## Requirements

- Linux, macOS, or Windows for the agent binary
- Local [llama.cpp](https://github.com/ggml-org/llama.cpp) server for native inference jobs
- Linux worker hosts for managed OCI execution
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) when OCI workloads need GPU acceleration
- CPU-only llama.cpp jobs work without a GPU when the local server supports the model

## Configuration

All configuration is via flags or environment variables:

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `-hub` | `RYV_HUB_URL` | `https://api.ryvion.ai` | Hub orchestrator URL |
| `-device` | `RYV_DEVICE` | auto-detected | Device type: `gpu` or `cpu` |
| `-gpus` | `RYV_GPUS` | auto-detected | GPU configuration |
| `-country` | `RYV_COUNTRY` | — | ISO country code for jurisdiction routing |
| `-key` | `RYV_KEY_PATH` | `~/.ryvion/node.key` | Path to Ed25519 node key |
| `-data` | `RYV_DATA_DIR` | `~/.ryvion/data` | Working directory for job artifacts |
| `-bind-token` | `RYV_BIND_TOKEN` | — | Token to bind node to a specific account |
| `-ui-port` | `RYV_UI_PORT` | `0` | Local status UI port (0 = disabled) |
| — | `RYV_CONTAINER_CPUS` | — | CPU limit for containers |
| — | `RYV_CONTAINER_MEMORY` | — | Memory limit for containers |
| — | `RYV_LLAMA_CPP_SERVER_URL` | — | Local llama.cpp server URL, for example `http://127.0.0.1:8080` |
| — | `RYV_LLAMA_CPP_MODEL` | — | Model id sent to `/v1/chat/completions` when a job does not specify one |
| — | `RYV_LLAMA_CPP_PROBE_TIMEOUT` | `2s` | llama.cpp health probe timeout |
| — | `RYV_LLAMA_CPP_HTTP_TIMEOUT` | `10m` | llama.cpp inference request timeout |
| — | `RYV_JOB_TIMEOUT` | `10m` | Maximum job execution time |
| — | `RYV_MAX_GPU_UTIL` | — | GPU utilization threshold |
| — | `RYV_LOG_LEVEL` | `info` | Log level: `debug`, `info`, `warn`, `error` |

`RYV_DEVICE_TYPE` and `-type` are still accepted as deprecated aliases for older launch scripts; new deployments should use `RYV_DEVICE` and `-device`.

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
    llamacpp/          Local llama.cpp client, spec validation, artifact writer
    runner/            OCI container workload execution
    blob/              Artifact upload flow
    nodekey/           Ed25519 key management
    update/            Signed auto-update (SHA256SUMS + Ed25519 sig)
```

**Job lifecycle:**
1. Node polls hub for assigned jobs
2. Hub returns a job with `executor_kind=llama_cpp` or a managed OCI image
3. llama.cpp jobs call the local OpenAI-compatible server and write a JSON artifact
4. OCI jobs pull the image and run with the configured isolation policy
5. Node uploads artifacts and submits a signed receipt to the hub without raw prompt/output in metadata

## Auto-updates

The node agent supports signed auto-updates. When a new release is published, running nodes will download and verify the update using Ed25519 signatures before applying it.

## License

Business Source License 1.1 — see [LICENSE](LICENSE).

You can use, modify, and run this software freely. You cannot use it to operate a competing distributed GPU compute network. The code converts to Apache 2.0 on April 4, 2030.
