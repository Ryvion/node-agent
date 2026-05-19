# Ryvion AI Node Agent - DigitalOcean Deployment

## Quick Start

1. Create an Ubuntu 22.04 LTS droplet.
2. Run the setup script:

```bash
curl -fsSL https://raw.githubusercontent.com/Ryvion/ryvion-node/main/deploy/digitalocean-setup.sh | bash
systemctl start ryvion-node
systemctl status ryvion-node
```

3. Configure a local llama.cpp server when the node should accept native
   inference jobs:

```bash
export RYV_LLAMA_CPP_SERVER_URL=http://127.0.0.1:8080
export RYV_LLAMA_CPP_MODEL=local-model
```

## Runtime Model

The node reports two active execution lanes:

- `llama_cpp`: local OpenAI-compatible llama.cpp server for AI inference jobs.
- `managed_oci`: isolated container execution for trusted custom/code workloads.

Managed OCI jobs run with network isolation by default. Native llama.cpp jobs do
not send prompts or generated text in receipt metadata; the node records hashes,
token counts, model id, timing, and uploaded artifact references.

## Verification

```bash
docker ps
docker-compose -f /opt/ryvion/docker-compose.yml logs --tail=100
docker exec ryvion-node pgrep ryvion-node
```

Connectivity check:

```bash
curl -I https://api.ryvion.ai/health
```

## GPU Hosts

GPU hosts can use NVIDIA Container Toolkit for managed OCI workloads. llama.cpp
GPU acceleration is configured in the local llama.cpp server itself; the node
only probes the local server and advertises readiness when it is reachable.
