Ryvion Node (Go)

What it does
- Loads or creates an Ed25519 node key
- Registers with Hub Orchestrator and sends signed heartbeats
- Polls for work, runs OCI workloads via Docker, uploads artifacts, and submits signed receipts

Entry point
- `cmd/ryvion-node/main.go`

Core packages
- `internal/hub`: typed API client for hub endpoints
- `internal/hw`: hardware detection + metrics sampling
- `internal/runner`: OCI workload execution
- `internal/blob`: artifact upload flow
- `internal/nodekey`: key management
