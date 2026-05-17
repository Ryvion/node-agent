# V7 Repo Context - ryvion-node

## V7 Positioning

Ryvion V7 is a verified execution fabric and compute object web, not just a GPU marketplace. The system coordinates work, evidence, receipts, placement, and object identity across hub-controlled planning and node-local runtime capacity.

`ryvion-node` is the single-binary runtime that turns operator hardware into verified V7 execution capacity. It runs on operator machines, reports what the machine can actually do, executes approved local work, and emits evidence that the hub can bind into receipts and graph objects.

## Node-Agent Role

`ryvion-node` should eventually support:

- V7 Capability Passport
- network profile probing
- upload/download/jitter/loss measurements
- local ModelLease state
- model residency / warmup / draining / eviction
- readiness challenge responses
- local CAS for content-addressed artifacts/model chunks/runner layers
- artifact manifest generation
- RYV3 evidence payloads
- safe sandbox policy
- proof-carrying runner output
- V7 heartbeat extension
- consumer-friendly onboarding without Docker as a hard requirement

`ryvion-node` owns local runtime capability, evidence, and execution support. It should keep those responsibilities local and concrete: detect hardware, measure network behavior, manage resident models, run allowlisted workloads, cache content-addressed data, and report verifiable outputs to `ryvion-hub`.

## Ryvion Hub Boundary

`ryvion-hub` already owns the V7 control-plane systems:

- RCOG
- RYV3GraphReceipt
- TranscriptDigest
- ExecutionPlan / RoleSlot
- UploadGate
- RiskGate
- RoleSlotAuction
- Stream events / Rollback
- DraftTicket
- FuzzyTrace
- V7 demo
- Shadow graph / shadow receipt
- Placement / AIMD / CoDel / DRR
- PathOracle / TrafficClass
- Count-Min Sketch / SPRT
- ObjectCDN / FEC
- Evidence frontier / audit sampler / trust certificate / formal specs

`ryvion-node` must not reimplement these `ryvion-hub` control-plane systems. It should provide the node-local facts and execution evidence those systems need: capability passports, path measurements, model residency state, CAS object availability, sandbox decisions, artifact manifests, and proof-carrying runner output.

## Design Laws

- Single-binary friendly first.
- Docker must not be required for consumer onboarding.
- Unsafe model formats must be rejected or sandboxed by default.
- Do not execute PyTorch pickle / arbitrary Python model code by default.
- Prefer GGUF / safetensors / explicit runner allowlists.
- Node must advertise real capabilities, not assumed capabilities.
- Upload is a first-class resource.
- Node must not accept tensor-heavy critical-path roles unless capability and upload budget allow it.
- ModelLease state prevents VRAM thrashing.
- Readiness challenges verify model residency before hot assignment.
- Local CAS stores content-addressed artifacts, not mutable unnamed blobs.
- Evidence payloads must reference CIDs/hashes, not raw mutable data.
- Do not store or expose customer secrets in local evidence.
- RYV3 evidence must bind runtime/model/output/artifact identity.
- V7 heartbeat extension must be additive and backwards-compatible.

## Roadmap

- TASK-NODE-001 - Add V7 ryvion-node repo context
- TASK-NODE-002 - Node Capability Passport
- TASK-NODE-003 - Network Profile Probe
- TASK-NODE-004 - Local ModelLease State Machine
- TASK-NODE-005 - Readiness Challenge Response
- TASK-NODE-006 - Local CAS
- TASK-NODE-007 - Artifact Manifest Generator
- TASK-NODE-008 - Sandbox Policy MVP
- TASK-NODE-009 - RYV3 Evidence Payload
- TASK-NODE-010 - Extend Heartbeat with V7 Capability Passport
- TASK-NODE-011 - Single-binary onboarding UX hardening
- TASK-NODE-012 - Runner allowlist / model format validation
- TASK-NODE-013 - ObjectCDN cache advertisement
- TASK-NODE-014 - Path probe reporting to hub
- TASK-NODE-015 - Model residency telemetry
- TASK-NODE-016 - Proof-carrying runner bridge

## Memory Plane Context

`ryvion-node` eventually participates in the V7 Memory Plane, but only when the hub selects it as eligible for the role and local measurements support the assignment.

- Global PagedAttention applies only if `ryvion-hub` selects the node as eligible.
- Remote Partial Attention applies only when model/KV residency and network budget allow it.
- KV-page and attention roles must be gated by upload budget and topology.
- Consumer nodes should default to upload-light roles: draft, audit, local inference, artifact generation, cache, readiness, and evidence.
- POP, Tier-S, and other high-bandwidth nodes can later handle tensor-heavy or KV-heavy roles.

The default local posture is conservative: advertise measured capability, refuse roles that exceed real local constraints, and produce evidence that binds runtime, model, output, and artifact identity without exposing secrets or mutable raw data.
