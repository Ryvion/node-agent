# V7 Task Index - node-agent

This index preserves the planned V7 node-agent task sequence locally so Codex can work without relying on a separate docs repository or hub-orch checkout.

## Current

- TASK-NODE-001 - Add V7 node-agent repo context. Current context setup for local architecture, task boundaries, and roadmap continuity.

## Planned

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

## Task Rules

Each V7 task should be isolated, additive, and feature-flagged when it touches runtime behavior. Future implementation tasks should preserve the hub-orch boundary from `V7_REPO_CONTEXT.md`: node-agent provides local capability, evidence, and execution support, while hub-orch owns planning, placement, graph, receipt, and control-plane systems.
