package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"strings"

	"github.com/Ryvion/node-agent/internal/blob"
	"github.com/Ryvion/node-agent/internal/hub"
	"github.com/Ryvion/node-agent/internal/inference"
	"github.com/Ryvion/node-agent/internal/runner"
)

type executionContext struct {
	client         *hub.Client
	gpus           string
	infMgr         *inference.Manager
	runtimeManager *runtimeManager
	gpuDetected    bool
}

type executionEngine interface {
	Kind() string
	Execute(context.Context, *hub.WorkAssignment, executionContext) (*runnerResultSnapshot, error)
}

type streamingEngine struct{}
type managedOCIEngine struct{}
type agentHostingEngine struct{}

func (streamingEngine) Kind() string  { return executorKindNativeStreaming }
func (managedOCIEngine) Kind() string { return executorKindManagedOCI }
func (agentHostingEngine) Kind() string {
	return executorKindAgentHosting
}

func selectExecutionEngine(work *hub.WorkAssignment) executionEngine {
	switch executorKindForAssignment(work) {
	case executorKindNativeStreaming:
		return streamingEngine{}
	case executorKindAgentHosting:
		return agentHostingEngine{}
	default:
		return managedOCIEngine{}
	}
}

func executorKindForAssignment(work *hub.WorkAssignment) string {
	if work == nil {
		return executorKindManagedOCI
	}
	if kind := strings.TrimSpace(work.ExecutorKind); kind != "" {
		return kind
	}
	if strings.EqualFold(strings.TrimSpace(work.Kind), executorKindAgentHosting) || isAgentHostingTask(work.SpecJSON) {
		return executorKindAgentHosting
	}
	if strings.EqualFold(strings.TrimSpace(work.Image), "streaming") {
		return executorKindNativeStreaming
	}
	return executorKindManagedOCI
}

func assuranceClassForAssignment(work *hub.WorkAssignment) string {
	if work == nil {
		return assuranceClassVerifiedGateway
	}
	if v := strings.TrimSpace(work.AssuranceClass); v != "" {
		return v
	}
	return assuranceClassVerifiedGateway
}

func (streamingEngine) Execute(ctx context.Context, work *hub.WorkAssignment, execCtx executionContext) (*runnerResultSnapshot, error) {
	if execCtx.infMgr == nil || !execCtx.infMgr.Healthy() {
		err := fmt.Errorf("inference manager is not healthy")
		relayStreamingFailure(ctx, execCtx.client, work.JobID, err)
		return nil, err
	}
	// Phase 1c.2: if the hub tagged the spec as task=embedding, run a
	// one-shot embedding through llama-server's /v1/embeddings and submit
	// the vector inline in the receipt. No SSE relay required — the hub
	// polls receipts and returns the vector to the buyer synchronously.
	if inference.IsEmbeddingJob(work.SpecJSON) {
		if err := execCtx.infMgr.RunEmbeddingJob(ctx, execCtx.client, work.JobID, work.SpecJSON); err != nil {
			return nil, err
		}
		return &runnerResultSnapshot{
			MeteringUnits: 1,
			Metadata: receiptMetadataBase(
				work,
				execCtx.runtimeManager.ReceiptMetadata(execCtx.gpuDetected),
				map[string]any{"executor": "llama-server", "task": "embedding"},
			),
		}, nil
	}
	if err := execCtx.infMgr.RunStreamingJob(ctx, execCtx.client, work.JobID, work.SpecJSON); err != nil {
		relayStreamingFailure(ctx, execCtx.client, work.JobID, err)
		return nil, err
	}
	return &runnerResultSnapshot{
		MeteringUnits: 1,
		Metadata: receiptMetadataBase(
			work,
			execCtx.runtimeManager.ReceiptMetadata(execCtx.gpuDetected),
			map[string]any{"executor": "llama-server"},
		),
	}, nil
}

func (agentHostingEngine) Execute(ctx context.Context, work *hub.WorkAssignment, execCtx executionContext) (*runnerResultSnapshot, error) {
	healthFn := func(uptimeSeconds int) bool {
		resp, err := execCtx.client.ReportAgentHealth(ctx, extractDeploymentID(work.SpecJSON), uptimeSeconds)
		if err != nil {
			return false
		}
		return resp.ShouldStop
	}

	result, runErr := runner.RunAgent(ctx, work.Image, work.SpecJSON, execCtx.gpus, healthFn)
	uptimeSeconds := 0
	if result != nil {
		uptimeSeconds = result.UptimeSeconds
	}
	hash := sha256.Sum256([]byte(work.JobID + fmt.Sprintf("%d", uptimeSeconds)))
	metadata := receiptMetadataBase(
		work,
		execCtx.runtimeManager.ReceiptMetadata(execCtx.gpuDetected),
		map[string]any{
			"executor":       "agent_hosting",
			"uptime_seconds": uptimeSeconds,
			"exit_code":      0,
		},
	)
	if result != nil {
		metadata["exit_code"] = result.ExitCode
	}
	if runErr != nil {
		metadata["error"] = runErr.Error()
	}
	receipt := hub.Receipt{
		JobID:         work.JobID,
		ResultHashHex: hex.EncodeToString(hash[:]),
		MeteringUnits: uint64(uptimeSeconds),
		Metadata:      metadata,
	}
	if err := submitReceiptWithRetry(ctx, execCtx.client, receipt); err != nil {
		return &runnerResultSnapshot{
			ResultHashHex: hex.EncodeToString(hash[:]),
			MeteringUnits: uint64(uptimeSeconds),
			Metadata:      metadata,
		}, err
	}
	return &runnerResultSnapshot{
		ResultHashHex: hex.EncodeToString(hash[:]),
		MeteringUnits: uint64(uptimeSeconds),
		Metadata:      metadata,
	}, runErr
}

func (managedOCIEngine) Execute(ctx context.Context, work *hub.WorkAssignment, execCtx executionContext) (*runnerResultSnapshot, error) {
	if strings.TrimSpace(work.Image) == "" || strings.TrimSpace(work.SpecJSON) == "" {
		rejectHash := sha256.Sum256([]byte(work.JobID + ":missing_spec"))
		rejectReceipt := hub.Receipt{
			JobID:         work.JobID,
			ResultHashHex: hex.EncodeToString(rejectHash[:]),
			MeteringUnits: 0,
			Metadata: receiptMetadataBase(
				work,
				execCtx.runtimeManager.ReceiptMetadata(execCtx.gpuDetected),
				map[string]any{
					"executor":  "node_agent",
					"exit_code": 1,
					"error":     "missing container image or spec",
				},
			),
		}
		if err := submitReceiptWithRetry(ctx, execCtx.client, rejectReceipt); err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("missing container image or spec")
	}

	result, runErr := runner.Run(ctx, work.Image, work.SpecJSON, execCtx.gpus)
	if result == nil {
		return nil, runErr
	}
	resultHash := result.Hash
	metadata := receiptMetadataBase(
		work,
		execCtx.runtimeManager.ReceiptMetadata(execCtx.gpuDetected),
		map[string]any{
			"executor":    "oci",
			"duration_ms": result.Duration.Milliseconds(),
			"exit_code":   result.ExitCode,
			"stderr_tail": result.Logs,
			"metrics":     result.Metrics,
		},
	)
	if strings.TrimSpace(result.OutputPath) != "" {
		uploadRes, uploadErr := blob.Upload(ctx, execCtx.client, work.JobID, result.OutputPath)
		if uploadErr == nil {
			metadata["blob_url"] = uploadRes.URL
			metadata["object_key"] = uploadRes.Key
			if strings.TrimSpace(uploadRes.Key) != "" {
				metadata["manifest_key"] = uploadRes.Key + ".manifest.json"
			}
			if strings.TrimSpace(uploadRes.Hash) != "" {
				metadata["artifact_sha256"] = uploadRes.Hash
				resultHash = uploadRes.Hash
			}
		}
		_ = os.Remove(result.OutputPath)
	}
	units := uint64(work.Units)
	if units == 0 {
		units = 1
	}
	receipt := hub.Receipt{
		JobID:         work.JobID,
		ResultHashHex: resultHash,
		MeteringUnits: units,
		Metadata:      metadata,
	}
	if err := submitReceiptWithRetry(ctx, execCtx.client, receipt); err != nil {
		return &runnerResultSnapshot{
			DurationMs:    result.Duration.Milliseconds(),
			ResultHashHex: resultHash,
			ExitCode:      result.ExitCode,
			MeteringUnits: units,
			BlobURL:       stringValue(metadata["blob_url"]),
			ObjectKey:     stringValue(metadata["object_key"]),
			Metadata:      metadata,
		}, err
	}
	return &runnerResultSnapshot{
		DurationMs:    result.Duration.Milliseconds(),
		ResultHashHex: resultHash,
		ExitCode:      result.ExitCode,
		MeteringUnits: units,
		BlobURL:       stringValue(metadata["blob_url"]),
		ObjectKey:     stringValue(metadata["object_key"]),
		Metadata:      metadata,
	}, runErr
}

func receiptMetadataBase(work *hub.WorkAssignment, extras ...map[string]any) map[string]any {
	out := map[string]any{
		"executor_kind":   executorKindForAssignment(work),
		"assurance_class": assuranceClassForAssignment(work),
	}
	for _, extra := range extras {
		for key, value := range extra {
			out[key] = value
		}
	}
	return out
}
