package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/Ryvion/ryvion-node/internal/blob"
	"github.com/Ryvion/ryvion-node/internal/hub"
	"github.com/Ryvion/ryvion-node/internal/inference"
	"github.com/Ryvion/ryvion-node/internal/runner"
)

func init() {
	// Surface payload-prefetch download time (GPU-idle-on-network for batch/EM/
	// training jobs) into work-loop diagnostics. See runner.PayloadPrefetchObserver.
	runner.PayloadPrefetchObserver = func(dur time.Duration, totalBytes int64, files int) {
		workLoopDiagnostics.RecordPayloadPrefetch(dur, totalBytes, files)
	}
}

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
type nativeReportEngine struct{}
type managedOCIEngine struct{}
type ryvionRuntimeEngine struct{}
type agentHostingEngine struct{}
type workCapsuleEngine struct{}
type nativeEMEngine struct{}

func (streamingEngine) Kind() string    { return executorKindNativeStreaming }
func (nativeReportEngine) Kind() string { return executorKindNativeReport }
func (managedOCIEngine) Kind() string   { return executorKindManagedOCI }
func (nativeEMEngine) Kind() string     { return executorKindNativeEM }
func (ryvionRuntimeEngine) Kind() string {
	return executorKindRyvionRuntime
}
func (agentHostingEngine) Kind() string {
	return executorKindAgentHosting
}
func (workCapsuleEngine) Kind() string { return executorKindWorkCapsule }

func selectExecutionEngine(work *hub.WorkAssignment) executionEngine {
	switch executorKindForAssignment(work) {
	case executorKindNativeStreaming:
		return streamingEngine{}
	case executorKindNativeReport:
		return nativeReportEngine{}
	case executorKindRyvionRuntime:
		return ryvionRuntimeEngine{}
	case executorKindAgentHosting:
		return agentHostingEngine{}
	case executorKindWorkCapsule:
		return workCapsuleEngine{}
	case executorKindNativeEM:
		return nativeEMEngine{}
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
	if strings.EqualFold(strings.TrimSpace(work.Kind), executorKindWorkCapsule) || isWorkCapsuleTask(work.SpecJSON) {
		return executorKindWorkCapsule
	}
	if strings.EqualFold(strings.TrimSpace(work.Kind), executorKindAgentHosting) || isAgentHostingTask(work.SpecJSON) {
		return executorKindAgentHosting
	}
	if isRyvionRuntimeTask(work.SpecJSON) {
		return executorKindRyvionRuntime
	}
	// EM (FDTD) jobs route to the native runtime when the hub omits a real
	// OCI ContainerImage (native-first) or explicitly tags the spec. A real
	// container image falls through to ManagedOCI (the EM OCI fallback lane).
	if strings.TrimSpace(work.Image) == "" && isNativeEMTask(work.SpecJSON) {
		return executorKindNativeEM
	}
	if strings.EqualFold(strings.TrimSpace(work.Image), executorKindNativeEM) {
		return executorKindNativeEM
	}
	if strings.EqualFold(strings.TrimSpace(work.Image), "streaming") {
		return executorKindNativeStreaming
	}
	if strings.EqualFold(strings.TrimSpace(work.Image), executorKindNativeReport) {
		return executorKindNativeReport
	}
	if strings.EqualFold(strings.TrimSpace(work.Image), "ryvion-runtime") || strings.EqualFold(strings.TrimSpace(work.Image), executorKindRyvionRuntime) {
		return executorKindRyvionRuntime
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
	if execCtx.infMgr != nil {
		if modelName, ok := inference.RequestedNativeModelForSpec(work.SpecJSON); ok {
			if err := execCtx.infMgr.SelectModelForNextStart(modelName); err != nil {
				relayStreamingFailure(ctx, execCtx.client, work.JobID, err)
				return nil, err
			}
		}
	}
	cleanupNative, readyErr := ensureNativeInferenceReadyForJob(ctx, execCtx.infMgr, os.Getenv)
	if cleanupNative != nil {
		defer cleanupNative()
	}
	if readyErr != nil {
		err := nativeInferenceReadinessError(execCtx.infMgr, readyErr)
		relayStreamingFailure(ctx, execCtx.client, work.JobID, err)
		return nil, err
	}
	// Phase 1c.2: if the hub tagged the spec as task=embedding, run a
	// one-shot embedding through llama-server's /v1/embeddings and submit
	// the vector inline in the receipt. No SSE relay required — the hub
	// polls receipts and returns the vector to the buyer synchronously.
	if inference.IsEmbeddingJob(work.SpecJSON) {
		metadata := receiptMetadataBase(
			work,
			execCtx.runtimeManager.ReceiptMetadata(execCtx.gpuDetected),
			map[string]any{"executor": "llama-server", "task": "embedding"},
		)
		if err := execCtx.infMgr.RunEmbeddingJob(ctx, execCtx.client, work.JobID, work.SpecJSON, metadata); err != nil {
			return nil, err
		}
		return &runnerResultSnapshot{
			MeteringUnits: 1,
			Metadata:      metadata,
		}, nil
	}
	metadata := receiptMetadataBase(
		work,
		execCtx.runtimeManager.ReceiptMetadata(execCtx.gpuDetected),
		map[string]any{"executor": "llama-server"},
	)
	if err := execCtx.infMgr.RunStreamingJob(ctx, execCtx.client, work.JobID, work.SpecJSON, metadata); err != nil {
		relayStreamingFailure(ctx, execCtx.client, work.JobID, err)
		return nil, err
	}
	return &runnerResultSnapshot{
		MeteringUnits: 1,
		Metadata:      metadata,
	}, nil
}

func ensureNativeInferenceReadyForJob(ctx context.Context, infMgr *inference.Manager, getenv func(string) string) (func(), error) {
	if infMgr == nil {
		return nil, fmt.Errorf("inference manager is not available")
	}
	if infMgr.Healthy() {
		return nil, nil
	}
	if !nativeInferenceJobLaunchEnabled(getenv) {
		return nil, fmt.Errorf("native inference manager disabled")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	runCtx, cancel := context.WithCancel(ctx)
	errCh := make(chan error, 1)
	go func() {
		errCh <- infMgr.Start(runCtx)
	}()
	cleanup := func() {
		cancel()
		infMgr.Stop()
	}

	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()
	for {
		if infMgr.Healthy() {
			return cleanup, nil
		}
		select {
		case <-ctx.Done():
			cleanup()
			return nil, ctx.Err()
		case err := <-errCh:
			cleanup()
			if err == nil || errors.Is(err, context.Canceled) {
				err = fmt.Errorf("native inference manager stopped before becoming healthy")
			}
			return nil, err
		case <-ticker.C:
		}
	}
}

func nativeInferenceReadinessError(infMgr *inference.Manager, cause error) error {
	if cause == nil {
		cause = fmt.Errorf("inference manager is not healthy")
	}
	reason := inference.BlockerNone
	if infMgr != nil {
		reason = infMgr.BlockerReason()
	}
	if reason == inference.BlockerNone {
		return fmt.Errorf("native inference manager not ready: %w", cause)
	}
	return fmt.Errorf("native inference manager not ready: %w; blocker=%s", cause, reason)
}

func nativeInferenceJobLaunchEnabled(getenv func(string) string) bool {
	if getenv == nil {
		getenv = os.Getenv
	}
	switch strings.ToLower(strings.TrimSpace(getenv(nativeInferenceJobLaunchOffEnv))) {
	case "0", "false", "no", "off", "disabled":
		return true
	case "1", "true", "yes", "on", "enabled":
		return false
	default:
		return true
	}
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

	var result *runner.Result
	var runErr error
	if runner.IsVerifierSessionSpec(work.SpecJSON) || usesVerifierSessionRunnerImage(work.Image) {
		result, runErr = runner.RunVerifierSession(ctx, work.Image, work.SpecJSON, execCtx.gpus)
	} else {
		result, runErr = runner.Run(ctx, work.Image, work.SpecJSON, execCtx.gpus)
	}
	if result == nil {
		return nil, runErr
	}
	resultHash := result.Hash
	metadata := receiptMetadataBase(
		work,
		execCtx.runtimeManager.ReceiptMetadata(execCtx.gpuDetected),
		result.Metadata,
		map[string]any{
			"executor":    "oci",
			"duration_ms": result.Duration.Milliseconds(),
			"exit_code":   result.ExitCode,
			"stderr_tail": result.Logs,
			"metrics":     result.Metrics,
		},
	)
	aborted := managedOCIExecutionAborted(ctx, runErr, result)
	if aborted {
		metadata = annotateManagedOCIAbortReceipt(ctx, metadata, runErr, result)
	}
	if !aborted && len(result.DraftPackets) > 0 {
		metadata["draft_packet_submission"] = submitSpeculativeDraftPackets(ctx, execCtx.client, result.DraftPackets)
	}
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
	submitCtx := ctx
	var submitCancel context.CancelFunc
	if aborted {
		submitCtx, submitCancel = context.WithTimeout(context.Background(), 30*time.Second)
		defer submitCancel()
	}
	if err := submitReceiptWithRetry(submitCtx, execCtx.client, receipt); err != nil {
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

func usesVerifierSessionRunnerImage(image string) bool {
	image = strings.ToLower(strings.TrimSpace(image))
	return strings.Contains(image, "ryvion-verifier-sglang") ||
		strings.Contains(image, "ryvion-verifier-contract-test") ||
		strings.Contains(image, "runtimes/verifier/sglang") ||
		strings.Contains(image, "runtimes/verifier/contract-test")
}

func submitSpeculativeDraftPackets(ctx context.Context, client interface {
	SubmitSpeculativeDraftPacket(context.Context, string, map[string]any) (hub.DraftPacketDecision, error)
	SubmitSpeculativeDraftPacketBatch(context.Context, string, []map[string]any) (hub.DraftPacketBatchDecision, error)
}, packets []map[string]any) map[string]any {
	summary := map[string]any{
		"attempted": len(packets),
		"accepted":  0,
		"failed":    0,
		"rejected":  0,
		"reasons":   map[string]int{},
	}
	if len(packets) == 0 || client == nil {
		return summary
	}
	submitCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	reasons := summary["reasons"].(map[string]int)
	if windowID, ok := commonSpeculativeWindowID(packets); ok {
		decision, err := client.SubmitSpeculativeDraftPacketBatch(submitCtx, windowID, packets)
		if err == nil {
			summary["accepted"] = decision.Accepted
			summary["rejected"] = decision.Rejected
			if decision.Attempted > 0 {
				summary["attempted"] = decision.Attempted
			}
			for _, packetDecision := range decision.Decisions {
				reason := strings.TrimSpace(packetDecision.Reason)
				if reason == "" {
					reason = "unknown"
				}
				reasons[reason]++
			}
			return summary
		}
		reasons["batch_submit_failed"]++
	}
	for _, packet := range packets {
		windowID := stringValue(packet["window_id"])
		decision, err := client.SubmitSpeculativeDraftPacket(submitCtx, windowID, packet)
		if err != nil {
			summary["failed"] = summary["failed"].(int) + 1
			reasons["submit_failed"]++
			continue
		}
		if decision.Accepted {
			summary["accepted"] = summary["accepted"].(int) + 1
		} else {
			summary["rejected"] = summary["rejected"].(int) + 1
		}
		reason := strings.TrimSpace(decision.Reason)
		if reason == "" {
			reason = "unknown"
		}
		reasons[reason]++
	}
	return summary
}

func commonSpeculativeWindowID(packets []map[string]any) (string, bool) {
	var windowID string
	for _, packet := range packets {
		next := strings.TrimSpace(stringValue(packet["window_id"]))
		if next == "" {
			return "", false
		}
		if windowID == "" {
			windowID = next
			continue
		}
		if next != windowID {
			return "", false
		}
	}
	return windowID, windowID != ""
}

func managedOCIExecutionAborted(ctx context.Context, runErr error, result *runner.Result) bool {
	if managedOCIResultDelivered(result) {
		return false
	}
	if ctx != nil && ctx.Err() != nil {
		return true
	}
	return errors.Is(runErr, context.Canceled) || errors.Is(runErr, context.DeadlineExceeded)
}

func managedOCIResultDelivered(result *runner.Result) bool {
	return result != nil && result.ExitCode == 0 && result.ReceiptComplete
}

func annotateManagedOCIAbortReceipt(ctx context.Context, metadata map[string]any, runErr error, result *runner.Result) map[string]any {
	if metadata == nil {
		metadata = map[string]any{}
	}
	if !managedOCIExecutionAborted(ctx, runErr, result) {
		return metadata
	}
	committedBeforeAbort := metadataBool(metadata, "committed_before_abort") || metadataBool(metadata, "accepted_before_abort")
	metadata["status"] = "aborted"
	metadata["execution_status"] = "aborted"
	metadata["abort_reason"] = managedOCIAbortReason(ctx, runErr)
	if _, exists := metadata["accepted_before_abort"]; !exists {
		metadata["accepted_before_abort"] = committedBeforeAbort
	}
	if _, exists := metadata["committed_before_abort"]; !exists {
		metadata["committed_before_abort"] = committedBeforeAbort
	}
	if committedBeforeAbort {
		metadata["billing_status"] = "committed_accepted_work_only"
	} else {
		metadata["billing_status"] = "not_billable_orphaned_compute"
		metadata["accepted_value"] = 0
	}
	return metadata
}

func managedOCIAbortReason(ctx context.Context, runErr error) string {
	if ctx != nil {
		switch {
		case errors.Is(ctx.Err(), context.DeadlineExceeded):
			return "context_deadline_exceeded"
		case errors.Is(ctx.Err(), context.Canceled):
			return "context_canceled"
		}
	}
	switch {
	case errors.Is(runErr, context.DeadlineExceeded):
		return "context_deadline_exceeded"
	case errors.Is(runErr, context.Canceled):
		return "context_canceled"
	default:
		return "abort_requested"
	}
}

func metadataBool(metadata map[string]any, key string) bool {
	if metadata == nil {
		return false
	}
	switch v := metadata[key].(type) {
	case bool:
		return v
	case string:
		return strings.EqualFold(strings.TrimSpace(v), "true") || strings.TrimSpace(v) == "1"
	default:
		return false
	}
}

// workCapsuleEnabled reports whether host-process WorkCapsule execution is
// explicitly opted in. WorkCapsule runs buyer-supplied shell commands directly
// on the operator host with NO container isolation, so it is disabled by
// default and must only be enabled on trusted single-tenant enterprise
// operators via RYV_ENABLE_WORK_CAPSULE.
func workCapsuleEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("RYV_ENABLE_WORK_CAPSULE"))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func (workCapsuleEngine) Execute(ctx context.Context, work *hub.WorkAssignment, execCtx executionContext) (*runnerResultSnapshot, error) {
	if !workCapsuleEnabled() {
		rejectHash := sha256.Sum256([]byte(work.JobID + ":work_capsule_disabled"))
		receipt := hub.Receipt{
			JobID:         work.JobID,
			ResultHashHex: hex.EncodeToString(rejectHash[:]),
			MeteringUnits: 0,
			Metadata: receiptMetadataBase(
				work,
				execCtx.runtimeManager.ReceiptMetadata(execCtx.gpuDetected),
				map[string]any{
					"executor":   executorKindWorkCapsule,
					"work_type":  "certified_change",
					"exit_code":  1,
					"error":      "work_capsule execution disabled on this node; set RYV_ENABLE_WORK_CAPSULE=1 only on trusted enterprise operators",
					"risk_level": "high",
				},
			),
		}
		if err := submitReceiptWithRetry(ctx, execCtx.client, receipt); err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("work_capsule execution disabled on this node")
	}
	if strings.TrimSpace(work.SpecJSON) == "" {
		rejectHash := sha256.Sum256([]byte(work.JobID + ":missing_work_capsule_spec"))
		receipt := hub.Receipt{
			JobID:         work.JobID,
			ResultHashHex: hex.EncodeToString(rejectHash[:]),
			MeteringUnits: 0,
			Metadata: receiptMetadataBase(
				work,
				execCtx.runtimeManager.ReceiptMetadata(execCtx.gpuDetected),
				map[string]any{
					"executor":   executorKindWorkCapsule,
					"work_type":  "certified_change",
					"exit_code":  1,
					"error":      "missing work capsule spec",
					"risk_level": "high",
				},
			),
		}
		if err := submitReceiptWithRetry(ctx, execCtx.client, receipt); err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("missing work capsule spec")
	}
	result, runErr := runner.RunWorkCapsule(ctx, work.SpecJSON)
	if result == nil {
		return nil, runErr
	}
	resultHash := result.Hash
	metadata := receiptMetadataBase(
		work,
		execCtx.runtimeManager.ReceiptMetadata(execCtx.gpuDetected),
		result.Metadata,
		map[string]any{
			"executor":    executorKindWorkCapsule,
			"work_type":   "certified_change",
			"duration_ms": result.Duration.Milliseconds(),
			"exit_code":   result.ExitCode,
			"stderr_tail": result.Logs,
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
	receipt := hub.Receipt{
		JobID:         work.JobID,
		ResultHashHex: resultHash,
		MeteringUnits: 1,
		Metadata:      metadata,
	}
	if err := submitReceiptWithRetry(ctx, execCtx.client, receipt); err != nil {
		return &runnerResultSnapshot{
			DurationMs:    result.Duration.Milliseconds(),
			ResultHashHex: resultHash,
			ExitCode:      result.ExitCode,
			MeteringUnits: 1,
			BlobURL:       stringValue(metadata["blob_url"]),
			ObjectKey:     stringValue(metadata["object_key"]),
			Metadata:      metadata,
		}, err
	}
	return &runnerResultSnapshot{
		DurationMs:    result.Duration.Milliseconds(),
		ResultHashHex: resultHash,
		ExitCode:      result.ExitCode,
		MeteringUnits: 1,
		BlobURL:       stringValue(metadata["blob_url"]),
		ObjectKey:     stringValue(metadata["object_key"]),
		Metadata:      metadata,
	}, runErr
}

// isNativeEMTask reports whether a job spec describes an electromagnetic (FDTD)
// simulation that should run through the native runtime bundle. It mirrors the
// task-detection pattern of isWorkCapsuleTask / isRyvionRuntimeTask.
func isNativeEMTask(specJSON string) bool {
	if strings.TrimSpace(specJSON) == "" {
		return false
	}
	var spec struct {
		Task          string `json:"task"`
		SchemaVersion string `json:"schema_version"`
		ExecutorKind  string `json:"executor_kind"`
	}
	if json.Unmarshal([]byte(specJSON), &spec) != nil {
		return false
	}
	if strings.EqualFold(strings.TrimSpace(spec.ExecutorKind), executorKindNativeEM) {
		return true
	}
	if strings.EqualFold(strings.TrimSpace(spec.Task), "em_simulation") {
		return true
	}
	return strings.HasPrefix(strings.TrimSpace(spec.SchemaVersion), "em.job.")
}

// Execute runs an EM (FDTD) simulation natively: it ensures the signed runtime
// bundle is present, stages the job into a sandboxed working dir, runs the
// engine offline, then reads the SAME result.npz/receipt.json/metrics.json
// contract used by the OCI lane and uploads the (small) artifact via blob.
func (nativeEMEngine) Execute(ctx context.Context, work *hub.WorkAssignment, execCtx executionContext) (*runnerResultSnapshot, error) {
	if strings.TrimSpace(work.SpecJSON) == "" {
		return submitNativeEMFailure(ctx, work, execCtx, "missing_em_spec", fmt.Errorf("missing EM job spec"))
	}

	result, runErr := runner.RunNativeEM(ctx, work.SpecJSON, execCtx.gpus, execCtx.client.NodeAuthToken(0))
	if result == nil {
		return submitNativeEMFailure(ctx, work, execCtx, "native_em_failed", runErr)
	}

	resultHash := result.Hash
	metadata := receiptMetadataBase(
		work,
		execCtx.runtimeManager.ReceiptMetadata(execCtx.gpuDetected),
		result.Metadata,
		map[string]any{
			"executor":    executorKindNativeEM,
			"task":        "em_simulation",
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
		} else {
			// Don't swallow: without object_key the hub has no result data and the
			// study can't build a dataset. Surface it so it's diagnosable.
			slog.Warn("EM result artifact upload failed; receipt will carry no object_key",
				"job_id", work.JobID, "error", uploadErr)
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

// submitNativeEMFailure writes a deterministic fail receipt so the hub gets a
// clean fail+refund instead of a hang (mirrors the EM runner fail() contract).
func submitNativeEMFailure(ctx context.Context, work *hub.WorkAssignment, execCtx executionContext, code string, runErr error) (*runnerResultSnapshot, error) {
	msg := code
	if runErr != nil && strings.TrimSpace(runErr.Error()) != "" {
		msg = runErr.Error()
	}
	sum := sha256.Sum256([]byte(work.JobID + ":" + code + ":" + msg))
	hash := hex.EncodeToString(sum[:])
	metadata := receiptMetadataBase(
		work,
		execCtx.runtimeManager.ReceiptMetadata(execCtx.gpuDetected),
		map[string]any{
			"executor":   executorKindNativeEM,
			"task":       "em_simulation",
			"exit_code":  1,
			"error_code": code,
			"error":      msg,
		},
	)
	_ = submitReceiptWithRetry(ctx, execCtx.client, hub.Receipt{
		JobID:         work.JobID,
		ResultHashHex: hash,
		MeteringUnits: 0,
		Metadata:      metadata,
	})
	return &runnerResultSnapshot{ResultHashHex: hash, Metadata: metadata, ExitCode: 1}, runErr
}

func receiptMetadataBase(work *hub.WorkAssignment, extras ...map[string]any) map[string]any {
	out := map[string]any{
		"executor_kind":   executorKindForAssignment(work),
		"assurance_class": assuranceClassForAssignment(work),
	}
	if energyReceipt := jobEnergyReceiptMetadata(work); len(energyReceipt) > 0 {
		out["energy_receipt"] = energyReceipt
	}
	for _, extra := range extras {
		for key, value := range extra {
			out[key] = value
		}
	}
	return out
}
