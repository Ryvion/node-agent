package runner

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/Ryvion/ryvion-node/internal/runtimeexec"
)

type Result struct {
	Hash            string
	ExitCode        int
	Logs            string
	OutputPath      string
	Duration        time.Duration
	Metrics         map[string]any
	Metadata        map[string]any
	DraftPackets    []map[string]any
	ReceiptComplete bool
}

// Run executes an OCI image with /work mounted and specJSON written to /work/job.json.
// The container is expected to write receipt.json and optional metrics.json/output artifact files.
func Run(ctx context.Context, image, specJSON, gpus string) (*Result, error) {
	if strings.TrimSpace(image) == "" {
		return nil, fmt.Errorf("image required")
	}

	if strings.TrimSpace(specJSON) == "" {
		specJSON = `{}`
	}

	workBase := resolveWorkBase(runtime.GOOS, os.Getenv)
	if workBase != "" {
		if err := os.MkdirAll(workBase, 0o755); err != nil {
			return nil, fmt.Errorf("create work dir: %w", err)
		}
	}
	workDir, err := os.MkdirTemp(workBase, "ryv_job_*")
	if err != nil {
		return nil, fmt.Errorf("create temp work dir: %w", err)
	}
	defer os.RemoveAll(workDir)

	if err := os.WriteFile(filepath.Join(workDir, "job.json"), []byte(specJSON), 0o644); err != nil {
		return nil, fmt.Errorf("write job.json: %w", err)
	}

	// Pre-download payload URL into the work directory so the container
	// (which runs with --network=none) can access it as a local file.
	if err := prefetchPayloadURL(ctx, specJSON, workDir); err != nil {
		slog.Warn("payload prefetch failed (non-fatal)", "error", err)
	}

	ociExec, err := resolveOCIExecutor()
	if err != nil {
		return nil, fmt.Errorf("OCI runtime not found: %w", err)
	}

	name := fmt.Sprintf("ryv_%s", filepath.Base(workDir))
	defer exec.Command(ociExec.command, ociCommandArgs(ociExec, "rm", "-f", name)...).Run()

	memLimit := strings.TrimSpace(os.Getenv("RYV_CONTAINER_MEMORY"))
	if memLimit == "" {
		memLimit = "8g"
	}
	cpuLimit := strings.TrimSpace(os.Getenv("RYV_CONTAINER_CPUS"))
	if cpuLimit == "" {
		cpuLimit = "4"
	}
	// Determine network mode: finetune/training jobs need network access to
	// download base models from HuggingFace. All other jobs run isolated.
	networkMode := "--network=none"
	if needsNetwork(specJSON) {
		networkMode = "--network=bridge"
	}

	// Pull the latest OCI image before running so cached stale images are refreshed.
	pullCtx, pullCancel := context.WithTimeout(ctx, 15*time.Minute)
	pullCmd := exec.CommandContext(pullCtx, ociExec.command, ociCommandArgs(ociExec, "pull", image)...)
	if pullOut, pullErr := pullCmd.CombinedOutput(); pullErr != nil {
		slog.Warn("managed OCI image pull failed (will try cached image)", "image", image, "error", pullErr, "output", string(pullOut[:min(len(pullOut), 200)]))
	} else {
		slog.Info("managed OCI image pull succeeded", "image", image)
	}
	pullCancel()

	args := baseOCIRunArgs(name, workDir, memLimit, cpuLimit, networkMode)
	if gpuArg := resolveGPUFlag(gpus); gpuArg != "" {
		args = append(args, "--gpus", gpuArg)
	} else if gpus == "auto" && isROCmAvailable() {
		// AMD ROCm GPU passthrough
		args = append(args, "--device=/dev/kfd", "--device=/dev/dri", "--group-add=video")
	}
	args = append(args, image)

	start := time.Now()
	cmd := exec.CommandContext(ctx, ociExec.command, ociCommandArgs(ociExec, args...)...)
	var out cappedBuffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	runErr := cmd.Run()
	duration := time.Since(start)

	if ctx.Err() != nil {
		stopContainerGracefully(ociExec, name, abortGracePeriod())
	}

	exitCode := 0
	if runErr != nil {
		if ee, ok := runErr.(*exec.ExitError); ok {
			exitCode = ee.ExitCode()
		} else {
			exitCode = -1
		}
	}

	receiptHash := readReceiptHash(
		filepath.Join(workDir, "receipt.json"),
		filepath.Join(workDir, "receipt.partial.json"),
	)
	receiptComplete := receiptFileHasHash(filepath.Join(workDir, "receipt.json"))
	metrics := readMetrics(filepath.Join(workDir, "metrics.json"), duration)
	probeSummary := readProbeSummary(
		filepath.Join(workDir, "probe_summary.json"),
		filepath.Join(workDir, "probe_summary.partial.json"),
	)
	verifierSessionReceipt := readVerifierSessionReceipt(
		filepath.Join(workDir, "verifier_session_receipt.json"),
		filepath.Join(workDir, "verifier_session_receipt.partial.json"),
	)
	draftPackets := readDraftPackets(
		filepath.Join(workDir, "draft_packets.json"),
		filepath.Join(workDir, "draft_packets.partial.json"),
	)
	artifactPath, _ := copyArtifact(workDir, workBase)

	hash := receiptHash
	if hash == "" {
		sum := sha256.Sum256(out.Bytes())
		hash = hex.EncodeToString(sum[:])
	}

	result := &Result{
		Hash:            hash,
		ExitCode:        exitCode,
		Logs:            out.Tail(32768),
		OutputPath:      artifactPath,
		Duration:        duration,
		Metrics:         metrics,
		Metadata:        runnerMetadata(probeSummary, verifierSessionReceipt),
		DraftPackets:    draftPackets,
		ReceiptComplete: receiptComplete,
	}
	return result, runErr
}

func RunVerifierSession(ctx context.Context, image, specJSON, gpus string) (*Result, error) {
	if strings.TrimSpace(image) == "" {
		return nil, fmt.Errorf("image required")
	}
	if strings.TrimSpace(specJSON) == "" {
		specJSON = `{}`
	}
	workBase := resolveWorkBase(runtime.GOOS, os.Getenv)
	if workBase == "" && runtime.GOOS != "windows" {
		workBase = "/tmp"
	}
	if workBase != "" {
		if err := os.MkdirAll(workBase, 0o755); err != nil {
			return nil, fmt.Errorf("create work dir: %w", err)
		}
	}
	workDir, err := os.MkdirTemp(workBase, "ryv_verifier_session_*")
	if err != nil {
		return nil, fmt.Errorf("create temp work dir: %w", err)
	}
	defer os.RemoveAll(workDir)

	if err := os.WriteFile(filepath.Join(workDir, "job.json"), []byte(specJSON), 0o644); err != nil {
		return nil, fmt.Errorf("write job.json: %w", err)
	}
	if err := prefetchPayloadURL(ctx, specJSON, workDir); err != nil {
		slog.Warn("payload prefetch failed (non-fatal)", "error", err)
	}
	ociExec, err := resolveOCIExecutor()
	if err != nil {
		return nil, fmt.Errorf("OCI runtime not found: %w", err)
	}
	name := fmt.Sprintf("ryv_%s", filepath.Base(workDir))
	defer exec.Command(ociExec.command, ociCommandArgs(ociExec, "rm", "-f", name)...).Run()

	memLimit := strings.TrimSpace(os.Getenv("RYV_CONTAINER_MEMORY"))
	if memLimit == "" {
		memLimit = "8g"
	}
	cpuLimit := strings.TrimSpace(os.Getenv("RYV_CONTAINER_CPUS"))
	if cpuLimit == "" {
		cpuLimit = "4"
	}
	socketPath := filepath.Join(workDir, "verifier_session.sock")
	args := baseOCIDetachedRunArgs(name, workDir, memLimit, cpuLimit, "--network=none", "/work/verifier_session.sock")
	if gpuArg := resolveGPUFlag(gpus); gpuArg != "" {
		args = append(args, "--gpus", gpuArg)
	} else if gpus == "auto" && isROCmAvailable() {
		args = append(args, "--device=/dev/kfd", "--device=/dev/dri", "--group-add=video")
	}
	args = append(args, image)

	start := time.Now()
	runOut, runErr := exec.CommandContext(ctx, ociExec.command, ociCommandArgs(ociExec, args...)...).CombinedOutput()
	if runErr != nil {
		duration := time.Since(start)
		return &Result{
			Hash:     sha256String(string(runOut)),
			ExitCode: -1,
			Logs:     string(runOut),
			Duration: duration,
			Metrics:  map[string]any{"duration_ms": duration.Milliseconds()},
		}, runErr
	}
	rpcErr := waitForVerifierSessionSocket(ctx, socketPath, 30*time.Second)
	var execResult VerifierSessionExecution
	if rpcErr == nil {
		execResult, rpcErr = ExecuteVerifierSessionRPC(ctx, socketPath, specJSON)
	}
	if ctx.Err() != nil {
		stopContainerGracefully(ociExec, name, abortGracePeriod())
	} else {
		waitCtx, waitCancel := context.WithTimeout(context.Background(), 30*time.Second)
		_, waitErr := exec.CommandContext(waitCtx, ociExec.command, ociCommandArgs(ociExec, "wait", name)...).CombinedOutput()
		waitCancel()
		if waitErr != nil {
			stopContainerGracefully(ociExec, name, abortGracePeriod())
		}
	}
	logsOut, _ := exec.Command(ociExec.command, ociCommandArgs(ociExec, "logs", name)...).CombinedOutput()
	duration := time.Since(start)
	receiptHash := readReceiptHash(
		filepath.Join(workDir, "receipt.json"),
		filepath.Join(workDir, "receipt.partial.json"),
	)
	receiptComplete := receiptFileHasHash(filepath.Join(workDir, "receipt.json"))
	metrics := readMetrics(filepath.Join(workDir, "metrics.json"), duration)
	probeSummary := readProbeSummary(
		filepath.Join(workDir, "probe_summary.json"),
		filepath.Join(workDir, "probe_summary.partial.json"),
	)
	verifierSessionReceipt := readVerifierSessionReceipt(
		filepath.Join(workDir, "verifier_session_receipt.json"),
		filepath.Join(workDir, "verifier_session_receipt.partial.json"),
	)
	artifactPath, _ := copyArtifact(workDir, workBase)
	hash := receiptHash
	if hash == "" {
		hash = sha256String(string(logsOut))
	}
	metadata := runnerMetadata(probeSummary, verifierSessionReceipt)
	if metadata == nil {
		metadata = map[string]any{}
	}
	metadata["verifier_session_rpc"] = map[string]any{
		"accepted_len":          execResult.AcceptedLen,
		"rollback_branch_count": len(execResult.RollbackBranchIDs),
		"tree_cid":              execResult.TreeCID,
	}
	return &Result{
		Hash:            hash,
		ExitCode:        0,
		Logs:            string(logsOut),
		OutputPath:      artifactPath,
		Duration:        duration,
		Metrics:         metrics,
		Metadata:        metadata,
		ReceiptComplete: receiptComplete,
	}, rpcErr
}

func sha256String(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])
}

func baseOCIRunArgs(name, workDir, memLimit, cpuLimit, networkMode string) []string {
	graceSeconds := int(abortGracePeriod().Seconds())
	if graceSeconds <= 0 {
		graceSeconds = 10
	}
	return []string{"run", "--name", name, "--rm", "-v", workDir + ":/work",
		"--memory", memLimit, "--memory-swap", memLimit, "--cpus", cpuLimit, "--pids-limit", "256",
		"--cpu-shares", "256",
		"--cap-drop=ALL",
		"--security-opt=no-new-privileges:true",
		"--env", "RYV_RECEIPT_PATH=/work/receipt.json",
		"--env", "RYV_PARTIAL_RECEIPT_PATH=/work/receipt.partial.json",
		"--env", "RYV_PROBE_SUMMARY_PATH=/work/probe_summary.json",
		"--env", "RYV_PARTIAL_PROBE_SUMMARY_PATH=/work/probe_summary.partial.json",
		"--env", "RYV_VERIFIER_SESSION_RECEIPT_PATH=/work/verifier_session_receipt.json",
		"--env", "RYV_PARTIAL_VERIFIER_SESSION_RECEIPT_PATH=/work/verifier_session_receipt.partial.json",
		"--env", fmt.Sprintf("RYV_ABORT_GRACE_SECONDS=%d", graceSeconds),
		networkMode}
}

func baseOCIDetachedRunArgs(name, workDir, memLimit, cpuLimit, networkMode, socketPath string) []string {
	graceSeconds := int(abortGracePeriod().Seconds())
	if graceSeconds <= 0 {
		graceSeconds = 10
	}
	return []string{"run", "--name", name, "-d", "-v", workDir + ":/work",
		"--memory", memLimit, "--memory-swap", memLimit, "--cpus", cpuLimit, "--pids-limit", "256",
		"--cpu-shares", "256",
		"--cap-drop=ALL",
		"--security-opt=no-new-privileges:true",
		"--env", "RYV_RECEIPT_PATH=/work/receipt.json",
		"--env", "RYV_PARTIAL_RECEIPT_PATH=/work/receipt.partial.json",
		"--env", "RYV_PROBE_SUMMARY_PATH=/work/probe_summary.json",
		"--env", "RYV_PARTIAL_PROBE_SUMMARY_PATH=/work/probe_summary.partial.json",
		"--env", "RYV_VERIFIER_SESSION_RECEIPT_PATH=/work/verifier_session_receipt.json",
		"--env", "RYV_PARTIAL_VERIFIER_SESSION_RECEIPT_PATH=/work/verifier_session_receipt.partial.json",
		"--env", "RYV_VERIFIER_SESSION_SOCKET=" + socketPath,
		"--env", fmt.Sprintf("RYV_ABORT_GRACE_SECONDS=%d", graceSeconds),
		networkMode}
}

func abortGracePeriod() time.Duration {
	raw := strings.TrimSpace(os.Getenv("RYV_ABORT_GRACE_PERIOD"))
	if raw != "" {
		if d, err := time.ParseDuration(raw); err == nil && d > 0 {
			return clampAbortGracePeriod(d)
		}
	}
	rawSeconds := strings.TrimSpace(os.Getenv("RYV_ABORT_GRACE_SECONDS"))
	if rawSeconds != "" {
		if seconds, err := strconv.Atoi(rawSeconds); err == nil && seconds > 0 {
			return clampAbortGracePeriod(time.Duration(seconds) * time.Second)
		}
	}
	return 10 * time.Second
}

func clampAbortGracePeriod(d time.Duration) time.Duration {
	if d < time.Second {
		return time.Second
	}
	if d > 60*time.Second {
		return 60 * time.Second
	}
	return d
}

func stopContainerGracefully(ociExec ociExecutor, name string, grace time.Duration) {
	seconds := int(grace.Seconds())
	if seconds <= 0 {
		seconds = 10
	}
	stopCtx, stopCancel := context.WithTimeout(context.Background(), grace+5*time.Second)
	stopErr := exec.CommandContext(stopCtx, ociExec.command, ociCommandArgs(ociExec, "stop", "--time", strconv.Itoa(seconds), name)...).Run()
	stopCancel()
	if stopErr == nil {
		return
	}
	killCtx, killCancel := context.WithTimeout(context.Background(), 10*time.Second)
	if err := exec.CommandContext(killCtx, ociExec.command, ociCommandArgs(ociExec, "kill", name)...).Run(); err != nil {
		slog.Warn("failed to kill timed-out container", "name", name, "stop_error", stopErr, "error", err)
	}
	killCancel()
}

// needsNetwork checks if a job spec requires network access inside the container.
// Currently only finetune jobs need this (to download HuggingFace base models).
func needsNetwork(specJSON string) bool {
	var spec map[string]any
	if json.Unmarshal([]byte(specJSON), &spec) != nil {
		return false
	}
	task, _ := spec["task"].(string)
	return task == "finetune"
}

// prefetchPayloadURL parses specJSON for payload_url, training_data_url, or
// audio_url fields and downloads them into workDir so the container (which
// runs with --network=none) can access them as local files.
func prefetchPayloadURL(ctx context.Context, specJSON, workDir string) error {
	var spec map[string]any
	if err := json.Unmarshal([]byte(specJSON), &spec); err != nil {
		return nil // not JSON, skip
	}
	downloads := map[string]string{
		"payload_url":       "payload.bin",
		"training_data_url": "training.jsonl",
		"audio_url":         "input_audio",
		"input_url":         "input.bin",
		"model_url":         "model.bin",
	}
	for field, filename := range downloads {
		rawURL, ok := spec[field].(string)
		if !ok || strings.TrimSpace(rawURL) == "" {
			continue
		}
		dest := filepath.Join(workDir, filename)
		if err := downloadToFile(ctx, rawURL, dest); err != nil {
			slog.Warn("prefetch download failed", "field", field, "url", rawURL[:min(len(rawURL), 80)], "error", err)
			continue
		}
		slog.Info("prefetched input file", "field", field, "dest", filename, "size", fileSize(dest))
	}
	return nil
}

func downloadToFile(ctx context.Context, rawURL, dest string) error {
	if err := validateDownloadURL(rawURL, allowLoopbackDownloads()); err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return err
	}
	client := restrictedHTTPClient(10*time.Minute, allowLoopbackDownloads())
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, resp.Body)
	return err
}

func fileSize(path string) int64 {
	fi, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return fi.Size()
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func resolveWorkBase(goos string, getenv func(string) string) string {
	if getenv == nil {
		getenv = os.Getenv
	}
	if workBase := strings.TrimSpace(getenv("RYV_WORK_DIR")); workBase != "" {
		return workBase
	}
	if goos != "windows" {
		return ""
	}
	programData := strings.TrimSpace(getenv("ProgramData"))
	if programData == "" {
		programData = `C:\ProgramData`
	}
	return filepath.Join(programData, "Ryvion", "work")
}

type ociExecutor struct {
	command    string
	prefixArgs []string
}

func resolveOCIExecutor() (ociExecutor, error) {
	execConfig, err := runtimeexec.ResolveExecutor(runtime.GOOS, os.Getenv)
	if err != nil {
		return ociExecutor{}, err
	}
	return ociExecutor{command: execConfig.Command, prefixArgs: execConfig.PrefixArgs}, nil
}

func ociCommandArgs(executor ociExecutor, args ...string) []string {
	out := append([]string{}, executor.prefixArgs...)
	return append(out, args...)
}

func resolveGPUFlag(gpus string) string {
	gpus = strings.TrimSpace(strings.ToLower(gpus))
	switch gpus {
	case "", "none", "off":
		return ""
	case "auto":
		if _, err := exec.LookPath("nvidia-smi"); err == nil {
			return "all"
		}
		return ""
	default:
		return gpus
	}
}

func isROCmAvailable() bool {
	_, err := os.Stat("/dev/kfd")
	return err == nil
}

func readReceiptHash(paths ...string) string {
	for _, path := range paths {
		if hash := receiptFileOutputHash(path); hash != "" {
			return hash
		}
	}
	return ""
}

func receiptFileHasHash(path string) bool {
	return receiptFileOutputHash(path) != ""
}

func receiptFileOutputHash(path string) string {
	if strings.HasSuffix(path, ".tmp") {
		return ""
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	var rec struct {
		OutputHash string `json:"output_hash"`
	}
	if err := json.Unmarshal(b, &rec); err != nil {
		return ""
	}
	return trimDigestPrefix(strings.TrimSpace(rec.OutputHash))
}

func readMetrics(path string, duration time.Duration) map[string]any {
	b, err := os.ReadFile(path)
	if err != nil {
		return map[string]any{"duration_ms": duration.Milliseconds()}
	}
	var metrics map[string]any
	if err := json.Unmarshal(b, &metrics); err != nil {
		return map[string]any{"duration_ms": duration.Milliseconds()}
	}
	if metrics == nil {
		metrics = map[string]any{}
	}
	if _, ok := metrics["duration_ms"]; !ok {
		metrics["duration_ms"] = duration.Milliseconds()
	}
	return metrics
}

func readProbeSummary(paths ...string) map[string]any {
	for _, path := range paths {
		if strings.HasSuffix(path, ".tmp") {
			continue
		}
		f, err := os.Open(path)
		if err != nil {
			continue
		}
		limited, readErr := io.ReadAll(io.LimitReader(f, 64<<10))
		_ = f.Close()
		if readErr != nil {
			continue
		}
		var raw map[string]any
		if err := json.Unmarshal(limited, &raw); err != nil {
			continue
		}
		if summary := sanitizeProbeSummary(raw); len(summary) > 0 {
			return summary
		}
	}
	return nil
}

func runnerMetadataFromProbeSummary(summary map[string]any) map[string]any {
	if len(summary) == 0 {
		return nil
	}
	return map[string]any{"probe_summary": summary}
}

func runnerMetadata(probeSummary, verifierSessionReceipt map[string]any) map[string]any {
	out := map[string]any{}
	if len(probeSummary) > 0 {
		out["probe_summary"] = probeSummary
	}
	if len(verifierSessionReceipt) > 0 {
		out["verifier_session"] = verifierSessionReceipt
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func readVerifierSessionReceipt(paths ...string) map[string]any {
	for _, path := range paths {
		if strings.HasSuffix(path, ".tmp") {
			continue
		}
		f, err := os.Open(path)
		if err != nil {
			continue
		}
		limited, readErr := io.ReadAll(io.LimitReader(f, 64<<10))
		_ = f.Close()
		if readErr != nil {
			continue
		}
		var raw map[string]any
		if err := json.Unmarshal(limited, &raw); err != nil {
			continue
		}
		if summary := sanitizeVerifierSessionReceipt(raw); len(summary) > 0 {
			return summary
		}
	}
	return nil
}

func readDraftPackets(paths ...string) []map[string]any {
	for _, path := range paths {
		if strings.HasSuffix(path, ".tmp") {
			continue
		}
		f, err := os.Open(path)
		if err != nil {
			continue
		}
		limited, readErr := io.ReadAll(io.LimitReader(f, 256<<10))
		_ = f.Close()
		if readErr != nil {
			continue
		}
		var raw any
		if err := json.Unmarshal(limited, &raw); err != nil {
			continue
		}
		packets := sanitizeDraftPackets(raw)
		if len(packets) > 0 {
			return packets
		}
	}
	return nil
}

func sanitizeDraftPackets(raw any) []map[string]any {
	var list []any
	switch typed := raw.(type) {
	case []any:
		list = typed
	case map[string]any:
		if packets, ok := typed["packets"].([]any); ok {
			list = packets
		}
	default:
		return nil
	}
	out := make([]map[string]any, 0, len(list))
	for _, item := range list {
		packet, ok := item.(map[string]any)
		if !ok {
			continue
		}
		safe := sanitizeDraftPacket(packet)
		if len(safe) > 0 {
			out = append(out, safe)
		}
	}
	return out
}

func sanitizeDraftPacket(raw map[string]any) map[string]any {
	out := map[string]any{}
	for _, key := range []string{
		"packet_id", "window_id", "workgraph_id", "role_id", "node_id", "parent_prefix_hash",
		"candidate_tokens", "model_hash", "drafter_model_id", "horizon", "confidence_bps",
		"energy_mwh", "deadline_ms", "signature", "submitted_at",
	} {
		value, ok := raw[key]
		if !ok || forbiddenDraftPacketKey(key) {
			continue
		}
		if key == "candidate_tokens" {
			tokens := sanitizeTokenList(value)
			if len(tokens) == 0 {
				continue
			}
			out[key] = tokens
			continue
		}
		out[key] = value
	}
	if len(out) == 0 || out["candidate_tokens"] == nil {
		return nil
	}
	return out
}

func sanitizeTokenList(value any) []any {
	raw, ok := value.([]any)
	if !ok {
		return nil
	}
	out := make([]any, 0, len(raw))
	for _, token := range raw {
		switch typed := token.(type) {
		case float64:
			if typed >= 0 && typed == float64(int64(typed)) {
				out = append(out, typed)
			}
		case int:
			if typed >= 0 {
				out = append(out, typed)
			}
		case int64:
			if typed >= 0 {
				out = append(out, typed)
			}
		}
	}
	return out
}

func forbiddenDraftPacketKey(key string) bool {
	k := strings.ToLower(strings.TrimSpace(key))
	k = strings.ReplaceAll(k, "-", "_")
	switch k {
	case "prompt", "raw_prompt", "prompt_text", "input_text", "raw_input",
		"output", "raw_output", "output_text", "response_text", "completion_text",
		"candidate_text", "candidate_text_preview",
		"raw_activation", "raw_activations", "raw_hidden_state", "hidden_state_values",
		"raw_logits", "logits", "raw_attention",
		"raw_sensor", "raw_media", "private_key", "secret", "api_key":
		return true
	default:
		return false
	}
}

func sanitizeVerifierSessionReceipt(raw map[string]any) map[string]any {
	if len(raw) == 0 {
		return nil
	}
	out := map[string]any{}
	for _, key := range []string{
		"schema_version", "receipt_type", "method", "session_id", "workgraph_id", "window_id",
		"tree_cid", "kv_epoch", "accepted_len", "rejected_reason", "commit_range",
		"rollback_branch_ids", "verifier_signature", "status", "latency_ms", "energy_mwh",
	} {
		value, ok := raw[key]
		if !ok || forbiddenVerifierSessionKey(key) {
			continue
		}
		switch typed := value.(type) {
		case map[string]any:
			out[key] = sanitizeVerifierSessionReceiptMap(typed)
		case []any:
			out[key] = sanitizeVerifierSessionReceiptList(typed)
		default:
			out[key] = typed
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func sanitizeVerifierSessionReceiptMap(raw map[string]any) map[string]any {
	out := map[string]any{}
	for key, value := range raw {
		if forbiddenVerifierSessionKey(key) {
			continue
		}
		switch typed := value.(type) {
		case map[string]any:
			nested := sanitizeVerifierSessionReceiptMap(typed)
			if len(nested) > 0 {
				out[key] = nested
			}
		case []any:
			list := sanitizeVerifierSessionReceiptList(typed)
			if len(list) > 0 {
				out[key] = list
			}
		default:
			out[key] = value
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func sanitizeVerifierSessionReceiptList(raw []any) []any {
	out := make([]any, 0, len(raw))
	for _, value := range raw {
		switch typed := value.(type) {
		case map[string]any:
			nested := sanitizeVerifierSessionReceiptMap(typed)
			if len(nested) > 0 {
				out = append(out, nested)
			}
		default:
			out = append(out, value)
		}
	}
	return out
}

func forbiddenVerifierSessionKey(key string) bool {
	k := strings.ToLower(strings.TrimSpace(key))
	k = strings.ReplaceAll(k, "-", "_")
	switch k {
	case "prompt", "raw_prompt", "prompt_text", "input_text", "raw_input",
		"output", "raw_output", "output_text", "response_text", "completion_text",
		"candidate_text", "candidate_text_preview",
		"raw_kv", "raw_kv_cache", "kv_cache", "kv_values", "raw_activation", "raw_activations",
		"raw_hidden_state", "hidden_state_values", "raw_logits", "logits", "raw_attention",
		"raw_sensor", "raw_media", "private_key", "secret", "api_key":
		return true
	default:
		return false
	}
}

func sanitizeProbeSummary(raw map[string]any) map[string]any {
	if len(raw) == 0 {
		return nil
	}
	out := map[string]any{}
	for _, key := range []string{
		"workgraph_id", "role_id", "model_hash", "probe_pack_cid",
		"feature_scores_bps", "confidence_bps", "answer_confidence_bps",
		"risk_flags", "accepted_tokens", "signature",
		"reasoning_performativity_bps", "eval_awareness_risk_bps",
		"early_exit_recommended", "verifier_signature",
	} {
		value, ok := raw[key]
		if !ok || forbiddenProbeSummaryKey(key) {
			continue
		}
		switch typed := value.(type) {
		case map[string]any:
			out[key] = sanitizeProbeSummaryMap(typed)
		case []any:
			out[key] = sanitizeProbeSummaryList(typed)
		default:
			out[key] = typed
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func sanitizeProbeSummaryMap(raw map[string]any) map[string]any {
	out := map[string]any{}
	for key, value := range raw {
		if forbiddenProbeSummaryKey(key) {
			continue
		}
		switch typed := value.(type) {
		case map[string]any:
			nested := sanitizeProbeSummaryMap(typed)
			if len(nested) > 0 {
				out[key] = nested
			}
		case []any:
			list := sanitizeProbeSummaryList(typed)
			if len(list) > 0 {
				out[key] = list
			}
		default:
			out[key] = value
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func sanitizeProbeSummaryList(raw []any) []any {
	out := make([]any, 0, len(raw))
	for _, value := range raw {
		switch typed := value.(type) {
		case map[string]any:
			nested := sanitizeProbeSummaryMap(typed)
			if len(nested) > 0 {
				out = append(out, nested)
			}
		default:
			out = append(out, value)
		}
	}
	return out
}

func forbiddenProbeSummaryKey(key string) bool {
	k := strings.ToLower(strings.TrimSpace(key))
	k = strings.ReplaceAll(k, "-", "_")
	switch k {
	case "prompt", "raw_prompt", "prompt_text", "input_text", "raw_input",
		"output", "raw_output", "output_text", "response_text", "completion_text",
		"raw_activation", "raw_activations", "activation_values", "raw_hidden_state", "hidden_state_values",
		"raw_logits", "logits", "logit_vector", "raw_attention", "attention_values",
		"raw_sensor", "raw_media", "private_key", "secret", "api_key":
		return true
	default:
		return false
	}
}

func copyArtifact(workDir, workBase string) (string, error) {
	return copyArtifactFrom(workDir, workBase, artifactCandidates(workDir))
}

// copyArtifactFrom copies the first readable, in-jail candidate to workBase.
// Callers can prepend preferred paths (e.g. the EM lane prefers result.json,
// the JSON mirror the hub parses, over the binary result.npz).
func copyArtifactFrom(workDir, workBase string, candidates []string) (string, error) {
	workRoot := canonicalPath(workDir)
	for _, src := range candidates {
		fi, err := os.Stat(src)
		if err != nil || fi.IsDir() {
			continue
		}
		resolved, err := filepath.EvalSymlinks(src)
		if err != nil {
			continue
		}
		if !isPathWithin(workRoot, canonicalPath(resolved)) {
			slog.Warn("artifact path traversal blocked", "path", src, "resolved", resolved)
			continue
		}
		targetDir := workBase
		if targetDir == "" {
			targetDir = os.TempDir()
		}
		dst, err := os.CreateTemp(targetDir, "ryv_artifact_*")
		if err != nil {
			return "", err
		}
		defer dst.Close()
		in, err := os.Open(src)
		if err != nil {
			return "", err
		}
		_, copyErr := io.Copy(dst, in)
		_ = in.Close()
		if copyErr != nil {
			_ = os.Remove(dst.Name())
			return "", copyErr
		}
		return dst.Name(), nil
	}
	return "", nil
}

func artifactCandidates(workDir string) []string {
	controlFiles := map[string]bool{
		"job.json":                                  true,
		"receipt.json":                              true,
		"receipt.partial.json":                      true,
		"receipt.partial.json.tmp":                  true,
		"metrics.json":                              true,
		"metrics.partial.json":                      true,
		"metrics.partial.json.tmp":                  true,
		"probe_summary.json":                        true,
		"probe_summary.partial.json":                true,
		"probe_summary.partial.json.tmp":            true,
		"verifier_session_receipt.json":             true,
		"verifier_session_receipt.partial.json":     true,
		"verifier_session_receipt.partial.json.tmp": true,
		"draft_packets.json":                        true,
		"draft_packets.partial.json":                true,
		"draft_packets.partial.json.tmp":            true,
	}
	seen := map[string]bool{}
	candidates := []string{}
	add := func(path string) {
		path = strings.TrimSpace(path)
		if path == "" {
			return
		}
		cleaned := filepath.Clean(path)
		if seen[cleaned] {
			return
		}
		seen[cleaned] = true
		candidates = append(candidates, cleaned)
	}

	add(filepath.Join(workDir, "output"))
	add(filepath.Join(workDir, "output.bin"))
	add(filepath.Join(workDir, "result.bin"))

	if outputName := metricsOutputName(filepath.Join(workDir, "metrics.json")); outputName != "" {
		base := filepath.Base(outputName)
		add(filepath.Join(workDir, base))
		// Runners (OCI image + native EM bundle) write the artifact into the
		// WORK_DIR/output/ subdir and report output_name as just the basename,
		// so also look there. Honor a relative output_name (e.g. output/x.npz).
		add(filepath.Join(workDir, "output", base))
		if !filepath.IsAbs(outputName) {
			add(filepath.Join(workDir, filepath.Clean(outputName)))
		}
	}

	// Scan workDir top level AND the conventional output/ subdir for any
	// non-control file the runner may have left as the artifact.
	scanDir := func(dir, prefix string) {
		entries, err := os.ReadDir(dir)
		if err != nil {
			return
		}
		for _, entry := range entries {
			if entry.IsDir() {
				continue
			}
			if controlFiles[strings.ToLower(entry.Name())] {
				continue
			}
			add(filepath.Join(prefix, entry.Name()))
		}
	}
	scanDir(workDir, workDir)
	scanDir(filepath.Join(workDir, "output"), filepath.Join(workDir, "output"))
	return candidates
}

func metricsOutputName(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	var metrics map[string]any
	if err := json.Unmarshal(b, &metrics); err != nil {
		return ""
	}
	v, _ := metrics["output_name"].(string)
	return strings.TrimSpace(v)
}

func canonicalPath(path string) string {
	if resolved, err := filepath.EvalSymlinks(path); err == nil {
		return filepath.Clean(resolved)
	}
	return filepath.Clean(path)
}

func isPathWithin(root, target string) bool {
	if root == "" || target == "" {
		return false
	}
	rel, err := filepath.Rel(root, target)
	if err != nil {
		return false
	}
	if rel == "." {
		return true
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(os.PathSeparator))
}

func trimDigestPrefix(v string) string {
	if i := strings.IndexByte(v, ':'); i > 0 {
		return v[i+1:]
	}
	return v
}

type cappedBuffer struct {
	buf []byte
}

func (c *cappedBuffer) Write(p []byte) (int, error) {
	c.buf = append(c.buf, p...)
	const limit = 1 << 20
	if len(c.buf) > limit {
		c.buf = c.buf[len(c.buf)-limit:]
	}
	return len(p), nil
}

func (c *cappedBuffer) Bytes() []byte {
	return c.buf
}

func (c *cappedBuffer) Tail(n int) string {
	if n <= 0 || len(c.buf) == 0 {
		return ""
	}
	if len(c.buf) <= n {
		return string(c.buf)
	}
	return string(c.buf[len(c.buf)-n:])
}
