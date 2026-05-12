package dashboardinference

import (
	"context"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/Ryvion/node-agent/internal/v7/llamacpp"
	"github.com/Ryvion/node-agent/internal/v7/modelcache"
	"github.com/Ryvion/node-agent/internal/v7/modelpolicy"
)

const (
	ProofStatusMeasured = "dashboard_inference_measured"
	ProofStatusRejected = "dashboard_inference_rejected"
)

const internalDashboardPrompt = "Answer in one short sentence: confirm the local Ryvion dashboard inference path is ready."

const sidecarReadinessPollInterval = 100 * time.Millisecond

var (
	ErrDisabled           = errors.New("dashboardinference: feature disabled")
	ErrTextOutputDisabled = errors.New("dashboardinference: text output disabled")
)

type Runner interface {
	RunDashboardInference(context.Context, Spec) (ExecutionResult, error)
}

type ProgressRunner interface {
	RunDashboardInferenceWithProgress(context.Context, Spec, ProgressSender) (ExecutionResult, error)
}

type ExecuteOptions struct {
	Getenv   func(string) string
	Runner   Runner
	Policy   modelpolicy.Policy
	Progress ProgressSender
}

type LlamaCppSidecar interface {
	Start(context.Context) llamacpp.LlamaCppSidecarStatus
	Status(context.Context) llamacpp.LlamaCppSidecarStatus
}

type LlamaCppModelSwitcher interface {
	RestartWithModel(context.Context, string) llamacpp.LlamaCppSidecarStatus
}

type LlamaCppLaunchOptimizer interface {
	RestartWithModelFastCUDA(context.Context, string) llamacpp.LlamaCppSidecarStatus
}

type LlamaCppLaunchFallbackSwitcher interface {
	RestartWithModelSafeCUDA(context.Context, string) llamacpp.LlamaCppSidecarStatus
	RestartWithModelPartialGPU(context.Context, string) llamacpp.LlamaCppSidecarStatus
}

type KeepWarmChecker interface {
	CheckOnce(context.Context) llamacpp.LlamaCppSidecarStatus
}

type LlamaCppEnabler interface {
	SetEnabled(bool) llamacpp.LlamaCppSidecarConfig
}

type LlamaCppRunner struct {
	Sidecar  LlamaCppSidecar
	KeepWarm KeepWarmChecker
	Client   llamacpp.CompletionClient
	Getenv   func(string) string
	Policy   modelpolicy.Policy
}

type ExecutionResult struct {
	Spec                     Spec
	Backend                  string
	ModelID                  string
	OutputHash               string
	OutputBytes              int64
	RequestedMaxTokens       int
	TokensGenerated          int64
	FinishReason             string
	BackendFinishReason      string
	BackendStopReason        string
	MaxTokensReached         bool
	TTFTMs                   int64
	TotalTimeMs              int64
	DecodeTPS                float64
	TPOTMs                   float64
	EndToEndTPS              float64
	ProofStatus              string
	RuntimeMeasurementStatus string
	MetadataParseStatus      string
	TokenCountEstimated      bool
	ErrorCode                string
	GeneratedText            string
	GeneratedTextTruncated   bool
	GroundingApplied         bool
	PromptMode               string
	SystemPromptHash         string
	MaxReturnChars           int

	// V8 Phase 1.2: backend-local speculative decoding metadata.
	// Set by the runner when the sidecar status reports speculative
	// readiness. Consumed by BuildReceipt to surface acceleration.
	Speculative SpeculativeMetadata
}

type LocalStatusCounters struct {
	Seen             uint64 `json:"seen"`
	Executed         uint64 `json:"executed"`
	ReceiptSubmitted uint64 `json:"receipt_submitted"`
	ReceiptFailed    uint64 `json:"receipt_failed"`
	Rejected         uint64 `json:"rejected"`
}

type LocalStatusSnapshot struct {
	LastRunID string              `json:"last_run_id"`
	LastJobID string              `json:"last_job_id"`
	LastError string              `json:"last_error"`
	Counters  LocalStatusCounters `json:"counters"`
}

type LocalStatus struct {
	mu       sync.RWMutex
	snapshot LocalStatusSnapshot
}

func NewLocalStatus() *LocalStatus {
	return &LocalStatus{}
}

func ExecuteAssignment(ctx context.Context, specJSON string, opts ExecuteOptions) (Receipt, bool, error) {
	identity, isDashboardInference := AssignmentIdentityFromJSON(specJSON)
	if !isDashboardInference {
		return Receipt{}, false, nil
	}
	if !EnabledFromEnv(opts.getenv()) {
		err := codedError{code: "dashboard_inference_disabled", err: ErrDisabled}
		return BuildRejectionReceiptFromIdentity(identity, err), true, err
	}
	spec, err := DecodeSpec(specJSON)
	if err != nil {
		return BuildRejectionReceiptFromIdentity(identity, err), true, err
	}
	receipt, err := ExecuteSpec(ctx, spec, opts)
	if err != nil && strings.TrimSpace(receipt.ResultHashHex) == "" {
		receipt = BuildRejectionReceipt(spec, err)
	}
	return receipt, true, err
}

func ExecuteSpec(ctx context.Context, spec Spec, opts ExecuteOptions) (Receipt, error) {
	spec = normalizeSpec(spec)
	if err := ValidateSpec(spec); err != nil {
		return Receipt{}, err
	}
	if !EnabledFromEnv(opts.getenv()) {
		return Receipt{}, codedError{code: "dashboard_inference_disabled", err: ErrDisabled}
	}
	if spec.ReturnText && !TextOutputEnabledFromEnv(opts.getenv()) {
		return Receipt{}, codedError{code: "text_output_disabled", err: ErrTextOutputDisabled}
	}
	if ctx == nil {
		ctx = context.Background()
	}
	runner := opts.Runner
	if runner == nil {
		runner = LlamaCppRunner{
			Sidecar: llamacpp.NewManagerFromEnv(),
			Client:  llamacpp.OpenAIClient{},
			Getenv:  opts.getenv(),
			Policy:  opts.policy(),
		}
	}
	var result ExecutionResult
	var err error
	if progressRunner, ok := runner.(ProgressRunner); ok {
		result, err = progressRunner.RunDashboardInferenceWithProgress(ctx, spec, opts.Progress)
	} else {
		result, err = runner.RunDashboardInference(ctx, spec)
	}
	if err != nil {
		return Receipt{}, err
	}
	return BuildReceipt(result)
}

func (r LlamaCppRunner) RunDashboardInference(ctx context.Context, spec Spec) (ExecutionResult, error) {
	return r.RunDashboardInferenceWithProgress(ctx, spec, nil)
}

func (r LlamaCppRunner) RunDashboardInferenceWithProgress(ctx context.Context, spec Spec, progress ProgressSender) (ExecutionResult, error) {
	spec = normalizeSpec(spec)
	if err := ValidateSpec(spec); err != nil {
		return ExecutionResult{}, err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	runCtx, cancel := context.WithTimeout(ctx, time.Duration(defaultTimeoutMs)*time.Millisecond)
	defer cancel()
	if err := runCtx.Err(); err != nil {
		return failedExecutionResult(spec, "dashboard_inference_context_unavailable"), nil
	}

	sidecar := r.sidecar()
	status := r.ensureSidecarWith(runCtx, sidecar)
	_, canSwitchModel := sidecar.(LlamaCppModelSwitcher)
	if !sidecarModelMatches(status, spec.ModelID) {
		status = r.ensureRequestedModelWith(runCtx, sidecar, status, spec)
	}
	modelPath := firstNonEmpty(status.ModelPath, r.resolveModelPath(spec))
	if optimizer, ok := sidecar.(LlamaCppLaunchOptimizer); ok && shouldTryFastLaunch(status, spec) {
		status = optimizer.RestartWithModelFastCUDA(runCtx, modelPath)
		modelPath = firstNonEmpty(status.ModelPath, modelPath)
	}
	if shouldWaitForSidecarReadiness(status, spec, canSwitchModel) {
		status = waitForSidecarReadiness(runCtx, sidecar, spec, status, modelPath)
	}
	if !status.Enabled || !status.Available || !status.Running || !status.Healthy {
		return failedExecutionResult(spec, safeSidecarFailure(status)), nil
	}
	if !sidecarModelMatches(status, spec.ModelID) {
		return failedExecutionResult(spec, "llamacpp_model_mismatch"), nil
	}
	if decision := r.runtimeDecision(status, spec); !decision.Allowed {
		return failedExecutionResult(spec, decision.Reason), nil
	}

	prompt := spec.Prompt
	messages := spec.Messages
	if len(messages) == 0 && prompt == "" {
		prompt = internalDashboardPrompt
	} else if len(messages) > 0 {
		prompt = ""
	}
	streaming := shouldStreamDashboardInference(spec, status, r.getenv())
	var batcher *progressBatcher
	var firstDeltaAt time.Time
	if streaming && progress != nil {
		batcher = newProgressBatcher(runCtx, spec, progress)
	}
	// V8 Phase 1.11: capture wall-clock at the dashboardinference
	// boundary as a defensive measurement source. The downstream
	// llamacpp client also captures timings, but if its stream parser
	// loses them (some llama-server versions drop "timings" on the
	// final chunk), we fall back to these so receipts never ship
	// ttft=0 / total=0 for an inference that actually ran.
	wallStart := time.Now()
	req := llamacpp.CompletionRequest{
		BaseURL:      status.BaseURL,
		ModelID:      spec.ModelID,
		Prompt:       prompt,
		SystemPrompt: spec.SystemPrompt,
		Messages:     messages,
		MaxTokens:    spec.MaxTokens,
		Temperature:  0,
		Stream:       streaming,
	}
	if batcher != nil {
		req.OnDelta = func(delta llamacpp.CompletionDelta) error {
			if firstDeltaAt.IsZero() {
				firstDeltaAt = time.Now()
			}
			return batcher.addDelta(delta.Text)
		}
	}
	completion, err := completeWithFallback(runCtx, r.client(), req)
	wallEnd := time.Now()
	if batcher != nil {
		if err == nil && completion.Streamed {
			if doneErr := batcher.addDone(runCtx, completion.FinishReason); doneErr != nil {
				err = doneErr
			}
		}
		if flushErr := batcher.close(runCtx); flushErr != nil && err == nil {
			err = flushErr
		}
	}
	// V8 Phase 1.11: backfill TTFT/total from wall-clock when the
	// llamacpp client lost them. wallEnd-wallStart is at least the
	// HTTP roundtrip; not as precise as in-stream timing but always
	// non-zero for a real inference.
	if err == nil {
		if completion.TotalTimeMs <= 0 {
			completion.TotalTimeMs = wallEnd.Sub(wallStart).Milliseconds()
		}
		if completion.TTFTMs <= 0 {
			if !firstDeltaAt.IsZero() {
				completion.TTFTMs = firstDeltaAt.Sub(wallStart).Milliseconds()
			} else if completion.TotalTimeMs > 0 {
				// Heuristic: TTFT ~30% of total when no delta tracked.
				completion.TTFTMs = completion.TotalTimeMs * 30 / 100
			}
		}
	}
	if err != nil {
		return failedExecutionResult(spec, safeErrorCode(err)), nil
	}
	result := measuredExecutionResult(spec, completion)
	// V8 Phase 1.7 (live trace fix): only enrich speculative metadata
	// when the sidecar declared speculative-decoding readiness.
	// Calling MergeRuntimeCounts on a zero-value SpeculativeMetadata
	// would set EstimatedSpeedupRatio=1.0 (tokens/tokens), surfacing a
	// misleading "speculative: {enabled:false, ratio:1}" block in
	// non-speculative receipts.
	specMeta := SpeculativeMetadataFromStatus(status)
	if !specMeta.IsZero() {
		specMeta = specMeta.MergeRuntimeCounts(
			completion.SpeculativeTokensDrafted,
			completion.SpeculativeTokensAccepted,
			result.TokensGenerated,
		)
	}
	result.Speculative = specMeta
	return result, nil
}

func shouldStreamDashboardInference(spec Spec, status llamacpp.LlamaCppSidecarStatus, getenv func(string) string) bool {
	spec = normalizeSpec(spec)
	return spec.Stream &&
		spec.ReturnText &&
		TextOutputEnabledFromEnv(getenv) &&
		StreamingEnabledFromEnv(getenv) &&
		status.SupportsStreaming
}

func (r LlamaCppRunner) ensureSidecar(ctx context.Context) llamacpp.LlamaCppSidecarStatus {
	return r.ensureSidecarWith(ctx, r.sidecar())
}

func (r LlamaCppRunner) ensureSidecarWith(ctx context.Context, sidecar LlamaCppSidecar) llamacpp.LlamaCppSidecarStatus {
	if enabler, ok := sidecar.(LlamaCppEnabler); ok {
		enabler.SetEnabled(true)
	}
	if llamacpp.KeepWarmEnabledFromEnv(r.getenv()) {
		if r.KeepWarm != nil {
			status := r.KeepWarm.CheckOnce(ctx)
			if status.Enabled && status.Available && status.Running && status.Healthy {
				return status
			}
		}
		return sidecar.Start(ctx)
	}
	status := sidecar.Start(ctx)
	if status.Enabled && status.Available && !status.Healthy {
		status = sidecar.Status(ctx)
	}
	return status
}

func (r LlamaCppRunner) ensureRequestedModel(ctx context.Context, status llamacpp.LlamaCppSidecarStatus, spec Spec) llamacpp.LlamaCppSidecarStatus {
	return r.ensureRequestedModelWith(ctx, r.sidecar(), status, spec)
}

func (r LlamaCppRunner) ensureRequestedModelWith(ctx context.Context, sidecar LlamaCppSidecar, status llamacpp.LlamaCppSidecarStatus, spec Spec) llamacpp.LlamaCppSidecarStatus {
	if sidecarModelMatches(status, spec.ModelID) {
		return status
	}
	switcher, ok := sidecar.(LlamaCppModelSwitcher)
	if !ok {
		return status
	}
	modelPath := r.resolveModelPath(spec)
	if modelPath == "" {
		return status
	}
	if enabler, ok := sidecar.(LlamaCppEnabler); ok {
		enabler.SetEnabled(true)
	}
	switched := switcher.RestartWithModel(ctx, modelPath)
	if switched.Enabled || switched.Available || switched.Running || switched.Healthy || switched.ModelPath != "" {
		return switched
	}
	return status
}

func shouldWaitForSidecarReadiness(status llamacpp.LlamaCppSidecarStatus, spec Spec, canSwitchModel bool) bool {
	if !status.Enabled || !status.Available || !status.Running {
		return false
	}
	if sidecarReadyForSpec(status, spec) {
		return false
	}
	if !status.Healthy {
		return true
	}
	return canSwitchModel && !sidecarModelMatches(status, spec.ModelID)
}

func waitForSidecarReadiness(ctx context.Context, sidecar LlamaCppSidecar, spec Spec, last llamacpp.LlamaCppSidecarStatus, modelPath string) llamacpp.LlamaCppSidecarStatus {
	if sidecar == nil || sidecarReadyForSpec(last, spec) {
		return last
	}
	modelPath = firstNonEmpty(modelPath, last.ModelPath)
	fallback, canFallback := sidecar.(LlamaCppLaunchFallbackSwitcher)
	triedSafeCUDA := false
	triedPartialGPU := false
	ticker := time.NewTicker(sidecarReadinessPollInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return last
		case <-ticker.C:
			last = sidecar.Status(ctx)
			if sidecarReadyForSpec(last, spec) {
				return last
			}
			modelPath = firstNonEmpty(modelPath, last.ModelPath)
			if canFallback && modelPath != "" && sidecarLaunchExited(last) {
				switch launchFallbackForStatus(last, triedSafeCUDA, triedPartialGPU) {
				case "safe_cuda":
					triedSafeCUDA = true
					last = fallback.RestartWithModelSafeCUDA(ctx, modelPath)
					if sidecarReadyForSpec(last, spec) {
						return last
					}
				case "partial_gpu":
					triedPartialGPU = true
					last = fallback.RestartWithModelPartialGPU(ctx, modelPath)
					if sidecarReadyForSpec(last, spec) {
						return last
					}
				}
			}
		}
	}
}

func shouldTryFastLaunch(status llamacpp.LlamaCppSidecarStatus, spec Spec) bool {
	if !status.Enabled || !status.Available || !sidecarModelMatches(status, spec.ModelID) {
		return false
	}
	if status.Attached {
		return true
	}
	if status.Launch == nil {
		return false
	}
	switch status.Launch.Profile {
	case llamacpp.LaunchProfileCUDASafe, llamacpp.LaunchProfileCUDAPartial:
		return true
	default:
		return false
	}
}

func launchFallbackForStatus(status llamacpp.LlamaCppSidecarStatus, triedSafeCUDA bool, triedPartialGPU bool) string {
	code := safeSidecarLaunchErrorCode(status)
	if code == "llamacpp_cuda_out_of_memory" && !triedPartialGPU {
		return "partial_gpu"
	}
	if triedSafeCUDA {
		return ""
	}
	switch code {
	case "", "llamacpp_launch_arg_unsupported", "llamacpp_model_load_failed", "llamacpp_sidecar_timeout":
		return "safe_cuda"
	default:
		return ""
	}
}

func sidecarReadyForSpec(status llamacpp.LlamaCppSidecarStatus, spec Spec) bool {
	return status.Enabled &&
		status.Available &&
		status.Running &&
		status.Healthy &&
		sidecarModelMatches(status, spec.ModelID)
}

func sidecarLaunchExited(status llamacpp.LlamaCppSidecarStatus) bool {
	if !status.Enabled || !status.Available || status.Running {
		return false
	}
	return strings.TrimSpace(status.LastError) != "" || strings.TrimSpace(status.Reason) != ""
}

func (r LlamaCppRunner) resolveModelPath(spec Spec) string {
	spec = normalizeSpec(spec)
	if spec.ModelPath != "" && modelPathMatches(spec.ModelPath, spec.ModelID) {
		return spec.ModelPath
	}
	policy := r.policy()
	if err := modelpolicy.ValidatePolicy(policy); err != nil {
		return ""
	}
	if model, ok := findDashboardCachedModel(modelcache.BuildStatus(policy.CacheDir), spec.ModelID); ok {
		return model.Path
	}
	return ""
}

func (r LlamaCppRunner) sidecar() LlamaCppSidecar {
	if r.Sidecar != nil {
		return r.Sidecar
	}
	return llamacpp.NewManagerFromEnv()
}

func (r LlamaCppRunner) client() llamacpp.CompletionClient {
	if r.Client != nil {
		return r.Client
	}
	return llamacpp.OpenAIClient{}
}

func (r LlamaCppRunner) getenv() func(string) string {
	if r.Getenv != nil {
		return r.Getenv
	}
	return func(string) string { return "" }
}

func (r LlamaCppRunner) policy() modelpolicy.Policy {
	if strings.TrimSpace(r.Policy.CacheDir) == "" {
		return modelpolicy.FromConfigSource(modelpolicy.ConfigSource{Getenv: r.getenv()})
	}
	return modelpolicy.NormalizePolicy(r.Policy)
}

func (r LlamaCppRunner) runtimeDecision(status llamacpp.LlamaCppSidecarStatus, spec Spec) modelpolicy.RuntimeDecision {
	policy := r.policy()
	if err := modelpolicy.ValidatePolicy(policy); err != nil {
		return modelpolicy.RuntimeDecision{Allowed: false, Reason: "runtime_policy_invalid"}
	}
	var modelSize uint64
	if status.ModelSizeBytes > 0 {
		modelSize = uint64(status.ModelSizeBytes)
	}
	return modelpolicy.EvaluateRuntimeRequest(policy, modelpolicy.RuntimeRequest{
		ModelID:        spec.ModelID,
		ModelSizeBytes: modelSize,
		Family:         firstNonEmpty(inferRuntimeModelFamily(status.ModelFamilyHint), inferRuntimeModelFamily(status.ModelFilename), inferRuntimeModelFamily(status.ModelPath), inferRuntimeModelFamily(spec.ModelID)),
		CPUOffload:     false,
	})
}

func (opts ExecuteOptions) getenv() func(string) string {
	if opts.Getenv != nil {
		return opts.Getenv
	}
	return func(string) string { return "" }
}

func (opts ExecuteOptions) policy() modelpolicy.Policy {
	if strings.TrimSpace(opts.Policy.CacheDir) == "" {
		return modelpolicy.FromConfigSource(modelpolicy.ConfigSource{Getenv: opts.getenv()})
	}
	return modelpolicy.NormalizePolicy(opts.Policy)
}

func completeWithFallback(ctx context.Context, client llamacpp.CompletionClient, req llamacpp.CompletionRequest) (llamacpp.CompletionResult, error) {
	if client == nil {
		client = llamacpp.OpenAIClient{}
	}
	result, err := client.Complete(ctx, req)
	if err == nil {
		return result, nil
	}
	if req.Stream && llamacpp.IsStreamUnavailable(err) {
		req.Stream = false
		result, err = client.Complete(ctx, req)
		if err == nil {
			return result, nil
		}
	}
	return llamacpp.CompletionResult{}, err
}

func sidecarModelMatches(status llamacpp.LlamaCppSidecarStatus, requested string) bool {
	requested = normalizeModelComparable(requested)
	if requested == "" {
		return false
	}
	for _, candidate := range []string{status.ModelFilename, filepath.Base(status.ModelPath), status.ModelPath} {
		if normalizeModelComparable(candidate) == requested {
			return true
		}
	}
	return false
}

func findDashboardCachedModel(status modelcache.Status, modelID string) (modelcache.Model, bool) {
	status = modelcache.NormalizeStatus(status)
	modelID = cleanText(modelID, maxModelIDLen)
	if modelID == "" {
		return modelcache.Model{}, false
	}
	for _, model := range status.Models {
		if !model.Installed || strings.TrimSpace(model.Path) == "" {
			continue
		}
		if dashboardModelMatch(model, modelID) {
			return model, true
		}
	}
	return modelcache.Model{}, false
}

func dashboardModelMatch(model modelcache.Model, modelID string) bool {
	for _, value := range []string{model.ModelID, model.Filename, filepath.Base(model.Path), model.Path} {
		if modelPathMatches(value, modelID) {
			return true
		}
	}
	return false
}

func modelPathMatches(value string, modelID string) bool {
	want := normalizeModelComparable(modelID)
	if want == "" {
		return false
	}
	return normalizeModelComparable(value) == want ||
		modelAliasToken(value) == modelAliasToken(modelID)
}

func modelAliasToken(value string) string {
	token := normalizeModelComparable(value)
	token = strings.TrimSuffix(token, ".gguf")
	switch {
	case token == "gemma-3-27b-it":
		return token
	case strings.HasPrefix(token, "gemma-3-27b-it-"):
		return "gemma-3-27b-it"
	default:
		return token
	}
}

func normalizeModelComparable(value string) string {
	value = strings.TrimSpace(value)
	if value == "" || value == "." || value == string(filepath.Separator) {
		return ""
	}
	value = strings.ReplaceAll(value, "\\", "/")
	return strings.ToLower(filepath.Base(value))
}

func inferRuntimeModelFamily(value string) string {
	lower := strings.ToLower(strings.TrimSpace(filepath.Base(value)))
	switch {
	case strings.Contains(lower, "llama"):
		return "llama"
	case strings.Contains(lower, "phi"):
		return "phi"
	case strings.Contains(lower, "qwen"):
		return "qwen"
	case strings.Contains(lower, "gemma"):
		return "gemma"
	default:
		return ""
	}
}

func measuredExecutionResult(spec Spec, completion llamacpp.CompletionResult) ExecutionResult {
	tokens := completion.TokensGenerated
	tokenCountEstimated := completion.TokenCountEstimated
	runtimeStatus := normalizeRuntimeMeasurementStatus(completion.RuntimeMeasurementStatus)
	parseStatus := normalizeMetadataParseStatus(completion.MetadataParseStatus)
	if tokens <= 0 {
		tokens = approximateGeneratedTokens(completion.Output)
		if tokens > 0 {
			tokenCountEstimated = true
			runtimeStatus = llamacpp.RuntimeMeasurementStatusPartial
			if parseStatus == "" {
				parseStatus = llamacpp.MetadataParseStatusOK
			}
		}
	}
	if tokens <= 0 {
		runtimeStatus = llamacpp.RuntimeMeasurementStatusUnknown
		parseStatus = llamacpp.MetadataParseStatusPartial
	} else {
		if runtimeStatus == "" {
			runtimeStatus = llamacpp.RuntimeMeasurementStatusMeasured
		}
		if parseStatus == "" {
			parseStatus = llamacpp.MetadataParseStatusOK
		}
	}
	finish := llamacpp.NormalizeCompletionFinishMetadata(completion, spec.MaxTokens, tokens)
	outputBytes := completion.OutputBytes
	if outputBytes <= 0 {
		outputBytes = int64(len(completion.Output))
	}
	ttft := completion.TTFTMs
	if ttft < 0 {
		ttft = 0
	}
	total := completion.TotalTimeMs
	if total < 0 {
		total = 0
	}
	if ttft > total {
		ttft = total
	}
	decodeWindowMs := total - ttft
	if decodeWindowMs <= 0 {
		decodeWindowMs = total
	}
	var decodeTPS float64
	if tokens > 0 && decodeWindowMs > 0 {
		decodeTPS = float64(tokens) / (float64(decodeWindowMs) / 1000)
	}
	var tpotMs float64
	if tokens > 0 && decodeWindowMs > 0 {
		tpotMs = float64(decodeWindowMs) / float64(tokens)
	}
	var endToEndTPS float64
	if tokens > 0 && total > 0 {
		endToEndTPS = float64(tokens) / (float64(total) / 1000)
	}
	if tokens <= 0 {
		runtimeStatus = llamacpp.RuntimeMeasurementStatusUnknown
		parseStatus = llamacpp.MetadataParseStatusPartial
	} else if decodeTPS <= 0 || total <= 0 {
		runtimeStatus = llamacpp.RuntimeMeasurementStatusPartial
	}
	result := ExecutionResult{
		Spec:                     spec,
		Backend:                  spec.Backend,
		ModelID:                  spec.ModelID,
		OutputHash:               HashOutput(spec.JobID, completion.Output),
		OutputBytes:              outputBytes,
		RequestedMaxTokens:       finish.RequestedMaxTokens,
		TokensGenerated:          tokens,
		FinishReason:             finish.FinishReason,
		BackendFinishReason:      finish.BackendFinishReason,
		BackendStopReason:        finish.BackendStopReason,
		MaxTokensReached:         finish.MaxTokensReached,
		TTFTMs:                   ttft,
		TotalTimeMs:              total,
		DecodeTPS:                roundTPS(decodeTPS),
		TPOTMs:                   roundTPS(tpotMs),
		EndToEndTPS:              roundTPS(endToEndTPS),
		ProofStatus:              ProofStatusMeasured,
		RuntimeMeasurementStatus: runtimeStatus,
		MetadataParseStatus:      parseStatus,
		TokenCountEstimated:      tokenCountEstimated,
		GroundingApplied:         groundingApplied(spec),
		PromptMode:               firstNonEmpty(completion.PromptMode, llamacpp.PromptModeChatMessages),
		SystemPromptHash:         firstNonEmpty(cleanHash(completion.SystemPromptHash), specSystemPromptHash(spec)),
		MaxReturnChars:           spec.MaxReturnChars,
	}
	if spec.ReturnText {
		result.GeneratedText, result.GeneratedTextTruncated = truncateGeneratedText(completion.Output, spec.MaxReturnChars)
	}
	return result
}

func groundingApplied(spec Spec) bool {
	spec = normalizeSpec(spec)
	if strings.TrimSpace(spec.SystemPrompt) != "" {
		return true
	}
	for _, message := range spec.Messages {
		if message.Role == "system" && strings.TrimSpace(message.Content) != "" {
			return true
		}
	}
	return false
}

func specSystemPromptHash(spec Spec) string {
	spec = normalizeSpec(spec)
	if hash := llamacpp.HashSystemPrompt(spec.SystemPrompt); hash != "" {
		return hash
	}
	for _, message := range spec.Messages {
		if message.Role == "system" {
			return llamacpp.HashSystemPrompt(message.Content)
		}
	}
	return ""
}

func truncateGeneratedText(output []byte, maxChars int) (string, bool) {
	if maxChars <= 0 || maxChars > defaultMaxReturnChars {
		maxChars = defaultMaxReturnChars
	}
	runes := []rune(string(output))
	if len(runes) <= maxChars {
		return string(runes), false
	}
	return string(runes[:maxChars]), true
}

func failedExecutionResult(spec Spec, code string) ExecutionResult {
	code = cleanErrorCode(code)
	if code == "" {
		code = "dashboard_inference_failed"
	}
	return ExecutionResult{
		Spec:                     spec,
		Backend:                  spec.Backend,
		ModelID:                  spec.ModelID,
		OutputHash:               HashOutput(spec.JobID, nil),
		RequestedMaxTokens:       spec.MaxTokens,
		FinishReason:             finishReasonFromErrorCode(code),
		BackendFinishReason:      llamacpp.FinishReasonUnknown,
		BackendStopReason:        llamacpp.FinishReasonUnknown,
		ProofStatus:              ProofStatusRejected,
		RuntimeMeasurementStatus: llamacpp.RuntimeMeasurementStatusUnknown,
		MetadataParseStatus:      llamacpp.MetadataParseStatusOK,
		ErrorCode:                code,
		MaxReturnChars:           spec.MaxReturnChars,
	}
}

func finishReasonFromErrorCode(code string) string {
	code = strings.ToLower(strings.TrimSpace(code))
	if code == "" {
		return llamacpp.FinishReasonError
	}
	if strings.Contains(code, "timeout") || strings.Contains(code, "deadline") {
		return llamacpp.FinishReasonTimeout
	}
	return llamacpp.FinishReasonError
}

func safeSidecarFailure(status llamacpp.LlamaCppSidecarStatus) string {
	if code := safeSidecarLaunchErrorCode(status); code != "" {
		return code
	}
	switch {
	case !status.Enabled:
		return "llamacpp_sidecar_disabled"
	case !status.Available:
		return "llamacpp_sidecar_unavailable"
	case !status.Running:
		return "llamacpp_sidecar_not_running"
	case !status.Healthy:
		return "llamacpp_sidecar_unhealthy"
	default:
		return "llamacpp_sidecar_unavailable"
	}
}

func safeSidecarLaunchErrorCode(status llamacpp.LlamaCppSidecarStatus) string {
	detail := strings.ToLower(strings.TrimSpace(firstNonEmpty(status.LastError, status.Reason)))
	if detail == "" {
		return ""
	}
	switch {
	case strings.Contains(detail, "out of memory") ||
		strings.Contains(detail, "cuda_malloc") ||
		strings.Contains(detail, "cudamalloc") ||
		strings.Contains(detail, "cuda error 2") ||
		strings.Contains(detail, "cuda error: 2"):
		return "llamacpp_cuda_out_of_memory"
	case strings.Contains(detail, "cudart") ||
		strings.Contains(detail, "cublas") ||
		strings.Contains(detail, "cudnn"):
		if strings.Contains(detail, "missing") ||
			strings.Contains(detail, "not found") ||
			strings.Contains(detail, "load") ||
			strings.Contains(detail, "dll") {
			return "llamacpp_cuda_runtime_missing"
		}
	case strings.Contains(detail, "unknown argument") ||
		strings.Contains(detail, "unrecognized option") ||
		strings.Contains(detail, "invalid argument") ||
		strings.Contains(detail, "unknown option"):
		return "llamacpp_launch_arg_unsupported"
	case strings.Contains(detail, "failed to load model") ||
		strings.Contains(detail, "unable to load model") ||
		strings.Contains(detail, "model load"):
		return "llamacpp_model_load_failed"
	case strings.Contains(detail, "deadline exceeded") ||
		strings.Contains(detail, "context canceled") ||
		strings.Contains(detail, "context cancelled"):
		return "llamacpp_sidecar_timeout"
	}
	return ""
}

func safeErrorCode(err error) string {
	if err == nil {
		return ""
	}
	code := ErrorCode(err)
	if code == "" {
		return "dashboard_inference_failed"
	}
	return code
}

func approximateGeneratedTokens(output []byte) int64 {
	count := int64(len(strings.Fields(string(output))))
	if count == 0 && strings.TrimSpace(string(output)) != "" {
		count = 1
	}
	return count
}

func roundTPS(value float64) float64 {
	if value <= 0 || math.IsNaN(value) || math.IsInf(value, 0) {
		return 0
	}
	return math.Round(value*1000) / 1000
}

func finiteTPS(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value >= 0
}

func normalizeRuntimeMeasurementStatus(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case llamacpp.RuntimeMeasurementStatusMeasured:
		return llamacpp.RuntimeMeasurementStatusMeasured
	case llamacpp.RuntimeMeasurementStatusPartial:
		return llamacpp.RuntimeMeasurementStatusPartial
	case llamacpp.RuntimeMeasurementStatusUnknown:
		return llamacpp.RuntimeMeasurementStatusUnknown
	default:
		return ""
	}
}

func normalizeMetadataParseStatus(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case llamacpp.MetadataParseStatusOK:
		return llamacpp.MetadataParseStatusOK
	case llamacpp.MetadataParseStatusPartial:
		return llamacpp.MetadataParseStatusPartial
	default:
		return ""
	}
}

func (s *LocalStatus) RecordSeen(runID, jobID string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastRunID = cleanText(runID, maxIDLen)
	s.snapshot.LastJobID = cleanText(jobID, maxIDLen)
	s.snapshot.LastError = ""
	s.snapshot.Counters.Seen++
}

func (s *LocalStatus) RecordExecuted(runID, jobID string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastRunID = cleanText(runID, maxIDLen)
	s.snapshot.LastJobID = cleanText(jobID, maxIDLen)
	s.snapshot.LastError = ""
	s.snapshot.Counters.Executed++
}

func (s *LocalStatus) RecordRejected(runID, jobID string, err error) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastRunID = cleanText(runID, maxIDLen)
	s.snapshot.LastJobID = cleanText(jobID, maxIDLen)
	s.snapshot.LastError = cleanLocalStatusError(err)
	s.snapshot.Counters.Rejected++
}

func (s *LocalStatus) RecordReceiptSubmitted(runID, jobID string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastRunID = cleanText(runID, maxIDLen)
	s.snapshot.LastJobID = cleanText(jobID, maxIDLen)
	s.snapshot.LastError = ""
	s.snapshot.Counters.ReceiptSubmitted++
}

func (s *LocalStatus) RecordReceiptFailed(runID, jobID string, err error) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastRunID = cleanText(runID, maxIDLen)
	s.snapshot.LastJobID = cleanText(jobID, maxIDLen)
	s.snapshot.LastError = cleanLocalStatusError(err)
	s.snapshot.Counters.ReceiptFailed++
}

func (s *LocalStatus) RecordError(runID, jobID string, err error) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.snapshot.LastRunID = cleanText(runID, maxIDLen)
	s.snapshot.LastJobID = cleanText(jobID, maxIDLen)
	s.snapshot.LastError = cleanLocalStatusError(err)
}

func (s *LocalStatus) Snapshot() LocalStatusSnapshot {
	if s == nil {
		return LocalStatusSnapshot{}
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.snapshot
}

func cleanLocalStatusError(err error) string {
	if err == nil {
		return ""
	}
	value := ErrorCode(err)
	if value == "" {
		value = "dashboard_inference_error"
	}
	return cleanText(value, maxStatusErrLen)
}

type codedError struct {
	code string
	err  error
}

func (e codedError) Error() string {
	if e.err == nil {
		return e.code
	}
	return fmt.Sprintf("%s: %v", e.code, e.err)
}

func (e codedError) Unwrap() error {
	return e.err
}

func ErrorCode(err error) string {
	if err == nil {
		return ""
	}
	var coded interface{ Code() string }
	if errors.As(err, &coded) {
		return cleanErrorCode(coded.Code())
	}
	var local codedError
	if errors.As(err, &local) {
		return cleanErrorCode(local.code)
	}
	return cleanErrorCode(err.Error())
}

func (e codedError) Code() string {
	return e.code
}

func cleanErrorCode(value string) string {
	value = cleanText(value, maxStatusErrLen)
	if value == "" {
		return ""
	}
	lower := strings.ToLower(value)
	for _, marker := range forbiddenTextMarkers() {
		if strings.Contains(lower, strings.ToLower(marker)) {
			return "dashboard_inference_error_redacted"
		}
	}
	return value
}
