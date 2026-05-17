package modelbench

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"runtime"
	"strings"
	"time"
)

const (
	defaultNativeBenchmarkAgentVersion = "dev"
	defaultNativeBenchmarkErrorCode    = "native_inference_failed"
)

type NativeInference interface {
	Healthy() bool
	EnsureModel(context.Context, string) error
	ServerURL() string
	ModelName() string
}

type NativeInferenceModelBenchmarkRunner struct {
	Native NativeInference

	AgentVersion     string
	OS               string
	Arch             string
	GPUDetected      bool
	GPUModel         string
	RuntimeAvailable func() bool
	HTTPClient       *http.Client
	Prompt           ModelBenchmarkPrompt
	Now              func() time.Time
}

func (r NativeInferenceModelBenchmarkRunner) RunModelBenchmark(ctx context.Context, spec ModelBenchmarkSpec) (ModelBenchmarkResult, error) {
	spec = normalizeModelBenchmarkSpec(spec)
	if err := ValidateModelBenchmarkSpec(spec); err != nil {
		return ModelBenchmarkResult{}, err
	}

	started := r.now()
	runCtx, cancel := context.WithTimeout(ctx, time.Duration(spec.TimeoutMs)*time.Millisecond)
	defer cancel()

	nativeSupported := r.nativeRuntimeAvailable()
	if !nativeSupported {
		return r.unavailableResult(spec, started, "native_runtime_unavailable"), nil
	}
	if nativeInferenceIsNil(r.Native) {
		return r.unavailableResult(spec, started, "native_inference_manager_unavailable"), nil
	}

	if err := r.Native.EnsureModel(runCtx, spec.ModelID); err != nil {
		return r.unavailableResult(spec, started, "native_model_unavailable"), nil
	}
	if !r.Native.Healthy() {
		return r.unavailableResult(spec, started, "native_inference_not_ready"), nil
	}

	serverURL := strings.TrimRight(strings.TrimSpace(r.Native.ServerURL()), "/")
	if serverURL == "" {
		return r.unavailableResult(spec, started, "native_server_url_unavailable"), nil
	}

	result, err := r.runChatCompletion(runCtx, spec, started, serverURL)
	if err != nil {
		return result, err
	}
	return result, nil
}

func (r NativeInferenceModelBenchmarkRunner) runChatCompletion(ctx context.Context, spec ModelBenchmarkSpec, started time.Time, serverURL string) (ModelBenchmarkResult, error) {
	reqBody, err := json.Marshal(nativeChatRequest{
		Model: spec.ModelID,
		Messages: []nativeChatMessage{
			{Role: "system", Content: "You are running a local native model benchmark self-test. Answer concisely."},
			{Role: "user", Content: string(r.benchmarkPrompt().Content)},
		},
		Stream:      true,
		MaxTokens:   spec.MaxTokens,
		Temperature: spec.Temperature,
	})
	if err != nil {
		return r.failedResult(spec, started, "native_request_marshal_failed"), ModelBenchmarkError{Code: "native_request_marshal_failed", Message: "failed to build local inference request"}
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, serverURL+"/v1/chat/completions", bytes.NewReader(reqBody))
	if err != nil {
		return r.failedResult(spec, started, "native_request_build_failed"), ModelBenchmarkError{Code: "native_request_build_failed", Message: "failed to build local inference request"}
	}
	req.Header.Set("Content-Type", "application/json")

	client := r.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return r.failedResult(spec, started, "native_request_failed"), ModelBenchmarkError{Code: "native_request_failed", Message: "local inference request failed"}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
		return r.failedResult(spec, started, "native_response_status_failed"), ModelBenchmarkError{Code: "native_response_status_failed", Message: fmt.Sprintf("local inference returned HTTP %d", resp.StatusCode)}
	}

	output, tokensGenerated, promptTokens, completionTokens, firstTokenAt, err := readNativeChatStream(resp.Body, started)
	if err != nil {
		return r.failedResult(spec, started, defaultNativeBenchmarkErrorCode), err
	}
	if output.Len() == 0 {
		return r.failedResult(spec, started, "native_empty_output"), ModelBenchmarkError{Code: "native_empty_output", Message: "local inference produced no output"}
	}
	if completionTokens != nil {
		tokensGenerated = *completionTokens
	}

	finished := r.now()
	wall := finished.Sub(started)
	if wall < 0 {
		wall = 0
	}
	var timeToFirstTokenMs int64
	if !firstTokenAt.IsZero() && !firstTokenAt.Before(started) {
		timeToFirstTokenMs = firstTokenAt.Sub(started).Milliseconds()
	}
	var tokensPerSecond float64
	if tokensGenerated > 0 && wall > 0 {
		tokensPerSecond = float64(tokensGenerated) / wall.Seconds()
	}

	result := ModelBenchmarkResult{
		RequestID:   spec.RequestID,
		JobID:       spec.JobID,
		ModelID:     spec.ModelID,
		PromptHash:  spec.PromptHash,
		RuntimeInfo: r.runtimeInfo(spec.ModelID, true, true),
		Metrics: ModelBenchmarkMetrics{
			StartedAtUnixMs:    started.UnixMilli(),
			FinishedAtUnixMs:   finished.UnixMilli(),
			WallTimeMs:         wall.Milliseconds(),
			TimeToFirstTokenMs: timeToFirstTokenMs,
			TokensGenerated:    tokensGenerated,
			TokensPerSecond:    tokensPerSecond,
			PromptTokens:       promptTokens,
			CompletionTokens:   completionTokens,
			ModelLoadState:     ModelBenchmarkModelLoadStateLoaded,
		},
		OutputHash:  hashBenchmarkOutput([]byte(output.String())),
		OutputBytes: int64(output.Len()),
		ProofStatus: ModelBenchmarkProofStatusMeasured,
	}
	if err := ValidateModelBenchmarkResult(result); err != nil {
		return result, err
	}
	return result, nil
}

func readNativeChatStream(body io.Reader, started time.Time) (strings.Builder, int64, *int64, *int64, time.Time, error) {
	var output strings.Builder
	var tokensGenerated int64
	var promptTokens *int64
	var completionTokens *int64
	var firstTokenAt time.Time

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data: "))
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			break
		}

		var chunk nativeChatStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return output, tokensGenerated, promptTokens, completionTokens, firstTokenAt, ModelBenchmarkError{Code: "native_stream_decode_failed", Message: "local inference stream was not valid JSON"}
		}
		if chunk.Error.Message != "" {
			return output, tokensGenerated, promptTokens, completionTokens, firstTokenAt, ModelBenchmarkError{Code: "native_stream_error", Message: "local inference stream reported an error"}
		}
		if chunk.Usage != nil {
			if chunk.Usage.PromptTokens >= 0 {
				v := chunk.Usage.PromptTokens
				promptTokens = &v
			}
			if chunk.Usage.CompletionTokens >= 0 {
				v := chunk.Usage.CompletionTokens
				completionTokens = &v
			}
		}
		for _, choice := range chunk.Choices {
			content := choice.Delta.Content
			if content == "" {
				content = choice.Text
			}
			if content == "" {
				continue
			}
			if firstTokenAt.IsZero() {
				firstTokenAt = time.Now()
				if firstTokenAt.Before(started) {
					firstTokenAt = started
				}
			}
			tokensGenerated++
			output.WriteString(content)
		}
	}
	if err := scanner.Err(); err != nil {
		return output, tokensGenerated, promptTokens, completionTokens, firstTokenAt, ModelBenchmarkError{Code: "native_stream_read_failed", Message: "failed reading local inference stream"}
	}
	return output, tokensGenerated, promptTokens, completionTokens, firstTokenAt, nil
}

func (r NativeInferenceModelBenchmarkRunner) unavailableResult(spec ModelBenchmarkSpec, started time.Time, code string) ModelBenchmarkResult {
	finished := r.now()
	if finished.Before(started) {
		finished = started
	}
	result := ModelBenchmarkResult{
		RequestID:   spec.RequestID,
		JobID:       spec.JobID,
		ModelID:     spec.ModelID,
		PromptHash:  spec.PromptHash,
		RuntimeInfo: r.runtimeInfo(spec.ModelID, r.nativeRuntimeAvailable(), false),
		Metrics: ModelBenchmarkMetrics{
			StartedAtUnixMs:    started.UnixMilli(),
			FinishedAtUnixMs:   finished.UnixMilli(),
			WallTimeMs:         finished.Sub(started).Milliseconds(),
			TimeToFirstTokenMs: 0,
			TokensGenerated:    0,
			TokensPerSecond:    0,
			ModelLoadState:     ModelBenchmarkModelLoadStateUnavailable,
			ErrorCode:          cleanModelBenchmarkCode(code),
		},
		OutputBytes: 0,
		ProofStatus: ModelBenchmarkProofStatusUnavailable,
	}
	return result
}

func (r NativeInferenceModelBenchmarkRunner) failedResult(spec ModelBenchmarkSpec, started time.Time, code string) ModelBenchmarkResult {
	finished := r.now()
	if finished.Before(started) {
		finished = started
	}
	result := ModelBenchmarkResult{
		RequestID:   spec.RequestID,
		JobID:       spec.JobID,
		ModelID:     spec.ModelID,
		PromptHash:  spec.PromptHash,
		RuntimeInfo: r.runtimeInfo(spec.ModelID, true, true),
		Metrics: ModelBenchmarkMetrics{
			StartedAtUnixMs:    started.UnixMilli(),
			FinishedAtUnixMs:   finished.UnixMilli(),
			WallTimeMs:         finished.Sub(started).Milliseconds(),
			TimeToFirstTokenMs: 0,
			TokensGenerated:    0,
			TokensPerSecond:    0,
			ModelLoadState:     ModelBenchmarkModelLoadStateFailed,
			ErrorCode:          cleanModelBenchmarkCode(code),
		},
		OutputBytes: 0,
		ProofStatus: ModelBenchmarkProofStatusFailed,
	}
	return result
}

func (r NativeInferenceModelBenchmarkRunner) runtimeInfo(modelID string, nativeSupported bool, nativeReady bool) ModelBenchmarkRuntimeInfo {
	agentVersion := strings.TrimSpace(r.AgentVersion)
	if agentVersion == "" {
		agentVersion = defaultNativeBenchmarkAgentVersion
	}
	osName := strings.TrimSpace(r.OS)
	if osName == "" {
		osName = runtime.GOOS
	}
	arch := strings.TrimSpace(r.Arch)
	if arch == "" {
		arch = runtime.GOARCH
	}
	return ModelBenchmarkRuntimeInfo{
		AgentVersion:             agentVersion,
		OS:                       osName,
		Arch:                     arch,
		NativeInferenceSupported: nativeSupported,
		NativeInferenceReady:     nativeReady,
		RuntimeKind:              ModelBenchmarkRuntimeKindNativeLocal,
		ModelID:                  strings.TrimSpace(modelID),
		ModelLoaded:              nativeReady,
		GPUDetected:              r.GPUDetected,
		GPUModel:                 strings.TrimSpace(r.GPUModel),
	}
}

func (r NativeInferenceModelBenchmarkRunner) nativeRuntimeAvailable() bool {
	if r.RuntimeAvailable != nil {
		return r.RuntimeAvailable()
	}
	return r.Native != nil
}

func (r NativeInferenceModelBenchmarkRunner) benchmarkPrompt() ModelBenchmarkPrompt {
	if len(r.Prompt.Content) > 0 || strings.TrimSpace(r.Prompt.Label) != "" {
		return r.Prompt
	}
	return DefaultModelBenchmarkPrompt()
}

func (r NativeInferenceModelBenchmarkRunner) now() time.Time {
	if r.Now != nil {
		if now := r.Now(); !now.IsZero() {
			return now
		}
	}
	return time.Now()
}

func hashBenchmarkOutput(output []byte) string {
	hash := sha256.New()
	hash.Write([]byte("ryvion:v7:model_benchmark_output:v1\n"))
	hash.Write(output)
	return "sha256:" + hex.EncodeToString(hash.Sum(nil))
}

func cleanModelBenchmarkCode(code string) string {
	code = strings.TrimSpace(code)
	if code == "" {
		return defaultNativeBenchmarkErrorCode
	}
	return code
}

func nativeInferenceIsNil(native NativeInference) bool {
	if native == nil {
		return true
	}
	value := reflect.ValueOf(native)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		return value.IsNil()
	default:
		return false
	}
}

type nativeChatRequest struct {
	Model       string              `json:"model"`
	Messages    []nativeChatMessage `json:"messages"`
	Stream      bool                `json:"stream"`
	MaxTokens   int                 `json:"max_tokens,omitempty"`
	Temperature float64             `json:"temperature,omitempty"`
}

type nativeChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type nativeChatStreamChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		Text string `json:"text"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
	} `json:"usage"`
	Error struct {
		Message string `json:"message"`
	} `json:"error"`
}
