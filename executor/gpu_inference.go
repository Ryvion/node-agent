//go:build containers

package executor

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/client"
)

type GPUInferenceExecutor struct {
	dockerClient *client.Client
}

type InferenceRequest struct {
	Model    string                 `json:"model"`
	Prompt   string                 `json:"prompt"`
	Params   map[string]interface{} `json:"params"`
	JobID    string                 `json:"job_id"`
	InputURL string                 `json:"input_url"`
}

type InferenceResult struct {
	JobID      string  `json:"job_id"`
	Output     string  `json:"output"`
	TokenCount int     `json:"token_count"`
	Duration   float64 `json:"duration_seconds"`
	GPUUsage   float64 `json:"gpu_usage_percent"`
	Error      string  `json:"error,omitempty"`
}

func NewGPUInferenceExecutor() (*GPUInferenceExecutor, error) {
	cli, err := client.NewClientWithOpts(client.FromEnv)
	if err != nil {
		return nil, fmt.Errorf("failed to create Docker client: %w", err)
	}
	return &GPUInferenceExecutor{dockerClient: cli}, nil
}

func (g *GPUInferenceExecutor) ExecuteStableDiffusion(ctx context.Context, req *InferenceRequest) (*InferenceResult, error) {
	workDir := filepath.Join("/tmp", fmt.Sprintf("sd_%s", req.JobID))
	if err := os.MkdirAll(workDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create work directory: %w", err)
	}
	defer os.RemoveAll(workDir)

	// Download input if URL provided
	if req.InputURL != "" {
		if err := g.downloadFile(req.InputURL, filepath.Join(workDir, "input.json")); err != nil {
			return nil, fmt.Errorf("failed to download input: %w", err)
		}
	}

	// Create input JSON for the container
	inputData := map[string]interface{}{
		"prompt": req.Prompt,
		"width":  getIntParam(req.Params, "width", 512),
		"height": getIntParam(req.Params, "height", 512),
		"steps":  getIntParam(req.Params, "steps", 20),
	}

	inputBytes, _ := json.Marshal(inputData)
	if err := os.WriteFile(filepath.Join(workDir, "request.json"), inputBytes, 0644); err != nil {
		return nil, fmt.Errorf("failed to write request: %w", err)
	}

	// Run Stable Diffusion container
	containerConfig := &container.Config{
		Image: "stabilityai/stable-diffusion:latest",
		Cmd: []string{
			"python", "/app/inference.py",
			"--input", "/work/request.json",
			"--output", "/work/output.png",
			"--json-output", "/work/result.json",
		},
		Env: []string{
			"CUDA_VISIBLE_DEVICES=0",
			"PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128",
		},
		WorkingDir: "/app",
	}

	hostConfig := &container.HostConfig{
		Binds: []string{fmt.Sprintf("%s:/work", workDir)},
		Resources: container.Resources{
			DeviceRequests: []container.DeviceRequest{
				{
					Count:        1,
					Capabilities: [][]string{{"gpu"}},
				},
			},
			Memory:   8 * 1024 * 1024 * 1024, // 8GB
			NanoCPUs: 4 * 1000000000,         // 4 CPUs
		},
		AutoRemove: true,
	}

	return g.runContainer(ctx, containerConfig, hostConfig, workDir, req.JobID)
}

func (g *GPUInferenceExecutor) ExecuteLLMInference(ctx context.Context, req *InferenceRequest) (*InferenceResult, error) {
	workDir := filepath.Join("/tmp", fmt.Sprintf("llm_%s", req.JobID))
	if err := os.MkdirAll(workDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create work directory: %w", err)
	}
	defer os.RemoveAll(workDir)

	// Create input for LLM
	inputData := map[string]interface{}{
		"model":       req.Model,
		"prompt":      req.Prompt,
		"max_tokens":  getIntParam(req.Params, "max_tokens", 100),
		"temperature": getFloatParam(req.Params, "temperature", 0.7),
	}

	inputBytes, _ := json.Marshal(inputData)
	if err := os.WriteFile(filepath.Join(workDir, "request.json"), inputBytes, 0644); err != nil {
		return nil, fmt.Errorf("failed to write request: %w", err)
	}

	var image string
	switch req.Model {
	case "llama2-7b", "llama2-13b":
		image = "huggingface/transformers-pytorch-gpu:4.21.0"
	case "codellama":
		image = "codellama/codellama:13b-instruct"
	default:
		image = "huggingface/transformers-pytorch-gpu:latest"
	}

	containerConfig := &container.Config{
		Image: image,
		Cmd: []string{
			"python", "-c", `
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

with open('/work/request.json') as f:
    req = json.load(f)

start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained(req['model'])
model = AutoModelForCausalLM.from_pretrained(req['model'], torch_dtype=torch.float16, device_map='auto')

inputs = tokenizer(req['prompt'], return_tensors='pt').to('cuda')
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=req['max_tokens'], temperature=req['temperature'])

result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
duration = time.time() - start_time

result = {
    'output': result_text,
    'token_count': len(outputs[0]),
    'duration_seconds': duration,
    'model': req['model']
}

with open('/work/result.json', 'w') as f:
    json.dump(result, f)
`,
		},
		Env: []string{
			"CUDA_VISIBLE_DEVICES=0",
			"TRANSFORMERS_CACHE=/work/cache",
			"HF_HOME=/work/cache",
		},
		WorkingDir: "/work",
	}

	hostConfig := &container.HostConfig{
		Binds: []string{fmt.Sprintf("%s:/work", workDir)},
		Resources: container.Resources{
			DeviceRequests: []container.DeviceRequest{
				{
					Count:        1,
					Capabilities: [][]string{{"gpu"}},
				},
			},
			Memory:   16 * 1024 * 1024 * 1024, // 16GB for LLMs
			NanoCPUs: 8 * 1000000000,          // 8 CPUs
		},
		AutoRemove: true,
	}

	return g.runContainer(ctx, containerConfig, hostConfig, workDir, req.JobID)
}

func (g *GPUInferenceExecutor) ExecuteWhisperTranscription(ctx context.Context, req *InferenceRequest) (*InferenceResult, error) {
	workDir := filepath.Join("/tmp", fmt.Sprintf("whisper_%s", req.JobID))
	if err := os.MkdirAll(workDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create work directory: %w", err)
	}
	defer os.RemoveAll(workDir)

	// Download audio file
	if req.InputURL == "" {
		return nil, fmt.Errorf("input_url required for Whisper transcription")
	}

	audioPath := filepath.Join(workDir, "input.wav")
	if err := g.downloadFile(req.InputURL, audioPath); err != nil {
		return nil, fmt.Errorf("failed to download audio: %w", err)
	}

	containerConfig := &container.Config{
		Image: "openai/whisper:latest",
		Cmd: []string{
			"whisper", "/work/input.wav",
			"--model", getStringParam(req.Params, "model", "base"),
			"--output_dir", "/work",
			"--output_format", "json",
			"--device", "cuda",
		},
		Env: []string{"CUDA_VISIBLE_DEVICES=0"},
	}

	hostConfig := &container.HostConfig{
		Binds: []string{fmt.Sprintf("%s:/work", workDir)},
		Resources: container.Resources{
			DeviceRequests: []container.DeviceRequest{
				{
					Count:        1,
					Capabilities: [][]string{{"gpu"}},
				},
			},
			Memory:   4 * 1024 * 1024 * 1024, // 4GB
			NanoCPUs: 2 * 1000000000,         // 2 CPUs
		},
		AutoRemove: true,
	}

	return g.runContainer(ctx, containerConfig, hostConfig, workDir, req.JobID)
}

func (g *GPUInferenceExecutor) runContainer(ctx context.Context, config *container.Config, hostConfig *container.HostConfig, workDir, jobID string) (*InferenceResult, error) {
	startTime := time.Now()

	// Create and start container
	resp, err := g.dockerClient.ContainerCreate(ctx, config, hostConfig, nil, nil, fmt.Sprintf("akatosh_%s", jobID))
	if err != nil {
		return nil, fmt.Errorf("failed to create container: %w", err)
	}

	if err := g.dockerClient.ContainerStart(ctx, resp.ID, types.ContainerStartOptions{}); err != nil {
		return nil, fmt.Errorf("failed to start container: %w", err)
	}

	// Wait for completion with timeout
	statusCh, errCh := g.dockerClient.ContainerWait(ctx, resp.ID, container.WaitConditionNotRunning)
	select {
	case err := <-errCh:
		if err != nil {
			return nil, fmt.Errorf("container wait error: %w", err)
		}
	case status := <-statusCh:
		if status.StatusCode != 0 {
			// Get container logs for debugging
			logs, _ := g.getContainerLogs(ctx, resp.ID)
			return &InferenceResult{
				JobID:    jobID,
				Error:    fmt.Sprintf("container exited with code %d. Logs: %s", status.StatusCode, logs),
				Duration: time.Since(startTime).Seconds(),
			}, nil
		}
	case <-ctx.Done():
		g.dockerClient.ContainerStop(context.Background(), resp.ID, nil)
		return nil, ctx.Err()
	}

	duration := time.Since(startTime)

	// Read result file
	resultPath := filepath.Join(workDir, "result.json")
	if _, err := os.Stat(resultPath); os.IsNotExist(err) {
		return &InferenceResult{
			JobID:    jobID,
			Error:    "no result.json produced",
			Duration: duration.Seconds(),
		}, nil
	}

	resultBytes, err := os.ReadFile(resultPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read result: %w", err)
	}

	var result map[string]interface{}
	if err := json.Unmarshal(resultBytes, &result); err != nil {
		return nil, fmt.Errorf("failed to parse result: %w", err)
	}

	return &InferenceResult{
		JobID:      jobID,
		Output:     getStringFromMap(result, "output"),
		TokenCount: getIntFromMap(result, "token_count"),
		Duration:   duration.Seconds(),
		GPUUsage:   g.measureGPUUsage(),
	}, nil
}

func (g *GPUInferenceExecutor) downloadFile(url, filepath string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

func (g *GPUInferenceExecutor) getContainerLogs(ctx context.Context, containerID string) (string, error) {
	logs, err := g.dockerClient.ContainerLogs(ctx, containerID, types.ContainerLogsOptions{
		ShowStdout: true,
		ShowStderr: true,
		Timestamps: false,
	})
	if err != nil {
		return "", err
	}
	defer logs.Close()

	logBytes, err := io.ReadAll(logs)
	if err != nil {
		return "", err
	}

	return string(logBytes), nil
}

func (g *GPUInferenceExecutor) measureGPUUsage() float64 {
	// Try to query utilization via nvidia-smi; return 0 if unavailable
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	// Query per-GPU utilization (no units, no header)
	cmd := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits")
	out, err := cmd.Output()
	if err != nil || len(out) == 0 {
		return 0.0
	}
	// Parse lines like: "12" or "12 %" depending on driver; we used nounits, so expect plain ints
	scanner := bufio.NewScanner(strings.NewReader(string(out)))
	var sum float64
	var count int
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		// Be tolerant: strip possible trailing '%' just in case
		line = strings.TrimSuffix(line, "%")
		v, err := strconv.ParseFloat(strings.TrimSpace(line), 64)
		if err != nil {
			continue
		}
		sum += v
		count++
	}
	if count == 0 {
		return 0.0
	}
	avg := sum / float64(count)
	if avg < 0 {
		avg = 0
	}
	if avg > 100 {
		avg = 100
	}
	return avg
}

// Helper functions
func getIntParam(params map[string]interface{}, key string, defaultVal int) int {
	if val, ok := params[key]; ok {
		if intVal, ok := val.(float64); ok {
			return int(intVal)
		}
	}
	return defaultVal
}

func getFloatParam(params map[string]interface{}, key string, defaultVal float64) float64 {
	if val, ok := params[key]; ok {
		if floatVal, ok := val.(float64); ok {
			return floatVal
		}
	}
	return defaultVal
}

func getStringParam(params map[string]interface{}, key string, defaultVal string) string {
	if val, ok := params[key]; ok {
		if strVal, ok := val.(string); ok {
			return strVal
		}
	}
	return defaultVal
}

func getStringFromMap(m map[string]interface{}, key string) string {
	if val, ok := m[key]; ok {
		if strVal, ok := val.(string); ok {
			return strVal
		}
	}
	return ""
}

func getIntFromMap(m map[string]interface{}, key string) int {
	if val, ok := m[key]; ok {
		if intVal, ok := val.(float64); ok {
			return int(intVal)
		}
	}
	return 0
}
