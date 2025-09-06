//go:build containers

package executor

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/client"
)

type WorkloadType string

const (
	WorkloadTypeInference   WorkloadType = "inference"
	WorkloadTypeTraining    WorkloadType = "training"
	WorkloadTypeTranscoding WorkloadType = "transcoding"
	WorkloadTypeRendering   WorkloadType = "rendering"
	WorkloadTypeMining      WorkloadType = "mining"
)

type WorkRequest struct {
	JobId      string `json:"job_id"`
	JobType    string `json:"job_type"`
	Parameters []byte `json:"parameters"`
}

type WorkloadExecutor struct {
	dockerClient    *client.Client
	gpuCapabilities GPUCapabilities
	activeJobs      map[string]*JobExecution
	hubBaseURL      string
}

type GPUCapabilities struct {
	Model       string
	VRAM        int64
	CUDACores   int
	TFLOPs      float64
	Utilization float64
}

type JobExecution struct {
	JobID       string
	ContainerID string
	StartTime   time.Time
	WorkType    WorkloadType
	Status      string
}

type WorkResult struct {
	JobID      string
	ResultHash string
	OutputData []byte
	Metrics    ExecutionMetrics
	Error      error
}

type ExecutionMetrics struct {
	Duration       time.Duration
	GPUUtilization float64
	PowerUsage     float64
	TokensPerSec   float64
}

func NewWorkloadExecutor() (*WorkloadExecutor, error) {
	cli, err := client.NewClientWithOpts(client.FromEnv)
	if err != nil {
		return nil, fmt.Errorf("failed to create Docker client: %w", err)
	}

	return &WorkloadExecutor{
		dockerClient:    cli,
		activeJobs:      make(map[string]*JobExecution),
		gpuCapabilities: detectGPUCapabilities(),
		hubBaseURL:      "https://ryvion-hub.onrender.com",
	}, nil
}

func (e *WorkloadExecutor) CanExecute(job *WorkRequest) bool {
	switch job.JobType {
	case "inference":
		return e.gpuCapabilities.VRAM >= 4096
	case "training":
		return e.gpuCapabilities.VRAM >= 8192
	case "transcoding":
		return e.gpuCapabilities.CUDACores > 0
	default:
		return false
	}
}

func (e *WorkloadExecutor) ExecuteJob(ctx context.Context, job *WorkRequest) (*WorkResult, error) {
	execution := &JobExecution{
		JobID:     job.JobId,
		StartTime: time.Now(),
		WorkType:  WorkloadType(job.JobType),
		Status:    "starting",
	}
	e.activeJobs[job.JobId] = execution

	switch WorkloadType(job.JobType) {
	case WorkloadTypeInference:
		return e.executeInferenceJob(ctx, job)
	case WorkloadTypeTraining:
		return e.executeTraining(ctx, job)
	case WorkloadTypeTranscoding:
		return e.executeTranscoding(ctx, job)
	case WorkloadTypeRendering:
		return e.executeRendering(ctx, job)
	default:
		return nil, fmt.Errorf("unsupported workload type: %s", job.JobType)
	}
}

func (e *WorkloadExecutor) ExecuteInference(ctx context.Context, jobID, jobType, payloadURL string) (*WorkResult, error) {
	execution := &JobExecution{
		JobID:     jobID,
		StartTime: time.Now(),
		WorkType:  WorkloadTypeInference,
		Status:    "running",
	}
	e.activeJobs[jobID] = execution

	gpuExecutor, err := NewGPUInferenceExecutor()
	if err == nil {
		var req *InferenceRequest
		switch jobType {
		case "stable-diffusion":
			req = &InferenceRequest{
				Model:    "stable-diffusion-v1-5",
				Prompt:   "high quality digital art",
				JobID:    jobID,
				InputURL: payloadURL,
				Params:   map[string]interface{}{"steps": 20, "width": 512, "height": 512},
			}
		case "llm-inference":
			req = &InferenceRequest{
				Model:    "llama2-7b",
				Prompt:   "Complete this text:",
				JobID:    jobID,
				InputURL: payloadURL,
				Params:   map[string]interface{}{"max_tokens": 100, "temperature": 0.7},
			}
		case "whisper-transcription":
			req = &InferenceRequest{
				Model:    "whisper",
				JobID:    jobID,
				InputURL: payloadURL,
				Params:   map[string]interface{}{"model": "base"},
			}
		default:
			// Default to Stable Diffusion
			req = &InferenceRequest{
				Model:    "stable-diffusion-v1-5",
				Prompt:   "digital art, high quality",
				JobID:    jobID,
				InputURL: payloadURL,
				Params:   map[string]interface{}{"steps": 15},
			}
		}

		var inferenceResult *InferenceResult
		switch jobType {
		case "stable-diffusion":
			inferenceResult, err = gpuExecutor.ExecuteStableDiffusion(ctx, req)
		case "llm-inference":
			inferenceResult, err = gpuExecutor.ExecuteLLMInference(ctx, req)
		case "whisper-transcription":
			inferenceResult, err = gpuExecutor.ExecuteWhisperTranscription(ctx, req)
		default:
			inferenceResult, err = gpuExecutor.ExecuteStableDiffusion(ctx, req)
		}

		if err == nil && inferenceResult.Error == "" {
			resultData, _ := json.Marshal(inferenceResult)

			var uploadURL string
			if inferenceResult.Output != "" && (filepath.Ext(inferenceResult.Output) != "" || len(inferenceResult.Output) > 100) {
				if url, uploadErr := e.uploadResultToHub(ctx, jobID, inferenceResult.Output, jobType); uploadErr == nil {
					uploadURL = url
					fmt.Printf("Result uploaded successfully: %s\n", uploadURL)
				} else {
					fmt.Printf("Warning: failed to upload result: %v\n", uploadErr)
				}
			}

			result := &WorkResult{
				JobID:      jobID,
				ResultHash: hashOutput(resultData),
				OutputData: resultData,
				Metrics: ExecutionMetrics{
					Duration:       time.Duration(inferenceResult.Duration * float64(time.Second)),
					GPUUtilization: inferenceResult.GPUUsage,
					PowerUsage:     e.measurePowerUsage(),
					TokensPerSec:   float64(inferenceResult.TokenCount) / inferenceResult.Duration,
				},
			}

			if uploadURL != "" {
				resultWithURL := make(map[string]interface{})
				json.Unmarshal(resultData, &resultWithURL)
				resultWithURL["upload_url"] = uploadURL
				result.OutputData, _ = json.Marshal(resultWithURL)
			}

			delete(e.activeJobs, jobID)
			return result, nil
		}

		errMsg := ""
		if inferenceResult != nil {
			errMsg = inferenceResult.Error
		}
		fmt.Printf("GPU inference failed for job %s: %v, error: %s. Falling back to simulation.\n",
			jobID, err, errMsg)
	}

	fmt.Printf("Using simulation fallback for job %s (GPU executor unavailable: %v)\n", jobID, err)

	time.Sleep(2 * time.Second)

	simulatedResult := map[string]interface{}{
		"job_id":      jobID,
		"job_type":    jobType,
		"output":      fmt.Sprintf("simulated_%s_output_for_%s", jobType, jobID),
		"simulated":   true,
		"message":     "Real GPU execution unavailable, using simulation",
		"payload_url": payloadURL,
	}

	resultData, _ := json.Marshal(simulatedResult)
	duration := time.Since(e.activeJobs[jobID].StartTime)

	result := &WorkResult{
		JobID:      jobID,
		ResultHash: hashOutput(resultData),
		OutputData: resultData,
		Metrics: ExecutionMetrics{
			Duration:       duration,
			GPUUtilization: 45.0,
			PowerUsage:     e.measurePowerUsage(),
			TokensPerSec:   20.0,
		},
	}

	delete(e.activeJobs, jobID)
	return result, nil
}

func (e *WorkloadExecutor) executeInferenceJob(ctx context.Context, job *WorkRequest) (*WorkResult, error) {
	var params InferenceParams
	if err := json.Unmarshal(job.Parameters, &params); err != nil {
		return nil, fmt.Errorf("invalid inference parameters: %w", err)
	}

	config := &container.Config{
		Image: params.ModelImage,
		Env: []string{
			fmt.Sprintf("MODEL_NAME=%s", params.ModelName),
			fmt.Sprintf("BATCH_SIZE=%d", params.BatchSize),
			"CUDA_VISIBLE_DEVICES=0",
		},
		Cmd: []string{
			"python", "inference.py",
			"--input", "/data/input.json",
			"--output", "/data/output.json",
		},
	}

	hostConfig := &container.HostConfig{
		Resources: container.Resources{
			DeviceRequests: []container.DeviceRequest{
				{
					Count:        1,
					Capabilities: [][]string{{"gpu"}},
				},
			},
			Memory:   8 * 1024 * 1024 * 1024,
			NanoCPUs: 4 * 1000000000,
		},
		AutoRemove: true,
	}

	resp, err := e.dockerClient.ContainerCreate(ctx, config, hostConfig, nil, nil, "")
	if err != nil {
		return nil, fmt.Errorf("failed to create container: %w", err)
	}

	e.activeJobs[job.JobId].ContainerID = resp.ID
	e.activeJobs[job.JobId].Status = "running"

	if err := e.dockerClient.ContainerStart(ctx, resp.ID, types.ContainerStartOptions{}); err != nil {
		return nil, fmt.Errorf("failed to start container: %w", err)
	}

	statusCh, errCh := e.dockerClient.ContainerWait(ctx, resp.ID, container.WaitConditionNotRunning)
	select {
	case err := <-errCh:
		if err != nil {
			return nil, fmt.Errorf("container wait error: %w", err)
		}
	case status := <-statusCh:
		if status.StatusCode != 0 {
			return nil, fmt.Errorf("container exited with code %d", status.StatusCode)
		}
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	output, err := e.getContainerOutput(ctx, resp.ID)
	if err != nil {
		return nil, err
	}

	duration := time.Since(e.activeJobs[job.JobId].StartTime)

	result := &WorkResult{
		JobID:      job.JobId,
		ResultHash: hashOutput(output),
		OutputData: output,
		Metrics: ExecutionMetrics{
			Duration:       duration,
			GPUUtilization: e.measureGPUUtilization(),
			PowerUsage:     e.measurePowerUsage(),
			TokensPerSec:   float64(params.BatchSize) / duration.Seconds(),
		},
	}

	delete(e.activeJobs, job.JobId)
	return result, nil
}

func (e *WorkloadExecutor) executeTranscoding(ctx context.Context, job *WorkRequest) (*WorkResult, error) {
	var params TranscodingParams
	if err := json.Unmarshal(job.Parameters, &params); err != nil {
		return nil, fmt.Errorf("invalid transcoding parameters: %w", err)
	}

	transcodingExecutor, err := NewVideoTranscodingExecutor()
	if err != nil {
		return nil, fmt.Errorf("failed to create transcoding executor: %w", err)
	}

	result, err := transcodingExecutor.ExecuteTranscoding(ctx, &VideoTranscodingRequest{
		JobID:       job.JobId,
		InputURL:    params.InputURL,
		OutputCodec: params.OutputCodec,
		Preset:      params.Preset,
		Bitrate:     params.Bitrate,
		Resolution:  "1920x1080",
	})

	if err != nil {
		return nil, fmt.Errorf("transcoding failed: %w", err)
	}

	resultData, _ := json.Marshal(result)
	duration := time.Since(e.activeJobs[job.JobId].StartTime)

	return &WorkResult{
		JobID:      job.JobId,
		ResultHash: hashOutput(resultData),
		OutputData: resultData,
		Metrics: ExecutionMetrics{
			Duration:       duration,
			GPUUtilization: result.GPUUsage,
			PowerUsage:     e.measurePowerUsage(),
			TokensPerSec:   result.FramesPerSecond,
		},
	}, nil
}

func (e *WorkloadExecutor) executeRendering(ctx context.Context, job *WorkRequest) (*WorkResult, error) {
	var params RenderingParams
	if err := json.Unmarshal(job.Parameters, &params); err != nil {
		return nil, fmt.Errorf("invalid rendering parameters: %w", err)
	}

	renderingExecutor, err := NewRenderingExecutor()
	if err != nil {
		return nil, fmt.Errorf("failed to create rendering executor: %w", err)
	}

	result, err := renderingExecutor.ExecuteRendering(ctx, &RenderingRequest{
		JobID:      job.JobId,
		ProjectURL: params.ProjectURL,
		StartFrame: params.StartFrame,
		EndFrame:   params.EndFrame,
		Resolution: params.Resolution,
		Engine:     "CYCLES",
		Quality:    "MEDIUM",
	})

	if err != nil {
		return nil, fmt.Errorf("rendering failed: %w", err)
	}

	resultData, _ := json.Marshal(result)
	duration := time.Since(e.activeJobs[job.JobId].StartTime)

	return &WorkResult{
		JobID:      job.JobId,
		ResultHash: hashOutput(resultData),
		OutputData: resultData,
		Metrics: ExecutionMetrics{
			Duration:       duration,
			GPUUtilization: result.GPUUsage,
			PowerUsage:     e.measurePowerUsage(),
			TokensPerSec:   float64(result.FramesCount) / result.Duration,
		},
	}, nil
}

func (e *WorkloadExecutor) executeTraining(ctx context.Context, job *WorkRequest) (*WorkResult, error) {
	var params TrainingParams
	if err := json.Unmarshal(job.Parameters, &params); err != nil {
		return nil, fmt.Errorf("invalid training parameters: %w", err)
	}

	trainingExecutor, err := NewModelTrainingExecutor()
	if err != nil {
		return nil, fmt.Errorf("failed to create training executor: %w", err)
	}

	result, err := trainingExecutor.ExecuteTraining(ctx, &ModelTrainingRequest{
		JobID:        job.JobId,
		ModelType:    params.ModelType,
		DatasetURL:   params.DatasetURL,
		BaseModelURL: params.BaseModelURL,
		Epochs:       params.Epochs,
		BatchSize:    params.BatchSize,
		LearningRate: params.LearningRate,
		MaxSteps:     params.MaxSteps,
	})

	if err != nil {
		return nil, fmt.Errorf("training failed: %w", err)
	}

	resultData, _ := json.Marshal(result)
	duration := time.Since(e.activeJobs[job.JobId].StartTime)

	return &WorkResult{
		JobID:      job.JobId,
		ResultHash: hashOutput(resultData),
		OutputData: resultData,
		Metrics: ExecutionMetrics{
			Duration:       duration,
			GPUUtilization: result.GPUUsage,
			PowerUsage:     e.measurePowerUsage(),
			TokensPerSec:   float64(result.StepsCompleted) / result.Duration,
		},
	}, nil
}

func (e *WorkloadExecutor) StopJob(jobID string) error {
	execution, exists := e.activeJobs[jobID]
	if !exists {
		return fmt.Errorf("job %s not found", jobID)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := e.dockerClient.ContainerStop(ctx, execution.ContainerID, nil); err != nil {
		return fmt.Errorf("failed to stop container: %w", err)
	}

	delete(e.activeJobs, jobID)
	return nil
}

func (e *WorkloadExecutor) GetActiveJobs() []JobExecution {
	jobs := make([]JobExecution, 0, len(e.activeJobs))
	for _, job := range e.activeJobs {
		jobs = append(jobs, *job)
	}
	return jobs
}

func (e *WorkloadExecutor) Cleanup() error {
	for jobID := range e.activeJobs {
		if err := e.StopJob(jobID); err != nil {
			return err
		}
	}
	return e.dockerClient.Close()
}

// Helper functions

func detectGPUCapabilities() GPUCapabilities {
	return GPUCapabilities{
		Model:     "NVIDIA RTX 3080",
		VRAM:      10240,
		CUDACores: 8704,
		TFLOPs:    29.77,
	}
}

func (e *WorkloadExecutor) measureGPUUtilization() float64 {
	return 85.5 // Placeholder
}

func (e *WorkloadExecutor) measurePowerUsage() float64 {
	return 250.0 // Watts, placeholder
}

func (e *WorkloadExecutor) getContainerOutput(ctx context.Context, containerID string) ([]byte, error) {
	logs, err := e.dockerClient.ContainerLogs(ctx, containerID, types.ContainerLogsOptions{
		ShowStdout: true,
		ShowStderr: true,
		Timestamps: false,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get container logs: %w", err)
	}
	defer logs.Close()

	output := make([]byte, 1024*1024) // 1MB buffer
	n, err := logs.Read(output)
	if err != nil && err.Error() != "EOF" {
		return nil, fmt.Errorf("failed to read container output: %w", err)
	}

	return output[:n], nil
}

func hashOutput(data []byte) string {
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}

type InferenceParams struct {
	ModelImage string `json:"model_image"`
	ModelName  string `json:"model_name"`
	BatchSize  int    `json:"batch_size"`
	InputData  string `json:"input_data"`
}

type TranscodingParams struct {
	InputURL    string `json:"input_url"`
	OutputCodec string `json:"output_codec"`
	Preset      string `json:"preset"`
	Bitrate     string `json:"bitrate"`
}

type RenderingParams struct {
	ProjectURL string `json:"project_url"`
	StartFrame int    `json:"start_frame"`
	EndFrame   int    `json:"end_frame"`
	Resolution string `json:"resolution"`
}

type TrainingParams struct {
	ModelType    string  `json:"model_type"`
	DatasetURL   string  `json:"dataset_url"`
	BaseModelURL string  `json:"base_model_url"`
	Epochs       int     `json:"epochs"`
	BatchSize    int     `json:"batch_size"`
	LearningRate float64 `json:"learning_rate"`
	MaxSteps     int     `json:"max_steps"`
}

func (e *WorkloadExecutor) uploadResultToHub(ctx context.Context, jobID, resultPath, resultType string) (string, error) {
	if _, err := os.Stat(resultPath); os.IsNotExist(err) {
		return "", fmt.Errorf("result file not found: %s", resultPath)
	}

	file, err := os.Open(resultPath)
	if err != nil {
		return "", fmt.Errorf("failed to open result file: %w", err)
	}
	defer file.Close()

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	writer.WriteField("job_id", jobID)
	writer.WriteField("type", resultType)

	part, err := writer.CreateFormFile("result", filepath.Base(resultPath))
	if err != nil {
		return "", fmt.Errorf("failed to create form file: %w", err)
	}

	_, err = io.Copy(part, file)
	if err != nil {
		return "", fmt.Errorf("failed to copy file to form: %w", err)
	}

	writer.Close()

	uploadURL := fmt.Sprintf("%s/api/results/upload", e.hubBaseURL)
	req, err := http.NewRequestWithContext(ctx, "POST", uploadURL, &buf)
	if err != nil {
		return "", fmt.Errorf("failed to create upload request: %w", err)
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("upload request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("upload failed with status %d: %s", resp.StatusCode, string(body))
	}

	return fmt.Sprintf("%s/api/results/%s", e.hubBaseURL, jobID), nil
}
