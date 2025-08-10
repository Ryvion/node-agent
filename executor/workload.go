package executor

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/client"
)

type WorkloadType string

const (
	WorkloadTypeInference    WorkloadType = "inference"
	WorkloadTypeTraining     WorkloadType = "training"
	WorkloadTypeTranscoding  WorkloadType = "transcoding"
	WorkloadTypeRendering    WorkloadType = "rendering"
	WorkloadTypeMining       WorkloadType = "mining"
)

// Local WorkRequest type (simplified version of proto.WorkRequest)
type WorkRequest struct {
	JobId      string `json:"job_id"`
	JobType    string `json:"job_type"`
	Parameters []byte `json:"parameters"`
}

type WorkloadExecutor struct {
	dockerClient    *client.Client
	gpuCapabilities GPUCapabilities
	activeJobs      map[string]*JobExecution
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
	JobID       string
	ResultHash  string
	OutputData  []byte
	Metrics     ExecutionMetrics
	Error       error
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
	case WorkloadTypeTranscoding:
		return e.executeTranscoding(ctx, job)
	case WorkloadTypeRendering:
		return e.executeRendering(ctx, job)
	default:
		return nil, fmt.Errorf("unsupported workload type: %s", job.JobType)
	}
}

// Simplified execution method for node agent
func (e *WorkloadExecutor) ExecuteInference(ctx context.Context, jobID, jobType, payloadURL string) (*WorkResult, error) {
	// Use default parameters for simplified interface
	params := InferenceParams{
		ModelImage: "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime",
		ModelName:  "default",
		BatchSize:  1,
		InputData:  payloadURL,
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
			Memory:     8 * 1024 * 1024 * 1024, // 8GB
			NanoCPUs:   4 * 1000000000,          // 4 CPUs
		},
		AutoRemove: true,
	}

	resp, err := e.dockerClient.ContainerCreate(ctx, config, hostConfig, nil, nil, "")
	if err != nil {
		return nil, fmt.Errorf("failed to create container: %w", err)
	}

	e.activeJobs[jobID].ContainerID = resp.ID
	e.activeJobs[jobID].Status = "running"

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

	duration := time.Since(e.activeJobs[jobID].StartTime)
	
	result := &WorkResult{
		JobID:      jobID,
		ResultHash: hashOutput(output),
		OutputData: output,
		Metrics: ExecutionMetrics{
			Duration:       duration,
			GPUUtilization: e.measureGPUUtilization(),
			PowerUsage:     e.measurePowerUsage(),
			TokensPerSec:   float64(params.BatchSize) / duration.Seconds(),
		},
	}

	delete(e.activeJobs, jobID)
	return result, nil
}

// Full workload job execution method
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
			Memory:     8 * 1024 * 1024 * 1024, // 8GB
			NanoCPUs:   4 * 1000000000,          // 4 CPUs
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

	// TODO: Implement full transcoding execution when Docker issues are resolved
	_ = params // Use params to avoid unused variable error
	
	return &WorkResult{
		JobID:      job.JobId,
		ResultHash: "transcoded_hash",
		Metrics: ExecutionMetrics{
			Duration: time.Since(e.activeJobs[job.JobId].StartTime),
		},
	}, nil
}

func (e *WorkloadExecutor) executeRendering(ctx context.Context, job *WorkRequest) (*WorkResult, error) {
	var params RenderingParams
	if err := json.Unmarshal(job.Parameters, &params); err != nil {
		return nil, fmt.Errorf("invalid rendering parameters: %w", err)
	}

	// TODO: Implement full rendering execution when Docker issues are resolved
	_ = params // Use params to avoid unused variable error
	
	return &WorkResult{
		JobID:      job.JobId,
		ResultHash: "rendered_hash",
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
	// Use nvidia-smi or similar to detect GPU
	// This is a placeholder implementation
	return GPUCapabilities{
		Model:     "NVIDIA RTX 3080",
		VRAM:      10240,
		CUDACores: 8704,
		TFLOPs:    29.77,
	}
}

func (e *WorkloadExecutor) measureGPUUtilization() float64 {
	// Use nvidia-smi to get current GPU utilization
	return 85.5 // Placeholder
}

func (e *WorkloadExecutor) measurePowerUsage() float64 {
	// Use nvidia-smi to get power usage
	return 250.0 // Watts, placeholder
}

func (e *WorkloadExecutor) getContainerOutput(ctx context.Context, containerID string) ([]byte, error) {
	// Read container logs as output
	logs, err := e.dockerClient.ContainerLogs(ctx, containerID, types.ContainerLogsOptions{
		ShowStdout: true,
		ShowStderr: true,
		Timestamps: false,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get container logs: %w", err)
	}
	defer logs.Close()
	
	// Read the logs content
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

// Parameter structs

type InferenceParams struct {
	ModelImage string `json:"model_image"`
	ModelName  string `json:"model_name"`
	BatchSize  int    `json:"batch_size"`
	InputData  string `json:"input_data"`
}

type TranscodingParams struct {
	InputURL     string `json:"input_url"`
	OutputCodec  string `json:"output_codec"`
	Preset       string `json:"preset"`
	Bitrate      string `json:"bitrate"`
}

type RenderingParams struct {
	ProjectURL  string `json:"project_url"`
	StartFrame  int    `json:"start_frame"`
	EndFrame    int    `json:"end_frame"`
	Resolution  string `json:"resolution"`
}
