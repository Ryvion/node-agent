//go:build containers

package executor

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/client"
)

type VideoTranscodingExecutor struct {
	dockerClient *client.Client
}

type VideoTranscodingRequest struct {
	JobID       string `json:"job_id"`
	InputURL    string `json:"input_url"`
	OutputCodec string `json:"output_codec"`
	Preset      string `json:"preset"`
	Bitrate     string `json:"bitrate"`
	Resolution  string `json:"resolution"`
}

type VideoTranscodingResult struct {
	JobID           string  `json:"job_id"`
	OutputURL       string  `json:"output_url"`
	Duration        float64 `json:"duration_seconds"`
	InputSize       int64   `json:"input_size_bytes"`
	OutputSize      int64   `json:"output_size_bytes"`
	CompressionRate float64 `json:"compression_rate"`
	FramesPerSecond float64 `json:"frames_per_second"`
	GPUUsage        float64 `json:"gpu_usage_percent"`
	Error           string  `json:"error,omitempty"`
}

func NewVideoTranscodingExecutor() (*VideoTranscodingExecutor, error) {
	cli, err := client.NewClientWithOpts(client.FromEnv)
	if err != nil {
		return nil, fmt.Errorf("failed to create Docker client: %w", err)
	}
	return &VideoTranscodingExecutor{dockerClient: cli}, nil
}

func (v *VideoTranscodingExecutor) ExecuteTranscoding(ctx context.Context, req *VideoTranscodingRequest) (*VideoTranscodingResult, error) {
	startTime := time.Now()

	// Create work directory
	workDir := filepath.Join("/tmp", fmt.Sprintf("transcode_%s", req.JobID))
	if err := os.MkdirAll(workDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create work directory: %w", err)
	}
	defer os.RemoveAll(workDir)

	// Download input video
	inputPath := filepath.Join(workDir, "input.mp4")
	inputSize, err := v.downloadFile(req.InputURL, inputPath)
	if err != nil {
		return nil, fmt.Errorf("failed to download input video: %w", err)
	}

	// Set up FFmpeg transcoding parameters
	outputPath := "/work/output.mp4"
	codec := req.OutputCodec
	if codec == "" {
		codec = "h264_nvenc" // NVIDIA GPU encoding
	}

	preset := req.Preset
	if preset == "" {
		preset = "fast"
	}

	bitrate := req.Bitrate
	if bitrate == "" {
		bitrate = "2000k"
	}

	// Build FFmpeg command
	ffmpegArgs := []string{
		"-hwaccel", "cuda", // Hardware acceleration
		"-hwaccel_output_format", "cuda", // Keep frames in GPU memory
		"-i", "/work/input.mp4", // Input file
		"-c:v", codec, // Video codec
		"-preset", preset, // Encoding preset
		"-b:v", bitrate, // Video bitrate
		"-c:a", "aac", // Audio codec
		"-b:a", "128k", // Audio bitrate
		"-f", "mp4", // Output format
		outputPath, // Output file
		"-y",       // Overwrite output
	}

	// Add resolution scaling if specified
	if req.Resolution != "" && req.Resolution != "original" {
		// Insert video filter before the last 3 arguments (-f mp4 output.mp4 -y)
		beforeEnd := ffmpegArgs[:len(ffmpegArgs)-3]
		afterEnd := ffmpegArgs[len(ffmpegArgs)-3:]
		beforeEnd = append(beforeEnd, "-vf", fmt.Sprintf("scale_cuda=%s", req.Resolution))
		ffmpegArgs = append(beforeEnd, afterEnd...)
	}

	// Create and run container
	config := &container.Config{
		Image: "jrottenberg/ffmpeg:4.4-nvidia",
		Cmd:   ffmpegArgs,
		Env: []string{
			"NVIDIA_VISIBLE_DEVICES=0",
			"NVIDIA_DRIVER_CAPABILITIES=compute,utility,video",
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
			Memory:   4 * 1024 * 1024 * 1024, // 4GB
			NanoCPUs: 4 * 1000000000,         // 4 CPUs
		},
		AutoRemove: true,
	}

	resp, err := v.dockerClient.ContainerCreate(ctx, config, hostConfig, nil, nil, fmt.Sprintf("akatosh_transcode_%s", req.JobID))
	if err != nil {
		return nil, fmt.Errorf("failed to create container: %w", err)
	}

	if err := v.dockerClient.ContainerStart(ctx, resp.ID, types.ContainerStartOptions{}); err != nil {
		return nil, fmt.Errorf("failed to start container: %w", err)
	}

	// Wait for completion with timeout (max 10 minutes for transcoding)
	timeoutCtx, cancel := context.WithTimeout(ctx, 10*time.Minute)
	defer cancel()

	statusCh, errCh := v.dockerClient.ContainerWait(timeoutCtx, resp.ID, container.WaitConditionNotRunning)
	select {
	case err := <-errCh:
		if err != nil {
			return nil, fmt.Errorf("container wait error: %w", err)
		}
	case status := <-statusCh:
		if status.StatusCode != 0 {
			logs, _ := v.getContainerLogs(ctx, resp.ID)
			return &VideoTranscodingResult{
				JobID:    req.JobID,
				Error:    fmt.Sprintf("FFmpeg exited with code %d. Logs: %s", status.StatusCode, logs),
				Duration: time.Since(startTime).Seconds(),
			}, nil
		}
	case <-timeoutCtx.Done():
		v.dockerClient.ContainerStop(context.Background(), resp.ID, nil)
		return nil, fmt.Errorf("transcoding timeout after 10 minutes")
	}

	// Check output file
	outputFilePath := filepath.Join(workDir, "output.mp4")
	outputStat, err := os.Stat(outputFilePath)
	if err != nil {
		return &VideoTranscodingResult{
			JobID:    req.JobID,
			Error:    "no output file produced",
			Duration: time.Since(startTime).Seconds(),
		}, nil
	}

	duration := time.Since(startTime).Seconds()
	outputSize := outputStat.Size()
	compressionRate := 1.0 - (float64(outputSize) / float64(inputSize))

	// Calculate approximate FPS (assume 30fps input for estimation)
	estimatedFPS := 30.0
	if duration > 0 {
		// Very rough estimation based on processing speed
		estimatedFPS = float64(inputSize) / (1024 * 1024) / duration * 2 // MB/s to rough FPS
		if estimatedFPS > 60 {
			estimatedFPS = 60
		}
	}

	// TODO: Upload output file to storage (S3, IPFS, etc.)
	outputURL := fmt.Sprintf("/tmp/transcoding_output_%s.mp4", req.JobID)

	return &VideoTranscodingResult{
		JobID:           req.JobID,
		OutputURL:       outputURL,
		Duration:        duration,
		InputSize:       inputSize,
		OutputSize:      outputSize,
		CompressionRate: compressionRate,
		FramesPerSecond: estimatedFPS,
		GPUUsage:        v.measureGPUUsage(),
	}, nil
}

func (v *VideoTranscodingExecutor) downloadFile(url, filepath string) (int64, error) {
	resp, err := http.Get(url)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	out, err := os.Create(filepath)
	if err != nil {
		return 0, err
	}
	defer out.Close()

	size, err := io.Copy(out, resp.Body)
	return size, err
}

func (v *VideoTranscodingExecutor) getContainerLogs(ctx context.Context, containerID string) (string, error) {
	logs, err := v.dockerClient.ContainerLogs(ctx, containerID, types.ContainerLogsOptions{
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

func (v *VideoTranscodingExecutor) measureGPUUsage() float64 {
	// TODO: Implement nvidia-ml-py or nvidia-smi parsing
	// For now return a reasonable estimate for transcoding
	return 80.0
}
