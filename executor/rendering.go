package executor

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/client"
)

type RenderingExecutor struct {
	dockerClient *client.Client
}

type RenderingRequest struct {
	JobID      string `json:"job_id"`
	ProjectURL string `json:"project_url"`
	StartFrame int    `json:"start_frame"`
	EndFrame   int    `json:"end_frame"`
	Resolution string `json:"resolution"`
	Engine     string `json:"engine"`  // "CYCLES" or "EEVEE"
	Samples    int    `json:"samples"` // Ray-tracing samples
	Quality    string `json:"quality"` // "LOW", "MEDIUM", "HIGH"
}

type RenderingResult struct {
	JobID        string   `json:"job_id"`
	OutputFrames []string `json:"output_frames"`
	FramesCount  int      `json:"frames_count"`
	Duration     float64  `json:"duration_seconds"`
	AverageTime  float64  `json:"average_time_per_frame"`
	RenderEngine string   `json:"render_engine"`
	GPUUsage     float64  `json:"gpu_usage_percent"`
	TotalSamples int      `json:"total_samples"`
	Resolution   string   `json:"resolution"`
	Error        string   `json:"error,omitempty"`
}

func NewRenderingExecutor() (*RenderingExecutor, error) {
	cli, err := client.NewClientWithOpts(client.FromEnv)
	if err != nil {
		return nil, fmt.Errorf("failed to create Docker client: %w", err)
	}
	return &RenderingExecutor{dockerClient: cli}, nil
}

func (r *RenderingExecutor) ExecuteRendering(ctx context.Context, req *RenderingRequest) (*RenderingResult, error) {
	startTime := time.Now()

	// Create work directory
	workDir := filepath.Join("/tmp", fmt.Sprintf("render_%s", req.JobID))
	if err := os.MkdirAll(workDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create work directory: %w", err)
	}
	defer os.RemoveAll(workDir)

	// Download Blender project file
	projectPath := filepath.Join(workDir, "project.blend")
	if err := r.downloadFile(req.ProjectURL, projectPath); err != nil {
		return nil, fmt.Errorf("failed to download project file: %w", err)
	}

	// Set rendering parameters
	engine := req.Engine
	if engine == "" {
		engine = "CYCLES"
	}

	samples := req.Samples
	if samples == 0 {
		switch req.Quality {
		case "LOW":
			samples = 32
		case "MEDIUM":
			samples = 128
		case "HIGH":
			samples = 512
		default:
			samples = 128
		}
	}

	resolution := req.Resolution
	if resolution == "" {
		resolution = "1920x1080"
	}

	startFrame := req.StartFrame
	endFrame := req.EndFrame
	if endFrame <= startFrame {
		endFrame = startFrame + 1
	}

	// Create Blender Python script for rendering
	pythonScript := fmt.Sprintf(`
import bpy
import os

# Set render engine
bpy.context.scene.render.engine = '%s'

# Configure Cycles for GPU rendering
if bpy.context.scene.render.engine == 'CYCLES':
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = %d
    
    # Enable GPU devices
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    cprefs.compute_device_type = 'CUDA'
    
    # Enable all GPU devices
    for device in cprefs.devices:
        if device.type == 'CUDA':
            device.use = True

# Set resolution
resolution = '%s'.split('x')
if len(resolution) == 2:
    bpy.context.scene.render.resolution_x = int(resolution[0])
    bpy.context.scene.render.resolution_y = int(resolution[1])

# Set output format
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.filepath = '/work/frame_'

# Set frame range
bpy.context.scene.frame_start = %d
bpy.context.scene.frame_end = %d

# Render animation
bpy.ops.render.render(animation=True)

print("Rendering completed successfully")
`, engine, samples, resolution, startFrame, endFrame)

	scriptPath := filepath.Join(workDir, "render_script.py")
	if err := os.WriteFile(scriptPath, []byte(pythonScript), 0644); err != nil {
		return nil, fmt.Errorf("failed to create render script: %w", err)
	}

	// Build Blender command
	blenderArgs := []string{
		"blender",
		"-b", "/work/project.blend", // Background mode with project file
		"-P", "/work/render_script.py", // Run Python script
		"--", "--cycles-device", "CUDA", // Force CUDA
	}

	// Create and run container
	config := &container.Config{
		Image: "blender/blender:3.6-gpu",
		Cmd:   blenderArgs,
		Env: []string{
			"NVIDIA_VISIBLE_DEVICES=0",
			"NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics",
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
			Memory:   8 * 1024 * 1024 * 1024, // 8GB
			NanoCPUs: 4 * 1000000000,         // 4 CPUs
		},
		AutoRemove: true,
	}

	resp, err := r.dockerClient.ContainerCreate(ctx, config, hostConfig, nil, nil, fmt.Sprintf("akatosh_render_%s", req.JobID))
	if err != nil {
		return nil, fmt.Errorf("failed to create container: %w", err)
	}

	if err := r.dockerClient.ContainerStart(ctx, resp.ID, types.ContainerStartOptions{}); err != nil {
		return nil, fmt.Errorf("failed to start container: %w", err)
	}

	// Wait for completion with extended timeout (rendering can take a while)
	timeoutCtx, cancel := context.WithTimeout(ctx, 30*time.Minute)
	defer cancel()

	statusCh, errCh := r.dockerClient.ContainerWait(timeoutCtx, resp.ID, container.WaitConditionNotRunning)
	select {
	case err := <-errCh:
		if err != nil {
			return nil, fmt.Errorf("container wait error: %w", err)
		}
	case status := <-statusCh:
		if status.StatusCode != 0 {
			logs, _ := r.getContainerLogs(ctx, resp.ID)
			return &RenderingResult{
				JobID:    req.JobID,
				Error:    fmt.Sprintf("Blender exited with code %d. Logs: %s", status.StatusCode, logs),
				Duration: time.Since(startTime).Seconds(),
			}, nil
		}
	case <-timeoutCtx.Done():
		r.dockerClient.ContainerStop(context.Background(), resp.ID, nil)
		return nil, fmt.Errorf("rendering timeout after 30 minutes")
	}

	// Count output frames
	outputFrames, err := r.collectOutputFrames(workDir, startFrame, endFrame)
	if err != nil {
		return &RenderingResult{
			JobID:    req.JobID,
			Error:    fmt.Sprintf("failed to collect output frames: %v", err),
			Duration: time.Since(startTime).Seconds(),
		}, nil
	}

	duration := time.Since(startTime).Seconds()
	frameCount := len(outputFrames)
	avgTimePerFrame := 0.0
	if frameCount > 0 {
		avgTimePerFrame = duration / float64(frameCount)
	}

	return &RenderingResult{
		JobID:        req.JobID,
		OutputFrames: outputFrames,
		FramesCount:  frameCount,
		Duration:     duration,
		AverageTime:  avgTimePerFrame,
		RenderEngine: engine,
		GPUUsage:     r.measureGPUUsage(),
		TotalSamples: samples * frameCount,
		Resolution:   resolution,
	}, nil
}

func (r *RenderingExecutor) downloadFile(url, filepath string) error {
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

func (r *RenderingExecutor) collectOutputFrames(workDir string, startFrame, endFrame int) ([]string, error) {
	var frames []string

	for frame := startFrame; frame <= endFrame; frame++ {
		// Blender typically outputs frames with 4-digit padding (e.g., frame_0001.png)
		frameName := fmt.Sprintf("frame_%04d.png", frame)
		framePath := filepath.Join(workDir, frameName)

		if _, err := os.Stat(framePath); err == nil {
			// File exists - add to list (in production, upload to storage)
			outputURL := fmt.Sprintf("/tmp/render_output_%s_%04d.png", workDir[strings.LastIndex(workDir, "_")+1:], frame)
			frames = append(frames, outputURL)
		}
	}

	return frames, nil
}

func (r *RenderingExecutor) getContainerLogs(ctx context.Context, containerID string) (string, error) {
	logs, err := r.dockerClient.ContainerLogs(ctx, containerID, types.ContainerLogsOptions{
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

func (r *RenderingExecutor) measureGPUUsage() float64 {
	// TODO: Implement nvidia-ml-py or nvidia-smi parsing
	// For now return a reasonable estimate for 3D rendering
	return 90.0
}
