package executor

import (
	"context"
	crand "crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"os/exec"
	"strings"
	"time"
)

// Run executes a job using available execution modes (docker, native), and only
// falls back to simulation if real execution paths are unavailable. The return
// values are the output hash (hex) and metering units consumed.
func Run(kind string, payloadURL string, suggestedUnits uint32) (resultHashHex string, units uint32, meta map[string]any) {
	meta = make(map[string]any)
	if os.Getenv("AK_EXECUTOR_MODE") == "docker" {
		LoadCatalogFromEnv()
		if ks, ok := LookupKind(kind); ok && ks.Image != "" {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()
			if h, code, logs, ok := TryDockerInference(ctx, ks.Image, ks.Args, 60*time.Second); ok {
				meta["executor"] = "docker/catalog"
				meta["exit_code"] = code
				meta["stderr_tail"] = logs
				return h, max1(suggestedUnits), meta
			}
		}
		ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
		defer cancel()
		switch kind {
		case "inference":
			if h, code, logs, ok := TryDockerInference(ctx, os.Getenv("AK_DOCKER_IMAGE_INFER"), []string{"--help"}, 30*time.Second); ok {
				meta["executor"] = "docker/inference"
				meta["exit_code"] = code
				meta["stderr_tail"] = logs
				return h, max1(suggestedUnits), meta
			}
		case "transcoding":
			if h, code, logs, ok := TryDockerInference(ctx, os.Getenv("AK_DOCKER_IMAGE_FFMPEG"), []string{"-version"}, 30*time.Second); ok {
				meta["executor"] = "docker/ffmpeg"
				meta["exit_code"] = code
				meta["stderr_tail"] = logs
				return h, max1(suggestedUnits), meta
			}
		}
	}
	if strings.EqualFold(kind, "transcoding") && strings.TrimSpace(payloadURL) != "" {
		argsEnv := strings.TrimSpace(os.Getenv("AK_FFMPEG_ARGS"))
		var args []string
		if argsEnv != "" {
			for _, p := range strings.Split(argsEnv, ",") {
				args = append(args, strings.TrimSpace(p))
			}
		} else {
			if hasNvidiaGPU() || os.Getenv("AK_FFMPEG_USE_NVENC") == "1" {
				args = []string{"-y", "-vsync", "0", "-hwaccel", "cuda", "-i", "{input}", "-c:v", "h264_nvenc", "-preset", "fast", "{output}"}
			} else {
				args = []string{"-y", "-i", "{input}", "-c:v", "libx264", "-preset", "veryfast", "{output}"}
			}
		}
		tout := 5 * time.Minute
		if v := strings.TrimSpace(os.Getenv("AK_FFMPEG_TIMEOUT_SEC")); v != "" {
			if n, err := time.ParseDuration(v + "s"); err == nil {
				tout = n
			}
		}
		ctx, cancel := context.WithTimeout(context.Background(), tout)
		defer cancel()
		if h, err := RunFFmpeg(ctx, TranscodeSpec{InputURL: payloadURL, Args: args, Timeout: tout}); err == nil && h != "" {
			meta["executor"] = "ffmpeg/native"
			return h, max1(suggestedUnits), meta
		}
	}
	if strings.EqualFold(kind, "embedding") || strings.EqualFold(kind, "embed") || strings.EqualFold(kind, "inference") {
		if cmd := strings.TrimSpace(os.Getenv("EMBED_CMD")); cmd != "" {
			in, out := tempPaths()
			if strings.TrimSpace(payloadURL) != "" {
				_ = downloadToFile(payloadURL, in)
			}
			argsEnv := strings.TrimSpace(os.Getenv("EMBED_ARGS"))
			var args []string
			if argsEnv != "" {
				for _, p := range strings.Split(argsEnv, ",") {
					args = append(args, strings.TrimSpace(p))
				}
			} else {
				args = []string{"-f", "{input}", "-o", "{output}"}
			}
			tout := 2 * time.Minute
			if v := strings.TrimSpace(os.Getenv("EMBED_TIMEOUT_SEC")); v != "" {
				if n, err := time.ParseDuration(v + "s"); err == nil {
					tout = n
				}
			}
			ctx, cancel := context.WithTimeout(context.Background(), tout)
			defer cancel()
			if h, err := RunEmbedding(ctx, EmbedSpec{InputFile: in, OutputFile: out, Timeout: tout, Cmd: cmd, Args: args}); err == nil && h != "" {
				meta["executor"] = "embed/native"
				return h, max1(suggestedUnits), meta
			}
		}
	}
	delay := 2 * time.Second
	units = suggestedUnits
	if units == 0 {
		units = 1
	}
	switch kind {
	case "inference":
		delay = 3 * time.Second
	case "transcoding":
		delay = 4 * time.Second
	case "rendering":
		delay = 5 * time.Second
	default:
		delay = 2 * time.Second
	}
	time.Sleep(delay)
	rand := make([]byte, 16)
	_, _ = crand.Read(rand)
	sum := sha256.Sum256(append([]byte(kind+"|"+payloadURL+"|"+time.Now().UTC().Format(time.RFC3339Nano)), rand...))
	meta["executor"] = "simulated"
	if os.Getenv("AK_REQUIRE_REAL") == "1" {
		return "", 0, meta
	}
	return hex.EncodeToString(sum[:]), units, meta
}

func max1(u uint32) uint32 {
	if u == 0 {
		return 1
	}
	return u
}

func hasNvidiaGPU() bool {
	if _, err := exec.LookPath("nvidia-smi"); err != nil {
		return false
	}
	return true
}

func tempPaths() (string, string) {
	in := os.TempDir() + "/ryvion_in" + time.Now().Format("150405.000")
	out := os.TempDir() + "/ryvion_out" + time.Now().Format("150405.000")
	return in, out
}
