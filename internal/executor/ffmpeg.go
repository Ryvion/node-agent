package executor

import (
    "context"
    "crypto/sha256"
    "encoding/hex"
    "io"
    "os"
    "os/exec"
    "time"
)

// TranscodeSpec defines parameters for an ffmpeg-based transcode job.
// Placeholders {input} and {output} in Args are expanded prior to execution.
type TranscodeSpec struct {
    InputURL  string
    OutputURL string // Optional pre-signed upload URL for artifacts
    Args      []string // Example: ["-i","{input}","-c:v","h264_nvenc","-b:v","4M","{output}"]
    Timeout   time.Duration
}

func RunFFmpeg(ctx context.Context, spec TranscodeSpec) (hash string, err error) {
    in := spec.InputURL
    // Default output path for the produced media file
    outPath := "/tmp/out.mp4"
    
    args := make([]string, 0, len(spec.Args))
    for _, a := range spec.Args {
        if a == "{input}" {
            a = in
        }
        if a == "{output}" {
            a = outPath
        }
        args = append(args, a)
    }
    
    cctx, cancel := context.WithTimeout(ctx, spec.Timeout)
    defer cancel()
    
    cmd := exec.CommandContext(cctx, "ffmpeg", args...)
    if out, err := cmd.CombinedOutput(); err != nil {
        // preserve output for debugging via assigned var
        _ = out
        return "", err
    }

    f, err := os.Open(outPath)
    if err != nil {
        return "", err
    }
    defer f.Close()
    
    h := sha256.New()
    if _, err := io.Copy(h, f); err != nil {
        return "", err
    }
    
    sum := hex.EncodeToString(h.Sum(nil))
    // TODO: upload to spec.OutputURL
    return sum, nil
}
