//go:build !containers

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

type TranscodeSpec struct {
    InputURL  string
    OutputURL string // presigned upload
    Args      []string // e.g. ["-i","{input}","-c:v","h264_nvenc","-b:v","4M","/tmp/out.mp4"]
    Timeout   time.Duration
}

func RunFFmpeg(ctx context.Context, spec TranscodeSpec) (hash string, err error) {
    in := spec.InputURL
    outPath := "/tmp/out.mp4" // keep simple
    
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
        return "", err
        _ = out
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