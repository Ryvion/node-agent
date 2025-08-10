package executor

import (
    "context"
    "crypto/sha256"
    crand "crypto/rand"
    "encoding/hex"
    "os"
    "time"
)

// Run executes or simulates a job of a given kind and payload.
// Returns a hex result hash and the metering units consumed.
// v0: simulate with a short delay and deterministic-ish hashing.
func Run(kind string, payloadURL string, suggestedUnits uint32) (resultHashHex string, units uint32, meta map[string]any) {
    meta = make(map[string]any)
    // Optional docker path when AK_EXECUTOR_MODE=docker and kind suggests inference/transcoding
    if os.Getenv("AK_EXECUTOR_MODE") == "docker" {
        // Load catalog once (no-op if already loaded)
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
    // Simulate varying duration & units per kind
    delay := 2 * time.Second
    units = suggestedUnits
    if units == 0 { units = 1 }
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
    // Mix in random bytes to avoid trivially predictable hashes
    rand := make([]byte, 16)
    _, _ = crand.Read(rand)
    sum := sha256.Sum256(append([]byte(kind+"|"+payloadURL+"|"+time.Now().UTC().Format(time.RFC3339Nano)), rand...))
    meta["executor"] = "simulated"
    return hex.EncodeToString(sum[:]), units, meta
}

func max1(u uint32) uint32 { if u == 0 { return 1 }; return u }
