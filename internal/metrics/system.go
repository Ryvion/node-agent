package metrics

import (
    crand "crypto/rand"
    "context"
    "math/rand"
    "os/exec"
    "runtime"
    "strconv"
    "strings"
    "time"
)

type CapSet struct {
    GPUModel      string
    CPUCores      uint32
    RAMBytes      uint64
    VRAMBytes     uint64
    Sensors       string
    BandwidthMbps uint64
}

type SampleSet struct {
    CPUUtil    float64
    MemUtil    float64
    GPUUtil    float64
    PowerWatts float64
}

func Capabilities(deviceType string) CapSet {
    // v0: very coarse introspection using stdlib only
    return CapSet{
        GPUModel:      "", // detect via nvidia-smi later
        CPUCores:      uint32(runtime.NumCPU()),
        RAMBytes:      8 * 1024 * 1024 * 1024, // placeholder
        VRAMBytes:     0,
        Sensors:       "",
        BandwidthMbps: 100,
    }
}

func Sample() SampleSet {
    return SampleSet{
        CPUUtil:    rand.Float64()*70 + 10,
        MemUtil:    rand.Float64()*70 + 10,
        GPUUtil:    rand.Float64()*70 + 10,
        PowerWatts: rand.Float64()*100 + 50,
    }
}

func RandomHash() []byte {
    b := make([]byte, 32)
    _, _ = crand.Read(b)
    return b
}

// GPUUtilSnapshot attempts to query instantaneous GPU utilization via nvidia-smi.
// Returns -1 if unavailable.
func GPUUtilSnapshot(ctx context.Context) float64 {
    if _, err := exec.LookPath("nvidia-smi"); err != nil { return -1 }
    cctx, cancel := context.WithTimeout(ctx, 2*time.Second)
    defer cancel()
    out, err := exec.CommandContext(cctx, "nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits").CombinedOutput()
    if err != nil { return -1 }
    lines := strings.Split(strings.TrimSpace(string(out)), "\n")
    if len(lines) == 0 { return -1 }
    // Parse first GPU util, fallback to max across lines if multiple
    var best float64 = -1
    for _, ln := range lines {
        ln = strings.TrimSpace(ln)
        if ln == "" { continue }
        if v, err := strconv.Atoi(ln); err == nil {
            if float64(v) > best { best = float64(v) }
        }
    }
    return best
}
