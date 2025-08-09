package metrics

import (
    crand "crypto/rand"
    "math/rand"
    "runtime"
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

