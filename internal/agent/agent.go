package agent

import (
    "bytes"
    "crypto/ed25519"
    crand "crypto/rand"
    "crypto/sha256"
    "encoding/hex"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "time"

    "github.com/akatosh/node-agent/internal/metrics"
)

type Agent struct {
    HubBaseURL string
    PubKey     ed25519.PublicKey
    PrivKey    ed25519.PrivateKey
    DeviceType string
}

func (a *Agent) Register() error {
    caps := metrics.Capabilities(a.DeviceType)
    // Build canonical message
    regMsg := registerMessage(a.PubKey, a.DeviceType, caps.GPUModel, caps.CPUCores, caps.RAMBytes, caps.VRAMBytes, caps.Sensors, caps.BandwidthMbps, 0, 0)
    body := map[string]any{
        "pubkey":            []byte(a.PubKey),
        "device_type":       a.DeviceType,
        "gpu_model":         caps.GPUModel,
        "cpu_cores":         caps.CPUCores,
        "ram_bytes":         caps.RAMBytes,
        "vram_bytes":        caps.VRAMBytes,
        "sensors":           caps.Sensors,
        "bandwidth_mbps":    caps.BandwidthMbps,
        "geohash_bucket":    uint64(0),
        "attestation_method": uint32(0),
        "signature":         ed25519.Sign(a.PrivKey, regMsg),
    }
    return postJSON(a.HubBaseURL+"/api/v1/node/register", body, nil)
}

func (a *Agent) HeartbeatOnce() error {
    m := metrics.Sample()
    hbMsg := heartbeatMessage(a.PubKey, time.Now().UnixMilli(), m.CPUUtil, m.MemUtil, m.GPUUtil, m.PowerWatts)
    body := map[string]any{
        "pubkey":       []byte(a.PubKey),
        "timestamp_ms": time.Now().UnixMilli(),
        "cpu_util":     m.CPUUtil,
        "mem_util":     m.MemUtil,
        "gpu_util":     m.GPUUtil,
        "power_watts":  m.PowerWatts,
        "signature":    ed25519.Sign(a.PrivKey, hbMsg),
    }
    return postJSON(a.HubBaseURL+"/api/v1/node/heartbeat", body, nil)
}

func (a *Agent) FetchAndRunWork() error {
    url := fmt.Sprintf("%s/api/v1/node/work?pubkey=%s", a.HubBaseURL, hex.EncodeToString(a.PubKey))
    resp, err := http.Get(url)
    if err != nil { return err }
    defer resp.Body.Close()
    if resp.StatusCode == 204 { return nil }
    if resp.StatusCode != 200 {
        b, _ := io.ReadAll(resp.Body)
        return fmt.Errorf("work req status %d: %s", resp.StatusCode, string(b))
    }
    var wa struct{
        JobID string `json:"job_id"`
        Kind string `json:"kind"`
        PayloadURL string `json:"payload_url"`
    }
    if err := json.NewDecoder(resp.Body).Decode(&wa); err != nil { return err }

    // v0: simulate doing the work
    time.Sleep(3 * time.Second)

    resHash := hex.EncodeToString(metrics.RandomHash())
    rcptMsg := receiptMessage(a.PubKey, wa.JobID, resHash, 1)
    receipt := map[string]any{
        "job_id":              wa.JobID,
        "pubkey":              []byte(a.PubKey),
        "result_hash":         resHash,
        "metering_units":      uint64(1),
        "green_multiplier_bps": uint32(0),
        "signature":           ed25519.Sign(a.PrivKey, rcptMsg),
    }
    return postJSON(a.HubBaseURL+"/api/v1/node/receipt", receipt, nil)
}

func (a *Agent) signRandom() []byte {
    buf := make([]byte, 32)
    _, _ = crand.Read(buf)
    sig := ed25519.Sign(a.PrivKey, buf)
    return sig
}

// Canonical message builders (must match server)
func registerMessage(pub ed25519.PublicKey, deviceType, gpuModel string, cpuCores uint32, ram, vram uint64, sensors string, bandwidth, geohash uint64, attest uint32) []byte {
    s := "AKT1|register|" +
        hex.EncodeToString(pub) + "|" +
        deviceType + "|" +
        gpuModel + "|" +
        itoaU32(cpuCores) + "|" +
        itoaU64(ram) + "|" +
        itoaU64(vram) + "|" +
        sensors + "|" +
        itoaU64(bandwidth) + "|" +
        itoaU64(geohash) + "|" +
        itoaU32(attest)
    sum := sha256.Sum256([]byte(s))
    return sum[:]
}

func heartbeatMessage(pub ed25519.PublicKey, ts int64, cpu, mem, gpu, watts float64) []byte {
    s := "AKT1|heartbeat|" +
        hex.EncodeToString(pub) + "|" +
        itoaI64(ts) + "|" + ftoa(cpu) + "|" + ftoa(mem) + "|" + ftoa(gpu) + "|" + ftoa(watts)
    sum := sha256.Sum256([]byte(s))
    return sum[:]
}

func receiptMessage(pub ed25519.PublicKey, jobID, resultHash string, units uint64) []byte {
    s := "AKT1|receipt|" + jobID + "|" + hex.EncodeToString(pub) + "|" + resultHash + "|" + itoaU64(units)
    sum := sha256.Sum256([]byte(s))
    return sum[:]
}

func itoaU64(v uint64) string { return fmtUint64(v) }
func itoaU32(v uint32) string { return fmtUint64(uint64(v)) }
func itoaI64(v int64) string  { return fmtInt64(v) }
func ftoa(f float64) string   { b, _ := json.Marshal(f); return string(b) }
func fmtUint64(v uint64) string {
    if v == 0 { return "0" }
    var buf [20]byte
    i := len(buf)
    for v > 0 { i--; buf[i] = byte('0' + v%10); v /= 10 }
    return string(buf[i:])
}
func fmtInt64(v int64) string {
    if v >= 0 { return fmtUint64(uint64(v)) }
    return "-" + fmtUint64(uint64(-v))
}

func postJSON(url string, body any, out any) error {
    b, _ := json.Marshal(body)
    resp, err := http.Post(url, "application/json", bytes.NewReader(b))
    if err != nil { return err }
    defer resp.Body.Close()
    if resp.StatusCode >= 300 {
        rb, _ := io.ReadAll(resp.Body)
        return fmt.Errorf("POST %s: %d %s", url, resp.StatusCode, string(rb))
    }
    if out != nil { return json.NewDecoder(resp.Body).Decode(out) }
    return nil
}
