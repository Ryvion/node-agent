package agent

import (
    "bytes"
    "context"
    "crypto/ed25519"
    crand "crypto/rand"
    "crypto/sha256"
    "encoding/hex"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "os"
    "time"

    "log"
    "strings"
    
    "github.com/akatosh/node-agent/internal/metrics"
    execsim "github.com/akatosh/node-agent/internal/executor"
    execreal "github.com/akatosh/node-agent/executor"
)

type Agent struct {
    HubBaseURL string
    PubKey     ed25519.PublicKey
    PrivKey    ed25519.PrivateKey
    DeviceType string
}

func (a *Agent) Register() error { return a.RegisterWithReferral("") }

func (a *Agent) RegisterWithReferral(referral string) error {
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
        "referral_code":     referral,
        "signature":         ed25519.Sign(a.PrivKey, regMsg),
    }
    if err := postJSON(a.HubBaseURL+"/api/v1/node/register", body, nil); err != nil { return err }
    // Solve a challenge to earn initial reputation (optional)
    _ = a.solveChallenge()
    return nil
}

// CreateReferral requests a referral code from the hub for this node.
func (a *Agent) CreateReferral() (string, error) {
    // Signature over sha256("AKT1|referral|" + hex(pubkey))
    msg := sha256.Sum256([]byte("AKT1|referral|" + hex.EncodeToString(a.PubKey)))
    sig := ed25519.Sign(a.PrivKey, msg[:])
    var out struct{ Code string `json:"code"` }
    body := map[string]any{"pubkey": []byte(a.PubKey), "signature": sig}
    if err := postJSON(a.HubBaseURL+"/api/v1/node/referral/create", body, &out); err != nil { return "", err }
    if out.Code == "" { return "", fmt.Errorf("no code returned") }
    return out.Code, nil
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
        JobPubkey string `json:"job_pubkey"`
        Kind string `json:"kind"`
        PayloadURL string `json:"payload_url"`
        Units uint32 `json:"units"`
    }
    if err := json.NewDecoder(resp.Body).Decode(&wa); err != nil { return err }

    // Execute workload (try real execution first, fallback to simulation)
    start := time.Now()
    var resHashHex string
    var units uint32
    var execMeta map[string]any
    
    // Try real Docker execution first
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
    defer cancel()
    
    if executor, err := execreal.NewWorkloadExecutor(); err == nil {
        // Create a simplified work request (avoiding proto dependency for now)
        if wa.Kind == "inference" || wa.Kind == "transcoding" || wa.Kind == "rendering" {
            if result, err := executor.ExecuteInference(ctx, wa.JobID, wa.Kind, wa.PayloadURL); err == nil {
                resHashHex = result.ResultHash
                units = uint32(len(result.OutputData))
                execMeta = map[string]any{
                    "executor": "docker",
                    "duration_ms": result.Metrics.Duration.Milliseconds(),
                    "gpu_util": result.Metrics.GPUUtilization,
                    "power_watts": result.Metrics.PowerUsage,
                }
            } else {
                log.Printf("Docker execution failed: %v, falling back to simulation", err)
                resHashHex, units, execMeta = execsim.Run(wa.Kind, wa.PayloadURL, wa.Units)
            }
        } else {
            // Unsupported job type, use simulation
            resHashHex, units, execMeta = execsim.Run(wa.Kind, wa.PayloadURL, wa.Units)
        }
    } else {
        log.Printf("Docker executor unavailable: %v, using simulation", err)
        resHashHex, units, execMeta = execsim.Run(wa.Kind, wa.PayloadURL, wa.Units)
    }
    
    if units == 0 { units = 1 }

    rcptMsg := receiptMessage(a.PubKey, wa.JobID, resHashHex, uint64(units))
    elapsed := time.Since(start)
    // Optional GPU util snapshot
    gpu := metrics.GPUUtilSnapshot(context.Background())
    receipt := map[string]any{
        "job_id":              wa.JobID,
        "pubkey":              []byte(a.PubKey),
        "result_hash":         resHashHex,
        "metering_units":      uint64(units),
        "green_multiplier_bps": uint32(0),
        "signature":           ed25519.Sign(a.PrivKey, rcptMsg),
        "metadata": map[string]any{
            "duration_ms": elapsed.Milliseconds(),
            "executor": execMeta["executor"],
            "exit_code": execMeta["exit_code"],
            "stderr_tail": execMeta["stderr_tail"],
            "gpu_util": gpu,
        },
    }
    if err := postJSON(a.HubBaseURL+"/api/v1/node/receipt", receipt, nil); err != nil { return err }
    // Optional on-chain receipt for Explorer visibility (DEV/opt-in)
    if getenvBool("AK_ONCHAIN_RECEIPTS") {
        _ = a.submitOnchainReceipt(wa.JobID, wa.JobPubkey, resHashHex, uint64(units))
    }
    return nil
}

func (a *Agent) signRandom() []byte {
    buf := make([]byte, 32)
    _, _ = crand.Read(buf)
    sig := ed25519.Sign(a.PrivKey, buf)
    return sig
}

func (a *Agent) solveChallenge() error {
    // Request a challenge
    var resp struct{ Nonce string `json:"nonce"`; ExpiresMs int64 `json:"expires_ms"` }
    if err := postJSON(a.HubBaseURL+"/api/v1/node/challenge/request", map[string]any{"pubkey": []byte(a.PubKey)}, &resp); err != nil { return err }
    if resp.Nonce == "" { return fmt.Errorf("no nonce") }
    // Sign
    msg := challengeMessage(resp.Nonce)
    sig := ed25519.Sign(a.PrivKey, msg)
    // Submit
    return postJSON(a.HubBaseURL+"/api/v1/node/challenge/solve", map[string]any{"pubkey": []byte(a.PubKey), "nonce": resp.Nonce, "signature": sig}, nil)
}

func getenvBool(key string) bool {
    return strings.ToLower(os.Getenv(key)) == "true"
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

func challengeMessage(nonce string) []byte {
    s := "AKT1|challenge|" + nonce
    sum := sha256.Sum256([]byte(s))
    return sum[:]
}

func (a *Agent) submitOnchainReceipt(jobID, jobPubHex, resultHashHex string, units uint64) error {
    // TODO: Implement on-chain receipt submission when Solana library is updated
    log.Printf("On-chain receipt disabled (Solana lib compatibility): job=%s result=%s units=%d", jobID, resultHashHex, units)
    return nil
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
