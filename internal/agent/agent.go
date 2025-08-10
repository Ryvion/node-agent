package agent

import (
    "bytes"
    "context"
    "crypto/ed25519"
    crand "crypto/rand"
    "crypto/sha256"
    "encoding/base64"
    "encoding/hex"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "os"
    "time"

    "github.com/akatosh/node-agent/internal/metrics"
    execsim "github.com/akatosh/node-agent/internal/executor"
    keyutil "github.com/akatosh/node-agent/internal/crypto"
    solana "github.com/gagliardetto/solana-go"
    "github.com/gagliardetto/solana-go/rpc"
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

    // Execute workload (simulated v0; pluggable executor)
    start := time.Now()
    resHashHex, units, execMeta := execsim.Run(wa.Kind, wa.PayloadURL, wa.Units)
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
    rpcURL := os.Getenv("AK_SOL_RPC")
    if rpcURL == "" { return nil }
    payer := solana.PublicKeyFromBytes(a.PubKey)
    // If an external Solana keypair is provided and matches the node pubkey, use it for signing.
    var ext solana.PrivateKey
    if kp := os.Getenv("AK_SOL_KEYPAIR"); kp != "" {
        if sk, err := keyutil.LoadSolanaKeypair(kp); err == nil {
            if sk.PublicKey().Equals(payer) { ext = sk }
        }
    }
    // Ask orchestrator to prepare instruction (it knows program IDs)
    var prep map[string]any
    body := map[string]any{
        "job_id": jobID,
        "node_pubkey": payer.String(),
        "result_hash_hex": resultHashHex,
        "units": units,
        "payer_pubkey": payer.String(),
    }
    if err := postJSON(a.HubBaseURL+"/api/v1/node/receipt/prepare", body, &prep); err != nil { return err }
    progID, _ := solana.PublicKeyFromBase58(prep["program_id"].(string))
    dataB64 := prep["data_base64"].(string)
    data, _ := base64.StdEncoding.DecodeString(dataB64)
    arr := prep["accounts"].([]any)
    keys := make([]solana.AccountMeta, 0, len(arr))
    for _, v := range arr {
        m := v.(map[string]any)
        pk, _ := solana.PublicKeyFromBase58(m["pubkey"].(string))
        keys = append(keys, solana.AccountMeta{PublicKey: pk, IsSigner: m["is_signer"].(bool), IsWritable: m["is_writable"].(bool)})
    }
    ix := solana.NewInstruction(progID, keys, data)
    bh := prep["recent_blockhash"].(string)
    tx, err := solana.NewTransaction([]solana.Instruction{ix}, solana.HashFromBase58(bh), solana.TransactionPayer(payer))
    if err != nil { return err }
    if _, err := tx.Sign(func(pub solana.PublicKey) (solana.PrivateKey, bool) {
        if !pub.Equals(payer) { return nil, false }
        if ext != nil { return ext, true }
        return solana.PrivateKey(a.PrivKey), true
    }); err != nil { return err }
    client := rpc.New(rpcURL)
    var sig solana.Signature
    var sendErr error
    for attempt := 0; attempt < 3; attempt++ {
        sig, sendErr = client.SendTransactionWithOpts(context.Background(), tx, rpc.TransactionOpts{SkipPreflight: false, PreflightCommitment: rpc.CommitmentProcessed})
        if sendErr == nil { break }
        time.Sleep(time.Duration(1<<attempt) * time.Second)
    }
    if sendErr != nil { return sendErr }
    // Log explorer URL
    cluster := ""
    low := strings.ToLower(rpcURL)
    if strings.Contains(low, "devnet") { cluster = "?cluster=devnet" } else if strings.Contains(low, "testnet") { cluster = "?cluster=testnet" }
    fmt.Printf("on-chain receipt tx: https://explorer.solana.com/tx/%s%s\n", sig.String(), cluster)
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
