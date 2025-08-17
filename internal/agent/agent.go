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
	"io/ioutil"
	"net/http"
	neturl "net/url"
	"os"
	"os/exec"
	"time"

	"log"
	"strings"

	execreal "github.com/akatosh/node-agent/executor"
	keysol "github.com/akatosh/node-agent/internal/crypto"
	execsim "github.com/akatosh/node-agent/internal/executor"
	"github.com/akatosh/node-agent/internal/metrics"
	runner "github.com/akatosh/node-agent/internal/runner"

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
	regMsg := registerMessageWithReferral(a.PubKey, a.DeviceType, caps.GPUModel, caps.CPUCores, caps.RAMBytes, caps.VRAMBytes, caps.Sensors, caps.BandwidthMbps, 0, 0, referral)
	body := map[string]any{
		"public_key_hex":     hex.EncodeToString(a.PubKey),
		"device_type":        a.DeviceType,
		"gpu_model":          caps.GPUModel,
		"cpu_cores":          caps.CPUCores,
		"ram_bytes":          caps.RAMBytes,
		"vram_bytes":         caps.VRAMBytes,
		"sensors":            caps.Sensors,
		"bandwidth_mbps":     caps.BandwidthMbps,
		"geohash_bucket":     uint64(0),
		"attestation_method": uint32(0),
		"referral_code":      referral,
		"signature":          ed25519.Sign(a.PrivKey, regMsg),
	}
	if err := postJSON(a.HubBaseURL+"/api/v1/node/register", body, nil); err != nil {
		log.Printf("Registration failed: %v", err)
		return err
	}
	log.Printf("Registration succeeded")
	_ = a.solveChallenge()
	_ = a.sendHealthReport()
	return nil
}

func (a *Agent) CreateReferral() (string, error) {
	msg := sha256.Sum256([]byte("AKT1|referral|" + hex.EncodeToString(a.PubKey)))
	sig := ed25519.Sign(a.PrivKey, msg[:])
	var out struct {
		Code string `json:"code"`
	}
	body := map[string]any{"pubkey": []byte(a.PubKey), "signature": sig}
	if err := postJSON(a.HubBaseURL+"/api/v1/node/referral/create", body, &out); err != nil {
		return "", err
	}
	if out.Code == "" {
		return "", fmt.Errorf("no code returned")
	}
	return out.Code, nil
}

func (a *Agent) HeartbeatOnce() error {
	m := metrics.Sample()
	hbMsg := heartbeatMessage(a.PubKey, time.Now().UnixMilli(), m.CPUUtil, m.MemUtil, m.GPUUtil, m.PowerWatts)
	body := map[string]any{
		"public_key_hex": hex.EncodeToString(a.PubKey),
		"timestamp_ms":   time.Now().UnixMilli(),
		"cpu_util":       m.CPUUtil,
		"mem_util":       m.MemUtil,
		"gpu_util":       m.GPUUtil,
		"power_watts":    m.PowerWatts,
		"signature":      ed25519.Sign(a.PrivKey, hbMsg),
	}
	return postJSON(a.HubBaseURL+"/api/v1/node/heartbeat", body, nil)
}

func (a *Agent) FetchAndRunWork() error {
	url := fmt.Sprintf("%s/api/v1/node/work?pubkey=%s", a.HubBaseURL, strings.ToLower(hex.EncodeToString(a.PubKey)))
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode == 204 {
		return nil
	}
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("work req status %d: %s", resp.StatusCode, string(b))
	}
	var wa struct {
		JobID      string `json:"job_id"`
		JobPubkey  string `json:"job_pubkey"`
		Kind       string `json:"kind"`
		PayloadURL string `json:"payload_url"`
		Units      uint32 `json:"units"`
		Image      string `json:"image"`
		SpecJSON   string `json:"spec_json"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&wa); err != nil {
		return err
	}

	start := time.Now()
	var resHashHex string
	var units uint32
	var execMeta map[string]any

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	if strings.TrimSpace(wa.Image) != "" && strings.TrimSpace(wa.SpecJSON) != "" && os.Getenv("AK_EXECUTOR_MODE") != "sim" {

		rr, rerr := runner.RunOCI(ctx, wa.Image, []byte(wa.SpecJSON), "auto")
		if rerr == nil {
			resHashHex = rr.ResultHash
			units = max(wa.Units)
			execMeta = map[string]any{"executor": "oci", "duration_ms": rr.Duration.Milliseconds(), "exit_code": rr.ExitCode, "stderr_tail": rr.LogsTail, "metrics": rr.Metrics}
			if strings.TrimSpace(os.Getenv("AK_UPLOAD")) == "1" && rr.OutputPath != "" {
				if blobURL, err := a.uploadArtifact(ctx, wa.JobID, rr.OutputPath); err == nil && blobURL != "" {
					if execMeta == nil {
						execMeta = map[string]any{}
					}
					execMeta["blob_url"] = blobURL
				}
			}
		} else {
			log.Printf("OCI runner failed: %v", rerr)
		}
	}

    if resHashHex == "" {
        if executor, err := execreal.NewWorkloadExecutor(); err == nil {
			if wa.Kind == "inference" || wa.Kind == "transcoding" || wa.Kind == "rendering" {
				if result, err := executor.ExecuteInference(ctx, wa.JobID, wa.Kind, wa.PayloadURL); err == nil {
					resHashHex = result.ResultHash
					units = uint32(len(result.OutputData))
					execMeta = map[string]any{
						"executor":    "docker",
						"duration_ms": result.Metrics.Duration.Milliseconds(),
						"gpu_util":    result.Metrics.GPUUtilization,
						"power_watts": result.Metrics.PowerUsage,
					}
                } else {
                    log.Printf("Docker execution failed: %v, trying native executors", err)
                    resHashHex, units, execMeta = execsim.Run(wa.Kind, wa.PayloadURL, wa.Units)
                }
            } else {
                resHashHex, units, execMeta = execsim.Run(wa.Kind, wa.PayloadURL, wa.Units)
            }
        } else {
            log.Printf("Docker executor unavailable: %v, trying native executors", err)
            resHashHex, units, execMeta = execsim.Run(wa.Kind, wa.PayloadURL, wa.Units)
        }
    }

    if units == 0 { units = 1 }
    // If real work is required and we still fell back to simulation, abort
    if os.Getenv("AK_REQUIRE_REAL") == "1" {
        if execMeta != nil {
            if ex, ok := execMeta["executor"].(string); ok && ex == "simulated" {
                return fmt.Errorf("real execution required but not available")
            }
        }
    }

	rcptMsg := receiptMessage(a.PubKey, wa.JobID, resHashHex, uint64(units))
	elapsed := time.Since(start)
	gpu := metrics.GPUUtilSnapshot(context.Background())
	receipt := map[string]any{
		"job_id":               wa.JobID,
		"pubkey":               []byte(a.PubKey),
		// Keep legacy field for backward compatibility
		"result_hash":          resHashHex,
		// New explicit hex field to avoid base64 vs hex ambiguity on the hub
		"result_hash_hex":      resHashHex,
		"metering_units":       uint64(units),
		"green_multiplier_bps": uint32(0),
		"signature":            ed25519.Sign(a.PrivKey, rcptMsg),
		"metadata": map[string]any{
			"duration_ms": elapsed.Milliseconds(),
			"executor":    execMeta["executor"],
			"exit_code":   execMeta["exit_code"],
			"stderr_tail": execMeta["stderr_tail"],
			"gpu_util":    gpu,
		},
	}
	if strings.TrimSpace(os.Getenv("AK_UPLOAD")) == "1" && strings.TrimSpace(wa.Image) != "" && strings.TrimSpace(wa.SpecJSON) != "" {
		if blobURL, err := a.uploadArtifact(ctx, wa.JobID, ""); err == nil && blobURL != "" {
			if md, ok := receipt["metadata"].(map[string]any); ok {
				md["blob_url"] = blobURL
			}
		}
	}
	if err := postJSON(a.HubBaseURL+"/api/v1/node/receipt", receipt, nil); err != nil {
		return err
	}
	if getenvBool("AK_ONCHAIN_RECEIPTS") {
		_ = a.submitOnchainReceipt(wa.JobID, wa.JobPubkey, resHashHex, uint64(units))
	}
	return nil
}

func (a *Agent) SavePayoutWallet(wallet string) error {
	ts := time.Now().UnixMilli()
	msg := sha256.Sum256([]byte("AKT1|payout|" + hex.EncodeToString(a.PubKey) + "|" + wallet + "|" + itoaI64(ts)))
	body := map[string]any{
		"pubkey":       []byte(a.PubKey),
		"wallet":       wallet,
		"timestamp_ms": ts,
		"signature":    ed25519.Sign(a.PrivKey, msg[:]),
	}
	return postJSON(a.HubBaseURL+"/api/v1/node/payout/save", body, nil)
}

func (a *Agent) signRandom() []byte {
	buf := make([]byte, 32)
	_, _ = crand.Read(buf)
	sig := ed25519.Sign(a.PrivKey, buf)
	return sig
}

func (a *Agent) solveChallenge() error {
	var resp struct {
		Nonce     string `json:"nonce"`
		ExpiresMs int64  `json:"expires_ms"`
	}
	if err := postJSON(a.HubBaseURL+"/api/v1/node/challenge/request", map[string]any{"pubkey": []byte(a.PubKey)}, &resp); err != nil {
		return err
	}
	if resp.Nonce == "" {
		return fmt.Errorf("no nonce")
	}
	msg := challengeMessage(resp.Nonce)
	sig := ed25519.Sign(a.PrivKey, msg)
	return postJSON(a.HubBaseURL+"/api/v1/node/challenge/solve", map[string]any{"pubkey": []byte(a.PubKey), "nonce": resp.Nonce, "signature": sig}, nil)
}

func getenvBool(key string) bool {
	return strings.ToLower(os.Getenv(key)) == "true"
}

func (a *Agent) sendHealthReport() error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	gpuReady := false
	dockerGPU := false
	msgParts := []string{}
	if _, err := exec.LookPath("nvidia-smi"); err == nil {
		gpuReady = true
		msgParts = append(msgParts, "nvidia-smi:ok")
	} else {
		msgParts = append(msgParts, "nvidia-smi:missing")
	}
	if _, err := exec.LookPath("docker"); err == nil {
		cctx, ccancel := context.WithTimeout(ctx, 6*time.Second)
		defer ccancel()
		cmd := exec.CommandContext(cctx, "docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:12.2.0-base-ubuntu20.04", "nvidia-smi", "-L")
		if out, err := cmd.CombinedOutput(); err == nil && len(out) > 0 {
			dockerGPU = true
			msgParts = append(msgParts, "docker-gpu:ok")
		} else {
			msgParts = append(msgParts, "docker-gpu:fail")
		}
	} else {
		msgParts = append(msgParts, "docker:missing")
	}
	ts := time.Now().UnixMilli()
	txt := strings.Join(msgParts, ",")
	b := []byte("AKT1|health|" + hex.EncodeToString(a.PubKey) + "|" + itoaI64(ts) + "|" + btoa(gpuReady) + "|" + btoa(dockerGPU) + "|" + txt)
	sum := sha256.Sum256(b)
	sig := ed25519.Sign(a.PrivKey, sum[:])
	body := map[string]any{
		"pubkey":       []byte(a.PubKey),
		"timestamp_ms": ts,
		"gpu_ready":    gpuReady,
		"docker_gpu":   dockerGPU,
		"message":      txt,
		"signature":    sig,
	}
	return postJSON(a.HubBaseURL+"/api/v1/node/health", body, nil)
}

func btoa(b bool) string {
	if b {
		return "1"
	}
	return "0"
}

func registerMessage(pub ed25519.PublicKey, deviceType, gpuModel string, cpuCores uint32, ram, vram uint64, sensors string, bandwidth, geohash uint64, attest uint32) []byte {
	return registerMessageWithReferral(pub, deviceType, gpuModel, cpuCores, ram, vram, sensors, bandwidth, geohash, attest, "")
}

func registerMessageWithReferral(pub ed25519.PublicKey, deviceType, gpuModel string, cpuCores uint32, ram, vram uint64, sensors string, bandwidth, geohash uint64, attest uint32, referral string) []byte {
	// Use simple string-based signature - the original working format
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
	kpPath := strings.TrimSpace(os.Getenv("AK_SOLANA_KEYPAIR"))
	rpcURL := strings.TrimSpace(os.Getenv("SOLANA_RPC_URL"))
	if kpPath == "" || rpcURL == "" {
		return nil
	}
	payer, err := keysol.LoadSolanaKeypair(kpPath)
	if err != nil {
		return err
	}
	var nodePK solana.PublicKey
	if len(a.PubKey) != 32 {
		return fmt.Errorf("invalid node ed25519 public key")
	}
	copy(nodePK[:], []byte(a.PubKey))
	var prepResp struct {
		ProgramID string `json:"program_id"`
		DataB64   string `json:"data_base64"`
		Accounts  []struct {
			Pubkey     string `json:"pubkey"`
			IsSigner   bool   `json:"is_signer"`
			IsWritable bool   `json:"is_writable"`
		} `json:"accounts"`
		RecentBlockhash string `json:"recent_blockhash"`
	}
	body := map[string]any{
		"job_id":               jobID,
		"node_pubkey":          nodePK.String(),
		"result_hash_hex":      strings.ToLower(resultHashHex),
		"units":                units,
		"green_multiplier_bps": uint32(10000),
		"payer_pubkey":         payer.PublicKey().String(),
	}
	if err := postJSON(a.HubBaseURL+"/api/v1/node/receipt/prepare", body, &prepResp); err != nil {
		return err
	}
	prog, err := solana.PublicKeyFromBase58(prepResp.ProgramID)
	if err != nil {
		return err
	}
	data, err := decodeB64(prepResp.DataB64)
	if err != nil {
		return err
	}
	metas := make([]*solana.AccountMeta, 0, len(prepResp.Accounts))
	for _, acc := range prepResp.Accounts {
		pk, err := solana.PublicKeyFromBase58(acc.Pubkey)
		if err != nil {
			return err
		}
		metas = append(metas, solana.NewAccountMeta(pk, acc.IsSigner, acc.IsWritable))
	}
	ix := solana.NewInstruction(prog, metas, data)
	client := rpc.New(rpcURL)
	bh := prepResp.RecentBlockhash
	if strings.TrimSpace(bh) == "" {
		got, err := client.GetLatestBlockhash(context.Background(), rpc.CommitmentFinalized)
		if err != nil {
			return err
		}
		bh = got.Value.Blockhash.String()
	}
	tx, err := solana.NewTransaction([]solana.Instruction{ix}, solana.MustHashFromBase58(bh), solana.TransactionPayer(payer.PublicKey()))
	if err != nil {
		return err
	}
	// Sign the transaction manually to avoid interface issues
	// Get the message bytes to sign
	messageBytes, err := tx.Message.MarshalBinary()
	if err != nil {
		return err
	}
	
	// Sign the message with the private key
	signature, err := payer.Sign(messageBytes)
	if err != nil {
		return err
	}
	
	// Add the signature to the transaction
	tx.Signatures = []solana.Signature{signature}
	_, err = client.SendTransactionWithOpts(context.Background(), tx, rpc.TransactionOpts{PreflightCommitment: rpc.CommitmentFinalized})
	return err
}

func itoaU64(v uint64) string            { return fmtUint64(v) }
func itoaU32(v uint32) string            { return fmtUint64(uint64(v)) }
func itoaI64(v int64) string             { return fmtInt64(v) }
func ftoa(f float64) string              { b, _ := json.Marshal(f); return string(b) }
func decodeB64(s string) ([]byte, error) { return base64.StdEncoding.DecodeString(s) }
func fmtUint64(v uint64) string {
	if v == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for v > 0 {
		i--
		buf[i] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[i:])
}
func fmtInt64(v int64) string {
	if v >= 0 {
		return fmtUint64(uint64(v))
	}
	return "-" + fmtUint64(uint64(-v))
}

func postJSON(url string, body any, out any) error {
    // Optional bind headers to auto-associate node to buyer or wallet
    bindTok := strings.TrimSpace(os.Getenv("AK_BIND_TOKEN"))
    wallet := strings.TrimSpace(os.Getenv("AK_WALLET"))
    if w := strings.TrimSpace(os.Getenv("SOLANA_WALLET")); w != "" {
        wallet = w
    }

    b, _ := json.Marshal(body)
    req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(b))
    if err != nil { return err }
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("User-Agent", "ryvion-node-agent/1.0")
    if bindTok != "" { req.Header.Set("X-Bind-Token", bindTok) }
    if wallet != "" { req.Header.Set("X-Wallet", wallet) }
    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    if resp.StatusCode >= 300 {
        rb, _ := io.ReadAll(resp.Body)
        return fmt.Errorf("POST %s: %d %s", url, resp.StatusCode, string(rb))
    }
    if out != nil {
        return json.NewDecoder(resp.Body).Decode(out)
    }
    return nil
}

func (a *Agent) uploadArtifact(ctx context.Context, jobID string, outPath string) (string, error) {

	if strings.TrimSpace(outPath) == "" {
		outPath = strings.TrimSpace(os.Getenv("AK_OUTPUT_PATH"))
	}
	if outPath == "" {
		return "", fmt.Errorf("no output path configured")
	}
	fi, err := os.Stat(outPath)
	if err != nil || fi.IsDir() {
		return "", fmt.Errorf("no artifact file")
	}
	size := uint64(fi.Size())

	msg := sha256.Sum256([]byte("AKT1|upload_prep|" + jobID + "|" + hex.EncodeToString(a.PubKey) + "|" + "application/octet-stream" + "|" + itoaU64(size)))
	body := map[string]any{
		"pubkey":       []byte(a.PubKey),
		"job_id":       jobID,
		"content_type": "application/octet-stream",
		"size_bytes":   size,
		"signature":    ed25519.Sign(a.PrivKey, msg[:]),
	}
	var prep struct {
		OK        bool   `json:"ok"`
		Provider  string `json:"provider"`
		PutURL    string `json:"put_url"`
		ExpiresAt string `json:"expires_at"`
	}
	if err := postJSON(a.HubBaseURL+"/api/v1/node/upload/prepare", body, &prep); err != nil {
		return "", err
	}
	if strings.TrimSpace(prep.PutURL) == "" {
		return "", fmt.Errorf("no put_url")
	}

	f, err := os.Open(outPath)
	if err != nil {
		return "", err
	}
	defer f.Close()
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, absolutize(a.HubBaseURL, prep.PutURL), f)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		b, _ := ioutil.ReadAll(resp.Body)
		return "", fmt.Errorf("put failed: %d %s", resp.StatusCode, string(b))
	}

	if strings.HasPrefix(prep.PutURL, "/") {
		var putResp struct {
			OK  bool   `json:"ok"`
			URL string `json:"url"`
		}

		if b, err := ioutil.ReadAll(resp.Body); err == nil && len(b) > 0 {
			_ = json.Unmarshal(b, &putResp)
			if strings.TrimSpace(putResp.URL) != "" {
				return absolutize(a.HubBaseURL, putResp.URL), nil
			}
		}

		return a.HubBaseURL + "/api/v1/blob/" + jobID, nil
	}
	return prep.PutURL, nil
}

func absolutize(base, maybeRel string) string {
	if strings.HasPrefix(maybeRel, "http://") || strings.HasPrefix(maybeRel, "https://") {
		return maybeRel
	}
	if strings.HasPrefix(maybeRel, "/") {
		u, err := neturl.Parse(base)
		if err != nil {
			return maybeRel
		}
		u.Path = maybeRel
		return u.String()
	}
	return maybeRel
}
