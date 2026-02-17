# Node-Agent Rewrite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite the node-agent from ~4,500 LOC / 23 files to ~740 LOC / 6 files with clean Go idioms, correct hub protocol (`RYV1|` signing prefix), and no dead code.

**Architecture:** Single binary with 5 internal packages: `nodekey` (keypair), `hub` (typed HTTP client), `hw` (hardware detection), `runner` (OCI execution), `blob` (artifact upload). Main loop: register → [heartbeat + fetch + execute + receipt] every 10s with graceful shutdown.

**Tech Stack:** Go 1.23, ed25519, `log/slog`, Docker CLI (via `os/exec`), no external dependencies.

---

### Task 1: Clean Slate — Delete Old Code, Create Structure

**Files:**
- Delete: everything under `internal/`, `pkg/`, `cmd/`
- Keep: `go.mod`, `go.sum`, `deploy/`, `docs/`, `Dockerfile`, `start.sh`, `.github/`
- Create: `cmd/ryvion-node/main.go` (stub), `internal/nodekey/nodekey.go`, `internal/hub/client.go`, `internal/hw/hw.go`, `internal/runner/oci.go`, `internal/blob/upload.go`

**Steps:**

1. Delete old source directories:
```bash
rm -rf internal/ pkg/ cmd/
```

2. Create new directory structure:
```bash
mkdir -p cmd/ryvion-node internal/nodekey internal/hub internal/hw internal/runner internal/blob
```

3. Update `go.mod` — remove the Docker dependency (we only use `docker` CLI via `os/exec`):
```bash
# Edit go.mod to remove docker dependencies, keep module path
```

4. Run `go mod tidy` to clean up dependencies.

5. Create stub `cmd/ryvion-node/main.go`:
```go
package main

func main() {}
```

6. Verify: `go build ./...` succeeds.

---

### Task 2: `internal/nodekey` — Key Management

**Files:**
- Create: `internal/nodekey/nodekey.go`

**Step 1: Write nodekey.go**

```go
// Package nodekey manages the node's ed25519 identity keypair.
package nodekey

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
)

// LoadOrCreate loads an existing ed25519 private key from path,
// or generates a new one and saves it. Returns an error instead of panicking.
func LoadOrCreate(path string) (ed25519.PublicKey, ed25519.PrivateKey, error) {
	if b, err := os.ReadFile(path); err == nil {
		sk, err := hex.DecodeString(string(b))
		if err == nil && len(sk) == ed25519.PrivateKeySize {
			priv := ed25519.PrivateKey(sk)
			return priv.Public().(ed25519.PublicKey), priv, nil
		}
	}

	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, nil, fmt.Errorf("generate key: %w", err)
	}

	if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return nil, nil, fmt.Errorf("create key directory: %w", err)
	}
	if err := os.WriteFile(path, []byte(hex.EncodeToString(priv)), 0600); err != nil {
		return nil, nil, fmt.Errorf("save key: %w", err)
	}

	return pub, priv, nil
}

// DefaultPath returns ~/.ryvion/node-key.
func DefaultPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "/root"
	}
	return filepath.Join(home, ".ryvion", "node-key")
}
```

**Step 2:** Verify: `go build ./...`

---

### Task 3: `internal/hw` — Hardware Detection

**Files:**
- Create: `internal/hw/hw.go`

**Step 1: Write hw.go**

Ported from the current `metrics/system.go` but cleaner:
- `Caps` struct with `DetectCaps(deviceType) Caps`
- `Metrics` struct with `SampleMetrics() Metrics`
- `GPUUtil(ctx) float64` for one-shot GPU snapshot
- All return zero on failure (no random numbers)
- No slog import — this is a pure detection package, caller logs

Key types:
```go
type Caps struct {
	GPUModel      string
	CPUCores      uint32
	RAMBytes      uint64
	VRAMBytes     uint64
	Sensors       string
	BandwidthMbps uint64
}

type Metrics struct {
	CPUUtil    float64
	MemUtil    float64
	GPUUtil    float64
	PowerWatts float64
}
```

Port detection functions from current code:
- `detectGPU()` → nvidia-smi query
- `detectRAM()` → /proc/meminfo, sysctl, wmic
- `sampleCPU()` → /proc/stat two-sample
- `sampleMem()` → /proc/meminfo
- `sampleGPU()` → nvidia-smi utilization query
- `samplePower()` → nvidia-smi power.draw query

**Step 2:** Verify: `go build ./...`

---

### Task 4: `internal/hub` — Hub API Client

**Files:**
- Create: `internal/hub/client.go`

This is the largest and most important file. It replaces the 595-line `agent.go` monolith.

**Step 1: Write types and constructor**

```go
package hub

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

type Client struct {
	baseURL   string
	pub       ed25519.PublicKey
	priv      ed25519.PrivateKey
	http      *http.Client
	bindToken string // optional X-Bind-Token header
	wallet    string // optional X-Wallet header
}

func New(baseURL string, pub ed25519.PublicKey, priv ed25519.PrivateKey, opts ...Option) *Client {
	c := &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		pub:     pub,
		priv:    priv,
		http:    &http.Client{Timeout: 30 * time.Second},
	}
	for _, o := range opts {
		o(c)
	}
	return c
}

type Option func(*Client)

func WithBindToken(t string) Option { return func(c *Client) { c.bindToken = t } }
func WithWallet(w string) Option    { return func(c *Client) { c.wallet = w } }
```

**Step 2: Write signing helpers**

The hub uses `"RYV1|"` prefix everywhere. Single sign method:

```go
func (c *Client) pubHex() string { return strings.ToLower(hex.EncodeToString(c.pub)) }

// sign computes SHA256("RYV1|" + parts joined by "|") and signs with ed25519.
func (c *Client) sign(parts ...string) []byte {
	msg := "RYV1|" + strings.Join(parts, "|")
	sum := sha256.Sum256([]byte(msg))
	return ed25519.Sign(c.priv, sum[:])
}

// signRaw signs the raw SHA256 hash (for messages built differently).
func (c *Client) signRaw(data []byte) []byte {
	sum := sha256.Sum256(data)
	return ed25519.Sign(c.priv, sum[:])
}

func i64str(v int64) string   { return strconv.FormatInt(v, 10) }
func u64str(v uint64) string  { return strconv.FormatUint(v, 10) }
func u32str(v uint32) string  { return strconv.FormatUint(uint64(v), 10) }
func f64json(v float64) string { b, _ := json.Marshal(v); return string(b) }
```

**Step 3: Write post helper**

```go
func (c *Client) post(ctx context.Context, path string, body any, out any) error {
	b, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}
	url := c.baseURL + path
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(b))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "ryvion-node/2.0")
	if c.bindToken != "" {
		req.Header.Set("X-Bind-Token", c.bindToken)
	}
	if c.wallet != "" {
		req.Header.Set("X-Wallet", c.wallet)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		rb, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("%s: %d %s", path, resp.StatusCode, string(rb))
	}
	if out != nil {
		return json.NewDecoder(resp.Body).Decode(out)
	}
	return nil
}
```

**Step 4: Write Register**

Hub expects: `public_key_hex`, `device_type`, `gpu_model`, `cpu_cores`, `ram_bytes`, `vram_bytes`, `sensors`, `bandwidth_mbps`, `geohash_bucket`, `attestation_method`, `referral_code`, `signature`.

Signature: `SHA256("RYV1|register|<pubhex>|<device_type>|<gpu_model>|<cpu_cores>|<ram>|<vram>|<sensors>|<bandwidth>|<geohash>|<attest>")`.

```go
type RegisterRequest struct {
	Caps       Caps
	DeviceType string
	Referral   string
}

type Caps struct {
	GPUModel      string
	CPUCores      uint32
	RAMBytes      uint64
	VRAMBytes     uint64
	Sensors       string
	BandwidthMbps uint64
}

func (c *Client) Register(ctx context.Context, req RegisterRequest) error {
	sig := c.sign("register",
		c.pubHex(),
		req.DeviceType,
		req.Caps.GPUModel,
		u32str(req.Caps.CPUCores),
		u64str(req.Caps.RAMBytes),
		u64str(req.Caps.VRAMBytes),
		req.Caps.Sensors,
		u64str(req.Caps.BandwidthMbps),
		"0", // geohash
		"0", // attestation
	)
	body := map[string]any{
		"public_key_hex":     c.pubHex(),
		"device_type":        req.DeviceType,
		"gpu_model":          req.Caps.GPUModel,
		"cpu_cores":          req.Caps.CPUCores,
		"ram_bytes":          req.Caps.RAMBytes,
		"vram_bytes":         req.Caps.VRAMBytes,
		"sensors":            req.Caps.Sensors,
		"bandwidth_mbps":     req.Caps.BandwidthMbps,
		"geohash_bucket":     uint64(0),
		"attestation_method": uint32(0),
		"referral_code":      req.Referral,
		"signature":          sig,
	}
	return c.post(ctx, "/api/v1/node/register", body, nil)
}
```

**Step 5: Write Heartbeat**

```go
func (c *Client) Heartbeat(ctx context.Context, m Metrics) error {
	ts := time.Now().UnixMilli()
	sig := c.sign("heartbeat", c.pubHex(), i64str(ts),
		f64json(m.CPUUtil), f64json(m.MemUtil), f64json(m.GPUUtil), f64json(m.PowerWatts))
	body := map[string]any{
		"public_key_hex": c.pubHex(),
		"timestamp_ms":   ts,
		"cpu_util":       m.CPUUtil,
		"mem_util":       m.MemUtil,
		"gpu_util":       m.GPUUtil,
		"power_watts":    m.PowerWatts,
		"signature":      sig,
	}
	return c.post(ctx, "/api/v1/node/heartbeat", body, nil)
}

type Metrics struct {
	CPUUtil    float64
	MemUtil    float64
	GPUUtil    float64
	PowerWatts float64
}
```

**Step 6: Write FetchWork**

```go
type WorkAssignment struct {
	JobID      string `json:"job_id"`
	Kind       string `json:"kind"`
	PayloadURL string `json:"payload_url"`
	Units      uint32 `json:"units"`
	Image      string `json:"image"`
	SpecJSON   string `json:"spec_json"`
}

func (c *Client) FetchWork(ctx context.Context) (*WorkAssignment, error) {
	ts := time.Now().UnixMilli()
	sig := c.sign("work", c.pubHex(), i64str(ts))

	url := fmt.Sprintf("%s/api/v1/node/work?pubkey=%s", c.baseURL, c.pubHex())
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("X-Node-Timestamp", i64str(ts))
	req.Header.Set("X-Node-Signature", hex.EncodeToString(sig))

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == 204 {
		return nil, nil
	}
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("work: %d %s", resp.StatusCode, string(b))
	}

	var wa struct {
		HasWork *bool  `json:"has_work"`
		WorkAssignment
	}
	if err := json.NewDecoder(resp.Body).Decode(&wa); err != nil {
		return nil, err
	}
	if wa.HasWork != nil && !*wa.HasWork {
		return nil, nil
	}
	if strings.TrimSpace(wa.JobID) == "" {
		return nil, nil
	}
	return &wa.WorkAssignment, nil
}
```

**Step 7: Write SubmitReceipt**

```go
type Receipt struct {
	JobID         string
	ResultHashHex string
	Units         uint64
	Metadata      map[string]any
}

func (c *Client) SubmitReceipt(ctx context.Context, r Receipt) error {
	sig := c.sign("receipt", r.JobID, c.pubHex(), r.ResultHashHex, u64str(r.Units))
	body := map[string]any{
		"job_id":          r.JobID,
		"public_key_hex":  c.pubHex(),
		"result_hash_hex": r.ResultHashHex,
		"metering_units":  r.Units,
		"signature":       sig,
		"metadata":        r.Metadata,
	}
	return c.post(ctx, "/api/v1/node/receipt", body, nil)
}
```

**Step 8: Write SolveChallenge and SendHealthReport**

```go
func (c *Client) SolveChallenge(ctx context.Context) error {
	var resp struct{ Nonce string `json:"nonce"` }
	if err := c.post(ctx, "/api/v1/node/challenge/request",
		map[string]any{"public_key_hex": c.pubHex()}, &resp); err != nil {
		return err
	}
	if resp.Nonce == "" {
		return fmt.Errorf("no nonce returned")
	}
	sig := c.sign("challenge", resp.Nonce)
	return c.post(ctx, "/api/v1/node/challenge/solve",
		map[string]any{"public_key_hex": c.pubHex(), "nonce": resp.Nonce, "signature": sig}, nil)
}

type HealthReport struct {
	GPUReady  bool
	DockerGPU bool
	Message   string
}

func (c *Client) SendHealthReport(ctx context.Context, h HealthReport) error {
	ts := time.Now().UnixMilli()
	gpuStr, dockerStr := "0", "0"
	if h.GPUReady { gpuStr = "1" }
	if h.DockerGPU { dockerStr = "1" }

	payload := []byte("RYV1|health|" + c.pubHex() + "|" + i64str(ts) + "|" + gpuStr + "|" + dockerStr + "|" + h.Message)
	sum := sha256.Sum256(payload)
	sig := ed25519.Sign(c.priv, sum[:])

	body := map[string]any{
		"public_key_hex": c.pubHex(),
		"timestamp_ms":   ts,
		"gpu_ready":      h.GPUReady,
		"docker_gpu":     h.DockerGPU,
		"message":        h.Message,
		"signature":      sig,
	}
	return c.post(ctx, "/api/v1/node/health", body, nil)
}
```

**Step 9: Write PrepareUpload and PresignManifest**

```go
type UploadToken struct {
	PutURL string `json:"put_url"`
	Key    string `json:"key"`
}

func (c *Client) PrepareUpload(ctx context.Context, jobID string, contentType string, size uint64) (*UploadToken, error) {
	msg := "RYV1|upload_prep|" + jobID + "|" + c.pubHex() + "|" + contentType + "|" + u64str(size)
	sum := sha256.Sum256([]byte(msg))
	sig := ed25519.Sign(c.priv, sum[:])

	body := map[string]any{
		"pubkey":       []byte(c.pub),
		"job_id":       jobID,
		"content_type": contentType,
		"size_bytes":   size,
		"signature":    sig,
	}
	var tok UploadToken
	if err := c.post(ctx, "/api/v1/node/upload/prepare", body, &tok); err != nil {
		return nil, err
	}
	if strings.TrimSpace(tok.PutURL) == "" {
		return nil, fmt.Errorf("no put_url returned")
	}
	return &tok, nil
}

func (c *Client) PresignManifest(ctx context.Context, key string) (string, error) {
	var resp struct{ URL string `json:"url"` }
	body := map[string]any{"key": key, "method": "PUT", "expiry_seconds": 900}
	if err := c.post(ctx, "/api/v1/blob/presign", body, &resp); err != nil {
		return "", err
	}
	return resp.URL, nil
}

// BaseURL returns the hub base URL for constructing absolute URLs from relative paths.
func (c *Client) BaseURL() string { return c.baseURL }

// PubKey returns the node's public key.
func (c *Client) PubKey() ed25519.PublicKey { return c.pub }

// PrivKey returns the node's private key (for blob signing).
func (c *Client) PrivKey() ed25519.PrivateKey { return c.priv }
```

**Step 10:** Verify: `go build ./...`

---

### Task 5: `internal/runner` — OCI Execution

**Files:**
- Create: `internal/runner/oci.go`

**Step 1: Write oci.go**

Cleaned up from current code. Key changes: `ryv_work_*` temp prefix, slog logging, proper error handling.

```go
package runner

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

type Result struct {
	Hash       string
	Duration   time.Duration
	ExitCode   int
	Logs       string
	Metrics    map[string]any
	OutputPath string // path to output file, if any
}

// Run executes a container image with the given job spec.
// It creates a temp workdir, writes job.json, runs docker, and reads results.
func Run(ctx context.Context, image string, specJSON []byte, workDir string) (*Result, error) {
	if image == "" {
		return nil, fmt.Errorf("image required")
	}

	dir, err := os.MkdirTemp(workDir, "ryv_*")
	if err != nil {
		return nil, fmt.Errorf("create workdir: %w", err)
	}
	defer os.RemoveAll(dir)

	if err := os.WriteFile(filepath.Join(dir, "job.json"), specJSON, 0644); err != nil {
		return nil, fmt.Errorf("write job.json: %w", err)
	}

	args := []string{"run", "--rm"}
	if hasGPU() {
		args = append(args, "--gpus", "all")
	}
	args = append(args, "-v", dir+":/work", image)

	start := time.Now()
	cmd := exec.CommandContext(ctx, "docker", args...)
	var buf limitBuf
	cmd.Stdout = &buf
	cmd.Stderr = &buf

	runErr := cmd.Run()
	dur := time.Since(start)

	exitCode := 0
	if runErr != nil {
		if ee, ok := runErr.(*exec.ExitError); ok {
			exitCode = ee.ExitCode()
		} else {
			return nil, fmt.Errorf("docker run: %w", runErr)
		}
	}

	hash, metrics := parseOutput(dir, buf.Bytes(), dur)
	outPath := filepath.Join(dir, "output")

	return &Result{
		Hash:       hash,
		Duration:   dur,
		ExitCode:   exitCode,
		Logs:       buf.Tail(2048),
		Metrics:    metrics,
		OutputPath: outPath,
	}, nil
}

func hasGPU() bool {
	_, err := exec.LookPath("nvidia-smi")
	return err == nil
}

func parseOutput(dir string, stdout []byte, dur time.Duration) (string, map[string]any) {
	var hash string
	if b, err := os.ReadFile(filepath.Join(dir, "receipt.json")); err == nil {
		var rec struct{ OutputHash string `json:"output_hash"` }
		if json.Unmarshal(b, &rec) == nil && rec.OutputHash != "" {
			hash = trimAlgo(rec.OutputHash)
		}
	}
	if hash == "" {
		sum := sha256.Sum256(stdout)
		hash = hex.EncodeToString(sum[:])
	}

	metrics := map[string]any{"duration_ms": dur.Milliseconds()}
	if b, err := os.ReadFile(filepath.Join(dir, "metrics.json")); err == nil {
		var m map[string]any
		if json.Unmarshal(b, &m) == nil {
			metrics = m
		}
	}
	return hash, metrics
}

func trimAlgo(s string) string {
	if i := strings.IndexByte(s, ':'); i >= 0 {
		return s[i+1:]
	}
	return s
}

type limitBuf struct{ b []byte }

func (l *limitBuf) Write(p []byte) (int, error) {
	l.b = append(l.b, p...)
	const max = 1 << 20
	if len(l.b) > max {
		l.b = l.b[len(l.b)-max:]
	}
	return len(p), nil
}

func (l *limitBuf) Bytes() []byte { return l.b }

func (l *limitBuf) Tail(n int) string {
	if len(l.b) <= n {
		return string(l.b)
	}
	return string(l.b[len(l.b)-n:])
}
```

**Step 2:** Verify: `go build ./...`

---

### Task 6: `internal/blob` — Artifact Upload

**Files:**
- Create: `internal/blob/upload.go`

**Step 1: Write upload.go**

Single clean implementation replacing the 100+ line inline code in agent.go:

```go
package blob

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

// Uploader handles artifact uploads to the hub's blob storage.
type Uploader struct {
	baseURL string
	pub     ed25519.PublicKey
	priv    ed25519.PrivateKey
	http    *http.Client
}

// New creates an Uploader from hub connection details.
func New(baseURL string, pub ed25519.PublicKey, priv ed25519.PrivateKey) *Uploader {
	return &Uploader{baseURL: baseURL, pub: pub, priv: priv, http: &http.Client{Timeout: 5 * time.Minute}}
}

// Result contains the upload outcome.
type Result struct {
	URL  string
	Key  string
	Hash string
}

// Upload reads the file at path, uploads it via presigned URL, and uploads a signed manifest.
// The prepareUpload and presignManifest functions are injected to avoid circular hub dependency.
func Upload(ctx context.Context, u *Uploader, jobID, filePath string,
	prepareUpload func(ctx context.Context, jobID, contentType string, size uint64) (putURL, key string, err error),
	presignManifest func(ctx context.Context, key string) (string, error),
) (*Result, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("read artifact: %w", err)
	}

	h := sha256.Sum256(data)
	fileHash := hex.EncodeToString(h[:])
	size := uint64(len(data))

	putURL, key, err := prepareUpload(ctx, jobID, "application/octet-stream", size)
	if err != nil {
		return nil, fmt.Errorf("prepare upload: %w", err)
	}

	absURL := absolutize(u.baseURL, putURL)

	req, err := http.NewRequestWithContext(ctx, http.MethodPut, absURL, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/octet-stream")

	// If relative URL (hub-managed blob), add node auth headers
	if strings.HasPrefix(putURL, "/") {
		ts := time.Now().UnixMilli()
		msg := fmt.Sprintf("RYV1|blob|%s|%s|%d|%d", jobID, hex.EncodeToString(u.pub), size, ts)
		sum := sha256.Sum256([]byte(msg))
		sig := ed25519.Sign(u.priv, sum[:])
		req.Header.Set("X-Node-Pubkey", strings.ToLower(hex.EncodeToString(u.pub)))
		req.Header.Set("X-Node-Timestamp", fmt.Sprintf("%d", ts))
		req.Header.Set("X-Node-Signature", hex.EncodeToString(sig))
	}

	resp, err := u.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("upload: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("upload: %d %s", resp.StatusCode, string(b))
	}

	// For hub-managed blobs, try to read the URL from the response
	resultURL := putURL
	if strings.HasPrefix(putURL, "/") {
		var pr struct{ URL string `json:"url"` }
		if b, err := io.ReadAll(resp.Body); err == nil && len(b) > 0 {
			if json.Unmarshal(b, &pr) == nil && pr.URL != "" {
				resultURL = absolutize(u.baseURL, pr.URL)
			}
		}
		if resultURL == putURL {
			resultURL = u.baseURL + "/api/v1/blob/" + jobID
		}
	}

	// Upload signed manifest
	if key != "" {
		uploadManifest(ctx, u, jobID, key, fileHash, size, presignManifest)
	}

	return &Result{URL: resultURL, Key: key, Hash: fileHash}, nil
}

func uploadManifest(ctx context.Context, u *Uploader, jobID, key, fileHash string, size uint64,
	presign func(ctx context.Context, key string) (string, error)) {

	manifest := map[string]any{
		"job_id":       jobID,
		"object_key":   key,
		"sha256":       fileHash,
		"size_bytes":   size,
		"node_pubkey":  strings.ToLower(hex.EncodeToString(u.pub)),
		"submitted_at": time.Now().UTC().Format(time.RFC3339),
	}
	mb, _ := json.Marshal(manifest)
	sum := sha256.Sum256(mb)
	sig := ed25519.Sign(u.priv, sum[:])
	manifest["signature_b64"] = base64.StdEncoding.EncodeToString(sig)
	mb, _ = json.Marshal(manifest)

	manKey := key + ".manifest.json"
	manURL, err := presign(ctx, manKey)
	if err != nil || manURL == "" {
		return
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPut, absolutize(u.baseURL, manURL), bytes.NewReader(mb))
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := u.http.Do(req)
	if err != nil {
		return
	}
	resp.Body.Close()
}

func absolutize(base, ref string) string {
	if strings.HasPrefix(ref, "http://") || strings.HasPrefix(ref, "https://") {
		return ref
	}
	if strings.HasPrefix(ref, "/") {
		u, err := url.Parse(base)
		if err != nil {
			return ref
		}
		u.Path = ref
		return u.String()
	}
	return ref
}
```

**Step 2:** Verify: `go build ./...`

---

### Task 7: `cmd/ryvion-node/main.go` — Entry Point

**Files:**
- Write: `cmd/ryvion-node/main.go`

**Step 1: Write main.go**

The entire orchestration loop in ~100 lines:

```go
package main

import (
	"context"
	"flag"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/Ryvion/node-agent/internal/blob"
	"github.com/Ryvion/node-agent/internal/hub"
	"github.com/Ryvion/node-agent/internal/hw"
	"github.com/Ryvion/node-agent/internal/nodekey"
	"github.com/Ryvion/node-agent/internal/runner"
)

func main() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo})))

	hubURL := flag.String("hub", "https://ryvion-hub.onrender.com", "hub URL")
	devType := flag.String("type", "", "device type (auto-detect if empty)")
	referral := flag.String("referral", "", "referral code")
	flag.Parse()

	if env := os.Getenv("RYV_HUB_URL"); env != "" && *hubURL == "https://ryvion-hub.onrender.com" {
		*hubURL = env
	}

	keyPath := os.Getenv("RYV_KEY_PATH")
	if keyPath == "" {
		keyPath = nodekey.DefaultPath()
	}

	pub, priv, err := nodekey.LoadOrCreate(keyPath)
	if err != nil {
		slog.Error("failed to load key", "error", err)
		os.Exit(1)
	}

	if *devType == "" {
		*devType = hw.AutoDetect()
	}

	var opts []hub.Option
	if t := strings.TrimSpace(os.Getenv("RYV_BIND_TOKEN")); t != "" {
		opts = append(opts, hub.WithBindToken(t))
	}
	if w := strings.TrimSpace(os.Getenv("RYV_WALLET")); w != "" {
		opts = append(opts, hub.WithWallet(w))
	}

	client := hub.New(*hubURL, pub, priv, opts...)
	slog.Info("node starting", "pubkey", client.PubHex(), "hub", *hubURL, "type", *devType)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	caps := hw.DetectCaps(*devType)
	if err := client.Register(ctx, hub.RegisterRequest{Caps: hub.Caps(caps), DeviceType: *devType, Referral: *referral}); err != nil {
		slog.Error("register failed", "error", err)
		os.Exit(1)
	}
	slog.Info("registered")

	if err := client.SolveChallenge(ctx); err != nil {
		slog.Warn("challenge failed", "error", err)
	}

	workDir := strings.TrimSpace(os.Getenv("RYV_WORK_DIR"))
	uploader := blob.New(*hubURL, pub, priv)

	tick := time.NewTicker(10 * time.Second)
	defer tick.Stop()

	for {
		cycle(ctx, client, uploader, workDir)
		select {
		case <-ctx.Done():
			slog.Info("shutting down")
			return
		case <-tick.C:
		}
	}
}

func cycle(ctx context.Context, c *hub.Client, up *blob.Uploader, workDir string) {
	metrics := hw.SampleMetrics()
	if err := c.Heartbeat(ctx, hub.Metrics(metrics)); err != nil {
		slog.Warn("heartbeat failed", "error", err)
	}

	work, err := c.FetchWork(ctx)
	if err != nil {
		slog.Warn("fetch work failed", "error", err)
		return
	}
	if work == nil {
		return
	}

	slog.Info("job received", "job_id", work.JobID, "image", work.Image)

	if strings.TrimSpace(work.Image) == "" {
		slog.Warn("job has no image, skipping", "job_id", work.JobID)
		return
	}

	result, err := runner.Run(ctx, work.Image, []byte(work.SpecJSON), workDir)
	if err != nil {
		slog.Error("execution failed", "error", err, "job_id", work.JobID)
		return
	}

	meta := map[string]any{
		"executor":    "oci",
		"duration_ms": result.Duration.Milliseconds(),
		"exit_code":   result.ExitCode,
		"stderr_tail": result.Logs,
	}

	// Upload artifact if output exists
	if fi, err := os.Stat(result.OutputPath); err == nil && !fi.IsDir() {
		br, err := blob.Upload(ctx, up, work.JobID, result.OutputPath,
			func(ctx context.Context, jobID, ct string, size uint64) (string, string, error) {
				tok, err := c.PrepareUpload(ctx, jobID, ct, size)
				if err != nil {
					return "", "", err
				}
				return tok.PutURL, tok.Key, nil
			},
			func(ctx context.Context, key string) (string, error) {
				return c.PresignManifest(ctx, key)
			},
		)
		if err != nil {
			slog.Error("artifact upload failed", "error", err, "job_id", work.JobID)
		} else {
			meta["blob_url"] = br.URL
			meta["object_key"] = br.Key
			meta["artifact_sha256"] = br.Hash
			if br.Hash != "" {
				result.Hash = br.Hash
			}
			slog.Info("artifact uploaded", "url", br.URL)
		}
	}

	units := uint64(work.Units)
	if units == 0 {
		units = 1
	}

	receipt := hub.Receipt{
		JobID:         work.JobID,
		ResultHashHex: result.Hash,
		Units:         units,
		Metadata:      meta,
	}
	if err := c.SubmitReceipt(ctx, receipt); err != nil {
		slog.Error("receipt failed", "error", err, "job_id", work.JobID)
		return
	}
	slog.Info("receipt submitted", "job_id", work.JobID, "hash", result.Hash)
}
```

**Step 2:** Verify: `go build ./...`

---

### Task 8: Update Dockerfile and start.sh

**Files:**
- Modify: `Dockerfile`
- Modify: `start.sh`
- Modify: `deploy/docker-compose.yml`

**Step 1:** Update Dockerfile to build `cmd/ryvion-node` instead of `cmd/node-agent`. Remove Python deps (not needed for OCI-only execution). Remove the docker dependency from go.mod.

**Step 2:** Update start.sh to exec `ryvion-node` instead of `node-agent`.

**Step 3:** Update docker-compose.yml command.

**Step 4:** Run `go mod tidy` to remove unused deps.

**Step 5:** Verify: `go build ./...` and `go vet ./...`

---

### Task 9: Final Verification

**Steps:**

1. `go build ./...` — compiles cleanly
2. `go vet ./...` — no issues
3. Verify zero `AKT1|` references: `grep -rn 'AKT1' .` → 0 results
4. Verify zero `AK_` references: `grep -rn 'AK_' .` → 0 results
5. Verify zero `ioutil` references: `grep -rn 'ioutil' .` → 0 results
6. Verify signing prefix is `RYV1|` everywhere: `grep -rn 'RYV1|' internal/` → only in hub/client.go and blob/upload.go
7. Count total LOC: `find . -name '*.go' -not -path './docs/*' | xargs wc -l`
8. Verify the module path in go.mod matches import paths
