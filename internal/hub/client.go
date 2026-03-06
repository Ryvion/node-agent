package hub

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
	"strconv"
	"strings"
	"time"
)

type Client struct {
	baseURL   string
	pub       ed25519.PublicKey
	priv      ed25519.PrivateKey
	http      *http.Client
	bindToken string
	wallet    string
	adminKey  string
	userAgent string
}

type Option func(*Client)

func WithHTTPClient(h *http.Client) Option {
	return func(c *Client) {
		if h != nil {
			c.http = h
		}
	}
}

func WithBindToken(token string) Option {
	return func(c *Client) { c.bindToken = strings.TrimSpace(token) }
}

func WithWallet(wallet string) Option {
	return func(c *Client) { c.wallet = strings.TrimSpace(wallet) }
}

func WithAdminKey(adminKey string) Option {
	return func(c *Client) { c.adminKey = strings.TrimSpace(adminKey) }
}

func WithUserAgent(ua string) Option {
	return func(c *Client) {
		if strings.TrimSpace(ua) != "" {
			c.userAgent = ua
		}
	}
}

func New(baseURL string, pub ed25519.PublicKey, priv ed25519.PrivateKey, opts ...Option) *Client {
	c := &Client{
		baseURL:   strings.TrimRight(strings.TrimSpace(baseURL), "/"),
		pub:       pub,
		priv:      priv,
		http:      &http.Client{Timeout: 30 * time.Second},
		userAgent: "ryvion-node/1.0",
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

func (c *Client) Register(ctx context.Context, caps Capabilities, deviceType, referral, declaredCountry string) error {
	pubHex := c.pubHex()
	body := registerRequest{
		PublicKeyHex:      pubHex,
		DeviceType:        strings.TrimSpace(deviceType),
		DeclaredCountry:   strings.ToUpper(strings.TrimSpace(declaredCountry)),
		GPUModel:          caps.GPUModel,
		CPUCores:          caps.CPUCores,
		RAMBytes:          caps.RAMBytes,
		VRAMBytes:         caps.VRAMBytes,
		Sensors:           caps.Sensors,
		BandwidthMbps:     caps.BandwidthMbps,
		GeohashBucket:     caps.GeohashBucket,
		AttestationMethod: caps.AttestationMethod,
		ReferralCode:      strings.TrimSpace(referral),
		TEESupported:      caps.TEESupported,
		TEEType:           caps.TEEType,
	}
	if body.DeviceType == "" {
		body.DeviceType = "cpu"
	}
	signParts := []string{
		"register",
		pubHex,
		body.DeviceType,
	}
	if body.DeclaredCountry != "" {
		signParts = append(signParts, body.DeclaredCountry)
	}
	signParts = append(signParts,
		body.GPUModel,
		strconv.FormatUint(uint64(body.CPUCores), 10),
		strconv.FormatUint(body.RAMBytes, 10),
		strconv.FormatUint(body.VRAMBytes, 10),
		body.Sensors,
		strconv.FormatUint(body.BandwidthMbps, 10),
		strconv.FormatUint(body.GeohashBucket, 10),
		strconv.FormatUint(uint64(body.AttestationMethod), 10),
	)
	body.Signature = c.sign(signParts...)
	return c.post(ctx, "/api/v1/node/register", body, nil)
}

func (c *Client) Heartbeat(ctx context.Context, metrics Metrics) (string, error) {
	pubHex := c.pubHex()
	ts := metrics.TimestampMs
	if ts == 0 {
		ts = time.Now().UnixMilli()
	}
	body := heartbeatRequest{
		PublicKeyHex: pubHex,
		TimestampMs:  ts,
		CPUUtil:      metrics.CPUUtil,
		MemUtil:      metrics.MemUtil,
		GPUUtil:      metrics.GPUUtil,
		PowerWatts:   metrics.PowerWatts,
		GPUThrottled: metrics.GPUThrottled,
	}
	body.Signature = c.sign(
		"heartbeat",
		pubHex,
		strconv.FormatInt(ts, 10),
		formatFloatJSON(body.CPUUtil),
		formatFloatJSON(body.MemUtil),
		formatFloatJSON(body.GPUUtil),
		formatFloatJSON(body.PowerWatts),
	)
	var resp struct {
		LatestVersion string `json:"latest_version"`
	}
	err := c.post(ctx, "/api/v1/node/heartbeat", body, &resp)
	return resp.LatestVersion, err
}

func (c *Client) FetchWork(ctx context.Context) (*WorkAssignment, error) {
	ts := time.Now().UnixMilli()
	pubHex := c.pubHex()

	u, err := url.Parse(c.absoluteURL("/api/v1/node/work"))
	if err != nil {
		return nil, err
	}
	q := u.Query()
	q.Set("pubkey", pubHex)
	q.Set("long_poll", "1")
	u.RawQuery = q.Encode()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", c.userAgent)
	req.Header.Set("X-Node-Timestamp", strconv.FormatInt(ts, 10))
	req.Header.Set("X-Node-Signature", hex.EncodeToString(c.sign("work", pubHex, strconv.FormatInt(ts, 10))))

	// Use longer timeout for long-polling (hub holds up to 25s)
	longPollClient := &http.Client{Timeout: 35 * time.Second}
	resp, err := longPollClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNoContent {
		return nil, nil
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))
		return nil, fmt.Errorf("GET %s: %d %s", u.String(), resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var out workResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if out.HasWork != nil && !*out.HasWork {
		return nil, nil
	}
	if out.HasWork == nil && strings.TrimSpace(out.JobID) == "" {
		return nil, nil
	}
	if strings.TrimSpace(out.JobID) == "" {
		return nil, fmt.Errorf("work assignment missing job_id")
	}
	return &WorkAssignment{
		JobID:        out.JobID,
		JobPubkey:    out.JobPubkey,
		Kind:         out.Kind,
		PayloadURL:   out.PayloadURL,
		PricePerUnit: out.PricePerUnit,
		Units:        out.Units,
		Image:        out.Image,
		SpecJSON:     out.SpecJSON,
	}, nil
}

func (c *Client) SubmitReceipt(ctx context.Context, receipt Receipt) error {
	jobID := strings.TrimSpace(receipt.JobID)
	if jobID == "" {
		return fmt.Errorf("job_id required")
	}
	hashHex := strings.TrimSpace(receipt.ResultHashHex)
	if hashHex == "" {
		return fmt.Errorf("result_hash_hex required")
	}
	units := receipt.MeteringUnits
	if units == 0 {
		units = 1
	}
	pubHex := c.pubHex()
	body := receiptRequest{
		JobID:         jobID,
		PublicKeyHex:  pubHex,
		ResultHashHex: hashHex,
		MeteringUnits: units,
		Metadata:      receipt.Metadata,
	}
	body.Signature = c.sign("receipt", jobID, pubHex, hashHex, strconv.FormatUint(units, 10))
	return c.post(ctx, "/api/v1/node/receipt", body, nil)
}

func (c *Client) SavePayout(ctx context.Context, stripeConnectID, currency string) error {
	stripeConnectID = strings.TrimSpace(stripeConnectID)
	if stripeConnectID == "" {
		return fmt.Errorf("stripe_connect_id required")
	}
	if currency = strings.TrimSpace(currency); currency == "" {
		currency = "CAD"
	}
	ts := time.Now().UnixMilli()
	pubHex := c.pubHex()
	body := payoutRequest{
		PublicKeyHex:    pubHex,
		StripeConnectID: stripeConnectID,
		Currency:        strings.ToUpper(currency),
		TimestampMs:     ts,
	}
	body.Signature = c.sign("payout", pubHex, stripeConnectID, strconv.FormatInt(ts, 10))
	return c.post(ctx, "/api/v1/node/payout/save", body, nil)
}

func (c *Client) SolveChallenge(ctx context.Context) error {
	var reqResp challengeResponse
	if err := c.post(ctx, "/api/v1/node/challenge/request", challengeRequest{PublicKeyHex: c.pubHex()}, &reqResp); err != nil {
		return err
	}
	if strings.TrimSpace(reqResp.Nonce) == "" {
		return fmt.Errorf("challenge nonce missing")
	}
	body := challengeSolveRequest{
		PublicKeyHex: c.pubHex(),
		Nonce:        reqResp.Nonce,
		Signature:    c.sign("challenge", reqResp.Nonce),
	}
	return c.post(ctx, "/api/v1/node/challenge/solve", body, nil)
}

func (c *Client) SendHealthReport(ctx context.Context, report HealthReport) error {
	ts := report.TimestampMs
	if ts == 0 {
		ts = time.Now().UnixMilli()
	}
	message := strings.TrimSpace(report.Message)
	pubHex := c.pubHex()
	body := healthRequest{
		PublicKeyHex: pubHex,
		TimestampMs:  ts,
		GPUReady:     report.GPUReady,
		DockerGPU:    report.DockerGPU,
		Message:      message,
	}
	body.Signature = c.sign(
		"health",
		pubHex,
		strconv.FormatInt(ts, 10),
		boolAsInt(report.GPUReady),
		boolAsInt(report.DockerGPU),
		message,
	)
	return c.post(ctx, "/api/v1/node/health", body, nil)
}

func (c *Client) PrepareUpload(ctx context.Context, jobID string, size uint64) (*UploadToken, error) {
	jobID = strings.TrimSpace(jobID)
	if jobID == "" {
		return nil, fmt.Errorf("job_id required")
	}
	body := uploadPrepareRequest{
		Pubkey:      []byte(c.pub),
		JobID:       jobID,
		ContentType: "application/octet-stream",
		SizeBytes:   size,
	}
	body.Signature = c.sign(
		"upload_prep",
		jobID,
		c.pubHex(),
		body.ContentType,
		strconv.FormatUint(size, 10),
	)

	var out UploadToken
	if err := c.post(ctx, "/api/v1/node/upload/prepare", body, &out); err != nil {
		return nil, err
	}
	if strings.TrimSpace(out.PutURL) == "" {
		return nil, fmt.Errorf("upload prepare response missing put_url")
	}
	return &out, nil
}

func (c *Client) PresignManifest(ctx context.Context, key string) (string, error) {
	key = strings.TrimSpace(key)
	if key == "" {
		return "", fmt.Errorf("key required")
	}
	body := blobPresignRequest{Key: key, Method: http.MethodPut, ExpirySeconds: 900}
	var out blobPresignResponse
	headers := map[string]string{}
	if c.adminKey != "" {
		headers["X-Admin-Key"] = c.adminKey
	}
	if err := c.postWithHeaders(ctx, "/api/v1/blob/presign", body, &out, headers); err != nil {
		return "", err
	}
	if strings.TrimSpace(out.URL) == "" {
		return "", fmt.Errorf("presign response missing url")
	}
	return out.URL, nil
}

func (c *Client) NodeAuthToken(tsMs int64) string {
	if tsMs == 0 {
		tsMs = time.Now().UnixMilli()
	}
	tsStr := strconv.FormatInt(tsMs, 10)
	sig := c.sign("node_auth", c.pubHex(), tsStr)
	return c.pubHex() + ":" + tsStr + ":" + base64.StdEncoding.EncodeToString(sig)
}

func (c *Client) PublicKeyHex() string {
	return c.pubHex()
}

// RedeemClaimCode sends a claim code to the hub to link this node to a buyer account.
func (c *Client) RedeemClaimCode(ctx context.Context, code string) error {
	body := map[string]string{"code": code}
	headers := map[string]string{"X-Node-Token": c.NodeAuthToken(0)}
	return c.postWithHeaders(ctx, "/api/v1/node/claim", body, nil, headers)
}

func (c *Client) AbsoluteURL(u string) string {
	return c.absoluteURL(u)
}

func (c *Client) BlobUploadHeaders(jobID string, size int64, tsMs int64) map[string]string {
	tsStr := strconv.FormatInt(tsMs, 10)
	sig := c.sign("blob", jobID, c.pubHex(), strconv.FormatInt(size, 10), tsStr)
	return map[string]string{
		"X-Node-Pubkey":    c.pubHex(),
		"X-Node-Timestamp": tsStr,
		"X-Node-Signature": hex.EncodeToString(sig),
	}
}

func (c *Client) StreamInference(ctx context.Context, jobID string, body io.Reader) error {
	jobID = strings.TrimSpace(jobID)
	if jobID == "" {
		return fmt.Errorf("job_id required")
	}
	ts := time.Now().UnixMilli()
	pubHex := c.pubHex()
	tsStr := strconv.FormatInt(ts, 10)
	sig := c.sign("stream", jobID, pubHex, tsStr)

	path := "/api/v1/node/inference/stream/" + jobID
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.absoluteURL(path), body)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "text/event-stream")
	req.Header.Set("Transfer-Encoding", "chunked")
	req.Header.Set("User-Agent", c.userAgent)
	req.Header.Set("X-Node-Pubkey", pubHex)
	req.Header.Set("X-Node-Timestamp", tsStr)
	req.Header.Set("X-Node-Signature", hex.EncodeToString(sig))

	// Send stream with no strict timeout so it can handle huge long-running models
	streamClient := &http.Client{}
	resp, err := streamClient.Do(req)
	if err != nil {
		return fmt.Errorf("stream inference: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		rb, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("stream inference %s: %d %s", path, resp.StatusCode, strings.TrimSpace(string(rb)))
	}
	return nil
}

func (c *Client) SignDigest(digest []byte) []byte {
	if len(digest) == 0 {
		return nil
	}
	return ed25519.Sign(c.priv, digest)
}

func (c *Client) Do(req *http.Request) (*http.Response, error) {
	return c.http.Do(req)
}

func (c *Client) sign(parts ...string) []byte {
	payload := "RYV1|" + strings.Join(parts, "|")
	sum := sha256.Sum256([]byte(payload))
	return ed25519.Sign(c.priv, sum[:])
}

func (c *Client) post(ctx context.Context, path string, body any, out any) error {
	return c.postWithHeaders(ctx, path, body, out, nil)
}

func (c *Client) postWithHeaders(ctx context.Context, path string, body any, out any, headers map[string]string) error {
	b, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.absoluteURL(path), bytes.NewReader(b))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", c.userAgent)
	if c.bindToken != "" {
		req.Header.Set("X-Bind-Token", c.bindToken)
	}
	if c.wallet != "" {
		req.Header.Set("X-Wallet", c.wallet)
	}
	for k, v := range headers {
		if strings.TrimSpace(v) != "" {
			req.Header.Set(k, v)
		}
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		rb, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))
		return fmt.Errorf("POST %s: %d %s", req.URL.String(), resp.StatusCode, strings.TrimSpace(string(rb)))
	}
	if out == nil || resp.StatusCode == http.StatusNoContent {
		return nil
	}
	rb, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if len(bytes.TrimSpace(rb)) == 0 {
		return nil
	}
	return json.Unmarshal(rb, out)
}

func (c *Client) pubHex() string {
	return strings.ToLower(hex.EncodeToString(c.pub))
}

func (c *Client) absoluteURL(maybeRelative string) string {
	if strings.HasPrefix(maybeRelative, "http://") || strings.HasPrefix(maybeRelative, "https://") {
		return maybeRelative
	}
	if strings.HasPrefix(maybeRelative, "/") {
		return c.baseURL + maybeRelative
	}
	return c.baseURL + "/" + maybeRelative
}

func formatFloatJSON(v float64) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func boolAsInt(v bool) string {
	if v {
		return "1"
	}
	return "0"
}

type Capabilities struct {
	GPUModel          string
	CPUCores          uint32
	RAMBytes          uint64
	VRAMBytes         uint64
	Sensors           string
	BandwidthMbps     uint64
	GeohashBucket     uint64
	AttestationMethod uint32
	TEESupported      bool
	TEEType           string
}

type Metrics struct {
	TimestampMs  int64
	CPUUtil      float64
	MemUtil      float64
	GPUUtil      float64
	PowerWatts   float64
	GPUThrottled bool // node is self-throttling due to operator GPU usage
}

type WorkAssignment struct {
	JobID        string
	JobPubkey    string
	Kind         string
	PayloadURL   string
	PricePerUnit uint64
	Units        uint32
	Image        string
	SpecJSON     string
}

type Receipt struct {
	JobID         string
	ResultHashHex string
	MeteringUnits uint64
	Metadata      map[string]any
}

type HealthReport struct {
	TimestampMs int64
	GPUReady    bool
	DockerGPU   bool
	Message     string
}

type UploadToken struct {
	OK        bool   `json:"ok"`
	Provider  string `json:"provider"`
	PutURL    string `json:"put_url"`
	ExpiresAt string `json:"expires_at"`
	Key       string `json:"key"`
}

type registerRequest struct {
	PublicKeyHex      string `json:"public_key_hex"`
	DeviceType        string `json:"device_type"`
	DeclaredCountry   string `json:"declared_country,omitempty"`
	GPUModel          string `json:"gpu_model"`
	CPUCores          uint32 `json:"cpu_cores"`
	RAMBytes          uint64 `json:"ram_bytes"`
	VRAMBytes         uint64 `json:"vram_bytes"`
	Sensors           string `json:"sensors"`
	BandwidthMbps     uint64 `json:"bandwidth_mbps"`
	GeohashBucket     uint64 `json:"geohash_bucket"`
	AttestationMethod uint32 `json:"attestation_method"`
	ReferralCode      string `json:"referral_code,omitempty"`
	TEESupported      bool   `json:"tee_supported"`
	TEEType           string `json:"tee_type"`
	Signature         []byte `json:"signature"`
}

type heartbeatRequest struct {
	PublicKeyHex string  `json:"public_key_hex"`
	TimestampMs  int64   `json:"timestamp_ms"`
	CPUUtil      float64 `json:"cpu_util"`
	MemUtil      float64 `json:"mem_util"`
	GPUUtil      float64 `json:"gpu_util"`
	PowerWatts   float64 `json:"power_watts"`
	GPUThrottled bool    `json:"gpu_throttled"`
	Signature    []byte  `json:"signature"`
}

type workResponse struct {
	HasWork      *bool  `json:"has_work"`
	JobID        string `json:"job_id"`
	JobPubkey    string `json:"job_pubkey"`
	Kind         string `json:"kind"`
	PayloadURL   string `json:"payload_url"`
	PricePerUnit uint64 `json:"price_per_unit"`
	Units        uint32 `json:"units"`
	Image        string `json:"image"`
	SpecJSON     string `json:"spec_json"`
}

type receiptRequest struct {
	JobID         string         `json:"job_id"`
	PublicKeyHex  string         `json:"public_key_hex"`
	ResultHashHex string         `json:"result_hash_hex"`
	MeteringUnits uint64         `json:"metering_units"`
	Signature     []byte         `json:"signature"`
	Metadata      map[string]any `json:"metadata,omitempty"`
}

type payoutRequest struct {
	PublicKeyHex    string `json:"public_key_hex"`
	StripeConnectID string `json:"stripe_connect_id"`
	Currency        string `json:"currency"`
	TimestampMs     int64  `json:"timestamp_ms"`
	Signature       []byte `json:"signature"`
}

type challengeResponse struct {
	Nonce     string `json:"nonce"`
	ExpiresMs int64  `json:"expires_ms"`
}

type challengeRequest struct {
	PublicKeyHex string `json:"public_key_hex"`
}

type challengeSolveRequest struct {
	PublicKeyHex string `json:"public_key_hex"`
	Nonce        string `json:"nonce"`
	Signature    []byte `json:"signature"`
}

type healthRequest struct {
	PublicKeyHex string `json:"public_key_hex"`
	TimestampMs  int64  `json:"timestamp_ms"`
	GPUReady     bool   `json:"gpu_ready"`
	DockerGPU    bool   `json:"docker_gpu"`
	Message      string `json:"message"`
	Signature    []byte `json:"signature"`
}

type uploadPrepareRequest struct {
	Pubkey      []byte `json:"pubkey"`
	JobID       string `json:"job_id"`
	ContentType string `json:"content_type"`
	SizeBytes   uint64 `json:"size_bytes"`
	Signature   []byte `json:"signature"`
}

type blobPresignRequest struct {
	Key           string `json:"key"`
	Method        string `json:"method"`
	ExpirySeconds int    `json:"expiry_seconds"`
}

type blobPresignResponse struct {
	OK        bool   `json:"ok"`
	URL       string `json:"url"`
	ExpiresAt string `json:"expires_at"`
}
