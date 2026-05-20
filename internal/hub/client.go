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

	heartbeat "github.com/Ryvion/ryvion-node/internal/hub/heartbeat"
	"github.com/Ryvion/ryvion-node/internal/hw"
	routedinference "github.com/Ryvion/ryvion-node/internal/inference/routed"
	netprofile "github.com/Ryvion/ryvion-node/internal/network/profile"
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
	grpc      *grpcTransport
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
		grpc:      defaultGRPCTransport(),
	}
	for _, opt := range opts {
		opt(c)
	}
	if c.grpc != nil {
		c.grpc.applyDefaultTarget(c.baseURL)
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
		CPUModel:          caps.CPUModel,
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

func (c *Client) Heartbeat(ctx context.Context, metrics Metrics) (HeartbeatResponse, error) {
	pubHex := c.pubHex()
	ts := metrics.TimestampMs
	if ts == 0 {
		ts = time.Now().UnixMilli()
	}
	body := heartbeatRequest{
		PublicKeyHex:   pubHex,
		TimestampMs:    ts,
		CPUUtil:        metrics.CPUUtil,
		MemUtil:        metrics.MemUtil,
		GPUUtil:        metrics.GPUUtil,
		PowerWatts:     metrics.PowerWatts,
		GPUThrottled:   metrics.GPUThrottled,
		SystemTimezone: detectIANATimezone(),
		NetworkProfile: metrics.NetworkProfile,
		V7:             metrics.V7Heartbeat,
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
	if c.useGRPCTransport() {
		if resp, err := c.heartbeatGRPC(ctx, body); err == nil || !c.shouldFallbackGRPC(err) {
			return resp, err
		}
	}
	var resp HeartbeatResponse
	err := c.post(ctx, "/api/v1/node/heartbeat", body, &resp)
	return resp, err
}

func (c *Client) FetchWork(ctx context.Context) (*WorkAssignment, error) {
	ts := time.Now().UnixMilli()
	pubHex := c.pubHex()
	sig := c.sign("work", pubHex, strconv.FormatInt(ts, 10))
	if c.useGRPCTransport() {
		if work, err := c.fetchWorkGRPC(ctx, pubHex, ts, sig, true); err == nil || !c.shouldFallbackGRPC(err) {
			return work, err
		}
	}

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
	req.Header.Set("X-Node-Signature", hex.EncodeToString(sig))

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
		JobID:               out.JobID,
		WorkGraphID:         out.WorkGraphID,
		JobPubkey:           out.JobPubkey,
		Kind:                out.Kind,
		PayloadURL:          out.PayloadURL,
		PricePerUnit:        out.PricePerUnit,
		Units:               out.Units,
		Image:               out.Image,
		SpecJSON:            out.SpecJSON,
		ExecutorKind:        out.ExecutorKind,
		AssuranceClass:      out.AssuranceClass,
		RuntimeRequirements: out.RuntimeRequirements,
	}, nil
}

func (c *Client) FetchWorkGraphAbort(ctx context.Context, workGraphID string) (*WorkGraphAbort, error) {
	workGraphID = strings.TrimSpace(workGraphID)
	if workGraphID == "" {
		return nil, nil
	}
	ts := time.Now().UnixMilli()
	pubHex := c.pubHex()
	sig := c.sign("workgraph_abort", pubHex, workGraphID, strconv.FormatInt(ts, 10))
	u, err := url.Parse(c.absoluteURL("/api/v1/node/workgraph-aborts"))
	if err != nil {
		return nil, err
	}
	q := u.Query()
	q.Set("pubkey", pubHex)
	q.Set("workgraph_id", workGraphID)
	u.RawQuery = q.Encode()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", c.userAgent)
	req.Header.Set("X-Node-Timestamp", strconv.FormatInt(ts, 10))
	req.Header.Set("X-Node-Signature", hex.EncodeToString(sig))
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))
		return nil, fmt.Errorf("GET %s: %d %s", u.String(), resp.StatusCode, strings.TrimSpace(string(body)))
	}
	var out workGraphAbortResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if !out.Aborted {
		return nil, nil
	}
	return &out.Abort, nil
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
	if c.useGRPCTransport() {
		if err := c.submitReceiptGRPC(ctx, body); err == nil || !c.shouldFallbackGRPC(err) {
			return err
		}
	}
	return c.post(ctx, "/api/v1/node/receipt", body, nil)
}

func (c *Client) SubmitSpeculativeDraftPacket(ctx context.Context, windowID string, packet map[string]any) (DraftPacketDecision, error) {
	windowID = strings.TrimSpace(windowID)
	if windowID == "" {
		return DraftPacketDecision{}, fmt.Errorf("window_id required")
	}
	if len(packet) == 0 {
		return DraftPacketDecision{}, fmt.Errorf("draft packet required")
	}
	if c.useGRPCTransport() {
		if batch, err := c.submitDraftPacketBatchGRPC(ctx, windowID, []map[string]any{packet}); err == nil {
			if len(batch.Decisions) > 0 {
				return batch.Decisions[0], nil
			}
			return DraftPacketDecision{
				SchemaVersion: batch.SchemaVersion,
				WindowID:      batch.WindowID,
				Accepted:      batch.Accepted > 0,
				Reason:        "accepted",
			}, nil
		} else if !c.shouldFallbackGRPC(err) {
			return DraftPacketDecision{}, err
		}
	}
	headers := map[string]string{
		"X-Node-Token": c.NodeAuthToken(0),
	}
	if c.adminKey != "" {
		headers["X-Admin-Key"] = c.adminKey
	}
	var out DraftPacketDecision
	err := c.postWithHeaders(ctx, "/api/v1/speculative/windows/"+url.PathEscape(windowID)+"/draft-packets", packet, &out, headers)
	return out, err
}

func (c *Client) SubmitSpeculativeDraftPacketBatch(ctx context.Context, windowID string, packets []map[string]any) (DraftPacketBatchDecision, error) {
	windowID = strings.TrimSpace(windowID)
	if windowID == "" {
		return DraftPacketBatchDecision{}, fmt.Errorf("window_id required")
	}
	if len(packets) == 0 {
		return DraftPacketBatchDecision{}, fmt.Errorf("draft packets required")
	}
	if c.useGRPCTransport() {
		if out, err := c.submitDraftPacketBatchGRPC(ctx, windowID, packets); err == nil || !c.shouldFallbackGRPC(err) {
			return out, err
		}
	}
	headers := map[string]string{
		"X-Node-Token": c.NodeAuthToken(0),
	}
	if c.adminKey != "" {
		headers["X-Admin-Key"] = c.adminKey
	}
	var out DraftPacketBatchDecision
	err := c.postWithHeaders(ctx, "/api/v1/speculative/windows/"+url.PathEscape(windowID)+"/draft-packets/batch", map[string]any{"packets": packets}, &out, headers)
	return out, err
}

type SpeculativeLiveLabSessionCommand struct {
	SchemaVersion       string         `json:"schema_version"`
	Command             string         `json:"command"`
	CommandID           string         `json:"command_id,omitempty"`
	Role                string         `json:"role,omitempty"`
	RunID               string         `json:"run_id,omitempty"`
	JobID               string         `json:"job_id,omitempty"`
	WorkGraphID         string         `json:"workgraph_id,omitempty"`
	SessionID           string         `json:"session_id,omitempty"`
	WindowID            string         `json:"window_id,omitempty"`
	WaveIndex           int            `json:"wave_index,omitempty"`
	RoleID              string         `json:"role_id,omitempty"`
	TargetNodeID        string         `json:"target_node_id,omitempty"`
	NodeID              string         `json:"node_id,omitempty"`
	Prompt              string         `json:"prompt,omitempty"`
	ParentPrefixHash    string         `json:"parent_prefix_hash,omitempty"`
	BranchCount         int            `json:"branch_count,omitempty"`
	Horizon             int            `json:"horizon,omitempty"`
	DeadlineMs          int            `json:"deadline_ms,omitempty"`
	FirstPacketTimeout  int            `json:"first_packet_timeout_ms,omitempty"`
	ModelHash           string         `json:"model_hash,omitempty"`
	DrafterModelID      string         `json:"drafter_model_id,omitempty"`
	AcceptedTokensTotal int            `json:"accepted_tokens_total,omitempty"`
	Reason              string         `json:"reason,omitempty"`
	Tree                map[string]any `json:"tree,omitempty"`
}

type SpeculativeLiveLabVerifierResult struct {
	JobID              string         `json:"job_id"`
	WindowID           string         `json:"window_id"`
	WaveIndex          int            `json:"wave_index"`
	AcceptedLen        int            `json:"accepted_len"`
	TreeCID            string         `json:"tree_cid,omitempty"`
	DurationMs         int64          `json:"duration_ms,omitempty"`
	AcceptedText       string         `json:"accepted_text,omitempty"`
	AcceptedTextHash   string         `json:"accepted_text_hash,omitempty"`
	AcceptedTextPublic bool           `json:"accepted_text_public,omitempty"`
	EOS                bool           `json:"eos,omitempty"`
	StopReason         string         `json:"stop_reason,omitempty"`
	ProbeSummary       map[string]any `json:"probe_summary,omitempty"`
}

func (c *Client) FetchSpeculativeLiveLabDraftCommand(ctx context.Context, runID string, jobID string) (SpeculativeLiveLabSessionCommand, error) {
	return c.fetchSpeculativeLiveLabSessionCommand(ctx, runID, jobID, "draft-command")
}

func (c *Client) FetchSpeculativeLiveLabVerifierCommand(ctx context.Context, runID string, jobID string) (SpeculativeLiveLabSessionCommand, error) {
	return c.fetchSpeculativeLiveLabSessionCommand(ctx, runID, jobID, "verifier-command")
}

func (c *Client) fetchSpeculativeLiveLabSessionCommand(ctx context.Context, runID string, jobID string, commandPath string) (SpeculativeLiveLabSessionCommand, error) {
	runID = strings.TrimSpace(runID)
	if runID == "" {
		return SpeculativeLiveLabSessionCommand{}, fmt.Errorf("run_id required")
	}
	if c.useGRPCTransport() {
		role := "target_verifier"
		if commandPath == "draft-command" {
			role = "draft_worker"
		}
		if out, err := c.fetchSpeculativeLiveLabSessionCommandGRPC(ctx, runID, jobID, role); err == nil || !c.shouldFallbackGRPC(err) {
			return out, err
		}
	}
	u := "/api/v1/node/speculative/live-lab/runs/" + url.PathEscape(runID) + "/" + commandPath
	if strings.TrimSpace(jobID) != "" {
		u += "?job_id=" + url.QueryEscape(strings.TrimSpace(jobID))
	}
	headers := map[string]string{"X-Node-Token": c.NodeAuthToken(0)}
	var out SpeculativeLiveLabSessionCommand
	if err := c.getWithHeaders(ctx, u, &out, headers); err != nil {
		return SpeculativeLiveLabSessionCommand{}, err
	}
	return out, nil
}

func (c *Client) SubmitSpeculativeLiveLabVerifierResult(ctx context.Context, runID string, result SpeculativeLiveLabVerifierResult) error {
	runID = strings.TrimSpace(runID)
	if runID == "" {
		return fmt.Errorf("run_id required")
	}
	result = sanitizeSpeculativeLiveLabVerifierResult(result)
	if c.useGRPCTransport() {
		if err := c.submitSpeculativeLiveLabVerifierResultGRPC(ctx, runID, result); err == nil || !c.shouldFallbackGRPC(err) {
			return err
		}
	}
	headers := map[string]string{"X-Node-Token": c.NodeAuthToken(0)}
	return c.postWithHeaders(ctx, "/api/v1/node/speculative/live-lab/runs/"+url.PathEscape(runID)+"/verifier-results", result, nil, headers)
}

func sanitizeSpeculativeLiveLabVerifierResult(result SpeculativeLiveLabVerifierResult) SpeculativeLiveLabVerifierResult {
	if strings.TrimSpace(result.AcceptedText) != "" && strings.TrimSpace(result.AcceptedTextHash) == "" {
		sum := sha256.Sum256([]byte(result.AcceptedText))
		result.AcceptedTextHash = "sha256:" + hex.EncodeToString(sum[:])
	}
	result.AcceptedText = ""
	return result
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

// Attest performs TEE attestation with the hub via challenge-response protocol.
func (c *Client) Attest(ctx context.Context, caps hw.CapSet) error {
	if !caps.TEESupported {
		return nil
	}

	// Step 1: Request challenge nonce
	var challenge struct {
		Nonce string `json:"nonce"`
	}
	if err := c.post(ctx, "/api/v1/node/attest/challenge",
		map[string]string{"public_key_hex": c.pubHex()}, &challenge); err != nil {
		return fmt.Errorf("attestation challenge: %w", err)
	}

	// Step 2: Generate attestation report with the nonce
	nonce, err := hex.DecodeString(challenge.Nonce)
	if err != nil {
		return fmt.Errorf("bad nonce: %w", err)
	}
	report := hw.GenerateAttestationReport(nonce)
	if report.ReportB64 == "" {
		return fmt.Errorf("failed to generate attestation report")
	}

	// Step 3: Submit for verification
	var result struct {
		Verified bool   `json:"verified"`
		Reason   string `json:"reason,omitempty"`
	}
	if err := c.post(ctx, "/api/v1/node/attest/verify", map[string]any{
		"public_key_hex": c.pubHex(),
		"method":         report.Method,
		"tee_type":       report.TEEType,
		"report_b64":     report.ReportB64,
		"nonce_hex":      report.NonceHex,
		"cert_chain":     report.CertChain,
	}, &result); err != nil {
		return fmt.Errorf("attestation verify: %w", err)
	}

	if !result.Verified {
		return fmt.Errorf("attestation rejected: %s", result.Reason)
	}
	return nil
}

// ReportAgentHealth sends a signed health check for a running agent deployment.
func (c *Client) ReportAgentHealth(ctx context.Context, deploymentID string, uptimeSeconds int) (AgentHealthResponse, error) {
	deploymentID = strings.TrimSpace(deploymentID)
	if deploymentID == "" {
		return AgentHealthResponse{}, fmt.Errorf("deployment_id required")
	}
	status := "healthy"
	ts := time.Now().UnixMilli()
	pubHex := c.pubHex()
	body := agentHealthRequest{
		PublicKeyHex:  pubHex,
		TimestampMs:   ts,
		Status:        status,
		UptimeSeconds: uptimeSeconds,
		Signature:     c.sign("agent_health", pubHex, deploymentID, strconv.FormatInt(ts, 10), strconv.Itoa(uptimeSeconds), status),
	}
	var out AgentHealthResponse
	if err := c.post(ctx, "/api/v1/node/agent-health/"+deploymentID, body, &out); err != nil {
		return AgentHealthResponse{}, err
	}
	return out, nil
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
		RuntimeGPU:   report.RuntimeGPU,
		Message:      message,
	}
	body.Signature = c.sign(
		"health",
		pubHex,
		strconv.FormatInt(ts, 10),
		boolAsInt(report.GPUReady),
		boolAsInt(report.RuntimeGPU),
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

func (c *Client) NodeModelSnapshotURL(modelID string) string {
	if c == nil || strings.TrimSpace(modelID) == "" {
		return ""
	}
	return c.absoluteURL("/api/v1/node/models/" + url.PathEscape(strings.TrimSpace(modelID)) + "/snapshot.tar.gz")
}

func (c *Client) PublicKeyHex() string {
	return c.pubHex()
}

func (c *Client) HeartbeatProbeTarget() string {
	if c == nil {
		return ""
	}
	return c.absoluteURL("/api/v1/ping")
}

func (c *Client) ProbeHubRTT(ctx context.Context) (time.Duration, error) {
	if c == nil {
		return 0, fmt.Errorf("nil hub client")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	probeCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if rtt, err := c.probeHubRTTBurst(probeCtx, http.MethodHead, "/api/v1/ping", 3); err == nil {
		return rtt, nil
	}
	return c.probeHubRTTBurst(probeCtx, http.MethodGet, "/healthz", 2)
}

func (c *Client) probeHubRTTBurst(ctx context.Context, method string, path string, attempts int) (time.Duration, error) {
	if attempts <= 0 {
		attempts = 1
	}
	var best time.Duration
	var lastErr error
	for i := 0; i < attempts; i++ {
		rtt, err := c.probeHubRTT(ctx, method, path)
		if err == nil {
			if best <= 0 || rtt < best {
				best = rtt
			}
		} else {
			lastErr = err
		}
		if i+1 >= attempts {
			break
		}
		select {
		case <-ctx.Done():
			if best > 0 {
				return best, nil
			}
			if lastErr != nil {
				return 0, lastErr
			}
			return 0, ctx.Err()
		case <-time.After(40 * time.Millisecond):
		}
	}
	if best > 0 {
		return best, nil
	}
	if lastErr != nil {
		return 0, lastErr
	}
	return 0, fmt.Errorf("%s %s: no RTT samples", method, path)
}

func (c *Client) probeHubRTT(ctx context.Context, method string, path string) (time.Duration, error) {
	req, err := http.NewRequestWithContext(ctx, method, c.absoluteURL(path), nil)
	if err != nil {
		return 0, err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", c.userAgent)
	start := time.Now()
	resp, err := c.http.Do(req)
	rtt := time.Since(start)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 512))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return 0, fmt.Errorf("%s %s: %d", method, req.URL.String(), resp.StatusCode)
	}
	return rtt, nil
}

// RedeemClaimCode sends a claim code to the hub to link this node to a buyer account.
func (c *Client) RedeemClaimCode(ctx context.Context, code string) error {
	body := map[string]string{"code": code}
	headers := map[string]string{"X-Node-Token": c.NodeAuthToken(0)}
	return c.postWithHeaders(ctx, "/api/v1/node/claim", body, nil, headers)
}

func (c *Client) CreateConnectAccount(ctx context.Context, email, country string) (string, error) {
	email = strings.TrimSpace(email)
	if email == "" {
		return "", fmt.Errorf("email required")
	}
	country = strings.ToUpper(strings.TrimSpace(country))
	if country == "" {
		return "", fmt.Errorf("country required")
	}
	var out connectCreateResponse
	if err := c.postWithHeaders(ctx, "/api/v1/node/connect/create", map[string]string{
		"email":   email,
		"country": country,
	}, &out, map[string]string{"X-Node-Token": c.NodeAuthToken(0)}); err != nil {
		return "", err
	}
	if strings.TrimSpace(out.AccountID) == "" {
		return "", fmt.Errorf("connect account response missing account_id")
	}
	return out.AccountID, nil
}

func (c *Client) ConnectOnboardingLink(ctx context.Context, accountID string) (string, error) {
	accountID = strings.TrimSpace(accountID)
	if accountID == "" {
		return "", fmt.Errorf("account_id required")
	}
	var out connectOnboardingResponse
	if err := c.postWithHeaders(ctx, "/api/v1/node/connect/onboarding-link", map[string]string{
		"account_id": accountID,
	}, &out, map[string]string{"X-Node-Token": c.NodeAuthToken(0)}); err != nil {
		return "", err
	}
	if strings.TrimSpace(out.URL) == "" {
		return "", fmt.Errorf("onboarding link response missing url")
	}
	return out.URL, nil
}

func (c *Client) ConnectStatus(ctx context.Context, accountID string) (bool, error) {
	accountID = strings.TrimSpace(accountID)
	if accountID == "" {
		return false, fmt.Errorf("account_id required")
	}
	var out connectStatusResponse
	if err := c.getWithHeaders(ctx, "/api/v1/node/connect/status?account_id="+url.QueryEscape(accountID), &out, map[string]string{
		"X-Node-Token": c.NodeAuthToken(0),
	}); err != nil {
		return false, err
	}
	return out.Onboarded, nil
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

func (c *Client) SendDashboardInferenceProgress(ctx context.Context, batch routedinference.ProgressBatch) error {
	if len(batch.Chunks) == 0 {
		return nil
	}
	pubHex := c.pubHex()
	nodeID := strings.TrimSpace(batch.NodeID)
	if nodeID == "" {
		nodeID = pubHex
	}
	body := dashboardInferenceProgressRequest{
		RunID:        strings.TrimSpace(batch.RunID),
		JobID:        strings.TrimSpace(batch.JobID),
		NodeID:       nodeID,
		PublicKeyHex: pubHex,
		SeqStart:     batch.SeqStart,
		Chunks:       append([]routedinference.ProgressChunk(nil), batch.Chunks...),
	}
	return c.postWithHeaders(ctx, "/api/v1/node/inference/chunks", body, nil, map[string]string{
		"X-Node-Pubkey": pubHex,
	})
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

func (c *Client) getWithHeaders(ctx context.Context, path string, out any, headers map[string]string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.absoluteURL(path), nil)
	if err != nil {
		return err
	}
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
		return fmt.Errorf("GET %s: %d %s", req.URL.String(), resp.StatusCode, strings.TrimSpace(string(rb)))
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
	CPUModel          string
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
	TimestampMs    int64
	CPUUtil        float64
	MemUtil        float64
	GPUUtil        float64
	PowerWatts     float64
	GPUThrottled   bool // node is self-throttling due to operator GPU usage
	NetworkProfile *netprofile.NetworkProfile
	V7Heartbeat    *heartbeat.V7HeartbeatPayload
}

type WorkAssignment struct {
	JobID               string
	WorkGraphID         string
	JobPubkey           string
	Kind                string
	PayloadURL          string
	PricePerUnit        uint64
	Units               uint32
	Image               string
	SpecJSON            string
	ExecutorKind        string
	AssuranceClass      string
	RuntimeRequirements RuntimeRequirements
}

type WorkGraphAbort struct {
	WorkGraphHash   string `json:"workgraph_hash"`
	AbortEpoch      int64  `json:"abort_epoch"`
	Reason          string `json:"reason"`
	IssuedAt        string `json:"issued_at"`
	NoCreditAfter   string `json:"no_credit_after"`
	NoCreditAfterMs int64  `json:"no_credit_after_ms"`
}

type RuntimeRequirements struct {
	NeedsGPU             bool     `json:"needs_gpu,omitempty"`
	NeedsManagedOCI      bool     `json:"needs_managed_oci,omitempty"`
	NeedsManagedOCIGPU   bool     `json:"needs_managed_oci_gpu,omitempty"`
	NeedsRyvionRuntime   bool     `json:"needs_ryvion_runtime,omitempty"`
	NeedsNativeStreaming bool     `json:"needs_native_streaming,omitempty"`
	NeedsNativeReport    bool     `json:"needs_native_report,omitempty"`
	NeedsAgentHosting    bool     `json:"needs_agent_hosting,omitempty"`
	Tooling              []string `json:"tooling,omitempty"`
	MinDiskGB            uint64   `json:"min_disk_gb,omitempty"`
	MinVRAMMB            uint32   `json:"min_vram_mb,omitempty"`
	Jurisdiction         string   `json:"jurisdiction,omitempty"`
	TrustLevel           string   `json:"trust_level,omitempty"`
}

type Receipt struct {
	JobID         string
	ResultHashHex string
	MeteringUnits uint64
	Metadata      map[string]any
}

type DraftPacketDecision struct {
	SchemaVersion string `json:"schema_version"`
	WindowID      string `json:"window_id"`
	Accepted      bool   `json:"accepted"`
	Reason        string `json:"reason"`
	PacketID      string `json:"packet_id"`
}

type DraftPacketBatchDecision struct {
	SchemaVersion string                `json:"schema_version"`
	WindowID      string                `json:"window_id"`
	Attempted     int                   `json:"attempted"`
	Accepted      int                   `json:"accepted"`
	Rejected      int                   `json:"rejected"`
	Decisions     []DraftPacketDecision `json:"decisions"`
}

type HealthReport struct {
	TimestampMs int64
	GPUReady    bool
	RuntimeGPU  bool
	Message     string
}

type AgentHealthResponse struct {
	OK         bool   `json:"ok"`
	ShouldStop bool   `json:"should_stop"`
	Status     string `json:"status"`
	JobStatus  string `json:"job_status"`
}

type HeartbeatResponse struct {
	LatestVersion        string `json:"latest_version,omitempty"`
	NodeID               string `json:"node_id,omitempty"`
	CountryCode          string `json:"country_code,omitempty"`
	LocationApproved     bool   `json:"location_approved,omitempty"`
	SovereignVerified    bool   `json:"sovereign_verified,omitempty"`
	VerificationSource   string `json:"verification_source,omitempty"`
	TrustReason          string `json:"trust_reason,omitempty"`
	V7SnapshotUpserted   *bool  `json:"v7_snapshot_upserted"`
	SnapshotModelCount   int    `json:"snapshot_model_count"`
	SnapshotBackendCount int    `json:"snapshot_backend_count"`
	HasCapabilityProfile bool   `json:"has_capability_profile"`
	HubInstanceID        string `json:"hub_instance_id,omitempty"`
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
	CPUModel          string `json:"cpu_model"`
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
	PublicKeyHex   string                        `json:"public_key_hex"`
	TimestampMs    int64                         `json:"timestamp_ms"`
	CPUUtil        float64                       `json:"cpu_util"`
	MemUtil        float64                       `json:"mem_util"`
	GPUUtil        float64                       `json:"gpu_util"`
	PowerWatts     float64                       `json:"power_watts"`
	GPUThrottled   bool                          `json:"gpu_throttled"`
	SystemTimezone string                        `json:"system_timezone,omitempty"`
	NetworkProfile *netprofile.NetworkProfile    `json:"network_profile,omitempty"`
	V7             *heartbeat.V7HeartbeatPayload `json:"v7,omitempty"`
	Signature      []byte                        `json:"signature"`
}

type workResponse struct {
	HasWork             *bool               `json:"has_work"`
	JobID               string              `json:"job_id"`
	WorkGraphID         string              `json:"workgraph_id"`
	JobPubkey           string              `json:"job_pubkey"`
	Kind                string              `json:"kind"`
	PayloadURL          string              `json:"payload_url"`
	PricePerUnit        uint64              `json:"price_per_unit"`
	Units               uint32              `json:"units"`
	Image               string              `json:"image"`
	SpecJSON            string              `json:"spec_json"`
	ExecutorKind        string              `json:"executor_kind"`
	AssuranceClass      string              `json:"assurance_class"`
	RuntimeRequirements RuntimeRequirements `json:"runtime_requirements"`
}

type workGraphAbortResponse struct {
	SchemaVersion string         `json:"schema_version"`
	Status        string         `json:"status"`
	Aborted       bool           `json:"aborted"`
	Abort         WorkGraphAbort `json:"abort"`
}

type receiptRequest struct {
	JobID         string         `json:"job_id"`
	PublicKeyHex  string         `json:"public_key_hex"`
	ResultHashHex string         `json:"result_hash_hex"`
	MeteringUnits uint64         `json:"metering_units"`
	Signature     []byte         `json:"signature"`
	Metadata      map[string]any `json:"metadata,omitempty"`
}

type dashboardInferenceProgressRequest struct {
	RunID        string                          `json:"run_id,omitempty"`
	JobID        string                          `json:"job_id,omitempty"`
	NodeID       string                          `json:"node_id,omitempty"`
	PublicKeyHex string                          `json:"public_key_hex,omitempty"`
	SeqStart     int64                           `json:"seq_start"`
	Chunks       []routedinference.ProgressChunk `json:"chunks"`
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
	RuntimeGPU   bool   `json:"managed_oci_gpu_ready"`
	Message      string `json:"message"`
	Signature    []byte `json:"signature"`
}

type agentHealthRequest struct {
	PublicKeyHex  string `json:"public_key_hex"`
	TimestampMs   int64  `json:"timestamp_ms"`
	Status        string `json:"status"`
	UptimeSeconds int    `json:"uptime_seconds"`
	Signature     []byte `json:"signature"`
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

type connectCreateResponse struct {
	AccountID string `json:"account_id"`
}

type connectOnboardingResponse struct {
	URL string `json:"url"`
}

type connectStatusResponse struct {
	AccountID string `json:"account_id"`
	Onboarded bool   `json:"onboarded"`
}
