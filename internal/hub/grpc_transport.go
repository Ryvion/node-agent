package hub

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"net"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	v7alphapb "github.com/Ryvion/node-agent/internal/genproto/v7alpha"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/structpb"
)

const (
	hubTransportHTTP = "http"
	hubTransportGRPC = "grpc"
	hubTransportAuto = "auto"
)

type grpcTransport struct {
	target   string
	mode     string
	insecure bool

	mu     sync.Mutex
	conn   *grpc.ClientConn
	client v7alphapb.V7NodeTransportServiceClient
}

func defaultGRPCTransport() *grpcTransport {
	target := firstNonEmptyEnv("RYV_NODE_HUB_GRPC_ADDR", "RYV_HUB_GRPC_ADDR")
	mode := normalizeHubTransportMode(os.Getenv("RYV_NODE_HUB_TRANSPORT"), target)
	return &grpcTransport{
		target:   strings.TrimSpace(target),
		mode:     mode,
		insecure: parseBoolEnv("RYV_NODE_HUB_GRPC_INSECURE") || parseBoolEnv("RYV_HUB_GRPC_INSECURE"),
	}
}

// WithGRPCTransport enables the production hub-node gRPC transport. mode is
// "grpc", "auto", or "http"; auto falls back to HTTP only for transport-level
// failures such as Unavailable or Unimplemented.
func WithGRPCTransport(target string, mode string, insecureTransport bool) Option {
	return func(c *Client) {
		c.grpc = &grpcTransport{
			target:   strings.TrimSpace(target),
			mode:     normalizeHubTransportMode(mode, target),
			insecure: insecureTransport,
		}
		if c.grpc != nil {
			c.grpc.applyDefaultTarget(c.baseURL)
		}
	}
}

func (c *Client) useGRPCTransport() bool {
	return c != nil && c.grpc != nil && c.grpc.enabled()
}

func (c *Client) shouldFallbackGRPC(err error) bool {
	if err == nil || c == nil || c.grpc == nil || c.grpc.required() {
		return false
	}
	switch status.Code(err) {
	case codes.Unavailable, codes.Unimplemented, codes.DeadlineExceeded:
		return true
	default:
		return false
	}
}

func (c *Client) registerGRPC(ctx context.Context, body registerRequest) error {
	client, err := c.v7NodeTransportClient(ctx)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(c.grpcMetadataContext(ctx), 15*time.Second)
	defer cancel()
	_, err = client.RegisterNode(ctx, &v7alphapb.RegisterNodeRequest{
		PublicKeyHex:      body.PublicKeyHex,
		DeviceType:        body.DeviceType,
		DeclaredCountry:   body.DeclaredCountry,
		GpuModel:          body.GPUModel,
		CpuCores:          body.CPUCores,
		RamBytes:          body.RAMBytes,
		VramBytes:         body.VRAMBytes,
		Sensors:           body.Sensors,
		BandwidthMbps:     body.BandwidthMbps,
		GeohashBucket:     body.GeohashBucket,
		AttestationMethod: body.AttestationMethod,
		ReferralCode:      body.ReferralCode,
		TeeSupported:      body.TEESupported,
		TeeType:           body.TEEType,
		Signature:         append([]byte(nil), body.Signature...),
	})
	return err
}

func (c *Client) heartbeatGRPC(ctx context.Context, body heartbeatRequest) (HeartbeatResponse, error) {
	client, err := c.v7NodeTransportClient(ctx)
	if err != nil {
		return HeartbeatResponse{}, err
	}
	var v7Struct *structpb.Struct
	if body.V7 != nil {
		v7Struct, err = structFromJSONValue(body.V7)
		if err != nil {
			return HeartbeatResponse{}, err
		}
	}
	ctx, cancel := context.WithTimeout(c.grpcMetadataContext(ctx), 10*time.Second)
	defer cancel()
	resp, err := client.SendHeartbeat(ctx, &v7alphapb.SendHeartbeatRequest{
		PublicKeyHex:   body.PublicKeyHex,
		TimestampMs:    body.TimestampMs,
		CpuUtil:        body.CPUUtil,
		MemUtil:        body.MemUtil,
		GpuUtil:        body.GPUUtil,
		PowerWatts:     body.PowerWatts,
		GpuThrottled:   body.GPUThrottled,
		SystemTimezone: body.SystemTimezone,
		V7:             v7Struct,
		Signature:      append([]byte(nil), body.Signature...),
	})
	if err != nil {
		return HeartbeatResponse{}, err
	}
	upserted := resp.GetV7SnapshotUpserted()
	return HeartbeatResponse{
		LatestVersion:        resp.GetLatestVersion(),
		NodeID:               resp.GetNodeId(),
		CountryCode:          resp.GetCountryCode(),
		LocationApproved:     resp.GetLocationApproved(),
		SovereignVerified:    resp.GetSovereignVerified(),
		VerificationSource:   resp.GetVerificationSource(),
		TrustReason:          resp.GetTrustReason(),
		V7SnapshotUpserted:   &upserted,
		SnapshotModelCount:   int(resp.GetSnapshotModelCount()),
		SnapshotBackendCount: int(resp.GetSnapshotBackendCount()),
		HasCapabilityProfile: resp.GetHasCapabilityProfile(),
		HubInstanceID:        resp.GetHubInstanceId(),
	}, nil
}

func (c *Client) fetchWorkGRPC(ctx context.Context, pubHex string, ts int64, signature []byte, longPoll bool) (*WorkAssignment, error) {
	client, err := c.v7NodeTransportClient(ctx)
	if err != nil {
		return nil, err
	}
	timeout := 10 * time.Second
	if longPoll {
		timeout = 35 * time.Second
	}
	ctx, cancel := context.WithTimeout(c.grpcMetadataContext(ctx), timeout)
	defer cancel()
	resp, err := client.LeaseWork(ctx, &v7alphapb.LeaseWorkRequest{
		PublicKeyHex: pubHex,
		TimestampMs:  ts,
		LongPoll:     longPoll,
		Signature:    append([]byte(nil), signature...),
	})
	if err != nil {
		return nil, err
	}
	if !resp.GetHasWork() || resp.GetAssignment() == nil {
		return nil, nil
	}
	assignment := resp.GetAssignment()
	if strings.TrimSpace(assignment.GetJobId()) == "" {
		return nil, fmt.Errorf("work assignment missing job_id")
	}
	return &WorkAssignment{
		JobID:               assignment.GetJobId(),
		JobPubkey:           assignment.GetJobPubkey(),
		Kind:                assignment.GetKind(),
		PayloadURL:          assignment.GetPayloadUrl(),
		PricePerUnit:        assignment.GetPricePerUnit(),
		Units:               assignment.GetUnits(),
		Image:               assignment.GetImage(),
		SpecJSON:            assignment.GetSpecJson(),
		ExecutorKind:        assignment.GetExecutorKind(),
		AssuranceClass:      assignment.GetAssuranceClass(),
		RuntimeRequirements: runtimeRequirementsFromProto(assignment.GetRuntimeRequirements()),
	}, nil
}

func (c *Client) submitReceiptGRPC(ctx context.Context, body receiptRequest) error {
	client, err := c.v7NodeTransportClient(ctx)
	if err != nil {
		return err
	}
	var metadataStruct *structpb.Struct
	if body.Metadata != nil {
		metadataStruct, err = structFromJSONValue(body.Metadata)
		if err != nil {
			return err
		}
	}
	ctx, cancel := context.WithTimeout(c.grpcMetadataContext(ctx), 30*time.Second)
	defer cancel()
	_, err = client.SubmitReceipt(ctx, &v7alphapb.SubmitReceiptRequest{
		JobId:         body.JobID,
		PublicKeyHex:  body.PublicKeyHex,
		ResultHashHex: body.ResultHashHex,
		MeteringUnits: body.MeteringUnits,
		Signature:     append([]byte(nil), body.Signature...),
		Metadata:      metadataStruct,
	})
	return err
}

func (c *Client) sendDashboardInferenceProgressGRPC(ctx context.Context, body dashboardInferenceProgressRequest) error {
	client, err := c.v7NodeTransportClient(ctx)
	if err != nil {
		return err
	}
	chunks := make([]*v7alphapb.V7DashboardInferenceProgressChunk, 0, len(body.Chunks))
	for _, chunk := range body.Chunks {
		chunks = append(chunks, &v7alphapb.V7DashboardInferenceProgressChunk{
			Seq:          chunk.Seq,
			Type:         chunk.Type,
			Text:         chunk.Text,
			FinishReason: chunk.FinishReason,
		})
	}
	ctx, cancel := context.WithTimeout(c.grpcMetadataContext(ctx), 10*time.Second)
	defer cancel()
	_, err = client.SubmitDashboardInferenceProgress(ctx, &v7alphapb.SubmitDashboardInferenceProgressRequest{
		RunId:        body.RunID,
		JobId:        body.JobID,
		NodeId:       body.NodeID,
		PublicKeyHex: body.PublicKeyHex,
		SeqStart:     body.SeqStart,
		Chunks:       chunks,
	})
	return err
}

func (c *Client) v7NodeTransportClient(ctx context.Context) (v7alphapb.V7NodeTransportServiceClient, error) {
	if c == nil || c.grpc == nil || !c.grpc.enabled() {
		return nil, status.Error(codes.Unavailable, "gRPC transport disabled")
	}
	return c.grpc.clientFor(ctx)
}

func (c *Client) Close() error {
	if c == nil || c.grpc == nil {
		return nil
	}
	return c.grpc.close()
}

func (c *Client) grpcMetadataContext(ctx context.Context) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return metadata.AppendToOutgoingContext(ctx,
		"x-ryvion-node-pubkey", c.pubHex(),
		"x-ryvion-agent-version", c.userAgent,
	)
}

func (t *grpcTransport) close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.conn == nil {
		return nil
	}
	err := t.conn.Close()
	t.conn = nil
	t.client = nil
	return err
}

func (t *grpcTransport) enabled() bool {
	return t != nil && t.mode != hubTransportHTTP && strings.TrimSpace(t.target) != ""
}

func (t *grpcTransport) required() bool {
	return t != nil && t.mode == hubTransportGRPC
}

func (t *grpcTransport) clientFor(ctx context.Context) (v7alphapb.V7NodeTransportServiceClient, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.client != nil {
		return t.client, nil
	}
	target := strings.TrimSpace(t.target)
	if target == "" {
		return nil, status.Error(codes.Unavailable, "gRPC target not configured")
	}
	opts := []grpc.DialOption{
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                60 * time.Second,
			Timeout:             10 * time.Second,
			PermitWithoutStream: true,
		}),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(4*1024*1024),
			grpc.MaxCallSendMsgSize(4*1024*1024),
		),
	}
	if t.insecure {
		opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	} else {
		opts = append(opts, grpc.WithTransportCredentials(credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12})))
	}
	conn, err := grpc.NewClient(target, opts...)
	if err != nil {
		return nil, err
	}
	t.conn = conn
	t.client = v7alphapb.NewV7NodeTransportServiceClient(conn)
	return t.client, nil
}

func (t *grpcTransport) applyDefaultTarget(baseURL string) {
	if t == nil || strings.TrimSpace(t.target) != "" || t.mode == hubTransportHTTP {
		return
	}
	u, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil || strings.TrimSpace(u.Host) == "" {
		return
	}
	host := u.Host
	if _, _, err := net.SplitHostPort(host); err != nil {
		switch strings.ToLower(u.Scheme) {
		case "https":
			host = net.JoinHostPort(host, "443")
		default:
			host = net.JoinHostPort(host, "80")
			t.insecure = true
		}
	}
	t.target = host
}

func runtimeRequirementsFromProto(req *v7alphapb.V7RuntimeRequirements) RuntimeRequirements {
	if req == nil {
		return RuntimeRequirements{}
	}
	return RuntimeRequirements{
		NeedsGPU:             req.GetNeedsGpu(),
		NeedsManagedOCI:      req.GetNeedsManagedOci(),
		NeedsManagedOCIGPU:   req.GetNeedsManagedOciGpu(),
		NeedsRyvionRuntime:   req.GetNeedsRyvionRuntime(),
		NeedsNativeStreaming: req.GetNeedsNativeStreaming(),
		NeedsNativeReport:    req.GetNeedsNativeReport(),
		NeedsAgentHosting:    req.GetNeedsAgentHosting(),
		Tooling:              append([]string(nil), req.GetTooling()...),
		MinDiskGB:            req.GetMinDiskGb(),
		MinVRAMMB:            req.GetMinVramMb(),
		Jurisdiction:         req.GetJurisdiction(),
		TrustLevel:           req.GetTrustLevel(),
	}
}

func structFromJSONValue(v any) (*structpb.Struct, error) {
	if v == nil {
		return nil, nil
	}
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	return structpb.NewStruct(m)
}

func normalizeHubTransportMode(mode string, target string) string {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case hubTransportGRPC:
		return hubTransportGRPC
	case hubTransportAuto:
		return hubTransportAuto
	case hubTransportHTTP:
		return hubTransportHTTP
	default:
		if strings.TrimSpace(target) != "" {
			return hubTransportAuto
		}
		return hubTransportHTTP
	}
}

func firstNonEmptyEnv(names ...string) string {
	for _, name := range names {
		if v := strings.TrimSpace(os.Getenv(name)); v != "" {
			return v
		}
	}
	return ""
}

func parseBoolEnv(name string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(name))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}
