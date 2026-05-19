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

	nodev1 "github.com/Ryvion/ryvion-protocol/gen/go/ryvion/node/v1"
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

	mu             sync.Mutex
	conn           *grpc.ClientConn
	gatewayClient  nodev1.NodeGatewayClient
	gatewayStream  nodev1.NodeGateway_ConnectClient
	gatewayPending map[string]chan nodeGatewayResponse
}

type nodeGatewayResponse struct {
	resp *nodev1.HubToNode
	err  error
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

func (c *Client) heartbeatGRPC(ctx context.Context, body heartbeatRequest) (HeartbeatResponse, error) {
	return c.heartbeatNodeGatewayGRPC(ctx, body)
}

func (c *Client) heartbeatNodeGatewayGRPC(ctx context.Context, body heartbeatRequest) (HeartbeatResponse, error) {
	var err error
	var capabilityStruct *structpb.Struct
	capabilityPayload := body.Capability
	if capabilityPayload != nil {
		capabilityStruct, err = structFromJSONValue(capabilityPayload)
		if err != nil {
			return HeartbeatResponse{}, err
		}
	}
	var networkStruct *structpb.Struct
	if body.NetworkProfile != nil {
		networkStruct, err = structFromJSONValue(body.NetworkProfile)
		if err != nil {
			return HeartbeatResponse{}, err
		}
	}
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	resp, err := c.nodeGatewayRoundTrip(ctx, &nodev1.NodeToHub{
		NodeId:          body.PublicKeyHex,
		MessageId:       fmt.Sprintf("heartbeat-%d", body.TimestampMs),
		CreatedAtUnixMs: body.TimestampMs,
		Payload: &nodev1.NodeToHub_Heartbeat{
			Heartbeat: &nodev1.NodeHeartbeat{
				PublicKeyHex:      body.PublicKeyHex,
				TimestampMs:       body.TimestampMs,
				CpuUtil:           body.CPUUtil,
				MemUtil:           body.MemUtil,
				GpuUtil:           body.GPUUtil,
				PowerWatts:        body.PowerWatts,
				GpuThrottled:      body.GPUThrottled,
				SystemTimezone:    body.SystemTimezone,
				CapabilityPayload: capabilityStruct,
				NetworkProfile:    networkStruct,
			},
		},
		Signature: &nodev1.Signature{
			KeyId:     body.PublicKeyHex,
			Algorithm: "ed25519",
			Signature: append([]byte(nil), body.Signature...),
		},
	})
	if err != nil {
		return HeartbeatResponse{}, err
	}
	ack := resp.GetHeartbeatAck()
	if ack == nil {
		return HeartbeatResponse{}, status.Error(codes.Internal, "heartbeat ack missing")
	}
	upserted := ack.GetCapabilityProfileUpserted()
	return HeartbeatResponse{
		LatestVersion:             ack.GetLatestVersion(),
		NodeID:                    ack.GetNodeId(),
		CountryCode:               ack.GetCountryCode(),
		LocationApproved:          ack.GetLocationApproved(),
		SovereignVerified:         ack.GetSovereignVerified(),
		VerificationSource:        ack.GetVerificationSource(),
		TrustReason:               ack.GetTrustReason(),
		CapabilityProfileUpserted: &upserted,
		ProfileRuntimeCount:       int(ack.GetProfileRuntimeCount()),
		ProfileBackendCount:       int(ack.GetProfileBackendCount()),
		HasCapabilityProfile:      ack.GetHasCapabilityProfile(),
		HubInstanceID:             ack.GetHubInstanceId(),
	}, nil
}

func (c *Client) fetchWorkGRPC(ctx context.Context, pubHex string, ts int64, signature []byte, longPoll bool) (*WorkAssignment, error) {
	return c.fetchWorkNodeGatewayGRPC(ctx, pubHex, ts, signature, longPoll)
}

func (c *Client) fetchWorkNodeGatewayGRPC(ctx context.Context, pubHex string, ts int64, signature []byte, longPoll bool) (*WorkAssignment, error) {
	timeout := 10 * time.Second
	if longPoll {
		timeout = 35 * time.Second
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	resp, err := c.nodeGatewayRoundTrip(ctx, &nodev1.NodeToHub{
		NodeId:          pubHex,
		MessageId:       fmt.Sprintf("work-lease-%d", ts),
		CreatedAtUnixMs: ts,
		Payload: &nodev1.NodeToHub_WorkLeaseRequest{
			WorkLeaseRequest: &nodev1.WorkLeaseRequest{
				PublicKeyHex: pubHex,
				TimestampMs:  ts,
				LongPoll:     longPoll,
			},
		},
		Signature: &nodev1.Signature{
			KeyId:     pubHex,
			Algorithm: "ed25519",
			Signature: append([]byte(nil), signature...),
		},
	})
	if err != nil {
		return nil, err
	}
	ack := resp.GetWorkLeaseAck()
	if ack == nil {
		return nil, status.Error(codes.Internal, "work lease ack missing")
	}
	if !ack.GetHasWork() || ack.GetAssignment() == nil {
		return nil, nil
	}
	assignment := ack.GetAssignment()
	if strings.TrimSpace(assignment.GetJobId()) == "" {
		return nil, fmt.Errorf("work assignment missing job_id")
	}
	return &WorkAssignment{
		JobID:               assignment.GetJobId(),
		WorkScopeID:         assignment.GetAbortScopeId(),
		JobPubkey:           assignment.GetJobPubkey(),
		Kind:                assignment.GetKind(),
		PayloadURL:          assignment.GetPayloadUrl(),
		Units:               assignment.GetUnits(),
		Image:               assignment.GetImage(),
		SpecJSON:            assignment.GetSpecJson(),
		ExecutorKind:        assignment.GetExecutorKind(),
		AssuranceClass:      assignment.GetAssuranceClass(),
		RuntimeRequirements: runtimeRequirementsFromNodeProto(assignment.GetRuntimeRequirements()),
	}, nil
}

func (c *Client) submitReceiptGRPC(ctx context.Context, body receiptRequest) error {
	return c.submitReceiptNodeGatewayGRPC(ctx, body)
}

func (c *Client) submitReceiptNodeGatewayGRPC(ctx context.Context, body receiptRequest) error {
	var err error
	var metadataStruct *structpb.Struct
	if body.Metadata != nil {
		metadataStruct, err = structFromJSONValue(body.Metadata)
		if err != nil {
			return err
		}
	}
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	resp, err := c.nodeGatewayRoundTrip(ctx, &nodev1.NodeToHub{
		NodeId:          body.PublicKeyHex,
		MessageId:       fmt.Sprintf("receipt-%s", body.JobID),
		CreatedAtUnixMs: time.Now().UnixMilli(),
		Payload: &nodev1.NodeToHub_Receipt{
			Receipt: &nodev1.WorkReceipt{
				JobId:         body.JobID,
				PublicKeyHex:  body.PublicKeyHex,
				ResultHashHex: body.ResultHashHex,
				MeteringUnits: body.MeteringUnits,
				Metadata:      metadataStruct,
			},
		},
		Signature: &nodev1.Signature{
			KeyId:     body.PublicKeyHex,
			Algorithm: "ed25519",
			Signature: append([]byte(nil), body.Signature...),
		},
	})
	if err != nil {
		return err
	}
	ack := resp.GetReceiptAck()
	if ack == nil {
		return status.Error(codes.Internal, "receipt ack missing")
	}
	if !ack.GetAccepted() {
		if reason := strings.TrimSpace(ack.GetReason()); reason != "" {
			return fmt.Errorf("receipt rejected: %s", reason)
		}
		return fmt.Errorf("receipt rejected")
	}
	return nil
}

func (c *Client) nodeGatewayRoundTrip(ctx context.Context, msg *nodev1.NodeToHub) (*nodev1.HubToNode, error) {
	if c == nil || c.grpc == nil || !c.grpc.enabled() {
		return nil, status.Error(codes.Unavailable, "gRPC transport disabled")
	}
	if msg == nil {
		return nil, status.Error(codes.InvalidArgument, "node gateway message required")
	}
	messageID := strings.TrimSpace(msg.GetMessageId())
	if messageID == "" {
		return nil, status.Error(codes.InvalidArgument, "node gateway message_id required")
	}
	return c.grpc.gatewayRoundTrip(ctx, c.grpcMetadataContext(context.Background()), msg, "ack_"+messageID)
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
	if t.gatewayStream != nil {
		_ = t.gatewayStream.CloseSend()
		t.gatewayStream = nil
	}
	pending := t.gatewayPending
	t.gatewayPending = nil
	if t.conn == nil {
		t.mu.Unlock()
		for _, ch := range pending {
			if ch != nil {
				ch <- nodeGatewayResponse{err: status.Error(codes.Canceled, "gRPC transport closed")}
			}
		}
		return nil
	}
	err := t.conn.Close()
	t.conn = nil
	t.gatewayClient = nil
	t.mu.Unlock()
	for _, ch := range pending {
		if ch != nil {
			ch <- nodeGatewayResponse{err: status.Error(codes.Canceled, "gRPC transport closed")}
		}
	}
	return err
}

func (t *grpcTransport) enabled() bool {
	return t != nil && t.mode != hubTransportHTTP && strings.TrimSpace(t.target) != ""
}

func (t *grpcTransport) required() bool {
	return t != nil && t.mode == hubTransportGRPC
}

func (t *grpcTransport) gatewayRoundTrip(ctx context.Context, streamCtx context.Context, msg *nodev1.NodeToHub, responseID string) (*nodev1.HubToNode, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	ch := make(chan nodeGatewayResponse, 1)
	stream, err := t.gatewayStreamFor(streamCtx)
	if err != nil {
		return nil, err
	}
	t.mu.Lock()
	if t.gatewayPending == nil {
		t.gatewayPending = make(map[string]chan nodeGatewayResponse)
	}
	if _, exists := t.gatewayPending[responseID]; exists {
		t.mu.Unlock()
		return nil, status.Error(codes.Aborted, "duplicate node gateway response id")
	}
	t.gatewayPending[responseID] = ch
	t.mu.Unlock()

	if err := stream.Send(msg); err != nil {
		t.removeGatewayPending(responseID)
		t.resetGatewayStream(stream, err)
		return nil, err
	}
	select {
	case out := <-ch:
		return out.resp, out.err
	case <-ctx.Done():
		t.removeGatewayPending(responseID)
		return nil, ctx.Err()
	}
}

func (t *grpcTransport) gatewayStreamFor(ctx context.Context) (nodev1.NodeGateway_ConnectClient, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.gatewayStream != nil {
		return t.gatewayStream, nil
	}
	conn, err := t.connForLocked(ctx)
	if err != nil {
		return nil, err
	}
	if t.gatewayClient == nil {
		t.gatewayClient = nodev1.NewNodeGatewayClient(conn)
	}
	stream, err := t.gatewayClient.Connect(ctx)
	if err != nil {
		return nil, err
	}
	t.gatewayStream = stream
	if t.gatewayPending == nil {
		t.gatewayPending = make(map[string]chan nodeGatewayResponse)
	}
	go t.recvGatewayStream(stream)
	return stream, nil
}

func (t *grpcTransport) recvGatewayStream(stream nodev1.NodeGateway_ConnectClient) {
	for {
		resp, err := stream.Recv()
		if err != nil {
			t.resetGatewayStream(stream, err)
			return
		}
		if resp == nil {
			continue
		}
		t.mu.Lock()
		ch := t.gatewayPending[resp.GetMessageId()]
		if ch != nil {
			delete(t.gatewayPending, resp.GetMessageId())
		}
		t.mu.Unlock()
		if ch != nil {
			ch <- nodeGatewayResponse{resp: resp}
		}
	}
}

func (t *grpcTransport) removeGatewayPending(responseID string) {
	t.mu.Lock()
	delete(t.gatewayPending, responseID)
	t.mu.Unlock()
}

func (t *grpcTransport) resetGatewayStream(stream nodev1.NodeGateway_ConnectClient, streamErr error) {
	if streamErr == nil {
		streamErr = status.Error(codes.Unavailable, "node gateway stream closed")
	}
	t.mu.Lock()
	if stream != nil && t.gatewayStream != stream {
		t.mu.Unlock()
		return
	}
	if t.gatewayStream != nil {
		_ = t.gatewayStream.CloseSend()
	}
	t.gatewayStream = nil
	pending := t.gatewayPending
	t.gatewayPending = nil
	t.mu.Unlock()

	for _, ch := range pending {
		if ch != nil {
			ch <- nodeGatewayResponse{err: streamErr}
		}
	}
}

func (t *grpcTransport) connForLocked(_ context.Context) (*grpc.ClientConn, error) {
	if t.conn != nil {
		return t.conn, nil
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
	return conn, nil
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

func runtimeRequirementsFromNodeProto(req *nodev1.RuntimeRequirements) RuntimeRequirements {
	if req == nil {
		return RuntimeRequirements{}
	}
	return RuntimeRequirements{
		NeedsGPU:           req.GetNeedsGpu(),
		NeedsManagedOCI:    req.GetNeedsManagedOci(),
		NeedsManagedOCIGPU: req.GetNeedsManagedOciGpu(),
		NeedsLlamaCPP:      req.GetNeedsLlamaCpp(),
		Tooling:            append([]string(nil), req.GetTooling()...),
		MinDiskGB:          req.GetMinDiskGb(),
		MinVRAMMB:          req.GetMinVramMb(),
		Jurisdiction:       req.GetJurisdiction(),
		TrustLevel:         req.GetTrustLevel(),
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
