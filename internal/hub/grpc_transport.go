package hub

import (
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	nodev1 "github.com/Ryvion/ryvion-protocol/gen/go/ryvion/node/v1"
	v7alphapb "github.com/Ryvion/ryvion-protocol/gen/go/ryvion/v7alpha"
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

	mu            sync.Mutex
	conn          *grpc.ClientConn
	client        v7alphapb.V7NodeTransportServiceClient
	gatewayClient nodev1.NodeGatewayClient
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

func nodeGatewayCompatFallback(err error) bool {
	return status.Code(err) == codes.Unimplemented || errors.Is(err, io.EOF)
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
	if resp, err := c.heartbeatNodeGatewayGRPC(ctx, body); err == nil {
		return resp, nil
	} else if !nodeGatewayCompatFallback(err) {
		return HeartbeatResponse{}, err
	}
	return c.heartbeatUnaryGRPC(ctx, body)
}

func (c *Client) heartbeatNodeGatewayGRPC(ctx context.Context, body heartbeatRequest) (HeartbeatResponse, error) {
	client, err := c.nodeGatewayClient(ctx)
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
	var networkStruct *structpb.Struct
	if body.NetworkProfile != nil {
		networkStruct, err = structFromJSONValue(body.NetworkProfile)
		if err != nil {
			return HeartbeatResponse{}, err
		}
	}
	ctx, cancel := context.WithTimeout(c.grpcMetadataContext(ctx), 10*time.Second)
	defer cancel()
	stream, err := client.Connect(ctx)
	if err != nil {
		return HeartbeatResponse{}, err
	}
	if err := stream.Send(&nodev1.NodeToHub{
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
				CapabilityPayload: v7Struct,
				NetworkProfile:    networkStruct,
			},
		},
		Signature: &nodev1.Signature{
			KeyId:     body.PublicKeyHex,
			Algorithm: "ed25519",
			Signature: append([]byte(nil), body.Signature...),
		},
	}); err != nil {
		return HeartbeatResponse{}, err
	}
	resp, err := stream.Recv()
	if closeErr := stream.CloseSend(); err == nil && closeErr != nil {
		err = closeErr
	}
	if err != nil {
		return HeartbeatResponse{}, err
	}
	ack := resp.GetHeartbeatAck()
	if ack == nil {
		return HeartbeatResponse{}, status.Error(codes.Internal, "heartbeat ack missing")
	}
	upserted := ack.GetV7SnapshotUpserted()
	return HeartbeatResponse{
		LatestVersion:        ack.GetLatestVersion(),
		NodeID:               ack.GetNodeId(),
		CountryCode:          ack.GetCountryCode(),
		LocationApproved:     ack.GetLocationApproved(),
		SovereignVerified:    ack.GetSovereignVerified(),
		VerificationSource:   ack.GetVerificationSource(),
		TrustReason:          ack.GetTrustReason(),
		V7SnapshotUpserted:   &upserted,
		SnapshotModelCount:   int(ack.GetSnapshotModelCount()),
		SnapshotBackendCount: int(ack.GetSnapshotBackendCount()),
		HasCapabilityProfile: ack.GetHasCapabilityProfile(),
		HubInstanceID:        ack.GetHubInstanceId(),
	}, nil
}

func (c *Client) heartbeatUnaryGRPC(ctx context.Context, body heartbeatRequest) (HeartbeatResponse, error) {
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
	if work, err := c.fetchWorkNodeGatewayGRPC(ctx, pubHex, ts, signature, longPoll); err == nil {
		return work, nil
	} else if !nodeGatewayCompatFallback(err) {
		return nil, err
	}
	return c.fetchWorkUnaryGRPC(ctx, pubHex, ts, signature, longPoll)
}

func (c *Client) fetchWorkNodeGatewayGRPC(ctx context.Context, pubHex string, ts int64, signature []byte, longPoll bool) (*WorkAssignment, error) {
	client, err := c.nodeGatewayClient(ctx)
	if err != nil {
		return nil, err
	}
	timeout := 10 * time.Second
	if longPoll {
		timeout = 35 * time.Second
	}
	ctx, cancel := context.WithTimeout(c.grpcMetadataContext(ctx), timeout)
	defer cancel()
	stream, err := client.Connect(ctx)
	if err != nil {
		return nil, err
	}
	if err := stream.Send(&nodev1.NodeToHub{
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
	}); err != nil {
		return nil, err
	}
	resp, err := stream.Recv()
	if closeErr := stream.CloseSend(); err == nil && closeErr != nil {
		err = closeErr
	}
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
		WorkGraphID:         assignment.GetWorkgraphId(),
		JobPubkey:           assignment.GetJobPubkey(),
		Kind:                assignment.GetKind(),
		PayloadURL:          assignment.GetPayloadUrl(),
		PricePerUnit:        assignment.GetPricePerUnit(),
		Units:               assignment.GetUnits(),
		Image:               assignment.GetImage(),
		SpecJSON:            assignment.GetSpecJson(),
		ExecutorKind:        assignment.GetExecutorKind(),
		AssuranceClass:      assignment.GetAssuranceClass(),
		RuntimeRequirements: runtimeRequirementsFromNodeProto(assignment.GetRuntimeRequirements()),
	}, nil
}

func (c *Client) fetchWorkUnaryGRPC(ctx context.Context, pubHex string, ts int64, signature []byte, longPoll bool) (*WorkAssignment, error) {
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
	if err := c.submitReceiptNodeGatewayGRPC(ctx, body); err == nil {
		return nil
	} else if !nodeGatewayCompatFallback(err) {
		return err
	}
	return c.submitReceiptUnaryGRPC(ctx, body)
}

func (c *Client) submitReceiptNodeGatewayGRPC(ctx context.Context, body receiptRequest) error {
	client, err := c.nodeGatewayClient(ctx)
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
	stream, err := client.Connect(ctx)
	if err != nil {
		return err
	}
	if err := stream.Send(&nodev1.NodeToHub{
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
	}); err != nil {
		return err
	}
	resp, err := stream.Recv()
	if closeErr := stream.CloseSend(); err == nil && closeErr != nil {
		err = closeErr
	}
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

func (c *Client) submitReceiptUnaryGRPC(ctx context.Context, body receiptRequest) error {
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

func (c *Client) submitDraftPacketBatchGRPC(ctx context.Context, windowID string, packets []map[string]any) (DraftPacketBatchDecision, error) {
	client, err := c.nodeGatewayClient(ctx)
	if err != nil {
		return DraftPacketBatchDecision{}, err
	}
	protoPackets := make([]*structpb.Struct, 0, len(packets))
	for _, packet := range packets {
		packetStruct, err := structFromJSONValue(packet)
		if err != nil {
			return DraftPacketBatchDecision{}, err
		}
		protoPackets = append(protoPackets, packetStruct)
	}
	ts := time.Now().UnixMilli()
	pubHex := c.pubHex()
	ctx, cancel := context.WithTimeout(c.grpcMetadataContext(ctx), 30*time.Second)
	defer cancel()
	stream, err := client.Connect(ctx)
	if err != nil {
		return DraftPacketBatchDecision{}, err
	}
	if err := stream.Send(&nodev1.NodeToHub{
		NodeId:          pubHex,
		MessageId:       fmt.Sprintf("draft-packets-%s-%d", windowID, ts),
		CreatedAtUnixMs: ts,
		Payload: &nodev1.NodeToHub_DraftPacketBatch{
			DraftPacketBatch: &nodev1.DraftPacketBatch{
				WindowId: windowID,
				Packets:  protoPackets,
			},
		},
		Signature: &nodev1.Signature{
			KeyId:     pubHex,
			Algorithm: "ed25519-node-auth",
			Signature: c.sign("node_auth", pubHex, strconv.FormatInt(ts, 10)),
		},
	}); err != nil {
		return DraftPacketBatchDecision{}, err
	}
	resp, err := stream.Recv()
	if closeErr := stream.CloseSend(); err == nil && closeErr != nil {
		err = closeErr
	}
	if err != nil {
		return DraftPacketBatchDecision{}, err
	}
	ack := resp.GetDraftPacketBatchAck()
	if ack == nil {
		return DraftPacketBatchDecision{}, status.Error(codes.Internal, "draft packet batch ack missing")
	}
	decisions := make([]DraftPacketDecision, 0, len(ack.GetDecisions()))
	for _, decision := range ack.GetDecisions() {
		if decision == nil {
			continue
		}
		decisions = append(decisions, DraftPacketDecision{
			PacketID: decision.GetPacketId(),
			Accepted: decision.GetAccepted(),
			Reason:   decision.GetReason(),
		})
	}
	return DraftPacketBatchDecision{
		SchemaVersion: ack.GetSchemaVersion(),
		WindowID:      ack.GetWindowId(),
		Attempted:     int(ack.GetAttempted()),
		Accepted:      int(ack.GetAccepted()),
		Rejected:      int(ack.GetRejected()),
		Decisions:     decisions,
	}, nil
}

func (c *Client) fetchForesightLiveLabSessionCommandGRPC(ctx context.Context, runID string, jobID string, role string) (ForesightLiveLabSessionCommand, error) {
	client, err := c.nodeGatewayClient(ctx)
	if err != nil {
		return ForesightLiveLabSessionCommand{}, err
	}
	ts := time.Now().UnixMilli()
	pubHex := c.pubHex()
	ctx, cancel := context.WithTimeout(c.grpcMetadataContext(ctx), 10*time.Second)
	defer cancel()
	stream, err := client.Connect(ctx)
	if err != nil {
		return ForesightLiveLabSessionCommand{}, err
	}
	if err := stream.Send(&nodev1.NodeToHub{
		NodeId:          pubHex,
		MessageId:       fmt.Sprintf("live-lab-command-%s-%d", runID, ts),
		CreatedAtUnixMs: ts,
		Payload: &nodev1.NodeToHub_LiveLabCommandRequest{
			LiveLabCommandRequest: &nodev1.LiveLabCommandRequest{
				RunId: runID,
				JobId: strings.TrimSpace(jobID),
				Role:  strings.TrimSpace(role),
			},
		},
		Signature: &nodev1.Signature{
			KeyId:     pubHex,
			Algorithm: "ed25519-node-auth",
			Signature: c.sign("node_auth", pubHex, strconv.FormatInt(ts, 10)),
		},
	}); err != nil {
		return ForesightLiveLabSessionCommand{}, err
	}
	resp, err := stream.Recv()
	if closeErr := stream.CloseSend(); err == nil && closeErr != nil {
		err = closeErr
	}
	if err != nil {
		return ForesightLiveLabSessionCommand{}, err
	}
	command := resp.GetLiveLabCommand()
	if command == nil {
		return ForesightLiveLabSessionCommand{}, status.Error(codes.Internal, "live lab command missing")
	}
	payload := map[string]any{}
	if command.GetPayload() != nil {
		payload = command.GetPayload().AsMap()
	}
	payload["schema_version"] = command.GetSchemaVersion()
	payload["run_id"] = command.GetRunId()
	payload["job_id"] = command.GetJobId()
	payload["role"] = command.GetRole()
	payload["command"] = command.GetCommand()
	payload["command_id"] = command.GetCommandId()
	payload["reason"] = command.GetReason()
	var out ForesightLiveLabSessionCommand
	raw, err := json.Marshal(payload)
	if err != nil {
		return ForesightLiveLabSessionCommand{}, err
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return ForesightLiveLabSessionCommand{}, err
	}
	return out, nil
}

func (c *Client) submitForesightLiveLabVerifierResultGRPC(ctx context.Context, runID string, result ForesightLiveLabVerifierResult) error {
	client, err := c.nodeGatewayClient(ctx)
	if err != nil {
		return err
	}
	var probeSummary *structpb.Struct
	if result.ProbeSummary != nil {
		probeSummary, err = structFromJSONValue(result.ProbeSummary)
		if err != nil {
			return err
		}
	}
	acceptedTextHash := ""
	if strings.TrimSpace(result.AcceptedText) != "" {
		sum := sha256.Sum256([]byte(result.AcceptedText))
		acceptedTextHash = "sha256:" + hex.EncodeToString(sum[:])
	}
	ts := time.Now().UnixMilli()
	pubHex := c.pubHex()
	ctx, cancel := context.WithTimeout(c.grpcMetadataContext(ctx), 10*time.Second)
	defer cancel()
	stream, err := client.Connect(ctx)
	if err != nil {
		return err
	}
	if err := stream.Send(&nodev1.NodeToHub{
		NodeId:          pubHex,
		MessageId:       fmt.Sprintf("live-lab-verifier-result-%s-%d", runID, ts),
		CreatedAtUnixMs: ts,
		Payload: &nodev1.NodeToHub_LiveLabVerifierResult{
			LiveLabVerifierResult: &nodev1.LiveLabVerifierResult{
				RunId:              runID,
				JobId:              result.JobID,
				WindowId:           result.WindowID,
				WaveIndex:          nonNegativeUint32(result.WaveIndex),
				AcceptedLen:        nonNegativeUint32(result.AcceptedLen),
				TreeCid:            result.TreeCID,
				DurationMs:         result.DurationMs,
				AcceptedTextHash:   acceptedTextHash,
				AcceptedTextPublic: result.AcceptedTextPublic,
				Eos:                result.EOS,
				StopReason:         result.StopReason,
				ProbeSummary:       probeSummary,
			},
		},
		Signature: &nodev1.Signature{
			KeyId:     pubHex,
			Algorithm: "ed25519-node-auth",
			Signature: c.sign("node_auth", pubHex, strconv.FormatInt(ts, 10)),
		},
	}); err != nil {
		return err
	}
	resp, err := stream.Recv()
	if closeErr := stream.CloseSend(); err == nil && closeErr != nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	ack := resp.GetLiveLabVerifierResultAck()
	if ack == nil {
		return status.Error(codes.Internal, "live lab verifier result ack missing")
	}
	if statusText := strings.TrimSpace(ack.GetStatus()); statusText != "" && statusText != "accepted" && statusText != "duplicate" && statusText != "ok" {
		return fmt.Errorf("live lab verifier result rejected: %s", statusText)
	}
	return nil
}

func nonNegativeUint32(value int) uint32 {
	if value <= 0 {
		return 0
	}
	return uint32(value)
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

func (c *Client) nodeGatewayClient(ctx context.Context) (nodev1.NodeGatewayClient, error) {
	if c == nil || c.grpc == nil || !c.grpc.enabled() {
		return nil, status.Error(codes.Unavailable, "gRPC transport disabled")
	}
	return c.grpc.gatewayClientFor(ctx)
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
	t.gatewayClient = nil
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
	conn, err := t.connForLocked(ctx)
	if err != nil {
		return nil, err
	}
	t.client = v7alphapb.NewV7NodeTransportServiceClient(conn)
	return t.client, nil
}

func (t *grpcTransport) gatewayClientFor(ctx context.Context) (nodev1.NodeGatewayClient, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.gatewayClient != nil {
		return t.gatewayClient, nil
	}
	conn, err := t.connForLocked(ctx)
	if err != nil {
		return nil, err
	}
	t.gatewayClient = nodev1.NewNodeGatewayClient(conn)
	return t.gatewayClient, nil
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

func runtimeRequirementsFromNodeProto(req *nodev1.RuntimeRequirements) RuntimeRequirements {
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
