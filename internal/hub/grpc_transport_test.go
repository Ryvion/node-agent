package hub

import (
	"context"
	"crypto/ed25519"
	"encoding/hex"
	"net"
	"strings"
	"sync/atomic"
	"testing"

	nodev1 "github.com/Ryvion/ryvion-protocol/gen/go/ryvion/node/v1"
	"google.golang.org/grpc"
)

func TestClientHeartbeatUsesNodeGatewayStreamWhenAvailable(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	fake := &fakeNodeGatewayServer{}
	grpcServer := grpc.NewServer()
	nodev1.RegisterNodeGatewayServer(grpcServer, fake)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	go func() {
		_ = grpcServer.Serve(listener)
	}()
	t.Cleanup(func() {
		grpcServer.Stop()
		_ = listener.Close()
	})

	client := New("http://127.0.0.1:1", pub, priv, WithGRPCTransport(listener.Addr().String(), "grpc", true))
	resp, err := client.Heartbeat(context.Background(), Metrics{TimestampMs: 456, CPUUtil: 5, MemUtil: 6, GPUUtil: 7, PowerWatts: 8})
	if err != nil {
		t.Fatalf("Heartbeat() error = %v", err)
	}
	if resp.NodeID != "node-gateway" || resp.HubInstanceID != "hub-gateway" || resp.CapabilityProfileUpserted == nil || !*resp.CapabilityProfileUpserted {
		t.Fatalf("unexpected heartbeat response: %#v", resp)
	}
	if fake.heartbeat == nil {
		t.Fatal("NodeGateway heartbeat was not received")
	}
	if fake.heartbeat.GetPublicKeyHex() != hex.EncodeToString(pub) || len(fake.signature.GetSignature()) == 0 {
		t.Fatalf("heartbeat did not carry signed node identity: heartbeat=%#v signature=%#v", fake.heartbeat, fake.signature)
	}
}

func TestClientReusesNodeGatewayStream(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	fake := &fakePersistentNodeGatewayServer{work: &nodev1.WorkAssignment{
		JobId:        "job-reuse",
		Kind:         "llama_cpp_inference",
		Image:        "llama_cpp",
		SpecJson:     `{"task":"llama_cpp_inference","prompt":"hi"}`,
		ExecutorKind: "llama_cpp",
	}}
	grpcServer := grpc.NewServer()
	nodev1.RegisterNodeGatewayServer(grpcServer, fake)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	go func() {
		_ = grpcServer.Serve(listener)
	}()
	t.Cleanup(func() {
		grpcServer.Stop()
		_ = listener.Close()
	})

	client := New("http://127.0.0.1:1", pub, priv, WithGRPCTransport(listener.Addr().String(), "grpc", true))
	defer client.Close()
	if _, err := client.Heartbeat(context.Background(), Metrics{TimestampMs: 457, CPUUtil: 1}); err != nil {
		t.Fatalf("Heartbeat() error = %v", err)
	}
	work, err := client.FetchWork(context.Background())
	if err != nil {
		t.Fatalf("FetchWork() error = %v", err)
	}
	if work == nil || work.JobID != "job-reuse" {
		t.Fatalf("unexpected work assignment: %#v", work)
	}
	if got := fake.connectCount.Load(); got != 1 {
		t.Fatalf("NodeGateway Connect calls = %d, want one persistent stream", got)
	}
}

func TestClientFetchWorkUsesNodeGatewayStreamWhenAvailable(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	fake := &fakeNodeGatewayServer{work: &nodev1.WorkAssignment{
		JobId:          "job-gateway",
		AbortScopeId:   "scope-gateway",
		WorkgraphId:    "wg-gateway",
		Kind:           "llama_cpp_inference",
		PayloadUrl:     "https://example.invalid/payload",
		Units:          2,
		Image:          "llama_cpp",
		SpecJson:       `{"task":"llama_cpp_inference","prompt":"hi"}`,
		JobPubkey:      "buyer-key",
		ExecutorKind:   "llama_cpp",
		AssuranceClass: "standard",
		RuntimeRequirements: &nodev1.RuntimeRequirements{
			NeedsGpu:        true,
			NeedsManagedOci: true,
			NeedsLlamaCpp:   true,
			MinVramMb:       8192,
			Tooling:         []string{"llama_cpp"},
		},
	}}
	grpcServer := grpc.NewServer()
	nodev1.RegisterNodeGatewayServer(grpcServer, fake)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	go func() {
		_ = grpcServer.Serve(listener)
	}()
	t.Cleanup(func() {
		grpcServer.Stop()
		_ = listener.Close()
	})

	client := New("http://127.0.0.1:1", pub, priv, WithGRPCTransport(listener.Addr().String(), "grpc", true))
	work, err := client.FetchWork(context.Background())
	if err != nil {
		t.Fatalf("FetchWork() error = %v", err)
	}
	if work == nil || work.JobID != "job-gateway" || work.WorkScopeID != "scope-gateway" || !work.RuntimeRequirements.NeedsGPU || !work.RuntimeRequirements.NeedsLlamaCPP {
		t.Fatalf("unexpected work assignment: %#v", work)
	}
	if fake.workLease == nil || fake.workLease.GetPublicKeyHex() != hex.EncodeToString(pub) || len(fake.signature.GetSignature()) == 0 {
		t.Fatalf("work lease did not carry signed node identity: lease=%#v signature=%#v", fake.workLease, fake.signature)
	}
}

func TestClientFetchWorkNodeGatewayNoWork(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	fake := &fakeNodeGatewayServer{}
	grpcServer := grpc.NewServer()
	nodev1.RegisterNodeGatewayServer(grpcServer, fake)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	go func() {
		_ = grpcServer.Serve(listener)
	}()
	t.Cleanup(func() {
		grpcServer.Stop()
		_ = listener.Close()
	})

	client := New("http://127.0.0.1:1", pub, priv, WithGRPCTransport(listener.Addr().String(), "grpc", true))
	work, err := client.FetchWork(context.Background())
	if err != nil {
		t.Fatalf("FetchWork() error = %v", err)
	}
	if work != nil {
		t.Fatalf("FetchWork() = %#v, want nil", work)
	}
}

func TestClientSubmitReceiptUsesNodeGatewayStreamWhenAvailable(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	fake := &fakeNodeGatewayServer{receiptAccepted: true}
	grpcServer := grpc.NewServer()
	nodev1.RegisterNodeGatewayServer(grpcServer, fake)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	go func() {
		_ = grpcServer.Serve(listener)
	}()
	t.Cleanup(func() {
		grpcServer.Stop()
		_ = listener.Close()
	})

	client := New("http://127.0.0.1:1", pub, priv, WithGRPCTransport(listener.Addr().String(), "grpc", true))
	err = client.SubmitReceipt(context.Background(), Receipt{
		JobID:         "job-gateway",
		ResultHashHex: "abcd",
		MeteringUnits: 3,
		Metadata:      map[string]any{"finish_reason": "stop"},
	})
	if err != nil {
		t.Fatalf("SubmitReceipt() error = %v", err)
	}
	if fake.receipt == nil || fake.receipt.GetJobId() != "job-gateway" || fake.receipt.GetPublicKeyHex() != hex.EncodeToString(pub) || len(fake.signature.GetSignature()) == 0 {
		t.Fatalf("receipt did not carry signed node identity: receipt=%#v signature=%#v", fake.receipt, fake.signature)
	}
}

func TestClientSubmitReceiptNodeGatewayRejectsAck(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	fake := &fakeNodeGatewayServer{receiptAccepted: false, receiptReason: "bad_hash"}
	grpcServer := grpc.NewServer()
	nodev1.RegisterNodeGatewayServer(grpcServer, fake)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	go func() {
		_ = grpcServer.Serve(listener)
	}()
	t.Cleanup(func() {
		grpcServer.Stop()
		_ = listener.Close()
	})

	client := New("http://127.0.0.1:1", pub, priv, WithGRPCTransport(listener.Addr().String(), "grpc", true))
	err = client.SubmitReceipt(context.Background(), Receipt{JobID: "job-gateway", ResultHashHex: "abcd", MeteringUnits: 3})
	if err == nil || !strings.Contains(err.Error(), "bad_hash") {
		t.Fatalf("SubmitReceipt() error = %v, want bad_hash rejection", err)
	}
}

type fakeNodeGatewayServer struct {
	nodev1.UnimplementedNodeGatewayServer
	heartbeat       *nodev1.NodeHeartbeat
	workLease       *nodev1.WorkLeaseRequest
	work            *nodev1.WorkAssignment
	receipt         *nodev1.WorkReceipt
	receiptAccepted bool
	receiptReason   string
	signature       *nodev1.Signature
}

type fakePersistentNodeGatewayServer struct {
	nodev1.UnimplementedNodeGatewayServer
	connectCount atomic.Int32
	work         *nodev1.WorkAssignment
}

func (s *fakePersistentNodeGatewayServer) Connect(stream nodev1.NodeGateway_ConnectServer) error {
	s.connectCount.Add(1)
	for {
		msg, err := stream.Recv()
		if err != nil {
			return err
		}
		if msg.GetHeartbeat() != nil {
			if err := stream.Send(&nodev1.HubToNode{
				MessageId:       "ack_" + msg.GetMessageId(),
				CreatedAtUnixMs: msg.GetCreatedAtUnixMs(),
				Payload: &nodev1.HubToNode_HeartbeatAck{
					HeartbeatAck: &nodev1.NodeHeartbeatAck{
						Ok:                        true,
						NodeId:                    "node-reuse",
						CapabilityProfileUpserted: true,
						ProfileRuntimeCount:       1,
						ProfileBackendCount:       1,
						HasCapabilityProfile:      true,
					},
				},
			}); err != nil {
				return err
			}
			continue
		}
		if msg.GetWorkLeaseRequest() != nil {
			if err := stream.Send(&nodev1.HubToNode{
				MessageId:       "ack_" + msg.GetMessageId(),
				CreatedAtUnixMs: msg.GetCreatedAtUnixMs(),
				Payload: &nodev1.HubToNode_WorkLeaseAck{
					WorkLeaseAck: &nodev1.WorkLeaseAck{
						HasWork:    s.work != nil,
						Assignment: s.work,
					},
				},
			}); err != nil {
				return err
			}
		}
	}
}

func (s *fakeNodeGatewayServer) Connect(stream nodev1.NodeGateway_ConnectServer) error {
	msg, err := stream.Recv()
	if err != nil {
		return err
	}
	s.signature = msg.GetSignature()
	if hb := msg.GetHeartbeat(); hb != nil {
		s.heartbeat = hb
		return stream.Send(&nodev1.HubToNode{
			MessageId:       "ack_" + msg.GetMessageId(),
			CreatedAtUnixMs: msg.GetCreatedAtUnixMs(),
			Payload: &nodev1.HubToNode_HeartbeatAck{
				HeartbeatAck: &nodev1.NodeHeartbeatAck{
					Ok:                        true,
					NodeId:                    "node-gateway",
					CapabilityProfileUpserted: true,
					ProfileRuntimeCount:       3,
					ProfileBackendCount:       2,
					HasCapabilityProfile:      true,
					HubInstanceId:             "hub-gateway",
				},
			},
		})
	}
	if lease := msg.GetWorkLeaseRequest(); lease != nil {
		s.workLease = lease
		ack := &nodev1.WorkLeaseAck{}
		if s.work != nil {
			ack.HasWork = true
			ack.Assignment = s.work
		}
		return stream.Send(&nodev1.HubToNode{
			MessageId:       "ack_" + msg.GetMessageId(),
			CreatedAtUnixMs: msg.GetCreatedAtUnixMs(),
			Payload: &nodev1.HubToNode_WorkLeaseAck{
				WorkLeaseAck: ack,
			},
		})
	}
	if receipt := msg.GetReceipt(); receipt != nil {
		s.receipt = receipt
		return stream.Send(&nodev1.HubToNode{
			MessageId:       "ack_" + msg.GetMessageId(),
			CreatedAtUnixMs: msg.GetCreatedAtUnixMs(),
			Payload: &nodev1.HubToNode_ReceiptAck{
				ReceiptAck: &nodev1.WorkReceiptAck{
					Accepted:  s.receiptAccepted,
					Status:    "received",
					ReceiptId: "receipt-gateway",
					Reason:    s.receiptReason,
				},
			},
		})
	}
	return stream.Send(&nodev1.HubToNode{
		MessageId:       "ack_" + msg.GetMessageId(),
		CreatedAtUnixMs: msg.GetCreatedAtUnixMs(),
		Payload: &nodev1.HubToNode_Ping{
			Ping: &nodev1.Ping{Nonce: msg.GetMessageId()},
		},
	})
}
