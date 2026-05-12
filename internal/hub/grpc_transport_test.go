package hub

import (
	"context"
	"crypto/ed25519"
	"encoding/hex"
	"net"
	"testing"

	v7alphapb "github.com/Ryvion/node-agent/internal/genproto/v7alpha"
	"google.golang.org/grpc"
)

func TestClientHeartbeatUsesGRPCTransportWhenConfigured(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	fake := &fakeV7NodeTransportServer{}
	grpcServer := grpc.NewServer()
	v7alphapb.RegisterV7NodeTransportServiceServer(grpcServer, fake)
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
	resp, err := client.Heartbeat(context.Background(), Metrics{TimestampMs: 123, CPUUtil: 1, MemUtil: 2, GPUUtil: 3, PowerWatts: 4})
	if err != nil {
		t.Fatalf("Heartbeat() error = %v", err)
	}
	if resp.NodeID != "node-grpc" || resp.HubInstanceID != "hub-grpc" || resp.V7SnapshotUpserted == nil || !*resp.V7SnapshotUpserted {
		t.Fatalf("unexpected heartbeat response: %#v", resp)
	}
	if fake.heartbeat == nil {
		t.Fatal("gRPC heartbeat was not received")
	}
	if fake.heartbeat.GetPublicKeyHex() != hex.EncodeToString(pub) || len(fake.heartbeat.GetSignature()) == 0 {
		t.Fatalf("heartbeat did not carry signed node identity: %#v", fake.heartbeat)
	}
}

type fakeV7NodeTransportServer struct {
	v7alphapb.UnimplementedV7NodeTransportServiceServer
	heartbeat *v7alphapb.SendHeartbeatRequest
}

func (s *fakeV7NodeTransportServer) SendHeartbeat(_ context.Context, req *v7alphapb.SendHeartbeatRequest) (*v7alphapb.SendHeartbeatResponse, error) {
	s.heartbeat = req
	return &v7alphapb.SendHeartbeatResponse{
		Ok:                   true,
		NodeId:               "node-grpc",
		V7SnapshotUpserted:   true,
		SnapshotModelCount:   2,
		SnapshotBackendCount: 1,
		HasCapabilityProfile: true,
		HubInstanceId:        "hub-grpc",
	}, nil
}
