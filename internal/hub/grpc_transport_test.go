package hub

import (
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"net"
	"strings"
	"sync/atomic"
	"testing"

	experimentsv1 "github.com/Ryvion/ryvion-protocol/gen/go/ryvion/experiments/v1"
	nodev1 "github.com/Ryvion/ryvion-protocol/gen/go/ryvion/node/v1"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/types/known/structpb"
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
	if resp.NodeID != "node-gateway" || resp.HubInstanceID != "hub-gateway" || resp.V7SnapshotUpserted == nil || !*resp.V7SnapshotUpserted {
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
		Kind:         "inference",
		SpecJson:     `{"task":"inference"}`,
		ExecutorKind: "native",
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
		WorkgraphId:    "wg-gateway",
		Kind:           "v7_dashboard_inference",
		PayloadUrl:     "https://example.invalid/payload",
		PricePerUnit:   9,
		Units:          2,
		Image:          "ryvion-verifier-sglang",
		SpecJson:       `{"task":"v7_dashboard_inference"}`,
		JobPubkey:      "buyer-key",
		ExecutorKind:   "native",
		AssuranceClass: "standard",
		RuntimeRequirements: &nodev1.RuntimeRequirements{
			NeedsGpu:           true,
			NeedsRyvionRuntime: true,
			MinVramMb:          8192,
			Tooling:            []string{"llama.cpp"},
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
	if work == nil || work.JobID != "job-gateway" || work.WorkGraphID != "wg-gateway" || !work.RuntimeRequirements.NeedsGPU {
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

func TestClientSubmitDraftPacketBatchUsesNodeGatewayStreamWhenAvailable(t *testing.T) {
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
	decision, err := client.SubmitSpeculativeDraftPacketBatch(context.Background(), "win-gateway", []map[string]any{
		{"packet_id": "pkt-a", "candidate_tokens": []int{1, 2, 3}},
	})
	if err != nil {
		t.Fatalf("SubmitSpeculativeDraftPacketBatch() error = %v", err)
	}
	if decision.WindowID != "win-gateway" || decision.Accepted != 1 || len(decision.Decisions) != 1 || decision.Decisions[0].PacketID != "pkt-a" {
		t.Fatalf("unexpected draft batch decision: %#v", decision)
	}
	if fake.draftBatch == nil || fake.draftBatch.GetWindowId() != "win-gateway" || len(fake.draftBatch.GetPackets()) != 1 || len(fake.signature.GetSignature()) == 0 {
		t.Fatalf("draft batch did not carry signed node identity: batch=%#v signature=%#v", fake.draftBatch, fake.signature)
	}
}

func TestClientFetchLiveLabCommandUsesExperimentalGatewayStreamWhenAvailable(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	payload, err := structpb.NewStruct(map[string]any{
		"schema_version":            "ryvion.speculative_live_lab.session_command.v1",
		"run_id":                    "flab-gateway",
		"job_id":                    "job-draft",
		"role":                      "draft_worker",
		"command":                   "generate_draft_packets",
		"command_id":                "flab-gateway:draft:1:win-gateway",
		"workgraph_id":              "wg-gateway",
		"session_id":                "sess-gateway",
		"window_id":                 "win-gateway",
		"wave_index":                1,
		"parent_prefix_hash":        "sha256:prefix",
		"branch_count":              2,
		"horizon":                   4,
		"first_packet_timeout_ms":   500,
		"accepted_tokens_total":     0,
		"production_valid":          false,
		"billing_status":            "not_billable_live_lab",
		"read_only_experiment_path": true,
	})
	if err != nil {
		t.Fatalf("command payload: %v", err)
	}
	fake := &fakeExperimentalLiveLabGatewayServer{liveLabCommand: &experimentsv1.LiveLabCommand{
		SchemaVersion: "ryvion.speculative_live_lab.session_command.v1",
		RunId:         "flab-gateway",
		JobId:         "job-draft",
		Role:          "draft_worker",
		Command:       "generate_draft_packets",
		CommandId:     "flab-gateway:draft:1:win-gateway",
		Payload:       payload,
	}}
	grpcServer := grpc.NewServer()
	experimentsv1.RegisterExperimentalLiveLabGatewayServer(grpcServer, fake)
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
	command, err := client.FetchSpeculativeLiveLabDraftCommand(context.Background(), "flab-gateway", "job-draft")
	if err != nil {
		t.Fatalf("FetchSpeculativeLiveLabDraftCommand() error = %v", err)
	}
	if command.Command != "generate_draft_packets" ||
		command.RunID != "flab-gateway" ||
		command.JobID != "job-draft" ||
		command.WindowID != "win-gateway" ||
		command.BranchCount != 2 ||
		command.Prompt != "" {
		t.Fatalf("unexpected live lab command: %#v", command)
	}
	if fake.liveLabCommandRequest == nil ||
		fake.liveLabCommandRequest.GetRole() != "draft_worker" ||
		fake.liveLabCommandRequest.GetRunId() != "flab-gateway" ||
		len(fake.signature.GetSignature()) == 0 {
		t.Fatalf("live lab command request missing signed gateway payload: request=%#v signature=%#v", fake.liveLabCommandRequest, fake.signature)
	}
}

func TestClientSubmitLiveLabVerifierResultUsesExperimentalGatewayStreamWhenAvailable(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatalf("GenerateKey() error = %v", err)
	}
	fake := &fakeExperimentalLiveLabGatewayServer{liveLabVerifierResultStatus: "accepted"}
	grpcServer := grpc.NewServer()
	experimentsv1.RegisterExperimentalLiveLabGatewayServer(grpcServer, fake)
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
	err = client.SubmitSpeculativeLiveLabVerifierResult(context.Background(), "flab-gateway", SpeculativeLiveLabVerifierResult{
		JobID:              "job-verify",
		WindowID:           "win-gateway",
		WaveIndex:          1,
		AcceptedLen:        3,
		TreeCID:            "cid-tree",
		DurationMs:         12,
		AcceptedText:       "private accepted text",
		AcceptedTextPublic: true,
		EOS:                true,
		StopReason:         "eos",
		ProbeSummary:       map[string]any{"backend": "sglang"},
	})
	if err != nil {
		t.Fatalf("SubmitSpeculativeLiveLabVerifierResult() error = %v", err)
	}
	if fake.liveLabVerifierResult == nil ||
		fake.liveLabVerifierResult.GetRunId() != "flab-gateway" ||
		fake.liveLabVerifierResult.GetJobId() != "job-verify" ||
		fake.liveLabVerifierResult.GetAcceptedLen() != 3 ||
		fake.liveLabVerifierResult.GetProbeSummary().AsMap()["backend"] != "sglang" {
		t.Fatalf("unexpected live lab verifier result: %#v", fake.liveLabVerifierResult)
	}
	sum := sha256.Sum256([]byte("private accepted text"))
	if got := fake.liveLabVerifierResult.GetAcceptedTextHash(); got != "sha256:"+hex.EncodeToString(sum[:]) {
		t.Fatalf("accepted text hash = %q, want sha256 hash", got)
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
	draftBatch      *nodev1.DraftPacketBatch
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
						Ok:                   true,
						NodeId:               "node-reuse",
						V7SnapshotUpserted:   true,
						SnapshotModelCount:   1,
						SnapshotBackendCount: 1,
						HasCapabilityProfile: true,
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
					Ok:                   true,
					NodeId:               "node-gateway",
					V7SnapshotUpserted:   true,
					SnapshotModelCount:   3,
					SnapshotBackendCount: 2,
					HasCapabilityProfile: true,
					HubInstanceId:        "hub-gateway",
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
	if draftBatch := msg.GetDraftPacketBatch(); draftBatch != nil {
		s.draftBatch = draftBatch
		packetID := ""
		if len(draftBatch.GetPackets()) > 0 && draftBatch.GetPackets()[0] != nil {
			packetID = draftBatch.GetPackets()[0].GetPacketId()
		}
		return stream.Send(&nodev1.HubToNode{
			MessageId:       "ack_" + msg.GetMessageId(),
			CreatedAtUnixMs: msg.GetCreatedAtUnixMs(),
			Payload: &nodev1.HubToNode_DraftPacketBatchAck{
				DraftPacketBatchAck: &nodev1.DraftPacketBatchAck{
					SchemaVersion: "ryvion.speculative.draft_packet_batch_decision.v1",
					WindowId:      draftBatch.GetWindowId(),
					Attempted:     1,
					Accepted:      1,
					Decisions: []*nodev1.DraftPacketDecision{{
						PacketId: packetID,
						Accepted: true,
						Reason:   "accepted",
					}},
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

type fakeExperimentalLiveLabGatewayServer struct {
	experimentsv1.UnimplementedExperimentalLiveLabGatewayServer
	liveLabCommandRequest       *experimentsv1.LiveLabCommandRequest
	liveLabCommand              *experimentsv1.LiveLabCommand
	liveLabVerifierResult       *experimentsv1.LiveLabVerifierResult
	liveLabVerifierResultStatus string
	signature                   *nodev1.Signature
}

func (s *fakeExperimentalLiveLabGatewayServer) Connect(stream experimentsv1.ExperimentalLiveLabGateway_ConnectServer) error {
	msg, err := stream.Recv()
	if err != nil {
		return err
	}
	s.signature = msg.GetSignature()
	if request := msg.GetCommandRequest(); request != nil {
		s.liveLabCommandRequest = request
		command := s.liveLabCommand
		if command == nil {
			command = &experimentsv1.LiveLabCommand{
				SchemaVersion: "ryvion.speculative_live_lab.session_command.v1",
				RunId:         request.GetRunId(),
				JobId:         request.GetJobId(),
				Role:          request.GetRole(),
				Command:       "wait",
				Reason:        "test_default",
			}
		}
		return stream.Send(&experimentsv1.LiveLabHubToNode{
			MessageId:       "ack_" + msg.GetMessageId(),
			CreatedAtUnixMs: msg.GetCreatedAtUnixMs(),
			Payload: &experimentsv1.LiveLabHubToNode_Command{
				Command: command,
			},
		})
	}
	if result := msg.GetVerifierResult(); result != nil {
		s.liveLabVerifierResult = result
		status := s.liveLabVerifierResultStatus
		if strings.TrimSpace(status) == "" {
			status = "accepted"
		}
		return stream.Send(&experimentsv1.LiveLabHubToNode{
			MessageId:       "ack_" + msg.GetMessageId(),
			CreatedAtUnixMs: msg.GetCreatedAtUnixMs(),
			Payload: &experimentsv1.LiveLabHubToNode_VerifierResultAck{
				VerifierResultAck: &experimentsv1.LiveLabVerifierResultAck{
					SchemaVersion: "ryvion.speculative_live_lab.verifier_result_ack.v1",
					Status:        status,
					RunId:         result.GetRunId(),
					WindowId:      result.GetWindowId(),
				},
			},
		})
	}
	return stream.Send(&experimentsv1.LiveLabHubToNode{
		MessageId:       "ack_" + msg.GetMessageId(),
		CreatedAtUnixMs: msg.GetCreatedAtUnixMs(),
	})
}
