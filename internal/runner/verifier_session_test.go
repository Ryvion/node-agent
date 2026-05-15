package runner

import (
	"bufio"
	"context"
	"encoding/json"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

func TestExecuteVerifierSessionRPCRunsPersistentCommitRollbackSequence(t *testing.T) {
	if testing.Short() {
		t.Skip("unix socket test skipped in short mode")
	}
	tmp, err := os.MkdirTemp("/tmp", "ryv_verifier_rpc_*")
	if err != nil {
		t.Fatalf("temp dir: %v", err)
	}
	defer os.RemoveAll(tmp)
	socketPath := filepath.Join(tmp, "verifier.sock")
	ln, err := net.Listen("unix", socketPath)
	if err != nil {
		t.Fatalf("listen unix socket: %v", err)
	}
	defer ln.Close()

	methods := make(chan string, 8)
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go func(conn net.Conn) {
				defer conn.Close()
				scanner := bufio.NewScanner(conn)
				for scanner.Scan() {
					var req verifierRPCRequest
					if err := json.Unmarshal(scanner.Bytes(), &req); err != nil {
						return
					}
					methods <- req.Method
					result := map[string]any{"status": req.Method + "_ok"}
					if req.Method == "verify_tree" {
						result["accepted_token_receipt"] = map[string]any{
							"accepted_len":        2,
							"tree_cid":            "sha256:tree",
							"rollback_branch_ids": []any{"br-reject"},
						}
						result["probe_summary"] = map[string]any{"confidence_bps": 9100}
					}
					_ = json.NewEncoder(conn).Encode(map[string]any{"jsonrpc": "2.0", "id": req.ID, "result": result})
				}
			}(conn)
		}
	}()

	spec := `{
		"schema_version":"ryvion.verify_tree_request.v1",
		"method":"verify_tree",
		"session":{"session_id":"sess-rpc","workgraph_id":"wg-rpc","model_hash":"sha256:model","prefix_hash":"sha256:prefix"},
		"tree":{"tree_cid":"sha256:tree","window_id":"win-rpc","branches":[{"branch_id":"br-ok","candidate_tokens":[1,2]}]}
	}`
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if _, err := ExecuteVerifierSessionRPC(ctx, socketPath, spec); err != nil {
		t.Fatalf("ExecuteVerifierSessionRPC() error = %v", err)
	}

	got := drainMethods(methods)
	want := []string{"start_session", "prefill", "verify_tree", "commit", "rollback", "close_session"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("methods = %v, want %v", got, want)
	}
}

func TestReadDraftPacketsSanitizesUnsafePayload(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "draft_packets.json")
	if err := os.WriteFile(path, []byte(`[
		{"packet_id":"pkt-1","window_id":"win","role_id":"draft","parent_prefix_hash":"sha256:prefix","candidate_tokens":[1,2],"model_hash":"sha256:model","signature":"sha256:sig","candidate_text_preview":"secret","prompt":"secret prompt"}
	]`), 0o644); err != nil {
		t.Fatalf("write draft packets: %v", err)
	}
	packets := readDraftPackets(path)
	if len(packets) != 1 {
		t.Fatalf("packets len = %d, want 1", len(packets))
	}
	if _, ok := packets[0]["candidate_text_preview"]; ok {
		t.Fatalf("unsafe candidate preview leaked: %#v", packets[0])
	}
	if _, ok := packets[0]["prompt"]; ok {
		t.Fatalf("unsafe prompt leaked: %#v", packets[0])
	}
	if tokens, ok := packets[0]["candidate_tokens"].([]any); !ok || len(tokens) != 2 {
		t.Fatalf("candidate tokens missing: %#v", packets[0])
	}
}

func drainMethods(methods <-chan string) []string {
	out := []string{}
	timer := time.NewTimer(100 * time.Millisecond)
	defer timer.Stop()
	for {
		select {
		case method := <-methods:
			out = append(out, method)
		case <-timer.C:
			return out
		}
	}
}
