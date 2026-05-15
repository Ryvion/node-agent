package runner

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"strings"
	"time"
)

type VerifierSessionExecution struct {
	AcceptedLen       int
	TreeCID           string
	RollbackBranchIDs []string
	ProbeSummary      map[string]any
}

type verifierRPCRequest struct {
	JSONRPC string         `json:"jsonrpc"`
	ID      string         `json:"id"`
	Method  string         `json:"method"`
	Params  map[string]any `json:"params,omitempty"`
}

type verifierRPCResponse struct {
	JSONRPC string         `json:"jsonrpc"`
	ID      string         `json:"id"`
	Result  map[string]any `json:"result,omitempty"`
	Error   *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

func IsVerifierSessionSpec(specJSON string) bool {
	job, ok := verifierJobFromSpec(specJSON)
	if !ok {
		return false
	}
	method := strings.TrimSpace(stringFromMap(job, "method"))
	if method == "verify_tree" {
		return true
	}
	task := strings.TrimSpace(stringFromMap(job, "task"))
	return task == "verifier_session_v8" || task == "v8_verifier_session"
}

func ExecuteVerifierSessionRPC(ctx context.Context, socketPath, specJSON string) (VerifierSessionExecution, error) {
	job, ok := verifierJobFromSpec(specJSON)
	if !ok {
		return VerifierSessionExecution{}, fmt.Errorf("verifier session job required")
	}
	session := mapFromMap(job, "session")
	tree := mapFromMap(job, "tree")
	if len(session) == 0 || len(tree) == 0 {
		return VerifierSessionExecution{}, fmt.Errorf("verifier session and tree required")
	}
	sessionID := strings.TrimSpace(stringFromMap(session, "session_id"))
	if sessionID == "" {
		sessionID = "sess-" + time.Now().UTC().Format("20060102150405")
		session["session_id"] = sessionID
	}
	if _, err := callVerifierRPC(ctx, socketPath, "start_session", map[string]any{"session": session}); err != nil {
		return VerifierSessionExecution{}, err
	}
	if _, err := callVerifierRPC(ctx, socketPath, "prefill", map[string]any{
		"session_id":  sessionID,
		"prefix_hash": stringFromMap(session, "prefix_hash"),
	}); err != nil {
		return VerifierSessionExecution{}, err
	}
	verifyResult, err := callVerifierRPC(ctx, socketPath, "verify_tree", map[string]any{
		"session":         session,
		"tree":            tree,
		"kv_cache_policy": mapFromMap(job, "kv_cache_policy"),
	})
	if err != nil {
		return VerifierSessionExecution{}, err
	}
	receipt := mapFromMap(verifyResult, "accepted_token_receipt")
	acceptedLen := intFromMap(receipt, "accepted_len")
	if acceptedLen <= 0 {
		acceptedLen = intFromMap(verifyResult, "accepted_len")
	}
	treeCID := strings.TrimSpace(stringFromMap(receipt, "tree_cid"))
	if treeCID == "" {
		treeCID = strings.TrimSpace(stringFromMap(tree, "tree_cid"))
	}
	rollbackBranchIDs := stringSliceFromMap(receipt, "rollback_branch_ids")
	if len(rollbackBranchIDs) == 0 {
		rollbackBranchIDs = stringSliceFromMap(verifyResult, "rollback_branch_ids")
	}
	if acceptedLen > 0 {
		if _, err := callVerifierRPC(ctx, socketPath, "commit", map[string]any{
			"session_id":   sessionID,
			"accepted_len": acceptedLen,
			"tree_cid":     treeCID,
		}); err != nil {
			return VerifierSessionExecution{}, err
		}
	}
	if len(rollbackBranchIDs) > 0 {
		if _, err := callVerifierRPC(ctx, socketPath, "rollback", map[string]any{
			"session_id":   sessionID,
			"branch_ids":   rollbackBranchIDs,
			"tree_cid":     treeCID,
			"accepted_len": acceptedLen,
		}); err != nil {
			return VerifierSessionExecution{}, err
		}
	}
	if _, err := callVerifierRPC(ctx, socketPath, "close_session", map[string]any{"session_id": sessionID}); err != nil {
		return VerifierSessionExecution{}, err
	}
	return VerifierSessionExecution{
		AcceptedLen:       acceptedLen,
		TreeCID:           treeCID,
		RollbackBranchIDs: rollbackBranchIDs,
		ProbeSummary:      mapFromMap(verifyResult, "probe_summary"),
	}, nil
}

func waitForVerifierSessionSocket(ctx context.Context, socketPath string, timeout time.Duration) error {
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	deadline := time.Now().Add(timeout)
	for {
		if _, err := os.Stat(socketPath); err == nil {
			conn, err := (&net.Dialer{Timeout: 250 * time.Millisecond}).DialContext(ctx, "unix", socketPath)
			if err == nil {
				_ = conn.Close()
				return nil
			}
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("verifier session socket not ready: %s", socketPath)
		}
		time.Sleep(50 * time.Millisecond)
	}
}

func callVerifierRPC(ctx context.Context, socketPath, method string, params map[string]any) (map[string]any, error) {
	conn, err := (&net.Dialer{Timeout: 2 * time.Second}).DialContext(ctx, "unix", socketPath)
	if err != nil {
		return nil, err
	}
	defer conn.Close()
	req := verifierRPCRequest{
		JSONRPC: "2.0",
		ID:      method + "-" + time.Now().UTC().Format("150405.000000000"),
		Method:  method,
		Params:  params,
	}
	if err := json.NewEncoder(conn).Encode(req); err != nil {
		return nil, err
	}
	scanner := bufio.NewScanner(conn)
	if !scanner.Scan() {
		if err := scanner.Err(); err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("verifier rpc %s returned no response", method)
	}
	var resp verifierRPCResponse
	if err := json.Unmarshal(scanner.Bytes(), &resp); err != nil {
		return nil, err
	}
	if resp.Error != nil {
		return nil, fmt.Errorf("verifier rpc %s failed: %s", method, resp.Error.Message)
	}
	return resp.Result, nil
}

func verifierJobFromSpec(specJSON string) (map[string]any, bool) {
	var root map[string]any
	if err := json.Unmarshal([]byte(specJSON), &root); err != nil {
		return nil, false
	}
	if nested := mapFromMap(root, "verifier_job"); len(nested) > 0 {
		return nested, true
	}
	if nested := mapFromMap(root, "verify_tree"); len(nested) > 0 {
		return nested, true
	}
	return root, len(root) > 0
}

func mapFromMap(raw map[string]any, key string) map[string]any {
	if raw == nil {
		return nil
	}
	switch typed := raw[key].(type) {
	case map[string]any:
		return typed
	default:
		return nil
	}
}

func stringFromMap(raw map[string]any, key string) string {
	if raw == nil {
		return ""
	}
	switch typed := raw[key].(type) {
	case string:
		return typed
	default:
		return ""
	}
}

func intFromMap(raw map[string]any, key string) int {
	if raw == nil {
		return 0
	}
	switch typed := raw[key].(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	case json.Number:
		v, _ := typed.Int64()
		return int(v)
	default:
		return 0
	}
}

func stringSliceFromMap(raw map[string]any, key string) []string {
	if raw == nil {
		return nil
	}
	switch typed := raw[key].(type) {
	case []string:
		return append([]string(nil), typed...)
	case []any:
		out := make([]string, 0, len(typed))
		for _, value := range typed {
			if s, ok := value.(string); ok && strings.TrimSpace(s) != "" {
				out = append(out, strings.TrimSpace(s))
			}
		}
		return out
	default:
		return nil
	}
}
