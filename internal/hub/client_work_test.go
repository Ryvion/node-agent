package hub

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestFetchWorkAcceptsAbortScopeID(t *testing.T) {
	t.Parallel()

	pub, priv := testKeyPair()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/node/work" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"has_work":       true,
			"job_id":         "job-123",
			"abort_scope_id": "scope-abc",
			"kind":           "custom_runtime",
			"image":          "ghcr.io/example/runner:latest",
			"spec_json":      `{"task":"custom_runtime"}`,
		})
	}))
	defer ts.Close()

	work, err := New(ts.URL, pub, priv).FetchWork(context.Background())
	if err != nil {
		t.Fatalf("FetchWork() error = %v", err)
	}
	if work == nil || work.WorkScopeID != "scope-abc" {
		t.Fatalf("FetchWork() work scope = %#v, want abort_scope_id", work)
	}
}
