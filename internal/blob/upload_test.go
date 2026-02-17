package blob

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/Ryvion/node-agent/internal/hub"
)

func TestUploadRelativeURLUsesPOST(t *testing.T) {
	pub, priv := testKeyPair(t)

	var (
		mu            sync.Mutex
		blobMethod    string
		uploadedBytes []byte
	)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v1/node/upload/prepare":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"ok":       true,
				"provider": "hub",
				"put_url":  "/api/v1/blob/job_1",
				"key":      "",
			})
		case "/api/v1/blob/job_1":
			mu.Lock()
			blobMethod = r.Method
			uploadedBytes, _ = io.ReadAll(r.Body)
			mu.Unlock()
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": "true", "url": "/api/v1/blob/job_1"})
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer ts.Close()

	file := writeTempArtifact(t, []byte("artifact-data"))
	client := hub.New(ts.URL, pub, priv, hub.WithHTTPClient(ts.Client()))

	res, err := Upload(context.Background(), client, "job_1", file)
	if err != nil {
		t.Fatalf("upload failed: %v", err)
	}
	if res == nil {
		t.Fatalf("expected result")
	}

	mu.Lock()
	defer mu.Unlock()
	if blobMethod != http.MethodPost {
		t.Fatalf("expected method POST, got %s", blobMethod)
	}
	if string(uploadedBytes) != "artifact-data" {
		t.Fatalf("uploaded bytes mismatch: %q", string(uploadedBytes))
	}
}

func TestUploadAbsoluteURLUsesPUT(t *testing.T) {
	pub, priv := testKeyPair(t)

	var (
		mu         sync.Mutex
		blobMethod string
		serverURL  string
	)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v1/node/upload/prepare":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"ok":       true,
				"provider": "s3",
				"put_url":  serverURL + "/external-put",
				"key":      "",
			})
		case "/external-put":
			mu.Lock()
			blobMethod = r.Method
			mu.Unlock()
			w.WriteHeader(http.StatusOK)
		default:
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
	}))
	defer ts.Close()
	serverURL = ts.URL

	file := writeTempArtifact(t, []byte("artifact-data"))
	client := hub.New(ts.URL, pub, priv, hub.WithHTTPClient(ts.Client()))

	if _, err := Upload(context.Background(), client, "job_1", file); err != nil {
		t.Fatalf("upload failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if blobMethod != http.MethodPut {
		t.Fatalf("expected method PUT, got %s", blobMethod)
	}
}

func testKeyPair(t *testing.T) (ed25519.PublicKey, ed25519.PrivateKey) {
	t.Helper()
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("generate key: %v", err)
	}
	return pub, priv
}

func writeTempArtifact(t *testing.T, b []byte) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "artifact.bin")
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	return path
}
