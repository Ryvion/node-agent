package blob

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/Ryvion/node-agent/internal/hub"
)

type UploadResult struct {
	URL  string
	Key  string
	Hash string
}

func Upload(ctx context.Context, client *hub.Client, jobID, filePath string) (*UploadResult, error) {
	if client == nil {
		return nil, fmt.Errorf("client required")
	}
	jobID = strings.TrimSpace(jobID)
	if jobID == "" {
		return nil, fmt.Errorf("job_id required")
	}
	filePath = strings.TrimSpace(filePath)
	if filePath == "" {
		return nil, fmt.Errorf("file path required")
	}

	fi, err := os.Stat(filePath)
	if err != nil {
		return nil, err
	}
	if fi.IsDir() {
		return nil, fmt.Errorf("artifact path is a directory")
	}
	size := fi.Size()

	hexHash, err := hashFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("hash artifact: %w", err)
	}

	uploadToken, err := client.PrepareUpload(ctx, jobID, uint64(size))
	if err != nil {
		return nil, err
	}

	f, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	putURL := client.AbsoluteURL(uploadToken.PutURL)
	method := http.MethodPut
	if strings.HasPrefix(uploadToken.PutURL, "/") {
		method = http.MethodPost
	}
	req, err := http.NewRequestWithContext(ctx, method, putURL, f)
	if err != nil {
		return nil, err
	}
	req.ContentLength = size
	req.Header.Set("Content-Type", "application/octet-stream")
	if strings.HasPrefix(uploadToken.PutURL, "/") {
		headers := client.BlobUploadHeaders(jobID, size, time.Now().UnixMilli())
		for k, v := range headers {
			req.Header.Set(k, v)
		}
	}

	// Use a dedicated HTTP client for uploads — the hub client has a 30s timeout
	// which is too short for large artifacts (models can be 500MB+).
	uploadClient := &http.Client{Timeout: 30 * time.Minute}
	if strings.HasPrefix(uploadToken.PutURL, "/") {
		// Hub proxy path — use the hub client (with auth headers)
		uploadClient = nil
	}
	var resp *http.Response
	if uploadClient != nil {
		resp, err = uploadClient.Do(req)
	} else {
		resp, err = client.Do(req)
	}
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		rb, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))
		return nil, fmt.Errorf("artifact PUT failed: %d %s", resp.StatusCode, strings.TrimSpace(string(rb)))
	}

	artifactURL := putURL
	if strings.HasPrefix(uploadToken.PutURL, "/") {
		var putResp struct {
			URL string `json:"url"`
		}
		if rb, err := io.ReadAll(io.LimitReader(resp.Body, 8192)); err == nil {
			if len(bytes.TrimSpace(rb)) > 0 && json.Unmarshal(rb, &putResp) == nil && strings.TrimSpace(putResp.URL) != "" {
				artifactURL = client.AbsoluteURL(putResp.URL)
			} else {
				artifactURL = client.AbsoluteURL("/api/v1/blob/" + jobID)
			}
		}
	}

	if strings.TrimSpace(uploadToken.Key) != "" {
		_ = uploadManifest(ctx, client, uploadToken.Key, jobID, uint64(size), hexHash)
	}

	return &UploadResult{URL: artifactURL, Key: uploadToken.Key, Hash: hexHash}, nil
}

func hashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func uploadManifest(ctx context.Context, client *hub.Client, objectKey, jobID string, size uint64, hashHex string) error {
	manifest := map[string]any{
		"job_id":       jobID,
		"object_key":   objectKey,
		"sha256":       hashHex,
		"size_bytes":   size,
		"node_pubkey":  client.PublicKeyHex(),
		"submitted_at": time.Now().UTC().Format(time.RFC3339),
	}

	payload, err := json.Marshal(manifest)
	if err != nil {
		return err
	}
	digest := sha256.Sum256(payload)
	manifest["signature_b64"] = base64.StdEncoding.EncodeToString(client.SignDigest(digest[:]))

	body, err := json.Marshal(manifest)
	if err != nil {
		return err
	}
	presignedURL, err := client.PresignManifest(ctx, objectKey+".manifest.json")
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, client.AbsoluteURL(presignedURL), bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		rb, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))
		return fmt.Errorf("manifest PUT failed: %d %s", resp.StatusCode, strings.TrimSpace(string(rb)))
	}
	return nil
}
