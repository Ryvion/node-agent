package storage

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

type ResultStorage struct {
	hubBaseURL string
	ipfsNode   string
	localDir   string
}

type UploadResult struct {
	URL       string `json:"url"`
	Hash      string `json:"hash"`
	Size      int64  `json:"size"`
	Type      string `json:"type"`
	Timestamp int64  `json:"timestamp"`
}

func NewResultStorage(bucketName, hubBaseURL string) (*ResultStorage, error) {
	localDir := "/tmp/ryvion_results"
	if err := os.MkdirAll(localDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create local storage directory: %w", err)
	}

	return &ResultStorage{
		hubBaseURL: hubBaseURL,
		ipfsNode:   "http://127.0.0.1:5001",
		localDir:   localDir,
	}, nil
}

func (rs *ResultStorage) UploadWorkloadResult(ctx context.Context, jobID string, resultPath string, resultType string) (*UploadResult, error) {
	fileInfo, err := os.Stat(resultPath)
	if err != nil {
		return nil, fmt.Errorf("result file not found: %w", err)
	}

	hash, err := rs.calculateFileHash(resultPath)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate file hash: %w", err)
	}

	timestamp := time.Now().Unix()

	ipfsURL, err := rs.UploadToIPFS(ctx, resultPath)
	if err == nil {
		uploadResult := &UploadResult{
			URL:       ipfsURL,
			Hash:      hash,
			Size:      fileInfo.Size(),
			Type:      resultType,
			Timestamp: timestamp,
		}

		if err := rs.registerResultWithHub(ctx, jobID, uploadResult); err != nil {
			fmt.Printf("Warning: failed to register IPFS result with hub: %v\n", err)
		}

		return uploadResult, nil
	}

	fmt.Printf("IPFS upload failed (%v), trying hub upload...\n", err)

	return rs.UploadToHub(ctx, jobID, resultPath, resultType)
}

func (rs *ResultStorage) UploadMultipleResults(ctx context.Context, jobID string, resultPaths []string, resultType string) ([]*UploadResult, error) {
	var results []*UploadResult

	for i, path := range resultPaths {
		if _, err := os.Stat(path); os.IsNotExist(err) {
			continue
		}

		result, err := rs.UploadWorkloadResult(ctx, fmt.Sprintf("%s_frame_%d", jobID, i), path, resultType)
		if err != nil {
			return results, fmt.Errorf("failed to upload file %s: %w", path, err)
		}

		results = append(results, result)

		time.Sleep(100 * time.Millisecond)
	}

	return results, nil
}

func (rs *ResultStorage) DownloadWorkloadInput(ctx context.Context, inputURL, destPath string) error {
	destDir := filepath.Dir(destPath)
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return fmt.Errorf("failed to create destination directory: %w", err)
	}

	resp, err := http.Get(inputURL)
	if err != nil {
		return fmt.Errorf("failed to download input: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download failed with status: %d", resp.StatusCode)
	}

	destFile, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("failed to create destination file: %w", err)
	}
	defer destFile.Close()

	_, err = io.Copy(destFile, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to write downloaded data: %w", err)
	}

	return nil
}

func (rs *ResultStorage) UploadToHub(ctx context.Context, jobID, resultPath, resultType string) (*UploadResult, error) {
	file, err := os.Open(resultPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open result file: %w", err)
	}
	defer file.Close()

	fileInfo, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("failed to get file info: %w", err)
	}

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	writer.WriteField("job_id", jobID)
	writer.WriteField("type", resultType)

	part, err := writer.CreateFormFile("result", filepath.Base(resultPath))
	if err != nil {
		return nil, fmt.Errorf("failed to create form file: %w", err)
	}

	_, err = io.Copy(part, file)
	if err != nil {
		return nil, fmt.Errorf("failed to copy file to form: %w", err)
	}

	writer.Close()

	uploadURL := fmt.Sprintf("%s/api/results/upload", rs.hubBaseURL)
	req, err := http.NewRequestWithContext(ctx, "POST", uploadURL, &buf)
	if err != nil {
		return nil, fmt.Errorf("failed to create upload request: %w", err)
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("upload request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("upload failed with status %d: %s", resp.StatusCode, string(body))
	}

	timestamp := time.Now().Unix()
	return &UploadResult{
		URL:       fmt.Sprintf("%s/api/results/%s", rs.hubBaseURL, jobID),
		Hash:      "hub_upload_hash",
		Size:      fileInfo.Size(),
		Type:      resultType,
		Timestamp: timestamp,
	}, nil
}

/*func (rs *ResultStorage) getContentType(resultType string) string {
	switch resultType {
	case "image", "stable-diffusion":
		return "image/png"
	case "video", "transcoding":
		return "video/mp4"
	case "audio", "whisper":
		return "audio/mpeg"
	case "model", "training":
		return "application/octet-stream"
	case "text", "llm-inference":
		return "text/plain"
	default:
		return "application/octet-stream"
	}
}*/

func (rs *ResultStorage) calculateFileHash(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer file.Close()

	hasher := sha256.New()
	if _, err := io.Copy(hasher, file); err != nil {
		return "", err
	}

	return hex.EncodeToString(hasher.Sum(nil)), nil
}

func (rs *ResultStorage) registerResultWithHub(ctx context.Context, jobID string, result *UploadResult) error {
	registerURL := fmt.Sprintf("%s/api/results/register", rs.hubBaseURL)

	reqBody := map[string]interface{}{
		"job_id":    jobID,
		"url":       result.URL,
		"hash":      result.Hash,
		"size":      result.Size,
		"type":      result.Type,
		"timestamp": result.Timestamp,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("failed to marshal registration data: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", registerURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("failed to create registration request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("registration request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("registration failed with status %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

func (rs *ResultStorage) UploadToIPFS(ctx context.Context, resultPath string) (string, error) {
	file, err := os.Open(resultPath)
	if err != nil {
		return "", fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	part, err := writer.CreateFormFile("file", filepath.Base(resultPath))
	if err != nil {
		return "", fmt.Errorf("failed to create form file: %w", err)
	}

	_, err = io.Copy(part, file)
	if err != nil {
		return "", fmt.Errorf("failed to copy file: %w", err)
	}

	writer.Close()
	ipfsURL := "http://127.0.0.1:5001/api/v0/add"
	req, err := http.NewRequestWithContext(ctx, "POST", ipfsURL, &buf)
	if err != nil {
		return "", fmt.Errorf("failed to create IPFS request: %w", err)
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("IPFS upload failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("IPFS upload failed with status: %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read IPFS response: %w", err)
	}

	var ipfsResponse struct {
		Hash string `json:"Hash"`
		Name string `json:"Name"`
		Size string `json:"Size"`
	}

	if err := json.Unmarshal(body, &ipfsResponse); err != nil {
		return "", fmt.Errorf("failed to parse IPFS response: %w", err)
	}

	return fmt.Sprintf("https://ipfs.io/ipfs/%s", ipfsResponse.Hash), nil
}
