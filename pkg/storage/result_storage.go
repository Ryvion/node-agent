package storage

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

const (
	HubUploadTimeout           = 10 * time.Minute
	IPFSUploadTimeout          = 5 * time.Minute
	HubRegisterTimeout         = 30 * time.Second
	DownloadInputTimeout       = 5 * time.Minute
	UploadMultipleResultsDelay = 100 * time.Millisecond
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

func NewResultStorage(hubBaseURL, ipfsNodeURL string) (*ResultStorage, error) {
	localDir := "/tmp/ryvion_results"
	if err := os.MkdirAll(localDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create local storage directory: %w", err)
	}

	return &ResultStorage{
		hubBaseURL: hubBaseURL,
		ipfsNode:   ipfsNodeURL,
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

	log.Printf("IPFS upload failed for job %s (%v), trying hub upload...", jobID, err)

	return rs.UploadToHub(ctx, jobID, resultPath, resultType)
}

func (rs *ResultStorage) UploadMultipleResults(ctx context.Context, jobID string, resultPaths []string, resultType string) ([]*UploadResult, error) {
	var results []*UploadResult

	for i, path := range resultPaths {

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:

		}

		if _, err := os.Stat(path); os.IsNotExist(err) {
			log.Printf("File %s does not exist, skipping...", path)
			continue
		}

		currentJobID := fmt.Sprintf("%s_frame_%d", jobID, i)
		result, err := rs.UploadWorkloadResult(ctx, currentJobID, path, resultType)
		if err != nil {
			return results, fmt.Errorf("failed to upload file %s for job %s: %w", path, currentJobID, err)
		}

		results = append(results, result)

		select {
		case <-time.After(UploadMultipleResultsDelay):
		case <-ctx.Done():
			return results, ctx.Err()
		}
	}

	return results, nil
}

func (rs *ResultStorage) DownloadWorkloadInput(ctx context.Context, inputURL, destPath string) error {
	destDir := filepath.Dir(destPath)
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return fmt.Errorf("failed to create destination directory: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "GET", inputURL, nil)
	if err != nil {
		return fmt.Errorf("failed to create download request: %w", err)
	}

	client := &http.Client{Timeout: DownloadInputTimeout}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to download input: %w", err)
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			log.Printf("Error closing response body during download: %v", closeErr)
		}
	}()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("download failed with status: %d: %s", resp.StatusCode, string(body))
	}

	destFile, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("failed to create destination file: %w", err)
	}
	defer func() {
		if closeErr := destFile.Close(); closeErr != nil {
			log.Printf("Error closing destination file during download: %v", closeErr)
		}
	}()

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
	defer func() {
		if closeErr := file.Close(); closeErr != nil {
			log.Printf("Error closing result file during upload: %v", closeErr)
		}
	}()

	fileInfo, err := file.Stat()
	if err != nil {
		return nil, fmt.Errorf("failed to get file info: %w", err)
	}

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	if err := writer.WriteField("job_id", jobID); err != nil {
		return nil, fmt.Errorf("failed to write job ID: %w", err)
	}
	if err := writer.WriteField("type", resultType); err != nil {
		return nil, fmt.Errorf("failed to write result type: %w", err)
	}

	part, err := writer.CreateFormFile("result", filepath.Base(resultPath))
	if err != nil {
		return nil, fmt.Errorf("failed to create form file: %w", err)
	}

	if _, err = io.Copy(part, file); err != nil {
		return nil, fmt.Errorf("failed to copy file to form: %w", err)
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to close writer: %w", err)
	}

	uploadURL := fmt.Sprintf("%s/api/results/upload", rs.hubBaseURL)
	req, err := http.NewRequestWithContext(ctx, "POST", uploadURL, &buf)
	if err != nil {
		return nil, fmt.Errorf("failed to create upload request: %w", err)
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{Timeout: HubUploadTimeout}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("upload request failed: %w", err)
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			log.Printf("Error closing response body during upload: %v", closeErr)
		}
	}()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("upload failed with status %d: %s", resp.StatusCode, string(body))
	}

	timestamp := time.Now().Unix()
	return &UploadResult{
		URL:       fmt.Sprintf("%s/api/results/%s", rs.hubBaseURL, jobID),
		Hash:      "hub_upload_hash_placeholder",
		Size:      fileInfo.Size(),
		Type:      resultType,
		Timestamp: timestamp,
	}, nil
}

func (rs *ResultStorage) calculateFileHash(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer func() {
		if closeErr := file.Close(); closeErr != nil {
			log.Printf("Error closing file during hash calculation: %v", closeErr)
		}
	}()

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

	client := &http.Client{Timeout: HubRegisterTimeout}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("registration request failed: %w", err)
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			log.Printf("Error closing response body during registration: %v", closeErr)
		}
	}()

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
	defer func() {
		if closeErr := file.Close(); closeErr != nil {
			log.Printf("Error closing file during IPFS upload: %v", closeErr)
		}
	}()

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	part, err := writer.CreateFormFile("file", filepath.Base(resultPath))
	if err != nil {
		return "", fmt.Errorf("failed to create form file: %w", err)
	}

	if _, err = io.Copy(part, file); err != nil {
		return "", fmt.Errorf("failed to copy file: %w", err)
	}

	if err := writer.Close(); err != nil {
		return "", fmt.Errorf("failed to close writer: %w", err)
	}

	ipfsURL := fmt.Sprintf("%s/api/v0/add", rs.ipfsNode)
	req, err := http.NewRequestWithContext(ctx, "POST", ipfsURL, &buf)
	if err != nil {
		return "", fmt.Errorf("failed to create IPFS request: %w", err)
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())

	client := &http.Client{Timeout: IPFSUploadTimeout}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("IPFS upload failed: %w", err)
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			log.Printf("Error closing response body during IPFS upload: %v", closeErr)
		}
	}()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("IPFS upload failed with status: %d: %s", resp.StatusCode, string(body))
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

	return fmt.Sprintf("ipfs://%s", ipfsResponse.Hash), nil
}
