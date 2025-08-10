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
	// Create local storage directory
	localDir := "/tmp/akatosh_results"
	if err := os.MkdirAll(localDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create local storage directory: %w", err)
	}

	return &ResultStorage{
		hubBaseURL: hubBaseURL,
		ipfsNode:   "http://127.0.0.1:5001", // Local IPFS node
		localDir:   localDir,
	}, nil
}

// UploadWorkloadResult uploads workload result files with multiple storage options
func (rs *ResultStorage) UploadWorkloadResult(ctx context.Context, jobID string, resultPath string, resultType string) (*UploadResult, error) {
	// Check if file exists
	fileInfo, err := os.Stat(resultPath)
	if err != nil {
		return nil, fmt.Errorf("result file not found: %w", err)
	}

	// Calculate file hash
	hash, err := rs.calculateFileHash(resultPath)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate file hash: %w", err)
	}

	timestamp := time.Now().Unix()
	
	// Try IPFS upload first
	ipfsURL, err := rs.UploadToIPFS(ctx, resultPath)
	if err == nil {
		uploadResult := &UploadResult{
			URL:       ipfsURL,
			Hash:      hash,
			Size:      fileInfo.Size(),
			Type:      resultType,
			Timestamp: timestamp,
		}

		// Register with hub
		if err := rs.registerResultWithHub(ctx, jobID, uploadResult); err != nil {
			fmt.Printf("Warning: failed to register IPFS result with hub: %v\n", err)
		}

		return uploadResult, nil
	}

	fmt.Printf("IPFS upload failed (%v), trying hub upload...\n", err)

	// Fallback to hub upload
	return rs.UploadToHub(ctx, jobID, resultPath, resultType)
}

// UploadMultipleResults uploads multiple result files (e.g., rendered frames)
func (rs *ResultStorage) UploadMultipleResults(ctx context.Context, jobID string, resultPaths []string, resultType string) ([]*UploadResult, error) {
	var results []*UploadResult
	
	for i, path := range resultPaths {
		if _, err := os.Stat(path); os.IsNotExist(err) {
			continue // Skip missing files
		}

		result, err := rs.UploadWorkloadResult(ctx, fmt.Sprintf("%s_frame_%d", jobID, i), path, resultType)
		if err != nil {
			return results, fmt.Errorf("failed to upload file %s: %w", path, err)
		}
		
		results = append(results, result)
		
		// Add small delay to avoid rate limiting
		time.Sleep(100 * time.Millisecond)
	}
	
	return results, nil
}

// DownloadWorkloadInput downloads input files for workload execution
func (rs *ResultStorage) DownloadWorkloadInput(ctx context.Context, inputURL, destPath string) error {
	// Create destination directory
	destDir := filepath.Dir(destPath)
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return fmt.Errorf("failed to create destination directory: %w", err)
	}

	// Download file
	resp, err := http.Get(inputURL)
	if err != nil {
		return fmt.Errorf("failed to download input: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download failed with status: %d", resp.StatusCode)
	}

	// Create destination file
	destFile, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("failed to create destination file: %w", err)
	}
	defer destFile.Close()

	// Copy data
	_, err = io.Copy(destFile, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to write downloaded data: %w", err)
	}

	return nil
}

// UploadToHub uploads result directly to hub using multipart form
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

	// Create multipart form
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)

	// Add job ID field
	writer.WriteField("job_id", jobID)
	writer.WriteField("type", resultType)

	// Add file field
	part, err := writer.CreateFormFile("result", filepath.Base(resultPath))
	if err != nil {
		return nil, fmt.Errorf("failed to create form file: %w", err)
	}

	_, err = io.Copy(part, file)
	if err != nil {
		return nil, fmt.Errorf("failed to copy file to form: %w", err)
	}

	writer.Close()

	// Create HTTP request
	uploadURL := fmt.Sprintf("%s/api/results/upload", rs.hubBaseURL)
	req, err := http.NewRequestWithContext(ctx, "POST", uploadURL, &buf)
	if err != nil {
		return nil, fmt.Errorf("failed to create upload request: %w", err)
	}

	req.Header.Set("Content-Type", writer.FormDataContentType())

	// Send request
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

	// Parse response (simplified - in real implementation parse JSON response)
	timestamp := time.Now().Unix()
	return &UploadResult{
		URL:       fmt.Sprintf("%s/api/results/%s", rs.hubBaseURL, jobID),
		Hash:      "hub_upload_hash", // Would be returned by hub
		Size:      fileInfo.Size(),
		Type:      resultType,
		Timestamp: timestamp,
	}, nil
}

// Helper methods

func (rs *ResultStorage) getContentType(resultType string) string {
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
}

func (rs *ResultStorage) calculateFileHash(filePath string) (string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return "", err
	}
	defer file.Close()

	// Calculate SHA256 hash
	hasher := sha256.New()
	if _, err := io.Copy(hasher, file); err != nil {
		return "", err
	}
	
	return hex.EncodeToString(hasher.Sum(nil)), nil
}

func (rs *ResultStorage) registerResultWithHub(ctx context.Context, jobID string, result *UploadResult) error {
	// Create registration request
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

// IPFS upload support (alternative to S3)
func (rs *ResultStorage) UploadToIPFS(ctx context.Context, resultPath string) (string, error) {
	// Open file
	file, err := os.Open(resultPath)
	if err != nil {
		return "", fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	// Create multipart form for IPFS API
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

	// Upload to IPFS node (assuming local node at 5001)
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

	// Parse IPFS response to get hash
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read IPFS response: %w", err)
	}

	// Parse JSON response from IPFS
	var ipfsResponse struct {
		Hash string `json:"Hash"`
		Name string `json:"Name"`
		Size string `json:"Size"`
	}
	
	if err := json.Unmarshal(body, &ipfsResponse); err != nil {
		// Fallback if parsing fails
		return "", fmt.Errorf("failed to parse IPFS response: %w", err)
	}

	return fmt.Sprintf("https://ipfs.io/ipfs/%s", ipfsResponse.Hash), nil
}