package modelprepare

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

var (
	ErrInvalidArtifactURI = errors.New("modelprepare: invalid artifact uri")
	ErrDownloadTooLarge   = errors.New("modelprepare: artifact exceeds allowed size")
	ErrDownloadSize       = errors.New("modelprepare: artifact size mismatch")
)

type DownloadOptions struct {
	HTTPClient        *http.Client
	MaxBytes          uint64
	ExpectedSizeBytes int64
	AllowFileURI      bool
	AllowInsecureHTTP bool
	AttachAuth        func(*http.Request, string)
}

type DownloadResult struct {
	Path  string `json:"path"`
	Bytes int64  `json:"bytes"`
}

func DownloadArtifact(ctx context.Context, artifactURI, destinationPath string, options DownloadOptions) (DownloadResult, error) {
	artifactURI = strings.TrimSpace(artifactURI)
	destinationPath = strings.TrimSpace(destinationPath)
	if artifactURI == "" || destinationPath == "" {
		return DownloadResult{}, fmt.Errorf("%w: artifact_uri and destination_path required", ErrInvalidArtifactURI)
	}
	parsed, err := url.Parse(artifactURI)
	if err != nil {
		return DownloadResult{}, fmt.Errorf("%w: parse artifact_uri", ErrInvalidArtifactURI)
	}
	if err := os.MkdirAll(filepath.Dir(destinationPath), 0o755); err != nil {
		return DownloadResult{}, err
	}

	var reader io.ReadCloser
	switch strings.ToLower(parsed.Scheme) {
	case "https":
		reader, err = openHTTPArtifact(ctx, artifactURI, options)
	case "http":
		if !options.AllowInsecureHTTP {
			return DownloadResult{}, fmt.Errorf("%w: http artifact_uri disabled", ErrInvalidArtifactURI)
		}
		reader, err = openHTTPArtifact(ctx, artifactURI, options)
	case "file":
		if !options.AllowFileURI {
			return DownloadResult{}, fmt.Errorf("%w: file artifact_uri disabled", ErrInvalidArtifactURI)
		}
		reader, err = openFileArtifact(parsed)
	default:
		return DownloadResult{}, fmt.Errorf("%w: unsupported artifact_uri scheme", ErrInvalidArtifactURI)
	}
	if err != nil {
		return DownloadResult{}, err
	}
	defer reader.Close()

	out, err := os.OpenFile(destinationPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return DownloadResult{}, err
	}
	wrote, copyErr := copyWithMax(out, reader, options.MaxBytes)
	closeErr := out.Close()
	if copyErr != nil || closeErr != nil {
		_ = os.Remove(destinationPath)
		if copyErr != nil {
			return DownloadResult{}, copyErr
		}
		return DownloadResult{}, closeErr
	}
	if options.ExpectedSizeBytes > 0 && wrote != options.ExpectedSizeBytes {
		_ = os.Remove(destinationPath)
		return DownloadResult{}, fmt.Errorf("%w: got %d want %d", ErrDownloadSize, wrote, options.ExpectedSizeBytes)
	}
	return DownloadResult{Path: destinationPath, Bytes: wrote}, nil
}

func openHTTPArtifact(ctx context.Context, artifactURI string, options DownloadOptions) (io.ReadCloser, error) {
	client := redirectSafeArtifactClient(options.HTTPClient)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, artifactURI, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/octet-stream")
	attachHTTPArtifactAuth(req, artifactURI)
	if options.AttachAuth != nil {
		options.AttachAuth(req, artifactURI)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		resp.Body.Close()
		if isHuggingFaceArtifactURI(artifactURI) && (resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden) {
			return nil, fmt.Errorf("modelprepare: huggingface artifact access denied; accept the model license and set HF_TOKEN or HUGGINGFACE_TOKEN")
		}
		return nil, fmt.Errorf("modelprepare: artifact download status %d", resp.StatusCode)
	}
	if options.MaxBytes > 0 && resp.ContentLength > int64(options.MaxBytes) {
		resp.Body.Close()
		return nil, ErrDownloadTooLarge
	}
	return resp.Body, nil
}

func redirectSafeArtifactClient(base *http.Client) *http.Client {
	if base == nil {
		base = &http.Client{Timeout: 30 * time.Minute}
	}
	clone := *base
	prior := clone.CheckRedirect
	clone.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		if len(via) > 0 && !sameArtifactHost(req.URL, via[len(via)-1].URL) {
			req.Header.Del("X-Node-Token")
		}
		if prior != nil {
			return prior(req, via)
		}
		return nil
	}
	return &clone
}

func sameArtifactHost(a, b *url.URL) bool {
	if a == nil || b == nil {
		return false
	}
	return strings.EqualFold(a.Scheme, b.Scheme) && strings.EqualFold(a.Host, b.Host)
}

func attachHTTPArtifactAuth(req *http.Request, artifactURI string) {
	if req == nil || !isHuggingFaceArtifactURI(artifactURI) {
		return
	}
	token := strings.TrimSpace(os.Getenv("HF_TOKEN"))
	if token == "" {
		token = strings.TrimSpace(os.Getenv("HUGGINGFACE_TOKEN"))
	}
	if token == "" {
		return
	}
	req.Header.Set("Authorization", "Bearer "+token)
}

func isHuggingFaceArtifactURI(artifactURI string) bool {
	parsed, err := url.Parse(strings.TrimSpace(artifactURI))
	if err != nil {
		return false
	}
	host := strings.ToLower(strings.TrimSpace(parsed.Hostname()))
	return host == "huggingface.co" || strings.HasSuffix(host, ".huggingface.co") || host == "hf.co" || strings.HasSuffix(host, ".hf.co")
}

func openFileArtifact(parsed *url.URL) (io.ReadCloser, error) {
	if parsed == nil {
		return nil, ErrInvalidArtifactURI
	}
	host := parsed.Hostname()
	if host != "" && !strings.EqualFold(host, "localhost") {
		if ip := net.ParseIP(host); ip == nil || !ip.IsLoopback() {
			return nil, fmt.Errorf("%w: file artifact host must be local", ErrInvalidArtifactURI)
		}
	}
	pathValue, err := url.PathUnescape(parsed.Path)
	if err != nil {
		return nil, err
	}
	if runtime.GOOS == "windows" {
		pathValue = strings.TrimPrefix(pathValue, "/")
	}
	pathValue = filepath.FromSlash(pathValue)
	info, err := os.Stat(pathValue)
	if err != nil {
		return nil, err
	}
	if info.IsDir() {
		return nil, fmt.Errorf("%w: file artifact is a directory", ErrInvalidArtifactURI)
	}
	return os.Open(pathValue)
}

func copyWithMax(dst io.Writer, src io.Reader, maxBytes uint64) (int64, error) {
	if maxBytes == 0 {
		return io.Copy(dst, src)
	}
	limited := io.LimitReader(src, int64(maxBytes)+1)
	wrote, err := io.Copy(dst, limited)
	if err != nil {
		return wrote, err
	}
	if uint64(wrote) > maxBytes {
		return wrote, ErrDownloadTooLarge
	}
	return wrote, nil
}
