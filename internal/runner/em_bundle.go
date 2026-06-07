package runner

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"crypto/ed25519"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// emBundleManifest describes a signed, self-contained FDTD runtime bundle. It is
// the native-EM analogue of the inference manager's model/binary descriptors.
// The hub serves it (per OS x GPU) and the node downloads + verifies + caches it
// to ~/.ryvion/runtimes/em/<engine>-<version>/ exactly like the inference
// runtime auto-update path.
type emBundleManifest struct {
	Engine        string `json:"engine"`         // "gprmax" | "openems" | "meep"
	EngineVersion string `json:"engine_version"` // pinned for QA determinism
	BundleURL     string `json:"bundle_url"`     // https URL of the archive (.tar.gz/.zip)
	BundleSHA256  string `json:"bundle_sha256"`  // expected archive digest
	// Entrypoint is the path (relative to the extracted bundle root) of the
	// executable/script that consumes /work/job.json and writes the result
	// contract. e.g. "bin/em-runner" or "run.py".
	Entrypoint string `json:"entrypoint"`
	OS         string `json:"os"`   // goos this bundle targets
	Arch       string `json:"arch"` // goarch this bundle targets
	// Signature is an Ed25519 signature (hex) over the canonical manifest bytes,
	// signed by the platform bundle-signing key. Verified before extraction.
	Signature string `json:"signature"`
}

// emRuntimeRoot returns ~/.ryvion/runtimes/em (overridable for tests/installs),
// mirroring imageRuntimeRoot() in the ryvion_runtime image lane.
func emRuntimeRoot() string {
	if root := strings.TrimSpace(os.Getenv("RYVION_EM_RUNTIME_ROOT")); root != "" {
		return root
	}
	if home, err := os.UserHomeDir(); err == nil && strings.TrimSpace(home) != "" {
		return filepath.Join(home, ".ryvion", "runtimes", "em")
	}
	return filepath.Join(os.TempDir(), "ryvion-em-runtime")
}

// emBundleSigningKey returns the Ed25519 public key (hex) used to verify EM
// bundle manifests. Configured by the installer/operator; when unset, signature
// verification is skipped only if RYV_EM_ALLOW_UNSIGNED_BUNDLE=1 (dev/test).
func emBundleSigningKey() ed25519.PublicKey {
	raw := strings.TrimSpace(os.Getenv("RYV_EM_BUNDLE_PUBKEY"))
	if raw == "" {
		return nil
	}
	b, err := hex.DecodeString(raw)
	if err != nil || len(b) != ed25519.PublicKeySize {
		return nil
	}
	return ed25519.PublicKey(b)
}

func emAllowUnsignedBundle() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("RYV_EM_ALLOW_UNSIGNED_BUNDLE"))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

// emBundleReadyMarker is written into the extracted bundle dir on success so a
// cached bundle is treated as ready without re-downloading (auto-update via the
// hub-advertised manifest digest, same model as the inference runtime).
const emBundleReadyMarker = ".em-bundle-ready"

// verifyManifestSignature checks the Ed25519 signature over the manifest's
// signed fields. The signed payload excludes the signature field itself.
func verifyManifestSignature(m emBundleManifest) error {
	pub := emBundleSigningKey()
	if pub == nil {
		if emAllowUnsignedBundle() {
			slog.Warn("EM bundle signature verification skipped (RYV_EM_ALLOW_UNSIGNED_BUNDLE=1)", "engine", m.Engine)
			return nil
		}
		return fmt.Errorf("EM bundle signing key not configured (set RYV_EM_BUNDLE_PUBKEY)")
	}
	sig, err := hex.DecodeString(strings.TrimSpace(m.Signature))
	if err != nil || len(sig) == 0 {
		return fmt.Errorf("EM bundle signature missing or malformed")
	}
	signed := m
	signed.Signature = ""
	payload, err := json.Marshal(signed)
	if err != nil {
		return fmt.Errorf("canonicalize manifest: %w", err)
	}
	if !ed25519.Verify(pub, payload, sig) {
		return fmt.Errorf("EM bundle signature verification failed")
	}
	return nil
}

// ensureEMBundle downloads, verifies, and caches the signed FDTD runtime bundle,
// returning the absolute path to its entrypoint executable/script. It mirrors
// the inference manager's native download/auto-update pattern: a cached bundle
// keyed by <engine>-<version> is reused; a digest mismatch forces re-download.
func ensureEMBundle(ctx context.Context, m emBundleManifest, nodeToken string) (string, error) {
	m.Engine = strings.TrimSpace(strings.ToLower(m.Engine))
	m.EngineVersion = strings.TrimSpace(m.EngineVersion)
	if m.Engine == "" || m.EngineVersion == "" {
		return "", fmt.Errorf("EM bundle manifest missing engine/engine_version")
	}
	if strings.TrimSpace(m.Entrypoint) == "" {
		return "", fmt.Errorf("EM bundle manifest missing entrypoint")
	}
	if err := verifyManifestSignature(m); err != nil {
		return "", err
	}

	root := emRuntimeRoot()
	bundleDir := filepath.Join(root, m.Engine+"-"+m.EngineVersion)
	entry := filepath.Join(bundleDir, filepath.FromSlash(strings.TrimPrefix(m.Entrypoint, "/")))
	marker := filepath.Join(bundleDir, emBundleReadyMarker)

	// Cache hit: ready marker present AND entrypoint exists -> reuse.
	if markerBytes, err := os.ReadFile(marker); err == nil {
		if strings.TrimSpace(string(markerBytes)) == m.BundleSHA256 || m.BundleSHA256 == "" {
			if _, statErr := os.Stat(entry); statErr == nil {
				return entry, nil
			}
		}
	}

	if strings.TrimSpace(m.BundleURL) == "" {
		return "", fmt.Errorf("EM bundle %s-%s not cached and no bundle_url provided", m.Engine, m.EngineVersion)
	}

	// Fresh download into a temp archive, verify digest, then extract.
	if err := os.MkdirAll(root, 0o755); err != nil {
		return "", fmt.Errorf("create EM runtime root: %w", err)
	}
	tmpArchive, err := os.CreateTemp(root, "em-bundle-*.dl")
	if err != nil {
		return "", err
	}
	tmpName := tmpArchive.Name()
	tmpArchive.Close()
	defer os.Remove(tmpName)

	if err := downloadEMBundle(ctx, m.BundleURL, tmpName, nodeToken); err != nil {
		return "", fmt.Errorf("download EM bundle: %w", err)
	}
	if strings.TrimSpace(m.BundleSHA256) != "" {
		got := fileSHA256(tmpName)
		if got == "" {
			return "", fmt.Errorf("hash EM bundle archive")
		}
		if !strings.EqualFold(got, strings.TrimSpace(m.BundleSHA256)) {
			return "", fmt.Errorf("EM bundle digest mismatch: got %s want %s", got, m.BundleSHA256)
		}
	}

	// Extract into a fresh dir (replace any partial cache).
	_ = os.RemoveAll(bundleDir)
	if err := os.MkdirAll(bundleDir, 0o755); err != nil {
		return "", err
	}
	if err := extractEMBundle(tmpName, bundleDir); err != nil {
		_ = os.RemoveAll(bundleDir)
		return "", fmt.Errorf("extract EM bundle: %w", err)
	}
	if _, statErr := os.Stat(entry); statErr != nil {
		_ = os.RemoveAll(bundleDir)
		return "", fmt.Errorf("EM bundle entrypoint %q not found after extract", m.Entrypoint)
	}
	if runtime.GOOS != "windows" {
		_ = os.Chmod(entry, 0o755)
	}
	_ = os.WriteFile(marker, []byte(m.BundleSHA256), 0o644)
	slog.Info("EM runtime bundle ready", "engine", m.Engine, "version", m.EngineVersion, "dir", bundleDir)
	return entry, nil
}

func downloadEMBundle(ctx context.Context, rawURL, dest, nodeToken string) error {
	if err := validateDownloadURL(rawURL, allowLoopbackDownloads()); err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return err
	}
	if strings.TrimSpace(nodeToken) != "" {
		req.Header.Set("X-Node-Token", strings.TrimSpace(nodeToken))
	}
	client := restrictedHTTPClient(30*time.Minute, allowLoopbackDownloads())
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, resp.Body)
	return err
}

// extractEMBundle unpacks a .tar.gz or .zip bundle into destDir with path-jail
// protection against archive traversal (Zip-Slip / tar traversal).
func extractEMBundle(archivePath, destDir string) error {
	lower := strings.ToLower(archivePath)
	// Sniff by magic bytes since the temp name has a .dl suffix.
	f, err := os.Open(archivePath)
	if err != nil {
		return err
	}
	magic := make([]byte, 4)
	_, _ = io.ReadFull(f, magic)
	_ = f.Close()
	isZip := len(magic) >= 2 && magic[0] == 'P' && magic[1] == 'K'
	isGzip := len(magic) >= 2 && magic[0] == 0x1f && magic[1] == 0x8b
	if isZip || strings.HasSuffix(lower, ".zip") {
		return extractZipBundle(archivePath, destDir)
	}
	if isGzip || strings.HasSuffix(lower, ".tar.gz") || strings.HasSuffix(lower, ".tgz") {
		return extractTarGzBundle(archivePath, destDir)
	}
	return fmt.Errorf("unsupported EM bundle archive format")
}

func safeJoin(destDir, name string) (string, error) {
	target := filepath.Join(destDir, filepath.FromSlash(name))
	root := filepath.Clean(destDir)
	cleaned := filepath.Clean(target)
	if cleaned != root && !strings.HasPrefix(cleaned, root+string(os.PathSeparator)) {
		return "", fmt.Errorf("unsafe archive path %q", name)
	}
	return cleaned, nil
}

func extractTarGzBundle(archivePath, destDir string) error {
	f, err := os.Open(archivePath)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		target, err := safeJoin(destDir, hdr.Name)
		if err != nil {
			return err
		}
		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			out, err := os.OpenFile(target, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, os.FileMode(hdr.Mode)&0o777)
			if err != nil {
				return err
			}
			if _, err := io.Copy(out, io.LimitReader(tr, 2<<30)); err != nil {
				out.Close()
				return err
			}
			out.Close()
		}
	}
}

func extractZipBundle(archivePath, destDir string) error {
	zr, err := zip.OpenReader(archivePath)
	if err != nil {
		return err
	}
	defer zr.Close()
	for _, zf := range zr.File {
		target, err := safeJoin(destDir, zf.Name)
		if err != nil {
			return err
		}
		if zf.FileInfo().IsDir() {
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}
		rc, err := zf.Open()
		if err != nil {
			return err
		}
		out, err := os.OpenFile(target, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, zf.Mode()&0o777)
		if err != nil {
			rc.Close()
			return err
		}
		if _, err := io.Copy(out, io.LimitReader(rc, 2<<30)); err != nil {
			out.Close()
			rc.Close()
			return err
		}
		out.Close()
		rc.Close()
	}
	return nil
}
