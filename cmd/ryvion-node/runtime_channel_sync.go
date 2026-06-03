package main

import (
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"
)

type runtimeChannelManifest struct {
	Channel      string                            `json:"channel"`
	Version      string                            `json:"version"`
	ManifestHash string                            `json:"manifest_hash"`
	Platforms    map[string]runtimeChannelPlatform `json:"platforms"`
}

type runtimeChannelPlatform struct {
	Provider string                 `json:"provider"`
	Mode     string                 `json:"mode"`
	Source   string                 `json:"source"`
	Artifact runtimeChannelArtifact `json:"artifact"`
}

type runtimeChannelArtifact struct {
	FileName string `json:"file_name"`
}

func runtimeAutoSyncEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("RYV_RUNTIME_AUTO_SYNC"))) {
	case "0", "false", "no", "off":
		return false
	default:
		return true
	}
}

func syncManagedRuntimeFromHub(ctx context.Context, hubURL string, runtimeMgr *runtimeManager) error {
	if runtimeMgr == nil || !runtimeAutoSyncEnabled() || ociLaneDisabled() {
		return nil
	}
	if runtime.GOOS != "windows" && runtime.GOOS != "linux" {
		return nil
	}
	manifest, err := fetchRuntimeChannelManifest(ctx, hubURL)
	if err != nil {
		return err
	}
	meta, ok := runtimeContractFromManifest(manifest, runtime.GOOS)
	if !ok {
		return fmt.Errorf("runtime channel has no platform entry for %s", runtime.GOOS)
	}
	current := runtimeMgr.contract
	needsSync := strings.TrimSpace(current.ManifestHash) != strings.TrimSpace(meta.ManifestHash) ||
		strings.TrimSpace(current.Version) != strings.TrimSpace(meta.Version)
	if !needsSync && runtime.GOOS != "darwin" {
		if _, ok := resolveFlux2LocalHelper(); !ok {
			needsSync = true
		}
	}
	if !needsSync {
		return nil
	}
	if err := runRuntimeBootstrap(ctx, hubURL); err != nil {
		return err
	}
	runtimeMgr.UpdateContract(meta)
	_, _ = mutateOperatorPreferences(func(p *operatorPreferences) {
		p.RuntimeChannel = meta.Channel
		p.RuntimeChannelVersion = meta.Version
		p.RuntimeProvider = meta.Provider
		p.RuntimeMode = meta.Mode
		p.RuntimeSource = meta.Source
		p.RuntimeArtifact = meta.Artifact
		p.RuntimeBinary = meta.Binary
		p.RuntimeBackendBinary = meta.Backend
		p.RuntimeEngineBinary = meta.Engine
		p.RuntimeEngineKind = meta.EngineKind
		p.RuntimeManifestHash = meta.ManifestHash
	})
	return nil
}

func fetchRuntimeChannelManifest(ctx context.Context, hubURL string) (runtimeChannelManifest, error) {
	var manifest runtimeChannelManifest
	base := strings.TrimRight(strings.TrimSpace(hubURL), "/")
	if base == "" {
		return manifest, fmt.Errorf("hub URL is empty")
	}
	endpoint := base + "/api/v1/runtime/channel/current"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return manifest, err
	}
	client := &http.Client{Timeout: 20 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return manifest, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return manifest, fmt.Errorf("runtime channel returned HTTP %d", resp.StatusCode)
	}
	if err := json.NewDecoder(resp.Body).Decode(&manifest); err != nil {
		return manifest, err
	}
	return manifest, nil
}

func runtimeContractFromManifest(manifest runtimeChannelManifest, goos string) (runtimeContractMetadata, bool) {
	platform, ok := manifest.Platforms[goos]
	if !ok {
		return runtimeContractMetadata{}, false
	}
	meta := runtimeContractMetadata{
		Channel:      strings.TrimSpace(manifest.Channel),
		Version:      strings.TrimSpace(manifest.Version),
		Provider:     strings.TrimSpace(platform.Provider),
		Mode:         strings.TrimSpace(platform.Mode),
		Source:       strings.TrimSpace(platform.Source),
		Artifact:     strings.TrimSpace(platform.Artifact.FileName),
		ManifestHash: strings.TrimSpace(manifest.ManifestHash),
	}
	switch goos {
	case "windows":
		programFiles := strings.TrimSpace(os.Getenv("ProgramFiles"))
		if programFiles == "" {
			programFiles = `C:\Program Files`
		}
		root := programFiles + `\Ryvion\runtime`
		meta.Binary = root + `\ryvion-runtime.cmd`
		meta.Backend = root + `\backend\ryvion-oci.cmd`
	case "linux":
		root := "/opt/ryvion/runtime"
		meta.Binary = root + "/ryvion-runtime"
		meta.Backend = root + "/backend/ryvion-oci"
	}
	return meta, true
}

// runtimeBootstrapPubKey returns the Ed25519 public key used to verify the hub's
// runtime-bootstrap script, from RYV_RUNTIME_BOOTSTRAP_PUBKEY (hex or base64).
// When unset, the script runs UNVERIFIED (migration default) with a warning.
func runtimeBootstrapPubKey() ed25519.PublicKey {
	raw := strings.TrimSpace(os.Getenv("RYV_RUNTIME_BOOTSTRAP_PUBKEY"))
	if raw == "" {
		return nil
	}
	if b, err := hex.DecodeString(raw); err == nil && len(b) == ed25519.PublicKeySize {
		return ed25519.PublicKey(b)
	}
	if b, err := base64.StdEncoding.DecodeString(raw); err == nil && len(b) == ed25519.PublicKeySize {
		return ed25519.PublicKey(b)
	}
	return nil
}

func isLoopbackHostname(host string) bool {
	host = strings.ToLower(strings.TrimSpace(host))
	return host == "localhost" || host == "127.0.0.1" || host == "::1"
}

// fetchBootstrapBytes downloads up to 8 MiB over HTTPS (http only for a loopback
// hub, for dev). Returns the body bytes.
func fetchBootstrapBytes(ctx context.Context, rawURL string) ([]byte, error) {
	u, err := url.Parse(rawURL)
	if err != nil || u.Hostname() == "" {
		return nil, fmt.Errorf("invalid bootstrap url")
	}
	if !strings.EqualFold(u.Scheme, "https") {
		if !(strings.EqualFold(u.Scheme, "http") && isLoopbackHostname(u.Hostname())) {
			return nil, fmt.Errorf("bootstrap url must use https")
		}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, err
	}
	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("bootstrap fetch HTTP %d", resp.StatusCode)
	}
	return io.ReadAll(io.LimitReader(resp.Body, 8<<20))
}

// verifyBootstrapScript enforces an Ed25519 signature over the script bytes when
// a verifying key is configured. The detached signature (hex or base64) is served
// at <scriptURL>.sig. Without a key it logs a warning and allows (migration).
func verifyBootstrapScript(ctx context.Context, scriptURL string, script []byte) error {
	pub := runtimeBootstrapPubKey()
	if pub == nil {
		slog.Warn("runtime bootstrap: running UNVERIFIED hub script — set RYV_RUNTIME_BOOTSTRAP_PUBKEY and serve <url>.sig to enforce signatures")
		return nil
	}
	sigRaw, err := fetchBootstrapBytes(ctx, scriptURL+".sig")
	if err != nil {
		return fmt.Errorf("bootstrap signature fetch failed: %w", err)
	}
	sigStr := strings.TrimSpace(string(sigRaw))
	var sig []byte
	if b, e := hex.DecodeString(sigStr); e == nil && len(b) == ed25519.SignatureSize {
		sig = b
	} else if b, e := base64.StdEncoding.DecodeString(sigStr); e == nil && len(b) == ed25519.SignatureSize {
		sig = b
	}
	if len(sig) == 0 {
		return fmt.Errorf("bootstrap signature malformed")
	}
	if !ed25519.Verify(pub, script, sig) {
		return fmt.Errorf("bootstrap signature verification failed")
	}
	return nil
}

func runRuntimeBootstrap(ctx context.Context, hubURL string) error {
	base := strings.TrimRight(strings.TrimSpace(hubURL), "/")
	if base == "" {
		return fmt.Errorf("hub URL is empty")
	}
	bootstrapCtx, cancel := context.WithTimeout(ctx, 15*time.Minute)
	defer cancel()

	var scriptURL, ext string
	switch runtime.GOOS {
	case "windows":
		scriptURL, ext = base+"/runtime/windows/bootstrap.ps1", ".ps1"
	case "linux":
		scriptURL, ext = base+"/runtime/linux/bootstrap.sh", ".sh"
	default:
		return nil
	}
	if _, err := url.ParseRequestURI(scriptURL); err != nil {
		return err
	}

	// Download the script, verify it (when a key is configured), then execute the
	// FILE — never pipe a network stream to a shell (iex / curl|bash), so the bytes
	// that run are exactly the bytes we fetched and verified.
	script, err := fetchBootstrapBytes(bootstrapCtx, scriptURL)
	if err != nil {
		return fmt.Errorf("runtime bootstrap download failed: %w", err)
	}
	if err := verifyBootstrapScript(bootstrapCtx, scriptURL, script); err != nil {
		return err
	}

	tmp, err := os.CreateTemp("", "ryv-bootstrap-*"+ext)
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(script); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}

	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = exec.CommandContext(bootstrapCtx, "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", tmpName)
	case "linux":
		_ = os.Chmod(tmpName, 0o700)
		cmd = exec.CommandContext(bootstrapCtx, "bash", tmpName)
	}
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("runtime bootstrap failed: %w: %s", err, tailString(string(out), 2048))
	}
	return nil
}

