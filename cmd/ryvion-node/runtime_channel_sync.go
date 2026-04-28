package main

import (
	"context"
	"encoding/json"
	"fmt"
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

func runRuntimeBootstrap(ctx context.Context, hubURL string) error {
	base := strings.TrimRight(strings.TrimSpace(hubURL), "/")
	if base == "" {
		return fmt.Errorf("hub URL is empty")
	}
	bootstrapCtx, cancel := context.WithTimeout(ctx, 15*time.Minute)
	defer cancel()
	switch runtime.GOOS {
	case "windows":
		scriptURL := base + "/runtime/windows/bootstrap.ps1"
		if _, err := url.ParseRequestURI(scriptURL); err != nil {
			return err
		}
		ps := "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; " +
			"iex ((New-Object System.Net.WebClient).DownloadString('" + scriptURL + "'))"
		cmd := exec.CommandContext(bootstrapCtx, "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps)
		out, err := cmd.CombinedOutput()
		if err != nil {
			return fmt.Errorf("runtime bootstrap failed: %w: %s", err, tailString(string(out), 2048))
		}
		return nil
	case "linux":
		scriptURL := base + "/runtime/linux/bootstrap.sh"
		if _, err := url.ParseRequestURI(scriptURL); err != nil {
			return err
		}
		cmd := exec.CommandContext(bootstrapCtx, "bash", "-lc", "curl -fsSL "+shellQuote(scriptURL)+" | bash")
		out, err := cmd.CombinedOutput()
		if err != nil {
			return fmt.Errorf("runtime bootstrap failed: %w: %s", err, tailString(string(out), 2048))
		}
		return nil
	default:
		return nil
	}
}

func shellQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "'\\''") + "'"
}
