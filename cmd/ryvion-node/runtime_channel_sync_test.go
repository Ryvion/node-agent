package main

import "testing"

func TestRuntimeAutoSyncRequiresExplicitOptIn(t *testing.T) {
	for _, raw := range []string{"", "0", "false", "off"} {
		t.Run("disabled="+raw, func(t *testing.T) {
			t.Setenv("RYV_RUNTIME_AUTO_SYNC", raw)
			if runtimeAutoSyncEnabled() {
				t.Fatalf("RYV_RUNTIME_AUTO_SYNC=%q must not provision host software", raw)
			}
		})
	}
	for _, raw := range []string{"1", "true", "yes", "on"} {
		t.Run("enabled="+raw, func(t *testing.T) {
			t.Setenv("RYV_RUNTIME_AUTO_SYNC", raw)
			if !runtimeAutoSyncEnabled() {
				t.Fatalf("RYV_RUNTIME_AUTO_SYNC=%q should explicitly enable sync", raw)
			}
		})
	}
}

func TestRuntimeContractFromManifestWindows(t *testing.T) {
	t.Setenv("ProgramFiles", `C:\Program Files`)
	manifest := runtimeChannelManifest{
		Channel:      "managed_oci_v1",
		Version:      "2026.04.28.1",
		ManifestHash: "hash123",
		Platforms: map[string]runtimeChannelPlatform{
			"windows": {
				Provider: "oci_desktop_adapter",
				Mode:     "host_package",
				Source:   "ryvion_runtime_kit",
				Artifact: runtimeChannelArtifact{FileName: "ryvion-runtime-kit-windows-amd64-2026.04.28.1.zip"},
			},
		},
	}
	meta, ok := runtimeContractFromManifest(manifest, "windows")
	if !ok {
		t.Fatal("expected windows platform")
	}
	if meta.Version != "2026.04.28.1" || meta.ManifestHash != "hash123" || meta.Provider != "oci_desktop_adapter" {
		t.Fatalf("unexpected metadata: %+v", meta)
	}
	if meta.Binary != `C:\Program Files\Ryvion\runtime\ryvion-runtime.cmd` {
		t.Fatalf("binary = %q", meta.Binary)
	}
	if meta.Backend != `C:\Program Files\Ryvion\runtime\backend\ryvion-oci.cmd` {
		t.Fatalf("backend = %q", meta.Backend)
	}
}

func TestRuntimeContractFromManifestLinux(t *testing.T) {
	manifest := runtimeChannelManifest{
		Channel:      "managed_oci_v1",
		Version:      "2026.04.28.1",
		ManifestHash: "hash123",
		Platforms: map[string]runtimeChannelPlatform{
			"linux": {
				Provider: "oci_linux_adapter",
				Mode:     "host_package",
				Source:   "ryvion_runtime_kit",
				Artifact: runtimeChannelArtifact{FileName: "ryvion-runtime-kit-linux-amd64-2026.04.28.1.tar.gz"},
			},
		},
	}
	meta, ok := runtimeContractFromManifest(manifest, "linux")
	if !ok {
		t.Fatal("expected linux platform")
	}
	if meta.Binary != "/opt/ryvion/runtime/ryvion-runtime" {
		t.Fatalf("binary = %q", meta.Binary)
	}
	if meta.Backend != "/opt/ryvion/runtime/backend/ryvion-oci" {
		t.Fatalf("backend = %q", meta.Backend)
	}
	if meta.Artifact != "ryvion-runtime-kit-linux-amd64-2026.04.28.1.tar.gz" {
		t.Fatalf("artifact = %q", meta.Artifact)
	}
}
