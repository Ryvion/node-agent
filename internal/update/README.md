Signed update manifest

Overview
- File `manifest.json`: JSON describing version and assets
- File `manifest.sig`: base64-encoded ed25519 signature over the raw bytes of `manifest.json`
- Signer key: 32-byte ed25519 public key distributed with the agent

Minimal schema
{
  "version": "v0.1.0",
  "created_at": "2025-08-10T12:34:56Z",
  "binaries": [
    { "name": "node-agent", "os": "windows", "arch": "amd64", "url": "https://…/agent-win.exe", "sha256": "…" },
    { "name": "node-agent", "os": "linux",   "arch": "amd64", "url": "https://…/agent-linux",   "sha256": "…" }
  ],
  "runners": [
    { "name": "ffmpeg-nvenc", "os": "linux", "arch": "amd64", "url": "https://…/ffmpeg.tgz", "sha256": "…" }
  ],
  "notes_url": "https://…/releases/v0.1.0"
}

Verification snippet (Go)
 m, raw, _ := update.ParseManifest(manifestBytes)
 if err := update.VerifyDetached(raw, sigB64, signerPubHex); err != nil { panic(err) }
 asset, ok := update.SelectAsset(m, "node-agent", runtime.GOOS, runtime.GOARCH)
 if ok {
   rc, _ := update.Fetch(asset.URL)
   defer rc.Close()
   if err := update.VerifySHA256Hex(rc, asset.SHA256); err != nil { panic(err) }
 }

Signing tip
- Avoid inline signatures inside JSON to prevent canonicalization issues; keep a detached `.sig` file.

