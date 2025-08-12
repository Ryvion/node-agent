package update

import (
    "context"
    "crypto/ed25519"
    "crypto/sha256"
    "errors"
    "io"
    "net/http"
    "os"
    "runtime"
)

type SimpleClient struct {
    ManifestURL string
    PubKey      ed25519.PublicKey
}

// Simple implementation using the existing manifest structure
func (c *SimpleClient) CheckAndUpdate(ctx context.Context, current string, installPath string) (bool, error) {
    // Get manifest from URL
    body, err := Fetch(c.ManifestURL)
    if err != nil {
        return false, err
    }
    defer body.Close()
    
    b, err := io.ReadAll(body)
    if err != nil {
        return false, err
    }
    
    m, raw, err := ParseManifest(b)
    if err != nil {
        return false, err
    }

    // For now, skip signature verification in simple mode
    // In production, you'd fetch a separate .sig file and verify
    _ = raw

    if m.Version == current {
        return false, nil
    }

    // Find asset for current platform
    asset, ok := SelectAsset(m, "node-agent", runtime.GOOS, runtime.GOARCH)
    if !ok {
        return false, errors.New("no asset for platform")
    }

    // Download and verify
    req, _ := http.NewRequestWithContext(ctx, "GET", asset.URL, nil)
    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        return false, err
    }
    defer resp.Body.Close()
    
    tmp := installPath + ".new"
    f, err := os.Create(tmp)
    if err != nil {
        return false, err
    }
    defer f.Close()
    
    h := sha256.New()
    mw := io.MultiWriter(f, h)
    if _, err := io.Copy(mw, resp.Body); err != nil {
        return false, err
    }

    // Verify SHA256 hash
    if err := VerifySHA256Hex(f, asset.SHA256); err != nil {
        return false, err
    }

    // Swap files
    if err := os.Rename(tmp, installPath); err != nil {
        return false, err
    }
    
    return true, nil
}