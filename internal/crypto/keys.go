package crypto

import (
    crand "crypto/rand"
    "crypto/ed25519"
    "encoding/hex"
    "log"
    "os"
    "path/filepath"
)

func keyPath() string {
    // Check for explicit config path first
    if p := os.Getenv("AK_KEY_PATH"); p != "" {
        return p
    }
    
    // Use a more reliable directory structure
    dataDir := os.Getenv("HOME")
    if dataDir == "" {
        dataDir = "/root" // Default for Docker containers
    }
    
    // Create .akatosh directory if it doesn't exist
    akatoshDir := filepath.Join(dataDir, ".akatosh")
    os.MkdirAll(akatoshDir, 0700)
    
    return filepath.Join(akatoshDir, "node-key")
}

// LoadOrCreateKey stores a hex-encoded ed25519 private key on disk (dev-only).
func LoadOrCreateKey() (ed25519.PublicKey, ed25519.PrivateKey) {
    kp := keyPath()
    if b, err := os.ReadFile(kp); err == nil {
        skBytes, err := hex.DecodeString(string(b))
        if err == nil && len(skBytes) == ed25519.PrivateKeySize {
            sk := ed25519.PrivateKey(skBytes)
            return sk.Public().(ed25519.PublicKey), sk
        }
    }
    pub, priv, err := ed25519.GenerateKey(crand.Reader)
    if err != nil { log.Fatalf("keygen failed: %v", err) }
    _ = os.WriteFile(kp, []byte(hex.EncodeToString(priv)), 0600)
    return pub, priv
}

