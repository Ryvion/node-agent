package crypto

import (
	"crypto/ed25519"
	crand "crypto/rand"
	"encoding/hex"
	"log"
	"os"
	"path/filepath"
)

func keyPath() string {
	if p := os.Getenv("AK_KEY_PATH"); p != "" {
		return p
	}

	dataDir := os.Getenv("HOME")
	if dataDir == "" {
		dataDir = "/root"
	}

	ryvionDir := filepath.Join(dataDir, ".ryvion")
	os.MkdirAll(ryvionDir, 0700)

	return filepath.Join(ryvionDir, "node-key")
}

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
	if err != nil {
		log.Fatalf("keygen failed: %v", err)
	}
	_ = os.WriteFile(kp, []byte(hex.EncodeToString(priv)), 0600)
	return pub, priv
}
