package nodekey

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// LoadOrCreate loads an Ed25519 private key from disk, or creates one if missing.
func LoadOrCreate(path string) (ed25519.PublicKey, ed25519.PrivateKey, error) {
	if strings.TrimSpace(path) == "" {
		defaultPath, err := defaultKeyPath()
		if err != nil {
			return nil, nil, err
		}
		path = defaultPath
	}

	if b, err := os.ReadFile(path); err == nil {
		raw := strings.TrimSpace(string(b))
		privBytes, decErr := hex.DecodeString(raw)
		if decErr == nil && len(privBytes) == ed25519.PrivateKeySize {
			priv := ed25519.PrivateKey(privBytes)
			pub, ok := priv.Public().(ed25519.PublicKey)
			if !ok {
				return nil, nil, errors.New("invalid ed25519 public key type")
			}
			return pub, priv, nil
		}
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, nil, fmt.Errorf("create key directory: %w", err)
	}

	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, nil, fmt.Errorf("generate key: %w", err)
	}
	if err := os.WriteFile(path, []byte(hex.EncodeToString(priv)), 0o600); err != nil {
		return nil, nil, fmt.Errorf("write key: %w", err)
	}
	return pub, priv, nil
}

func defaultKeyPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("resolve user home: %w", err)
	}
	return filepath.Join(home, ".ryvion", "node-key"), nil
}
