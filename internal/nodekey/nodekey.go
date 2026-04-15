package nodekey

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// Load loads an existing Ed25519 private key from disk.
func Load(path string) (ed25519.PublicKey, ed25519.PrivateKey, error) {
	if strings.TrimSpace(path) == "" {
		defaultPath, err := defaultKeyPath()
		if err != nil {
			return nil, nil, err
		}
		path = defaultPath
	}

	b, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	raw := strings.TrimSpace(string(b))
	privBytes, decErr := hex.DecodeString(raw)
	if decErr != nil || len(privBytes) != ed25519.PrivateKeySize {
		return nil, nil, errors.New("invalid ed25519 private key encoding")
	}
	priv := ed25519.PrivateKey(privBytes)
	pub, ok := priv.Public().(ed25519.PublicKey)
	if !ok {
		return nil, nil, errors.New("invalid ed25519 public key type")
	}
	return pub, priv, nil
}

// LoadOrCreate loads an Ed25519 private key from disk, or creates one if missing.
func LoadOrCreate(path string) (ed25519.PublicKey, ed25519.PrivateKey, error) {
	if pub, priv, err := Load(path); err == nil {
		return pub, priv, nil
	}
	if strings.TrimSpace(path) == "" {
		defaultPath, err := defaultKeyPath()
		if err != nil {
			return nil, nil, err
		}
		path = defaultPath
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

// PublicKeyHex returns the node's public key as a hex string.
func PublicKeyHex(path string) (string, error) {
	pub, _, err := Load(path)
	if err != nil {
		return "", err
	}
	return hex.EncodeToString(pub), nil
}

func defaultKeyPath() (string, error) {
	if runtime.GOOS == "windows" {
		pd := os.Getenv("ProgramData")
		if pd == "" {
			pd = `C:\ProgramData`
		}
		return filepath.Join(pd, "Ryvion", "node-key"), nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("resolve user home: %w", err)
	}
	return filepath.Join(home, ".ryvion", "node-key"), nil
}
