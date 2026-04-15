package nodekey

import (
	"bytes"
	"crypto/ed25519"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadOrCreateCreatesAndReloads(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "node-key")

	pub1, priv1, err := LoadOrCreate(path)
	if err != nil {
		t.Fatalf("load/create failed: %v", err)
	}
	if len(priv1) != ed25519.PrivateKeySize {
		t.Fatalf("unexpected private key size: %d", len(priv1))
	}
	if len(pub1) != ed25519.PublicKeySize {
		t.Fatalf("unexpected public key size: %d", len(pub1))
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read key file: %v", err)
	}
	decoded, err := hex.DecodeString(string(raw))
	if err != nil {
		t.Fatalf("hex decode key file: %v", err)
	}
	if !bytes.Equal(decoded, priv1) {
		t.Fatalf("stored key mismatch")
	}

	pub2, priv2, err := LoadOrCreate(path)
	if err != nil {
		t.Fatalf("reload failed: %v", err)
	}
	if !bytes.Equal(pub1, pub2) {
		t.Fatalf("public key changed across reload")
	}
	if !bytes.Equal(priv1, priv2) {
		t.Fatalf("private key changed across reload")
	}
}

func TestPublicKeyHexMatchesLoadedPublicKey(t *testing.T) {
	tmp := t.TempDir()
	path := filepath.Join(tmp, "node-key")

	pub, _, err := LoadOrCreate(path)
	if err != nil {
		t.Fatalf("load/create failed: %v", err)
	}

	got, err := PublicKeyHex(path)
	if err != nil {
		t.Fatalf("public key hex failed: %v", err)
	}
	want := hex.EncodeToString(pub)
	if got != want {
		t.Fatalf("public key hex = %q, want %q", got, want)
	}
}
