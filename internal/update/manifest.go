package update

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"runtime"
	"strings"
)

type Manifest struct {
	Version   string  `json:"version"`
	CreatedAt string  `json:"created_at"`
	Binaries  []Asset `json:"binaries"`
	Runners   []Asset `json:"runners"`
	NotesURL  string  `json:"notes_url,omitempty"`
}

type Asset struct {
	Name   string `json:"name"`
	OS     string `json:"os"`
	Arch   string `json:"arch"`
	URL    string `json:"url"`
	SHA256 string `json:"sha256"`
}

func ParseManifest(b []byte) (Manifest, []byte, error) {
	var m Manifest
	if err := json.Unmarshal(b, &m); err != nil {
		return Manifest{}, nil, err
	}
	return m, b, nil
}

func VerifyDetached(payload []byte, sigB64 string, pubKeyHex string) error {
	pk, err := hex.DecodeString(strings.TrimSpace(pubKeyHex))
	if err != nil {
		return fmt.Errorf("invalid pubkey hex: %w", err)
	}
	sig, err := base64.StdEncoding.DecodeString(strings.TrimSpace(sigB64))
	if err != nil {
		return fmt.Errorf("invalid signature b64: %w", err)
	}
	if len(pk) != ed25519.PublicKeySize {
		return errors.New("bad pubkey length")
	}
	if len(sig) != ed25519.SignatureSize {
		return errors.New("bad signature length")
	}
	if !ed25519.Verify(ed25519.PublicKey(pk), payload, sig) {
		return errors.New("signature verify failed")
	}
	return nil
}

func SelectAsset(m Manifest, name, goos, goarch string) (Asset, bool) {
	norm := func(s string) string { return strings.ToLower(strings.TrimSpace(s)) }
	name = norm(name)
	goos = norm(goos)
	goarch = norm(goarch)
	pick := func(list []Asset) (Asset, bool) {
		for _, a := range list {
			if norm(a.Name) == name && norm(a.OS) == goos && norm(a.Arch) == goarch {
				return a, true
			}
		}
		return Asset{}, false
	}
	if a, ok := pick(m.Binaries); ok {
		return a, true
	}
	if a, ok := pick(m.Runners); ok {
		return a, true
	}
	return Asset{}, false
}

func VerifySHA256Hex(r io.Reader, expectedHex string) error {
	h := sha256.New()
	if _, err := io.Copy(h, r); err != nil {
		return err
	}
	sum := h.Sum(nil)
	got := hex.EncodeToString(sum)
	if !strings.EqualFold(got, strings.TrimSpace(expectedHex)) {
		return fmt.Errorf("sha256 mismatch: got %s want %s", got, expectedHex)
	}
	return nil
}

func Fetch(url string) (io.ReadCloser, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != 200 {
		resp.Body.Close()
		return nil, fmt.Errorf("http %d", resp.StatusCode)
	}
	return resp.Body, nil
}

func CurrentPlatform() (string, string) { return runtime.GOOS, runtime.GOARCH }
