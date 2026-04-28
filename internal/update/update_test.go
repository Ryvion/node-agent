package update

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestFetchExpectedChecksumParsesBaseName(t *testing.T) {
	name := expectedArchiveFilename()
	if name == "" {
		t.Skip("unsupported platform")
	}
	want := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("generate key: %v", err)
	}
	t.Setenv("RYV_UPDATE_PUBKEY_B64", base64.StdEncoding.EncodeToString(pub))

	checksums := fmt.Sprintf("%s  releases/%s\n", want, name)
	sigB64 := base64.StdEncoding.EncodeToString(ed25519.Sign(priv, []byte(checksums)))

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v1/downloads/checksums":
			fmt.Fprint(w, checksums)
		case "/api/v1/downloads/checksums.sig":
			fmt.Fprint(w, sigB64)
		default:
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
	}))
	defer srv.Close()

	got, err := fetchExpectedChecksum(context.Background(), srv.URL, name)
	if err != nil {
		t.Fatalf("fetchExpectedChecksum error: %v", err)
	}
	if got != want {
		t.Fatalf("checksum = %q, want %q", got, want)
	}
}

func TestFetchExpectedChecksumRejectsInvalidSignature(t *testing.T) {
	name := expectedArchiveFilename()
	if name == "" {
		t.Skip("unsupported platform")
	}
	want := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	pub, _, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("generate key: %v", err)
	}
	t.Setenv("RYV_UPDATE_PUBKEY_B64", base64.StdEncoding.EncodeToString(pub))

	checksums := fmt.Sprintf("%s  %s\n", want, name)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v1/downloads/checksums":
			fmt.Fprint(w, checksums)
		case "/api/v1/downloads/checksums.sig":
			fmt.Fprint(w, base64.StdEncoding.EncodeToString([]byte("bad-signature")))
		default:
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
	}))
	defer srv.Close()

	_, err = fetchExpectedChecksum(context.Background(), srv.URL, name)
	if err == nil || !strings.Contains(err.Error(), "signature") {
		t.Fatalf("expected signature error, got %v", err)
	}
}

func TestRewriteLaunchAgentBinaryContentReplacesExistingPath(t *testing.T) {
	input := `<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/ryvion-node</string>
        <string>-ui-port</string>
        <string>45890</string>
    </array>
</dict>
</plist>`
	got, changed := rewriteLaunchAgentBinaryContent(input, "/usr/local/bin/ryvion-node", "/Users/daniel/.ryvion/bin/ryvion-node")
	if !changed {
		t.Fatal("expected plist content to change")
	}
	if !strings.Contains(got, "<string>/Users/daniel/.ryvion/bin/ryvion-node</string>") {
		t.Fatalf("expected new binary path in plist, got:\n%s", got)
	}
	if strings.Contains(got, "<string>/usr/local/bin/ryvion-node</string>") {
		t.Fatalf("old binary path still present:\n%s", got)
	}
}

func TestRewriteLaunchAgentBinaryContentFallsBackToFirstProgramArgument(t *testing.T) {
	input := `<plist version="1.0"><dict><key>ProgramArguments</key><array><string>/old/path</string><string>-ui-port</string></array></dict></plist>`
	got, changed := rewriteLaunchAgentBinaryContent(input, "/different/path", "/Users/daniel/.ryvion/bin/ryvion-node")
	if !changed {
		t.Fatal("expected fallback replacement to change content")
	}
	if !strings.Contains(got, "<string>/Users/daniel/.ryvion/bin/ryvion-node</string>") {
		t.Fatalf("expected fallback binary path replacement, got %s", got)
	}
}

func TestSecureHexEqual(t *testing.T) {
	a := "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	if !secureHexEqual(a, a) {
		t.Fatal("expected equal checksums")
	}
	if secureHexEqual(a, "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb") {
		t.Fatal("expected non-equal checksums")
	}
}
