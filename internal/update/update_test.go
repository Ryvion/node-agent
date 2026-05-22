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

func TestNeedsUpdateDevBuildsAutoUpdate(t *testing.T) {
	// Default-on for unstamped builds: operator installed via `go install`
	// (or older releases that didn't set main.version) should still pick up
	// new releases. Pre-fix this returned false, leaving nodes stuck on
	// whatever they originally compiled.
	if !NeedsUpdate("dev", "v1.2.154") {
		t.Fatal("expected dev build to auto-update to v1.2.154")
	}
	if !NeedsUpdate("", "v1.2.154") {
		t.Fatal("expected empty version to auto-update to v1.2.154")
	}
}

func TestNeedsUpdateRespectsExplicitDisable(t *testing.T) {
	// Active developers can opt out so a local `go run` won't get clobbered.
	t.Setenv("RYV_DISABLE_AUTO_UPDATE", "1")
	if NeedsUpdate("dev", "v1.2.154") {
		t.Fatal("expected RYV_DISABLE_AUTO_UPDATE=1 to suppress dev auto-update")
	}
}

func TestNeedsUpdateSemverComparison(t *testing.T) {
	if !NeedsUpdate("v1.2.150", "v1.2.154") {
		t.Fatal("expected v1.2.150 → v1.2.154 update")
	}
	if NeedsUpdate("v1.2.154", "v1.2.154") {
		t.Fatal("expected same version to skip update")
	}
	if NeedsUpdate("v1.2.155", "v1.2.154") {
		t.Fatal("expected newer current to skip update")
	}
}

func TestNeedsUpdateEmptyLatest(t *testing.T) {
	// Hub may not have advertised a version yet — never update on empty.
	if NeedsUpdate("v1.2.150", "") {
		t.Fatal("expected empty latest to never trigger update")
	}
	if NeedsUpdate("dev", "") {
		t.Fatal("expected empty latest to never trigger update from dev")
	}
}

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
	origKey, origBase := testSigningPublicKeyB64, releaseAssetBaseURL
	defer func() { testSigningPublicKeyB64, releaseAssetBaseURL = origKey, origBase }()
	testSigningPublicKeyB64 = base64.StdEncoding.EncodeToString(pub)

	const version = "v9.9.9"
	checksums := fmt.Sprintf("%s  ryvion-node-%s/%s\n", want, version, name)
	sigB64 := base64.StdEncoding.EncodeToString(ed25519.Sign(priv, []byte(checksums)))

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/" + version + "/SHA256SUMS":
			fmt.Fprint(w, checksums)
		case "/" + version + "/SHA256SUMS.sig":
			fmt.Fprint(w, sigB64)
		default:
			t.Errorf("unexpected path %q", r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	releaseAssetBaseURL = srv.URL

	got, err := fetchExpectedChecksum(context.Background(), version, name)
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
	origKey, origBase := testSigningPublicKeyB64, releaseAssetBaseURL
	defer func() { testSigningPublicKeyB64, releaseAssetBaseURL = origKey, origBase }()
	testSigningPublicKeyB64 = base64.StdEncoding.EncodeToString(pub)

	const version = "v9.9.9"
	checksums := fmt.Sprintf("%s  %s\n", want, name)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/" + version + "/SHA256SUMS":
			fmt.Fprint(w, checksums)
		case "/" + version + "/SHA256SUMS.sig":
			fmt.Fprint(w, base64.StdEncoding.EncodeToString([]byte("bad-signature")))
		default:
			t.Errorf("unexpected path %q", r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	releaseAssetBaseURL = srv.URL

	_, err = fetchExpectedChecksum(context.Background(), version, name)
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

func TestSplitWindowsServiceImageArgs(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "quoted path with args",
			in:   `"C:\Program Files\Ryvion\ryvion-node.exe" -hub https://ryvion-hub.fly.dev -ui-port 45890`,
			want: ` -hub https://ryvion-hub.fly.dev -ui-port 45890`,
		},
		{
			name: "unquoted path with args",
			in:   `C:\Ryvion\ryvion-node.exe -hub https://ryvion-hub.fly.dev`,
			want: ` -hub https://ryvion-hub.fly.dev`,
		},
		{
			name: "no args",
			in:   `"C:\Program Files\Ryvion\ryvion-node.exe"`,
			want: ``,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := splitWindowsServiceImageArgs(tt.in); got != tt.want {
				t.Fatalf("args = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestWindowsInstallRootFromExe(t *testing.T) {
	if got := windowsInstallRootFromExe(`C:\Program Files\Ryvion\ryvion-node.exe`); got != `C:\Program Files\Ryvion` {
		t.Fatalf("install root = %q", got)
	}
	if got := windowsInstallRootFromExe(`C:\Program Files\Ryvion\updates\ryvion-node-abcd.exe`); got != `C:\Program Files\Ryvion` {
		t.Fatalf("staged install root = %q", got)
	}
}

func TestWindowsCanonicalExePath(t *testing.T) {
	if got := windowsCanonicalExePath(`C:\Program Files\Ryvion`); got != `C:\Program Files\Ryvion\ryvion-node.exe` {
		t.Fatalf("canonical exe path = %q", got)
	}
}

func TestWindowsCanonicalRefreshCommandCopiesStagedUpdateToPathBinary(t *testing.T) {
	args := windowsCanonicalRefreshCommand(
		`C:\Program Files\Ryvion\updates\ryvion-node-abcd.exe`,
		`C:\Program Files\Ryvion\ryvion-node.exe`,
	)
	got := strings.Join(args, "\x00")
	if !strings.Contains(got, "cmd.exe\x00/C\x00start\x00\x00powershell.exe") {
		t.Fatalf("unexpected command prefix: %#v", args)
	}
	if !strings.Contains(got, "Copy-Item -LiteralPath $src -Destination $dst -Force") {
		t.Fatalf("refresh command does not copy staged binary to canonical path: %#v", args)
	}
	if !strings.Contains(got, `$src = 'C:\Program Files\Ryvion\updates\ryvion-node-abcd.exe'`) {
		t.Fatalf("staged path was not safely quoted: %#v", args)
	}
	if !strings.Contains(got, `$dst = 'C:\Program Files\Ryvion\ryvion-node.exe'`) {
		t.Fatalf("canonical path was not safely quoted: %#v", args)
	}
}

func TestWindowsServiceStartCommandQuotesServiceName(t *testing.T) {
	args := windowsServiceStartCommand(`Ryvion'Node`)
	got := strings.Join(args, "\x00")
	if !strings.Contains(got, "cmd.exe\x00/C\x00start\x00\x00powershell.exe") {
		t.Fatalf("unexpected command prefix: %#v", args)
	}
	if !strings.Contains(got, "Start-Service -Name 'Ryvion''Node'") {
		t.Fatalf("service name was not PowerShell single-quoted safely: %#v", args)
	}
}

func TestWindowsServiceNameDefaultAndOverride(t *testing.T) {
	t.Setenv("RYVION_WINDOWS_SERVICE", "")
	if got := windowsServiceName(); got != "RyvionNode" {
		t.Fatalf("default service name = %q, want RyvionNode", got)
	}
	t.Setenv("RYVION_WINDOWS_SERVICE", "CustomRyvionNode")
	if got := windowsServiceName(); got != "CustomRyvionNode" {
		t.Fatalf("override service name = %q, want CustomRyvionNode", got)
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
