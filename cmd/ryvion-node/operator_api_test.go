package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestAllowLocalOrigin(t *testing.T) {
	t.Parallel()

	allowed := []string{
		"http://localhost:1420",
		"http://127.0.0.1:45890",
		"tauri://localhost",
		"https://tauri.localhost",
	}
	blocked := []string{
		"",
		"https://ryvion.ai",
		"http://example.com",
		"file://local",
	}

	for _, origin := range allowed {
		if !allowLocalOrigin(origin) {
			t.Fatalf("expected origin %q to be allowed", origin)
		}
	}
	for _, origin := range blocked {
		if allowLocalOrigin(origin) {
			t.Fatalf("expected origin %q to be blocked", origin)
		}
	}
}

func TestOperatorAPIPort(t *testing.T) {
	t.Parallel()

	old, hadOld := os.LookupEnv("RYV_UI_PORT")
	defer func() {
		if hadOld {
			_ = os.Setenv("RYV_UI_PORT", old)
			return
		}
		_ = os.Unsetenv("RYV_UI_PORT")
	}()

	_ = os.Unsetenv("RYV_UI_PORT")
	if got := operatorAPIPort(""); got != defaultOperatorAPIPort {
		t.Fatalf("expected default port %q, got %q", defaultOperatorAPIPort, got)
	}
	if got := operatorAPIPort("5000"); got != "5000" {
		t.Fatalf("expected flag port 5000, got %q", got)
	}

	_ = os.Setenv("RYV_UI_PORT", "61234")
	if got := operatorAPIPort("5000"); got != "61234" {
		t.Fatalf("expected env override port 61234, got %q", got)
	}
}

func TestLogRingWriteTail(t *testing.T) {
	t.Parallel()

	ring := newLogRing(3)
	if _, err := ring.Write([]byte("line one\nline two")); err != nil {
		t.Fatalf("write 1: %v", err)
	}
	if _, err := ring.Write([]byte("\nline three\nline four\n")); err != nil {
		t.Fatalf("write 2: %v", err)
	}

	got := ring.tail(10)
	want := []string{"line two", "line three", "line four"}
	if len(got) != len(want) {
		t.Fatalf("expected %d lines, got %d: %#v", len(want), len(got), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("line %d: expected %q, got %q", i, want[i], got[i])
		}
	}
}

func TestStatusTokenParsing(t *testing.T) {
	t.Parallel()

	msg := "docker-cli:present, docker-ready:1, disk_gb:512, native-inference-ready:1"
	if !statusToken(msg, "docker-ready:1") {
		t.Fatal("expected docker-ready token")
	}
	if statusToken(msg, "docker-ready:0") {
		t.Fatal("did not expect docker-ready:0 token")
	}
	if got := statusTokenUint(msg, "disk_gb:"); got != 512 {
		t.Fatalf("expected disk_gb 512, got %d", got)
	}
}

func TestSplitStatusTokens(t *testing.T) {
	t.Parallel()

	got := splitStatusTokens("docker-cli:present, docker-ready:1, , native-inference-ready:1 ")
	want := []string{"docker-cli:present", "docker-ready:1", "native-inference-ready:1"}
	if len(got) != len(want) {
		t.Fatalf("expected %d tokens, got %d: %#v", len(want), len(got), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token %d: expected %q, got %q", i, want[i], got[i])
		}
	}
}

func TestDeriveSovereignPosture(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name            string
		registered      bool
		declaredCountry string
		runtimeReady    bool
		nativeReady     bool
		wantReady       bool
		wantStatus      string
	}{
		{
			name:       "missing country blocks review",
			registered: true,
			wantStatus: "country_missing",
		},
		{
			name:            "registration pending blocks review",
			declaredCountry: "CA",
			runtimeReady:    true,
			wantStatus:      "registration_pending",
		},
		{
			name:            "runtime unavailable blocks review",
			registered:      true,
			declaredCountry: "CA",
			wantStatus:      "runtime_unavailable",
		},
		{
			name:            "local prerequisites satisfied",
			registered:      true,
			declaredCountry: "CA",
			runtimeReady:    true,
			wantReady:       true,
			wantStatus:      "review_ready",
		},
		{
			name:            "native path also satisfies prerequisites",
			registered:      true,
			declaredCountry: "DE",
			nativeReady:     true,
			wantReady:       true,
			wantStatus:      "review_ready",
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			gotReady, gotStatus, gotDetail := deriveSovereignPosture(tc.registered, tc.declaredCountry, tc.runtimeReady, tc.nativeReady)
			if gotReady != tc.wantReady {
				t.Fatalf("ready = %v, want %v", gotReady, tc.wantReady)
			}
			if gotStatus != tc.wantStatus {
				t.Fatalf("status = %q, want %q", gotStatus, tc.wantStatus)
			}
			if gotDetail == "" {
				t.Fatal("expected non-empty detail")
			}
		})
	}
}

func TestNormalizeDeclaredCountry(t *testing.T) {
	t.Parallel()

	if got := normalizeDeclaredCountry("ca"); got != "CA" {
		t.Fatalf("normalizeDeclaredCountry() = %q, want %q", got, "CA")
	}
	if got := normalizeDeclaredCountry(" c1 "); got != "" {
		t.Fatalf("normalizeDeclaredCountry() = %q, want empty", got)
	}
	if got := normalizeDeclaredCountry("CAN"); got != "" {
		t.Fatalf("normalizeDeclaredCountry() = %q, want empty", got)
	}
}

func TestUpdatePublicAIOptInPreservesOtherPreferences(t *testing.T) {
	prevResolver := operatorConfigPathResolver
	configPath := filepath.Join(t.TempDir(), "config.json")
	operatorConfigPathResolver = func() (string, error) {
		return configPath, nil
	}
	defer func() {
		operatorConfigPathResolver = prevResolver
	}()

	if err := saveOperatorPreferences(operatorPreferences{
		PublicAIOptIn:         false,
		DeclaredCountry:       "CA",
		RuntimeChannel:        "managed_oci_v1",
		RuntimeChannelVersion: "2026.04.14",
		RuntimeProvider:       "oci_linux_adapter",
		RuntimeMode:           "host_package",
		RuntimeSource:         "ryvion_runtime_kit",
		RuntimeArtifact:       "artifact.tar.gz",
		RuntimeBackendBinary:  "/opt/ryvion/runtime/backend/ryvion-oci",
		RuntimeManifestHash:   "abc123",
	}); err != nil {
		t.Fatalf("saveOperatorPreferences() error = %v", err)
	}

	state := &operatorRuntime{}
	if err := state.updatePublicAIOptIn(true); err != nil {
		t.Fatalf("updatePublicAIOptIn() error = %v", err)
	}

	got, err := loadOperatorPreferences()
	if err != nil {
		t.Fatalf("loadOperatorPreferences() error = %v", err)
	}
	if !got.PublicAIOptIn {
		t.Fatal("expected public AI opt-in to be updated")
	}
	if got.DeclaredCountry != "CA" || got.RuntimeChannel != "managed_oci_v1" || got.RuntimeArtifact != "artifact.tar.gz" {
		t.Fatalf("expected unrelated preferences to be preserved, got %+v", got)
	}
}
