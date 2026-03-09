package main

import (
	"os"
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
