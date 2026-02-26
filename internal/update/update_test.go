package update

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestFetchExpectedChecksumParsesBaseName(t *testing.T) {
	name := expectedArchiveFilename()
	if name == "" {
		t.Skip("unsupported platform")
	}
	want := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/downloads/checksums" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		fmt.Fprintf(w, "%s  releases/%s\n", want, name)
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

func TestSecureHexEqual(t *testing.T) {
	a := "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	if !secureHexEqual(a, a) {
		t.Fatal("expected equal checksums")
	}
	if secureHexEqual(a, "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb") {
		t.Fatal("expected non-equal checksums")
	}
}
