package runner

import (
	"crypto/ed25519"
	"encoding/hex"
	"testing"
)

func emSign(t *testing.T, m emBundleManifest, priv ed25519.PrivateKey) emBundleManifest {
	t.Helper()
	m.Signature = hex.EncodeToString(ed25519.Sign(priv, emManifestSigningBytes(m)))
	return m
}

func TestVerifyManifestSignature_Modes(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(nil)
	_, otherPriv, _ := ed25519.GenerateKey(nil)
	base := emBundleManifest{
		Engine: "gprmax", EngineVersion: "1.0", BundleURL: "https://h/b.tgz",
		BundleSHA256: "deadbeef", Entrypoint: "run.py",
	}
	keyHex := hex.EncodeToString(pub)

	t.Run("require+valid signature passes", func(t *testing.T) {
		t.Setenv("RYV_EM_BUNDLE_PUBKEY", keyHex)
		t.Setenv("RYV_EM_REQUIRE_SIGNED_BUNDLE", "1")
		t.Setenv("RYV_EM_ALLOW_UNSIGNED_BUNDLE", "")
		if err := verifyManifestSignature(emSign(t, base, priv)); err != nil {
			t.Fatalf("want pass, got %v", err)
		}
	})

	t.Run("require rejects unsigned even with allow flag", func(t *testing.T) {
		t.Setenv("RYV_EM_BUNDLE_PUBKEY", keyHex)
		t.Setenv("RYV_EM_REQUIRE_SIGNED_BUNDLE", "1")
		t.Setenv("RYV_EM_ALLOW_UNSIGNED_BUNDLE", "1") // must be ignored
		if err := verifyManifestSignature(base); err == nil {
			t.Fatal("want rejection of unsigned under require")
		}
	})

	t.Run("require rejects forged signature", func(t *testing.T) {
		t.Setenv("RYV_EM_BUNDLE_PUBKEY", keyHex)
		t.Setenv("RYV_EM_REQUIRE_SIGNED_BUNDLE", "1")
		if err := verifyManifestSignature(emSign(t, base, otherPriv)); err == nil {
			t.Fatal("want rejection of signature from wrong key")
		}
	})

	t.Run("migration: unsigned allowed with flag (fleet not broken)", func(t *testing.T) {
		t.Setenv("RYV_EM_BUNDLE_PUBKEY", "")
		t.Setenv("RYV_EM_REQUIRE_SIGNED_BUNDLE", "")
		t.Setenv("RYV_EM_ALLOW_UNSIGNED_BUNDLE", "1")
		if err := verifyManifestSignature(base); err != nil {
			t.Fatalf("migration unsigned should pass, got %v", err)
		}
	})

	t.Run("unsigned rejected by default (no allow flag)", func(t *testing.T) {
		t.Setenv("RYV_EM_BUNDLE_PUBKEY", "")
		t.Setenv("RYV_EM_REQUIRE_SIGNED_BUNDLE", "")
		t.Setenv("RYV_EM_ALLOW_UNSIGNED_BUNDLE", "")
		if err := verifyManifestSignature(base); err == nil {
			t.Fatal("unsigned with no allowance should be rejected")
		}
	})

	t.Run("forged signature rejected even without require when key present", func(t *testing.T) {
		t.Setenv("RYV_EM_BUNDLE_PUBKEY", keyHex)
		t.Setenv("RYV_EM_REQUIRE_SIGNED_BUNDLE", "")
		t.Setenv("RYV_EM_ALLOW_UNSIGNED_BUNDLE", "1")
		if err := verifyManifestSignature(emSign(t, base, otherPriv)); err == nil {
			t.Fatal("tampered signature must be rejected when a key is present")
		}
	})
}

// TestEMManifestSigningBytes_Canonical pins the exact signed message so it stays
// byte-for-byte in sync with ryvion-runtimes tools/sign_descriptor.py. os/arch are
// set here but MUST be excluded from the signed bytes (portable signature).
func TestEMManifestSigningBytes_Canonical(t *testing.T) {
	m := emBundleManifest{
		Engine: "gprmax", EngineVersion: "1.0", BundleURL: "https://h/b.tgz",
		BundleSHA256: "deadbeef", Entrypoint: "run.py", OS: "linux", Arch: "amd64",
	}
	got := string(emManifestSigningBytes(m))
	want := "ryvion-em-bundle-v1\ngprmax\n1.0\nhttps://h/b.tgz\ndeadbeef\nrun.py"
	if got != want {
		t.Fatalf("canonical signing bytes drift:\n got=%q\nwant=%q", got, want)
	}
}

func TestEMBundleEntrypointJail(t *testing.T) {
	if _, err := safeJoin("/tmp/bundle", "../../etc/passwd"); err == nil {
		t.Fatal("safeJoin must reject traversal entrypoint")
	}
	if _, err := safeJoin("/tmp/bundle", "bin/run.py"); err != nil {
		t.Fatalf("safeJoin must allow a normal entrypoint, got %v", err)
	}
}
