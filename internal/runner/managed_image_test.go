package runner

import "testing"

func TestValidateManagedRunnerImage(t *testing.T) {
	allowed := []string{
		"ghcr.io/ryvion/llm-runner:latest",
		"ghcr.io/ryvion/em-fdtd-runner:0.1.0",
		"GHCR.IO/Ryvion/whisper-runner:latest", // case-insensitive
		"ghcr.io/ryvion/llm-runner@sha256:" + "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
		"registry.example.com/foo@sha256:" + "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef", // pinned digest anywhere
	}
	for _, img := range allowed {
		if err := validateManagedRunnerImage(img); err != nil {
			t.Errorf("expected %q allowed, got %v", img, err)
		}
	}

	rejected := []string{
		"",
		"docker.io/library/alpine:latest",
		"evilregistry.com/cryptominer:latest",
		"ghcr.io/attacker/payload:latest",
		"ghcr.io/ryvion-evil/payload:latest", // prefix must be ghcr.io/ryvion/ exactly
		"alpine",
	}
	for _, img := range rejected {
		if err := validateManagedRunnerImage(img); err == nil {
			t.Errorf("expected %q rejected, got nil error", img)
		}
	}
}
