package inference

import (
	"math"
	"path/filepath"
	"strings"
	"testing"
)

func TestValidateAvailableDiskSpaceIncludesDownloadAndReserve(t *testing.T) {
	const gib = uint64(1024 * 1024 * 1024)
	if err := validateAvailableDiskSpace(15*gib, 5*gib); err != nil {
		t.Fatalf("exact download plus reserve should fit: %v", err)
	}
	if err := validateAvailableDiskSpace(15*gib-1, 5*gib); err == nil {
		t.Fatal("download must not consume the 10 GiB operator reserve")
	}
}

func TestValidateAvailableDiskSpaceRejectsOverflow(t *testing.T) {
	err := validateAvailableDiskSpace(math.MaxUint64, math.MaxUint64)
	if err == nil || !strings.Contains(err.Error(), "supported disk accounting") {
		t.Fatalf("expected overflow-safe rejection, got %v", err)
	}
}

func TestAvailableDiskBytesUsesNearestExistingParent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "models", "pending", "model.gguf")
	available, err := AvailableDiskBytes(path)
	if err != nil {
		t.Fatalf("probe future download path: %v", err)
	}
	if available == 0 {
		t.Fatal("expected non-zero available disk capacity")
	}
}
