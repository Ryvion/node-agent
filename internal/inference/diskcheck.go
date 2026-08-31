package inference

import (
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
)

const minFreeSpaceBytes uint64 = 10 * 1024 * 1024 * 1024 // 10 GiB

func checkDiskSpace(path string) error {
	return checkDiskSpaceFor(path, 0)
}

func checkDiskSpaceFor(path string, requiredBytes uint64) error {
	freeBytes, err := AvailableDiskBytes(path)
	if err != nil {
		return fmt.Errorf("inspect available disk space: %w", err)
	}
	return validateAvailableDiskSpace(freeBytes, requiredBytes)
}

func validateAvailableDiskSpace(freeBytes, requiredBytes uint64) error {
	if requiredBytes > math.MaxUint64-minFreeSpaceBytes {
		return fmt.Errorf("download size exceeds supported disk accounting")
	}
	neededBytes := minFreeSpaceBytes + requiredBytes
	if freeBytes < neededBytes {
		return fmt.Errorf(
			"insufficient disk space: %d MiB free, need %d MiB for the download plus %d MiB reserve",
			freeBytes/1024/1024,
			requiredBytes/1024/1024,
			minFreeSpaceBytes/1024/1024,
		)
	}
	return nil
}

func availableDownloadBudget(path string) (uint64, error) {
	freeBytes, err := AvailableDiskBytes(path)
	if err != nil {
		return 0, fmt.Errorf("inspect available disk space: %w", err)
	}
	if err := validateAvailableDiskSpace(freeBytes, 0); err != nil {
		return 0, err
	}
	return freeBytes - minFreeSpaceBytes, nil
}

// AvailableDiskBytes reports writable capacity on the filesystem that will
// contain path. Download destinations commonly do not exist yet, so walk to
// the nearest existing parent instead of failing or probing the process CWD.
func AvailableDiskBytes(path string) (uint64, error) {
	probe := strings.TrimSpace(path)
	if probe == "" {
		probe = "."
	}
	for {
		if _, err := os.Stat(probe); err == nil {
			return availableDiskBytes(probe)
		} else if !os.IsNotExist(err) {
			return 0, err
		}
		parent := filepath.Dir(probe)
		if parent == probe {
			return 0, fmt.Errorf("no existing parent for %q", path)
		}
		probe = parent
	}
}

func validateGGUF(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	magic := make([]byte, 4)
	if _, err := io.ReadFull(f, magic); err != nil {
		return fmt.Errorf("failed to read GGUF header: %w", err)
	}
	if string(magic) != "GGUF" {
		return fmt.Errorf("invalid GGUF file: magic bytes %x", magic)
	}
	return nil
}
