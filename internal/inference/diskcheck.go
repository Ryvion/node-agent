package inference

import (
	"fmt"
	"io"
	"os"
	"runtime"
	"syscall"
)

const minFreeSpaceBytes = 10 * 1024 * 1024 * 1024 // 10 GB

func checkDiskSpace(path string) error {
	if runtime.GOOS == "windows" {
		return nil // Skip on Windows — no simple syscall
	}
	var stat syscall.Statfs_t
	if err := syscall.Statfs(path, &stat); err != nil {
		return nil // Best effort — don't block on stat failure
	}
	freeBytes := stat.Bavail * uint64(stat.Bsize)
	if freeBytes < minFreeSpaceBytes {
		return fmt.Errorf("insufficient disk space: %d MB free, need at least %d MB", freeBytes/1024/1024, minFreeSpaceBytes/1024/1024)
	}
	return nil
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
