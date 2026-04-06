//go:build linux

package inference

import (
	"fmt"
	"syscall"
)

func checkDiskSpace(path string) error {
	var stat syscall.Statfs_t
	if err := syscall.Statfs(path, &stat); err != nil {
		return nil
	}
	freeBytes := stat.Bavail * uint64(stat.Bsize)
	if freeBytes < minFreeSpaceBytes {
		return fmt.Errorf("insufficient disk space: %d MB free, need at least %d MB", freeBytes/1024/1024, minFreeSpaceBytes/1024/1024)
	}
	return nil
}
