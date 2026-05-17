//go:build linux

package hardware

import (
	"os"

	"golang.org/x/sys/unix"
)

func defaultDiskFreeBytes(path string) (uint64, error) {
	existing, err := existingPath(path, os.Stat)
	if err != nil {
		return 0, err
	}
	var stat unix.Statfs_t
	if err := unix.Statfs(existing, &stat); err != nil {
		return 0, err
	}
	return stat.Bavail * uint64(stat.Bsize), nil
}
