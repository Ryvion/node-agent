//go:build !linux

package inference

func checkDiskSpace(_ string) error {
	return nil // Disk check not supported on this platform
}
