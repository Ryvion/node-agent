//go:build !linux && !darwin && !windows

package inference

import "fmt"

func availableDiskBytes(_ string) (uint64, error) {
	return 0, fmt.Errorf("disk accounting is unsupported on this platform")
}
