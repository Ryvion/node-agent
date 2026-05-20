package inference

import (
	"fmt"
	"io"
	"os"
)

const minFreeSpaceBytes = 10 * 1024 * 1024 * 1024 // 10 GB

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
