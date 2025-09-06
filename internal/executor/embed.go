package executor

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
	"os/exec"
	"time"
)

// Embedding wrapper that executes an external embedding binary defined by EMBED_CMD (for example, "llama-embed").
type EmbedSpec struct {
	InputFile  string
	OutputFile string
	Timeout    time.Duration
	Cmd        string
	Args       []string
}

func RunEmbedding(ctx context.Context, spec EmbedSpec) (hash string, err error) {
	if spec.Cmd == "" {
		spec.Cmd = "llama-embed"
	}

	args := make([]string, 0, len(spec.Args))
	for _, a := range spec.Args {
		if a == "{input}" {
			a = spec.InputFile
		}
		if a == "{output}" {
			a = spec.OutputFile
		}
		args = append(args, a)
	}

	cctx, cancel := context.WithTimeout(ctx, spec.Timeout)
	defer cancel()

	cmd := exec.CommandContext(cctx, spec.Cmd, args...)
	if out, err := cmd.CombinedOutput(); err != nil {
		_ = out
		return "", err
	}

	f, err := os.Open(spec.OutputFile)
	if err != nil {
		return "", err
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}

	sum := hex.EncodeToString(h.Sum(nil))
	return sum, nil
}
