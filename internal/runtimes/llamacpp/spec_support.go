package llamacpp

import (
	"bytes"
	"context"
	"log/slog"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// Custom speculative-decoding flags accepted only by the Ryvion llama.cpp fork.
// Stock upstream llama-server builds reject them with
//
//	error: invalid argument: --spec-draft-n-max
//
// and exit immediately, which crash-loops the runtime and takes the node's
// native streaming inference offline. The node agent auto-updates independently
// of the bundled llama-server binary, so a newer agent (which defaults
// SpecType to ngram-simple) can end up paired with a stock binary. We
// feature-detect the binary and strip these flags when unsupported, falling
// back to standard decoding instead of trusting config blindly.
var unsupportedSpecFlags = map[string]bool{
	"--spec-type":        true,
	"--spec-draft-n-max": true,
	"--spec-draft-n-min": true,
}

const specFlagProbeTimeout = 5 * time.Second

// specFlagSupportProbe is the binary-capability probe, overridable in tests.
var specFlagSupportProbe = probeSpecFlagSupport

var (
	specFlagSupportMu    sync.Mutex
	specFlagSupportCache = map[string]bool{}
)

// specCompatibleArgs strips the custom speculative flags from a llama-server
// argument list when the configured binary cannot parse them. It is a no-op
// when the binary supports them, or when no such flags are present (so a
// supporting binary keeps speculative decoding).
func specCompatibleArgs(serverPath string, args []string) []string {
	if hasUnsupportedSpecFlag(args) && !serverSupportsSpecFlags(serverPath) {
		return stripUnsupportedSpecFlags(args)
	}
	return args
}

// hasUnsupportedSpecFlag reports whether args contain any custom --spec-* flag
// (matching both "--flag value" and "--flag=value" forms).
func hasUnsupportedSpecFlag(args []string) bool {
	for _, a := range args {
		if unsupportedSpecFlags[specFlagName(a)] {
			return true
		}
	}
	return false
}

// stripUnsupportedSpecFlags removes the custom --spec-* flags and their values
// from a llama-server argument list, leaving a stock-compatible command line.
func stripUnsupportedSpecFlags(args []string) []string {
	out := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		a := args[i]
		if !unsupportedSpecFlags[specFlagName(a)] {
			out = append(out, a)
			continue
		}
		// Drop the flag. For the space-separated form ("--spec-type ngram-simple")
		// also drop the following value token; the "--flag=value" form is
		// self-contained.
		if !strings.ContainsRune(a, '=') && i+1 < len(args) && !strings.HasPrefix(args[i+1], "-") {
			i++
		}
	}
	return out
}

func specFlagName(arg string) string {
	if eq := strings.IndexByte(arg, '='); eq > 0 {
		return arg[:eq]
	}
	return arg
}

// serverSupportsSpecFlags reports whether the llama-server binary at serverPath
// understands the custom --spec-type speculative flags. The result is cached
// per binary path (the binary does not change mid-process). Fail-closed: an
// empty path or a probe error reports "unsupported", so we never feed a binary
// flags it cannot parse.
func serverSupportsSpecFlags(serverPath string) bool {
	serverPath = strings.TrimSpace(serverPath)
	if serverPath == "" {
		return false
	}
	specFlagSupportMu.Lock()
	defer specFlagSupportMu.Unlock()
	if v, ok := specFlagSupportCache[serverPath]; ok {
		return v
	}
	v := specFlagSupportProbe(serverPath)
	specFlagSupportCache[serverPath] = v
	if !v {
		slog.Warn("llama-server build does not support --spec-type; disabling speculative decoding, using standard decoding", "server", serverPath)
	}
	return v
}

// probeSpecFlagSupport runs `<serverPath> --help` with the same library search
// path the launcher uses (so the binary's dynamic libraries resolve) and
// reports whether the help text advertises the --spec-type flag.
func probeSpecFlagSupport(serverPath string) bool {
	ctx, cancel := context.WithTimeout(context.Background(), specFlagProbeTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, serverPath, "--help")
	configureProcessCommand(cmd, serverPath)
	out, _ := cmd.CombinedOutput()
	return bytes.Contains(out, []byte("--spec-type"))
}
