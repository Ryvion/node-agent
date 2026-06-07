package inference

import (
	"bytes"
	"context"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// Custom speculative-decoding flags accepted ONLY by the Ryvion llama.cpp fork.
// Stock upstream llama-server builds reject them with
//
//	error: invalid argument: --spec-type
//
// and exit immediately. Because streamingSpeculativeLaunchForModel defaults the
// method to ngram-simple, these flags are added on EVERY native launch — so a
// stock binary crash-loops and takes the node's native streaming inference
// offline for every model, not just speculative ones. The agent auto-updates
// independently of the bundled llama-server binary, so a newer agent can be
// paired with a stock binary.
//
// This mirrors internal/runtimes/llamacpp/spec_support.go, which already gates the
// V7 sidecar's launches. The legacy native streaming path (runServerNative, used
// by RunStreamingJob) builds its own command line and was never wired through that
// guard — leaving stock-binary nodes process-failed. Kept in this package to keep
// the streaming launcher self-contained (no cross-package dependency), matching the
// existing duplication of the jinja/reasoning helpers.
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
// argument list when the configured binary cannot parse them. It is a no-op when
// the binary supports them, or when no such flags are present (so a supporting
// fork binary keeps speculative decoding).
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
// understands the custom --spec-type speculative flags. Cached per binary path
// (the binary does not change mid-process). Fail-closed: an empty path or a probe
// error reports "unsupported", so we never feed a binary flags it cannot parse.
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
// path the launcher uses (so the binary's shared libraries resolve) and reports
// whether the help text advertises the --spec-type flag.
func probeSpecFlagSupport(serverPath string) bool {
	ctx, cancel := context.WithTimeout(context.Background(), specFlagProbeTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, serverPath, "--help")
	binDir := filepath.Dir(serverPath)
	env := os.Environ()
	if runtime.GOOS == "windows" {
		env = append(env, "PATH="+binDir+";"+os.Getenv("PATH"))
	} else {
		env = append(env, "DYLD_LIBRARY_PATH="+binDir, "LD_LIBRARY_PATH="+binDir)
	}
	cmd.Env = env
	out, _ := cmd.CombinedOutput()
	return bytes.Contains(out, []byte("--spec-type"))
}
