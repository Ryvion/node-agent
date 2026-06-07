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

// Legacy speculative-decoding flags used by older llama.cpp / Ryvion launches.
// Current upstream llama-server builds reject some of these with
//
//	error: the argument has been removed. use --spec-draft-n-max or --spec-ngram-mod-n-max
//
// and exit immediately. Because streamingSpeculativeLaunchForModel defaults to
// ngram speculative decoding, these flags are added on EVERY native launch — so
// an incompatible binary crash-loops and takes native streaming inference offline
// for every model, not just speculative ones. The agent auto-updates independently
// of the bundled llama-server binary, so a newer agent can be paired with a
// binary whose speculative CLI has changed.
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

var legacySpecLaunchFlags = map[string]bool{
	"--spec-type":          true,
	"--model-draft":        true,
	"--draft-max":          true,
	"--draft":              true,
	"--draft-n":            true,
	"--draft-min":          true,
	"--draft-n-min":        true,
	"--draft-p-min":        true,
	"--n-gpu-layers-draft": true,
}

const specFlagProbeTimeout = 5 * time.Second

// specFlagSupportProbe is the binary-capability probe, overridable in tests.
var specFlagSupportProbe = probeSpecFlagSupport
var legacySpecFlagSupportProbe = probeLegacySpecFlagSupport

var (
	specFlagSupportMu        sync.Mutex
	specFlagSupportCache     = map[string]bool{}
	legacySpecFlagSupportMu  sync.Mutex
	legacySpecFlagSupportMap = map[string]bool{}
)

// specCompatibleArgs strips the custom speculative flags from a llama-server
// argument list when the configured binary cannot parse them. It is a no-op when
// the binary supports them, or when no such flags are present (so a supporting
// fork binary keeps speculative decoding).
func specCompatibleArgs(serverPath string, args []string) []string {
	if hasLegacySpecLaunchFlag(args) && !serverSupportsLegacySpecFlags(serverPath) {
		slog.Warn("llama-server build does not support this legacy speculative CLI; disabling speculative decoding, using standard decoding", "server", serverPath)
		return stripLegacySpecLaunchFlags(args)
	}
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
	return stripFlagValuePairs(args, unsupportedSpecFlags)
}

func hasLegacySpecLaunchFlag(args []string) bool {
	for _, a := range args {
		if legacySpecLaunchFlags[specFlagName(a)] {
			return true
		}
	}
	return false
}

func stripLegacySpecLaunchFlags(args []string) []string {
	return stripFlagValuePairs(args, legacySpecLaunchFlags)
}

func stripFlagValuePairs(args []string, flags map[string]bool) []string {
	out := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		a := args[i]
		if !flags[specFlagName(a)] {
			out = append(out, a)
			continue
		}
		// Drop the flag. For the space-separated form ("--flag value")
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

// serverSupportsLegacySpecFlags reports whether the binary still accepts the
// older speculative launch syntax emitted by streamingSpeculativeLaunchForModel
// (`--draft-max`, `--draft-min`, etc.). Current b9180 upstream advertises those
// names only as removed aliases, so the node must strip the whole legacy
// speculative launch instead of letting llama-server exit before model load.
func serverSupportsLegacySpecFlags(serverPath string) bool {
	serverPath = strings.TrimSpace(serverPath)
	if serverPath == "" {
		return false
	}
	legacySpecFlagSupportMu.Lock()
	defer legacySpecFlagSupportMu.Unlock()
	if v, ok := legacySpecFlagSupportMap[serverPath]; ok {
		return v
	}
	v := legacySpecFlagSupportProbe(serverPath)
	legacySpecFlagSupportMap[serverPath] = v
	return v
}

// probeSpecFlagSupport runs `<serverPath> --help` with the same library search
// path the launcher uses (so the binary's shared libraries resolve) and reports
// whether the help text advertises the --spec-type flag.
func probeSpecFlagSupport(serverPath string) bool {
	out := llamaServerHelpOutput(serverPath)
	return bytes.Contains(out, []byte("--spec-type"))
}

func probeLegacySpecFlagSupport(serverPath string) bool {
	out := llamaServerHelpOutput(serverPath)
	if !bytes.Contains(out, []byte("--draft-max")) {
		return false
	}
	return !bytes.Contains(out, []byte("the argument has been removed"))
}

func llamaServerHelpOutput(serverPath string) []byte {
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
	return out
}
