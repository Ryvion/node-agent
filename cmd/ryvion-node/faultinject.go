//go:build faultinject

package main

import (
	"log/slog"
	"os"
	"strings"
)

// maybeFaultInject simulates a post-startup crash so the Layer-2 crash-loop
// rollback can be exercised end-to-end on a real node. It is ONLY compiled with
// `-tags faultinject` and crashes ONLY when RYV_FAULT_CRASH_VERSION exactly
// matches this build's version — so the rolled-back-to (good) binary, built at a
// different version, runs cleanly and the drill recovers. Never present in a
// production build.
func maybeFaultInject(version string) {
	want := strings.TrimSpace(os.Getenv("RYV_FAULT_CRASH_VERSION"))
	if want != "" && want == strings.TrimSpace(version) {
		slog.Error("FAULT INJECTION: simulating post-startup crash for rollback drill", "version", version)
		os.Exit(7)
	}
}
