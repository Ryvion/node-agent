//go:build !faultinject

package main

// maybeFaultInject is a NO-OP in production builds. The crashing variant is only
// compiled with `-tags faultinject` (for the auto-update rollback drill), so a
// production binary can never self-destruct through this path.
func maybeFaultInject(version string) {}
