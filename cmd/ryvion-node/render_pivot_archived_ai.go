package main

import (
	"context"
	"time"

	"github.com/Ryvion/ryvion-node/internal/hub"
	usefulenergy "github.com/Ryvion/ryvion-node/internal/telemetry/usefulenergy"
)

type jobEnergyStart struct {
	startedAt time.Time
	snapshot  usefulenergy.Snapshot
}

func startWorkGraphAbortMonitor(context.Context, interface{}, *hub.WorkAssignment, context.CancelFunc) func() {
	return func() {}
}

func jobEnergyReceiptMetadata(*hub.WorkAssignment) map[string]any {
	return nil
}

func processOptionalSpeculativeNativeDraftHotSession(context.Context, *hub.Client, *hub.WorkAssignment, *runtimeManager, bool) (bool, *runnerResultSnapshot, error) {
	return false, nil, nil
}

func processOptionalSpeculativeNativeVerifierHotSession(context.Context, *hub.Client, *hub.WorkAssignment, *runtimeManager, bool) (bool, *runnerResultSnapshot, error) {
	return false, nil, nil
}

func processOptionalSpeculativeNativeDraft(context.Context, *hub.Client, *hub.WorkAssignment, *runtimeManager, bool) (bool, *runnerResultSnapshot, error) {
	return false, nil, nil
}

func processOptionalSpeculativeNativeVerifier(context.Context, *hub.Client, *hub.WorkAssignment, *runtimeManager, bool) (bool, *runnerResultSnapshot, error) {
	return false, nil, nil
}
