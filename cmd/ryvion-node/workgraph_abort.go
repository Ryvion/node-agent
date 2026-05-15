package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/Ryvion/node-agent/internal/hub"
)

type workGraphAbortChecker interface {
	FetchWorkGraphAbort(context.Context, string) (*hub.WorkGraphAbort, error)
}

func startWorkGraphAbortMonitor(ctx context.Context, client workGraphAbortChecker, work *hub.WorkAssignment, cancel context.CancelFunc) func() {
	return startWorkGraphAbortMonitorWithInterval(ctx, client, work, cancel, workGraphAbortPollInterval(os.Getenv))
}

func startWorkGraphAbortMonitorWithInterval(ctx context.Context, client workGraphAbortChecker, work *hub.WorkAssignment, cancel context.CancelFunc, interval time.Duration) func() {
	workGraphID := workGraphIDForAssignment(work)
	if ctx == nil || client == nil || cancel == nil || workGraphID == "" {
		return func() {}
	}
	if interval <= 0 {
		interval = 2 * time.Second
	}
	monitorCtx, stop := context.WithCancel(ctx)
	go func() {
		check := func() bool {
			checkCtx, checkCancel := context.WithTimeout(monitorCtx, 5*time.Second)
			abort, err := client.FetchWorkGraphAbort(checkCtx, workGraphID)
			checkCancel()
			if err != nil {
				slog.Debug("workgraph abort check failed", "job_id", work.JobID, "error", err)
				return false
			}
			if abort == nil {
				return false
			}
			slog.Info("workgraph abort received; cancelling active runner",
				"job_id", work.JobID,
				"workgraph_hash", abort.WorkGraphHash,
				"abort_epoch", abort.AbortEpoch,
				"reason", abort.Reason,
			)
			cancel()
			return true
		}
		if check() {
			return
		}
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-monitorCtx.Done():
				return
			case <-ticker.C:
				if check() {
					return
				}
			}
		}
	}()
	return stop
}

func workGraphIDForAssignment(work *hub.WorkAssignment) string {
	if work == nil {
		return ""
	}
	if id := strings.TrimSpace(work.WorkGraphID); id != "" {
		return id
	}
	var spec map[string]any
	if json.Unmarshal([]byte(work.SpecJSON), &spec) != nil {
		return ""
	}
	for _, key := range []string{"workgraph_id", "work_graph_id"} {
		if id, ok := spec[key].(string); ok && strings.TrimSpace(id) != "" {
			return strings.TrimSpace(id)
		}
	}
	if nested, ok := spec["workgraph"].(map[string]any); ok {
		if id, ok := nested["graph_id"].(string); ok && strings.TrimSpace(id) != "" {
			return strings.TrimSpace(id)
		}
		if id, ok := nested["workgraph_id"].(string); ok && strings.TrimSpace(id) != "" {
			return strings.TrimSpace(id)
		}
	}
	return ""
}

func workGraphAbortPollInterval(getenv func(string) string) time.Duration {
	if getenv == nil {
		getenv = os.Getenv
	}
	raw := strings.TrimSpace(getenv("RYV_WORKGRAPH_ABORT_POLL_INTERVAL"))
	if raw != "" {
		if d, err := time.ParseDuration(raw); err == nil && d > 0 {
			return clampWorkGraphAbortPollInterval(d)
		}
	}
	rawMs := strings.TrimSpace(getenv("RYV_WORKGRAPH_ABORT_POLL_INTERVAL_MS"))
	if rawMs != "" {
		if ms, err := strconv.Atoi(rawMs); err == nil && ms > 0 {
			return clampWorkGraphAbortPollInterval(time.Duration(ms) * time.Millisecond)
		}
	}
	return 2 * time.Second
}

func clampWorkGraphAbortPollInterval(interval time.Duration) time.Duration {
	if interval < 250*time.Millisecond {
		return 250 * time.Millisecond
	}
	if interval > 30*time.Second {
		return 30 * time.Second
	}
	return interval
}
