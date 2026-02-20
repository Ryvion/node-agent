//go:build windows

package main

import (
	"context"
	"log/slog"

	"golang.org/x/sys/windows/svc"
)

func isWindowsService() bool {
	ok, err := svc.IsWindowsService()
	if err != nil {
		slog.Warn("failed to detect service mode", "error", err)
		return false
	}
	return ok
}

func runAsWindowsService() {
	if err := svc.Run("RyvionNode", &ryvionService{}); err != nil {
		slog.Error("windows service run failed", "error", err)
	}
}

type ryvionService struct{}

func (s *ryvionService) Execute(args []string, r <-chan svc.ChangeRequest, changes chan<- svc.Status) (bool, uint32) {
	const accepted = svc.AcceptStop | svc.AcceptShutdown
	changes <- svc.Status{State: svc.StartPending}

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		runNode(ctx)
	}()

	changes <- svc.Status{State: svc.Running, Accepts: accepted}
	slog.Info("windows service running")

	for {
		select {
		case c := <-r:
			switch c.Cmd {
			case svc.Interrogate:
				changes <- c.CurrentStatus
			case svc.Stop, svc.Shutdown:
				slog.Info("windows service stop requested")
				changes <- svc.Status{State: svc.StopPending}
				cancel()
				<-done
				return false, 0
			}
		case <-done:
			return false, 0
		}
	}
}
