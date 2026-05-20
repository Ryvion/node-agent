package main

import (
	"testing"
	"time"

	"github.com/Ryvion/ryvion-node/internal/hw"
	"github.com/Ryvion/ryvion-node/internal/network/profile"
)

func TestBuildV7HeartbeatPayloadIncludesHubNetworkTelemetry(t *testing.T) {
	previous := hubNetworkProfile
	hubNetworkProfile = netprofile.NewRollingHubProfile(8, "https://api.ryvion.ai/api/v1/ping")
	t.Cleanup(func() { hubNetworkProfile = previous })

	hubNetworkProfile.RecordRTT(42*time.Millisecond, time.UnixMilli(1770000000000))
	hubNetworkProfile.RecordRTT(58*time.Millisecond, time.UnixMilli(1770000000100))

	payload, err := buildV7HeartbeatPayloadForNode("node-network", hw.CapSet{CPUCores: 8, RAMBytes: 16 << 30}, "cpu", "US", nil, nil)
	if err != nil {
		t.Fatalf("buildV7HeartbeatPayloadForNode() error = %v", err)
	}
	if payload.NetworkProfile == nil {
		t.Fatal("NetworkProfile = nil, want hub moving average")
	}
	if payload.NetworkProfile.RTTMsP95 <= 0 || payload.NetworkProfile.JitterMsP95 <= 0 {
		t.Fatalf("NetworkProfile = %#v, want ping and jitter", payload.NetworkProfile)
	}
}
