package netprofile

import (
	"testing"
	"time"
)

func TestRollingHubProfileComputesMovingPingAndJitter(t *testing.T) {
	rolling := NewRollingHubProfile(4, "https://api.ryvion.ai/api/v1/ping")
	for _, sample := range []time.Duration{
		40 * time.Millisecond,
		50 * time.Millisecond,
		70 * time.Millisecond,
		90 * time.Millisecond,
	} {
		rolling.RecordRTT(sample, time.UnixMilli(1770000000000))
	}

	profile, ok := rolling.Snapshot()
	if !ok {
		t.Fatal("Snapshot() ok = false, want true")
	}
	if profile.SampleCount != 4 {
		t.Fatalf("sample_count = %d, want 4", profile.SampleCount)
	}
	if profile.ProbeTarget != "https://api.ryvion.ai/api/v1/ping" {
		t.Fatalf("probe_target = %q", profile.ProbeTarget)
	}
	if profile.RTTMsP50 <= 0 || profile.RTTMsP95 <= profile.RTTMsP50 {
		t.Fatalf("unexpected RTT stats: p50=%f p95=%f", profile.RTTMsP50, profile.RTTMsP95)
	}
	if profile.JitterMsP95 <= 0 {
		t.Fatalf("jitter_ms_p95 = %f, want positive jitter", profile.JitterMsP95)
	}
}

func TestRollingHubProfileKeepsBoundedWindow(t *testing.T) {
	rolling := NewRollingHubProfile(2, "")
	rolling.RecordRTT(10*time.Millisecond, time.UnixMilli(1))
	rolling.RecordRTT(20*time.Millisecond, time.UnixMilli(2))
	rolling.RecordRTT(100*time.Millisecond, time.UnixMilli(3))

	profile, ok := rolling.Snapshot()
	if !ok {
		t.Fatal("Snapshot() ok = false, want true")
	}
	if profile.SampleCount != 2 {
		t.Fatalf("sample_count = %d, want bounded window of 2", profile.SampleCount)
	}
	if profile.RTTMsP50 < 20 || profile.RTTMsP95 < 20 {
		t.Fatalf("old sample was not evicted: %#v", profile)
	}
}
