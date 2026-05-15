package netprofile

import (
	"strings"
	"sync"
	"time"
)

const defaultRollingHubProfileWindow = 8

// RollingHubProfile keeps a small moving window of hub round-trip samples.
// It is intentionally local and lightweight: the heartbeat path records the
// actual HTTP heartbeat duration, and the next V7 heartbeat advertises the
// aggregate as a network profile.
type RollingHubProfile struct {
	mu     sync.Mutex
	max    int
	target string
	rtts   []float64
	atMs   int64
}

func NewRollingHubProfile(maxSamples int, target string) *RollingHubProfile {
	if maxSamples <= 0 {
		maxSamples = defaultRollingHubProfileWindow
	}
	return &RollingHubProfile{
		max:    maxSamples,
		target: cleanRollingTarget(target),
	}
}

func (r *RollingHubProfile) SetTarget(target string) {
	if r == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.target = cleanRollingTarget(target)
}

func (r *RollingHubProfile) RecordRTT(duration time.Duration, observedAt time.Time) {
	if r == nil || duration <= 0 {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.max <= 0 {
		r.max = defaultRollingHubProfileWindow
	}
	r.rtts = append(r.rtts, durationMillis(duration))
	if len(r.rtts) > r.max {
		copy(r.rtts, r.rtts[len(r.rtts)-r.max:])
		r.rtts = r.rtts[:r.max]
	}
	if observedAt.IsZero() {
		observedAt = time.Now()
	}
	r.atMs = observedAt.UnixMilli()
}

func (r *RollingHubProfile) Snapshot() (NetworkProfile, bool) {
	if r == nil {
		return NetworkProfile{}, false
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.rtts) == 0 {
		return NetworkProfile{}, false
	}
	rtts := append([]float64(nil), r.rtts...)
	p50, p95 := EstimateStats(rtts)
	_, jitterP95 := EstimateStats(rollingJitterSamples(rtts))
	profile := NetworkProfile{
		RTTMsP50:         p50,
		RTTMsP95:         p95,
		JitterMsP95:      jitterP95,
		LossRateP95:      0,
		ProbeTarget:      r.target,
		SampleCount:      len(rtts),
		MeasuredAtUnixMs: r.atMs,
	}
	if profile.MeasuredAtUnixMs <= 0 {
		profile.MeasuredAtUnixMs = time.Now().UnixMilli()
	}
	return profile, ValidateNetworkProfile(profile) == nil
}

func rollingJitterSamples(rtts []float64) []float64 {
	if len(rtts) < 2 {
		return nil
	}
	jitters := make([]float64, 0, len(rtts)-1)
	for i := 1; i < len(rtts); i++ {
		delta := rtts[i] - rtts[i-1]
		if delta < 0 {
			delta = -delta
		}
		jitters = append(jitters, delta)
	}
	return jitters
}

func cleanRollingTarget(target string) string {
	target = strings.TrimSpace(target)
	target = strings.ReplaceAll(target, "\r", "")
	target = strings.ReplaceAll(target, "\n", "")
	target = strings.ReplaceAll(target, "\t", "")
	if len(target) > 256 {
		return target[:256]
	}
	return target
}
