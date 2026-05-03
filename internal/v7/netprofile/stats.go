package netprofile

import (
	"math"
	"sort"
)

func EstimateStats(samples []float64) (p50, p95 float64) {
	clean := make([]float64, 0, len(samples))
	for _, sample := range samples {
		if math.IsNaN(sample) || math.IsInf(sample, 0) || sample < 0 {
			continue
		}
		clean = append(clean, sample)
	}
	if len(clean) == 0 {
		return 0, 0
	}
	sort.Float64s(clean)
	return percentile(clean, 50), percentile(clean, 95)
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if len(sorted) == 1 {
		return sorted[0]
	}
	if p <= 0 {
		return sorted[0]
	}
	if p >= 100 {
		return sorted[len(sorted)-1]
	}
	pos := (p / 100) * float64(len(sorted)-1)
	lower := int(math.Floor(pos))
	upper := int(math.Ceil(pos))
	if lower == upper {
		return sorted[lower]
	}
	weight := pos - float64(lower)
	return sorted[lower]*(1-weight) + sorted[upper]*weight
}
