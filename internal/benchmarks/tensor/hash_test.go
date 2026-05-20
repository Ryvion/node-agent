package tensorplane

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestHashTensorPageDeterministic(t *testing.T) {
	page := testFloat32Page(t, []float32{
		1, 0,
		0, 1,
	}, []float32{
		2, 4,
		10, 20,
	})

	first, err := HashTensorPage(page)
	if err != nil {
		t.Fatalf("first HashTensorPage() error = %v", err)
	}
	second, err := HashTensorPage(page)
	if err != nil {
		t.Fatalf("second HashTensorPage() error = %v", err)
	}
	if first == "" || first != second {
		t.Fatalf("HashTensorPage() = %q then %q, want deterministic non-empty hash", first, second)
	}
}

func TestHashTensorPageChangesWithTensorValue(t *testing.T) {
	page := testFloat32Page(t, []float32{
		1, 0,
		0, 1,
	}, []float32{
		2, 4,
		10, 20,
	})

	first, err := HashTensorPage(page)
	if err != nil {
		t.Fatalf("first HashTensorPage() error = %v", err)
	}
	binary.LittleEndian.PutUint32(page.ValueData[0:4], math.Float32bits(3))
	second, err := HashTensorPage(page)
	if err != nil {
		t.Fatalf("second HashTensorPage() error = %v", err)
	}
	if first == second {
		t.Fatalf("page hash unchanged after tensor mutation: %s", first)
	}
}

func TestHashTensorPartialAttentionSummaryExcludesComputeTime(t *testing.T) {
	page := testFloat32Page(t, []float32{
		1, 0,
		0, 1,
	}, []float32{
		2, 4,
		10, 20,
	})
	summary, err := ComputeTensorPartialAttentionSummary(testAttentionQuery(), page)
	if err != nil {
		t.Fatalf("ComputeTensorPartialAttentionSummary() error = %v", err)
	}
	changed := summary
	changed.ComputeTimeUs += 1000

	first, err := HashTensorPartialAttentionSummary(summary)
	if err != nil {
		t.Fatalf("first HashTensorPartialAttentionSummary() error = %v", err)
	}
	second, err := HashTensorPartialAttentionSummary(changed)
	if err != nil {
		t.Fatalf("second HashTensorPartialAttentionSummary() error = %v", err)
	}
	if first != second {
		t.Fatalf("summary hash changed with compute_time_us: %s vs %s", first, second)
	}
}
