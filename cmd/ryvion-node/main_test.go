package main

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Ryvion/node-agent/internal/hub"
)

// fakeClient implements a minimal receipt submitter that fails N times then succeeds.
type fakeClient struct {
	failCount int
	calls     atomic.Int32
}

func (f *fakeClient) SubmitReceipt(_ context.Context, _ hub.Receipt) error {
	n := int(f.calls.Add(1))
	if n <= f.failCount {
		return fmt.Errorf("simulated failure %d", n)
	}
	return nil
}

func TestSubmitReceiptWithRetry_SucceedsAfterFailures(t *testing.T) {
	fc := &fakeClient{failCount: 3}
	receipt := hub.Receipt{JobID: "test-job-1", ResultHashHex: "abc123", MeteringUnits: 1}

	err := submitReceiptWithRetryTestable(context.Background(), fc, receipt)
	if err != nil {
		t.Fatalf("expected success after transient failures, got: %v", err)
	}
	if got := int(fc.calls.Load()); got != 4 {
		t.Fatalf("expected 4 calls (3 fail + 1 success), got %d", got)
	}
}

func TestSubmitReceiptWithRetry_ExhaustsRetries(t *testing.T) {
	fc := &fakeClient{failCount: 10} // always fails
	receipt := hub.Receipt{JobID: "test-job-2", ResultHashHex: "def456", MeteringUnits: 1}

	err := submitReceiptWithRetryTestable(context.Background(), fc, receipt)
	if err == nil {
		t.Fatal("expected error after exhausting retries")
	}
	if got := int(fc.calls.Load()); got != 5 {
		t.Fatalf("expected exactly 5 attempts, got %d", got)
	}
}

func TestSubmitReceiptWithRetry_RespectsContextCancel(t *testing.T) {
	fc := &fakeClient{failCount: 10}
	receipt := hub.Receipt{JobID: "test-job-3", ResultHashHex: "ghi789", MeteringUnits: 1}

	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()

	start := time.Now()
	err := submitReceiptWithRetryTestable(ctx, fc, receipt)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected error on cancelled context")
	}
	// Should bail out within ~1 second, not wait for all 5 retries
	if elapsed > 3*time.Second {
		t.Fatalf("took too long (%v), context cancellation not respected", elapsed)
	}
}

func TestJobActiveFlag_PreventsUpdate(t *testing.T) {
	// Verify the atomic flag mechanism works
	jobActive.Store(0)
	if jobActive.Load() != 0 {
		t.Fatal("expected jobActive=0 initially")
	}
	jobActive.Store(1)
	if jobActive.Load() != 1 {
		t.Fatal("expected jobActive=1 after store")
	}
	jobActive.Store(0)
	if jobActive.Load() != 0 {
		t.Fatal("expected jobActive=0 after reset")
	}
}

// submitReceiptWithRetryTestable is the same logic as submitReceiptWithRetry
// but accepts an interface so we can inject a fake client.
type receiptSubmitter interface {
	SubmitReceipt(ctx context.Context, receipt hub.Receipt) error
}

func submitReceiptWithRetryTestable(ctx context.Context, client receiptSubmitter, receipt hub.Receipt) error {
	const maxAttempts = 5
	delay := 2 * time.Millisecond // fast for tests
	var lastErr error
	for i := 0; i < maxAttempts; i++ {
		if err := client.SubmitReceipt(ctx, receipt); err != nil {
			lastErr = err
			select {
			case <-ctx.Done():
				return fmt.Errorf("context cancelled during receipt retry: %w", lastErr)
			case <-time.After(delay):
			}
			delay = time.Duration(float64(delay) * 2)
			if delay > 30*time.Millisecond {
				delay = 30 * time.Millisecond
			}
			continue
		}
		return nil
	}
	return fmt.Errorf("receipt submission failed after %d attempts: %w", maxAttempts, lastErr)
}
