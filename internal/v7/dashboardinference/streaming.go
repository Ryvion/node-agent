package dashboardinference

import (
	"context"
	"errors"
	"strings"
	"sync"
	"time"
)

const (
	defaultChunkBatchSize     = 16
	defaultChunkBatchInterval = 150 * time.Millisecond

	ProgressChunkTypeDelta         = "delta"
	ProgressChunkTypeDone          = "done"
	ProgressChunkTypeTokenCommit   = "token.commit"
	ProgressChunkTypeTokenFinalize = "token.finalize"
)

var ErrStreamProgressFailed = errors.New("dashboardinference: stream progress failed")

type ProgressChunk struct {
	Seq          int64  `json:"seq,omitempty"`
	Type         string `json:"type,omitempty"`
	Text         string `json:"text,omitempty"`
	FinishReason string `json:"finish_reason,omitempty"`
}

type ProgressBatch struct {
	RunID    string
	JobID    string
	NodeID   string
	SeqStart int64
	Chunks   []ProgressChunk
}

type ProgressSender interface {
	SendDashboardInferenceProgress(context.Context, ProgressBatch) error
}

type progressBatcher struct {
	ctx      context.Context
	cancel   context.CancelFunc
	sender   ProgressSender
	spec     Spec
	maxBatch int
	v8Events bool

	mu       sync.Mutex
	flushMu  sync.Mutex
	pending  []ProgressChunk
	nextSeq  int64
	err      error
	disabled bool

	done chan struct{}
}

func newProgressBatcher(ctx context.Context, spec Spec, sender ProgressSender, v8Events bool) *progressBatcher {
	if ctx == nil {
		ctx = context.Background()
	}
	batchCtx, cancel := context.WithCancel(ctx)
	b := &progressBatcher{
		ctx:      batchCtx,
		cancel:   cancel,
		sender:   sender,
		spec:     normalizeSpec(spec),
		maxBatch: defaultChunkBatchSize,
		v8Events: v8Events,
		done:     make(chan struct{}),
	}
	go b.loop()
	return b
}

func (b *progressBatcher) addDelta(text string) error {
	if b == nil || b.sender == nil {
		return nil
	}
	text = strings.TrimRight(text, "\x00")
	if text == "" {
		return nil
	}
	if err := b.currentErr(); err != nil {
		return nil
	}

	b.mu.Lock()
	if b.disabled {
		b.mu.Unlock()
		return nil
	}
	b.nextSeq++
	chunk := ProgressChunk{
		Seq:  b.nextSeq,
		Type: b.deltaChunkType(),
		Text: text,
	}
	b.pending = append(b.pending, chunk)
	shouldFlush := len(b.pending) >= b.maxBatch
	b.mu.Unlock()

	if shouldFlush {
		return b.flush(b.ctx)
	}
	return nil
}

func (b *progressBatcher) addDone(ctx context.Context, finishReason string) error {
	if b == nil || b.sender == nil {
		return nil
	}
	finishReason = cleanFinishReason(finishReason)
	if finishReason == "" {
		finishReason = "unknown"
	}
	if err := b.currentErr(); err != nil {
		return nil
	}
	b.mu.Lock()
	if b.disabled {
		b.mu.Unlock()
		return nil
	}
	b.nextSeq++
	b.pending = append(b.pending, ProgressChunk{
		Seq:          b.nextSeq,
		Type:         b.doneChunkType(),
		FinishReason: finishReason,
	})
	b.mu.Unlock()
	return b.flush(ctx)
}

func (b *progressBatcher) deltaChunkType() string {
	if b != nil && b.v8Events {
		return ProgressChunkTypeTokenCommit
	}
	return ProgressChunkTypeDelta
}

func (b *progressBatcher) doneChunkType() string {
	if b != nil && b.v8Events {
		return ProgressChunkTypeTokenFinalize
	}
	return ProgressChunkTypeDone
}

func (b *progressBatcher) close(ctx context.Context) error {
	if b == nil {
		return nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	flushErr := b.flush(ctx)
	b.cancel()
	<-b.done
	_ = flushErr
	return nil
}

func (b *progressBatcher) loop() {
	defer close(b.done)
	ticker := time.NewTicker(defaultChunkBatchInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			_ = b.flush(b.ctx)
		case <-b.ctx.Done():
			return
		}
	}
}

func (b *progressBatcher) flush(ctx context.Context) error {
	if b == nil || b.sender == nil {
		return nil
	}
	if err := b.currentErr(); err != nil {
		return nil
	}
	if ctx == nil {
		ctx = context.Background()
	}

	b.flushMu.Lock()
	defer b.flushMu.Unlock()

	for {
		b.mu.Lock()
		if b.disabled || len(b.pending) == 0 {
			b.mu.Unlock()
			return nil
		}
		batchSize := len(b.pending)
		if batchSize > b.maxBatch {
			batchSize = b.maxBatch
		}
		chunks := append([]ProgressChunk(nil), b.pending[:batchSize]...)
		b.pending = append([]ProgressChunk(nil), b.pending[batchSize:]...)
		b.mu.Unlock()

		batch := ProgressBatch{
			RunID:    b.spec.RunID,
			JobID:    b.spec.JobID,
			NodeID:   b.spec.TargetNodeID,
			SeqStart: chunks[0].Seq,
			Chunks:   chunks,
		}
		if err := b.sender.SendDashboardInferenceProgress(ctx, batch); err != nil {
			safeErr := codedError{code: "dashboard_inference_stream_progress_failed", err: ErrStreamProgressFailed}
			b.disable(safeErr)
			return nil
		}
	}
}

func (b *progressBatcher) currentErr() error {
	if b == nil {
		return nil
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.err
}

func (b *progressBatcher) setErr(err error) {
	if b == nil || err == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.err == nil {
		b.err = err
	}
}

func (b *progressBatcher) disable(err error) {
	if b == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.disabled = true
	b.pending = nil
	if err != nil && b.err == nil {
		b.err = err
	}
}
