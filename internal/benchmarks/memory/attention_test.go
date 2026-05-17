package memorybench

import (
	"errors"
	"math"
	"strings"
	"testing"
)

func TestComputePartialAttentionSummaryMatchesNaiveSoftmax(t *testing.T) {
	request := SyntheticAttentionRequest{
		RequestID:       "request-1",
		JobID:           "job-1",
		ShardID:         "shard-a",
		Seed:            7,
		Logits:          []float64{1.25, -0.5, 2.75, 0.25},
		Values:          [][]float64{{1, 2, 3}, {-1, 0, 1}, {4, -2, 0.5}, {0.25, 0.5, -0.75}},
		ValueDim:        3,
		CreatedAtUnixMs: 1_800_000_000_123,
	}

	response, err := ComputePartialAttentionSummary(request)
	if err != nil {
		t.Fatalf("ComputePartialAttentionSummary() error = %v", err)
	}
	naive, err := NaiveAttentionOutput(request.Logits, request.Values)
	if err != nil {
		t.Fatalf("NaiveAttentionOutput() error = %v", err)
	}

	if response.Summary.LocalMax != 2.75 {
		t.Fatalf("local max = %v, want 2.75", response.Summary.LocalMax)
	}
	if response.Summary.TokenCount != len(request.Logits) {
		t.Fatalf("token count = %d, want %d", response.Summary.TokenCount, len(request.Logits))
	}
	if response.Summary.ValueDim != request.ValueDim {
		t.Fatalf("value dim = %d, want %d", response.Summary.ValueDim, request.ValueDim)
	}
	if response.Summary.DType != SyntheticAttentionDTypeFloat64 {
		t.Fatalf("dtype = %q, want %q", response.Summary.DType, SyntheticAttentionDTypeFloat64)
	}
	if response.Summary.ExpSum <= 0 {
		t.Fatalf("exp sum = %v, want positive", response.Summary.ExpSum)
	}
	for dim := range response.Summary.WeightedValue {
		got := response.Summary.WeightedValue[dim] / response.Summary.ExpSum
		if !nearlyEqual(got, naive[dim]) {
			t.Fatalf("normalized weighted value[%d] = %v, want %v", dim, got, naive[dim])
		}
	}
}

func TestComputePartialAttentionSummaryRejectsInvalidDimensions(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(*SyntheticAttentionRequest)
		want string
	}{
		{
			name: "missing request id",
			edit: func(request *SyntheticAttentionRequest) {
				request.RequestID = " "
			},
			want: "request_id required",
		},
		{
			name: "missing job id",
			edit: func(request *SyntheticAttentionRequest) {
				request.JobID = ""
			},
			want: "job_id required",
		},
		{
			name: "zero value dim",
			edit: func(request *SyntheticAttentionRequest) {
				request.ValueDim = 0
			},
			want: "value_dim must be greater than zero",
		},
		{
			name: "logits values mismatch",
			edit: func(request *SyntheticAttentionRequest) {
				request.Values = request.Values[:1]
			},
			want: "logits and values length mismatch",
		},
		{
			name: "value length mismatch",
			edit: func(request *SyntheticAttentionRequest) {
				request.Values[0] = []float64{1}
			},
			want: "length must equal value_dim",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request := validSyntheticAttentionRequest()
			tc.edit(&request)

			_, err := ComputePartialAttentionSummary(request)
			if !errors.Is(err, ErrInvalidSyntheticAttentionRequest) || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("ComputePartialAttentionSummary() error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestComputePartialAttentionSummaryRejectsNaNAndInf(t *testing.T) {
	for _, tc := range []struct {
		name string
		edit func(*SyntheticAttentionRequest)
		want string
	}{
		{
			name: "nan logit",
			edit: func(request *SyntheticAttentionRequest) {
				request.Logits[0] = math.NaN()
			},
			want: "logit 0 must be finite",
		},
		{
			name: "inf logit",
			edit: func(request *SyntheticAttentionRequest) {
				request.Logits[1] = math.Inf(1)
			},
			want: "logit 1 must be finite",
		},
		{
			name: "nan value",
			edit: func(request *SyntheticAttentionRequest) {
				request.Values[0][0] = math.NaN()
			},
			want: "value 0 component 0 must be finite",
		},
		{
			name: "inf value",
			edit: func(request *SyntheticAttentionRequest) {
				request.Values[1][1] = math.Inf(-1)
			},
			want: "value 1 component 1 must be finite",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request := validSyntheticAttentionRequest()
			tc.edit(&request)

			_, err := ComputePartialAttentionSummary(request)
			if !errors.Is(err, ErrInvalidSyntheticAttentionRequest) || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("ComputePartialAttentionSummary() error = %v, want %q", err, tc.want)
			}
		})
	}
}

func TestComputePartialAttentionSummaryRejectsEmptyTokens(t *testing.T) {
	request := validSyntheticAttentionRequest()
	request.Logits = nil
	request.Values = nil

	_, err := ComputePartialAttentionSummary(request)
	if !errors.Is(err, ErrInvalidSyntheticAttentionRequest) || !strings.Contains(err.Error(), "token_count must be greater than zero") {
		t.Fatalf("ComputePartialAttentionSummary() error = %v, want token count rejection", err)
	}
}

func validSyntheticAttentionRequest() SyntheticAttentionRequest {
	return SyntheticAttentionRequest{
		RequestID:       "request-valid",
		JobID:           "job-valid",
		ShardID:         "shard-valid",
		Seed:            42,
		Logits:          []float64{0.5, 1.5},
		Values:          [][]float64{{1, 2}, {3, 4}},
		ValueDim:        2,
		CreatedAtUnixMs: 1_800_000_000_456,
	}
}

func nearlyEqual(got, want float64) bool {
	const epsilon = 1e-12
	diff := got - want
	if diff < 0 {
		diff = -diff
	}
	return diff <= epsilon
}
