package tensorplane

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestValidateTensorPageAcceptsFloat32Fixture(t *testing.T) {
	page := testFloat32Page(t, []float32{
		1, 0,
		0, 1,
	}, []float32{
		2, 4,
		10, 20,
	})
	if err := ValidateTensorPage(page); err != nil {
		t.Fatalf("ValidateTensorPage() error = %v", err)
	}
}

func TestValidateTensorPageRejectsInvalidByteLengths(t *testing.T) {
	page := testFloat32Page(t, []float32{
		1, 0,
		0, 1,
	}, []float32{
		2, 4,
		10, 20,
	})
	page.KeyData = page.KeyData[:len(page.KeyData)-1]

	if err := ValidateTensorPage(page); err == nil {
		t.Fatal("ValidateTensorPage() error = nil, want invalid key_data length")
	}
}

func TestDecodeTensorFloatRejectsNaNAndInf(t *testing.T) {
	nanBytes := make([]byte, 4)
	binary.LittleEndian.PutUint32(nanBytes, math.Float32bits(float32(math.NaN())))
	if _, err := decodeTensorFloat(TensorDTypeFloat32, nanBytes, 0); err == nil {
		t.Fatal("decodeTensorFloat(NaN) error = nil")
	}

	infBytes := make([]byte, 2)
	binary.LittleEndian.PutUint16(infBytes, 0x7c00)
	if _, err := decodeTensorFloat(TensorDTypeFloat16, infBytes, 0); err == nil {
		t.Fatal("decodeTensorFloat(+Inf float16) error = nil")
	}
}

func TestValidateTensorPageRejectsNaN(t *testing.T) {
	page := testFloat32Page(t, []float32{
		1, 0,
		0, 1,
	}, []float32{
		2, 4,
		10, 20,
	})
	binary.LittleEndian.PutUint32(page.ValueData[0:4], math.Float32bits(float32(math.NaN())))

	if err := ValidateTensorPage(page); err == nil {
		t.Fatal("ValidateTensorPage() error = nil, want NaN rejection")
	}
}

func TestValidateAttentionQueryRejectsNaN(t *testing.T) {
	page := testFloat32Page(t, []float32{
		1, 0,
		0, 1,
	}, []float32{
		2, 4,
		10, 20,
	})
	query := testAttentionQuery()
	query.QueryVector[0] = float32(math.NaN())

	if err := ValidateAttentionQuery(query, page); err == nil {
		t.Fatal("ValidateAttentionQuery() error = nil, want NaN rejection")
	}
}

func TestOnlineSoftmaxSummaryClonesWeightedValue(t *testing.T) {
	summary := TensorPartialAttentionSummary{
		LocalMax:      2,
		ExpSum:        3,
		WeightedValue: []float64{4, 5},
		TokenCount:    6,
		ValueDim:      2,
		DType:         TensorDTypeFloat32,
	}

	compat := summary.OnlineSoftmaxSummary()
	compat.WeightedValue[0] = 99
	if summary.WeightedValue[0] == 99 {
		t.Fatal("OnlineSoftmaxSummary() returned aliased weighted_value")
	}
}

func testAttentionQuery() AttentionQuery {
	return AttentionQuery{
		RequestID:       "req-1",
		JobID:           "job-1",
		ModelID:         "model-a",
		LayerIndex:      2,
		HeadIndex:       0,
		QueryVector:     []float32{1, 0},
		Scale:           1,
		DType:           TensorDTypeFloat32,
		CreatedAtUnixMs: 1,
	}
}

func testFloat32Page(t *testing.T, keys []float32, values []float32) TensorPage {
	t.Helper()
	page := TensorPage{
		PageID: TensorPageID{
			ModelID:       "model-a",
			LayerIndex:    2,
			HeadStart:     0,
			HeadCount:     1,
			TokenStart:    0,
			TokenCount:    2,
			PageID:        "page-1",
			DType:         TensorDTypeFloat32,
			LayoutVersion: TensorLayoutSimpleContiguousV1,
		},
		DType: TensorDTypeFloat32,
		Shape: TensorShape{
			Heads:    1,
			Tokens:   2,
			HeadDim:  2,
			ValueDim: 2,
			PageSize: 2,
		},
		KeyData:   encodeFloat32Tensor(keys),
		ValueData: encodeFloat32Tensor(values),
	}
	if err := ValidateTensorPage(page); err != nil {
		t.Fatalf("fixture page invalid: %v", err)
	}
	return page
}

func encodeFloat32Tensor(values []float32) []byte {
	out := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(out[i*4:i*4+4], math.Float32bits(value))
	}
	return out
}
