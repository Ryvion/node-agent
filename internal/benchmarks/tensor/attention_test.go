package tensorplane

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestComputeTensorPartialAttentionSummaryFloat32(t *testing.T) {
	page := testFloat32Page(t, []float32{
		1, 0,
		0, 1,
	}, []float32{
		2, 4,
		10, 20,
	})
	query := testAttentionQuery()

	summary, err := ComputeTensorPartialAttentionSummary(query, page)
	if err != nil {
		t.Fatalf("ComputeTensorPartialAttentionSummary() error = %v", err)
	}

	wantExpSum := 1 + math.Exp(-1)
	wantWeighted := []float64{
		2 + math.Exp(-1)*10,
		4 + math.Exp(-1)*20,
	}
	assertClose(t, "local_max", summary.LocalMax, 1, 1e-12)
	assertClose(t, "exp_sum", summary.ExpSum, wantExpSum, 1e-12)
	assertClose(t, "weighted_value[0]", summary.WeightedValue[0], wantWeighted[0], 1e-12)
	assertClose(t, "weighted_value[1]", summary.WeightedValue[1], wantWeighted[1], 1e-12)
	if summary.TokenCount != 2 || summary.ValueDim != 2 {
		t.Fatalf("summary dimensions = token_count %d value_dim %d, want 2/2", summary.TokenCount, summary.ValueDim)
	}
	if summary.PageHash == "" || summary.SummaryHash == "" {
		t.Fatalf("summary hashes missing: page=%q summary=%q", summary.PageHash, summary.SummaryHash)
	}
}

func TestComputeTensorPartialAttentionSummaryFloat16(t *testing.T) {
	page := testFloat16Page(t, []float32{
		0.5, 1.0,
		1.5, -0.5,
	}, []float32{
		0.25, 2,
		4, -1,
	})
	query := AttentionQuery{
		RequestID:       "req-f16",
		JobID:           "job-f16",
		ModelID:         "model-a",
		LayerIndex:      2,
		HeadIndex:       0,
		QueryVector:     []float32{2, -1},
		Scale:           0.5,
		DType:           TensorDTypeFloat16,
		CreatedAtUnixMs: 1,
	}

	summary, err := ComputeTensorPartialAttentionSummary(query, page)
	if err != nil {
		t.Fatalf("ComputeTensorPartialAttentionSummary() error = %v", err)
	}

	logit0 := ((2 * 0.5) + (-1 * 1.0)) * 0.5
	logit1 := ((2 * 1.5) + (-1 * -0.5)) * 0.5
	localMax := math.Max(logit0, logit1)
	weight0 := math.Exp(logit0 - localMax)
	weight1 := math.Exp(logit1 - localMax)
	assertClose(t, "local_max", summary.LocalMax, localMax, 1e-6)
	assertClose(t, "exp_sum", summary.ExpSum, weight0+weight1, 1e-6)
	assertClose(t, "weighted_value[0]", summary.WeightedValue[0], weight0*0.25+weight1*4, 1e-6)
	assertClose(t, "weighted_value[1]", summary.WeightedValue[1], weight0*2+weight1*-1, 1e-6)
}

func TestComputeTensorPartialAttentionSummaryRejectsNaNInTensor(t *testing.T) {
	page := testFloat32Page(t, []float32{
		1, 0,
		0, 1,
	}, []float32{
		2, 4,
		10, 20,
	})
	binary.LittleEndian.PutUint32(page.KeyData[0:4], math.Float32bits(float32(math.NaN())))

	if _, err := ComputeTensorPartialAttentionSummary(testAttentionQuery(), page); err == nil {
		t.Fatal("ComputeTensorPartialAttentionSummary() error = nil, want NaN rejection")
	}
}

func TestTensorPartialAttentionSummaryHashDeterministic(t *testing.T) {
	page := testFloat32Page(t, []float32{
		1, 0,
		0, 1,
	}, []float32{
		2, 4,
		10, 20,
	})
	query := testAttentionQuery()

	first, err := ComputeTensorPartialAttentionSummary(query, page)
	if err != nil {
		t.Fatalf("first ComputeTensorPartialAttentionSummary() error = %v", err)
	}
	second, err := ComputeTensorPartialAttentionSummary(query, page)
	if err != nil {
		t.Fatalf("second ComputeTensorPartialAttentionSummary() error = %v", err)
	}
	if first.SummaryHash != second.SummaryHash {
		t.Fatalf("summary_hash differs: %s vs %s", first.SummaryHash, second.SummaryHash)
	}
	if first.PageHash != second.PageHash {
		t.Fatalf("page_hash differs: %s vs %s", first.PageHash, second.PageHash)
	}
}

func TestChangingTensorValueChangesSummaryHash(t *testing.T) {
	page := testFloat32Page(t, []float32{
		1, 0,
		0, 1,
	}, []float32{
		2, 4,
		10, 20,
	})
	query := testAttentionQuery()

	first, err := ComputeTensorPartialAttentionSummary(query, page)
	if err != nil {
		t.Fatalf("first ComputeTensorPartialAttentionSummary() error = %v", err)
	}
	binary.LittleEndian.PutUint32(page.ValueData[0:4], math.Float32bits(3))
	second, err := ComputeTensorPartialAttentionSummary(query, page)
	if err != nil {
		t.Fatalf("second ComputeTensorPartialAttentionSummary() error = %v", err)
	}
	if first.SummaryHash == second.SummaryHash {
		t.Fatalf("summary_hash unchanged after tensor mutation: %s", first.SummaryHash)
	}
	if first.PageHash == second.PageHash {
		t.Fatalf("page_hash unchanged after tensor mutation: %s", first.PageHash)
	}
}

func TestTensorPayloadEstimateScalesWithValueDimNotTokenCount(t *testing.T) {
	twoDimTwoTokens := EstimateTensorPartialAttentionPayloadBytes(2)
	twoDimFourTokens := tensorPayloadEstimateForFixture(t, 4, 2)
	threeDimTwoTokens := EstimateTensorPartialAttentionPayloadBytes(3)

	if twoDimTwoTokens != twoDimFourTokens {
		t.Fatalf("same value_dim estimate differs by token_count: %d vs %d", twoDimTwoTokens, twoDimFourTokens)
	}
	if threeDimTwoTokens <= twoDimTwoTokens {
		t.Fatalf("larger value_dim estimate = %d, want > %d", threeDimTwoTokens, twoDimTwoTokens)
	}
}

func TestTensorPlaneCoreDoesNotCallSyntheticMemoryBenchGenerator(t *testing.T) {
	needle := "GenerateSynthetic" + "AttentionRequest"
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	dir := filepath.Dir(file)
	paths, err := filepath.Glob(filepath.Join(dir, "*.go"))
	if err != nil {
		t.Fatalf("filepath.Glob() error = %v", err)
	}
	for _, path := range paths {
		if strings.HasSuffix(path, "_test.go") {
			continue
		}
		contents, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("ReadFile(%s) error = %v", path, err)
		}
		if strings.Contains(string(contents), needle) {
			t.Fatalf("%s references synthetic MemoryBench generator", filepath.Base(path))
		}
	}
}

func testFloat16Page(t *testing.T, keys []float32, values []float32) TensorPage {
	t.Helper()
	page := testFloat32Page(t, []float32{
		1, 0,
		0, 1,
	}, []float32{
		2, 4,
		10, 20,
	})
	page.PageID.PageID = "page-f16"
	page.PageID.DType = TensorDTypeFloat16
	page.DType = TensorDTypeFloat16
	page.KeyData = encodeFloat16Tensor(keys)
	page.ValueData = encodeFloat16Tensor(values)
	if err := ValidateTensorPage(page); err != nil {
		t.Fatalf("fixture float16 page invalid: %v", err)
	}
	return page
}

func tensorPayloadEstimateForFixture(t *testing.T, tokens int, valueDim int) int64 {
	t.Helper()
	keyValues := make([]float32, tokens*2)
	valueValues := make([]float32, tokens*valueDim)
	page := TensorPage{
		PageID: TensorPageID{
			ModelID:       "model-a",
			LayerIndex:    2,
			HeadStart:     0,
			HeadCount:     1,
			TokenStart:    0,
			TokenCount:    tokens,
			PageID:        "page-estimate",
			DType:         TensorDTypeFloat32,
			LayoutVersion: TensorLayoutSimpleContiguousV1,
		},
		DType: TensorDTypeFloat32,
		Shape: TensorShape{
			Heads:    1,
			Tokens:   tokens,
			HeadDim:  2,
			ValueDim: valueDim,
			PageSize: tokens,
		},
		KeyData:   encodeFloat32Tensor(keyValues),
		ValueData: encodeFloat32Tensor(valueValues),
	}
	query := testAttentionQuery()
	query.QueryVector = []float32{0, 0}
	summary, err := ComputeTensorPartialAttentionSummary(query, page)
	if err != nil {
		t.Fatalf("ComputeTensorPartialAttentionSummary() error = %v", err)
	}
	return summary.PayloadBytesEstimate
}

func encodeFloat16Tensor(values []float32) []byte {
	out := make([]byte, len(values)*2)
	for i, value := range values {
		binary.LittleEndian.PutUint16(out[i*2:i*2+2], float32ToFloat16Bits(value))
	}
	return out
}

func float32ToFloat16Bits(value float32) uint16 {
	bits := math.Float32bits(value)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits >> 23) & 0xff)
	frac := bits & 0x7fffff
	if exp == 0xff {
		if frac == 0 {
			return sign | 0x7c00
		}
		return sign | 0x7e00
	}
	exp16 := exp - 127 + 15
	if exp16 >= 0x1f {
		return sign | 0x7c00
	}
	if exp16 <= 0 {
		if exp16 < -10 {
			return sign
		}
		frac |= 0x800000
		shift := uint(14 - exp16)
		rounded := uint16(frac >> shift)
		if (frac>>(shift-1))&1 == 1 {
			rounded++
		}
		return sign | rounded
	}
	roundedFrac := uint16(frac >> 13)
	if frac&0x00001000 != 0 {
		roundedFrac++
		if roundedFrac == 0x0400 {
			roundedFrac = 0
			exp16++
			if exp16 >= 0x1f {
				return sign | 0x7c00
			}
		}
	}
	return sign | uint16(exp16<<10) | roundedFrac
}

func assertClose(t *testing.T, name string, got float64, want float64, tolerance float64) {
	t.Helper()
	if math.Abs(got-want) > tolerance {
		t.Fatalf("%s = %.17g, want %.17g", name, got, want)
	}
}
