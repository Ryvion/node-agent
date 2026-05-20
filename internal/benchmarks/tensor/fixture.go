package tensorplane

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"strings"
)

const (
	TensorPlaneFixtureVersion = "tensorplane/local-fixture/v1"

	defaultTensorPlaneProbeTokens   = 128
	defaultTensorPlaneProbeHeadDim  = 64
	defaultTensorPlaneProbeValueDim = 64
	defaultTensorPlaneProbeSeed     = int64(42)
)

var ErrInvalidTensorPlaneFixture = errors.New("tensorplane: invalid fixture")

type TensorPlaneFixtureConfig struct {
	Tokens   int
	HeadDim  int
	ValueDim int
	DType    TensorDType
	Seed     int64
}

type TensorPlaneFixture struct {
	Version      string         `json:"version"`
	PageID       string         `json:"page_id"`
	ModelID      string         `json:"model_id"`
	LayerIndex   int            `json:"layer_index"`
	DType        TensorDType    `json:"dtype"`
	Shape        TensorShape    `json:"shape"`
	TensorPageID TensorPageID   `json:"tensor_page_id"`
	Query        AttentionQuery `json:"query"`
	KeyData      []byte         `json:"key_data"`
	ValueData    []byte         `json:"value_data"`
	KeyHash      string         `json:"key_hash"`
	ValueHash    string         `json:"value_hash"`
	PageHash     string         `json:"page_hash,omitempty"`
}

func DefaultTensorPlaneFixtureConfig() TensorPlaneFixtureConfig {
	return TensorPlaneFixtureConfig{
		Tokens:   defaultTensorPlaneProbeTokens,
		HeadDim:  defaultTensorPlaneProbeHeadDim,
		ValueDim: defaultTensorPlaneProbeValueDim,
		DType:    TensorDTypeFloat32,
		Seed:     defaultTensorPlaneProbeSeed,
	}
}

func BuildTensorPlaneFixture(config TensorPlaneFixtureConfig) (TensorPlaneFixture, error) {
	config.DType = NormalizeTensorDType(config.DType)
	if err := ValidateTensorPlaneFixtureConfig(config); err != nil {
		return TensorPlaneFixture{}, err
	}

	keyValues := make([]float32, config.Tokens*config.HeadDim)
	valueValues := make([]float32, config.Tokens*config.ValueDim)
	queryVector := make([]float32, config.HeadDim)
	for i := range keyValues {
		keyValues[i] = deterministicTensorPlaneFloat(config.Seed, 1, i, 0.75)
	}
	for i := range valueValues {
		valueValues[i] = deterministicTensorPlaneFloat(config.Seed, 2, i, 1.25)
	}
	for i := range queryVector {
		queryVector[i] = deterministicTensorPlaneFloat(config.Seed, 3, i, 0.5)
	}

	keyData, err := encodeTensorPlaneFloats(config.DType, keyValues)
	if err != nil {
		return TensorPlaneFixture{}, err
	}
	valueData, err := encodeTensorPlaneFloats(config.DType, valueValues)
	if err != nil {
		return TensorPlaneFixture{}, err
	}

	pageID := TensorPageID{
		ModelID:       "tensorplane-local-fixture",
		LayerIndex:    0,
		HeadStart:     0,
		HeadCount:     1,
		TokenStart:    0,
		TokenCount:    config.Tokens,
		PageID:        fmt.Sprintf("tensorplane-page-%d-%d-%d-%d-%s", config.Seed, config.Tokens, config.HeadDim, config.ValueDim, config.DType),
		DType:         config.DType,
		LayoutVersion: TensorLayoutSimpleContiguousV1,
	}
	shape := TensorShape{
		Heads:    1,
		Tokens:   config.Tokens,
		HeadDim:  config.HeadDim,
		ValueDim: config.ValueDim,
		PageSize: config.Tokens,
	}
	query := AttentionQuery{
		RequestID:       fmt.Sprintf("tensorplane-selftest-%d-%d-%d-%d-%s", config.Seed, config.Tokens, config.HeadDim, config.ValueDim, config.DType),
		JobID:           fmt.Sprintf("tensorplane-selftest-job-%d", config.Seed),
		ModelID:         pageID.ModelID,
		LayerIndex:      pageID.LayerIndex,
		HeadIndex:       pageID.HeadStart,
		QueryVector:     queryVector,
		Scale:           1 / math.Sqrt(float64(config.HeadDim)),
		DType:           config.DType,
		CreatedAtUnixMs: deterministicTensorPlaneCreatedAtUnixMs(config.Seed),
	}
	fixture := TensorPlaneFixture{
		Version:      TensorPlaneFixtureVersion,
		PageID:       pageID.PageID,
		ModelID:      pageID.ModelID,
		LayerIndex:   pageID.LayerIndex,
		DType:        config.DType,
		Shape:        shape,
		TensorPageID: pageID,
		Query:        query,
		KeyData:      keyData,
		ValueData:    valueData,
		KeyHash:      sha256HexBytes(keyData),
		ValueHash:    sha256HexBytes(valueData),
	}
	page, _, err := TensorPlaneFixturePageAndQuery(fixture)
	if err != nil {
		return TensorPlaneFixture{}, err
	}
	pageHash, err := HashTensorPage(page)
	if err != nil {
		return TensorPlaneFixture{}, err
	}
	fixture.PageHash = pageHash
	if err := ValidateTensorPlaneFixture(fixture); err != nil {
		return TensorPlaneFixture{}, err
	}
	return fixture, nil
}

func ValidateTensorPlaneFixtureConfig(config TensorPlaneFixtureConfig) error {
	var errs []error
	if config.Tokens <= 0 {
		errs = append(errs, fmt.Errorf("%w: tokens must be greater than zero", ErrInvalidTensorShape))
	}
	if config.HeadDim <= 0 {
		errs = append(errs, fmt.Errorf("%w: head_dim must be greater than zero", ErrInvalidTensorShape))
	}
	if config.ValueDim <= 0 {
		errs = append(errs, fmt.Errorf("%w: value_dim must be greater than zero", ErrInvalidTensorShape))
	}
	if config.Tokens > 0 && config.HeadDim > 0 {
		if elements, ok := checkedMultiply(config.Tokens, config.HeadDim); !ok {
			errs = append(errs, fmt.Errorf("%w: key tensor element count overflow", ErrInvalidTensorShape))
		} else if _, ok := checkedMultiply(elements, 4); !ok {
			errs = append(errs, fmt.Errorf("%w: key tensor byte count overflow", ErrInvalidTensorShape))
		}
	}
	if config.Tokens > 0 && config.ValueDim > 0 {
		if elements, ok := checkedMultiply(config.Tokens, config.ValueDim); !ok {
			errs = append(errs, fmt.Errorf("%w: value tensor element count overflow", ErrInvalidTensorShape))
		} else if _, ok := checkedMultiply(elements, 4); !ok {
			errs = append(errs, fmt.Errorf("%w: value tensor byte count overflow", ErrInvalidTensorShape))
		}
	}
	switch NormalizeTensorDType(config.DType) {
	case TensorDTypeFloat32, TensorDTypeFloat16:
	default:
		errs = append(errs, fmt.Errorf("%w: dtype must be float32 or float16", ErrInvalidTensorDType))
	}
	return errors.Join(errs...)
}

func deterministicTensorPlaneCreatedAtUnixMs(seed int64) int64 {
	return 1_800_000_000_000 + int64(splitmix64(uint64(seed))%1_000_000)
}

func ValidateTensorPlaneFixture(fixture TensorPlaneFixture) error {
	page, query, err := TensorPlaneFixturePageAndQuery(fixture)
	if err != nil {
		return err
	}
	var errs []error
	if fixture.Version != TensorPlaneFixtureVersion {
		errs = append(errs, fmt.Errorf("%w: unsupported version %q", ErrInvalidTensorPlaneFixture, fixture.Version))
	}
	if strings.TrimSpace(fixture.PageID) != page.PageID.PageID {
		errs = append(errs, fmt.Errorf("%w: page_id must match tensor_page_id.page_id", ErrInvalidTensorPlaneFixture))
	}
	if strings.TrimSpace(fixture.ModelID) != page.PageID.ModelID {
		errs = append(errs, fmt.Errorf("%w: model_id must match tensor_page_id.model_id", ErrInvalidTensorPlaneFixture))
	}
	if fixture.LayerIndex != page.PageID.LayerIndex {
		errs = append(errs, fmt.Errorf("%w: layer_index must match tensor_page_id.layer_index", ErrInvalidTensorPlaneFixture))
	}
	if NormalizeTensorDType(fixture.DType) != page.DType {
		errs = append(errs, fmt.Errorf("%w: dtype must match tensor_page_id dtype", ErrInvalidTensorPlaneFixture))
	}
	if fixture.KeyHash == "" || fixture.KeyHash != sha256HexBytes(fixture.KeyData) {
		errs = append(errs, fmt.Errorf("%w: key_hash mismatch", ErrInvalidTensorPlaneFixture))
	}
	if fixture.ValueHash == "" || fixture.ValueHash != sha256HexBytes(fixture.ValueData) {
		errs = append(errs, fmt.Errorf("%w: value_hash mismatch", ErrInvalidTensorPlaneFixture))
	}
	if err := ValidateTensorPage(page); err != nil {
		errs = append(errs, err)
	}
	if err := ValidateAttentionQuery(query, page); err != nil {
		errs = append(errs, err)
	}
	if strings.TrimSpace(fixture.PageHash) != "" {
		pageHash, err := HashTensorPage(page)
		if err != nil {
			errs = append(errs, err)
		} else if fixture.PageHash != pageHash {
			errs = append(errs, fmt.Errorf("%w: page_hash mismatch", ErrInvalidTensorPlaneFixture))
		}
	}
	return errors.Join(errs...)
}

func TensorPlaneFixturePageAndQuery(fixture TensorPlaneFixture) (TensorPage, AttentionQuery, error) {
	pageID := normalizeTensorPageID(fixture.TensorPageID)
	dtype := NormalizeTensorDType(fixture.DType)
	if dtype == "" {
		dtype = pageID.DType
	}
	page := TensorPage{
		PageID:    pageID,
		DType:     dtype,
		Shape:     fixture.Shape,
		KeyData:   append([]byte(nil), fixture.KeyData...),
		ValueData: append([]byte(nil), fixture.ValueData...),
	}
	query := normalizeAttentionQuery(fixture.Query)
	query.QueryVector = append([]float32(nil), fixture.Query.QueryVector...)
	return page, query, nil
}

func encodeTensorPlaneFloats(dtype TensorDType, values []float32) ([]byte, error) {
	dtype = NormalizeTensorDType(dtype)
	elementBytes, err := tensorDTypeElementBytes(dtype)
	if err != nil {
		return nil, err
	}
	if dtype != TensorDTypeFloat32 && dtype != TensorDTypeFloat16 {
		return nil, fmt.Errorf("%w: dtype must be float32 or float16", ErrInvalidTensorDType)
	}
	out := make([]byte, len(values)*elementBytes)
	for i, value := range values {
		if !finiteFloat64(float64(value)) {
			return nil, fmt.Errorf("%w: tensor value %d must be finite", ErrInvalidTensorPage, i)
		}
		switch dtype {
		case TensorDTypeFloat32:
			binary.LittleEndian.PutUint32(out[i*4:i*4+4], math.Float32bits(value))
		case TensorDTypeFloat16:
			binary.LittleEndian.PutUint16(out[i*2:i*2+2], float32ToTensorPlaneFloat16Bits(value))
		}
	}
	return out, nil
}

func deterministicTensorPlaneFloat(seed int64, stream uint64, index int, amplitude float64) float32 {
	mixed := splitmix64(uint64(seed) + stream*0x9e3779b97f4a7c15 + uint64(index)*0xbf58476d1ce4e5b9)
	unit := float64(mixed>>11) * (1.0 / (1 << 53))
	return float32((unit*2 - 1) * amplitude)
}

func splitmix64(value uint64) uint64 {
	value += 0x9e3779b97f4a7c15
	value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9
	value = (value ^ (value >> 27)) * 0x94d049bb133111eb
	return value ^ (value >> 31)
}

func float32ToTensorPlaneFloat16Bits(value float32) uint16 {
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

func sha256HexBytes(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
