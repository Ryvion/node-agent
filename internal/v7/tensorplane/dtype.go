package tensorplane

import (
	"encoding/binary"
	"fmt"
	"math"
	"strings"
)

const (
	TensorDTypeFloat32  TensorDType = "float32"
	TensorDTypeFloat16  TensorDType = "float16"
	TensorDTypeBFloat16 TensorDType = "bfloat16"
)

func NormalizeTensorDType(dtype TensorDType) TensorDType {
	return TensorDType(strings.ToLower(strings.TrimSpace(string(dtype))))
}

func ValidateTensorDType(dtype TensorDType) error {
	switch NormalizeTensorDType(dtype) {
	case TensorDTypeFloat32, TensorDTypeFloat16, TensorDTypeBFloat16:
		return nil
	default:
		return fmt.Errorf("%w: dtype must be float32, float16, or bfloat16", ErrInvalidTensorDType)
	}
}

func tensorDTypeElementBytes(dtype TensorDType) (int, error) {
	switch NormalizeTensorDType(dtype) {
	case TensorDTypeFloat32:
		return 4, nil
	case TensorDTypeFloat16, TensorDTypeBFloat16:
		return 2, nil
	default:
		return 0, fmt.Errorf("%w: unsupported dtype %q", ErrInvalidTensorDType, dtype)
	}
}

func decodeTensorFloat(dtype TensorDType, data []byte, elementIndex int) (float32, error) {
	dtype = NormalizeTensorDType(dtype)
	elementBytes, err := tensorDTypeElementBytes(dtype)
	if err != nil {
		return 0, err
	}
	offset := elementIndex * elementBytes
	if elementIndex < 0 || offset < 0 || offset+elementBytes > len(data) {
		return 0, fmt.Errorf("%w: element index %d out of range", ErrInvalidTensorPage, elementIndex)
	}

	var value float32
	switch dtype {
	case TensorDTypeFloat32:
		value = math.Float32frombits(binary.LittleEndian.Uint32(data[offset : offset+4]))
	case TensorDTypeFloat16:
		value = float16ToFloat32(binary.LittleEndian.Uint16(data[offset : offset+2]))
	case TensorDTypeBFloat16:
		value = bfloat16ToFloat32(binary.LittleEndian.Uint16(data[offset : offset+2]))
	default:
		return 0, fmt.Errorf("%w: unsupported dtype %q", ErrInvalidTensorDType, dtype)
	}
	if !finiteFloat64(float64(value)) {
		return 0, fmt.Errorf("%w: decoded tensor value at element %d must be finite", ErrInvalidTensorPage, elementIndex)
	}
	return value, nil
}

func float16ToFloat32(bits uint16) float32 {
	sign := uint32(bits&0x8000) << 16
	exp := (bits >> 10) & 0x1f
	frac := bits & 0x03ff

	switch exp {
	case 0:
		if frac == 0 {
			return math.Float32frombits(sign)
		}
		exp32 := uint32(127 - 14)
		for frac&0x0400 == 0 {
			frac <<= 1
			exp32--
		}
		frac &= 0x03ff
		return math.Float32frombits(sign | (exp32 << 23) | (uint32(frac) << 13))
	case 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | (uint32(frac) << 13))
	default:
		exp32 := uint32(exp) + (127 - 15)
		return math.Float32frombits(sign | (exp32 << 23) | (uint32(frac) << 13))
	}
}

func bfloat16ToFloat32(bits uint16) float32 {
	return math.Float32frombits(uint32(bits) << 16)
}
