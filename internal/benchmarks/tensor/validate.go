package tensorplane

import (
	"errors"
	"fmt"
	"math"
	"strings"
)

var (
	ErrInvalidTensorDType          = errors.New("tensorplane: invalid dtype")
	ErrInvalidTensorShape          = errors.New("tensorplane: invalid tensor shape")
	ErrInvalidTensorPageID         = errors.New("tensorplane: invalid tensor page id")
	ErrInvalidTensorPage           = errors.New("tensorplane: invalid tensor page")
	ErrInvalidAttentionQuery       = errors.New("tensorplane: invalid attention query")
	ErrInvalidPartialAttentionHash = errors.New("tensorplane: invalid partial attention hash")
)

func ValidateTensorShape(shape TensorShape) error {
	var errs []error
	if shape.Heads <= 0 {
		errs = append(errs, fmt.Errorf("%w: heads must be greater than zero", ErrInvalidTensorShape))
	}
	if shape.Tokens <= 0 {
		errs = append(errs, fmt.Errorf("%w: tokens must be greater than zero", ErrInvalidTensorShape))
	}
	if shape.HeadDim <= 0 {
		errs = append(errs, fmt.Errorf("%w: head_dim must be greater than zero", ErrInvalidTensorShape))
	}
	if shape.ValueDim <= 0 {
		errs = append(errs, fmt.Errorf("%w: value_dim must be greater than zero", ErrInvalidTensorShape))
	}
	if shape.PageSize < 0 {
		errs = append(errs, fmt.Errorf("%w: page_size must be non-negative", ErrInvalidTensorShape))
	} else if shape.PageSize > 0 && shape.Tokens > shape.PageSize {
		errs = append(errs, fmt.Errorf("%w: tokens must be less than or equal to page_size", ErrInvalidTensorShape))
	}
	return errors.Join(errs...)
}

func ValidateTensorPageID(pageID TensorPageID) error {
	pageID = normalizeTensorPageID(pageID)
	var errs []error
	if pageID.ModelID == "" {
		errs = append(errs, fmt.Errorf("%w: model_id required", ErrInvalidTensorPageID))
	}
	if pageID.LayerIndex < 0 {
		errs = append(errs, fmt.Errorf("%w: layer_index must be non-negative", ErrInvalidTensorPageID))
	}
	if pageID.HeadStart < 0 {
		errs = append(errs, fmt.Errorf("%w: head_start must be non-negative", ErrInvalidTensorPageID))
	}
	if pageID.HeadCount <= 0 {
		errs = append(errs, fmt.Errorf("%w: head_count must be greater than zero", ErrInvalidTensorPageID))
	}
	if pageID.TokenStart < 0 {
		errs = append(errs, fmt.Errorf("%w: token_start must be non-negative", ErrInvalidTensorPageID))
	}
	if pageID.TokenCount <= 0 {
		errs = append(errs, fmt.Errorf("%w: token_count must be greater than zero", ErrInvalidTensorPageID))
	}
	if pageID.PageID == "" {
		errs = append(errs, fmt.Errorf("%w: page_id required", ErrInvalidTensorPageID))
	}
	if err := ValidateTensorDType(pageID.DType); err != nil {
		errs = append(errs, fmt.Errorf("dtype: %w", err))
	}
	if pageID.LayoutVersion == "" {
		errs = append(errs, fmt.Errorf("%w: layout_version required", ErrInvalidTensorPageID))
	}
	return errors.Join(errs...)
}

func ValidateTensorPage(page TensorPage) error {
	page = normalizeTensorPage(page)
	var errs []error
	if err := ValidateTensorPageID(page.PageID); err != nil {
		errs = append(errs, err)
	}
	if err := ValidateTensorShape(page.Shape); err != nil {
		errs = append(errs, err)
	}
	if err := ValidateTensorDType(page.DType); err != nil {
		errs = append(errs, err)
	}
	if page.DType != page.PageID.DType {
		errs = append(errs, fmt.Errorf("%w: dtype must match page_id dtype", ErrInvalidTensorPage))
	}
	if page.Shape.Heads > 0 && page.PageID.HeadCount > 0 && page.Shape.Heads != page.PageID.HeadCount {
		errs = append(errs, fmt.Errorf("%w: shape heads must match page_id head_count", ErrInvalidTensorPage))
	}
	if page.Shape.Tokens > 0 && page.PageID.TokenCount > 0 && page.Shape.Tokens != page.PageID.TokenCount {
		errs = append(errs, fmt.Errorf("%w: shape tokens must match page_id token_count", ErrInvalidTensorPage))
	}
	if page.PageID.LayoutVersion != "" && page.PageID.LayoutVersion != TensorLayoutSimpleContiguousV1 {
		errs = append(errs, fmt.Errorf("%w: unsupported layout_version %q", ErrInvalidTensorPage, page.PageID.LayoutVersion))
	}
	if len(page.KeyData) == 0 {
		errs = append(errs, fmt.Errorf("%w: key_data required", ErrInvalidTensorPage))
	}
	if len(page.ValueData) == 0 {
		errs = append(errs, fmt.Errorf("%w: value_data required", ErrInvalidTensorPage))
	}
	if err := validateTensorPageByteLengths(page); err != nil {
		errs = append(errs, err)
	}
	if err := validateTensorPageDataFinite(page); err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

func ValidateAttentionQuery(query AttentionQuery, page TensorPage) error {
	query = normalizeAttentionQuery(query)
	page = normalizeTensorPage(page)
	var errs []error
	if query.RequestID == "" {
		errs = append(errs, fmt.Errorf("%w: request_id required", ErrInvalidAttentionQuery))
	}
	if query.JobID == "" {
		errs = append(errs, fmt.Errorf("%w: job_id required", ErrInvalidAttentionQuery))
	}
	if query.ModelID == "" {
		errs = append(errs, fmt.Errorf("%w: model_id required", ErrInvalidAttentionQuery))
	} else if page.PageID.ModelID != "" && query.ModelID != page.PageID.ModelID {
		errs = append(errs, fmt.Errorf("%w: model_id must match page model_id", ErrInvalidAttentionQuery))
	}
	if query.LayerIndex < 0 {
		errs = append(errs, fmt.Errorf("%w: layer_index must be non-negative", ErrInvalidAttentionQuery))
	} else if query.LayerIndex != page.PageID.LayerIndex {
		errs = append(errs, fmt.Errorf("%w: layer_index must match page layer_index", ErrInvalidAttentionQuery))
	}
	if err := ValidateTensorDType(query.DType); err != nil {
		errs = append(errs, err)
	} else if query.DType != page.DType {
		errs = append(errs, fmt.Errorf("%w: dtype must match page dtype", ErrInvalidAttentionQuery))
	}
	if len(query.QueryVector) != page.Shape.HeadDim {
		errs = append(errs, fmt.Errorf("%w: query_vector length must equal head_dim", ErrInvalidAttentionQuery))
	}
	for i, value := range query.QueryVector {
		if !finiteFloat64(float64(value)) {
			errs = append(errs, fmt.Errorf("%w: query_vector[%d] must be finite", ErrInvalidAttentionQuery, i))
		}
	}
	if !finiteFloat64(query.Scale) {
		errs = append(errs, fmt.Errorf("%w: scale must be finite", ErrInvalidAttentionQuery))
	}
	if query.CreatedAtUnixMs < 0 {
		errs = append(errs, fmt.Errorf("%w: created_at_unix_ms must be non-negative", ErrInvalidAttentionQuery))
	}
	if _, err := resolveQueryLocalHead(query, page); err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

func validateTensorPageByteLengths(page TensorPage) error {
	elementBytes, err := tensorDTypeElementBytes(page.DType)
	if err != nil {
		return err
	}
	keyElements, ok := checkedTensorElementCount(page.Shape.Heads, page.Shape.Tokens, page.Shape.HeadDim)
	if !ok {
		return fmt.Errorf("%w: key tensor element count overflow", ErrInvalidTensorPage)
	}
	valueElements, ok := checkedTensorElementCount(page.Shape.Heads, page.Shape.Tokens, page.Shape.ValueDim)
	if !ok {
		return fmt.Errorf("%w: value tensor element count overflow", ErrInvalidTensorPage)
	}
	keyBytes, ok := checkedMultiply(keyElements, elementBytes)
	if !ok {
		return fmt.Errorf("%w: key_data byte count overflow", ErrInvalidTensorPage)
	}
	valueBytes, ok := checkedMultiply(valueElements, elementBytes)
	if !ok {
		return fmt.Errorf("%w: value_data byte count overflow", ErrInvalidTensorPage)
	}
	var errs []error
	if len(page.KeyData) != keyBytes {
		errs = append(errs, fmt.Errorf("%w: key_data length = %d, want %d", ErrInvalidTensorPage, len(page.KeyData), keyBytes))
	}
	if len(page.ValueData) != valueBytes {
		errs = append(errs, fmt.Errorf("%w: value_data length = %d, want %d", ErrInvalidTensorPage, len(page.ValueData), valueBytes))
	}
	return errors.Join(errs...)
}

func validateTensorPageDataFinite(page TensorPage) error {
	if len(page.KeyData) == 0 || len(page.ValueData) == 0 {
		return nil
	}
	keyElements, ok := checkedTensorElementCount(page.Shape.Heads, page.Shape.Tokens, page.Shape.HeadDim)
	if !ok {
		return nil
	}
	valueElements, ok := checkedTensorElementCount(page.Shape.Heads, page.Shape.Tokens, page.Shape.ValueDim)
	if !ok {
		return nil
	}
	var errs []error
	for i := 0; i < keyElements; i++ {
		if _, err := decodeTensorFloat(page.DType, page.KeyData, i); err != nil {
			errs = append(errs, fmt.Errorf("key_data[%d]: %w", i, err))
			break
		}
	}
	for i := 0; i < valueElements; i++ {
		if _, err := decodeTensorFloat(page.DType, page.ValueData, i); err != nil {
			errs = append(errs, fmt.Errorf("value_data[%d]: %w", i, err))
			break
		}
	}
	return errors.Join(errs...)
}

func resolveQueryLocalHead(query AttentionQuery, page TensorPage) (int, error) {
	query = normalizeAttentionQuery(query)
	page = normalizeTensorPage(page)
	headIndex := query.HeadIndex
	if query.HeadCount > 0 {
		if query.HeadStart < 0 {
			return 0, fmt.Errorf("%w: head_start must be non-negative", ErrInvalidAttentionQuery)
		}
		if query.HeadCount != 1 {
			return 0, fmt.Errorf("%w: only single-head attention queries are supported", ErrInvalidAttentionQuery)
		}
		headIndex = query.HeadStart
	} else if query.HeadCount < 0 {
		return 0, fmt.Errorf("%w: head_count must be non-negative", ErrInvalidAttentionQuery)
	}
	if headIndex < 0 {
		return 0, fmt.Errorf("%w: head_index must be non-negative", ErrInvalidAttentionQuery)
	}
	localHead := headIndex - page.PageID.HeadStart
	if localHead < 0 || localHead >= page.PageID.HeadCount {
		return 0, fmt.Errorf("%w: requested head %d outside page head range [%d,%d)", ErrInvalidAttentionQuery, headIndex, page.PageID.HeadStart, page.PageID.HeadStart+page.PageID.HeadCount)
	}
	if localHead >= page.Shape.Heads {
		return 0, fmt.Errorf("%w: requested head %d outside shape heads", ErrInvalidAttentionQuery, headIndex)
	}
	return localHead, nil
}

func checkedTensorElementCount(a int, b int, c int) (int, bool) {
	ab, ok := checkedMultiply(a, b)
	if !ok {
		return 0, false
	}
	return checkedMultiply(ab, c)
}

func checkedMultiply(a int, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	if a == 0 || b == 0 {
		return 0, true
	}
	maxInt := int(^uint(0) >> 1)
	if a > maxInt/b {
		return 0, false
	}
	return a * b, true
}

func normalizeTensorPage(page TensorPage) TensorPage {
	page.PageID = normalizeTensorPageID(page.PageID)
	page.DType = NormalizeTensorDType(page.DType)
	page.Hash = strings.TrimSpace(page.Hash)
	return page
}

func normalizeTensorPageID(pageID TensorPageID) TensorPageID {
	pageID.ModelID = strings.TrimSpace(pageID.ModelID)
	pageID.PageID = strings.TrimSpace(pageID.PageID)
	pageID.DType = NormalizeTensorDType(pageID.DType)
	pageID.LayoutVersion = strings.TrimSpace(pageID.LayoutVersion)
	return pageID
}

func normalizeAttentionQuery(query AttentionQuery) AttentionQuery {
	query.RequestID = strings.TrimSpace(query.RequestID)
	query.JobID = strings.TrimSpace(query.JobID)
	query.ModelID = strings.TrimSpace(query.ModelID)
	query.DType = NormalizeTensorDType(query.DType)
	return query
}

func finiteFloat64(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
