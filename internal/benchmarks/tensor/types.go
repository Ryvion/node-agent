package tensorplane

const (
	TensorLayoutSimpleContiguousV1 = "tensorplane/simple-contiguous/v1"
)

type TensorDType string

type TensorShape struct {
	Heads    int `json:"heads"`
	Tokens   int `json:"tokens"`
	HeadDim  int `json:"head_dim"`
	ValueDim int `json:"value_dim"`
	PageSize int `json:"page_size,omitempty"`
}

type TensorPageID struct {
	ModelID       string      `json:"model_id"`
	LayerIndex    int         `json:"layer_index"`
	HeadStart     int         `json:"head_start"`
	HeadCount     int         `json:"head_count"`
	TokenStart    int         `json:"token_start"`
	TokenCount    int         `json:"token_count"`
	PageID        string      `json:"page_id"`
	DType         TensorDType `json:"dtype"`
	LayoutVersion string      `json:"layout_version"`
}

type TensorPage struct {
	PageID    TensorPageID `json:"page_id"`
	DType     TensorDType  `json:"dtype"`
	Shape     TensorShape  `json:"shape"`
	KeyData   []byte       `json:"key_data"`
	ValueData []byte       `json:"value_data"`
	Hash      string       `json:"hash,omitempty"`
}

type AttentionQuery struct {
	RequestID       string      `json:"request_id"`
	JobID           string      `json:"job_id"`
	ModelID         string      `json:"model_id"`
	LayerIndex      int         `json:"layer_index"`
	HeadIndex       int         `json:"head_index,omitempty"`
	HeadStart       int         `json:"head_start,omitempty"`
	HeadCount       int         `json:"head_count,omitempty"`
	QueryVector     []float32   `json:"query_vector"`
	Scale           float64     `json:"scale"`
	DType           TensorDType `json:"dtype"`
	CreatedAtUnixMs int64       `json:"created_at_unix_ms"`
}

type TensorPartialAttentionSummary struct {
	RequestID            string       `json:"request_id"`
	JobID                string       `json:"job_id"`
	PageID               TensorPageID `json:"page_id"`
	LocalMax             float64      `json:"local_max"`
	ExpSum               float64      `json:"exp_sum"`
	WeightedValue        []float64    `json:"weighted_value"`
	TokenCount           int          `json:"token_count"`
	ValueDim             int          `json:"value_dim"`
	DType                TensorDType  `json:"dtype"`
	PageHash             string       `json:"page_hash"`
	SummaryHash          string       `json:"summary_hash"`
	ComputeTimeUs        int64        `json:"compute_time_us"`
	PayloadBytesEstimate int64        `json:"payload_bytes_estimate"`
}

type TensorOnlineSoftmaxSummary struct {
	LocalMax      float64     `json:"local_max"`
	ExpSum        float64     `json:"exp_sum"`
	WeightedValue []float64   `json:"weighted_value"`
	TokenCount    int         `json:"token_count"`
	ValueDim      int         `json:"value_dim"`
	DType         TensorDType `json:"dtype"`
}

func (s TensorPartialAttentionSummary) OnlineSoftmaxSummary() TensorOnlineSoftmaxSummary {
	return TensorOnlineSoftmaxSummary{
		LocalMax:      s.LocalMax,
		ExpSum:        s.ExpSum,
		WeightedValue: append([]float64(nil), s.WeightedValue...),
		TokenCount:    s.TokenCount,
		ValueDim:      s.ValueDim,
		DType:         s.DType,
	}
}
