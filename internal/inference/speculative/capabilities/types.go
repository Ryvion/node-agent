package speculative

const (
	EnvDisableSpeculativeDecoding = "RYV_NODE_DISABLE_SPECULATIVE_DECODING"
	EnvDisableDraftModels         = "RYV_NODE_DISABLE_DRAFT_MODELS"
	EnvDisableNgramSpeculation    = "RYV_NODE_DISABLE_NGRAM_SPECULATION"

	MethodNgram      = "ngram"
	MethodDraftModel = "draft_model"
	MethodNativeMTP  = "native_mtp"
	MethodEAGLE      = "eagle"
	MethodEAGLE3     = "eagle3"
	MethodMedusa     = "medusa"
	MethodReDrafter  = "redrafter"

	BenchmarkStatusNotRun = "not_run"
)

type DecodingCapability struct {
	Supported                    bool     `json:"supported"`
	Enabled                      bool     `json:"enabled"`
	Methods                      []string `json:"methods"`
	DefaultMethod                string   `json:"default_method"`
	SupportsStreaming            bool     `json:"supports_streaming"`
	SupportsLosslessVerification bool     `json:"supports_lossless_verification"`
	MaxSpeculativeTokens         int      `json:"max_speculative_tokens"`
}

type Profile struct {
	TargetModelID       string           `json:"target_model_id"`
	DraftModelID        string           `json:"draft_model_id,omitempty"`
	Method              string           `json:"method"`
	Backend             string           `json:"backend"`
	Acceleration        []string         `json:"acceleration"`
	Runnable            bool             `json:"runnable"`
	TargetResident      bool             `json:"target_resident"`
	DraftResident       bool             `json:"draft_resident"`
	WarmPair            bool             `json:"warm_pair"`
	TokenizerCompatible bool             `json:"tokenizer_compatible"`
	MemoryEstimateBytes uint64           `json:"memory_estimate_bytes"`
	BlockedReasons      []string         `json:"blocked_reasons"`
	LastBenchmark       BenchmarkSummary `json:"last_benchmark"`
}

type BenchmarkSummary struct {
	Status                  string  `json:"status"`
	DecodeTokensPerSecond   float64 `json:"decode_tokens_per_second,omitempty"`
	EndToEndTokensPerSecond float64 `json:"end_to_end_tokens_per_second,omitempty"`
	ImprovementRatio        float64 `json:"improvement_ratio,omitempty"`
	UpdatedAtUnixMs         int64   `json:"updated_at_unix_ms,omitempty"`
}

type Report struct {
	SpeculativeDecoding DecodingCapability `json:"speculative_decoding"`
	SpeculativeProfiles []Profile          `json:"speculative_profiles"`
}
