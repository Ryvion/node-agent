package readiness

type ReadinessDecision string

const (
	ReadinessDecisionReady    ReadinessDecision = "ready"
	ReadinessDecisionNotReady ReadinessDecision = "not_ready"
	ReadinessDecisionExpired  ReadinessDecision = "expired"
	ReadinessDecisionRejected ReadinessDecision = "rejected"

	ReadinessReady    = ReadinessDecisionReady
	ReadinessNotReady = ReadinessDecisionNotReady
	ReadinessExpired  = ReadinessDecisionExpired
	ReadinessRejected = ReadinessDecisionRejected

	ReadinessReasonReady               = "ready"
	ReadinessReasonExpired             = "expired"
	ReadinessReasonRejected            = "rejected"
	ReadinessReasonMismatch            = "mismatch"
	ReadinessReasonNotResident         = "not_resident"
	ReadinessReasonLatencyTooHigh      = "latency_too_high"
	ReadinessReasonSparseLogitMismatch = "sparse_logit_mismatch"
	ReadinessReasonInvalidLocalState   = "invalid_local_state"

	SignaturePlaceholderUnsignedV1 = "unsigned-readiness-response-v1"
)

type ReadinessResponse struct {
	ChallengeID          string            `json:"challenge_id"`
	NodeID               string            `json:"node_id"`
	ModelID              string            `json:"model_id"`
	QuantizationID       string            `json:"quantization_id,omitempty"`
	RespondedAtUnixMs    int64             `json:"responded_at_unix_ms"`
	LatencyMs            int64             `json:"latency_ms"`
	SparseLogitCheckHash string            `json:"sparse_logit_check_hash,omitempty"`
	Decision             ReadinessDecision `json:"decision"`
	Reason               string            `json:"reason,omitempty"`
	SignaturePlaceholder string            `json:"signature_placeholder"`
}
