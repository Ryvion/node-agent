package dashboardinference

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/Ryvion/node-agent/internal/v7/llamacpp"
)

var ErrInvalidReceipt = errors.New("dashboardinference: invalid receipt")

type Receipt struct {
	JobID         string
	ResultHashHex string
	MeteringUnits uint64
	Metadata      map[string]any
}

type ReceiptMetadata struct {
	RequestID                string  `json:"request_id"`
	RunID                    string  `json:"run_id"`
	JobID                    string  `json:"job_id"`
	Backend                  string  `json:"backend"`
	ModelID                  string  `json:"model_id"`
	OutputHash               string  `json:"output_hash"`
	OutputBytes              int64   `json:"output_bytes"`
	RequestedMaxTokens       int     `json:"requested_max_tokens"`
	TokensGenerated          int64   `json:"tokens_generated"`
	FinishReason             string  `json:"finish_reason"`
	BackendFinishReason      string  `json:"backend_finish_reason"`
	BackendStopReason        string  `json:"backend_stop_reason"`
	MaxTokensReached         bool    `json:"max_tokens_reached"`
	TTFTMs                   int64   `json:"ttft_ms"`
	TotalTimeMs              int64   `json:"total_time_ms"`
	DecodeTPS                float64 `json:"decode_tps"`
	TPOTMs                   float64 `json:"tpot_ms"`
	EndToEndTPS              float64 `json:"end_to_end_tps"`
	ProofStatus              string  `json:"proof_status"`
	RuntimeMeasurementStatus string  `json:"runtime_measurement_status"`
	MetadataParseStatus      string  `json:"metadata_parse_status"`
	TokenCountEstimated      bool    `json:"token_count_estimated,omitempty"`
	PromptHash               string  `json:"prompt_hash,omitempty"`
	PromptProfileID          string  `json:"prompt_profile_id,omitempty"`
	ErrorCode                string  `json:"error_code,omitempty"`
	GeneratedText            string  `json:"generated_text,omitempty"`
	GeneratedTextTruncated   bool    `json:"generated_text_truncated,omitempty"`
	GroundingApplied         bool    `json:"grounding_applied"`
	PromptMode               string  `json:"prompt_mode,omitempty"`
	SystemPromptHash         string  `json:"system_prompt_hash,omitempty"`
	MaxReturnChars           int     `json:"max_return_chars"`
}

func BuildReceipt(result ExecutionResult) (Receipt, error) {
	result = normalizeExecutionResult(result)
	if err := ValidateSpec(result.Spec); err != nil {
		return Receipt{}, err
	}
	metadata := receiptMetadataFromResult(result)
	if err := validateReceiptMetadata(result.Spec, metadata); err != nil {
		return Receipt{}, err
	}
	hashHex, err := HashReceiptMetadata(result.Spec.JobID, metadata)
	if err != nil {
		return Receipt{}, err
	}
	meteringUnits := uint64(0)
	if metadata.ProofStatus == ProofStatusMeasured {
		meteringUnits = 1
	}
	return Receipt{
		JobID:         result.Spec.JobID,
		ResultHashHex: hashHex,
		MeteringUnits: meteringUnits,
		Metadata: map[string]any{
			Task: metadata.MapWithResultHash(hashHex),
		},
	}, nil
}

func BuildRejectionReceipt(spec Spec, runErr error) Receipt {
	spec = normalizeSpec(spec)
	identity := AssignmentIdentity{
		RequestID:       spec.RequestID,
		RunID:           spec.RunID,
		JobID:           spec.JobID,
		Backend:         spec.Backend,
		ModelID:         spec.ModelID,
		PromptHash:      spec.PromptHash,
		PromptProfileID: spec.PromptProfileID,
	}
	return BuildRejectionReceiptFromIdentity(identity, runErr)
}

func BuildRejectionReceiptFromIdentity(identity AssignmentIdentity, runErr error) Receipt {
	jobID := cleanText(identity.JobID, maxIDLen)
	if jobID == "" {
		jobID = "v7-dashboard-inference-rejected"
	}
	backend := normalizeBackendName(identity.Backend)
	if backend == "" {
		backend = llamacpp.BackendName
	}
	metadata := ReceiptMetadata{
		RequestID:                firstNonEmpty(cleanText(identity.RequestID, maxIDLen), "unknown_request"),
		RunID:                    firstNonEmpty(cleanText(identity.RunID, maxIDLen), "unknown_run"),
		JobID:                    jobID,
		Backend:                  backend,
		ModelID:                  firstNonEmpty(cleanText(identity.ModelID, maxModelIDLen), "unknown_model"),
		OutputHash:               HashOutput(jobID, nil),
		OutputBytes:              0,
		RequestedMaxTokens:       0,
		TokensGenerated:          0,
		FinishReason:             llamacpp.FinishReasonError,
		BackendFinishReason:      llamacpp.FinishReasonUnknown,
		BackendStopReason:        llamacpp.FinishReasonUnknown,
		TTFTMs:                   0,
		TotalTimeMs:              0,
		DecodeTPS:                0,
		TPOTMs:                   0,
		EndToEndTPS:              0,
		ProofStatus:              ProofStatusRejected,
		RuntimeMeasurementStatus: llamacpp.RuntimeMeasurementStatusUnknown,
		MetadataParseStatus:      llamacpp.MetadataParseStatusOK,
		PromptHash:               cleanHash(identity.PromptHash),
		PromptProfileID:          cleanText(identity.PromptProfileID, maxIDLen),
		ErrorCode:                ErrorCode(runErr),
		MaxReturnChars:           defaultMaxReturnChars,
	}
	if metadata.ErrorCode == "" {
		metadata.ErrorCode = "dashboard_inference_rejected"
	}
	hashHex, _ := HashReceiptMetadata(jobID, metadata)
	return Receipt{
		JobID:         jobID,
		ResultHashHex: hashHex,
		MeteringUnits: 0,
		Metadata: map[string]any{
			Task: metadata.MapWithResultHash(hashHex),
		},
	}
}

func HashReceiptMetadata(jobID string, metadata ReceiptMetadata) (string, error) {
	jobID = cleanText(jobID, maxIDLen)
	if jobID == "" {
		return "", fmt.Errorf("%w: job_id required for result hash", ErrInvalidReceipt)
	}
	if err := validateReceiptMetadataForHash(metadata); err != nil {
		return "", err
	}
	envelope := struct {
		Task     string         `json:"task"`
		JobID    string         `json:"job_id"`
		Metadata map[string]any `json:"v7_dashboard_inference"`
	}{
		Task:     Task,
		JobID:    jobID,
		Metadata: metadata.Map(),
	}
	encoded, err := json.Marshal(envelope)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:]), nil
}

func HashOutput(jobID string, output []byte) string {
	hash := sha256.New()
	hash.Write([]byte("ryvion:v7:dashboard_inference_output:v1\n"))
	hash.Write([]byte("job_id:"))
	hash.Write([]byte(cleanText(jobID, maxIDLen)))
	hash.Write([]byte("\nbytes:"))
	hash.Write([]byte(fmt.Sprintf("%d\n", len(output))))
	hash.Write(output)
	return "sha256:" + hex.EncodeToString(hash.Sum(nil))
}

func (m ReceiptMetadata) Map() map[string]any {
	m = m.clone()
	out := map[string]any{
		"request_id":                 m.RequestID,
		"run_id":                     m.RunID,
		"job_id":                     m.JobID,
		"backend":                    m.Backend,
		"model_id":                   m.ModelID,
		"output_hash":                m.OutputHash,
		"output_bytes":               m.OutputBytes,
		"requested_max_tokens":       m.RequestedMaxTokens,
		"tokens_generated":           m.TokensGenerated,
		"finish_reason":              m.FinishReason,
		"backend_finish_reason":      m.BackendFinishReason,
		"backend_stop_reason":        m.BackendStopReason,
		"max_tokens_reached":         m.MaxTokensReached,
		"ttft_ms":                    m.TTFTMs,
		"total_time_ms":              m.TotalTimeMs,
		"decode_tps":                 m.DecodeTPS,
		"tpot_ms":                    m.TPOTMs,
		"end_to_end_tps":             m.EndToEndTPS,
		"p50_ttft_ms":                m.TTFTMs,
		"p50_decode_tps":             m.DecodeTPS,
		"p50_end_to_end_tps":         m.EndToEndTPS,
		"proof_status":               m.ProofStatus,
		"runtime_measurement_status": m.RuntimeMeasurementStatus,
		"metadata_parse_status":      m.MetadataParseStatus,
		"generated_text_truncated":   m.GeneratedTextTruncated,
		"grounding_applied":          m.GroundingApplied,
		"max_return_chars":           m.MaxReturnChars,
	}
	if m.TokenCountEstimated {
		out["token_count_estimated"] = true
	}
	if m.PromptHash != "" {
		out["prompt_hash"] = m.PromptHash
	}
	if m.PromptProfileID != "" {
		out["prompt_profile_id"] = m.PromptProfileID
	}
	if m.ErrorCode != "" {
		out["error_code"] = m.ErrorCode
	}
	if m.GeneratedText != "" {
		out["generated_text"] = m.GeneratedText
		out["generated_text_truncated"] = m.GeneratedTextTruncated
	}
	if m.PromptMode != "" {
		out["prompt_mode"] = m.PromptMode
	}
	if m.SystemPromptHash != "" {
		out["system_prompt_hash"] = m.SystemPromptHash
	}
	return out
}

func (m ReceiptMetadata) MapWithResultHash(resultHashHex string) map[string]any {
	out := m.Map()
	if hashHex64(resultHashHex) {
		out["result_hash_hex"] = strings.TrimSpace(resultHashHex)
	}
	return out
}

func ReceiptProofStatus(receipt Receipt) string {
	if receipt.Metadata == nil {
		return ""
	}
	taskMetadata, ok := receipt.Metadata[Task].(map[string]any)
	if !ok {
		return ""
	}
	status, _ := taskMetadata["proof_status"].(string)
	return cleanText(status, maxIDLen)
}

func (m ReceiptMetadata) clone() ReceiptMetadata {
	m.RequestID = cleanText(m.RequestID, maxIDLen)
	m.RunID = cleanText(m.RunID, maxIDLen)
	m.JobID = cleanText(m.JobID, maxIDLen)
	m.Backend = normalizeBackendName(m.Backend)
	m.ModelID = cleanText(m.ModelID, maxModelIDLen)
	m.OutputHash = cleanHash(m.OutputHash)
	m.ProofStatus = cleanText(m.ProofStatus, maxIDLen)
	m.PromptHash = cleanHash(m.PromptHash)
	m.PromptProfileID = cleanText(m.PromptProfileID, maxIDLen)
	m.ErrorCode = cleanErrorCode(m.ErrorCode)
	m.PromptMode = cleanPromptMode(m.PromptMode)
	m.SystemPromptHash = cleanHash(m.SystemPromptHash)
	if m.OutputBytes < 0 {
		m.OutputBytes = 0
	}
	if m.RequestedMaxTokens < 0 {
		m.RequestedMaxTokens = 0
	}
	if m.TokensGenerated < 0 {
		m.TokensGenerated = 0
	}
	m.FinishReason = cleanFinishReason(m.FinishReason)
	m.BackendFinishReason = cleanFinishDetail(m.BackendFinishReason)
	m.BackendStopReason = cleanFinishDetail(m.BackendStopReason)
	if m.FinishReason == "" {
		if m.ProofStatus == ProofStatusRejected {
			m.FinishReason = finishReasonFromErrorCode(m.ErrorCode)
		} else {
			m.FinishReason = llamacpp.FinishReasonUnknown
		}
	}
	if m.BackendFinishReason == "" {
		m.BackendFinishReason = llamacpp.FinishReasonUnknown
	}
	if m.BackendStopReason == "" {
		m.BackendStopReason = llamacpp.FinishReasonUnknown
	}
	if m.TTFTMs < 0 {
		m.TTFTMs = 0
	}
	if m.TotalTimeMs < 0 {
		m.TotalTimeMs = 0
	}
	m.DecodeTPS = roundTPS(m.DecodeTPS)
	m.TPOTMs = roundTPS(m.TPOTMs)
	m.EndToEndTPS = roundTPS(m.EndToEndTPS)
	if m.ProofStatus == "" {
		m.ProofStatus = ProofStatusRejected
	}
	m.RuntimeMeasurementStatus = normalizeRuntimeMeasurementStatus(m.RuntimeMeasurementStatus)
	m.MetadataParseStatus = normalizeMetadataParseStatus(m.MetadataParseStatus)
	if m.TokensGenerated <= 0 {
		m.RuntimeMeasurementStatus = llamacpp.RuntimeMeasurementStatusUnknown
		if m.MetadataParseStatus == "" {
			m.MetadataParseStatus = llamacpp.MetadataParseStatusPartial
		}
	} else if m.RuntimeMeasurementStatus == "" {
		m.RuntimeMeasurementStatus = llamacpp.RuntimeMeasurementStatusMeasured
	}
	if m.MetadataParseStatus == "" {
		m.MetadataParseStatus = llamacpp.MetadataParseStatusOK
	}
	if m.MaxReturnChars <= 0 || m.MaxReturnChars > defaultMaxReturnChars {
		m.MaxReturnChars = defaultMaxReturnChars
	}
	if m.ProofStatus == ProofStatusRejected {
		m.GeneratedText = ""
		m.GeneratedTextTruncated = false
		m.TokenCountEstimated = false
		m.RuntimeMeasurementStatus = llamacpp.RuntimeMeasurementStatusUnknown
		if m.FinishReason == llamacpp.FinishReasonUnknown {
			m.FinishReason = llamacpp.FinishReasonError
		}
	} else if m.GeneratedText != "" {
		text, truncated := truncateGeneratedText([]byte(m.GeneratedText), defaultMaxReturnChars)
		m.GeneratedText = text
		m.GeneratedTextTruncated = m.GeneratedTextTruncated || truncated
	}
	return m
}

func receiptMetadataFromResult(result ExecutionResult) ReceiptMetadata {
	result = normalizeExecutionResult(result)
	return ReceiptMetadata{
		RequestID:                result.Spec.RequestID,
		RunID:                    result.Spec.RunID,
		JobID:                    result.Spec.JobID,
		Backend:                  firstNonEmpty(result.Backend, result.Spec.Backend),
		ModelID:                  firstNonEmpty(result.ModelID, result.Spec.ModelID),
		OutputHash:               firstNonEmpty(result.OutputHash, HashOutput(result.Spec.JobID, nil)),
		OutputBytes:              result.OutputBytes,
		RequestedMaxTokens:       result.RequestedMaxTokens,
		TokensGenerated:          result.TokensGenerated,
		FinishReason:             result.FinishReason,
		BackendFinishReason:      result.BackendFinishReason,
		BackendStopReason:        result.BackendStopReason,
		MaxTokensReached:         result.MaxTokensReached,
		TTFTMs:                   result.TTFTMs,
		TotalTimeMs:              result.TotalTimeMs,
		DecodeTPS:                result.DecodeTPS,
		TPOTMs:                   result.TPOTMs,
		EndToEndTPS:              result.EndToEndTPS,
		ProofStatus:              result.ProofStatus,
		RuntimeMeasurementStatus: result.RuntimeMeasurementStatus,
		MetadataParseStatus:      result.MetadataParseStatus,
		TokenCountEstimated:      result.TokenCountEstimated,
		PromptHash:               result.Spec.PromptHash,
		PromptProfileID:          result.Spec.PromptProfileID,
		ErrorCode:                result.ErrorCode,
		GeneratedText:            result.GeneratedText,
		GeneratedTextTruncated:   result.GeneratedTextTruncated,
		GroundingApplied:         result.GroundingApplied,
		PromptMode:               result.PromptMode,
		SystemPromptHash:         result.SystemPromptHash,
		MaxReturnChars:           result.MaxReturnChars,
	}.clone()
}

func normalizeExecutionResult(result ExecutionResult) ExecutionResult {
	result.Spec = normalizeSpec(result.Spec)
	result.Backend = normalizeBackendName(firstNonEmpty(result.Backend, result.Spec.Backend))
	result.ModelID = cleanText(firstNonEmpty(result.ModelID, result.Spec.ModelID), maxModelIDLen)
	result.OutputHash = cleanHash(result.OutputHash)
	result.ProofStatus = cleanText(result.ProofStatus, maxIDLen)
	if result.ProofStatus == "" {
		result.ProofStatus = ProofStatusRejected
	}
	result.ErrorCode = cleanErrorCode(result.ErrorCode)
	result.PromptMode = cleanPromptMode(result.PromptMode)
	result.SystemPromptHash = cleanHash(result.SystemPromptHash)
	finish := llamacpp.NormalizeCompletionFinishMetadata(llamacpp.CompletionResult{
		RequestedMaxTokens:  result.RequestedMaxTokens,
		TokensGenerated:     result.TokensGenerated,
		FinishReason:        result.FinishReason,
		BackendFinishReason: result.BackendFinishReason,
		BackendStopReason:   result.BackendStopReason,
		MaxTokensReached:    result.MaxTokensReached,
	}, result.Spec.MaxTokens, result.TokensGenerated)
	result.RequestedMaxTokens = finish.RequestedMaxTokens
	result.FinishReason = finish.FinishReason
	result.BackendFinishReason = finish.BackendFinishReason
	result.BackendStopReason = finish.BackendStopReason
	result.MaxTokensReached = finish.MaxTokensReached
	if result.ProofStatus == ProofStatusRejected {
		result.FinishReason = finishReasonFromErrorCode(result.ErrorCode)
	}
	if result.MaxReturnChars <= 0 || result.MaxReturnChars > defaultMaxReturnChars {
		result.MaxReturnChars = result.Spec.MaxReturnChars
	}
	if !result.Spec.ReturnText || result.ProofStatus != ProofStatusMeasured {
		result.GeneratedText = ""
		result.GeneratedTextTruncated = false
	} else if result.GeneratedText != "" {
		text, truncated := truncateGeneratedText([]byte(result.GeneratedText), result.Spec.MaxReturnChars)
		result.GeneratedText = text
		result.GeneratedTextTruncated = result.GeneratedTextTruncated || truncated
	}
	if result.OutputBytes < 0 {
		result.OutputBytes = 0
	}
	if result.TokensGenerated < 0 {
		result.TokensGenerated = 0
	}
	if result.TTFTMs < 0 {
		result.TTFTMs = 0
	}
	if result.TotalTimeMs < 0 {
		result.TotalTimeMs = 0
	}
	result.DecodeTPS = roundTPS(result.DecodeTPS)
	result.TPOTMs = roundTPS(result.TPOTMs)
	result.EndToEndTPS = roundTPS(result.EndToEndTPS)
	result.RuntimeMeasurementStatus = normalizeRuntimeMeasurementStatus(result.RuntimeMeasurementStatus)
	result.MetadataParseStatus = normalizeMetadataParseStatus(result.MetadataParseStatus)
	if result.TokensGenerated <= 0 {
		result.RuntimeMeasurementStatus = llamacpp.RuntimeMeasurementStatusUnknown
		if result.MetadataParseStatus == "" {
			result.MetadataParseStatus = llamacpp.MetadataParseStatusPartial
		}
		result.TokenCountEstimated = false
	} else if result.RuntimeMeasurementStatus == "" {
		result.RuntimeMeasurementStatus = llamacpp.RuntimeMeasurementStatusMeasured
	}
	if result.MetadataParseStatus == "" {
		result.MetadataParseStatus = llamacpp.MetadataParseStatusOK
	}
	return result
}

func validateReceiptMetadata(spec Spec, metadata ReceiptMetadata) error {
	spec = normalizeSpec(spec)
	metadata = metadata.clone()
	var errs []error
	if metadata.RequestID != spec.RequestID {
		errs = append(errs, fmt.Errorf("%w: receipt request_id must match spec", ErrInvalidReceipt))
	}
	if metadata.RunID != spec.RunID {
		errs = append(errs, fmt.Errorf("%w: receipt run_id must match spec", ErrInvalidReceipt))
	}
	if metadata.JobID != spec.JobID {
		errs = append(errs, fmt.Errorf("%w: receipt job_id must match spec", ErrInvalidReceipt))
	}
	if metadata.Backend != spec.Backend {
		errs = append(errs, fmt.Errorf("%w: receipt backend must match spec", ErrInvalidReceipt))
	}
	if metadata.ModelID != spec.ModelID {
		errs = append(errs, fmt.Errorf("%w: receipt model_id must match spec", ErrInvalidReceipt))
	}
	if metadata.PromptHash != cleanHash(spec.PromptHash) {
		errs = append(errs, fmt.Errorf("%w: receipt prompt_hash must match spec when present", ErrInvalidReceipt))
	}
	if metadata.PromptProfileID != spec.PromptProfileID {
		errs = append(errs, fmt.Errorf("%w: receipt prompt_profile_id must match spec when present", ErrInvalidReceipt))
	}
	if err := validateReceiptMetadataForHash(metadata); err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

func validateReceiptMetadataForHash(metadata ReceiptMetadata) error {
	metadata = metadata.clone()
	var errs []error
	if metadata.RequestID == "" {
		errs = append(errs, fmt.Errorf("%w: receipt request_id required", ErrInvalidReceipt))
	}
	if metadata.RunID == "" {
		errs = append(errs, fmt.Errorf("%w: receipt run_id required", ErrInvalidReceipt))
	}
	if metadata.JobID == "" {
		errs = append(errs, fmt.Errorf("%w: receipt job_id required", ErrInvalidReceipt))
	}
	if metadata.Backend == "" {
		errs = append(errs, fmt.Errorf("%w: receipt backend required", ErrInvalidReceipt))
	}
	if metadata.ModelID == "" {
		errs = append(errs, fmt.Errorf("%w: receipt model_id required", ErrInvalidReceipt))
	}
	if err := validateSHA256ID(metadata.OutputHash, "receipt output_hash", ErrInvalidReceipt); err != nil {
		errs = append(errs, err)
	}
	if metadata.OutputBytes < 0 || metadata.TokensGenerated < 0 || metadata.TTFTMs < 0 || metadata.TotalTimeMs < 0 {
		errs = append(errs, fmt.Errorf("%w: receipt metrics must be non-negative", ErrInvalidReceipt))
	}
	if metadata.RequestedMaxTokens < 0 || metadata.MaxReturnChars <= 0 {
		errs = append(errs, fmt.Errorf("%w: receipt token/display caps must be valid", ErrInvalidReceipt))
	}
	if cleanFinishReason(metadata.FinishReason) == "" {
		errs = append(errs, fmt.Errorf("%w: receipt finish_reason unknown %q", ErrInvalidReceipt, metadata.FinishReason))
	}
	if !finiteTPS(metadata.DecodeTPS) || !finiteTPS(metadata.TPOTMs) || !finiteTPS(metadata.EndToEndTPS) {
		errs = append(errs, fmt.Errorf("%w: receipt tps metrics must be finite", ErrInvalidReceipt))
	}
	if normalizeRuntimeMeasurementStatus(metadata.RuntimeMeasurementStatus) == "" {
		errs = append(errs, fmt.Errorf("%w: receipt runtime_measurement_status unknown %q", ErrInvalidReceipt, metadata.RuntimeMeasurementStatus))
	}
	if normalizeMetadataParseStatus(metadata.MetadataParseStatus) == "" {
		errs = append(errs, fmt.Errorf("%w: receipt metadata_parse_status unknown %q", ErrInvalidReceipt, metadata.MetadataParseStatus))
	}
	if metadata.PromptHash != "" {
		if err := validateSHA256ID(metadata.PromptHash, "receipt prompt_hash", ErrInvalidReceipt); err != nil {
			errs = append(errs, err)
		}
	}
	if metadata.PromptMode != "" && cleanPromptMode(metadata.PromptMode) == "" {
		errs = append(errs, fmt.Errorf("%w: receipt prompt_mode unknown %q", ErrInvalidReceipt, metadata.PromptMode))
	}
	if metadata.SystemPromptHash != "" {
		if err := validateSHA256ID(metadata.SystemPromptHash, "receipt system_prompt_hash", ErrInvalidReceipt); err != nil {
			errs = append(errs, err)
		}
	}
	switch metadata.ProofStatus {
	case ProofStatusMeasured:
		if metadata.ErrorCode != "" {
			errs = append(errs, fmt.Errorf("%w: measured receipt must not include error_code", ErrInvalidReceipt))
		}
	case ProofStatusRejected:
		if metadata.ErrorCode == "" {
			errs = append(errs, fmt.Errorf("%w: rejected receipt requires error_code", ErrInvalidReceipt))
		}
		if metadata.GeneratedText != "" {
			errs = append(errs, fmt.Errorf("%w: rejected receipt must not include generated_text", ErrInvalidReceipt))
		}
	default:
		errs = append(errs, fmt.Errorf("%w: receipt proof_status unknown %q", ErrInvalidReceipt, metadata.ProofStatus))
	}
	return errors.Join(errs...)
}

func ReceiptJSONContainsNoRawText(receipt Receipt) bool {
	raw, err := json.Marshal(receipt.Metadata)
	if err != nil {
		return false
	}
	lower := bytes.ToLower(raw)
	for _, marker := range forbiddenReceiptJSONMarkers() {
		if bytes.Contains(lower, bytes.ToLower([]byte(marker))) {
			return false
		}
	}
	return true
}

func forbiddenReceiptJSONMarkers() []string {
	markers := forbiddenTextMarkers()
	for idx, marker := range markers {
		if marker == "generated_text" {
			markers[idx] = `"generated_text":`
		} else if marker == "messages" {
			markers[idx] = `"messages"`
		}
	}
	return markers
}

func forbiddenTextMarkers() []string {
	return []string{
		internalDashboardPrompt,
		"raw_prompt",
		"prompt_text",
		"messages",
		"input_text",
		"output_text",
		"generated_text",
		"raw_output",
		"completion",
		"token_logprobs",
		"logprobs",
		"tokens array",
		`"token"`,
		"key_data",
		"value_data",
		"query_vector",
		"tensor_bytes",
		"secret",
	}
}

func cleanFinishReason(value string) string {
	switch cleanFinishDetail(value) {
	case llamacpp.FinishReasonStop:
		return llamacpp.FinishReasonStop
	case llamacpp.FinishReasonLength:
		return llamacpp.FinishReasonLength
	case llamacpp.FinishReasonMaxTokens:
		return llamacpp.FinishReasonMaxTokens
	case llamacpp.FinishReasonTimeout:
		return llamacpp.FinishReasonTimeout
	case llamacpp.FinishReasonError:
		return llamacpp.FinishReasonError
	case llamacpp.FinishReasonUnknown:
		return llamacpp.FinishReasonUnknown
	default:
		return ""
	}
}

func cleanPromptMode(value string) string {
	value = strings.ToLower(cleanText(value, maxIDLen))
	value = strings.ReplaceAll(value, "-", "_")
	value = strings.ReplaceAll(value, " ", "_")
	switch value {
	case "",
		llamacpp.PromptModeChatMessages,
		llamacpp.PromptModeTemplate,
		llamacpp.PromptModeRawCompletion:
		return value
	default:
		return ""
	}
}

func cleanFinishDetail(value string) string {
	value = strings.ToLower(cleanText(value, maxIDLen))
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	lower := strings.ToLower(value)
	for _, marker := range forbiddenTextMarkers() {
		if strings.Contains(lower, strings.ToLower(marker)) {
			return ""
		}
	}
	value = strings.NewReplacer(" ", "_", "-", "_").Replace(value)
	for _, r := range value {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '_' || r == '.' || r == ':' {
			continue
		}
		return ""
	}
	return value
}

func cleanHash(value string) string {
	value = strings.TrimSpace(value)
	if validateSHA256ID(value, "hash", ErrInvalidReceipt) != nil {
		return ""
	}
	return value
}

func hashHex64(value string) bool {
	value = strings.TrimSpace(value)
	if len(value) != 64 {
		return false
	}
	for _, r := range value {
		if (r < '0' || r > '9') && (r < 'a' || r > 'f') {
			return false
		}
	}
	return true
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			return trimmed
		}
	}
	return ""
}
