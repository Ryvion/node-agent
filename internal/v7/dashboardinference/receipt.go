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
	RequestID              string  `json:"request_id"`
	RunID                  string  `json:"run_id"`
	JobID                  string  `json:"job_id"`
	Backend                string  `json:"backend"`
	ModelID                string  `json:"model_id"`
	OutputHash             string  `json:"output_hash"`
	OutputBytes            int64   `json:"output_bytes"`
	TokensGenerated        int64   `json:"tokens_generated"`
	TTFTMs                 int64   `json:"ttft_ms"`
	TotalTimeMs            int64   `json:"total_time_ms"`
	DecodeTPS              float64 `json:"decode_tps"`
	EndToEndTPS            float64 `json:"end_to_end_tps"`
	ProofStatus            string  `json:"proof_status"`
	PromptHash             string  `json:"prompt_hash,omitempty"`
	PromptProfileID        string  `json:"prompt_profile_id,omitempty"`
	ErrorCode              string  `json:"error_code,omitempty"`
	GeneratedText          string  `json:"generated_text,omitempty"`
	GeneratedTextTruncated bool    `json:"generated_text_truncated,omitempty"`
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
			Task: metadata.Map(),
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
		RequestID:       firstNonEmpty(cleanText(identity.RequestID, maxIDLen), "unknown_request"),
		RunID:           firstNonEmpty(cleanText(identity.RunID, maxIDLen), "unknown_run"),
		JobID:           jobID,
		Backend:         backend,
		ModelID:         firstNonEmpty(cleanText(identity.ModelID, maxModelIDLen), "unknown_model"),
		OutputHash:      HashOutput(jobID, nil),
		OutputBytes:     0,
		TokensGenerated: 0,
		TTFTMs:          0,
		TotalTimeMs:     0,
		DecodeTPS:       0,
		EndToEndTPS:     0,
		ProofStatus:     ProofStatusRejected,
		PromptHash:      cleanHash(identity.PromptHash),
		PromptProfileID: cleanText(identity.PromptProfileID, maxIDLen),
		ErrorCode:       ErrorCode(runErr),
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
			Task: metadata.Map(),
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
		"request_id":       m.RequestID,
		"run_id":           m.RunID,
		"job_id":           m.JobID,
		"backend":          m.Backend,
		"model_id":         m.ModelID,
		"output_hash":      m.OutputHash,
		"output_bytes":     m.OutputBytes,
		"tokens_generated": m.TokensGenerated,
		"ttft_ms":          m.TTFTMs,
		"total_time_ms":    m.TotalTimeMs,
		"decode_tps":       m.DecodeTPS,
		"end_to_end_tps":   m.EndToEndTPS,
		"proof_status":     m.ProofStatus,
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
	if m.OutputBytes < 0 {
		m.OutputBytes = 0
	}
	if m.TokensGenerated < 0 {
		m.TokensGenerated = 0
	}
	if m.TTFTMs < 0 {
		m.TTFTMs = 0
	}
	if m.TotalTimeMs < 0 {
		m.TotalTimeMs = 0
	}
	m.DecodeTPS = roundTPS(m.DecodeTPS)
	m.EndToEndTPS = roundTPS(m.EndToEndTPS)
	if m.ProofStatus == "" {
		m.ProofStatus = ProofStatusRejected
	}
	if m.ProofStatus == ProofStatusRejected {
		m.GeneratedText = ""
		m.GeneratedTextTruncated = false
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
		RequestID:              result.Spec.RequestID,
		RunID:                  result.Spec.RunID,
		JobID:                  result.Spec.JobID,
		Backend:                firstNonEmpty(result.Backend, result.Spec.Backend),
		ModelID:                firstNonEmpty(result.ModelID, result.Spec.ModelID),
		OutputHash:             firstNonEmpty(result.OutputHash, HashOutput(result.Spec.JobID, nil)),
		OutputBytes:            result.OutputBytes,
		TokensGenerated:        result.TokensGenerated,
		TTFTMs:                 result.TTFTMs,
		TotalTimeMs:            result.TotalTimeMs,
		DecodeTPS:              result.DecodeTPS,
		EndToEndTPS:            result.EndToEndTPS,
		ProofStatus:            result.ProofStatus,
		PromptHash:             result.Spec.PromptHash,
		PromptProfileID:        result.Spec.PromptProfileID,
		ErrorCode:              result.ErrorCode,
		GeneratedText:          result.GeneratedText,
		GeneratedTextTruncated: result.GeneratedTextTruncated,
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
	result.EndToEndTPS = roundTPS(result.EndToEndTPS)
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
	if !finiteTPS(metadata.DecodeTPS) || !finiteTPS(metadata.EndToEndTPS) {
		errs = append(errs, fmt.Errorf("%w: receipt tps metrics must be finite", ErrInvalidReceipt))
	}
	if metadata.PromptHash != "" {
		if err := validateSHA256ID(metadata.PromptHash, "receipt prompt_hash", ErrInvalidReceipt); err != nil {
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
	for _, marker := range forbiddenTextMarkers() {
		if bytes.Contains(lower, bytes.ToLower([]byte(marker))) {
			return false
		}
	}
	return true
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

func cleanHash(value string) string {
	value = strings.TrimSpace(value)
	if validateSHA256ID(value, "hash", ErrInvalidReceipt) != nil {
		return ""
	}
	return value
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			return trimmed
		}
	}
	return ""
}
