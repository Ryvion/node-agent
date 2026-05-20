package modelprepare

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"path"
	"path/filepath"
	"strings"
)

const (
	PrepareTask         = "v7_prepare_model"
	PrepareFlagEnv      = "RYV_NODE_V7_MODEL_PREPARE"
	AllowFileURIFlagEnv = "RYV_NODE_V7_MODEL_PREPARE_ALLOW_FILE_URI"

	defaultPrepareTimeoutMs = int64(30 * 60 * 1000)
	maxPrepareTimeoutMs     = int64(4 * 60 * 60 * 1000)
	maxPrepareTextLen       = 256
	maxPreparePathLen       = 2048

	backendLlamaCPP = "llama.cpp"
)

var ErrInvalidPrepareSpec = errors.New("modelprepare: invalid prepare spec")

type PrepareSpec struct {
	Task                     string `json:"task"`
	PrepareID                string `json:"prepare_id"`
	RequestID                string `json:"request_id"`
	JobID                    string `json:"job_id"`
	ModelID                  string `json:"model_id"`
	ArtifactURI              string `json:"artifact_uri"`
	ArtifactSHA256           string `json:"artifact_sha256"`
	ArtifactSizeBytes        int64  `json:"artifact_size_bytes"`
	Backend                  string `json:"backend"`
	KeepWarm                 bool   `json:"keep_warm"`
	RunBenchmarkAfterPrepare bool   `json:"run_benchmark_after_prepare"`
	TimeoutMs                int64  `json:"timeout_ms"`
	ModelFamily              string `json:"model_family,omitempty"`
	ArtifactFormat           string `json:"artifact_format,omitempty"`
}

type PrepareAssignmentIdentity struct {
	PrepareID string
	RequestID string
	JobID     string
	ModelID   string
}

func IsPrepareSpecJSON(specJSON string) bool {
	var header struct {
		Task string `json:"task"`
	}
	if json.Unmarshal([]byte(strings.TrimSpace(specJSON)), &header) != nil {
		return false
	}
	return strings.TrimSpace(header.Task) == PrepareTask
}

func PrepareEnabledFromEnv(getenv func(string) string) bool {
	if getenv == nil {
		return false
	}
	return strings.TrimSpace(getenv(PrepareFlagEnv)) == "1"
}

func FileURIAllowedFromEnv(getenv func(string) string) bool {
	if getenv == nil {
		return false
	}
	return strings.TrimSpace(getenv(AllowFileURIFlagEnv)) == "1"
}

func PrepareAssignmentIdentityFromJSON(specJSON string) (PrepareAssignmentIdentity, bool) {
	var header struct {
		Task      string `json:"task"`
		PrepareID string `json:"prepare_id"`
		RequestID string `json:"request_id"`
		JobID     string `json:"job_id"`
		ModelID   string `json:"model_id"`
	}
	if json.Unmarshal([]byte(strings.TrimSpace(specJSON)), &header) != nil {
		return PrepareAssignmentIdentity{}, false
	}
	if strings.TrimSpace(header.Task) != PrepareTask {
		return PrepareAssignmentIdentity{}, false
	}
	return PrepareAssignmentIdentity{
		PrepareID: cleanPrepareText(header.PrepareID, maxPrepareTextLen),
		RequestID: cleanPrepareText(header.RequestID, maxPrepareTextLen),
		JobID:     cleanPrepareText(header.JobID, maxPrepareTextLen),
		ModelID:   cleanPrepareText(header.ModelID, maxPrepareTextLen),
	}, true
}

func DecodePrepareSpec(specJSON string) (PrepareSpec, error) {
	raw := strings.TrimSpace(specJSON)
	if raw == "" {
		return PrepareSpec{}, fmt.Errorf("%w: spec_json required", ErrInvalidPrepareSpec)
	}
	var spec PrepareSpec
	if err := json.Unmarshal([]byte(raw), &spec); err != nil {
		return PrepareSpec{}, fmt.Errorf("%w: decode spec_json: %v", ErrInvalidPrepareSpec, err)
	}
	var aliases map[string]json.RawMessage
	if err := json.Unmarshal([]byte(raw), &aliases); err == nil {
		if spec.ModelFamily == "" {
			spec.ModelFamily = decodeStringAlias(aliases, "family")
		}
		if spec.ArtifactFormat == "" {
			spec.ArtifactFormat = firstNonEmptyPrepare(
				decodeStringAlias(aliases, "format"),
				decodeStringAlias(aliases, "model_format"),
			)
		}
	}
	spec = NormalizePrepareSpec(spec)
	if err := ValidatePrepareSpec(spec); err != nil {
		return PrepareSpec{}, err
	}
	return spec, nil
}

func ValidatePrepareSpec(spec PrepareSpec) error {
	spec = NormalizePrepareSpec(spec)
	var errs []error
	if spec.Task == "" {
		errs = append(errs, fmt.Errorf("%w: task required", ErrInvalidPrepareSpec))
	} else if spec.Task != PrepareTask {
		errs = append(errs, fmt.Errorf("%w: task must be %q", ErrInvalidPrepareSpec, PrepareTask))
	}
	if spec.PrepareID == "" {
		errs = append(errs, fmt.Errorf("%w: prepare_id required", ErrInvalidPrepareSpec))
	}
	if spec.RequestID == "" {
		errs = append(errs, fmt.Errorf("%w: request_id required", ErrInvalidPrepareSpec))
	}
	if spec.JobID == "" {
		errs = append(errs, fmt.Errorf("%w: job_id required", ErrInvalidPrepareSpec))
	}
	if spec.ModelID == "" {
		errs = append(errs, fmt.Errorf("%w: model_id required", ErrInvalidPrepareSpec))
	}
	if spec.ArtifactURI == "" {
		errs = append(errs, fmt.Errorf("%w: artifact_uri required", ErrInvalidPrepareSpec))
	} else if _, err := url.Parse(spec.ArtifactURI); err != nil {
		errs = append(errs, fmt.Errorf("%w: artifact_uri invalid", ErrInvalidPrepareSpec))
	}
	if spec.ArtifactSizeBytes <= 0 {
		errs = append(errs, fmt.Errorf("%w: artifact_size_bytes must be greater than zero", ErrInvalidPrepareSpec))
	}
	if spec.ArtifactSHA256 != "" {
		if normalized := NormalizeSHA256(spec.ArtifactSHA256); normalized == "" {
			errs = append(errs, fmt.Errorf("%w: artifact_sha256 must be sha256:<64 hex>", ErrInvalidPrepareSpec))
		}
	}
	if spec.Backend == "" {
		errs = append(errs, fmt.Errorf("%w: backend required", ErrInvalidPrepareSpec))
	} else if spec.Backend != backendLlamaCPP {
		errs = append(errs, fmt.Errorf("%w: backend must be %q", ErrInvalidPrepareSpec, backendLlamaCPP))
	}
	if spec.TimeoutMs <= 0 {
		errs = append(errs, fmt.Errorf("%w: timeout_ms must be greater than zero", ErrInvalidPrepareSpec))
	} else if spec.TimeoutMs > maxPrepareTimeoutMs {
		errs = append(errs, fmt.Errorf("%w: timeout_ms exceeds maximum %d", ErrInvalidPrepareSpec, maxPrepareTimeoutMs))
	}
	return errors.Join(errs...)
}

func NormalizePrepareSpec(spec PrepareSpec) PrepareSpec {
	spec.Task = strings.TrimSpace(spec.Task)
	spec.PrepareID = cleanPrepareText(spec.PrepareID, maxPrepareTextLen)
	spec.RequestID = cleanPrepareText(spec.RequestID, maxPrepareTextLen)
	spec.JobID = cleanPrepareText(spec.JobID, maxPrepareTextLen)
	spec.ModelID = cleanPrepareText(spec.ModelID, maxPrepareTextLen)
	spec.ArtifactURI = cleanPrepareText(spec.ArtifactURI, maxPreparePathLen)
	spec.ArtifactSHA256 = NormalizeSHA256(spec.ArtifactSHA256)
	spec.Backend = normalizeBackend(spec.Backend)
	if spec.Backend == "" {
		spec.Backend = backendLlamaCPP
	}
	if spec.TimeoutMs == 0 {
		spec.TimeoutMs = defaultPrepareTimeoutMs
	}
	spec.ModelFamily = cleanPrepareText(strings.ToLower(firstNonEmptyPrepare(spec.ModelFamily, inferModelFamily(spec.ModelID))), 64)
	spec.ArtifactFormat = cleanPrepareText(strings.ToLower(firstNonEmptyPrepare(spec.ArtifactFormat, inferArtifactFormat(spec.ModelID), inferArtifactFormat(spec.ArtifactURI))), 64)
	return spec
}

func decodeStringAlias(raw map[string]json.RawMessage, key string) string {
	value, ok := raw[key]
	if !ok {
		return ""
	}
	var out string
	if json.Unmarshal(value, &out) != nil {
		return ""
	}
	return out
}

func normalizeBackend(value string) string {
	value = strings.TrimSpace(value)
	normalized := strings.ToLower(value)
	normalized = strings.NewReplacer(".", "", "_", "", "-", "", " ", "").Replace(normalized)
	if normalized == "llamacpp" {
		return backendLlamaCPP
	}
	return cleanPrepareText(value, maxPrepareTextLen)
}

func inferArtifactFormat(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	if parsed, err := url.Parse(value); err == nil && parsed.Path != "" {
		value = path.Base(parsed.Path)
	}
	ext := strings.TrimPrefix(strings.ToLower(filepath.Ext(value)), ".")
	if ext == "" {
		return ""
	}
	return ext
}

func inferModelFamily(modelID string) string {
	lower := strings.ToLower(strings.TrimSpace(modelID))
	switch {
	case strings.Contains(lower, "llama"):
		return "llama"
	case strings.Contains(lower, "phi"):
		return "phi"
	case strings.Contains(lower, "qwen"):
		return "qwen"
	case strings.Contains(lower, "gemma"):
		return "gemma"
	default:
		return ""
	}
}

func cleanPrepareText(value string, maxRunes int) string {
	value = strings.Join(strings.Fields(strings.TrimSpace(value)), " ")
	if value == "" || maxRunes <= 0 {
		return ""
	}
	runes := []rune(value)
	if len(runes) > maxRunes {
		runes = runes[:maxRunes]
	}
	return strings.TrimSpace(string(runes))
}

func firstNonEmptyPrepare(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}
