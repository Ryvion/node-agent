package modelwarm

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

const (
	WarmTask    = "v7_warm_model"
	WarmFlagEnv = "RYV_NODE_V7_MODEL_WARM"

	defaultWarmTimeoutMs = int64(10 * 60 * 1000)
	maxWarmTimeoutMs     = int64(60 * 60 * 1000)
	maxWarmTextLen       = 256
	maxWarmPathLen       = 2048

	backendLlamaCPP = "llama.cpp"
)

var ErrInvalidWarmSpec = errors.New("modelwarm: invalid warm spec")

type WarmSpec struct {
	Task                  string `json:"task"`
	RequestID             string `json:"request_id"`
	WarmID                string `json:"warm_id"`
	JobID                 string `json:"job_id"`
	ModelID               string `json:"model_id"`
	Backend               string `json:"backend"`
	RunBenchmarkAfterWarm bool   `json:"run_benchmark_after_warm"`
	TimeoutMs             int64  `json:"timeout_ms"`
}

type WarmAssignmentIdentity struct {
	WarmID    string
	RequestID string
	JobID     string
	ModelID   string
}

func IsWarmSpecJSON(specJSON string) bool {
	var header struct {
		Task string `json:"task"`
	}
	if json.Unmarshal([]byte(strings.TrimSpace(specJSON)), &header) != nil {
		return false
	}
	return strings.TrimSpace(header.Task) == WarmTask
}

func WarmEnabledFromEnv(getenv func(string) string) bool {
	if getenv == nil {
		return false
	}
	return strings.TrimSpace(getenv(WarmFlagEnv)) == "1"
}

func WarmAssignmentIdentityFromJSON(specJSON string) (WarmAssignmentIdentity, bool) {
	var header struct {
		Task      string `json:"task"`
		WarmID    string `json:"warm_id"`
		RequestID string `json:"request_id"`
		JobID     string `json:"job_id"`
		ModelID   string `json:"model_id"`
	}
	if json.Unmarshal([]byte(strings.TrimSpace(specJSON)), &header) != nil {
		return WarmAssignmentIdentity{}, false
	}
	if strings.TrimSpace(header.Task) != WarmTask {
		return WarmAssignmentIdentity{}, false
	}
	return WarmAssignmentIdentity{
		WarmID:    cleanWarmText(header.WarmID, maxWarmTextLen),
		RequestID: cleanWarmText(header.RequestID, maxWarmTextLen),
		JobID:     cleanWarmText(header.JobID, maxWarmTextLen),
		ModelID:   cleanWarmText(header.ModelID, maxWarmTextLen),
	}, true
}

func DecodeWarmSpec(specJSON string) (WarmSpec, error) {
	raw := strings.TrimSpace(specJSON)
	if raw == "" {
		return WarmSpec{}, fmt.Errorf("%w: spec_json required", ErrInvalidWarmSpec)
	}
	var spec WarmSpec
	if err := json.Unmarshal([]byte(raw), &spec); err != nil {
		return WarmSpec{}, fmt.Errorf("%w: decode spec_json: %v", ErrInvalidWarmSpec, err)
	}
	spec = NormalizeWarmSpec(spec)
	if err := ValidateWarmSpec(spec); err != nil {
		return WarmSpec{}, err
	}
	return spec, nil
}

func ValidateWarmSpec(spec WarmSpec) error {
	spec = NormalizeWarmSpec(spec)
	var errs []error
	if spec.Task == "" {
		errs = append(errs, fmt.Errorf("%w: task required", ErrInvalidWarmSpec))
	} else if spec.Task != WarmTask {
		errs = append(errs, fmt.Errorf("%w: task must be %q", ErrInvalidWarmSpec, WarmTask))
	}
	if spec.RequestID == "" {
		errs = append(errs, fmt.Errorf("%w: request_id required", ErrInvalidWarmSpec))
	}
	if spec.WarmID == "" {
		errs = append(errs, fmt.Errorf("%w: warm_id required", ErrInvalidWarmSpec))
	}
	if spec.JobID == "" {
		errs = append(errs, fmt.Errorf("%w: job_id required", ErrInvalidWarmSpec))
	}
	if spec.ModelID == "" {
		errs = append(errs, fmt.Errorf("%w: model_id required", ErrInvalidWarmSpec))
	}
	if spec.Backend == "" {
		errs = append(errs, fmt.Errorf("%w: backend required", ErrInvalidWarmSpec))
	} else if spec.Backend != backendLlamaCPP {
		errs = append(errs, fmt.Errorf("%w: backend must be %q", ErrInvalidWarmSpec, backendLlamaCPP))
	}
	if spec.TimeoutMs <= 0 {
		errs = append(errs, fmt.Errorf("%w: timeout_ms must be greater than zero", ErrInvalidWarmSpec))
	} else if spec.TimeoutMs > maxWarmTimeoutMs {
		errs = append(errs, fmt.Errorf("%w: timeout_ms exceeds maximum %d", ErrInvalidWarmSpec, maxWarmTimeoutMs))
	}
	return errors.Join(errs...)
}

func NormalizeWarmSpec(spec WarmSpec) WarmSpec {
	spec.Task = strings.TrimSpace(spec.Task)
	spec.RequestID = cleanWarmText(spec.RequestID, maxWarmTextLen)
	spec.WarmID = cleanWarmText(spec.WarmID, maxWarmTextLen)
	spec.JobID = cleanWarmText(spec.JobID, maxWarmTextLen)
	spec.ModelID = cleanWarmText(spec.ModelID, maxWarmTextLen)
	spec.Backend = normalizeBackend(spec.Backend)
	if spec.Backend == "" {
		spec.Backend = backendLlamaCPP
	}
	if spec.TimeoutMs == 0 {
		spec.TimeoutMs = defaultWarmTimeoutMs
	}
	return spec
}

func normalizeBackend(value string) string {
	value = strings.TrimSpace(value)
	normalized := strings.ToLower(value)
	normalized = strings.NewReplacer(".", "", "_", "", "-", "", " ", "").Replace(normalized)
	if normalized == "llamacpp" {
		return backendLlamaCPP
	}
	return cleanWarmText(value, maxWarmTextLen)
}

func cleanWarmText(value string, maxRunes int) string {
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

func firstNonEmptyWarm(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}
