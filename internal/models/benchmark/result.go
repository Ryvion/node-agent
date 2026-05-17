package modelbench

import "strings"

type ModelBenchmarkResult struct {
	RequestID   string                    `json:"request_id"`
	JobID       string                    `json:"job_id"`
	NodeID      string                    `json:"node_id,omitempty"`
	ModelID     string                    `json:"model_id"`
	PromptHash  string                    `json:"prompt_hash"`
	RuntimeInfo ModelBenchmarkRuntimeInfo `json:"runtime_info"`
	Metrics     ModelBenchmarkMetrics     `json:"metrics"`
	OutputHash  string                    `json:"output_hash,omitempty"`
	OutputBytes int64                     `json:"output_bytes"`
	ProofStatus ModelBenchmarkProofStatus `json:"proof_status"`
}

func normalizeModelBenchmarkResult(result ModelBenchmarkResult) ModelBenchmarkResult {
	result.RequestID = strings.TrimSpace(result.RequestID)
	result.JobID = strings.TrimSpace(result.JobID)
	result.NodeID = strings.TrimSpace(result.NodeID)
	result.ModelID = strings.TrimSpace(result.ModelID)
	result.PromptHash = strings.TrimSpace(result.PromptHash)
	result.OutputHash = strings.TrimSpace(result.OutputHash)
	result.RuntimeInfo = normalizeModelBenchmarkRuntimeInfo(result.RuntimeInfo)
	result.Metrics = normalizeModelBenchmarkMetrics(result.Metrics)
	return result
}

func normalizeModelBenchmarkRuntimeInfo(info ModelBenchmarkRuntimeInfo) ModelBenchmarkRuntimeInfo {
	info.AgentVersion = strings.TrimSpace(info.AgentVersion)
	info.OS = strings.TrimSpace(info.OS)
	info.Arch = strings.TrimSpace(info.Arch)
	info.RuntimeKind = strings.TrimSpace(info.RuntimeKind)
	info.ModelID = strings.TrimSpace(info.ModelID)
	info.GPUModel = strings.TrimSpace(info.GPUModel)
	return info
}

func normalizeModelBenchmarkMetrics(metrics ModelBenchmarkMetrics) ModelBenchmarkMetrics {
	metrics.ModelLoadState = ModelBenchmarkModelLoadState(strings.TrimSpace(string(metrics.ModelLoadState)))
	metrics.ErrorCode = strings.TrimSpace(metrics.ErrorCode)
	return metrics
}
