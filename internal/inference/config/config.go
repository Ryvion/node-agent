package inferenceconfig

import "strings"

const (
	EnvDisableV7Inference = "RYV_NODE_DISABLE_V7_INFERENCE"
	EnvDisableTextOutput  = "RYV_NODE_DISABLE_V7_TEXT_OUTPUT"
	EnvDisableStreaming   = "RYV_NODE_DISABLE_V7_STREAMING"
	EnvDisableModelWarm   = "RYV_NODE_DISABLE_MODEL_WARM"
)

func V7InferenceEnabled(getenv func(string) string) bool {
	return !envBool(getenvValue(getenv, EnvDisableV7Inference))
}

func TextOutputEnabled(getenv func(string) string) bool {
	return V7InferenceEnabled(getenv) && !envBool(getenvValue(getenv, EnvDisableTextOutput))
}

func StreamingEnabled(getenv func(string) string) bool {
	return TextOutputEnabled(getenv) && !envBool(getenvValue(getenv, EnvDisableStreaming))
}

func ModelWarmEnabled(getenv func(string) string) bool {
	return !envBool(getenvValue(getenv, EnvDisableModelWarm))
}

func envBool(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "yes", "on", "enabled":
		return true
	default:
		return false
	}
}

func getenvValue(getenv func(string) string, key string) string {
	if getenv == nil {
		return ""
	}
	return getenv(key)
}
