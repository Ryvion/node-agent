package sandbox

type RunnerKind string

const (
	RunnerKindNativeLlama   RunnerKind = "native_llama"
	RunnerKindManagedOCI    RunnerKind = "managed_oci"
	RunnerKindRyvionRuntime RunnerKind = "ryvion_runtime"
	RunnerKindAgentHosting  RunnerKind = "agent_hosting"
	RunnerKindCustom        RunnerKind = "custom"
)

func validRunnerKind(kind RunnerKind) bool {
	switch kind {
	case RunnerKindNativeLlama,
		RunnerKindManagedOCI,
		RunnerKindRyvionRuntime,
		RunnerKindAgentHosting,
		RunnerKindCustom:
		return true
	default:
		return false
	}
}

func runnerKindIn(kind RunnerKind, allowed []RunnerKind) bool {
	for _, candidate := range allowed {
		if kind == candidate {
			return true
		}
	}
	return false
}
