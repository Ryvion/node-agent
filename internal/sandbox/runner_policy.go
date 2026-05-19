package sandbox

type RunnerKind string

const (
	RunnerKindManagedOCI    RunnerKind = "managed_oci"
	RunnerKindRyvionRuntime RunnerKind = "ryvion_runtime"
	RunnerKindBlender       RunnerKind = "blender"
	RunnerKindMediaTool     RunnerKind = "media_tool"
	RunnerKindCustom        RunnerKind = "custom"
)

func validRunnerKind(kind RunnerKind) bool {
	switch kind {
	case RunnerKindManagedOCI,
		RunnerKindRyvionRuntime,
		RunnerKindBlender,
		RunnerKindMediaTool,
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
