package sandbox

import "testing"

func TestRunnerAllowlistAllowsExactRunnerEntry(t *testing.T) {
	allowlist := RunnerAllowlist{
		Entries: []RunnerAllowlistEntry{
			{
				RunnerKind:          RunnerKindCustom,
				RunnerImageOrBinary: "ghcr.io/ryvion/runner-safe:v1",
			},
		},
	}

	request := RunnerRequest{
		RunnerKind:          RunnerKindCustom,
		RunnerImageOrBinary: "ghcr.io/ryvion/runner-safe:v1",
	}

	if !allowlist.Allows(request) {
		t.Fatalf("Allows() = false, want true")
	}
}

func TestRunnerAllowlistRejectsCustomKindOnlyEntry(t *testing.T) {
	allowlist := RunnerAllowlist{
		Entries: []RunnerAllowlistEntry{
			{RunnerKind: RunnerKindCustom},
		},
	}

	request := RunnerRequest{
		RunnerKind:          RunnerKindCustom,
		RunnerImageOrBinary: "ghcr.io/example/custom-runner:v1",
	}

	if allowlist.Allows(request) {
		t.Fatalf("Allows() = true, want false for custom runner without exact image or binary")
	}
}

func TestRunnerAllowlistRejectsEmptyEntry(t *testing.T) {
	allowlist := RunnerAllowlist{
		Entries: []RunnerAllowlistEntry{{}},
	}

	request := RunnerRequest{
		RunnerKind: RunnerKindRyvionRuntime,
	}

	if allowlist.Allows(request) {
		t.Fatalf("Allows() = true, want false for empty entry")
	}
}

func TestRunnerAllowlistAllowsBuiltInKindEntry(t *testing.T) {
	allowlist := RunnerAllowlist{
		Entries: []RunnerAllowlistEntry{
			{RunnerKind: RunnerKindRyvionRuntime},
		},
	}

	request := RunnerRequest{
		RunnerKind: RunnerKindRyvionRuntime,
	}

	if !allowlist.Allows(request) {
		t.Fatalf("Allows() = false, want true")
	}
}
