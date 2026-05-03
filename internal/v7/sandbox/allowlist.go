package sandbox

import "strings"

type RunnerAllowlist struct {
	Entries                []RunnerAllowlistEntry `json:"entries,omitempty"`
	RunnerKinds            []RunnerKind           `json:"runner_kinds,omitempty"`
	RunnerImagesOrBinaries []string               `json:"runner_images_or_binaries,omitempty"`
}

type RunnerAllowlistEntry struct {
	RunnerKind          RunnerKind `json:"runner_kind,omitempty"`
	RunnerImageOrBinary string     `json:"runner_image_or_binary,omitempty"`
}

func (allowlist RunnerAllowlist) Allows(request RunnerRequest) bool {
	runnerRef := normalizeRunnerImageOrBinary(request.RunnerImageOrBinary)

	for _, entry := range allowlist.Entries {
		if entry.RunnerKind != "" && entry.RunnerKind != request.RunnerKind {
			continue
		}

		entryRef := normalizeRunnerImageOrBinary(entry.RunnerImageOrBinary)
		if entryRef == "" {
			if entry.RunnerKind != "" && request.RunnerKind != RunnerKindCustom {
				return true
			}
			continue
		}
		if runnerRef != "" && entryRef == runnerRef {
			return true
		}
	}

	for _, kind := range allowlist.RunnerKinds {
		if kind == request.RunnerKind && request.RunnerKind != RunnerKindCustom {
			return true
		}
	}

	for _, allowedRef := range allowlist.RunnerImagesOrBinaries {
		if runnerRef != "" && normalizeRunnerImageOrBinary(allowedRef) == runnerRef {
			return true
		}
	}

	return false
}

func normalizeRunnerImageOrBinary(value string) string {
	value = strings.TrimSpace(value)
	value = strings.Trim(value, `"'`)
	return value
}
