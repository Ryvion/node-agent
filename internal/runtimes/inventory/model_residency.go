package runtimeinventory

func BuildModelResidencySnapshot(status RuntimeStatus) []ModelResidencySnapshot {
	status = normalizeRuntimeStatus(status)
	if status.NativeModel == "" {
		return []ModelResidencySnapshot{}
	}
	supportsTextGeneration := status.SupportsTextGeneration || status.NativeInferenceReady
	supportsStreaming := status.SupportsStreaming || status.NativeInferenceReady
	return []ModelResidencySnapshot{{
		ModelID:                 status.NativeModel,
		RuntimeKind:             status.RuntimeKind,
		Backend:                 status.Backend,
		Loaded:                  status.ModelLoaded,
		Warm:                    status.ModelLoaded && status.NativeInferenceReady,
		SupportsTextGeneration:  supportsTextGeneration,
		SupportsStreaming:       supportsStreaming,
		SupportsKVAccess:        false,
		SupportsTensorPlane:     false,
		SupportsTensorPlaneDemo: status.SupportsTensorPlaneDemo,
		Reason:                  status.Reason,
	}}
}
