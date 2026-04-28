package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
)

type operatorPreferences struct {
	PublicAIOptIn         bool   `json:"public_ai_opt_in"`
	PublicAIOptInSet      bool   `json:"-"`
	DeclaredCountry       string `json:"declared_country,omitempty"`
	RuntimeChannel        string `json:"runtime_channel,omitempty"`
	RuntimeChannelVersion string `json:"runtime_channel_version,omitempty"`
	RuntimeProvider       string `json:"runtime_provider,omitempty"`
	RuntimeMode           string `json:"runtime_mode,omitempty"`
	RuntimeSource         string `json:"runtime_source,omitempty"`
	RuntimeArtifact       string `json:"runtime_artifact,omitempty"`
	RuntimeBinary         string `json:"runtime_binary,omitempty"`
	RuntimeBackendBinary  string `json:"runtime_backend_binary,omitempty"`
	RuntimeEngineBinary   string `json:"runtime_engine_binary,omitempty"`
	RuntimeEngineKind     string `json:"runtime_engine_kind,omitempty"`
	RuntimeManifestHash   string `json:"runtime_manifest_hash,omitempty"`
}

type runtimeContractMetadata struct {
	Channel      string
	Version      string
	Provider     string
	Mode         string
	Source       string
	Artifact     string
	Binary       string
	Backend      string
	Engine       string
	EngineKind   string
	ManifestHash string
}

var operatorConfigPathResolver = defaultOperatorConfigPath

func defaultOperatorConfigPath() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, ".ryvion", "config.json"), nil
}

func loadOperatorPreferences() (operatorPreferences, error) {
	path, err := operatorConfigPathResolver()
	if err != nil {
		return operatorPreferences{}, err
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return operatorPreferences{}, nil
		}
		return operatorPreferences{}, err
	}
	var prefs operatorPreferences
	if err := json.Unmarshal(data, &prefs); err != nil {
		return operatorPreferences{}, err
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err == nil {
		_, prefs.PublicAIOptInSet = raw["public_ai_opt_in"]
	}
	return prefs, nil
}

func saveOperatorPreferences(prefs operatorPreferences) error {
	path, err := operatorConfigPathResolver()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(prefs, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	tmp, err := os.CreateTemp(filepath.Dir(path), ".config-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		_ = os.Remove(tmpName)
		return err
	}
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		_ = os.Remove(tmpName)
		return err
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpName)
		return err
	}
	if err := os.Rename(tmpName, path); err != nil {
		_ = os.Remove(path)
		if retryErr := os.Rename(tmpName, path); retryErr != nil {
			_ = os.Remove(tmpName)
			return retryErr
		}
	}
	return nil
}

func mutateOperatorPreferences(mutator func(*operatorPreferences)) (operatorPreferences, error) {
	prefs, err := loadOperatorPreferences()
	if err != nil {
		return operatorPreferences{}, err
	}
	mutator(&prefs)
	if err := saveOperatorPreferences(prefs); err != nil {
		return operatorPreferences{}, err
	}
	return prefs, nil
}

func resolveInitialPublicAIOptIn() (bool, error) {
	if raw := strings.TrimSpace(os.Getenv("RYV_PUBLIC_AI")); raw != "" {
		return parsePublicAIOptIn(raw), nil
	}
	prefs, err := loadOperatorPreferences()
	if err != nil {
		return false, err
	}
	if prefs.PublicAIOptIn {
		return true, nil
	}
	if prefs.PublicAIOptInSet {
		return false, nil
	}
	// Operators who explicitly disable the managed OCI lane are clearly here to
	// run native inference; auto-opt them into the public AI lane so they earn
	// streaming work without a second toggle. Phase 1 friction removal.
	if ociLaneDisabledFromEnv() {
		return true, nil
	}
	// Before operator preferences existed, Ryvion nodes defaulted public AI on
	// unless RYV_PUBLIC_AI=0 was set. Preserve that behavior for existing config
	// files that do not contain an explicit public_ai_opt_in decision; otherwise
	// auto-updates can silently remove working buyer-facing inference capacity.
	return true, nil
}

// ociLaneDisabledFromEnv mirrors the logic in runtime_manager.go without
// importing it here (operator_preferences.go is loaded before the runtime
// manager is constructed).
func ociLaneDisabledFromEnv() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("RYV_DISABLE_OCI"))) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

func resolveInitialDeclaredCountry(flagValue string) (string, error) {
	if raw := strings.TrimSpace(os.Getenv("RYV_DECLARED_COUNTRY")); raw != "" {
		return strings.ToUpper(raw), nil
	}
	if raw := strings.TrimSpace(flagValue); raw != "" {
		return strings.ToUpper(raw), nil
	}
	prefs, err := loadOperatorPreferences()
	if err != nil {
		return "", err
	}
	return strings.ToUpper(strings.TrimSpace(prefs.DeclaredCountry)), nil
}

func parsePublicAIOptIn(raw string) bool {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func resolveRuntimeContractMetadata(defaultVersion string) (runtimeContractMetadata, error) {
	prefs, err := loadOperatorPreferences()
	if err != nil {
		return runtimeContractMetadata{}, err
	}

	meta := runtimeContractMetadata{
		Channel:      strings.TrimSpace(prefs.RuntimeChannel),
		Version:      strings.TrimSpace(prefs.RuntimeChannelVersion),
		Provider:     strings.TrimSpace(prefs.RuntimeProvider),
		Mode:         strings.TrimSpace(prefs.RuntimeMode),
		Source:       strings.TrimSpace(prefs.RuntimeSource),
		Artifact:     strings.TrimSpace(prefs.RuntimeArtifact),
		Binary:       strings.TrimSpace(prefs.RuntimeBinary),
		Backend:      strings.TrimSpace(prefs.RuntimeBackendBinary),
		Engine:       strings.TrimSpace(prefs.RuntimeEngineBinary),
		EngineKind:   strings.TrimSpace(prefs.RuntimeEngineKind),
		ManifestHash: strings.TrimSpace(prefs.RuntimeManifestHash),
	}

	if value := strings.TrimSpace(os.Getenv("RYV_RUNTIME_CHANNEL")); value != "" {
		meta.Channel = value
	}
	if value := strings.TrimSpace(os.Getenv("RYV_RUNTIME_CHANNEL_VERSION")); value != "" {
		meta.Version = value
	}
	if value := strings.TrimSpace(os.Getenv("RYV_RUNTIME_PROVIDER")); value != "" {
		meta.Provider = value
	}
	if value := strings.TrimSpace(os.Getenv("RYV_RUNTIME_MODE")); value != "" {
		meta.Mode = value
	}
	if value := strings.TrimSpace(os.Getenv("RYV_RUNTIME_SOURCE")); value != "" {
		meta.Source = value
	}
	if value := strings.TrimSpace(os.Getenv("RYV_RUNTIME_ARTIFACT")); value != "" {
		meta.Artifact = value
	}
	if value := strings.TrimSpace(os.Getenv("RYV_RUNTIME_BINARY")); value != "" {
		meta.Binary = value
	}
	if value := strings.TrimSpace(os.Getenv("RYV_RUNTIME_BACKEND_BINARY")); value != "" {
		meta.Backend = value
	}
	if value := strings.TrimSpace(os.Getenv("RYV_RUNTIME_ENGINE_BINARY")); value != "" {
		meta.Engine = value
	}
	if value := strings.TrimSpace(os.Getenv("RYV_RUNTIME_ENGINE_KIND")); value != "" {
		meta.EngineKind = value
	}
	if value := strings.TrimSpace(os.Getenv("RYV_RUNTIME_MANIFEST_HASH")); value != "" {
		meta.ManifestHash = value
	}

	if meta.Version == "" {
		meta.Version = strings.TrimSpace(defaultVersion)
	}
	if meta.ManifestHash == "" {
		meta.ManifestHash = computeRuntimeManifestHash(meta)
	}
	return meta, nil
}

func computeRuntimeManifestHash(meta runtimeContractMetadata) string {
	payload := strings.Join([]string{
		strings.TrimSpace(meta.Channel),
		strings.TrimSpace(meta.Version),
		strings.TrimSpace(meta.Provider),
		strings.TrimSpace(meta.Mode),
		strings.TrimSpace(meta.Source),
		strings.TrimSpace(meta.Artifact),
		strings.TrimSpace(meta.Binary),
		strings.TrimSpace(meta.Backend),
	}, "|")
	sum := sha256.Sum256([]byte(payload))
	return hex.EncodeToString(sum[:])
}
