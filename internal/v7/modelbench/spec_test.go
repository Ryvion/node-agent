package modelbench

import (
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"
)

func TestValidateModelBenchmarkSpecValidPasses(t *testing.T) {
	if err := ValidateModelBenchmarkSpec(validModelBenchmarkSpec()); err != nil {
		t.Fatalf("ValidateModelBenchmarkSpec() error = %v", err)
	}
}

func TestValidateModelBenchmarkSpecMissingFieldsRejected(t *testing.T) {
	tests := []struct {
		name      string
		mutate    func(*ModelBenchmarkSpec)
		wantField string
	}{
		{
			name:      "request id",
			mutate:    func(spec *ModelBenchmarkSpec) { spec.RequestID = "" },
			wantField: "request_id",
		},
		{
			name:      "job id",
			mutate:    func(spec *ModelBenchmarkSpec) { spec.JobID = "" },
			wantField: "job_id",
		},
		{
			name:      "model id",
			mutate:    func(spec *ModelBenchmarkSpec) { spec.ModelID = "" },
			wantField: "model_id",
		},
		{
			name:      "prompt hash",
			mutate:    func(spec *ModelBenchmarkSpec) { spec.PromptHash = "" },
			wantField: "prompt_hash",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			spec := validModelBenchmarkSpec()
			test.mutate(&spec)

			err := ValidateModelBenchmarkSpec(spec)
			if !errors.Is(err, ErrInvalidModelBenchmarkSpec) || !strings.Contains(err.Error(), test.wantField) {
				t.Fatalf("ValidateModelBenchmarkSpec() error = %v, want %s invalid", err, test.wantField)
			}
		})
	}
}

func TestValidateModelBenchmarkSpecMaxTokensCapEnforced(t *testing.T) {
	spec := validModelBenchmarkSpec()
	spec.MaxTokens = MaxModelBenchmarkTokens + 1

	err := ValidateModelBenchmarkSpec(spec)
	if !errors.Is(err, ErrInvalidModelBenchmarkSpec) || !strings.Contains(err.Error(), "max_tokens") {
		t.Fatalf("ValidateModelBenchmarkSpec() error = %v, want max_tokens cap", err)
	}
}

func TestValidateModelBenchmarkSpecLongDecodeProfilePasses(t *testing.T) {
	profile, err := GetBenchmarkPromptProfile(string(BenchmarkPromptProfileLongDecodeProbe))
	if err != nil {
		t.Fatalf("GetBenchmarkPromptProfile() error = %v", err)
	}
	spec := validModelBenchmarkSpec()
	spec.PromptProfileID = string(profile.ID)
	spec.PromptLabel = profile.Label
	spec.PromptHash = BenchmarkPromptHash(profile)
	spec.MaxTokens = 128

	if err := ValidateModelBenchmarkSpec(spec); err != nil {
		t.Fatalf("ValidateModelBenchmarkSpec() error = %v", err)
	}
}

func TestValidateModelBenchmarkSpecRejectsUnknownPromptProfile(t *testing.T) {
	spec := validModelBenchmarkSpec()
	spec.PromptProfileID = "missing_probe"

	err := ValidateModelBenchmarkSpec(spec)
	if !errors.Is(err, ErrInvalidModelBenchmarkSpec) || !strings.Contains(err.Error(), "prompt_hash") {
		t.Fatalf("ValidateModelBenchmarkSpec() error = %v, want prompt profile rejection", err)
	}
	if _, resolveErr := ResolveBenchmarkPromptForSpec(spec); !errors.Is(resolveErr, ErrUnknownBenchmarkPromptProfile) {
		t.Fatalf("ResolveBenchmarkPromptForSpec() error = %v, want unknown profile", resolveErr)
	}
}

func TestValidateModelBenchmarkSpecRejectsPromptHashMismatch(t *testing.T) {
	longProfile, err := GetBenchmarkPromptProfile(string(BenchmarkPromptProfileLongDecodeProbe))
	if err != nil {
		t.Fatalf("GetBenchmarkPromptProfile(long) error = %v", err)
	}
	shortProfile, err := GetBenchmarkPromptProfile(string(BenchmarkPromptProfileShortTTFTProbe))
	if err != nil {
		t.Fatalf("GetBenchmarkPromptProfile(short) error = %v", err)
	}
	spec := validModelBenchmarkSpec()
	spec.PromptProfileID = string(longProfile.ID)
	spec.PromptLabel = longProfile.Label
	spec.PromptHash = BenchmarkPromptHash(shortProfile)

	err = ValidateModelBenchmarkSpec(spec)
	if !errors.Is(err, ErrInvalidModelBenchmarkSpec) || !strings.Contains(err.Error(), "prompt_hash") {
		t.Fatalf("ValidateModelBenchmarkSpec() error = %v, want prompt_hash mismatch", err)
	}
}

func TestModelBenchmarkSpecJSONDoesNotExposeRawPromptText(t *testing.T) {
	profile, err := GetBenchmarkPromptProfile(string(BenchmarkPromptProfileLongDecodeProbe))
	if err != nil {
		t.Fatalf("GetBenchmarkPromptProfile() error = %v", err)
	}
	spec := validModelBenchmarkSpec()
	spec.PromptProfileID = string(profile.ID)
	spec.PromptLabel = profile.Label
	spec.PromptHash = BenchmarkPromptHash(profile)

	encoded, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("json.Marshal(spec) error = %v", err)
	}
	text := string(encoded)
	for _, forbidden := range []string{
		"Continue the numbered sequence",
		"Do not stop early",
		"prompt_text",
		"prompt_content",
	} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("spec JSON leaked raw prompt text %q: %s", forbidden, text)
		}
	}
	if !strings.Contains(text, `"prompt_profile_id":"long_decode_probe"`) {
		t.Fatalf("spec JSON missing prompt_profile_id: %s", text)
	}
}

func TestResolveBenchmarkPromptForSpecDefaultsAndLegacyPrompt(t *testing.T) {
	defaultProfile, err := ResolveBenchmarkPromptForSpec(ModelBenchmarkSpec{})
	if err != nil {
		t.Fatalf("ResolveBenchmarkPromptForSpec(empty) error = %v", err)
	}
	if defaultProfile.ID != BenchmarkPromptProfileShortTTFTProbe {
		t.Fatalf("default profile = %q, want %q", defaultProfile.ID, BenchmarkPromptProfileShortTTFTProbe)
	}

	legacyPrompt := DefaultModelBenchmarkPrompt()
	spec := validModelBenchmarkSpec()
	spec.PromptProfileID = ""
	spec.PromptLabel = legacyPrompt.Label
	spec.PromptHash = HashBenchmarkPrompt(legacyPrompt)
	if err := ValidateModelBenchmarkSpec(spec); err != nil {
		t.Fatalf("ValidateModelBenchmarkSpec(legacy default) error = %v", err)
	}
	resolved, err := ResolveBenchmarkPromptForSpec(spec)
	if err != nil {
		t.Fatalf("ResolveBenchmarkPromptForSpec(legacy default) error = %v", err)
	}
	if resolved.Label != legacyPrompt.Label {
		t.Fatalf("resolved legacy label = %q, want %q", resolved.Label, legacyPrompt.Label)
	}
}

func TestValidateModelBenchmarkSpecRejectsNonFiniteTemperature(t *testing.T) {
	for _, value := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		spec := validModelBenchmarkSpec()
		spec.Temperature = value

		err := ValidateModelBenchmarkSpec(spec)
		if !errors.Is(err, ErrInvalidModelBenchmarkSpec) || !strings.Contains(err.Error(), "temperature") {
			t.Fatalf("ValidateModelBenchmarkSpec(%v) error = %v, want temperature finite", value, err)
		}
	}
}

func TestHashBenchmarkPromptDeterministic(t *testing.T) {
	prompt := ModelBenchmarkPrompt{
		Label:   "fixed-readiness-smoke",
		Content: []byte("Return one short readiness token."),
	}

	first := HashBenchmarkPrompt(prompt)
	second := HashBenchmarkPrompt(prompt)
	if first != second {
		t.Fatalf("hashes differ for same prompt: %q vs %q", first, second)
	}
	if !strings.HasPrefix(first, "sha256:") || len(first) != len("sha256:")+64 {
		t.Fatalf("hash = %q, want sha256:<64 hex>", first)
	}

	prompt.Content = []byte("Return a different token.")
	if changed := HashBenchmarkPrompt(prompt); changed == first {
		t.Fatalf("changed prompt produced same hash %q", first)
	}
}

func validModelBenchmarkSpec() ModelBenchmarkSpec {
	profile, err := GetBenchmarkPromptProfile(string(BenchmarkPromptProfileShortTTFTProbe))
	if err != nil {
		panic(err)
	}
	return ModelBenchmarkSpec{
		Task:            ModelBenchmarkTask,
		RequestID:       "request-modelbench-1",
		JobID:           "job-modelbench-1",
		ModelID:         "llama-local-7b-q4",
		PromptProfileID: string(profile.ID),
		PromptLabel:     profile.Label,
		PromptHash:      BenchmarkPromptHash(profile),
		MaxTokens:       64,
		Temperature:     0.2,
		TimeoutMs:       30_000,
		CreatedAtUnixMs: 1_800_000_000_000,
	}
}
