package modelbench

import (
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
	return ModelBenchmarkSpec{
		Task:            ModelBenchmarkTask,
		RequestID:       "request-modelbench-1",
		JobID:           "job-modelbench-1",
		ModelID:         "llama-local-7b-q4",
		PromptLabel:     "fixed-readiness-smoke",
		PromptHash:      modelBenchHash("Return one short readiness token."),
		MaxTokens:       64,
		Temperature:     0.2,
		TimeoutMs:       30_000,
		CreatedAtUnixMs: 1_800_000_000_000,
	}
}
