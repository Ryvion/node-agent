package modelbench

import (
	"errors"
	"fmt"
	"strings"
)

type BenchmarkPromptProfileID string

const (
	BenchmarkPromptProfileShortTTFTProbe             BenchmarkPromptProfileID = "short_ttft_probe"
	BenchmarkPromptProfileLongDecodeProbe            BenchmarkPromptProfileID = "long_decode_probe"
	BenchmarkPromptProfileWarmupProbe                BenchmarkPromptProfileID = "warmup_probe"
	BenchmarkPromptProfileDeterministicCountingProbe BenchmarkPromptProfileID = "deterministic_counting_probe"
	BenchmarkPromptProfileDeterministicCounting      BenchmarkPromptProfileID = BenchmarkPromptProfileDeterministicCountingProbe
)

var ErrUnknownBenchmarkPromptProfile = errors.New("modelbench: unknown benchmark prompt profile")

type BenchmarkPromptProfile struct {
	ID               BenchmarkPromptProfileID `json:"profile_id"`
	Label            string                   `json:"prompt_label"`
	Purpose          string                   `json:"purpose"`
	DefaultMaxTokens int                      `json:"default_max_tokens,omitempty"`
}

type BenchmarkPromptDefinition struct {
	BenchmarkPromptProfile
	Prompt ModelBenchmarkPrompt `json:"-"`
}

var benchmarkPromptProfiles = []BenchmarkPromptDefinition{
	{
		BenchmarkPromptProfile: BenchmarkPromptProfile{
			ID:               BenchmarkPromptProfileShortTTFTProbe,
			Label:            string(BenchmarkPromptProfileShortTTFTProbe),
			Purpose:          "measure first-token latency",
			DefaultMaxTokens: 16,
		},
		Prompt: ModelBenchmarkPrompt{
			Label:   string(BenchmarkPromptProfileShortTTFTProbe),
			Content: []byte("Reply with exactly: ready."),
		},
	},
	{
		BenchmarkPromptProfile: BenchmarkPromptProfile{
			ID:               BenchmarkPromptProfileLongDecodeProbe,
			Label:            string(BenchmarkPromptProfileLongDecodeProbe),
			Purpose:          "force longer deterministic generation for decode throughput",
			DefaultMaxTokens: 256,
		},
		Prompt: ModelBenchmarkPrompt{
			Label:   string(BenchmarkPromptProfileLongDecodeProbe),
			Content: []byte("Continue the numbered sequence from 1 to 200, separated by commas. Do not stop early."),
		},
	},
	{
		BenchmarkPromptProfile: BenchmarkPromptProfile{
			ID:               BenchmarkPromptProfileWarmupProbe,
			Label:            string(BenchmarkPromptProfileWarmupProbe),
			Purpose:          "warm local model and runtime before measurement",
			DefaultMaxTokens: 8,
		},
		Prompt: ModelBenchmarkPrompt{
			Label:   string(BenchmarkPromptProfileWarmupProbe),
			Content: []byte("Return the word warm."),
		},
	},
	{
		BenchmarkPromptProfile: BenchmarkPromptProfile{
			ID:               BenchmarkPromptProfileDeterministicCountingProbe,
			Label:            string(BenchmarkPromptProfileDeterministicCountingProbe),
			Purpose:          "produce a stable output shape for repeated benchmark checks",
			DefaultMaxTokens: 96,
		},
		Prompt: ModelBenchmarkPrompt{
			Label:   string(BenchmarkPromptProfileDeterministicCountingProbe),
			Content: []byte("Count from 1 to 50 in order, separated by commas. Stop after 50."),
		},
	},
}

func KnownBenchmarkPromptProfiles() []BenchmarkPromptDefinition {
	out := make([]BenchmarkPromptDefinition, 0, len(benchmarkPromptProfiles))
	for _, profile := range benchmarkPromptProfiles {
		out = append(out, cloneBenchmarkPromptDefinition(profile))
	}
	return out
}

func GetBenchmarkPromptProfile(id string) (BenchmarkPromptDefinition, error) {
	id = strings.TrimSpace(id)
	if id == "" {
		return BenchmarkPromptDefinition{}, fmt.Errorf("%w: profile id required", ErrUnknownBenchmarkPromptProfile)
	}
	for _, profile := range benchmarkPromptProfiles {
		if string(profile.ID) == id {
			return cloneBenchmarkPromptDefinition(profile), nil
		}
	}
	return BenchmarkPromptDefinition{}, fmt.Errorf("%w: %s", ErrUnknownBenchmarkPromptProfile, id)
}

func BenchmarkPromptHash(profile BenchmarkPromptDefinition) string {
	prompt := profile.Prompt
	if strings.TrimSpace(prompt.Label) == "" {
		prompt.Label = profile.Label
	}
	return HashBenchmarkPrompt(prompt)
}

func ResolveBenchmarkPromptForSpec(spec ModelBenchmarkSpec) (BenchmarkPromptDefinition, error) {
	profileID := strings.TrimSpace(spec.PromptProfileID)
	promptLabel := strings.TrimSpace(spec.PromptLabel)
	promptHash := strings.TrimSpace(spec.PromptHash)

	if profileID != "" {
		profile, err := GetBenchmarkPromptProfile(profileID)
		if err != nil {
			return BenchmarkPromptDefinition{}, err
		}
		if err := matchBenchmarkPromptSpec(profile, promptLabel, promptHash); err != nil {
			return BenchmarkPromptDefinition{}, err
		}
		return profile, nil
	}

	if promptHash != "" {
		for _, profile := range benchmarkPromptProfiles {
			profile = cloneBenchmarkPromptDefinition(profile)
			if promptLabel != "" && promptLabel != profile.Label {
				continue
			}
			if promptHash == BenchmarkPromptHash(profile) {
				return profile, nil
			}
		}
		legacy := legacyBenchmarkPromptDefinition()
		if (promptLabel == "" || promptLabel == legacy.Label) && promptHash == BenchmarkPromptHash(legacy) {
			return legacy, nil
		}
		return BenchmarkPromptDefinition{}, fmt.Errorf("%w: prompt_hash does not match a known benchmark prompt profile", ErrInvalidModelBenchmarkSpec)
	}

	if promptLabel != "" {
		for _, profile := range benchmarkPromptProfiles {
			profile = cloneBenchmarkPromptDefinition(profile)
			if promptLabel == profile.Label {
				return profile, nil
			}
		}
		legacy := legacyBenchmarkPromptDefinition()
		if promptLabel == legacy.Label {
			return legacy, nil
		}
		return BenchmarkPromptDefinition{}, fmt.Errorf("%w: prompt_label does not match a known benchmark prompt profile", ErrInvalidModelBenchmarkSpec)
	}

	profile, _ := GetBenchmarkPromptProfile(string(BenchmarkPromptProfileShortTTFTProbe))
	return profile, nil
}

func matchBenchmarkPromptSpec(profile BenchmarkPromptDefinition, promptLabel string, promptHash string) error {
	if promptLabel != "" && promptLabel != profile.Label {
		return fmt.Errorf("%w: prompt_label %q does not match profile %q", ErrInvalidModelBenchmarkSpec, promptLabel, profile.ID)
	}
	if promptHash != "" && promptHash != BenchmarkPromptHash(profile) {
		return fmt.Errorf("%w: prompt_hash does not match profile %q", ErrInvalidModelBenchmarkSpec, profile.ID)
	}
	return nil
}

func benchmarkPromptBindingValid(spec ModelBenchmarkSpec) bool {
	if strings.TrimSpace(spec.PromptHash) == "" {
		return true
	}
	if strings.TrimSpace(spec.PromptProfileID) == "" && !knownBenchmarkPromptLabel(spec.PromptLabel) {
		return true
	}
	_, err := ResolveBenchmarkPromptForSpec(spec)
	return err == nil
}

func knownBenchmarkPromptLabel(label string) bool {
	label = strings.TrimSpace(label)
	if label == "" {
		return false
	}
	for _, profile := range benchmarkPromptProfiles {
		if label == profile.Label {
			return true
		}
	}
	return label == legacyBenchmarkPromptDefinition().Label
}

func legacyBenchmarkPromptDefinition() BenchmarkPromptDefinition {
	prompt := DefaultModelBenchmarkPrompt()
	return BenchmarkPromptDefinition{
		BenchmarkPromptProfile: BenchmarkPromptProfile{
			Label:            prompt.Label,
			Purpose:          "legacy native model benchmark self-test prompt",
			DefaultMaxTokens: defaultModelBenchmarkSelfTestMaxTokens,
		},
		Prompt: prompt,
	}
}

func cloneBenchmarkPromptDefinition(profile BenchmarkPromptDefinition) BenchmarkPromptDefinition {
	out := profile
	out.Prompt.Content = append([]byte(nil), profile.Prompt.Content...)
	return out
}
