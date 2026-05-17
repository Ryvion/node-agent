package modelbench

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
)

func TestKnownBenchmarkPromptProfilesDeterministic(t *testing.T) {
	first := KnownBenchmarkPromptProfiles()
	second := KnownBenchmarkPromptProfiles()

	wantIDs := []BenchmarkPromptProfileID{
		BenchmarkPromptProfileShortTTFTProbe,
		BenchmarkPromptProfileLongDecodeProbe,
		BenchmarkPromptProfileWarmupProbe,
		BenchmarkPromptProfileDeterministicCounting,
	}
	if len(first) != len(wantIDs) {
		t.Fatalf("len(KnownBenchmarkPromptProfiles()) = %d, want %d", len(first), len(wantIDs))
	}
	var firstIDs []BenchmarkPromptProfileID
	var secondIDs []BenchmarkPromptProfileID
	for i := range first {
		firstIDs = append(firstIDs, first[i].ID)
		secondIDs = append(secondIDs, second[i].ID)
		if first[i].ID != wantIDs[i] {
			t.Fatalf("profile[%d].ID = %q, want %q", i, first[i].ID, wantIDs[i])
		}
	}
	if !reflect.DeepEqual(firstIDs, secondIDs) {
		t.Fatalf("profile IDs changed between calls: %v vs %v", firstIDs, secondIDs)
	}
}

func TestBenchmarkPromptProfilesHaveLabelsAndHashes(t *testing.T) {
	for _, profile := range KnownBenchmarkPromptProfiles() {
		if strings.TrimSpace(profile.Label) == "" {
			t.Fatalf("profile %q has empty label", profile.ID)
		}
		if strings.TrimSpace(profile.Purpose) == "" {
			t.Fatalf("profile %q has empty purpose", profile.ID)
		}
		hash := BenchmarkPromptHash(profile)
		if !strings.HasPrefix(hash, "sha256:") || len(hash) != len("sha256:")+64 {
			t.Fatalf("profile %q hash = %q, want sha256:<64 hex>", profile.ID, hash)
		}
	}
}

func TestBenchmarkPromptHashDeterministicAndMaterialSensitive(t *testing.T) {
	profile, err := GetBenchmarkPromptProfile(string(BenchmarkPromptProfileLongDecodeProbe))
	if err != nil {
		t.Fatalf("GetBenchmarkPromptProfile() error = %v", err)
	}
	first := BenchmarkPromptHash(profile)
	second := BenchmarkPromptHash(profile)
	if first != second {
		t.Fatalf("hashes differ for same profile: %q vs %q", first, second)
	}

	profile.Prompt.Content = []byte("Different benchmark-only prompt material.")
	if changed := BenchmarkPromptHash(profile); changed == first {
		t.Fatalf("changed prompt material produced same hash %q", first)
	}
}

func TestGetBenchmarkPromptProfileRejectsUnknown(t *testing.T) {
	_, err := GetBenchmarkPromptProfile("missing_probe")
	if !errors.Is(err, ErrUnknownBenchmarkPromptProfile) {
		t.Fatalf("GetBenchmarkPromptProfile() error = %v, want unknown profile", err)
	}
}

func TestBenchmarkPromptDefinitionJSONDoesNotExposeRawPrompt(t *testing.T) {
	profile, err := GetBenchmarkPromptProfile(string(BenchmarkPromptProfileLongDecodeProbe))
	if err != nil {
		t.Fatalf("GetBenchmarkPromptProfile() error = %v", err)
	}
	encoded, err := json.Marshal(profile)
	if err != nil {
		t.Fatalf("json.Marshal(profile) error = %v", err)
	}
	text := string(encoded)
	for _, forbidden := range []string{
		"Continue the numbered sequence",
		"Do not stop early",
		"prompt_text",
		"prompt_content",
	} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("profile JSON leaked raw prompt text %q: %s", forbidden, text)
		}
	}
	if !strings.Contains(text, string(BenchmarkPromptProfileLongDecodeProbe)) {
		t.Fatalf("profile JSON missing label/id: %s", text)
	}
}
