package inference

import "testing"

func TestSupportedNativeChatModelsGatesGemmaByVRAM(t *testing.T) {
	t.Setenv("HF_TOKEN", "test-token")

	lowVRAM := SupportedNativeChatModels(12 * 1024 * 1024 * 1024)
	for _, model := range lowVRAM {
		if model == "gemma-3-27b-it" {
			t.Fatal("expected Gemma 27B to be hidden below 16GB VRAM")
		}
	}

	enoughVRAM := SupportedNativeChatModels(16 * 1024 * 1024 * 1024)
	found := false
	for _, model := range enoughVRAM {
		if model == "gemma-3-27b-it" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected Gemma 27B to be advertised on 16GB VRAM nodes")
	}
}

func TestSupportedNativeChatModelsRequiresTokenForGatedGemma(t *testing.T) {
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HUGGINGFACE_TOKEN", "")

	models := SupportedNativeChatModels(16 * 1024 * 1024 * 1024)
	for _, model := range models {
		if model == "gemma-3-27b-it" {
			t.Fatal("expected gated Gemma model to stay hidden without a Hugging Face token")
		}
	}
}
