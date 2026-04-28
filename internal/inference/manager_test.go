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

func TestSupportedNativeChatModelsAllowsDriverReservedVRAMOn16GBCards(t *testing.T) {
	t.Setenv("HF_TOKEN", "test-token")

	reportedVRAM := uint64(17171480576) // RTX 4070 Ti SUPER can report just under exact 16 GiB.
	models := SupportedNativeChatModels(reportedVRAM)
	for _, model := range models {
		if model == "gemma-3-27b-it" {
			return
		}
	}
	t.Fatalf("expected Gemma 27B to be advertised for 16GB-class GPU with %d reported bytes", reportedVRAM)
}

func TestSupportedNativeChatModelsRequiresTokenForGatedGemma(t *testing.T) {
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HUGGINGFACE_TOKEN", "")

	models := SupportedNativeChatModels(16 * 1024 * 1024 * 1024)
	found := false
	for _, model := range models {
		if model == "gemma-3-27b-it" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected gated Gemma model to be advertised through Ryvion platform-managed model downloads")
	}
}

func TestSupportedNativeChatModelsCanDisablePlatformGatedModels(t *testing.T) {
	t.Setenv("HF_TOKEN", "")
	t.Setenv("HUGGINGFACE_TOKEN", "")
	t.Setenv("RYV_DISABLE_PLATFORM_MODEL_DOWNLOADS", "1")

	models := SupportedNativeChatModels(16 * 1024 * 1024 * 1024)
	for _, model := range models {
		if model == "gemma-3-27b-it" {
			t.Fatal("expected gated Gemma model to stay hidden when platform downloads are disabled and no local token is configured")
		}
	}
}
