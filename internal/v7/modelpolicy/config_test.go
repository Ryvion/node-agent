package modelpolicy

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestFromConfigSourceParsesEnvPolicy(t *testing.T) {
	t.Parallel()

	env := map[string]string{
		EnvModelAutoDownload:           "1",
		EnvModelMaxSingleGB:            "12",
		EnvModelMaxCacheGB:             "80",
		EnvModelCacheDir:               "~/models",
		EnvModelAllowedFamilies:        "llama, qwen, phi, llama",
		EnvModelAllowedFormats:         "gguf,safetensors",
		EnvModelKeepWarmIDs:            "Llama-3.2-3B-Instruct-Q4_K_M.gguf,phi-4.Q5_K_M.gguf",
		EnvModelEvictionPolicy:         "LRU",
		EnvModelAllowLicenseRestricted: "true",
	}
	policy := FromConfigSource(ConfigSource{
		Getenv: func(name string) string {
			return env[name]
		},
		UserHomeDir: func() (string, error) {
			return "/Users/tester", nil
		},
		GOOS: "darwin",
	})

	if !policy.AutoDownload || !policy.AllowLicenseRestricted {
		t.Fatalf("boolean policy fields = %+v, want true", policy)
	}
	if policy.MaxSingleModelBytes != 12*bytesPerGiB || policy.MaxCacheBytes != 80*bytesPerGiB {
		t.Fatalf("size bytes = %d/%d, want 12GiB/80GiB", policy.MaxSingleModelBytes, policy.MaxCacheBytes)
	}
	if policy.CacheDir != "/Users/tester/models" {
		t.Fatalf("cache_dir = %q, want expanded home dir", policy.CacheDir)
	}
	if got, want := strings.Join(policy.AllowedFamilies, ","), "llama,qwen,phi"; got != want {
		t.Fatalf("allowed_families = %q, want %q", got, want)
	}
	if got, want := strings.Join(policy.AllowedFormats, ","), "gguf,safetensors"; got != want {
		t.Fatalf("allowed_formats = %q, want %q", got, want)
	}
	if got, want := strings.Join(policy.KeepWarmModelIDs, ","), "Llama-3.2-3B-Instruct-Q4_K_M.gguf,phi-4.Q5_K_M.gguf"; got != want {
		t.Fatalf("keep_warm_model_ids = %q, want %q", got, want)
	}
	if policy.EvictionPolicy != DefaultEvictionPolicy {
		t.Fatalf("eviction_policy = %q, want lru", policy.EvictionPolicy)
	}
	if err := ValidatePolicy(policy); err != nil {
		t.Fatalf("ValidatePolicy() error = %v", err)
	}
}

func TestDefaultPolicyValues(t *testing.T) {
	t.Parallel()

	policy := FromConfigSource(ConfigSource{
		Getenv: func(name string) string {
			return ""
		},
		UserHomeDir: func() (string, error) {
			return "/home/operator", nil
		},
		GOOS: "linux",
	})

	if policy.AutoDownload {
		t.Fatalf("auto_download = true, want false")
	}
	if policy.MaxSingleModelBytes != DefaultMaxSingleModelGB*bytesPerGiB {
		t.Fatalf("max_single_model_bytes = %d", policy.MaxSingleModelBytes)
	}
	if policy.MaxCacheBytes != DefaultMaxCacheGB*bytesPerGiB {
		t.Fatalf("max_cache_bytes = %d", policy.MaxCacheBytes)
	}
	if policy.CacheDir != "/home/operator/.ryvion/models" {
		t.Fatalf("cache_dir = %q", policy.CacheDir)
	}
	if got, want := strings.Join(policy.AllowedFamilies, ","), "llama,phi,qwen,gemma"; got != want {
		t.Fatalf("allowed_families = %q, want %q", got, want)
	}
	if got, want := strings.Join(policy.AllowedFormats, ","), "gguf"; got != want {
		t.Fatalf("allowed_formats = %q, want %q", got, want)
	}
	if policy.EvictionPolicy != DefaultEvictionPolicy {
		t.Fatalf("eviction_policy = %q", policy.EvictionPolicy)
	}
	if policy.AllowLicenseRestricted {
		t.Fatalf("allow_license_restricted = true, want false")
	}
}

func TestNormalizePolicyAppliesCaps(t *testing.T) {
	t.Parallel()

	ids := make([]string, 0, 40)
	for i := 0; i < 40; i++ {
		ids = append(ids, string(rune('a'+(i%26)))+string(rune('a'+((i/26)%26)))+strings.Repeat("x", 190)+".gguf")
	}
	policy := NormalizePolicy(Policy{
		MaxSingleModelBytes: uint64(maxPolicySingleSizeGB+100) * bytesPerGiB,
		MaxCacheBytes:       uint64(maxPolicyCacheSizeGB+100) * bytesPerGiB,
		CacheDir:            strings.Repeat("p", 700),
		AllowedFamilies:     []string{"llama", "llama", strings.Repeat("q", 80)},
		AllowedFormats:      []string{"gguf", "gguf"},
		KeepWarmModelIDs:    ids,
		EvictionPolicy:      strings.Repeat("e", 80),
	})

	if policy.MaxSingleModelBytes != uint64(maxPolicySingleSizeGB)*bytesPerGiB {
		t.Fatalf("max_single_model_bytes = %d, cap not applied", policy.MaxSingleModelBytes)
	}
	if policy.MaxCacheBytes != uint64(maxPolicyCacheSizeGB)*bytesPerGiB {
		t.Fatalf("max_cache_bytes = %d, cap not applied", policy.MaxCacheBytes)
	}
	if len(policy.CacheDir) != maxPolicyPathLen {
		t.Fatalf("cache_dir len = %d, want cap %d", len(policy.CacheDir), maxPolicyPathLen)
	}
	if len(policy.KeepWarmModelIDs) != maxPolicyListItems {
		t.Fatalf("keep_warm_model_ids len = %d, want %d", len(policy.KeepWarmModelIDs), maxPolicyListItems)
	}
	for _, id := range policy.KeepWarmModelIDs {
		if len(id) != maxPolicyItemLen {
			t.Fatalf("keep warm id len = %d, want %d", len(id), maxPolicyItemLen)
		}
	}
	if len(policy.EvictionPolicy) != maxPolicyCompactLen {
		t.Fatalf("eviction_policy len = %d, want %d", len(policy.EvictionPolicy), maxPolicyCompactLen)
	}
}

func TestWindowsCacheDirHandling(t *testing.T) {
	t.Parallel()

	policy := FromConfigSource(ConfigSource{
		Getenv: func(name string) string {
			if name == EnvModelCacheDir {
				return `C:\Ryvion\models`
			}
			return ""
		},
		UserHomeDir: func() (string, error) {
			return `C:\Users\operator`, nil
		},
		GOOS: "windows",
	})
	if policy.CacheDir != `C:\Ryvion\models` {
		t.Fatalf("cache_dir = %q, want windows path preserved", policy.CacheDir)
	}

	defaultPolicy := FromConfigSource(ConfigSource{
		Getenv: func(name string) string {
			return ""
		},
		UserHomeDir: func() (string, error) {
			return `C:\Users\operator`, nil
		},
		GOOS: "windows",
	})
	if defaultPolicy.CacheDir != `C:\Users\operator\.ryvion\models` {
		t.Fatalf("default cache_dir = %q", defaultPolicy.CacheDir)
	}
}

func TestPolicyJSONHasNoRawTensorPromptOutputOrSecrets(t *testing.T) {
	t.Parallel()

	raw, err := json.Marshal(FromConfigSource(ConfigSource{
		Getenv: func(name string) string {
			return ""
		},
		UserHomeDir: func() (string, error) {
			return "/home/operator", nil
		},
	}))
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	body := strings.ToLower(string(raw))
	for _, forbidden := range []string{"raw_prompt", "prompt_text", "model_output", "output_text", "generated_text", "key_data", "value_data", "query_vector", "tensor_bytes", "raw_tensor", "secret", "token"} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("policy JSON contains forbidden marker %q: %s", forbidden, raw)
		}
	}
}
