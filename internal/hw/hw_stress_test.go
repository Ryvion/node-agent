package hw

import (
	"fmt"
	"testing"
)

// ---------------------------------------------------------------------------
// 1. TestStress_ParseROCmVRAMFree
// ---------------------------------------------------------------------------

func TestStress_ParseROCmVRAMFree(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		json string
		want uint64
	}{
		// rocm-smi v5+ format
		{
			name: "v5_format_normal",
			json: `{"card0":{"VRAM Total Memory (B)":"25769803776","VRAM Total Used Memory (B)":"1073741824"}}`,
			want: 25769803776 - 1073741824,
		},
		{
			name: "v5_format_zero_used",
			json: `{"card0":{"VRAM Total Memory (B)":"25769803776","VRAM Total Used Memory (B)":"0"}}`,
			want: 25769803776, // used=0 means all VRAM is free
		},
		{
			name: "v5_format_all_free",
			json: `{"card0":{"VRAM Total Memory (B)":"25769803776","VRAM Total Used Memory (B)":"1"}}`,
			want: 25769803776 - 1,
		},
		// Older format
		{
			name: "older_format_normal",
			json: `{"card0":{"vram Total Memory":"25769803776","vram Total Used Memory":"1073741824"}}`,
			want: 25769803776 - 1073741824,
		},
		{
			name: "older_format_large_used",
			json: `{"card0":{"vram Total Memory":"8589934592","vram Total Used Memory":"4294967296"}}`,
			want: 8589934592 - 4294967296,
		},
		// Multiple cards: picks first card with valid data
		{
			name: "multiple_cards_first_valid",
			json: `{"card0":{"VRAM Total Memory (B)":"16000000000","VRAM Total Used Memory (B)":"1000000000"},"card1":{"VRAM Total Memory (B)":"8000000000","VRAM Total Used Memory (B)":"500000000"}}`,
			want: 16000000000 - 1000000000, // or card1 depending on map iteration; both valid
		},
		// Values as float64 (JSON numbers without quotes)
		{
			name: "values_as_numbers",
			json: `{"card0":{"VRAM Total Memory (B)":25769803776,"VRAM Total Used Memory (B)":1073741824}}`,
			want: 25769803776 - 1073741824,
		},
		{
			name: "mixed_string_and_number",
			json: `{"card0":{"VRAM Total Memory (B)":"25769803776","VRAM Total Used Memory (B)":1073741824}}`,
			want: 25769803776 - 1073741824,
		},
		// Empty JSON
		{
			name: "empty_json",
			json: `{}`,
			want: 0,
		},
		// Malformed JSON
		{
			name: "malformed_json",
			json: `{this is not valid json`,
			want: 0,
		},
		{
			name: "empty_string",
			json: ``,
			want: 0,
		},
		// Zero total
		{
			name: "zero_total",
			json: `{"card0":{"VRAM Total Memory (B)":"0","VRAM Total Used Memory (B)":"0"}}`,
			want: 0,
		},
		// Used > Total → return 0
		{
			name: "used_greater_than_total",
			json: `{"card0":{"VRAM Total Memory (B)":"1000","VRAM Total Used Memory (B)":"2000"}}`,
			want: 0,
		},
		{
			name: "used_equals_total",
			json: `{"card0":{"VRAM Total Memory (B)":"5000","VRAM Total Used Memory (B)":"5000"}}`,
			want: 0, // total > used is false when equal
		},
		// Giant values (64GB VRAM)
		{
			name: "giant_64GB",
			json: `{"card0":{"VRAM Total Memory (B)":"68719476736","VRAM Total Used Memory (B)":"1073741824"}}`,
			want: 68719476736 - 1073741824,
		},
		{
			name: "giant_128GB",
			json: `{"card0":{"VRAM Total Memory (B)":"137438953472","VRAM Total Used Memory (B)":"34359738368"}}`,
			want: 137438953472 - 34359738368,
		},
		// Card with no total key
		{
			name: "missing_total_key",
			json: `{"card0":{"VRAM Total Used Memory (B)":"1073741824"}}`,
			want: 0,
		},
		// Card with no used key
		{
			name: "missing_used_key",
			json: `{"card0":{"VRAM Total Memory (B)":"25769803776"}}`,
			want: 0, // used == 0, but v==0 is skipped, so total=25G, used=0 → total > used → returns total - used? No: used is 0 (not set); total>used(0) → returns total
		},
		// Non-numeric values
		{
			name: "non_numeric_total",
			json: `{"card0":{"VRAM Total Memory (B)":"notanumber","VRAM Total Used Memory (B)":"1073741824"}}`,
			want: 0,
		},
		// Whitespace in values
		{
			name: "whitespace_in_values",
			json: `{"card0":{"VRAM Total Memory (B)":" 25769803776 ","VRAM Total Used Memory (B)":" 1073741824 "}}`,
			want: 25769803776 - 1073741824,
		},
		// Nested but wrong structure
		{
			name: "wrong_nesting",
			json: `{"VRAM Total Memory (B)":"25769803776"}`,
			want: 0,
		},
		// Null value
		{
			name: "null_values",
			json: `{"card0":{"VRAM Total Memory (B)":null,"VRAM Total Used Memory (B)":null}}`,
			want: 0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := parseROCmVRAMFree([]byte(tc.json))
			// For multiple cards, map iteration is nondeterministic; accept either card's result
			if tc.name == "multiple_cards_first_valid" {
				alt := uint64(8000000000 - 500000000)
				if got != tc.want && got != alt {
					t.Errorf("parseROCmVRAMFree() = %d, want %d or %d", got, tc.want, alt)
				}
				return
			}
			// missing_used_key: used stays 0 because v==0 is skipped; total=25769803776, used=0 → total>used → returns total
			if tc.name == "missing_used_key" {
				if got != 25769803776 {
					t.Errorf("parseROCmVRAMFree() = %d, want %d", got, uint64(25769803776))
				}
				return
			}
			if got != tc.want {
				t.Errorf("parseROCmVRAMFree() = %d, want %d", got, tc.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// 2. TestStress_ParseROCmVRAMTotal
// ---------------------------------------------------------------------------

func TestStress_ParseROCmVRAMTotal(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		json string
		want uint64
	}{
		{
			name: "v5_format_normal",
			json: `{"card0":{"VRAM Total Memory (B)":"25769803776","VRAM Total Used Memory (B)":"1073741824"}}`,
			want: 25769803776,
		},
		{
			name: "older_format",
			json: `{"card0":{"vram Total Memory":"25769803776","vram Total Used Memory":"1073741824"}}`,
			want: 25769803776,
		},
		{
			name: "values_as_numbers",
			json: `{"card0":{"VRAM Total Memory (B)":25769803776,"VRAM Total Used Memory (B)":1073741824}}`,
			want: 25769803776,
		},
		{
			name: "empty_json",
			json: `{}`,
			want: 0,
		},
		{
			name: "malformed_json",
			json: `not json`,
			want: 0,
		},
		{
			name: "empty_string",
			json: ``,
			want: 0,
		},
		{
			name: "zero_total",
			json: `{"card0":{"VRAM Total Memory (B)":"0","VRAM Total Used Memory (B)":"0"}}`,
			want: 0,
		},
		{
			name: "giant_64GB",
			json: `{"card0":{"VRAM Total Memory (B)":"68719476736","VRAM Total Used Memory (B)":"1073741824"}}`,
			want: 68719476736,
		},
		{
			name: "giant_128GB",
			json: `{"card0":{"VRAM Total Memory (B)":"137438953472","VRAM Total Used Memory (B)":"0"}}`,
			want: 137438953472,
		},
		{
			name: "multiple_cards",
			json: `{"card0":{"VRAM Total Memory (B)":"16000000000","VRAM Total Used Memory (B)":"1000"},"card1":{"VRAM Total Memory (B)":"8000000000","VRAM Total Used Memory (B)":"500"}}`,
			want: 16000000000, // or 8000000000; first valid
		},
		{
			name: "non_numeric_total",
			json: `{"card0":{"VRAM Total Memory (B)":"abc","VRAM Total Used Memory (B)":"1073741824"}}`,
			want: 0,
		},
		{
			name: "whitespace_in_values",
			json: `{"card0":{"VRAM Total Memory (B)":" 25769803776 ","VRAM Total Used Memory (B)":" 1073741824 "}}`,
			want: 25769803776,
		},
		{
			name: "null_value",
			json: `{"card0":{"VRAM Total Memory (B)":null}}`,
			want: 0,
		},
		{
			name: "wrong_structure",
			json: `{"VRAM Total Memory (B)":"25769803776"}`,
			want: 0,
		},
		{
			name: "mixed_string_and_number",
			json: `{"card0":{"VRAM Total Memory (B)":"8589934592","VRAM Total Used Memory (B)":4294967296}}`,
			want: 8589934592,
		},
		{
			name: "only_used_key",
			json: `{"card0":{"VRAM Total Used Memory (B)":"1073741824"}}`,
			want: 0,
		},
		{
			name: "case_insensitive_total",
			json: `{"card0":{"VRAM TOTAL Memory (B)":"12345678"}}`,
			want: 12345678,
		},
		{
			name: "negative_looking_string",
			json: `{"card0":{"VRAM Total Memory (B)":"-1"}}`,
			want: 0, // ParseUint fails on negative
		},
		{
			name: "float_string_value",
			json: `{"card0":{"VRAM Total Memory (B)":"25769803776.5"}}`,
			want: 0, // ParseUint fails on float
		},
		{
			name: "small_total_1MB",
			json: `{"card0":{"VRAM Total Memory (B)":"1048576"}}`,
			want: 1048576,
		},
		{
			name: "array_instead_of_object",
			json: `[{"VRAM Total Memory (B)":"12345"}]`,
			want: 0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := parseROCmVRAMTotal([]byte(tc.json))
			if tc.name == "multiple_cards" {
				alt := uint64(8000000000)
				if got != tc.want && got != alt {
					t.Errorf("parseROCmVRAMTotal() = %d, want %d or %d", got, tc.want, alt)
				}
				return
			}
			if got != tc.want {
				t.Errorf("parseROCmVRAMTotal() = %d, want %d", got, tc.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// 3. TestStress_StringVal
// ---------------------------------------------------------------------------

func TestStress_StringVal(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		val  any
		want string
	}{
		{"string_simple", "hello", "hello"},
		{"string_empty", "", ""},
		{"string_numeric", "12345", "12345"},
		{"string_whitespace", "  spaced  ", "  spaced  "},
		{"float64_zero", float64(0), "0"},
		{"float64_positive", float64(25769803776), "25769803776"},
		{"float64_small", float64(42), "42"},
		{"float64_large", float64(137438953472), "137438953472"},
		{"nil_value", nil, ""},
		{"bool_true", true, ""},
		{"bool_false", false, ""},
		{"int_value", 42, ""},       // only float64 and string handled
		{"int64_value", int64(9), ""}, // not handled
		{"slice_value", []string{"a"}, ""},
		{"map_value", map[string]int{"a": 1}, ""},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := stringVal(tc.val)
			if got != tc.want {
				t.Errorf("stringVal(%v) = %q, want %q", tc.val, got, tc.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// 4. TestStress_ParseWindowsSizedMemory_Exhaustive
// ---------------------------------------------------------------------------

func TestStress_ParseWindowsSizedMemory_Exhaustive(t *testing.T) {
	t.Parallel()

	tests := []struct {
		input string
		want  uint64
	}{
		// Standard cases
		{"24564 MB", 24564 * 1024 * 1024},
		{"24 GB", 24 * 1024 * 1024 * 1024},
		{"3,072 MB", 3072 * 1024 * 1024},
		{"", 0},
		{"n/a", 0},

		// TB
		{"1 TB", 1 * 1024 * 1024 * 1024 * 1024},
		{"0.5 TB", uint64(0.5 * 1024 * 1024 * 1024 * 1024)},
		{"2 TB", 2 * 1024 * 1024 * 1024 * 1024},

		// Large MB
		{"2048 MB", 2048 * 1024 * 1024},
		{"65536 MB", 65536 * 1024 * 1024},

		// KB
		{"1024 KB", 1024 * 1024},
		{"512 KB", 512 * 1024},

		// Raw bytes (no unit suffix)
		{"1073741824", 1073741824},
		{"4294967296", 4294967296},

		// Edge cases
		{"0 GB", 0},           // v <= 0 returns 0
		{"0 MB", 0},           // v <= 0 returns 0
		{"0.0 GB", 0},         // 0.0 <= 0 returns 0
		{"-1 GB", 0},          // negative: v <= 0 returns 0
		{"-500 MB", 0},        // negative

		// Fractional
		{"1.5 GB", uint64(1.5 * 1024 * 1024 * 1024)},
		{"0.25 GB", uint64(0.25 * 1024 * 1024 * 1024)},
		{"10.5 MB", uint64(10.5 * 1024 * 1024)},

		// Case insensitive (parseWindowsSizedMemory uppercases input)
		{"24 gb", 24 * 1024 * 1024 * 1024},
		{"512 mb", 512 * 1024 * 1024},
		{"1 tb", 1 * 1024 * 1024 * 1024 * 1024},

		// Thousands separator variations (commas are stripped)
		{"1,024 MB", 1024 * 1024 * 1024},
		{"16,384 MB", 16384 * 1024 * 1024},
		{"1,048,576 KB", 1048576 * 1024},

		// Suffix-only (no space, unit appended to number)
		{"24GB", 24 * 1024 * 1024 * 1024},
		{"512MB", 512 * 1024 * 1024},
		{"1TB", 1 * 1024 * 1024 * 1024 * 1024},

		// Junk input
		{"not a number", 0},
		{"MB", 0},                // no numeric part after stripping
		{"   ", 0},               // whitespace only
	}

	for _, tc := range tests {
		t.Run(fmt.Sprintf("input=%q", tc.input), func(t *testing.T) {
			t.Parallel()
			got := parseWindowsSizedMemory(tc.input)
			if got != tc.want {
				t.Errorf("parseWindowsSizedMemory(%q) = %d, want %d", tc.input, got, tc.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// 5. TestStress_WindowsGPUTiering
// ---------------------------------------------------------------------------

func TestStress_WindowsGPUTiering(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		gpu  string
		want int
	}{
		// Discrete GPUs → tier 1
		{"nvidia_rtx_4090", "NVIDIA GeForce RTX 4090", 1},
		{"nvidia_rtx_3080", "NVIDIA GeForce RTX 3080", 1},
		{"nvidia_rtx_4070_ti", "NVIDIA GeForce RTX 4070 Ti", 1},
		{"amd_rx_7900_xtx", "AMD Radeon RX 7900 XTX", 1},
		{"amd_rx_6800", "AMD Radeon RX 6800", 1},
		{"nvidia_a100", "NVIDIA A100-SXM4-80GB", 1},
		{"nvidia_quadro", "NVIDIA Quadro RTX 6000", 1},

		// Integrated GPUs → tier 0
		{"intel_uhd_630", "Intel(R) UHD Graphics 630", 0},
		{"intel_uhd_770", "Intel(R) UHD Graphics 770", 0},
		{"intel_hd_graphics", "Intel(R) HD Graphics 530", 0},
		{"intel_iris_xe", "Intel(R) Iris(R) Xe Graphics", 0},
		{"intel_iris_plus", "Intel(R) Iris Plus Graphics", 0},
		{"amd_vega_8", "AMD Radeon Vega 8 Graphics", 0},
		{"amd_vega_3", "AMD Radeon Vega 3 Graphics", 0},
		{"amd_vega_6", "AMD Radeon Vega 6 Graphics", 0},
		{"amd_vega_7", "AMD Radeon Vega 7 Graphics", 0},
		{"amd_vega_10", "AMD Radeon Vega 10 Graphics", 0},
		{"amd_vega_11", "AMD Radeon Vega 11 Graphics", 0},
		{"amd_radeon_tm", "AMD Radeon(TM) Graphics", 0},
		{"amd_radeon_graphics_plain", "Radeon Graphics", 0},

		// Empty → -1
		{"empty_string", "", -1},
		{"whitespace_only", "   ", -1},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := windowsGPUTier(tc.gpu)
			if got != tc.want {
				t.Errorf("windowsGPUTier(%q) = %d, want %d", tc.gpu, got, tc.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// 6. TestStress_IsIgnoredWindowsGPU
// ---------------------------------------------------------------------------

func TestStress_IsIgnoredWindowsGPU(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		gpu     string
		ignored bool
	}{
		// Ignored
		{"microsoft_basic_display", "Microsoft Basic Display Adapter", true},
		{"microsoft_basic_lower", "microsoft basic display adapter", true},
		{"microsoft_basic_mixed", "Microsoft BASIC Display", true},
		{"vmware_svga", "VMware SVGA 3D", true},
		{"vmware_lower", "vmware svga", true},
		{"virtualbox_graphics", "VirtualBox Graphics Adapter", true},
		{"parallels_display", "Parallels Display Adapter", true},
		{"hyper_v_video", "Microsoft Hyper-V Video", true},
		{"remote_desktop_display", "Microsoft Remote Desktop Display", true},

		// Not ignored
		{"nvidia_rtx_4090", "NVIDIA GeForce RTX 4090", false},
		{"amd_rx_7900", "AMD Radeon RX 7900 XTX", false},
		{"intel_uhd", "Intel(R) UHD Graphics 630", false},
		{"nvidia_a100", "NVIDIA A100", false},
		{"empty_string", "", false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := isIgnoredWindowsGPU(tc.gpu)
			if got != tc.ignored {
				t.Errorf("isIgnoredWindowsGPU(%q) = %v, want %v", tc.gpu, got, tc.ignored)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// 7. TestStress_PickBestWindowsGPU
// ---------------------------------------------------------------------------

func TestStress_PickBestWindowsGPU(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		gpus     []windowsGPUInfo
		wantName string
		wantVRAM uint64
	}{
		{
			name: "discrete_over_integrated",
			gpus: []windowsGPUInfo{
				{Name: "Intel(R) UHD Graphics 630", VRAMBytes: 4294967296},
				{Name: "NVIDIA GeForce RTX 4090", VRAMBytes: 25769803776},
			},
			wantName: "NVIDIA GeForce RTX 4090",
			wantVRAM: 25769803776,
		},
		{
			name: "discrete_over_integrated_reversed",
			gpus: []windowsGPUInfo{
				{Name: "NVIDIA GeForce RTX 4090", VRAMBytes: 25769803776},
				{Name: "Intel(R) Iris(R) Xe Graphics", VRAMBytes: 128 * 1024 * 1024},
			},
			wantName: "NVIDIA GeForce RTX 4090",
			wantVRAM: 25769803776,
		},
		{
			name: "multiple_discrete_picks_highest_vram",
			gpus: []windowsGPUInfo{
				{Name: "NVIDIA GeForce RTX 3060", VRAMBytes: 12884901888},
				{Name: "NVIDIA GeForce RTX 4090", VRAMBytes: 25769803776},
			},
			wantName: "NVIDIA GeForce RTX 4090",
			wantVRAM: 25769803776,
		},
		{
			name: "multiple_discrete_equal_vram_picks_first",
			gpus: []windowsGPUInfo{
				{Name: "NVIDIA GeForce RTX 4080", VRAMBytes: 16000000000},
				{Name: "AMD Radeon RX 7800 XT", VRAMBytes: 16000000000},
			},
			wantName: "NVIDIA GeForce RTX 4080",
			wantVRAM: 16000000000,
		},
		{
			name: "all_ignored",
			gpus: []windowsGPUInfo{
				{Name: "Microsoft Basic Display Adapter", VRAMBytes: 0},
				{Name: "VMware SVGA 3D", VRAMBytes: 0},
			},
			wantName: "",
			wantVRAM: 0,
		},
		{
			name:     "empty_list",
			gpus:     []windowsGPUInfo{},
			wantName: "",
			wantVRAM: 0,
		},
		{
			name:     "nil_list",
			gpus:     nil,
			wantName: "",
			wantVRAM: 0,
		},
		{
			name: "only_integrated",
			gpus: []windowsGPUInfo{
				{Name: "Intel(R) UHD Graphics 630", VRAMBytes: 2147483648},
				{Name: "AMD Radeon Vega 8 Graphics", VRAMBytes: 1073741824},
			},
			wantName: "Intel(R) UHD Graphics 630",
			wantVRAM: 2147483648,
		},
		{
			name: "ignored_plus_discrete",
			gpus: []windowsGPUInfo{
				{Name: "Microsoft Basic Display Adapter", VRAMBytes: 0},
				{Name: "NVIDIA GeForce RTX 3080", VRAMBytes: 10737418240},
			},
			wantName: "NVIDIA GeForce RTX 3080",
			wantVRAM: 10737418240,
		},
		{
			name: "ignored_plus_integrated_plus_discrete",
			gpus: []windowsGPUInfo{
				{Name: "VMware SVGA 3D", VRAMBytes: 0},
				{Name: "Intel(R) UHD Graphics 770", VRAMBytes: 2147483648},
				{Name: "AMD Radeon RX 7900 XTX", VRAMBytes: 25769803776},
			},
			wantName: "AMD Radeon RX 7900 XTX",
			wantVRAM: 25769803776,
		},
		{
			name: "empty_name_skipped",
			gpus: []windowsGPUInfo{
				{Name: "", VRAMBytes: 99999999999},
				{Name: "NVIDIA GeForce RTX 4090", VRAMBytes: 25769803776},
			},
			wantName: "NVIDIA GeForce RTX 4090",
			wantVRAM: 25769803776,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := pickBestWindowsGPU(tc.gpus)
			if got.Name != tc.wantName {
				t.Errorf("pickBestWindowsGPU() name = %q, want %q", got.Name, tc.wantName)
			}
			if got.VRAMBytes != tc.wantVRAM {
				t.Errorf("pickBestWindowsGPU() vram = %d, want %d", got.VRAMBytes, tc.wantVRAM)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// 8. TestStress_ParseDarwinCPU_EdgeCases
// ---------------------------------------------------------------------------

func TestStress_ParseDarwinCPU_EdgeCases(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		line     string
		wantUser float64
		wantSys  float64
	}{
		{
			name:     "normal",
			line:     "CPU usage: 5.26% user, 3.94% sys, 90.78% idle",
			wantUser: 5.26,
			wantSys:  3.94,
		},
		{
			name:     "zero_idle",
			line:     "CPU usage: 80.5% user, 19.5% sys, 0.0% idle",
			wantUser: 80.5,
			wantSys:  19.5,
		},
		{
			name:     "high_cpu",
			line:     "CPU usage: 99.9% user, 0.1% sys, 0.0% idle",
			wantUser: 99.9,
			wantSys:  0.1,
		},
		{
			name:     "extra_whitespace",
			line:     "  CPU usage:   12.5% user ,  3.2% sys ,  84.3% idle  ",
			wantUser: 12.5,
			wantSys:  3.2,
		},
		{
			name:     "missing_sys",
			line:     "CPU usage: 100% user",
			wantUser: 100,
			wantSys:  0,
		},
		{
			name:     "missing_user",
			line:     "CPU usage: 50% sys, 50% idle",
			wantUser: 0,
			wantSys:  50,
		},
		{
			name:     "non_standard_format_no_prefix",
			line:     "something else entirely",
			wantUser: 0,
			wantSys:  0,
		},
		{
			name:     "empty_string",
			line:     "",
			wantUser: 0,
			wantSys:  0,
		},
		{
			name:     "prefix_only",
			line:     "CPU usage:",
			wantUser: 0,
			wantSys:  0,
		},
		{
			name:     "zero_user_and_sys",
			line:     "CPU usage: 0.0% user, 0.0% sys, 100.0% idle",
			wantUser: 0.0,
			wantSys:  0.0,
		},
		{
			name:     "integer_values",
			line:     "CPU usage: 10% user, 5% sys, 85% idle",
			wantUser: 10,
			wantSys:  5,
		},
		{
			name:     "prefixed_with_other_text",
			line:     "Processes: 300 total, 5 running. CPU usage: 7.5% user, 2.5% sys, 90.0% idle",
			wantUser: 7.5,
			wantSys:  2.5,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			gotUser, gotSys := parseDarwinCPU(tc.line)
			if gotUser != tc.wantUser {
				t.Errorf("parseDarwinCPU(%q) user = %f, want %f", tc.line, gotUser, tc.wantUser)
			}
			if gotSys != tc.wantSys {
				t.Errorf("parseDarwinCPU(%q) sys = %f, want %f", tc.line, gotSys, tc.wantSys)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// 9. TestStress_ParseVMStatLine_Exhaustive
// ---------------------------------------------------------------------------

func TestStress_ParseVMStatLine_Exhaustive(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		line   string
		prefix string
		want   uint64
	}{
		{
			name:   "normal_with_period",
			line:   "Pages active:                            163235.",
			prefix: "Pages active:",
			want:   163235,
		},
		{
			name:   "large_value_millions",
			line:   "Pages active:                          12345678.",
			prefix: "Pages active:",
			want:   12345678,
		},
		{
			name:   "very_large_value",
			line:   "Pages active:                        9999999999.",
			prefix: "Pages active:",
			want:   9999999999,
		},
		{
			name:   "no_trailing_period",
			line:   "Pages active:                            163235",
			prefix: "Pages active:",
			want:   163235,
		},
		{
			name:   "wrong_prefix",
			line:   "Pages free:                                4599.",
			prefix: "Pages active:",
			want:   0,
		},
		{
			name:   "empty_string",
			line:   "",
			prefix: "Pages active:",
			want:   0,
		},
		{
			name:   "empty_prefix",
			line:   "Pages active:                            163235.",
			prefix: "",
			want:   0, // empty prefix never matches
		},
		{
			name:   "wired_down",
			line:   "Pages wired down:                         80000.",
			prefix: "Pages wired down:",
			want:   80000,
		},
		{
			name:   "compressor",
			line:   "Pages occupied by compressor:              45000.",
			prefix: "Pages occupied by compressor:",
			want:   45000,
		},
		{
			name:   "value_zero",
			line:   "Pages active:                                 0.",
			prefix: "Pages active:",
			want:   0,
		},
		{
			name:   "value_one",
			line:   "Pages active:                                 1.",
			prefix: "Pages active:",
			want:   1,
		},
		{
			name:   "non_numeric_value",
			line:   "Pages active:                            foobar.",
			prefix: "Pages active:",
			want:   0,
		},
		{
			name:   "prefix_only_no_value",
			line:   "Pages active:",
			prefix: "Pages active:",
			want:   0,
		},
		{
			name:   "extra_whitespace_in_value",
			line:   "Pages active:     163235   .",
			prefix: "Pages active:",
			want:   0, // TrimSuffix "." won't catch "   ." and the space makes ParseUint fail
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := parseVMStatLine(tc.line, tc.prefix)
			if got != tc.want {
				t.Errorf("parseVMStatLine(%q, %q) = %d, want %d", tc.line, tc.prefix, got, tc.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// 10. TestStress_SplitQuoted
// ---------------------------------------------------------------------------

func TestStress_SplitQuoted(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		s    string
		want []string
	}{
		{
			name: "normal_quoted",
			s:    `"foo" "bar" "baz"`,
			want: []string{"foo", "bar", "baz"},
		},
		{
			name: "mixed_quoted_unquoted",
			s:    `foo "bar baz" qux`,
			want: []string{"foo", "bar baz", "qux"},
		},
		{
			name: "empty_quotes",
			s:    `"" "test"`,
			want: []string{"test"},
		},
		{
			name: "no_quotes",
			s:    `foo bar baz`,
			want: []string{"foo", "bar", "baz"},
		},
		{
			name: "empty_string",
			s:    ``,
			want: nil,
		},
		{
			name: "only_spaces",
			s:    `   `,
			want: nil,
		},
		{
			name: "single_quoted_word",
			s:    `"hello"`,
			want: []string{"hello"},
		},
		{
			name: "single_unquoted_word",
			s:    `hello`,
			want: []string{"hello"},
		},
		{
			name: "tabs_as_separator",
			s:    "foo\t\"bar baz\"\tqux",
			want: []string{"foo", "bar baz", "qux"},
		},
		{
			name: "multiple_spaces_between",
			s:    `foo    bar    baz`,
			want: []string{"foo", "bar", "baz"},
		},
		{
			name: "lspci_mm_style",
			s:    `00:02.0 "0300" "8086" "Intel Corporation" "UHD Graphics 630"`,
			want: []string{"00:02.0", "0300", "8086", "Intel Corporation", "UHD Graphics 630"},
		},
		{
			name: "quoted_with_numbers",
			s:    `"0302" "Advanced Micro Devices" "Navi 31"`,
			want: []string{"0302", "Advanced Micro Devices", "Navi 31"},
		},
		{
			name: "leading_trailing_spaces",
			s:    `  "hello"  "world"  `,
			want: []string{"hello", "world"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := splitQuoted(tc.s)
			if len(got) != len(tc.want) {
				t.Fatalf("splitQuoted(%q) = %v (len %d), want %v (len %d)", tc.s, got, len(got), tc.want, len(tc.want))
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Errorf("splitQuoted(%q)[%d] = %q, want %q", tc.s, i, got[i], tc.want[i])
				}
			}
		})
	}
}
