package inference

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestWriteHubStreamErrorUsesOpenAIErrorShape(t *testing.T) {
	var out strings.Builder
	writeHubStreamError(&out, `insufficient VRAM: "1202" MB free`)

	line := strings.TrimSpace(out.String())
	if !strings.HasPrefix(line, "data: ") {
		t.Fatalf("stream error line = %q, want data prefix", line)
	}
	var payload struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &payload); err != nil {
		t.Fatalf("stream error JSON invalid: %v; line=%q", err, line)
	}
	if payload.Error.Message != `insufficient VRAM: "1202" MB free` || payload.Error.Type != "node_error" {
		t.Fatalf("payload = %+v", payload.Error)
	}
}
