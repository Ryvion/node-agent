package inference

import (
	"encoding/json"
	"testing"
)

func TestToolCallAccumulator_AssemblesFragmentedDeltas(t *testing.T) {
	a := newToolCallAccumulator()
	// First delta carries id + name + the start of the arguments.
	a.add(json.RawMessage(`[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"loc"}}]`))
	// Later deltas stream more argument fragments (no id/name).
	a.add(json.RawMessage(`[{"index":0,"function":{"arguments":"ation\":\"Paris\"}"}}]`))

	if !a.hasAny() {
		t.Fatal("expected accumulated tool calls")
	}
	var calls []struct {
		Index    int    `json:"index"`
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	}
	if err := json.Unmarshal(a.assembled(), &calls); err != nil {
		t.Fatalf("assembled is not valid json: %v", err)
	}
	if len(calls) != 1 {
		t.Fatalf("want 1 call, got %d", len(calls))
	}
	c := calls[0]
	if c.ID != "call_1" || c.Type != "function" || c.Function.Name != "get_weather" {
		t.Errorf("bad call metadata: %+v", c)
	}
	if c.Function.Arguments != `{"location":"Paris"}` {
		t.Errorf("arguments not concatenated across deltas: %q", c.Function.Arguments)
	}
}

func TestToolCallAccumulator_MultipleCallsOrderedByIndex(t *testing.T) {
	a := newToolCallAccumulator()
	a.add(json.RawMessage(`[{"index":1,"id":"b","type":"function","function":{"name":"two","arguments":"{}"}}]`))
	a.add(json.RawMessage(`[{"index":0,"id":"a","type":"function","function":{"name":"one","arguments":"{}"}}]`))
	var calls []struct {
		Index int    `json:"index"`
		ID    string `json:"id"`
	}
	if err := json.Unmarshal(a.assembled(), &calls); err != nil {
		t.Fatalf("invalid json: %v", err)
	}
	if len(calls) != 2 || calls[0].Index != 0 || calls[1].Index != 1 {
		t.Fatalf("calls not ordered by index: %s", a.assembled())
	}
}

func TestToolCallAccumulator_EmptyAndInvalidInputs(t *testing.T) {
	a := newToolCallAccumulator()
	if a.hasAny() || a.assembled() != nil {
		t.Fatal("fresh accumulator should be empty")
	}
	a.add(nil)
	a.add(json.RawMessage(`not-json`))
	if a.hasAny() {
		t.Error("nil/invalid deltas must not register a tool call")
	}
}
