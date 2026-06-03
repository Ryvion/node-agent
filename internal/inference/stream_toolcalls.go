package inference

import (
	"encoding/json"
	"sort"
	"strings"
)

// toolCallAccumulator assembles OpenAI streaming tool_call deltas — which arrive
// fragmented (id/name in the first delta for an index, function.arguments
// streamed across many subsequent deltas) — into a complete tool_calls array to
// record in the receipt. This is the node side of end-to-end tool calling: the
// hub reads the assembled array from the receipt metadata and surfaces it as
// structured tool_calls in the /v1/chat response.
type toolCallAccumulator struct {
	byIdx map[int]*accumulatedToolCall
}

type accumulatedToolCall struct {
	id   string
	typ  string
	name string
	args strings.Builder
}

func newToolCallAccumulator() *toolCallAccumulator {
	return &toolCallAccumulator{byIdx: map[int]*accumulatedToolCall{}}
}

// add merges one chunk's `delta.tool_calls` array into the accumulator.
func (a *toolCallAccumulator) add(raw json.RawMessage) {
	if len(raw) == 0 {
		return
	}
	var deltas []struct {
		Index    *int   `json:"index"`
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &deltas); err != nil {
		return
	}
	for _, d := range deltas {
		idx := 0
		if d.Index != nil {
			idx = *d.Index
		}
		tc, ok := a.byIdx[idx]
		if !ok {
			tc = &accumulatedToolCall{}
			a.byIdx[idx] = tc
		}
		if d.ID != "" {
			tc.id = d.ID
		}
		if d.Type != "" {
			tc.typ = d.Type
		}
		if d.Function.Name != "" {
			tc.name = d.Function.Name
		}
		if d.Function.Arguments != "" {
			tc.args.WriteString(d.Function.Arguments)
		}
	}
}

func (a *toolCallAccumulator) hasAny() bool { return len(a.byIdx) > 0 }

// assembled returns the complete tool_calls array as OpenAI-shaped JSON, ordered
// by delta index. Returns nil when no tool calls were seen.
func (a *toolCallAccumulator) assembled() json.RawMessage {
	if !a.hasAny() {
		return nil
	}
	idxs := make([]int, 0, len(a.byIdx))
	for idx := range a.byIdx {
		idxs = append(idxs, idx)
	}
	sort.Ints(idxs)

	type fn struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	}
	type call struct {
		Index    int    `json:"index"`
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function fn     `json:"function"`
	}
	out := make([]call, 0, len(idxs))
	for _, idx := range idxs {
		c := a.byIdx[idx]
		typ := c.typ
		if typ == "" {
			typ = "function"
		}
		out = append(out, call{Index: idx, ID: c.id, Type: typ, Function: fn{Name: c.name, Arguments: c.args.String()}})
	}
	b, err := json.Marshal(out)
	if err != nil {
		return nil
	}
	return b
}
