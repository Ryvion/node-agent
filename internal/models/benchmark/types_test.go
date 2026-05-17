package modelbench

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestModelBenchmarkResultContainsNoRawPromptOrOutputFields(t *testing.T) {
	for _, typ := range []reflect.Type{
		reflect.TypeOf(ModelBenchmarkResult{}),
		reflect.TypeOf(ModelBenchmarkRuntimeInfo{}),
		reflect.TypeOf(ModelBenchmarkMetrics{}),
	} {
		for i := 0; i < typ.NumField(); i++ {
			field := typ.Field(i)
			name := strings.ToLower(field.Name)
			tag := strings.ToLower(field.Tag.Get("json"))
			for _, forbidden := range []string{
				"rawprompt",
				"prompttext",
				"promptcontent",
				"promptbytes",
				"rawoutput",
				"outputtext",
				"outputcontent",
				"outputdata",
				"transcript",
			} {
				if strings.Contains(name, forbidden) || strings.Contains(tag, forbidden) {
					t.Fatalf("%s.%s exposes forbidden raw field via json tag %q", typ.Name(), field.Name, field.Tag.Get("json"))
				}
			}
		}
	}
}

func TestModelBenchmarkPromptContentOmittedFromJSON(t *testing.T) {
	encoded, err := json.Marshal(ModelBenchmarkPrompt{
		Label:   "fixed-readiness-smoke",
		Content: []byte("raw benchmark prompt content"),
	})
	if err != nil {
		t.Fatalf("json.Marshal(ModelBenchmarkPrompt) error = %v", err)
	}
	if strings.Contains(string(encoded), "raw benchmark prompt content") {
		t.Fatalf("encoded prompt leaked content: %s", encoded)
	}
	if !strings.Contains(string(encoded), "fixed-readiness-smoke") {
		t.Fatalf("encoded prompt missing label: %s", encoded)
	}
}

func TestModelBenchmarkErrorFormatsCodeAndMessage(t *testing.T) {
	err := ModelBenchmarkError{Code: "runtime_not_ready", Message: "native inference is disabled"}
	if got := err.Error(); got != "runtime_not_ready: native inference is disabled" {
		t.Fatalf("Error() = %q", got)
	}
}

func modelBenchHash(value string) string {
	return HashBenchmarkPrompt(ModelBenchmarkPrompt{
		Label:   "fixed-smoke",
		Content: []byte(value),
	})
}
