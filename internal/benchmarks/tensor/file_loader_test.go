package tensorplane

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestTensorPlaneFixtureRoundTripFromJSON(t *testing.T) {
	fixture := testTensorPlaneFixture(t, TensorDTypeFloat32)
	encoded, err := MarshalTensorPlaneFixture(fixture)
	if err != nil {
		t.Fatalf("MarshalTensorPlaneFixture() error = %v", err)
	}
	loaded, err := ParseTensorPlaneFixture(encoded)
	if err != nil {
		t.Fatalf("ParseTensorPlaneFixture() error = %v", err)
	}

	page, query, err := TensorPlaneFixturePageAndQuery(loaded)
	if err != nil {
		t.Fatalf("TensorPlaneFixturePageAndQuery() error = %v", err)
	}
	if err := ValidateTensorPage(page); err != nil {
		t.Fatalf("loaded page invalid: %v", err)
	}
	if err := ValidateAttentionQuery(query, page); err != nil {
		t.Fatalf("loaded query invalid: %v", err)
	}
}

func TestParseTensorPlaneFixtureRejectsCorruptedTensorBytes(t *testing.T) {
	fixture := testTensorPlaneFixture(t, TensorDTypeFloat32)
	fixture.KeyData[0] ^= 0xff
	encoded, err := json.Marshal(fixture)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}

	if _, err := ParseTensorPlaneFixture(encoded); err == nil {
		t.Fatal("ParseTensorPlaneFixture() error = nil, want corrupted key_data rejection")
	}
}

func TestParseTensorPlaneFixtureRejectsWrongShape(t *testing.T) {
	fixture := testTensorPlaneFixture(t, TensorDTypeFloat32)
	fixture.Shape.HeadDim++
	encoded, err := json.Marshal(fixture)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}

	if _, err := ParseTensorPlaneFixture(encoded); err == nil {
		t.Fatal("ParseTensorPlaneFixture() error = nil, want wrong shape rejection")
	}
}

func TestTensorPlaneFixtureHasNoRawPromptOrOutputFields(t *testing.T) {
	fixture := testTensorPlaneFixture(t, TensorDTypeFloat32)
	encoded, err := MarshalTensorPlaneFixture(fixture)
	if err != nil {
		t.Fatalf("MarshalTensorPlaneFixture() error = %v", err)
	}

	assertNoRawPromptOrOutputKeys(t, encoded)
}

func TestParseTensorPlaneFixtureRejectsUnknownRawPromptField(t *testing.T) {
	fixture := testTensorPlaneFixture(t, TensorDTypeFloat32)
	encoded, err := MarshalTensorPlaneFixture(fixture)
	if err != nil {
		t.Fatalf("MarshalTensorPlaneFixture() error = %v", err)
	}
	corrupted := strings.Replace(string(encoded), `{"version":`, `{"prompt":"raw prompt","version":`, 1)

	if _, err := ParseTensorPlaneFixture([]byte(corrupted)); err == nil {
		t.Fatal("ParseTensorPlaneFixture() error = nil, want unknown prompt field rejection")
	}
}

func testTensorPlaneFixture(t *testing.T, dtype TensorDType) TensorPlaneFixture {
	t.Helper()
	fixture, err := BuildTensorPlaneFixture(TensorPlaneFixtureConfig{
		Tokens:   8,
		HeadDim:  4,
		ValueDim: 5,
		DType:    dtype,
		Seed:     99,
	})
	if err != nil {
		t.Fatalf("BuildTensorPlaneFixture() error = %v", err)
	}
	return fixture
}

func assertNoRawPromptOrOutputKeys(t *testing.T, encoded []byte) {
	t.Helper()
	var value any
	if err := json.Unmarshal(encoded, &value); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	var walk func(any)
	walk = func(current any) {
		switch typed := current.(type) {
		case map[string]any:
			for key, child := range typed {
				lower := strings.ToLower(key)
				if strings.Contains(lower, "prompt") || strings.Contains(lower, "output") {
					t.Fatalf("serialized JSON contains raw prompt/output key %q", key)
				}
				walk(child)
			}
		case []any:
			for _, child := range typed {
				walk(child)
			}
		}
	}
	walk(value)
}
