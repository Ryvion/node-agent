package tensorplane

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"
)

const maxTensorPlaneFixtureFileBytes = 256 << 20

func MarshalTensorPlaneFixture(fixture TensorPlaneFixture) ([]byte, error) {
	if err := ValidateTensorPlaneFixture(fixture); err != nil {
		return nil, err
	}
	encoded, err := json.Marshal(fixture)
	if err != nil {
		return nil, err
	}
	return append(encoded, '\n'), nil
}

func ParseTensorPlaneFixture(data []byte) (TensorPlaneFixture, error) {
	if len(bytes.TrimSpace(data)) == 0 {
		return TensorPlaneFixture{}, fmt.Errorf("%w: fixture file is empty", ErrInvalidTensorPlaneFixture)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var fixture TensorPlaneFixture
	if err := decoder.Decode(&fixture); err != nil {
		return TensorPlaneFixture{}, fmt.Errorf("%w: decode fixture JSON: %v", ErrInvalidTensorPlaneFixture, err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return TensorPlaneFixture{}, fmt.Errorf("%w: fixture JSON must contain one object", ErrInvalidTensorPlaneFixture)
	}
	if err := ValidateTensorPlaneFixture(fixture); err != nil {
		return TensorPlaneFixture{}, err
	}
	return fixture, nil
}

func WriteTensorPlaneFixtureFile(path string, fixture TensorPlaneFixture) error {
	path = strings.TrimSpace(path)
	if path == "" {
		return fmt.Errorf("%w: fixture path required", ErrInvalidTensorPlaneFixture)
	}
	encoded, err := MarshalTensorPlaneFixture(fixture)
	if err != nil {
		return err
	}
	return os.WriteFile(path, encoded, 0o600)
}

func LoadTensorPlaneFixtureFile(path string) (TensorPlaneFixture, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return TensorPlaneFixture{}, fmt.Errorf("%w: fixture path required", ErrInvalidTensorPlaneFixture)
	}
	file, err := os.Open(path)
	if err != nil {
		return TensorPlaneFixture{}, err
	}
	defer file.Close()

	limited := io.LimitReader(file, maxTensorPlaneFixtureFileBytes+1)
	data, err := io.ReadAll(limited)
	if err != nil {
		return TensorPlaneFixture{}, err
	}
	if len(data) > maxTensorPlaneFixtureFileBytes {
		return TensorPlaneFixture{}, fmt.Errorf("%w: fixture file exceeds %d bytes", ErrInvalidTensorPlaneFixture, maxTensorPlaneFixtureFileBytes)
	}
	return ParseTensorPlaneFixture(data)
}
