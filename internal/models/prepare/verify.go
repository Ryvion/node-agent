package modelprepare

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
	"strings"
)

func NormalizeSHA256(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	value = strings.TrimPrefix(value, "sha256:")
	if len(value) != sha256.Size*2 {
		return ""
	}
	if _, err := hex.DecodeString(value); err != nil {
		return ""
	}
	return "sha256:" + value
}

func HashFileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, f); err != nil {
		return "", err
	}
	return "sha256:" + hex.EncodeToString(hash.Sum(nil)), nil
}

func VerifyFileSHA256(path string, expected string) (bool, string, error) {
	expected = NormalizeSHA256(expected)
	actual, err := HashFileSHA256(path)
	if err != nil {
		return false, "", err
	}
	if expected == "" {
		return false, actual, nil
	}
	return strings.EqualFold(actual, expected), actual, nil
}
