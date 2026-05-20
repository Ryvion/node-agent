package localcas

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
)

const (
	objectIDPrefix    = "sha256:"
	objectIDHexLength = sha256.Size * 2
)

var ErrInvalidObjectID = errors.New("localcas: invalid object id")

type ObjectID string

func ValidateObjectID(id ObjectID) error {
	value := string(id)
	if len(value) != len(objectIDPrefix)+objectIDHexLength {
		return fmt.Errorf("%w: must be sha256:<64 lowercase hex>", ErrInvalidObjectID)
	}
	if !strings.HasPrefix(value, objectIDPrefix) {
		return fmt.Errorf("%w: missing sha256 prefix", ErrInvalidObjectID)
	}

	hashHex := value[len(objectIDPrefix):]
	for _, ch := range hashHex {
		if (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') {
			continue
		}
		return fmt.Errorf("%w: hash must use lowercase hex", ErrInvalidObjectID)
	}
	return nil
}

func HashBytes(data []byte) ObjectID {
	sum := sha256.Sum256(data)
	return ObjectID(objectIDPrefix + hex.EncodeToString(sum[:]))
}

func objectIDHashHex(id ObjectID) (string, error) {
	if err := ValidateObjectID(id); err != nil {
		return "", err
	}
	return string(id)[len(objectIDPrefix):], nil
}
