package hub

import (
	"crypto/ed25519"
	"crypto/sha256"
)

func testKeyPair() (ed25519.PublicKey, ed25519.PrivateKey) {
	seed := make([]byte, ed25519.SeedSize)
	for i := range seed {
		seed[i] = byte(i + 1)
	}
	priv := ed25519.NewKeyFromSeed(seed)
	pub := priv.Public().(ed25519.PublicKey)
	return pub, priv
}

func signPayload(parts ...string) []byte {
	joined := "RYV1|"
	for i, p := range parts {
		if i > 0 {
			joined += "|"
		}
		joined += p
	}
	sum := sha256.Sum256([]byte(joined))
	return sum[:]
}
