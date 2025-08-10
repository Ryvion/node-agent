package crypto

import (
    "encoding/json"
    "fmt"
    "os"
    solana "github.com/gagliardetto/solana-go"
)

// LoadSolanaKeypair loads a solana PrivateKey from a JSON file (array format like Solana CLI).
func LoadSolanaKeypair(path string) (solana.PrivateKey, error) {
    b, err := os.ReadFile(path)
    if err != nil { return nil, err }
    var arr []byte
    if err := json.Unmarshal(b, &arr); err != nil { return nil, fmt.Errorf("invalid keypair file: %w", err) }
    if len(arr) != 64 { return nil, fmt.Errorf("invalid key length: %d", len(arr)) }
    return solana.PrivateKey(arr), nil
}

