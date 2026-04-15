package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/Ryvion/node-agent/internal/nodekey"
)

func runIdentity() {
	shortLen := 0
	args := os.Args[2:]
	for i := 0; i < len(args); i++ {
		switch strings.TrimSpace(args[i]) {
		case "--short":
			shortLen = 16
			if i+1 < len(args) {
				if n, err := strconv.Atoi(strings.TrimSpace(args[i+1])); err == nil && n > 0 {
					shortLen = n
					i++
				}
			}
		default:
			fmt.Fprintln(os.Stderr, "Usage: ryvion-node identity [--short [N]]")
			os.Exit(1)
		}
	}

	pubHex, err := nodekey.PublicKeyHex(strings.TrimSpace(os.Getenv("RYV_KEY_PATH")))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: failed to load node identity: %v\n", err)
		os.Exit(1)
	}
	if shortLen > 0 && len(pubHex) > shortLen {
		pubHex = pubHex[:shortLen]
	}
	fmt.Println(pubHex)
}
