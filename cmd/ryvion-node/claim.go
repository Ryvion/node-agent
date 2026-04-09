package main

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/Ryvion/node-agent/internal/hub"
	"github.com/Ryvion/node-agent/internal/nodekey"
)

func runClaim() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "Usage: ryvion-node claim <CODE>")
		fmt.Fprintln(os.Stderr, "  Get a claim code from your Ryvion dashboard → Operator tab → Link a Node")
		os.Exit(1)
	}
	code := strings.TrimSpace(strings.ToUpper(os.Args[2]))
	if code == "" {
		fmt.Fprintln(os.Stderr, "Error: claim code cannot be empty")
		os.Exit(1)
	}

	hubURL := strings.TrimSpace(os.Getenv("RYV_HUB_URL"))
	if hubURL == "" {
		hubURL = "https://api.ryvion.ai"
	}

	pub, priv, err := nodekey.LoadOrCreate(strings.TrimSpace(os.Getenv("RYV_KEY_PATH")))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: failed to load node key: %v\n", err)
		os.Exit(1)
	}

	client := hub.New(hubURL, pub, priv, hub.WithUserAgent("ryvion-node/"+version))

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	fmt.Printf("Linking node to account with code %s...\n", code)
	if err := client.RedeemClaimCode(ctx, code); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("Node linked successfully!")
}
