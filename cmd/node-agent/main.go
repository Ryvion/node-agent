package main

import (
    "crypto/ed25519"
    "encoding/hex"
    "flag"
    "log"
    "time"

    "github.com/akatosh/node-agent/internal/agent"
    "github.com/akatosh/node-agent/internal/crypto"
)

func main() {
    hub := flag.String("hub", "http://localhost:8080", "hub orchestrator base URL")
    deviceType := flag.String("type", "gpu", "device type: gpu|cpu|mobile|iot")
    flag.Parse()

    pk, sk := crypto.LoadOrCreateKey()
    log.Printf("node pubkey: %s", hex.EncodeToString(ed25519.PublicKey(pk)))

    a := agent.Agent{
        HubBaseURL: *hub,
        PubKey:     ed25519.PublicKey(pk),
        PrivKey:    ed25519.PrivateKey(sk),
        DeviceType: *deviceType,
    }
    if err := a.Register(); err != nil { log.Fatalf("register failed: %v", err) }
    log.Printf("registered; starting heartbeats and work loop")

    ticker := time.NewTicker(10 * time.Second)
    defer ticker.Stop()
    for {
        if err := a.HeartbeatOnce(); err != nil { log.Printf("heartbeat err: %v", err) }
        if err := a.FetchAndRunWork(); err != nil { log.Printf("work err: %v", err) }
        <-ticker.C
    }
}

