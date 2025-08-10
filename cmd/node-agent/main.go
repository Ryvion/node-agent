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
    referral := flag.String("referral", "", "optional referral code")
    printRef := flag.Bool("print-referral", false, "print a referral code for this node and exit")
    refSrv := flag.Int("referral-server", 0, "start a local referral HTTP server on this port (0=disabled)")
    flag.Parse()

    pk, sk := crypto.LoadOrCreateKey()
    log.Printf("node pubkey: %s", hex.EncodeToString(ed25519.PublicKey(pk)))

    a := agent.Agent{
        HubBaseURL: *hub,
        PubKey:     ed25519.PublicKey(pk),
        PrivKey:    ed25519.PrivateKey(sk),
        DeviceType: *deviceType,
    }
    if *printRef {
        code, err := a.CreateReferral()
        if err != nil { log.Fatalf("referral error: %v", err) }
        log.Printf("referral code: %s", code)
        return
    }
    if err := a.RegisterWithReferral(*referral); err != nil { log.Fatalf("register failed: %v", err) }
    log.Printf("registered; starting heartbeats and work loop")

    if *refSrv > 0 {
        go func() {
            mux := http.NewServeMux()
            mux.HandleFunc("/referral", func(w http.ResponseWriter, r *http.Request) {
                if r.Method != http.MethodGet { w.WriteHeader(405); return }
                code, err := a.CreateReferral()
                if err != nil { w.WriteHeader(500); _, _ = w.Write([]byte(err.Error())); return }
                _, _ = w.Write([]byte(code))
            })
            addr := ":"+itoa(*refSrv)
            log.Printf("referral server on %s", addr)
            _ = http.ListenAndServe(addr, mux)
        }()
    }

    ticker := time.NewTicker(10 * time.Second)
    defer ticker.Stop()
    for {
        if err := a.HeartbeatOnce(); err != nil { log.Printf("heartbeat err: %v", err) }
        if err := a.FetchAndRunWork(); err != nil { log.Printf("work err: %v", err) }
        <-ticker.C
    }
}

func itoa(n int) string { return string([]byte(fmt.Sprintf("%d", n))) }
