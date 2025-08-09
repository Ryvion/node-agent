Node Agent (Go)

What it does
- Generates/loads an ed25519 key
- Registers with the Hub, sends heartbeats
- Polls for work; simulates execution; submits receipts

Dev
- Build: `go build ./cmd/node-agent`
- Run: `./node-agent -hub http://localhost:8080`

Config
- The agent stores a dev key at `~/.akatosh-node-key`

