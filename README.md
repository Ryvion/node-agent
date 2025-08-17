Node Agent (Go)

What it does
- Generates/loads an ed25519 key
- Registers with the Hub, sends heartbeats
- Polls for work; simulates execution; submits receipts

Dev
- Build: `go build ./cmd/node-agent`
- Run: `./node-agent -hub http://localhost:8080 -ui-port 8090`

Config
- The agent stores a dev key at `~/.akatosh-node-key`
- UI stub at `http://localhost:<ui-port>` supports Pause/Resume and shows last heartbeat/error.
# Test auto-update
