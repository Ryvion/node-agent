package tray

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	neturl "net/url"
	"sync/atomic"
	"time"
)

type Controller struct {
	Paused     *int32
	LastErr    *string
	LastBeat   *int64
	PubHex     string
	SavePayout func(wallet string) error
	Hub        string
}

func Start(addr string, ctrl Controller) {
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = fmt.Fprint(w, `<!doctype html><html><head><title>Akatosh Node</title><style>body{font-family:system-ui,Arial;margin:20px}button{padding:8px 12px;margin-right:8px}code{background:#f5f5f5;padding:2px 4px;border-radius:4px}</style></head><body>
        <h3>Akatosh Node</h3>
        <div id="st">Loading…</div>
        <div style="margin-top:8px">
          <button onclick="fetch('/api/pause',{method:'POST'}).then(load)">Pause</button>
          <button onclick="fetch('/api/resume',{method:'POST'}).then(load)">Resume</button>
        </div>
        <h4 style="margin-top:16px">Wallet</h4>
        <div>Node pubkey (hex): <code id="pubhex">...</code> <button onclick="copyPub()">Copy</button></div>
        <div style="margin-top:8px">Payout wallet (Solana base58): <input id="wallet" size="60" placeholder="Your base58 wallet"/> <button onclick="saveWallet()">Save</button></div>
        <small>Use a Solana address to receive payouts when enabled.</small>
        <script>
        async function load(){ const r=await fetch('/api/status'); const j=await r.json(); document.getElementById('st').innerText='Paused: '+j.paused+' | Last heartbeat: '+j.last_heartbeat_ms+' | Last error: '+(j.last_error||''); const rp=await fetch('/api/pubkey'); const p=await rp.json(); document.getElementById('pubhex').innerText=p.pubhex||''; try{ const cw=await fetch('/api/payout/current'); if(cw.ok){ const w=await cw.json(); if(w.wallet){ document.getElementById('wallet').value=w.wallet } } }catch(e){} }
        async function copyPub(){ try{ const rp=await fetch('/api/pubkey'); const p=await rp.json(); await navigator.clipboard.writeText(p.pubhex||''); alert('Copied'); }catch(e){ alert('Copy failed: '+e); } }
        async function saveWallet(){ try{ const w=document.getElementById('wallet').value.trim(); if(!w){ alert('Enter wallet'); return } const res=await fetch('/api/payout',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({wallet:w})}); if(!res.ok){ const t=await res.text(); throw new Error('HTTP '+res.status+' '+t) } alert('Saved payout wallet'); }catch(e){ alert('Save failed: '+e); } }
        load();
        </script>
        </body></html>`)
	})
	mux.HandleFunc("/api/status", func(w http.ResponseWriter, r *http.Request) {
		st := map[string]any{
			"paused":            atomic.LoadInt32(ctrl.Paused) == 1,
			"last_error":        derefString(ctrl.LastErr),
			"last_heartbeat_ms": atomic.LoadInt64(ctrl.LastBeat),
			"ts":                time.Now().UnixMilli(),
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(st)
	})
	mux.HandleFunc("/api/pubkey", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"pubhex": ctrl.PubHex})
	})
	mux.HandleFunc("/api/pause", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(405)
			return
		}
		atomic.StoreInt32(ctrl.Paused, 1)
		w.WriteHeader(204)
	})
	mux.HandleFunc("/api/resume", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(405)
			return
		}
		atomic.StoreInt32(ctrl.Paused, 0)
		w.WriteHeader(204)
	})
	mux.HandleFunc("/api/payout", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(405)
			return
		}
		var req struct {
			Wallet string `json:"wallet"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		if ctrl.SavePayout == nil {
			http.Error(w, "not supported", 400)
			return
		}
		if err := ctrl.SavePayout(req.Wallet); err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": true})
	})
	mux.HandleFunc("/api/payout/current", func(w http.ResponseWriter, r *http.Request) {
		if ctrl.Hub == "" || ctrl.PubHex == "" {
			http.Error(w, "missing hub/pubkey", http.StatusInternalServerError)
			return
		}
		u, _ := neturl.Parse(ctrl.Hub + "/api/v1/node/payout")
		q := u.Query()
		q.Set("pubkey", ctrl.PubHex)
		u.RawQuery = q.Encode()
		resp, err := http.Get(u.String())
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		_, _ = io.Copy(w, resp.Body)
	})
	go func() {
		log.Printf("tray stub UI on %s", addr)
		_ = http.ListenAndServe(addr, mux)
	}()
}

func derefString(p *string) string {
	if p == nil {
		return ""
	}
	return *p
}
