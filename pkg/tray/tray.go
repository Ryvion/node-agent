package tray

import (
	"embed"
	"encoding/json"
	"html/template"
	"io"
	"log"
	"net/http"
	neturl "net/url"
	"sync/atomic"
	"time"
)

//go:embed web/*.html
var content embed.FS

var indexTemplate *template.Template

func init() {
	var err error
	indexTemplate, err = template.ParseFS(content, "web/index.html")
	if err != nil {
		log.Fatalf("failed to parse index template: %v", err)
	}
}

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

		err := indexTemplate.Execute(w, nil)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			log.Printf("template execution error: %v", err)
		}
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
