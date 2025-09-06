package main

import (
	"bytes"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os/exec"
	"sync/atomic"
	"time"

	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/Ryvion/node-agent/internal/agent"
	"github.com/Ryvion/node-agent/internal/crypto"
	upd "github.com/Ryvion/node-agent/internal/update"
	"github.com/Ryvion/node-agent/pkg/tray"
)

func main() {
	hub := flag.String("hub", "https://ryvion-hub.onrender.com", "hub orchestrator base URL")
	deviceType := flag.String("type", "", "device type: gpu|cpu|mobile|iot (auto-detect if empty)")
	referral := flag.String("referral", "", "optional referral code")
	printRef := flag.Bool("print-referral", false, "print a referral code for this node and exit")
	refSrv := flag.Int("referral-server", 0, "start a local referral HTTP server on this port (0=disabled)")
	uiPort := flag.Int("ui-port", 0, "start a local status UI on this port (0=disabled)")
	flag.Parse()

	cfg := loadConfig()
	if cfg.Hub != "" && *hub == "https://ryvion-hub.onrender.com" {
		*hub = cfg.Hub
	}
	if cfg.DeviceType != "" && *deviceType == "" {
		*deviceType = cfg.DeviceType
	}

	if *deviceType == "" {
		*deviceType = autoDetectDeviceType()
	}
	if cfg.UIPort > 0 && *uiPort == 0 {
		*uiPort = cfg.UIPort
	}

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
		if err != nil {
			log.Fatalf("referral error: %v", err)
		}
		log.Printf("referral code: %s", code)
		return
	}
	if err := a.RegisterWithReferral(*referral); err != nil {
		log.Fatalf("register failed: %v", err)
	}
	log.Printf("registered; starting heartbeats and work loop")

	if os.Getenv("AK_AUTO_UPDATE") == "1" {
		go autoUpdateLoop()
	}

	var paused int32 = 0
	var lastErr string
	var lastBeat int64
	if *uiPort > 0 {
		pubhex := hex.EncodeToString(ed25519.PublicKey(pk))
		go tray.Start(":"+itoa(*uiPort), tray.Controller{Paused: &paused, LastErr: &lastErr, LastBeat: &lastBeat, PubHex: pubhex, SavePayout: func(wallet string) error { return a.SavePayoutWallet(wallet) }, Hub: *hub})
	}

	if *refSrv > 0 {
		go func() {
			mux := http.NewServeMux()
			mux.HandleFunc("/referral", func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodGet {
					w.WriteHeader(405)
					return
				}
				code, err := a.CreateReferral()
				if err != nil {
					w.WriteHeader(500)
					_, _ = w.Write([]byte(err.Error()))
					return
				}
				_, _ = w.Write([]byte(code))
			})
			addr := ":" + itoa(*refSrv)
			log.Printf("referral server on %s", addr)
			_ = http.ListenAndServe(addr, mux)
		}()
	}

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	for {
		if err := a.HeartbeatOnce(); err != nil {
			log.Printf("heartbeat err: %v", err)
			lastErr = err.Error()
		}
		atomic.StoreInt64(&lastBeat, time.Now().UnixMilli())
		if atomic.LoadInt32(&paused) == 0 {
			if err := a.FetchAndRunWork(); err != nil {
				log.Printf("work err: %v", err)
				lastErr = err.Error()
			}
		}
		<-ticker.C
	}
}

type appConfig struct {
	Hub        string `json:"hub"`
	UIPort     int    `json:"ui_port"`
	DeviceType string `json:"device_type"`
}

func loadConfig() appConfig {
	paths := []string{}
	if v := os.Getenv("AK_CONFIG_PATH"); v != "" {
		paths = append(paths, v)
	}
	if ph := os.Getenv("PROGRAMDATA"); ph != "" {
		paths = append(paths, ph+"\\Ryvion\\config.json")
	}
	if home, err := os.UserHomeDir(); err == nil {
		paths = append(paths, home+"/.ryvion/config.json")
	}
	for _, p := range paths {
		b, err := os.ReadFile(p)
		if err == nil && len(b) > 0 {
			var c appConfig
			if json.Unmarshal(b, &c) == nil {
				return c
			}
		}
	}
	return appConfig{}
}

func itoa(n int) string { return fmt.Sprintf("%d", n) }

func autoDetectDeviceType() string {
	if _, err := exec.LookPath("nvidia-smi"); err == nil {
		if out, err := exec.Command("nvidia-smi", "-L").CombinedOutput(); err == nil && len(out) > 0 {
			log.Printf("GPU detected, using device type: gpu")
			return "gpu"
		}
	}

	if _, err := os.Stat("/system/build.prop"); err == nil {
		log.Printf("Android detected, using device type: mobile")
		return "mobile"
	}

	log.Printf("No GPU detected, using device type: cpu")
	return "cpu"
}

func autoUpdateLoop() {
	base := strings.TrimRight(os.Getenv("AK_UPDATE_URL"), "/")
	pub := strings.TrimSpace(os.Getenv("AK_UPDATE_PUBKEY"))
	applyMode := strings.ToLower(strings.TrimSpace(os.Getenv("AK_UPDATE_APPLY")))
	if base == "" || pub == "" {
		log.Printf("auto-update disabled: AK_UPDATE_URL or AK_UPDATE_PUBKEY not set")
		return
	}
	interval := 6 * time.Hour
	if v := strings.TrimSpace(os.Getenv("AK_UPDATE_INTERVAL_MIN")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			interval = time.Duration(n) * time.Minute
		}
	}
	for {
		if err := checkAndStageUpdate(base, pub, applyMode); err != nil {
			log.Printf("update: %v", err)
		}
		t := time.NewTimer(interval)
		<-t.C
	}
}

func checkAndStageUpdate(baseURL, signerPubHex, applyMode string) error {
	manURL := baseURL + "/manifest.json"
	sigURL := baseURL + "/manifest.sig"
	rc, err := upd.Fetch(manURL)
	if err != nil {
		return fmt.Errorf("fetch manifest: %w", err)
	}
	mb, err := io.ReadAll(rc)
	rc.Close()
	if err != nil {
		return fmt.Errorf("read manifest: %w", err)
	}
	src, err := upd.Fetch(sigURL)
	if err != nil {
		return fmt.Errorf("fetch signature: %w", err)
	}
	sigb, err := io.ReadAll(src)
	src.Close()
	if err != nil {
		return fmt.Errorf("read signature: %w", err)
	}
	if err := upd.VerifyDetached(mb, string(bytes.TrimSpace(sigb)), signerPubHex); err != nil {
		return fmt.Errorf("signature verify failed: %w", err)
	}
	m, raw, err := upd.ParseManifest(mb)
	_ = raw
	if err != nil {
		return fmt.Errorf("parse manifest: %w", err)
	}
	goos, goarch := runtime.GOOS, runtime.GOARCH
	asset, ok := upd.SelectAsset(m, "node-agent", goos, goarch)
	if !ok {
		return fmt.Errorf("no asset for %s/%s", goos, goarch)
	}
	self, err := os.Executable()
	if err != nil {
		return fmt.Errorf("executable path: %w", err)
	}
	curHash, err := sha256File(self)
	if err != nil {
		return fmt.Errorf("hash current: %w", err)
	}
	if strings.EqualFold(curHash, asset.SHA256) {
		return nil
	}
	rdr, err := upd.Fetch(asset.URL)
	if err != nil {
		return fmt.Errorf("download: %w", err)
	}
	defer rdr.Close()
	buf := &bytes.Buffer{}
	tee := io.TeeReader(rdr, buf)
	if err := upd.VerifySHA256Hex(tee, asset.SHA256); err != nil {
		return fmt.Errorf("sha256 verify: %w", err)
	}
	dir := filepath.Dir(self)
	stage := filepath.Join(dir, filepath.Base(self)+".next")
	if err := os.WriteFile(stage, buf.Bytes(), 0o755); err != nil {
		return fmt.Errorf("write staged binary: %w", err)
	}
	log.Printf("update: staged new binary at %s (version %s)", stage, m.Version)
	if applyMode == "exit" {
		log.Printf("update: exiting to allow supervisor to swap to .next")
		os.Exit(0)
	}
	return nil
}

func sha256File(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
