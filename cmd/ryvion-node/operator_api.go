package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Ryvion/node-agent/internal/hub"
	"github.com/Ryvion/node-agent/internal/hw"
	"github.com/Ryvion/node-agent/internal/inference"
)

const defaultOperatorAPIPort = "45890"

var (
	operatorRuntimeState *operatorRuntime
	operatorLogBuffer    = newLogRing(400)
)

type operatorRuntime struct {
	mu sync.RWMutex

	version         string
	hubURL          string
	deviceType      string
	declaredCountry string
	publicKeyHex    string
	caps            hw.CapSet
	client          *hub.Client

	registered        bool
	lastRegisterError string
	lastHeartbeatAt   time.Time
	lastHeartbeatErr  string
	latestVersion     string
	lastMetrics       hw.Metrics
	lastHealthReport  hub.HealthReport
	lastClaimAt       time.Time
	lastClaimError    string
	lastPayoutAt      time.Time
	lastPayoutError   string
	currentJob        *operatorJob
	recentJobs        []operatorJob
	infMgr            *inference.Manager
}

type operatorJob struct {
	JobID          string         `json:"job_id"`
	Kind           string         `json:"kind"`
	Image          string         `json:"image,omitempty"`
	Status         string         `json:"status"`
	StartedAt      time.Time      `json:"started_at"`
	CompletedAt    time.Time      `json:"completed_at,omitempty"`
	DurationMs     int64          `json:"duration_ms,omitempty"`
	ResultHashHex  string         `json:"result_hash_hex,omitempty"`
	BlobURL        string         `json:"blob_url,omitempty"`
	ExitCode       int            `json:"exit_code,omitempty"`
	Error          string         `json:"error,omitempty"`
	MeteringUnits  uint64         `json:"metering_units,omitempty"`
	ReceiptMeta    map[string]any `json:"receipt_metadata,omitempty"`
	DeliveryObject string         `json:"delivery_object,omitempty"`
}

type operatorStatusResponse struct {
	Version         string             `json:"version"`
	HubURL          string             `json:"hub_url"`
	PublicKeyHex    string             `json:"public_key_hex"`
	DeviceType      string             `json:"device_type"`
	DeclaredCountry string             `json:"declared_country,omitempty"`
	Registered      bool               `json:"registered"`
	RegisterError   string             `json:"register_error,omitempty"`
	LatestVersion   string             `json:"latest_version,omitempty"`
	LastHeartbeatAt time.Time          `json:"last_heartbeat_at,omitempty"`
	LastHeartbeatErr string            `json:"last_heartbeat_error,omitempty"`
	Machine         operatorMachine    `json:"machine"`
	Runtime         operatorRuntimeInfo `json:"runtime"`
	Metrics         hw.Metrics         `json:"metrics"`
	CurrentJob      *operatorJob       `json:"current_job,omitempty"`
	RecentJobs      []operatorJob      `json:"recent_jobs"`
	LastClaimAt     time.Time          `json:"last_claim_at,omitempty"`
	LastClaimError  string             `json:"last_claim_error,omitempty"`
	LastPayoutAt    time.Time          `json:"last_payout_at,omitempty"`
	LastPayoutError string             `json:"last_payout_error,omitempty"`
}

type operatorMachine struct {
	CPUCores  uint32 `json:"cpu_cores"`
	RAMBytes  uint64 `json:"ram_bytes"`
	GPUModel  string `json:"gpu_model,omitempty"`
	VRAMBytes uint64 `json:"vram_bytes,omitempty"`
}

type operatorRuntimeInfo struct {
	LocalAPIURL                string `json:"local_api_url"`
	StatusMessage              string `json:"status_message,omitempty"`
	DockerCLIPresent           bool   `json:"docker_cli_present"`
	DockerReady                bool   `json:"docker_ready"`
	DockerGPUEnabled           bool   `json:"docker_gpu_enabled"`
	GPUReady                   bool   `json:"gpu_ready"`
	SpatialReady               bool   `json:"spatial_ready"`
	NativeInferenceSupported   bool   `json:"native_inference_supported"`
	NativeInferenceReady       bool   `json:"native_inference_ready"`
	NativeModel                string `json:"native_model,omitempty"`
	DiskGB                     uint64 `json:"disk_gb,omitempty"`
}

type logRing struct {
	mu      sync.Mutex
	pending bytes.Buffer
	lines   []string
	limit   int
}

func newOperatorRuntime(version, hubURL, deviceType, declaredCountry string, caps hw.CapSet, client *hub.Client) *operatorRuntime {
	return &operatorRuntime{
		version:         strings.TrimSpace(version),
		hubURL:          strings.TrimSpace(hubURL),
		deviceType:      strings.TrimSpace(deviceType),
		declaredCountry: strings.ToUpper(strings.TrimSpace(declaredCountry)),
		publicKeyHex:    client.PublicKeyHex(),
		caps:            caps,
		client:          client,
		recentJobs:      make([]operatorJob, 0, 20),
	}
}

func newLogRing(limit int) *logRing {
	if limit <= 0 {
		limit = 200
	}
	return &logRing{limit: limit}
}

func (l *logRing) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()

	n, err := l.pending.Write(p)
	if err != nil {
		return n, err
	}
	for {
		data := l.pending.Bytes()
		idx := bytes.IndexByte(data, '\n')
		if idx < 0 {
			break
		}
		line := strings.TrimSpace(string(data[:idx]))
		l.appendLine(line)
		l.pending.Next(idx + 1)
	}
	return len(p), nil
}

func (l *logRing) appendLine(line string) {
	if line == "" {
		return
	}
	l.lines = append(l.lines, line)
	if len(l.lines) > l.limit {
		l.lines = append([]string(nil), l.lines[len(l.lines)-l.limit:]...)
	}
}

func (l *logRing) tail(limit int) []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	if limit <= 0 || limit > len(l.lines) {
		limit = len(l.lines)
	}
	out := make([]string, limit)
	copy(out, l.lines[len(l.lines)-limit:])
	return out
}

func (s *operatorRuntime) setInferenceManager(infMgr *inference.Manager) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.infMgr = infMgr
}

func (s *operatorRuntime) setRegistered(ok bool, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.registered = ok
	if err != nil {
		s.lastRegisterError = strings.TrimSpace(err.Error())
		return
	}
	s.lastRegisterError = ""
}

func (s *operatorRuntime) recordHeartbeat(metrics hw.Metrics, latestVersion string, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastMetrics = metrics
	if err != nil {
		s.lastHeartbeatErr = strings.TrimSpace(err.Error())
		return
	}
	s.lastHeartbeatAt = time.Now()
	s.lastHeartbeatErr = ""
	if latestVersion != "" {
		s.latestVersion = latestVersion
	}
}

func (s *operatorRuntime) recordHealthReport(report hub.HealthReport) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastHealthReport = report
}

func (s *operatorRuntime) startJob(work *hub.WorkAssignment) {
	if work == nil {
		return
	}
	job := operatorJob{
		JobID:     strings.TrimSpace(work.JobID),
		Kind:      strings.TrimSpace(work.Kind),
		Image:     strings.TrimSpace(work.Image),
		Status:    "running",
		StartedAt: time.Now(),
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.currentJob = &job
}

func (s *operatorRuntime) finishJob(work *hub.WorkAssignment, result *runnerResultSnapshot, runErr error) {
	if work == nil {
		return
	}
	now := time.Now()
	s.mu.Lock()
	defer s.mu.Unlock()

	job := operatorJob{
		JobID:       strings.TrimSpace(work.JobID),
		Kind:        strings.TrimSpace(work.Kind),
		Image:       strings.TrimSpace(work.Image),
		Status:      "completed",
		StartedAt:   now,
		CompletedAt: now,
	}
	if s.currentJob != nil && s.currentJob.JobID == job.JobID {
		job.StartedAt = s.currentJob.StartedAt
		if !job.StartedAt.IsZero() {
			job.DurationMs = now.Sub(job.StartedAt).Milliseconds()
		}
	}
	if runErr != nil {
		job.Status = "failed"
		job.Error = strings.TrimSpace(runErr.Error())
	}
	if result != nil {
		if result.DurationMs > 0 {
			job.DurationMs = result.DurationMs
		}
		job.ResultHashHex = result.ResultHashHex
		job.ExitCode = result.ExitCode
		job.MeteringUnits = result.MeteringUnits
		job.BlobURL = result.BlobURL
		job.DeliveryObject = result.ObjectKey
		job.ReceiptMeta = result.Metadata
		if result.ExitCode != 0 && job.Status == "completed" {
			job.Status = "failed"
		}
	}
	s.currentJob = nil
	s.recentJobs = append([]operatorJob{job}, s.recentJobs...)
	if len(s.recentJobs) > 20 {
		s.recentJobs = append([]operatorJob(nil), s.recentJobs[:20]...)
	}
}

func (s *operatorRuntime) recordClaim(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastClaimAt = time.Now()
	if err != nil {
		s.lastClaimError = strings.TrimSpace(err.Error())
		return
	}
	s.lastClaimError = ""
}

func (s *operatorRuntime) recordPayout(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastPayoutAt = time.Now()
	if err != nil {
		s.lastPayoutError = strings.TrimSpace(err.Error())
		return
	}
	s.lastPayoutError = ""
}

func (s *operatorRuntime) recentJobsSnapshot() ([]operatorJob, *operatorJob) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	var current *operatorJob
	if s.currentJob != nil {
		cp := *s.currentJob
		current = &cp
	}
	out := make([]operatorJob, len(s.recentJobs))
	copy(out, s.recentJobs)
	return out, current
}

func (s *operatorRuntime) statusSnapshot(apiPort string) operatorStatusResponse {
	s.mu.RLock()
	caps := s.caps
	metrics := s.lastMetrics
	registered := s.registered
	registerErr := s.lastRegisterError
	latestVersion := s.latestVersion
	lastHeartbeatAt := s.lastHeartbeatAt
	lastHeartbeatErr := s.lastHeartbeatErr
	lastClaimAt := s.lastClaimAt
	lastClaimErr := s.lastClaimError
	lastPayoutAt := s.lastPayoutAt
	lastPayoutErr := s.lastPayoutError
	current := s.currentJob
	recent := make([]operatorJob, len(s.recentJobs))
	copy(recent, s.recentJobs)
	report := s.lastHealthReport
	infMgr := s.infMgr
	s.mu.RUnlock()

	var currentJob *operatorJob
	if current != nil {
		cp := *current
		currentJob = &cp
	}

	runtimeInfo := operatorRuntimeInfo{
		LocalAPIURL: fmt.Sprintf("http://127.0.0.1:%s", apiPort),
		StatusMessage: report.Message,
		DockerCLIPresent:         statusToken(report.Message, "docker-cli:present"),
		DockerReady:              statusToken(report.Message, "docker-ready:1"),
		DockerGPUEnabled:         statusToken(report.Message, "docker-gpu:ok"),
		GPUReady:                 report.GPUReady,
		SpatialReady:             statusToken(report.Message, "spatial-ready:1"),
		NativeInferenceSupported: statusToken(report.Message, "native-inference:supported"),
		NativeInferenceReady:     statusToken(report.Message, "native-inference-ready:1"),
		DiskGB:                   statusTokenUint(report.Message, "disk_gb:"),
	}
	if infMgr != nil {
		runtimeInfo.NativeModel = infMgr.ModelName()
	}

	return operatorStatusResponse{
		Version:          s.version,
		HubURL:           s.hubURL,
		PublicKeyHex:     s.publicKeyHex,
		DeviceType:       s.deviceType,
		DeclaredCountry:  s.declaredCountry,
		Registered:       registered,
		RegisterError:    registerErr,
		LatestVersion:    latestVersion,
		LastHeartbeatAt:  lastHeartbeatAt,
		LastHeartbeatErr: lastHeartbeatErr,
		Machine: operatorMachine{
			CPUCores:  caps.CPUCores,
			RAMBytes:  caps.RAMBytes,
			GPUModel:  caps.GPUModel,
			VRAMBytes: caps.VRAMBytes,
		},
		Runtime:         runtimeInfo,
		Metrics:         metrics,
		CurrentJob:      currentJob,
		RecentJobs:      recent,
		LastClaimAt:     lastClaimAt,
		LastClaimError:  lastClaimErr,
		LastPayoutAt:    lastPayoutAt,
		LastPayoutError: lastPayoutErr,
	}
}

func startOperatorAPIServer(ctx context.Context, state *operatorRuntime, port string) {
	if state == nil || strings.TrimSpace(port) == "" || port == "0" {
		return
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{
			"ok":       true,
			"version":  state.version,
			"api_port": port,
		})
	})
	mux.HandleFunc("GET /api/v1/operator/status", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, state.statusSnapshot(port))
	})
	mux.HandleFunc("GET /api/v1/operator/jobs", func(w http.ResponseWriter, r *http.Request) {
		jobs, current := state.recentJobsSnapshot()
		writeJSON(w, http.StatusOK, map[string]any{
			"current_job": current,
			"jobs":        jobs,
		})
	})
	mux.HandleFunc("GET /api/v1/operator/logs", func(w http.ResponseWriter, r *http.Request) {
		limit := 200
		if raw := strings.TrimSpace(r.URL.Query().Get("limit")); raw != "" {
			if n, err := strconv.Atoi(raw); err == nil && n > 0 && n <= 1000 {
				limit = n
			}
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"lines": operatorLogBuffer.tail(limit),
		})
	})
	mux.HandleFunc("POST /api/v1/operator/claim", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Code string `json:"code"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": "invalid_json"})
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 15*time.Second)
		defer cancel()
		err := state.client.RedeemClaimCode(ctx, strings.TrimSpace(body.Code))
		state.recordClaim(err)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"status": "claimed"})
	})
	mux.HandleFunc("POST /api/v1/operator/payout", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			StripeConnectID string `json:"stripe_connect_id"`
			Currency        string `json:"currency"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": "invalid_json"})
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 15*time.Second)
		defer cancel()
		err := state.client.SavePayout(ctx, strings.TrimSpace(body.StripeConnectID), strings.TrimSpace(body.Currency))
		state.recordPayout(err)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"status": "saved"})
	})
	mux.HandleFunc("POST /api/v1/operator/connect/create", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Email   string `json:"email"`
			Country string `json:"country"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": "invalid_json"})
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 20*time.Second)
		defer cancel()
		accountID, err := state.client.CreateConnectAccount(ctx, strings.TrimSpace(body.Email), strings.TrimSpace(body.Country))
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"account_id": accountID})
	})
	mux.HandleFunc("POST /api/v1/operator/connect/onboarding-link", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			AccountID string `json:"account_id"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": "invalid_json"})
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 20*time.Second)
		defer cancel()
		url, err := state.client.ConnectOnboardingLink(ctx, strings.TrimSpace(body.AccountID))
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"url": url})
	})
	mux.HandleFunc("GET /api/v1/operator/connect/status", func(w http.ResponseWriter, r *http.Request) {
		accountID := strings.TrimSpace(r.URL.Query().Get("account_id"))
		if accountID == "" {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": "account_id required"})
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 15*time.Second)
		defer cancel()
		onboarded, err := state.client.ConnectStatus(ctx, accountID)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"account_id": accountID,
			"onboarded":  onboarded,
		})
	})

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		setLocalOperatorCORS(w, r)
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		mux.ServeHTTP(w, r)
	})

	srv := &http.Server{
		Addr:              "127.0.0.1:" + port,
		Handler:           handler,
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
	}()
	go func() {
		slog.Info("operator API listening", "addr", srv.Addr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			slog.Warn("operator API stopped", "error", err)
		}
	}()
}

func setLocalOperatorCORS(w http.ResponseWriter, r *http.Request) {
	origin := strings.TrimSpace(r.Header.Get("Origin"))
	if allowLocalOrigin(origin) {
		w.Header().Set("Access-Control-Allow-Origin", origin)
		w.Header().Set("Vary", "Origin")
	}
	w.Header().Set("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
}

func allowLocalOrigin(origin string) bool {
	origin = strings.ToLower(strings.TrimSpace(origin))
	if origin == "" {
		return false
	}
	return strings.HasPrefix(origin, "http://localhost:") ||
		strings.HasPrefix(origin, "http://127.0.0.1:") ||
		origin == "tauri://localhost" ||
		origin == "https://tauri.localhost"
}

func operatorAPIPort(flagValue string) string {
	if env := strings.TrimSpace(os.Getenv("RYV_UI_PORT")); env != "" {
		flagValue = env
	}
	flagValue = strings.TrimSpace(flagValue)
	if flagValue == "" {
		return defaultOperatorAPIPort
	}
	return flagValue
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func statusToken(msg, token string) bool {
	token = strings.ToLower(strings.TrimSpace(token))
	if token == "" {
		return false
	}
	for _, part := range strings.Split(strings.ToLower(msg), ",") {
		if strings.TrimSpace(part) == token {
			return true
		}
	}
	return false
}

func statusTokenUint(msg, prefix string) uint64 {
	prefix = strings.ToLower(strings.TrimSpace(prefix))
	if prefix == "" {
		return 0
	}
	for _, part := range strings.Split(strings.ToLower(msg), ",") {
		part = strings.TrimSpace(part)
		if !strings.HasPrefix(part, prefix) {
			continue
		}
		raw := strings.TrimSpace(strings.TrimPrefix(part, prefix))
		v, err := strconv.ParseUint(raw, 10, 64)
		if err == nil {
			return v
		}
	}
	return 0
}

type runnerResultSnapshot struct {
	DurationMs    int64
	ResultHashHex string
	BlobURL       string
	ObjectKey     string
	MeteringUnits uint64
	ExitCode      int
	Metadata      map[string]any
}
