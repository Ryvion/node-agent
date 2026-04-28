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

	version            string
	hubURL             string
	deviceType         string
	declaredCountry    string
	verifiedCountry    string
	locationApproved   bool
	sovereignVerified  bool
	verificationSource string
	trustReason        string
	publicKeyHex       string
	publicAIOptIn      bool
	caps               hw.CapSet
	client             *hub.Client

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
	runtimeMgr        *runtimeManager
}

type operatorJob struct {
	JobID          string         `json:"job_id"`
	Kind           string         `json:"kind"`
	Image          string         `json:"image,omitempty"`
	ExecutorKind   string         `json:"executor_kind,omitempty"`
	AssuranceClass string         `json:"assurance_class,omitempty"`
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
	Version          string              `json:"version"`
	HubURL           string              `json:"hub_url"`
	PublicKeyHex     string              `json:"public_key_hex"`
	DeviceType       string              `json:"device_type"`
	DeclaredCountry  string              `json:"declared_country,omitempty"`
	VerifiedCountry  string              `json:"verified_country,omitempty"`
	Registered       bool                `json:"registered"`
	RegisterError    string              `json:"register_error,omitempty"`
	LatestVersion    string              `json:"latest_version,omitempty"`
	LastHeartbeatAt  time.Time           `json:"last_heartbeat_at,omitempty"`
	LastHeartbeatErr string              `json:"last_heartbeat_error,omitempty"`
	Machine          operatorMachine     `json:"machine"`
	Runtime          operatorRuntimeInfo `json:"runtime"`
	Metrics          operatorMetrics     `json:"metrics"`
	CurrentJob       *operatorJob        `json:"current_job,omitempty"`
	RecentJobs       []operatorJob       `json:"recent_jobs"`
	LastClaimAt      time.Time           `json:"last_claim_at,omitempty"`
	LastClaimError   string              `json:"last_claim_error,omitempty"`
	LastPayoutAt     time.Time           `json:"last_payout_at,omitempty"`
	LastPayoutError  string              `json:"last_payout_error,omitempty"`
}

type operatorMachine struct {
	CPUCores  uint32 `json:"cpu_cores"`
	RAMBytes  uint64 `json:"ram_bytes"`
	GPUModel  string `json:"gpu_model,omitempty"`
	VRAMBytes uint64 `json:"vram_bytes,omitempty"`
}

type operatorRuntimeInfo struct {
	LocalAPIURL              string `json:"local_api_url"`
	StatusMessage            string `json:"status_message,omitempty"`
	RuntimeReady             bool   `json:"runtime_ready"`
	RuntimeGPUReady          bool   `json:"runtime_gpu_ready"`
	RuntimeWarming           bool   `json:"runtime_warming"`
	RuntimeHealth            string `json:"runtime_health,omitempty"`
	RuntimePosture           string `json:"runtime_posture,omitempty"`
	RuntimeDetail            string `json:"runtime_detail,omitempty"`
	RuntimeVersion           string `json:"runtime_version,omitempty"`
	RuntimeChannel           string `json:"runtime_channel,omitempty"`
	RuntimeProvider          string `json:"runtime_provider,omitempty"`
	RuntimeMode              string `json:"runtime_mode,omitempty"`
	RuntimeSource            string `json:"runtime_source,omitempty"`
	RuntimeArtifact          string `json:"runtime_artifact,omitempty"`
	RuntimeBinary            string `json:"runtime_binary,omitempty"`
	RuntimeBackend           string `json:"runtime_backend,omitempty"`
	RuntimeEngine            string `json:"runtime_engine,omitempty"`
	RuntimeEngineKind        string `json:"runtime_engine_kind,omitempty"`
	RuntimeBackendPresent    bool   `json:"runtime_backend_present"`
	RuntimeManifestHash      string `json:"runtime_manifest_hash,omitempty"`
	ManagedOCIGPUReady       bool   `json:"managed_oci_gpu_ready"`
	GPUReady                 bool   `json:"gpu_ready"`
	SpatialReady             bool   `json:"spatial_ready"`
	PublicAIOptIn            bool   `json:"public_ai_opt_in"`
	PublicAIReady            bool   `json:"public_ai_ready"`
	NativeInferenceSupported bool   `json:"native_inference_supported"`
	NativeInferenceReady     bool   `json:"native_inference_ready"`
	PublicInferenceReady     bool   `json:"public_inference_ready"`
	SovereignReviewReady     bool   `json:"sovereign_review_ready"`
	SovereignStatus          string `json:"sovereign_status,omitempty"`
	SovereignDetail          string `json:"sovereign_detail,omitempty"`
	VerifiedCountry          string `json:"verified_country,omitempty"`
	LocationApproved         bool   `json:"location_approved"`
	SovereignVerified        bool   `json:"sovereign_verified"`
	VerificationSource       string `json:"verification_source,omitempty"`
	TrustReason              string `json:"trust_reason,omitempty"`
	NativeModel              string `json:"native_model,omitempty"`
	DiskGB                   uint64 `json:"disk_gb,omitempty"`
}

type operatorMetrics struct {
	TimestampMs  int64   `json:"timestamp_ms,omitempty"`
	CPUUtil      float64 `json:"cpu_util,omitempty"`
	MemUtil      float64 `json:"mem_util,omitempty"`
	GPUUtil      float64 `json:"gpu_util,omitempty"`
	PowerWatts   float64 `json:"power_watts,omitempty"`
	GPUThrottled bool    `json:"gpu_throttled,omitempty"`
}

type operatorDiagnosticCheck struct {
	Key      string `json:"key"`
	Label    string `json:"label"`
	Ready    bool   `json:"ready"`
	Detail   string `json:"detail,omitempty"`
	Severity string `json:"severity,omitempty"`
}

type operatorDiagnosticIssue struct {
	Key       string    `json:"key"`
	Message   string    `json:"message"`
	UpdatedAt time.Time `json:"updated_at,omitempty"`
}

type operatorDiagnosticsResponse struct {
	Version         string                    `json:"version"`
	LatestVersion   string                    `json:"latest_version,omitempty"`
	LocalAPIURL     string                    `json:"local_api_url"`
	DeclaredCountry string                    `json:"declared_country,omitempty"`
	VerifiedCountry string                    `json:"verified_country,omitempty"`
	RuntimeChecks   []operatorDiagnosticCheck `json:"runtime_checks"`
	Recommendations []string                  `json:"recommendations"`
	Issues          []operatorDiagnosticIssue `json:"issues"`
	StatusTokens    []string                  `json:"status_tokens,omitempty"`
	LogTail         []string                  `json:"log_tail"`
	LastHeartbeatAt time.Time                 `json:"last_heartbeat_at,omitempty"`
	LastClaimAt     time.Time                 `json:"last_claim_at,omitempty"`
	LastPayoutAt    time.Time                 `json:"last_payout_at,omitempty"`
}

type logRing struct {
	mu      sync.Mutex
	pending bytes.Buffer
	lines   []string
	limit   int
}

func newOperatorRuntime(version, hubURL, deviceType, declaredCountry string, publicAIOptIn bool, caps hw.CapSet, client *hub.Client) *operatorRuntime {
	return &operatorRuntime{
		version:         strings.TrimSpace(version),
		hubURL:          strings.TrimSpace(hubURL),
		deviceType:      strings.TrimSpace(deviceType),
		declaredCountry: strings.ToUpper(strings.TrimSpace(declaredCountry)),
		publicKeyHex:    client.PublicKeyHex(),
		publicAIOptIn:   publicAIOptIn,
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

func (s *operatorRuntime) setRuntimeManager(runtimeMgr *runtimeManager) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.runtimeMgr = runtimeMgr
}

func (s *operatorRuntime) publicAIOptInEnabled() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.publicAIOptIn
}

func (s *operatorRuntime) updatePublicAIOptIn(enabled bool) error {
	if _, err := mutateOperatorPreferences(func(prefs *operatorPreferences) {
		prefs.PublicAIOptIn = enabled
		prefs.PublicAIOptOut = !enabled
	}); err != nil {
		return err
	}

	s.mu.Lock()
	s.publicAIOptIn = enabled
	caps := s.caps
	infMgr := s.infMgr
	runtimeMgr := s.runtimeMgr
	client := s.client
	s.mu.Unlock()

	if runtimeMgr == nil {
		return nil
	}

	report := buildHealthReport(caps, infMgr, runtimeMgr)
	s.recordHealthReport(report)

	if client != nil {
		reportCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := client.SendHealthReport(reportCtx, report); err != nil {
			slog.Warn("public AI preference health sync failed", "error", err)
		}
	}

	return nil
}

func (s *operatorRuntime) updateDeclaredCountry(country string) error {
	country = normalizeDeclaredCountry(country)
	if _, err := mutateOperatorPreferences(func(prefs *operatorPreferences) {
		prefs.DeclaredCountry = country
	}); err != nil {
		return err
	}

	s.mu.Lock()
	s.declaredCountry = country
	s.mu.Unlock()
	return nil
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

func (s *operatorRuntime) recordHeartbeat(metrics hw.Metrics, heartbeat hub.HeartbeatResponse, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lastMetrics = metrics
	if err != nil {
		s.lastHeartbeatErr = strings.TrimSpace(err.Error())
		return
	}
	s.lastHeartbeatAt = time.Now()
	s.lastHeartbeatErr = ""
	if heartbeat.LatestVersion != "" {
		s.latestVersion = heartbeat.LatestVersion
	}
	s.verifiedCountry = normalizeDeclaredCountry(heartbeat.CountryCode)
	s.locationApproved = heartbeat.LocationApproved
	s.sovereignVerified = heartbeat.SovereignVerified
	s.verificationSource = strings.TrimSpace(heartbeat.VerificationSource)
	s.trustReason = strings.TrimSpace(heartbeat.TrustReason)
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
		JobID:          strings.TrimSpace(work.JobID),
		Kind:           strings.TrimSpace(work.Kind),
		Image:          strings.TrimSpace(work.Image),
		ExecutorKind:   executorKindForAssignment(work),
		AssuranceClass: assuranceClassForAssignment(work),
		Status:         "running",
		StartedAt:      time.Now(),
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
		JobID:          strings.TrimSpace(work.JobID),
		Kind:           strings.TrimSpace(work.Kind),
		Image:          strings.TrimSpace(work.Image),
		ExecutorKind:   executorKindForAssignment(work),
		AssuranceClass: assuranceClassForAssignment(work),
		Status:         "completed",
		StartedAt:      now,
		CompletedAt:    now,
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
	declaredCountry := s.declaredCountry
	verifiedCountry := s.verifiedCountry
	locationApproved := s.locationApproved
	sovereignVerified := s.sovereignVerified
	verificationSource := s.verificationSource
	trustReason := s.trustReason
	report := s.lastHealthReport
	infMgr := s.infMgr
	runtimeMgr := s.runtimeMgr
	publicAIOptIn := s.publicAIOptIn
	s.mu.RUnlock()

	report = freshOperatorHealthReport(caps, infMgr, runtimeMgr, report)

	var currentJob *operatorJob
	if current != nil {
		cp := *current
		currentJob = &cp
	}

	nativeSupported := inference.NativeRuntimeAvailable()
	nativeReady := nativeSupported && infMgr != nil && infMgr.Healthy()
	runtimeReady := statusToken(report.Message, "runtime-ready:1") || statusToken(report.Message, "docker-ready:1")
	runtimeGPUReady := statusToken(report.Message, "runtime-gpu-ready:1") || statusToken(report.Message, "docker-gpu:ok")
	runtimeHealth := statusTokenValue(report.Message, "runtime-health:")
	runtimePosture, runtimeDetail, runtimeWarming := deriveRuntimePosture(runtimeReady, runtimeHealth)
	sovereignReviewReady, sovereignStatus, sovereignDetail := deriveSovereignPosture(registered, declaredCountry, verifiedCountry, locationApproved, sovereignVerified, trustReason, runtimeReady, runtimeHealth, nativeReady)

	runtimeInfo := operatorRuntimeInfo{
		LocalAPIURL:              fmt.Sprintf("http://127.0.0.1:%s", apiPort),
		StatusMessage:            report.Message,
		RuntimeReady:             runtimeReady,
		RuntimeGPUReady:          runtimeGPUReady,
		RuntimeWarming:           runtimeWarming,
		RuntimeHealth:            runtimeHealth,
		RuntimePosture:           runtimePosture,
		RuntimeDetail:            runtimeDetail,
		RuntimeVersion:           statusTokenValue(report.Message, "runtime-version:"),
		RuntimeChannel:           statusTokenValue(report.Message, "runtime-channel:"),
		RuntimeProvider:          statusTokenValue(report.Message, "runtime-provider:"),
		RuntimeMode:              statusTokenValue(report.Message, "runtime-mode:"),
		RuntimeSource:            statusTokenValue(report.Message, "runtime-source:"),
		RuntimeArtifact:          statusTokenValue(report.Message, "runtime-artifact:"),
		RuntimeBinary:            statusTokenValue(report.Message, "runtime-binary:"),
		RuntimeBackend:           statusTokenValue(report.Message, "runtime-backend:"),
		RuntimeEngine:            statusTokenValue(report.Message, "runtime-engine:"),
		RuntimeEngineKind:        statusTokenValue(report.Message, "runtime-engine-kind:"),
		RuntimeManifestHash:      statusTokenValue(report.Message, "runtime-manifest-hash:"),
		GPUReady:                 report.GPUReady,
		SpatialReady:             statusToken(report.Message, "spatial-ready:1"),
		PublicAIOptIn:            publicAIOptIn,
		PublicAIReady:            publicAIOptIn,
		NativeInferenceSupported: nativeSupported,
		NativeInferenceReady:     nativeReady,
		PublicInferenceReady:     publicAIOptIn && nativeReady,
		SovereignReviewReady:     sovereignReviewReady,
		SovereignStatus:          sovereignStatus,
		SovereignDetail:          sovereignDetail,
		VerifiedCountry:          verifiedCountry,
		LocationApproved:         locationApproved,
		SovereignVerified:        sovereignVerified,
		VerificationSource:       verificationSource,
		TrustReason:              trustReason,
		DiskGB:                   statusTokenUint(report.Message, "disk_gb:"),
	}
	if infMgr != nil {
		runtimeInfo.NativeModel = infMgr.ModelName()
	}
	runtimeInfo.RuntimeBackendPresent = runtimeInfo.RuntimeBackend != ""
	runtimeInfo.ManagedOCIGPUReady = runtimeInfo.RuntimeGPUReady

	return operatorStatusResponse{
		Version:          s.version,
		HubURL:           s.hubURL,
		PublicKeyHex:     s.publicKeyHex,
		DeviceType:       s.deviceType,
		DeclaredCountry:  declaredCountry,
		VerifiedCountry:  verifiedCountry,
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
		Runtime: runtimeInfo,
		Metrics: operatorMetrics{
			CPUUtil:    metrics.CPUUtil,
			MemUtil:    metrics.MemUtil,
			GPUUtil:    metrics.GPUUtil,
			PowerWatts: metrics.PowerWatts,
		},
		CurrentJob:      currentJob,
		RecentJobs:      recent,
		LastClaimAt:     lastClaimAt,
		LastClaimError:  lastClaimErr,
		LastPayoutAt:    lastPayoutAt,
		LastPayoutError: lastPayoutErr,
	}
}

func (s *operatorRuntime) diagnosticsSnapshot(apiPort string) operatorDiagnosticsResponse {
	s.mu.RLock()
	version := s.version
	latestVersion := s.latestVersion
	caps := s.caps
	declaredCountry := s.declaredCountry
	verifiedCountry := s.verifiedCountry
	locationApproved := s.locationApproved
	sovereignVerified := s.sovereignVerified
	trustReason := s.trustReason
	registered := s.registered
	lastHeartbeatAt := s.lastHeartbeatAt
	lastHeartbeatErr := s.lastHeartbeatErr
	lastRegisterErr := s.lastRegisterError
	lastClaimAt := s.lastClaimAt
	lastClaimErr := s.lastClaimError
	lastPayoutAt := s.lastPayoutAt
	lastPayoutErr := s.lastPayoutError
	report := s.lastHealthReport
	infMgr := s.infMgr
	runtimeMgr := s.runtimeMgr
	publicAIOptIn := s.publicAIOptIn
	s.mu.RUnlock()

	report = freshOperatorHealthReport(caps, infMgr, runtimeMgr, report)

	nativeReady := inference.NativeRuntimeAvailable() && infMgr != nil && infMgr.Healthy()
	publicInferenceReady := publicAIOptIn && nativeReady
	runtimeReady := statusToken(report.Message, "runtime-ready:1") || statusToken(report.Message, "docker-ready:1")
	runtimeHealth := statusTokenValue(report.Message, "runtime-health:")
	runtimePosture, runtimeDetail, runtimeWarming := deriveRuntimePosture(runtimeReady, runtimeHealth)
	sovereignReviewReady, _, sovereignDetail := deriveSovereignPosture(registered, declaredCountry, verifiedCountry, locationApproved, sovereignVerified, trustReason, runtimeReady, runtimeHealth, nativeReady)
	managedRuntimeDetail := "Required for video transcode, embeddings, agent hosting, and other OCI workloads."
	if runtimeWarming {
		managedRuntimeDetail = runtimeDetail
	}
	managedRuntimeSeverity := "warn"
	if runtimeWarming {
		managedRuntimeSeverity = "neutral"
	}

	runtimeChecks := []operatorDiagnosticCheck{
		{
			Key:      "runtime_cli",
			Label:    "Execution runtime",
			Ready:    statusTokenValue(report.Message, "runtime-backend:") != "",
			Detail:   "Required to inspect and run managed OCI workloads.",
			Severity: "warn",
		},
		{
			Key:      "managed_runtime",
			Label:    "Managed OCI runtime",
			Ready:    runtimeReady,
			Detail:   managedRuntimeDetail,
			Severity: managedRuntimeSeverity,
		},
		{
			Key:      "gpu_runtime",
			Label:    "GPU runtime",
			Ready:    report.GPUReady || statusToken(report.Message, "runtime-gpu-ready:1") || statusToken(report.Message, "docker-gpu:ok"),
			Detail:   "Required for GPU-backed workload classes.",
			Severity: "neutral",
		},
		{
			Key:      "public_ai_opt_in",
			Label:    "Public AI opt-in",
			Ready:    publicAIOptIn,
			Detail:   "Required before buyer-facing AI jobs can land on this machine.",
			Severity: "warn",
		},
		{
			Key:      "native_inference",
			Label:    "Native inference",
			Ready:    publicInferenceReady,
			Detail:   "Required before public chat inference can use the native runtime on this machine.",
			Severity: "neutral",
		},
		{
			Key:      "sovereign_review",
			Label:    "Sovereign review posture",
			Ready:    sovereignReviewReady,
			Detail:   sovereignDetail,
			Severity: "neutral",
		},
		{
			Key:      "spatial_runtime",
			Label:    "Spatial runtime",
			Ready:    statusToken(report.Message, "spatial-ready:1"),
			Detail:   "Required for spatial staging workloads.",
			Severity: "neutral",
		},
	}
	if infMgr != nil && infMgr.ModelName() != "" {
		for i := range runtimeChecks {
			if runtimeChecks[i].Key == "native_inference" {
				runtimeChecks[i].Detail = fmt.Sprintf("%s (model: %s)", runtimeChecks[i].Detail, infMgr.ModelName())
			}
		}
	}

	var issues []operatorDiagnosticIssue
	if strings.TrimSpace(lastRegisterErr) != "" {
		issues = append(issues, operatorDiagnosticIssue{Key: "register", Message: lastRegisterErr})
	}
	if strings.TrimSpace(lastHeartbeatErr) != "" {
		issues = append(issues, operatorDiagnosticIssue{Key: "heartbeat", Message: lastHeartbeatErr, UpdatedAt: lastHeartbeatAt})
	}
	if strings.TrimSpace(lastClaimErr) != "" {
		issues = append(issues, operatorDiagnosticIssue{Key: "claim", Message: lastClaimErr, UpdatedAt: lastClaimAt})
	}
	if strings.TrimSpace(lastPayoutErr) != "" {
		issues = append(issues, operatorDiagnosticIssue{Key: "payout", Message: lastPayoutErr, UpdatedAt: lastPayoutAt})
	}

	recommendations := make([]string, 0, 6)
	if !runtimeReady {
		if runtimeWarming {
			recommendations = append(recommendations, "Leave the node online while the managed runtime finishes first-run startup, then refresh status before attempting repair again.")
		} else {
			recommendations = append(recommendations, "Repair or start the execution runtime before login so managed OCI workloads can land immediately.")
		}
	}
	if !nativeReady {
		recommendations = append(recommendations, "Load or repair the native model path if you want low-latency gateway inference without OCI startup.")
	}
	if nativeReady && !publicInferenceReady {
		recommendations = append(recommendations, "Enable public participation only on machines you explicitly want exposed to buyer-facing AI jobs.")
	}
	if strings.TrimSpace(verifiedCountry) == "" && strings.TrimSpace(declaredCountry) == "" {
		recommendations = append(recommendations, "Wait for hub location verification or set an ISO country code on the node if you need a local sovereign posture hint immediately.")
	}
	if runtimePosture == "warming" && len(issues) == 0 {
		recommendations = append(recommendations, "Managed OCI runtime is still warming in the background. Gateway-native inference can continue using the local model path while OCI startup finishes.")
	}
	if len(issues) == 0 && len(recommendations) == 0 {
		recommendations = append(recommendations, "Runtime posture is clean. Keep the node service and execution runtime healthy to preserve workload eligibility.")
	}

	return operatorDiagnosticsResponse{
		Version:         version,
		LatestVersion:   latestVersion,
		LocalAPIURL:     fmt.Sprintf("http://127.0.0.1:%s", apiPort),
		DeclaredCountry: declaredCountry,
		VerifiedCountry: verifiedCountry,
		RuntimeChecks:   runtimeChecks,
		Recommendations: recommendations,
		Issues:          issues,
		StatusTokens:    splitStatusTokens(report.Message),
		LogTail:         operatorLogBuffer.tail(80),
		LastHeartbeatAt: lastHeartbeatAt,
		LastClaimAt:     lastClaimAt,
		LastPayoutAt:    lastPayoutAt,
	}
}

func freshOperatorHealthReport(caps hw.CapSet, infMgr *inference.Manager, runtimeMgr *runtimeManager, fallback hub.HealthReport) hub.HealthReport {
	if runtimeMgr == nil {
		return fallback
	}
	report := buildHealthReport(caps, infMgr, runtimeMgr)
	if strings.TrimSpace(report.Message) == "" {
		return fallback
	}
	return report
}

func deriveRuntimePosture(runtimeReady bool, runtimeHealth string) (string, string, bool) {
	health := strings.ToLower(strings.TrimSpace(runtimeHealth))
	switch {
	case runtimeReady || health == "ready":
		return "ready", "Managed OCI runtime is ready for container-backed workloads.", false
	case health == "warming":
		return "warming", "Managed OCI runtime is installed and still finishing background startup.", true
	case health == "degraded":
		return "degraded", "Managed OCI runtime is installed but not yet healthy for OCI workloads.", false
	case health == "missing", health == "":
		return "unavailable", "Managed OCI runtime is not available on this machine.", false
	default:
		return health, fmt.Sprintf("Managed OCI runtime health is reported as %s.", health), false
	}
}

func deriveSovereignPosture(registered bool, declaredCountry string, verifiedCountry string, locationApproved bool, sovereignVerified bool, trustReason string, runtimeReady bool, runtimeHealth string, nativeReady bool) (bool, string, string) {
	declaredCountry = normalizeDeclaredCountry(declaredCountry)
	verifiedCountry = normalizeDeclaredCountry(verifiedCountry)
	if declaredCountry != "" && verifiedCountry != "" && declaredCountry != verifiedCountry {
		return false, "country_mismatch", fmt.Sprintf("Declared country %s does not match hub-verified country %s. Fix the local setting or investigate location verification before sovereign routing.", declaredCountry, verifiedCountry)
	}
	if verifiedCountry == "" && declaredCountry == "" {
		return false, "country_missing", "Country has not been verified by the hub yet. Ryvion should infer it automatically from network verification, but you can set RYV_DECLARED_COUNTRY=CA or save the ISO country code in Operator Settings as a fallback hint."
	}
	if !registered {
		return false, "registration_pending", "The node must register successfully before sovereign routing review can proceed."
	}
	if verifiedCountry != "" && (!locationApproved || !sovereignVerified) {
		if strings.TrimSpace(trustReason) != "" {
			return false, "trust_review_pending", trustReason
		}
		return false, "trust_review_pending", "Hub location verification is present, but sovereign approval has not been granted yet."
	}
	if !runtimeReady && !nativeReady {
		if strings.EqualFold(strings.TrimSpace(runtimeHealth), "warming") {
			return false, "runtime_warming", "Managed execution runtime is still warming in the background. Sovereign review can continue once local runtime readiness turns green, or sooner if native inference becomes healthy."
		}
		return false, "runtime_unavailable", "Bring either the native inference path or the managed execution runtime online before sovereign workloads can be considered."
	}
	if verifiedCountry != "" {
		return true, "review_ready", fmt.Sprintf("Hub-verified country %s is present and local runtime prerequisites are satisfied. Final sovereign eligibility still depends on hub trust review and jurisdiction policy.", verifiedCountry)
	}
	return true, "review_ready", "Local prerequisites are satisfied. Final sovereign eligibility still depends on hub trust review and jurisdiction policy."
}

func normalizeDeclaredCountry(raw string) string {
	value := strings.ToUpper(strings.TrimSpace(raw))
	if value == "" {
		return ""
	}
	if len(value) != 2 {
		return ""
	}
	for _, r := range value {
		if r < 'A' || r > 'Z' {
			return ""
		}
	}
	return value
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
	mux.HandleFunc("POST /api/v1/operator/preferences/public-ai", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Enabled *bool `json:"enabled"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": "invalid_json"})
			return
		}
		if body.Enabled == nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": "enabled required"})
			return
		}
		if err := state.updatePublicAIOptIn(*body.Enabled); err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, state.statusSnapshot(port))
	})
	mux.HandleFunc("POST /api/v1/operator/preferences/declared-country", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Country *string `json:"country"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": "invalid_json"})
			return
		}
		if body.Country == nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": "country required"})
			return
		}
		rawCountry := strings.TrimSpace(*body.Country)
		if rawCountry != "" && normalizeDeclaredCountry(rawCountry) == "" {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": "country must be an ISO 3166-1 alpha-2 code"})
			return
		}
		if err := state.updateDeclaredCountry(rawCountry); err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]any{"error": err.Error()})
			return
		}
		writeJSON(w, http.StatusOK, state.statusSnapshot(port))
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
	mux.HandleFunc("GET /api/v1/operator/diagnostics", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, state.diagnosticsSnapshot(port))
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
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      15 * time.Second,
		IdleTimeout:       60 * time.Second,
		MaxHeaderBytes:    1 << 20,
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

func statusTokenValue(msg, prefix string) string {
	prefix = strings.ToLower(strings.TrimSpace(prefix))
	if prefix == "" {
		return ""
	}
	for _, part := range strings.Split(msg, ",") {
		part = strings.TrimSpace(part)
		if strings.HasPrefix(strings.ToLower(part), prefix) {
			return strings.TrimSpace(part[len(prefix):])
		}
	}
	return ""
}

func splitStatusTokens(msg string) []string {
	parts := strings.Split(msg, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		out = append(out, part)
	}
	return out
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
