package inference

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Ryvion/ryvion-node/internal/runtimeexec"
)

const (
	defaultPort       = "8081"
	defaultThreads    = "4"
	defaultGPULayers  = "99"
	defaultCtxSize    = "16384"
	healthCheckPeriod = 5 * time.Second
	// Cold CUDA loads of larger native models (phi-4 ~9 GB Q4_K_M, gemma-4
	// 26B ~17 GB) can spend 60–180 s in cudaMalloc + tensor copies before
	// llama-server's /health responds. The previous 120 s ceiling silently
	// flipped these nodes to public-inference-ready:0 even though they were
	// still warming up. Override via RYV_INFERENCE_STARTUP_TIMEOUT_SECONDS.
	defaultStartupTimeout = 240 * time.Second
	// GPU drivers commonly reserve a small slice of VRAM, so a 16 GB card can
	// report slightly below 16 GiB. Keep model eligibility aligned with the
	// hub's displayed/capability GB values instead of hiding valid cards.
	modelVRAMReserveTolerance = 256 * 1024 * 1024
	nodeTokenHeader           = "X-Node-Token"
)

// resolvedStartupTimeout returns the cold-start budget for llama-server's
// /health probe. RYV_INFERENCE_STARTUP_TIMEOUT_SECONDS overrides the default
// when slow GPUs need extra time (e.g. shared CUDA contexts or paging in a
// 17 GB model file from spinning disk).
func resolvedStartupTimeout() time.Duration {
	raw := strings.TrimSpace(os.Getenv("RYV_INFERENCE_STARTUP_TIMEOUT_SECONDS"))
	if raw == "" {
		return defaultStartupTimeout
	}
	secs, err := strconv.Atoi(raw)
	if err != nil || secs <= 0 {
		return defaultStartupTimeout
	}
	return time.Duration(secs) * time.Second
}

// ModelMode distinguishes chat-style decoder models from encoder models that
// llama-server serves via its /v1/embeddings endpoint. An empty Mode defaults
// to chat — preserving the prior ModelConfig shape.
type ModelMode string

const (
	ModeChat      ModelMode = ""
	ModeEmbedding ModelMode = "embedding"
)

type ModelConfig struct {
	FileName string
	URL      string
	// PlatformPath points at a Ryvion hub model artifact endpoint. It lets the
	// platform handle gated/model-license downloads instead of asking each
	// operator to configure third-party tokens.
	PlatformPath string
	// Mode switches llama-server between chat/completion serving (default)
	// and embedding serving (--embedding flag, /v1/embeddings endpoint).
	Mode                    ModelMode
	MinVRAMBytes            uint64
	RequiresHuggingFaceAuth bool
	// MmprojFileName + MmprojURL are the multimodal vision projector
	// artifact for vision-capable models like Gemma 4 26B and Nemotron
	// 3 Nano Omni 30B. When set, the file is downloaded alongside the
	// GGUF and llama-server is launched with --mmproj pointing at it,
	// which is what makes the model actually consume image_url parts
	// from OpenAI multimodal messages. Empty for text-only models.
	MmprojFileName     string
	MmprojURL          string
	MmprojPlatformPath string
}

// NativeModels maps UI model names to GGUF downloads
var NativeModels = map[string]ModelConfig{
	"ryvion-llama-3.2-3b": {FileName: "Llama-3.2-3B-Instruct-Q4_K_M.gguf", URL: "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"},
	"phi-4":               {FileName: "phi-4-Q4_K_M.gguf", URL: "https://huggingface.co/bartowski/phi-4-GGUF/resolve/main/phi-4-Q4_K_M.gguf", MinVRAMBytes: 8 * 1024 * 1024 * 1024},
	"qwen3-8b-reasoning": {
		FileName:                "Qwen3-8B-Q4_K_M.gguf",
		URL:                     "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
		PlatformPath:            "/api/v1/node/models/qwen3-8b-reasoning/download",
		MinVRAMBytes:            8 * 1024 * 1024 * 1024,
		RequiresHuggingFaceAuth: false,
	},
	"gpt-oss-20b": {
		FileName:                "gpt-oss-20b-mxfp4.gguf",
		URL:                     "https://huggingface.co/ggml-org/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-mxfp4.gguf",
		PlatformPath:            "/api/v1/node/models/gpt-oss-20b/download",
		MinVRAMBytes:            16 * 1024 * 1024 * 1024,
		RequiresHuggingFaceAuth: false,
	},
	"gemma-4-26b-a4b-it": {
		FileName: "gemma-4-26B-A4B-it-Q4_K_M.gguf",
		URL:      "https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-Q4_K_M.gguf",
		// PlatformPath kept as a fallback so operators can opt into the
		// Ryvion-managed proxy via a flag flip without code changes, but the
		// public ggml-org GGUF mirror does not require an HF token (verified
		// 2026-05: HEAD returns 302 → CDN with no auth header). Keeping
		// RequiresHuggingFaceAuth=true was hiding the model from operators
		// who didn't set HF_TOKEN themselves, which defeated the whole point
		// of having the public mirror.
		PlatformPath: "/api/v1/node/models/gemma-4-26b-a4b-it/download",
		// 14 GiB floor — Q4_K_M Gemma 26B fits in ~13-15GB VRAM with all
		// layers offloaded; llama.cpp's `-ngl` partial-offload handles the
		// edge. A 16GB card (RTX 4070 Ti SUPER, RTX 4080) reports ~16,376 MiB
		// available after driver overhead, comfortably above 14 GiB. LM Studio
		// runs Gemma 31B on the same hardware via the same offload mechanism.
		MinVRAMBytes:            14 * 1024 * 1024 * 1024,
		RequiresHuggingFaceAuth: false,
		// Vision projector — auto-downloaded alongside the model so
		// llama-server launches with --mmproj and the model can answer
		// about images natively (no OCR fallback needed). ~2 GB on disk.
		MmprojFileName:     "mmproj-gemma-4-26B-A4B-it-F16.gguf",
		MmprojURL:          "https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF/resolve/main/mmproj-gemma-4-26B-A4B-it-F16.gguf",
		MmprojPlatformPath: "/api/v1/node/models/gemma-4-26b-a4b-it/artifacts/mmproj/download",
	},
	"tinyllama": {FileName: "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", URL: "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"},
	"nemotron-3-nano-omni-30b-a3b": {
		// NVIDIA Nemotron 3 Nano Omni 30B-A3B — MoE model (30B total /
		// 3B active per token), multimodal (vision via mmproj).
		//
		// ggml-org published the GGUF at ggml-org/NVIDIA-Nemotron-3-Nano-Omni.
		// The previously-configured ggml-org/Nemotron-3-Nano-Omni-30B-A3B-GGUF
		// path now 404s (upstream renamed the repo) — that dead URL, not gating,
		// was why every download failed. The repo is PUBLIC, so the node pulls
		// it straight from the HF CDN: no token, no hub proxy.
		//
		// Q4_K_M is ~24.5 GB — larger than any fleet GPU. It does NOT fit via
		// `--n-gpu-layers` offload (that places whole layers on the GPU and OOMs
		// on a 24 GB card, never mind 16 GB). The streaming launcher instead passes
		// `--cpu-moe`, keeping the MoE expert weights in system RAM so only
		// attention + KV + mmproj (~6-8 GB) sit on the GPU. That is why the gate is
		// 14GB: the GPU side genuinely fits a 14-16GB card — the operator just needs
		// ~24 GB of system RAM for the experts. See streamingMoECPUOffloadArgsForModel.
		FileName:                "nemotron-3-nano-omni-30b-a3b-Q4_K_M.gguf",
		URL:                     "https://huggingface.co/ggml-org/NVIDIA-Nemotron-3-Nano-Omni/resolve/main/nemotron-3-nano-omni-ga_v1.0-Q4_K_M.gguf",
		PlatformPath:            "/api/v1/node/models/nemotron-3-nano-omni-30b-a3b/download",
		MinVRAMBytes:            14 * 1024 * 1024 * 1024,
		RequiresHuggingFaceAuth: false,
		// Vision projector for the Omni multimodal mode. Same auto-
		// download + --mmproj launch flow as Gemma 4.
		MmprojFileName:     "mmproj-nemotron-3-nano-omni-30b-a3b-F16.gguf",
		MmprojURL:          "https://huggingface.co/ggml-org/NVIDIA-Nemotron-3-Nano-Omni/resolve/main/mmproj-nemotron-3-nano-omni-ga_v1.0.gguf",
		MmprojPlatformPath: "/api/v1/node/models/nemotron-3-nano-omni-30b-a3b/artifacts/mmproj/download",
	},
	// Phase 1c: native embeddings. nomic-embed-text-v1.5 is 137M params,
	// 768-dim, matches OpenAI text-embedding-3-small quality on MTEB, and
	// the Q4_K_M GGUF is ~90MB. llama-server serves it via /v1/embeddings
	// when launched with --embedding.
	"nomic-embed-text-v1.5": {
		FileName: "nomic-embed-text-v1.5.Q4_K_M.gguf",
		URL:      "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf",
		Mode:     ModeEmbedding,
	},
}

// SupportedNativeChatModels returns chat models this node is willing to advertise
// to the hub. Large gated models stay hidden unless local hardware is realistically
// capable; the hub then uses these tokens as a hard capability gate.
//
// Note: this is the *hardware-eligible* list — it does NOT check whether the
// GGUF is on disk. The hub may still route a job to a node whose model file
// hasn't been downloaded yet; the node will then download on demand. Startup
// prewarm intentionally selects a smaller subset so operators do not fill disks
// by advertising one large model family.
func SupportedNativeChatModels(vramBytes uint64) []string {
	out := make([]string, 0, len(NativeModels))
	for id, cfg := range NativeModels {
		if cfg.Mode == ModeEmbedding {
			continue
		}
		if !meetsModelVRAMRequirement(vramBytes, cfg.MinVRAMBytes) {
			continue
		}
		if cfg.RequiresHuggingFaceAuth && huggingFaceToken() == "" && !platformManagedGatedModelsEnabled() {
			continue
		}
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

// ActiveDownloads returns a snapshot of every in-flight model download. The
// agent reports this set in its heartbeat status as `download:<id>:<pct>:<mb_done>:<mb_total>`
// tokens; the hub parses those tokens and surfaces them to the buyer's
// playground so the cold-start UI shows actual progress instead of a
// generic spinner.
func (m *Manager) ActiveDownloads() []ActiveDownload {
	if m == nil {
		return nil
	}
	m.downloadsMu.RLock()
	defer m.downloadsMu.RUnlock()
	out := make([]ActiveDownload, 0, len(m.activeDownloads))
	for _, d := range m.activeDownloads {
		out = append(out, *d)
	}
	return out
}

func (m *Manager) registerDownload(modelID string) *ActiveDownload {
	d := &ActiveDownload{
		ModelID:   modelID,
		StartedAt: time.Now(),
	}
	m.downloadsMu.Lock()
	m.activeDownloads[modelID] = d
	m.downloadsMu.Unlock()
	return d
}

func (m *Manager) updateDownloadProgress(modelID string, bytesDone, bytesTotal int64) {
	m.downloadsMu.Lock()
	defer m.downloadsMu.Unlock()
	d, ok := m.activeDownloads[modelID]
	if !ok {
		return
	}
	d.BytesDone = bytesDone
	d.BytesTotal = bytesTotal
}

func (m *Manager) completeDownload(modelID string) {
	m.downloadsMu.Lock()
	defer m.downloadsMu.Unlock()
	delete(m.activeDownloads, modelID)
}

// modelDownloadBytes reports the bytes fetched so far for a model's GGUF when a
// download is currently in flight. EnsureModel uses it to tell an actively-
// progressing first-time load apart from a genuinely stalled one, so a slow
// multi-GB download isn't killed mid-transfer by the startup stall clock.
func (m *Manager) modelDownloadBytes(modelID string) (bytesDone int64, downloading bool) {
	m.downloadsMu.RLock()
	defer m.downloadsMu.RUnlock()
	d, ok := m.activeDownloads[modelID]
	if !ok {
		return 0, false
	}
	return d.BytesDone, true
}

// modelIDForFilename reverse-maps a GGUF filename to the canonical model ID
// in NativeModels. Used by the download path to register progress under the
// public-facing model id (e.g. "qwen3-8b-reasoning") rather than the raw
// GGUF basename (e.g. "Qwen3-8B-Q4_K_M.gguf"). Returns "" when nothing
// matches — caller should fall back to using the basename so progress is
// at least observable, just not tied to a catalog entry.
func modelIDForFilename(filename string) string {
	base := filepath.Base(filename)
	for id, cfg := range NativeModels {
		if cfg.FileName == base {
			return id
		}
	}
	return ""
}

// ModelFileDownloaded reports whether the GGUF for a given native model id is
// already on disk AND has a valid GGUF magic header. Used by the heartbeat
// advertiser to mark models with a `native-model-ready:` token (in addition
// to the always-emitted `model:` capability token) so the hub can prefer
// warm nodes for scheduling.
//
// Cheap: just stats the file and reads the first 4 bytes — no full-file
// hash. False positives (advertising a file that llama-server later
// fails to load for some non-magic reason) are still possible but
// vastly less likely than with a bare size check, which would happily
// approve an 11 GB partial download.
func (m *Manager) ModelFileDownloaded(modelID string) bool {
	cfg, ok := NativeModels[strings.TrimSpace(modelID)]
	if !ok || cfg.FileName == "" {
		return false
	}
	modelPath := filepath.Join(m.dataDir, "models", cfg.FileName)
	info, err := os.Stat(modelPath)
	if err != nil || info.IsDir() {
		return false
	}
	// 1 MiB is well below the smallest real GGUF in the registry
	// (TinyLlama Q4_K_M is ~700 MB), so anything smaller is clearly junk.
	if info.Size() <= 1<<20 {
		return false
	}
	// Magic-byte check — catches truncated downloads, CDN error pages
	// that returned 200 with HTML, etc. validateGGUF reads just 4 bytes.
	return validateGGUF(modelPath) == nil
}

// ReadyNativeChatModels returns the subset of SupportedNativeChatModels whose
// GGUF is already on disk and ready to serve without a download stall. This
// is what should be advertised to operators / surfaced in the dashboard's
// "this node is ready for X" view; the broader SupportedNativeChatModels list
// stays the source of truth for hardware eligibility.
func (m *Manager) ReadyNativeChatModels(vramBytes uint64) []string {
	supported := SupportedNativeChatModels(vramBytes)
	ready := make([]string, 0, len(supported))
	for _, id := range supported {
		if m.ModelFileDownloaded(id) {
			ready = append(ready, id)
		}
	}
	return ready
}

// PrewarmEligibleModels downloads a small selected set of GGUFs in the
// background so the common smoke-test/default paths are warm without filling
// an operator's disk with every large hardware-eligible model. Downloads are
// sequential to avoid disk thrashing + bandwidth contention on the operator's
// link.
//
// Selection:
//   - default/lean: tinyllama, ryvion-llama-3.2-3b, qwen3-8b-reasoning
//   - RYV_MODEL_PREWARM_MODE=all: every hardware-eligible chat model
//   - RYV_MODEL_PREWARM_MODE=off: no startup prewarm
//   - RYV_PREWARM_MODELS=a,b,c: exactly those supported model IDs
//
// Safe to call multiple times — already-downloaded files are skipped via the
// os.Stat check in the existing download loop. Designed to run in a
// goroutine fired at node-agent startup:
//
//	go infMgr.PrewarmEligibleModels(ctx, caps.VRAMBytes)
//
// Cancellation propagates through ctx; the download loop polls for it
// between chunks.
func (m *Manager) PrewarmEligibleModels(ctx context.Context, vramBytes uint64) {
	supported := SupportedNativeChatModels(vramBytes)
	supported = selectPrewarmModels(supported, os.Getenv)
	if len(supported) == 0 {
		slog.Info("prewarm: no models selected")
		return
	}
	modelsDir := filepath.Join(m.dataDir, "models")
	if err := os.MkdirAll(modelsDir, 0o755); err != nil {
		slog.Warn("prewarm: cannot create models dir", "error", err)
		return
	}
	for _, id := range supported {
		if ctx.Err() != nil {
			return
		}
		cfg, ok := NativeModels[id]
		if !ok || cfg.FileName == "" {
			continue
		}
		modelPath := filepath.Join(modelsDir, cfg.FileName)
		// "Already on disk" = exists + valid GGUF magic. A partial /
		// corrupt file from a previously interrupted download must be
		// deleted here, otherwise the inference loop later tries to
		// load it and llama-server keeps failing until the 5-min
		// EnsureModel deadline fires (the "timeout waiting for X to
		// start" error operators were hitting). downloadModelFile
		// also validates after fetch, but doing it here avoids the
		// extra round-trip when the file is already good.
		if info, err := os.Stat(modelPath); err == nil && !info.IsDir() && info.Size() > 1<<20 {
			if vErr := validateGGUF(modelPath); vErr == nil {
				continue // already on disk + valid
			} else {
				slog.Warn("prewarm: discarding invalid GGUF, will re-download",
					"model", id, "path", modelPath, "error", vErr)
				_ = os.Remove(modelPath)
			}
		}
		if err := checkDiskSpace(m.dataDir); err != nil {
			slog.Warn("prewarm: disk space check failed, aborting prewarm",
				"model", id, "error", err)
			return
		}
		downloadURL := m.modelDownloadURL(cfg)
		slog.Info("prewarming model", "model", id, "url", redactDownloadURL(downloadURL))
		if err := m.downloadModelFile(ctx, downloadURL, modelPath); err != nil {
			slog.Warn("prewarm: model download failed (will retry on first job)",
				"model", id, "error", err)
			// Don't return — try the next model. The failed one will be
			// retried on demand from the existing job-execution path.
			continue
		}
		slog.Info("prewarmed model", "model", id)
		// Vision projector (mmproj), if any — same opportunistic prewarm.
		if cfg.MmprojURL != "" && cfg.MmprojFileName != "" {
			mmprojPath := filepath.Join(modelsDir, cfg.MmprojFileName)
			if info, err := os.Stat(mmprojPath); err == nil && !info.IsDir() && info.Size() > 1<<20 {
				continue
			}
			if err := m.downloadModelFile(ctx, m.mmprojDownloadURL(cfg), mmprojPath); err != nil {
				slog.Warn("prewarm: mmproj download failed (model will run text-only until retry)",
					"model", id, "error", err)
				_ = os.Remove(mmprojPath)
			}
		}
	}
}

func meetsModelVRAMRequirement(vramBytes, minVRAMBytes uint64) bool {
	if minVRAMBytes == 0 {
		return true
	}
	if vramBytes >= minVRAMBytes {
		return true
	}
	if minVRAMBytes <= modelVRAMReserveTolerance {
		return false
	}
	return vramBytes+modelVRAMReserveTolerance >= minVRAMBytes
}

// platformServerURL returns the correct llama.cpp release URL for the current OS/arch.
func platformServerURL() string {
	const base = "https://github.com/ggml-org/llama.cpp/releases/download/b9180/"
	switch runtime.GOOS + "/" + runtime.GOARCH {
	case "darwin/arm64":
		return base + "llama-b9180-bin-macos-arm64.tar.gz"
	case "darwin/amd64":
		return base + "llama-b9180-bin-macos-x64.tar.gz"
	case "linux/amd64":
		// CUDA 12 build for Linux GPU nodes
		return base + "llama-b9180-bin-ubuntu-x64.tar.gz"
	case "linux/arm64":
		return base + "llama-b9180-bin-ubuntu-arm64.tar.gz"
	case "windows/amd64":
		accelerator := strings.TrimSpace(os.Getenv("RYV_LLAMA_CPP_WINDOWS_ACCELERATOR"))
		if accelerator == "" {
			accelerator = strings.TrimSpace(os.Getenv("RYV_WINDOWS_LLAMA_ACCELERATOR"))
		}
		accelerator = strings.ToLower(accelerator)
		if accelerator == "vulkan" {
			return base + "llama-b9180-bin-win-vulkan-x64.zip"
		}
		if accelerator == "cuda" || accelerator == "nvidia" {
			return base + "llama-b9180-bin-win-cuda-12.4-x64.zip"
		}
		if _, err := exec.LookPath("nvidia-smi"); err == nil {
			return base + "llama-b9180-bin-win-cuda-12.4-x64.zip"
		}
		return base + "llama-b9180-bin-win-vulkan-x64.zip"
	case "windows/arm64":
		return base + "llama-b9180-bin-win-cpu-arm64.zip"
	default:
		// Windows or unsupported — caller should check
		return ""
	}
}

// NativeRuntimeAvailable reports whether this host can run the bundled native
// inference server with the current defaults.
func NativeRuntimeAvailable() bool {
	if strings.TrimSpace(os.Getenv("RYV_SERVER_URL")) != "" {
		return true
	}
	return platformServerURL() != ""
}

// BlockerReason is a short, dashboard-safe token explaining why the native
// inference path is not currently ready. Operators see these via the
// native-inference-blocker:<reason> token in the heartbeat StatusMessage.
type BlockerReason string

const (
	BlockerNone                BlockerReason = ""
	BlockerNotInstalled        BlockerReason = "not-installed"
	BlockerBinaryMissing       BlockerReason = "binary-missing"
	BlockerStartupTimeout      BlockerReason = "health-check-timeout"
	BlockerProcessFailed       BlockerReason = "process-failed"
	BlockerModelDownloadFail   BlockerReason = "model-download-failed"
	BlockerDiskSpace           BlockerReason = "disk-space"
	BlockerStarting            BlockerReason = "starting"
	BlockerNotStarted          BlockerReason = "not-started"
	BlockerPlatformUnsupported BlockerReason = "platform-unsupported"
)

type Manager struct {
	dataDir           string
	port              string
	threads           string
	gpuLayers         string
	ctxSize           string
	serverURL         string
	serverURLExplicit bool
	serverPath        string
	hubURL            string
	nodeToken         func(int64) string
	// startupStallTimeout bounds how long EnsureModel waits with NO forward
	// progress (model not yet loaded AND no download bytes arriving) before it
	// declares "timeout waiting to start". 0 means use the 300s default. It is
	// a stall clock, not a total-time cap: an actively-downloading GGUF keeps
	// resetting it (see EnsureModel). Overridable so tests don't wait minutes.
	startupStallTimeout time.Duration

	mu              sync.RWMutex
	healthy         bool
	blockerReason   BlockerReason
	cmd             *exec.Cmd
	cancel          context.CancelFunc
	activeModelName string
	activeModelPath string
	activeModelMode ModelMode
	speculative     streamingSpeculativeLaunch

	// activeDownloads tracks in-flight GGUF downloads keyed by model ID.
	// Populated by downloadModelFile when it starts a transfer, cleared
	// on success or failure. Surfaced to operators (and through them to
	// buyers via heartbeat) so the cold-start UI can show real download
	// progress instead of a generic spinner. Concurrency-safe via its
	// own mutex — independent of `mu` to avoid lock-ordering issues.
	downloadsMu     sync.RWMutex
	activeDownloads map[string]*ActiveDownload

	// downloadLocks serializes downloads of the same GGUF so two
	// concurrent callers — typically the background prewarmer and the
	// on-demand inference loop — don't both `os.Create(dst+".tmp")` and
	// truncate each other's bytes. Race symptom: the progress bar
	// flip-flops between two competing writers' offsets, sometimes
	// decreasing. Per-model lock means different models can still
	// download in parallel; only same-model callers serialize. sync.Map
	// gives us atomic LoadOrStore for the lock-creation race.
	downloadLocks sync.Map
}

// ActiveDownload describes an in-flight GGUF transfer. Snapshot semantics:
// callers receive a value copy via ActiveDownloads(), so reading is
// allocation-free past the slice and safe to render in tight UI loops.
type ActiveDownload struct {
	ModelID    string
	BytesDone  int64
	BytesTotal int64
	StartedAt  time.Time
}

func (m *Manager) SetHubAuth(hubURL string, nodeAuthToken func(int64) string) {
	if m == nil {
		return
	}
	m.hubURL = strings.TrimRight(strings.TrimSpace(hubURL), "/")
	m.nodeToken = nodeAuthToken
}

func New(dataDir string) *Manager {
	if dataDir == "" {
		home, _ := os.UserHomeDir()
		dataDir = filepath.Join(home, ".ryvion")
	}
	port := envOr("RYV_INFERENCE_PORT", defaultPort)
	serverURL, serverURLExplicit := envOrExplicit("RYV_SERVER_URL", platformServerURL())
	mgr := &Manager{
		activeDownloads:   map[string]*ActiveDownload{},
		dataDir:           dataDir,
		port:              port,
		threads:           envOr("RYV_INFERENCE_THREADS", defaultThreads),
		gpuLayers:         envOr("RYV_GPU_LAYERS", defaultGPULayers),
		ctxSize:           envOr("RYV_CTX_SIZE", defaultCtxSize),
		serverURL:         serverURL,
		serverURLExplicit: serverURLExplicit,
		activeModelName:   "ryvion-llama-3.2-3b",
		blockerReason:     BlockerNotStarted,
	}
	if serverURL == "" {
		mgr.blockerReason = BlockerPlatformUnsupported
	}
	return mgr
}

// amdSmokeTestPassed tracks whether the AMD GPU dry-run succeeded.
var amdSmokeTestPassed bool

func (m *Manager) Start(ctx context.Context) error {
	if m.serverURL == "" {
		slog.Info("inference manager: no llama-server binary available for this platform, skipping",
			"os", runtime.GOOS, "arch", runtime.GOARCH)
		m.setBlockerReason(BlockerPlatformUnsupported)
		<-ctx.Done()
		return ctx.Err()
	}

	m.setBlockerReason(BlockerStarting)

	binDir := filepath.Join(m.dataDir, "bin")
	modelDir := filepath.Join(m.dataDir, "models")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		m.setBlockerReason(BlockerDiskSpace)
		return fmt.Errorf("create bin dir: %w", err)
	}
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		m.setBlockerReason(BlockerDiskSpace)
		return fmt.Errorf("create model dir: %w", err)
	}

	serverPath := filepath.Join(binDir, serverBinaryName())
	installServer, installReason := shouldInstallServer(serverPath, serverSourceMarkerPath(serverPath), m.serverURL, m.serverURLExplicit)
	if installServer {
		if err := checkDiskSpace(m.dataDir); err != nil {
			m.setBlockerReason(BlockerDiskSpace)
			return fmt.Errorf("disk space check: %w", err)
		}
		slog.Info("downloading llama-server", "url", m.serverURL, "reason", installReason)
		if err := downloadAndExtractServer(ctx, m.serverURL, serverPath); err != nil {
			m.setBlockerReason(BlockerBinaryMissing)
			return fmt.Errorf("download llama-server: %w", err)
		}
		if err := writeServerSourceMarker(serverSourceMarkerPath(serverPath), m.serverURL); err != nil {
			slog.Warn("failed to write llama-server source marker", "path", serverSourceMarkerPath(serverPath), "error", err)
		}
		slog.Info("llama-server downloaded", "path", serverPath)
	}
	m.serverPath = serverPath

	// AMD GPU smoke test — verify ROCm compatibility before accepting work
	if _, err := os.Stat("/dev/kfd"); err == nil {
		slog.Info("AMD GPU detected, running compatibility smoke test")
		if err := m.runAMDSmokeTest(ctx, modelDir); err != nil {
			slog.Error("AMD GPU smoke test FAILED — GPU inference may not work",
				"error", err,
				"hint", "check ROCm version and gfx architecture compatibility")
			// Don't return error — still allow CPU inference and native mode
		} else {
			amdSmokeTestPassed = true
			slog.Info("AMD GPU smoke test PASSED — ROCm inference is operational")
		}
	}

	// Start server with auto-restart
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		m.mu.RLock()
		currentModel := m.activeModelName
		customPath := m.activeModelPath
		m.mu.RUnlock()

		var modelPath string
		if customPath != "" && strings.HasSuffix(customPath, ".gguf") {
			// Custom model — path was already set by EnsureCustomModel
			if _, err := os.Stat(customPath); err == nil {
				modelPath = customPath
			}
		}
		var activeMode ModelMode
		if modelPath == "" {
			// Native registry model
			cfg, ok := NativeModels[currentModel]
			if !ok {
				currentModel = "ryvion-llama-3.2-3b"
				cfg = NativeModels[currentModel]
			}
			activeMode = cfg.Mode
			modelPath = filepath.Join(modelDir, cfg.FileName)
			// Download trigger: file missing OR present but failing GGUF
			// magic-byte validation. The second case catches truncated /
			// corrupt files left behind by a previously interrupted
			// download — without this check llama-server would start
			// with a bad GGUF and crash, the loop would relaunch, and
			// EnsureModel would eventually return "timeout waiting for
			// X to start" to the buyer. downloadModelFile is idempotent
			// + per-model locked, so re-entering it is safe.
			fileMissing := false
			if info, statErr := os.Stat(modelPath); statErr != nil || info.IsDir() || info.Size() <= 1<<20 {
				fileMissing = true
			} else if vErr := validateGGUF(modelPath); vErr != nil {
				slog.Warn("on-disk GGUF failed magic-byte validation, removing + re-downloading",
					"model", currentModel, "path", modelPath, "error", vErr)
				_ = os.Remove(modelPath)
				fileMissing = true
			}
			if fileMissing {
				if err := checkDiskSpace(m.dataDir); err != nil {
					slog.Error("disk space check failed before model download", "error", err)
					m.setBlockerReason(BlockerDiskSpace)
					time.Sleep(5 * time.Second)
					continue
				}
				downloadURL := m.modelDownloadURL(cfg)
				slog.Info("downloading model", "model", currentModel, "url", redactDownloadURL(downloadURL))
				if err := m.downloadModelFile(ctx, downloadURL, modelPath); err != nil {
					slog.Error("failed to download model", "error", err)
					m.setBlockerReason(BlockerModelDownloadFail)
					time.Sleep(5 * time.Second)
					continue
				}
				slog.Info("model downloaded", "path", modelPath)
			}
			// Vision projector (mmproj) — auto-downloaded alongside the
			// GGUF for vision-capable models. ~2 GB on disk for Gemma 4
			// / Nemotron Omni. Failure is non-fatal: the model still
			// runs in text-only mode without the projector, and the
			// playground's OCR fallback handles image attachments.
			if cfg.MmprojURL != "" && cfg.MmprojFileName != "" {
				mmprojPath := filepath.Join(modelDir, cfg.MmprojFileName)
				if _, err := os.Stat(mmprojPath); os.IsNotExist(err) {
					slog.Info("downloading vision projector",
						"model", currentModel,
						"mmproj", cfg.MmprojFileName,
						"url", redactDownloadURL(m.mmprojDownloadURL(cfg)),
					)
					if err := m.downloadModelFile(ctx, m.mmprojDownloadURL(cfg), mmprojPath); err != nil {
						slog.Warn("vision projector download failed; model will run text-only",
							"model", currentModel,
							"error", err,
						)
						// Cleanup partial file so a subsequent retry is clean.
						_ = os.Remove(mmprojPath)
					} else {
						slog.Info("vision projector downloaded", "path", mmprojPath)
					}
				}
			}
		}

		m.mu.Lock()
		m.activeModelPath = modelPath
		m.activeModelMode = activeMode
		m.mu.Unlock()

		slog.Info("starting llama-server", "port", m.port, "model", modelPath, "mode", string(activeMode))
		if err := m.runServer(ctx); err != nil {
			slog.Warn("llama-server exited", "error", err)
			m.setBlockerReason(BlockerProcessFailed)
		}
		m.setHealthy(false)
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(3 * time.Second):
			slog.Info("restarting llama-server")
		}
	}
}

func (m *Manager) Stop() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.stopServerLocked()
	m.healthy = false
	m.cancel = nil
	m.cmd = nil
}

func (m *Manager) Healthy() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.healthy
}

// BlockerReason returns a short token explaining why Healthy() is false.
// Returns BlockerNone when the manager is healthy. The reason feeds the
// native-inference-blocker:<reason> heartbeat token so operators can see a
// concrete cause (binary missing, health check timeout, etc.) on the dashboard.
func (m *Manager) BlockerReason() BlockerReason {
	if m == nil {
		return BlockerNotStarted
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.healthy {
		return BlockerNone
	}
	if m.blockerReason == BlockerNone {
		return BlockerNotStarted
	}
	return m.blockerReason
}

func (m *Manager) ServerURL() string {
	return "http://localhost:" + m.port
}

func (m *Manager) ModelName() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.activeModelName
}

// SelectModelForNextStart updates the native model that Start should launch
// when the manager is currently idle. If a different model is already running,
// it stops that process so the normal Start/EnsureModel path can launch the
// requested model directly instead of booting the old/default model first.
func (m *Manager) SelectModelForNextStart(modelName string) error {
	if m == nil {
		return fmt.Errorf("inference manager is not available")
	}
	modelName = strings.TrimSpace(modelName)
	if modelName == "" {
		modelName = "ryvion-llama-3.2-3b"
	}
	cfg, ok := NativeModels[modelName]
	if !ok {
		return fmt.Errorf("model %s not supported in native registry", modelName)
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if m.activeModelName == modelName {
		if !m.healthy && m.cancel == nil {
			m.activeModelPath = ""
			m.activeModelMode = cfg.Mode
			m.blockerReason = BlockerStarting
		}
		return nil
	}
	m.activeModelName = modelName
	m.activeModelPath = ""
	m.activeModelMode = cfg.Mode
	m.healthy = false
	m.blockerReason = BlockerStarting
	m.stopServerLocked()
	return nil
}

func (m *Manager) EnsureModel(ctx context.Context, modelName string) error {
	modelName = strings.TrimSpace(modelName)
	if modelName == "" {
		modelName = "ryvion-llama-3.2-3b"
	}
	cfg, ok := NativeModels[modelName]
	if !ok {
		return fmt.Errorf("model %s not supported in native registry", modelName)
	}

	m.mu.RLock()
	current := m.activeModelName
	m.mu.RUnlock()

	if current != modelName {
		slog.Info("switching native model", "from", current, "to", modelName)

		m.mu.Lock()
		m.activeModelName = modelName
		m.activeModelPath = ""
		m.activeModelMode = cfg.Mode
		m.healthy = false
		m.blockerReason = BlockerStarting
		m.stopServerLocked()
		m.mu.Unlock()
	}

	restartRequested := current != modelName
	stallTimeout := m.startupStallTimeout
	if stallTimeout <= 0 {
		stallTimeout = 300 * time.Second
	}
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	var lastMismatchIDs []string

	// Stall clock, not a total-time cap. A first-time load of a large reasoning
	// GGUF (qwen3 ~5GB, gpt-oss ~12GB) on a slow / home uplink can take well
	// over the bare startup window — on a MacBook this is exactly the
	// "first-time model load … timeout waiting to start" failure. Treat active
	// download progress as liveness: while bytes keep arriving we push
	// lastProgressAt forward, so the model is never declared stuck mid-transfer.
	// The surrounding job context (30 min for streaming inference) remains the
	// real upper bound; this only catches a genuine stall (no model loaded AND
	// no download progress for stallTimeout).
	lastProgressAt := time.Now()
	var lastDownloadBytes int64 = -1

	for {
		probeCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
		ok, ids, err := m.loadedModelMatches(probeCtx, modelName)
		cancel()
		if ok {
			return nil
		} else if err == nil && len(ids) > 0 && !restartRequested {
			lastMismatchIDs = append(lastMismatchIDs[:0], ids...)
			slog.Warn("native model state mismatch; restarting llama-server",
				"requested_model", modelName,
				"served_models", strings.Join(ids, ","),
			)
			m.mu.Lock()
			m.healthy = false
			m.blockerReason = BlockerStarting
			m.stopServerLocked()
			m.mu.Unlock()
			restartRequested = true
		} else if err == nil && len(ids) > 0 {
			lastMismatchIDs = append(lastMismatchIDs[:0], ids...)
		}

		// Reset the stall clock whenever the GGUF download advances.
		if done, downloading := m.modelDownloadBytes(modelName); downloading && done != lastDownloadBytes {
			lastDownloadBytes = done
			lastProgressAt = time.Now()
		}
		if time.Since(lastProgressAt) >= stallTimeout {
			m.setBlockerReason(BlockerStartupTimeout)
			if len(lastMismatchIDs) > 0 {
				return fmt.Errorf("timeout waiting for %s to start; last loaded model mismatch: requested %s (%s), llama-server reports %s", modelName, modelName, cfg.FileName, strings.Join(lastMismatchIDs, ","))
			}
			return fmt.Errorf("timeout waiting for %s to start", modelName)
		}

		select {
		case <-ctx.Done():
			if len(lastMismatchIDs) > 0 {
				return fmt.Errorf("model %s not ready before context deadline; last loaded model mismatch: requested %s (%s), llama-server reports %s: %w", modelName, modelName, cfg.FileName, strings.Join(lastMismatchIDs, ","), ctx.Err())
			}
			return ctx.Err()
		case <-ticker.C:
		}
	}
}

// EnsureCustomModel downloads a custom GGUF model from a URL and hot-swaps the server to use it.
func (m *Manager) EnsureCustomModel(ctx context.Context, modelName, modelURL string) error {
	m.mu.RLock()
	current := m.activeModelName
	m.mu.RUnlock()

	// Download model if not already cached
	modelsDir := filepath.Join(m.dataDir, "models")
	os.MkdirAll(modelsDir, 0755)
	// Use a hash of the URL as the filename to cache per-URL
	h := sha256.Sum256([]byte(modelURL))
	fileName := modelName + "-" + hex.EncodeToString(h[:8]) + ".gguf"
	modelPath := filepath.Join(modelsDir, fileName)

	if _, err := os.Stat(modelPath); err != nil {
		if err := checkDiskSpace(m.dataDir); err != nil {
			return fmt.Errorf("disk space check: %w", err)
		}
		slog.Info("downloading custom model", "name", modelName, "path", modelPath)
		if err := downloadFile(ctx, modelURL, modelPath); err != nil {
			return fmt.Errorf("download custom model: %w", err)
		}
		if err := validateGGUF(modelPath); err != nil {
			os.Remove(modelPath)
			return fmt.Errorf("invalid custom model file: %w", err)
		}
		slog.Info("custom model downloaded", "name", modelName, "path", modelPath)
	}

	// If already loaded, skip restart
	if current == fileName {
		return nil
	}

	slog.Info("switching to custom model", "from", current, "to", modelName)

	m.mu.Lock()
	m.activeModelName = fileName
	m.activeModelPath = modelPath
	m.healthy = false
	m.blockerReason = BlockerStarting
	if m.cancel != nil {
		m.cancel()
	}
	m.mu.Unlock()

	// Wait for server to become healthy with the new model
	for i := 0; i < 300; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(1 * time.Second):
			m.mu.RLock()
			hc := m.healthy
			m.mu.RUnlock()
			if hc {
				return nil
			}
		}
	}
	return fmt.Errorf("timeout waiting for custom model %s to start", modelName)
}

func (m *Manager) stopServerLocked() {
	if m.cancel != nil {
		m.cancel()
	}
	if m.cmd != nil && m.cmd.Process != nil {
		_ = m.cmd.Process.Kill()
	}
	m.cancel = nil
	m.cmd = nil
}

func (m *Manager) setHealthy(v bool) {
	m.mu.Lock()
	m.healthy = v
	if v {
		m.blockerReason = BlockerNone
	}
	m.mu.Unlock()
}

// setBlockerReason records the most recent reason the native inference path
// is blocked. Callers should set this whenever they fail to advance the
// startup pipeline (binary missing, model download failure, health timeout,
// process crash, etc.) so BlockerReason() reflects an actionable token.
func (m *Manager) setBlockerReason(reason BlockerReason) {
	if m == nil {
		return
	}
	m.mu.Lock()
	m.blockerReason = reason
	if reason != BlockerNone {
		m.healthy = false
	}
	m.mu.Unlock()
}

func (m *Manager) runServer(ctx context.Context) error {
	m.setHealthy(false)
	m.setStreamingSpeculativeLaunch(streamingSpeculativeLaunch{})

	m.mu.RLock()
	modelPath := m.activeModelPath
	port := m.port
	m.mu.RUnlock()

	// Use the host-native llama-server by default. Containerized inference is
	// opt-in because the generic OCI path has fixed memory/device constraints
	// that are a poor fit for large local GGUFs such as Nemotron 30B.
	if m.useContainerizedInference() {
		return m.runServerContainerized(ctx, modelPath, port)
	}

	// Fallback: native mode
	return m.runServerNative(ctx, modelPath, port)
}

// useContainerizedInference reports whether a managed OCI backend is available for sandboxed inference.
// On Windows, always prefer native mode — native inference is more reliable than GPU passthrough through the container backend
// containers is unreliable (OOM kills, exit 137). The native llama-server.exe with CUDA works better.
func (m *Manager) useContainerizedInference() bool {
	if envBool("RYV_NATIVE_INFERENCE_ONLY") {
		return false
	}
	if !envBool("RYV_CONTAINERIZED_NATIVE_INFERENCE") {
		return false
	}
	if runtime.GOOS == "windows" {
		return false
	}
	backend, err := runtimeexec.ResolveBackendCommand(runtime.GOOS, os.Getenv)
	if err != nil {
		return false
	}
	cmd := exec.Command(backend, "info")
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard
	return cmd.Run() == nil
}

// runServerContainerized runs llama-server inside the container backend with GPU passthrough.
func (m *Manager) runServerContainerized(ctx context.Context, modelPath, port string) error {
	serverCtx, cancel := context.WithCancel(ctx)
	m.mu.Lock()
	m.cancel = cancel
	m.mu.Unlock()
	defer cancel()

	modelDir := filepath.Dir(modelPath)
	modelFile := filepath.Base(modelPath)

	image := os.Getenv("RYV_INFERENCE_IMAGE")
	if image == "" {
		image = "ghcr.io/ggml-org/llama.cpp:server"
	}

	args := []string{
		"run", "--rm",
		"--name", "ryvion-inference",
		// Security constraints
		"--security-opt=no-new-privileges:true",
		"--cap-drop=ALL",
		"--memory=8g",
		"--pids-limit=256",
	}

	// Detect GPU type and add passthrough
	if _, err := exec.Command("nvidia-smi").Output(); err == nil {
		args = append(args, "--gpus", "all")
	} else if _, err := os.Stat("/dev/kfd"); err == nil {
		// ROCm (AMD) GPU
		args = append(args, "--device=/dev/kfd", "--device=/dev/dri", "--group-add=video")
	}

	// Mount model directory read-only, expose the port
	args = append(args,
		"-v", modelDir+":/models:ro",
		"-p", port+":"+port,
		image,
		"--model", "/models/"+modelFile,
		"--port", port,
		"--host", "0.0.0.0",
		"--threads", m.threads,
		"--ctx-size", m.ctxSize,
	)

	// GPU layers (skip on macOS where Metal isn't available inside containers)
	if runtime.GOOS != "darwin" {
		args = append(args, "--n-gpu-layers", m.gpuLayers)
	}

	// Vision projector auto-detection. Same logic as the native path —
	// if a sibling .mmproj file exists, mount the model dir read-only
	// (already done above) and pass --mmproj using the in-container
	// path so the model can answer about images.
	if mmprojPath := findVisionProjectorFor(modelPath); mmprojPath != "" {
		mmprojFile := filepath.Base(mmprojPath)
		args = append(args, "--mmproj", "/models/"+mmprojFile)
		slog.Info("vision projector found — enabling multimodal in container",
			"model", modelFile,
			"mmproj", mmprojFile,
		)
	}

	slog.Info("starting containerized llama-server",
		"image", image,
		"model", modelFile,
		"port", port,
	)

	ociExec, err := runtimeexec.ResolveExecutor(runtime.GOOS, os.Getenv)
	if err != nil {
		slog.Warn("containerized inference executor unavailable, falling back to native", "error", err)
		return m.runServerNative(ctx, modelPath, port)
	}
	execArgs := append(append([]string{}, ociExec.PrefixArgs...), args...)
	cmd := exec.CommandContext(serverCtx, ociExec.Command, execArgs...)
	// Send container output to a log file
	logPath := filepath.Join(m.dataDir, "llama-server.log")
	if logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644); err == nil {
		cmd.Stdout = logFile
		cmd.Stderr = logFile
		defer logFile.Close()
	} else {
		cmd.Stdout = io.Discard
		cmd.Stderr = io.Discard
	}

	m.mu.Lock()
	m.cmd = cmd
	m.mu.Unlock()

	if err := cmd.Start(); err != nil {
		slog.Warn("containerized inference failed to start, falling back to native", "error", err)
		return m.runServerNative(ctx, modelPath, port)
	}

	go m.healthLoop(serverCtx)

	waitErr := cmd.Wait()

	// Clean up container on exit (may already be removed by --rm, but be safe)
	cleanupArgs := append(append([]string{}, ociExec.PrefixArgs...), "rm", "-f", "ryvion-inference")
	cleanup := exec.Command(ociExec.Command, cleanupArgs...)
	cleanup.Stdout = io.Discard
	cleanup.Stderr = io.Discard
	cleanup.Run()

	if waitErr != nil {
		slog.Error("containerized llama-server exited with error",
			"error", waitErr,
			"model", modelPath,
			"log_file", logPath,
		)
	}
	return waitErr
}

// prewarmPriority maps a model id to a small integer "download me first"
// rank. Lower = earlier. The intent: the smallest + most-frequently-used
// chat models come up within ~1 min of node boot; the giant reasoning /
// multimodal models can take their time.
//
// Ordering rationale (in increasing rank):
//
//	0 tinyllama          — 700 MB, fastest possible smoke-test path
//	1 ryvion-llama-3.2-3b — 2 GB, the default chat model
//	2 phi-4              — 9 GB, popular for code/reasoning, fits 8 GB GPUs
//	3 qwen3-8b-reasoning — 5 GB, primary reasoning model
//	4 gpt-oss-20b        — 11 GB, secondary reasoning model
//	5 gemma-4-26b-a4b-it — 17 GB, vision + bigger, slower
//	6 nemotron-3-...     — 17 GB, vision + MoE, slower
var prewarmPriority = map[string]int{
	"tinyllama":                    0,
	"ryvion-llama-3.2-3b":          1,
	"phi-4":                        2,
	"qwen3-8b-reasoning":           3,
	"gpt-oss-20b":                  4,
	"gemma-4-26b-a4b-it":           5,
	"nemotron-3-nano-omni-30b-a3b": 6,
	"nomic-embed-text-v1.5":        7, // embedding model, only matters for RAG
}

var defaultPrewarmModels = map[string]struct{}{
	"tinyllama":           {},
	"ryvion-llama-3.2-3b": {},
	"qwen3-8b-reasoning":  {},
}

func selectPrewarmModels(supported []string, getenv func(string) string) []string {
	if getenv == nil {
		getenv = os.Getenv
	}
	explicit := strings.TrimSpace(getenv("RYV_PREWARM_MODELS"))
	if explicit != "" {
		return selectExplicitPrewarmModels(supported, explicit)
	}

	mode := strings.ToLower(strings.TrimSpace(getenv("RYV_MODEL_PREWARM_MODE")))
	switch mode {
	case "all":
		return sortPrewarmByPriority(supported)
	case "off", "none", "disabled", "0", "false":
		return nil
	case "", "default", "lean", "small":
		return selectDefaultPrewarmModels(supported)
	default:
		slog.Warn("prewarm: unknown mode, using lean default", "mode", mode)
		return selectDefaultPrewarmModels(supported)
	}
}

func selectDefaultPrewarmModels(supported []string) []string {
	sorted := sortPrewarmByPriority(supported)
	out := make([]string, 0, len(defaultPrewarmModels))
	for _, id := range sorted {
		if _, ok := defaultPrewarmModels[id]; ok {
			out = append(out, id)
		}
	}
	return out
}

func selectExplicitPrewarmModels(supported []string, raw string) []string {
	supportedSet := make(map[string]struct{}, len(supported))
	for _, id := range supported {
		supportedSet[strings.ToLower(strings.TrimSpace(id))] = struct{}{}
	}
	seen := make(map[string]struct{})
	out := make([]string, 0)
	for _, id := range splitPrewarmModelList(raw) {
		if _, ok := supportedSet[id]; !ok {
			continue
		}
		if _, dup := seen[id]; dup {
			continue
		}
		seen[id] = struct{}{}
		out = append(out, id)
	}
	return out
}

func splitPrewarmModelList(raw string) []string {
	parts := strings.FieldsFunc(raw, func(r rune) bool {
		return r == ',' || r == ';' || r == ' ' || r == '\n' || r == '\t'
	})
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		id := strings.ToLower(strings.TrimSpace(part))
		if id != "" {
			out = append(out, id)
		}
	}
	return out
}

// sortPrewarmByPriority returns a copy of `ids` sorted by prewarmPriority
// (lower first). Unknown ids land after all known ones, in their original
// alphabetical order, so future additions to NativeModels still get
// prewarmed — just at the tail of the queue.
func sortPrewarmByPriority(ids []string) []string {
	out := make([]string, len(ids))
	copy(out, ids)
	const unknown = 1000
	sort.SliceStable(out, func(i, j int) bool {
		pi, ok := prewarmPriority[out[i]]
		if !ok {
			pi = unknown
		}
		pj, ok := prewarmPriority[out[j]]
		if !ok {
			pj = unknown
		}
		if pi != pj {
			return pi < pj
		}
		return out[i] < out[j]
	})
	return out
}

// streamingFamilyHintForModel returns "qwen", "gpt-oss", "deepseek", "llama",
// "phi" etc. based on the GGUF filename. Used only to gate
// reasoning-related llama-server flags in the streaming path. Kept in
// this package (rather than imported from internal/runtimes/llamacpp) to
// avoid an import cycle and to make the streaming launcher self-contained.
func streamingFamilyHintForModel(modelPath string) string {
	lower := strings.ToLower(filepath.Base(strings.TrimSpace(modelPath)))
	switch {
	case strings.Contains(lower, "qwen"):
		return "qwen"
	case strings.Contains(lower, "gpt-oss"), strings.Contains(lower, "gptoss"):
		return "gpt-oss"
	case strings.Contains(lower, "deepseek"):
		return "deepseek"
	case strings.Contains(lower, "phi"):
		return "phi"
	case strings.Contains(lower, "gemma"):
		return "gemma"
	case strings.Contains(lower, "nemotron"):
		return "nemotron"
	case strings.Contains(lower, "llama"):
		return "llama"
	default:
		return ""
	}
}

// streamingJinjaRequiredForModel reports whether llama-server needs the
// `--jinja` flag to correctly apply this model's chat template — in
// particular to honor the `reasoning_effort` body field and to emit
// `<think>` segments / Harmony channels through the proper output paths.
// Qwen, DeepSeek-R1 and GPT-OSS all need it.
func streamingJinjaRequiredForModel(modelPath string) bool {
	switch streamingFamilyHintForModel(modelPath) {
	case "qwen", "deepseek", "gpt-oss":
		return true
	default:
		return false
	}
}

// streamingReasoningFormatForModel returns the llama-server
// `--reasoning-format` value that routes a model's thinking output into
// the `reasoning_content` SSE delta field. Without this flag the model's
// reasoning lands in `delta.content` with raw `<think>` / Harmony
// markers — the frontend's defensive regex tries to scrape them out, but
// cleanly separating reasoning_content upstream is much more reliable.
//
//   - "deepseek": parses `<think>...</think>` (Qwen3, DeepSeek R1).
//   - "auto":     auto-detect (required for GPT-OSS Harmony per
//     llama.cpp discussions/15396).
func streamingReasoningFormatForModel(modelPath string) string {
	switch streamingFamilyHintForModel(modelPath) {
	case "qwen", "deepseek":
		return "deepseek"
	case "gpt-oss":
		return "auto"
	default:
		return ""
	}
}

// streamingMoECPUOffloadArgsForModel returns extra llama-server flags that keep
// a Mixture-of-Experts model's expert tensors in system RAM instead of VRAM.
// Returns nil for models that fit on the GPU normally.
//
// Nemotron 3 Nano Omni 30B-A3B is a 30B-total / 3B-active MoE whose Q4_K_M GGUF
// is ~24.5 GB on disk — larger than the VRAM of every card in the fleet (24 GB on
// the RTX 3090 / RX 7900 XTX, 16 GB on the 4070 Ti). Passing only
// `--n-gpu-layers 99` asks llama-server to place all ~24.5 GB on the GPU; it
// cannot, so it aborts with a CUDA/ROCm out-of-memory at load. The manager then
// never reaches Healthy() and every chat request fails with the buyer-facing
// "inference manager is not healthy".
//
// `--cpu-moe` keeps the expert weights (the bulk of the model) on the CPU and
// leaves only attention/router/embeddings + KV cache + the vision projector on
// the GPU (~6-8 GB). Only 3B params are active per token, so throughput stays
// usable. This is the standard way to serve a large MoE on limited VRAM and is
// what actually makes the model run on a 16-24 GB card. The flag was added to
// llama.cpp around b6000; the node ships b9180. The operator needs ~24 GB of free
// system RAM for the offloaded experts.
func streamingMoECPUOffloadArgsForModel(modelPath string) []string {
	// Every Nemotron model in the registry is the 30B-A3B Omni MoE. Other MoE
	// models we ship (gpt-oss-20b) fit fully on a 16 GB GPU, so they are left on
	// the GPU for full throughput.
	if streamingFamilyHintForModel(modelPath) == "nemotron" {
		return []string{"--cpu-moe"}
	}
	return nil
}

// runServerNative runs llama-server directly on the host.
func (m *Manager) runServerNative(ctx context.Context, modelPath, port string) error {
	serverCtx, cancel := context.WithCancel(ctx)
	m.mu.Lock()
	m.cancel = cancel
	m.mu.Unlock()
	defer cancel()

	args := []string{
		"--model", modelPath,
		"--port", port,
		"--host", "127.0.0.1",
		"--threads", m.threads,
		"--ctx-size", m.ctxSize,
		"--log-disable",
	}

	// Reasoning-model flags. CRITICAL: without these, Qwen3-reasoning
	// emits its `<think>...</think>` blocks inline in `content` instead
	// of the OpenAI-style `delta.reasoning_content` SSE field — and
	// GPT-OSS's Harmony channel format goes raw. The frontend's
	// cold-start indicator only stops when content OR reasoning_content
	// arrives; without these flags the model spends 30-120s reasoning
	// silently before the first content token, and the buyer sees a
	// frozen UI with no thinking block.
	//
	// The companion sidecar manager (internal/runtimes/llamacpp/manager.go)
	// already gates these flags by family hint. Duplicating the tiny
	// switch here keeps RunStreamingJob — which uses runServerNative
	// directly, NOT the sidecar — in sync. Without this duplication,
	// every streaming Qwen3/GPT-OSS request hangs until cold-start
	// auto-aborts at 5 min.
	if streamingJinjaRequiredForModel(modelPath) {
		args = append(args, "--jinja")
	}
	if rf := streamingReasoningFormatForModel(modelPath); rf != "" {
		args = append(args, "--reasoning-format", rf)
	}

	// Vision projector auto-detection. If a .mmproj file lives alongside
	// the GGUF (same dir, common naming conventions), pass --mmproj so
	// llama-server starts in multimodal mode and accepts OpenAI-style
	// {type:"image_url",...} parts in chat completions. Without this,
	// even a vision-capable model like Gemma 4 26B silently drops
	// image_url parts and answers as if the image weren't there.
	if mmprojPath := findVisionProjectorFor(modelPath); mmprojPath != "" {
		args = append(args, "--mmproj", mmprojPath)
		slog.Info("vision projector found — enabling multimodal",
			"model", modelPath,
			"mmproj", mmprojPath,
		)
	}

	m.mu.RLock()
	mode := m.activeModelMode
	m.mu.RUnlock()
	if mode == ModeEmbedding {
		// --embedding routes the server's /v1/embeddings endpoint at this
		// model. Pooling=mean is the sensible default for encoder-style
		// models like nomic-embed-text / bge.
		args = append(args, "--embedding", "--pooling", "mean")
	}

	speculative := streamingSpeculativeLaunchForModel(modelPath, filepath.Join(m.dataDir, "models"), os.Getenv)
	if len(speculative.Args) > 0 {
		args = append(args, speculative.Args...)
		m.setStreamingSpeculativeLaunch(speculative)
	}

	// GPU offloading: Metal on macOS, CUDA on Linux/Windows
	// --n-gpu-layers=99 offloads all layers to GPU when available.
	// llama.cpp gracefully falls back to CPU if no GPU is detected.
	switch runtime.GOOS {
	case "darwin":
		// Metal acceleration (ARM64 and AMD64 with Metal)
		args = append(args, "--n-gpu-layers", m.gpuLayers)
	case "linux", "windows":
		// CUDA acceleration if available; llama.cpp ignores flag if no GPU
		args = append(args, "--n-gpu-layers", m.gpuLayers)
	}

	// Large Mixture-of-Experts models (Nemotron 3 Nano Omni 30B-A3B) exceed the
	// VRAM of every fleet GPU when fully offloaded — the 24.5 GB Q4_K_M does not
	// fit even a 24 GB card. Keep the expert tensors in system RAM with --cpu-moe
	// so the GPU holds only attention + KV cache + the vision projector. Combined
	// with --n-gpu-layers above, every non-expert layer still runs on the GPU.
	// Without this the launch OOMs at load and the manager reports
	// "inference manager is not healthy".
	if moeArgs := streamingMoECPUOffloadArgsForModel(modelPath); len(moeArgs) > 0 {
		args = append(args, moeArgs...)
		slog.Info("large MoE model — offloading expert tensors to CPU",
			"model", modelPath,
			"flags", moeArgs,
		)
	}

	// Stock upstream llama-server rejects the Ryvion fork's --spec-* flags and
	// exits immediately, crash-looping native streaming for EVERY model (the
	// speculative flags are added on every launch). Probe the binary and strip
	// them when unsupported, falling back to standard decoding. Mirrors the V7
	// sidecar's spec_support.go — this legacy streaming path (used by
	// RunStreamingJob) had not been wired through that guard, which left
	// stock-binary nodes reporting native-inference-blocker:process-failed.
	args = specCompatibleArgs(m.serverPath, args)

	cmd := exec.CommandContext(serverCtx, m.serverPath, args...)
	// Send llama-server output to a log file to avoid mixing with JSON slog.
	logPath := filepath.Join(m.dataDir, "llama-server.log")
	if logFile, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644); err == nil {
		cmd.Stdout = logFile
		cmd.Stderr = logFile
		defer logFile.Close()
	} else {
		cmd.Stdout = io.Discard
		cmd.Stderr = io.Discard
	}
	// Set library path so llama-server finds its shared libs
	binDir := filepath.Dir(m.serverPath)
	env := os.Environ()
	if runtime.GOOS == "windows" {
		// On Windows, colocated CUDA/llama DLLs are resolved through PATH.
		env = append(env, "PATH="+binDir+";"+os.Getenv("PATH"))
	} else {
		env = append(env,
			"DYLD_LIBRARY_PATH="+binDir,
			"LD_LIBRARY_PATH="+binDir,
		)
	}
	cmd.Env = env

	m.mu.Lock()
	m.cmd = cmd
	m.mu.Unlock()

	slog.Info("launching llama-server (native)",
		"binary", m.serverPath,
		"model", modelPath,
		"port", port,
		"threads", m.threads,
		"gpu_layers", m.gpuLayers,
		"ctx_size", m.ctxSize,
		"speculative_method", speculative.Method,
		"os", runtime.GOOS,
		"arch", runtime.GOARCH,
	)

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start llama-server: %w", err)
	}

	// Lower priority so operator workloads (games, etc.) take precedence.
	if cmd.Process != nil {
		setLowPriority(cmd.Process.Pid)
	}

	// Wait for health check to pass
	go m.healthLoop(serverCtx)

	waitErr := cmd.Wait()
	if waitErr != nil {
		slog.Error("llama-server process exited with error",
			"error", waitErr,
			"model", modelPath,
			"log_file", logPath,
		)
	}
	return waitErr
}

func (m *Manager) healthLoop(ctx context.Context) {
	url := m.ServerURL() + "/health"
	client := &http.Client{Timeout: 3 * time.Second}

	// Initial startup: wait up to resolvedStartupTimeout for first healthy response.
	startupBudget := resolvedStartupTimeout()
	deadline := time.After(startupBudget)
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-deadline:
			slog.Warn("llama-server failed to become healthy within timeout",
				"timeout_seconds", int(startupBudget.Seconds()))
			m.setBlockerReason(BlockerStartupTimeout)
			return
		case <-ticker.C:
			if checkHealth(ctx, client, url) {
				slog.Info("llama-server is healthy", "url", m.ServerURL())
				m.setHealthy(true)
				goto monitoring
			}
		}
	}

monitoring:
	ticker.Reset(healthCheckPeriod)
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if !checkHealth(ctx, client, url) {
				slog.Warn("llama-server health check failed")
				m.setHealthy(false)
				m.setBlockerReason(BlockerProcessFailed)
			} else if !m.Healthy() {
				slog.Info("llama-server recovered")
				m.setHealthy(true)
			}
		}
	}
}

func checkHealth(ctx context.Context, client *http.Client, url string) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return false
	}
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// downloadFile downloads a URL to a local file with progress logging.
func downloadFile(ctx context.Context, url, dst string) error {
	return downloadFileWithAuth(ctx, url, dst, attachModelDownloadAuth)
}

func (m *Manager) modelDownloadURL(cfg ModelConfig) string {
	return m.artifactDownloadURL(cfg.URL, cfg.PlatformPath, cfg.RequiresHuggingFaceAuth)
}

func (m *Manager) mmprojDownloadURL(cfg ModelConfig) string {
	return m.artifactDownloadURL(cfg.MmprojURL, cfg.MmprojPlatformPath, cfg.RequiresHuggingFaceAuth)
}

func (m *Manager) artifactDownloadURL(upstreamURL string, platformPath string, requiresHuggingFaceAuth bool) string {
	// Route through the hub's model-artifact proxy ONLY for models this node
	// cannot pull from upstream by itself — i.e. a GATED HuggingFace repo when
	// the operator has set no local HF token. The hub then attaches its own
	// token on the node's behalf.
	//
	// For current public platform models the hub endpoint is now a redirector:
	// R2 private cache when present, otherwise Hugging Face upstream. That keeps
	// one central cache control point without streaming multi-GB files through
	// the hub process. Operators can force direct upstream with
	// RYV_DISABLE_PLATFORM_MODEL_CACHE=1.
	localHFToken := huggingFaceToken()
	if requiresHuggingFaceAuth && localHFToken != "" {
		return upstreamURL
	}
	gatedWithoutLocalToken := requiresHuggingFaceAuth && localHFToken == ""
	if !gatedWithoutLocalToken && platformModelCacheDownloadsEnabled() &&
		strings.TrimSpace(platformPath) != "" &&
		strings.TrimSpace(m.hubURL) != "" && m.nodeToken != nil {
		return strings.TrimRight(m.hubURL, "/") + platformPath
	}
	if gatedWithoutLocalToken && platformManagedGatedModelsEnabled() &&
		strings.TrimSpace(platformPath) != "" &&
		strings.TrimSpace(m.hubURL) != "" && m.nodeToken != nil {
		return strings.TrimRight(m.hubURL, "/") + platformPath
	}
	return upstreamURL
}

func (m *Manager) downloadModelFile(ctx context.Context, url, dst string) error {
	// Track this download so the heartbeat can surface progress to operators
	// and (through them) to buyers staring at the cold-start UI. modelID is
	// resolved from the dst filename when possible — falls back to the
	// basename when the file isn't in the native registry (e.g. a custom
	// finetuned model from EnsureCustomModel).
	modelID := modelIDForFilename(dst)
	if modelID == "" {
		modelID = filepath.Base(dst)
	}

	// Per-model lock — prevents the background prewarmer and the on-demand
	// inference loop from concurrently creating the same `<dst>.tmp` file,
	// which would truncate each other's bytes and cause the buyer-visible
	// progress bar to flip-flop / reset.
	lockI, _ := m.downloadLocks.LoadOrStore(modelID, &sync.Mutex{})
	lock := lockI.(*sync.Mutex)
	lock.Lock()
	defer lock.Unlock()

	// Re-check after acquiring the lock — the OTHER goroutine may have
	// finished the download while we were waiting. But ALSO validate the
	// GGUF magic header so a corrupt/partial file from a previous
	// interrupted run gets deleted and re-downloaded instead of being
	// fed to llama-server (which would crash on every retry until the
	// 5-min EnsureModel deadline fires — surfaces as the buyer-visible
	// "timeout waiting for <model> to start" error).
	if info, err := os.Stat(dst); err == nil && !info.IsDir() && info.Size() > 1<<20 {
		if vErr := validateGGUF(dst); vErr == nil {
			return nil
		} else {
			slog.Warn("on-disk GGUF failed validation, re-downloading",
				"model", modelID, "path", dst, "error", vErr)
			_ = os.Remove(dst)
		}
	}

	m.registerDownload(modelID)
	defer m.completeDownload(modelID)
	if err := downloadFileWithAuthAndProgress(ctx, url, dst, m.attachModelDownloadAuth, func(done, total int64) {
		m.updateDownloadProgress(modelID, done, total)
	}); err != nil {
		return err
	}
	// Validate the freshly-downloaded file. A non-GGUF response (e.g. an
	// HTML error page from the CDN that returned 200 but with wrong body,
	// or a truncated transfer with status 200 OK that ended early) would
	// otherwise sit on disk forever and brick this model on every run.
	if err := validateGGUF(dst); err != nil {
		_ = os.Remove(dst)
		return fmt.Errorf("downloaded GGUF failed validation (%s): %w", filepath.Base(dst), err)
	}
	return nil
}

func (m *Manager) attachModelDownloadAuth(req *http.Request, rawURL string) {
	attachModelDownloadAuth(req, rawURL)
	if req == nil || m == nil || m.nodeToken == nil || !isRyvionModelArtifactURL(rawURL) {
		return
	}
	req.Header.Set(nodeTokenHeader, m.nodeToken(0))
}

func downloadFileWithAuth(ctx context.Context, url, dst string, attachAuth func(*http.Request, string)) error {
	return downloadFileWithAuthAndProgress(ctx, url, dst, attachAuth, nil)
}

// downloadFileWithAuthAndProgress is the underlying download primitive. The
// optional onProgress callback fires as bytes are written, at the same
// 5-second cadence as the existing progressWriter logging (cheap to call;
// just updates a small counter struct guarded by the Manager's mutex).
func downloadFileWithAuthAndProgress(
	ctx context.Context,
	url, dst string,
	attachAuth func(*http.Request, string),
	onProgress func(bytesDone, bytesTotal int64),
) error {
	tmp := dst + ".tmp"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	if attachAuth != nil {
		attachAuth(req, url)
	}
	client := redirectSafeDownloadClient(&http.Client{Timeout: 30 * time.Minute})
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		if isRyvionModelArtifactURL(url) {
			return fmt.Errorf("download %s: HTTP %d (Ryvion platform model artifact unavailable; hub must configure model cache or upstream model token)", url, resp.StatusCode)
		}
		if isHuggingFaceURL(url) && (resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden) {
			return fmt.Errorf("download %s: HTTP %d (Hugging Face access denied; accept the model license and set HF_TOKEN or HUGGINGFACE_TOKEN for gated models)", url, resp.StatusCode)
		}
		return fmt.Errorf("download %s: HTTP %d", url, resp.StatusCode)
	}

	f, err := os.Create(tmp)
	if err != nil {
		return err
	}

	total := resp.ContentLength
	pw := &progressWriter{dst: f, total: total, label: filepath.Base(dst), onProgress: onProgress}
	if _, err := io.Copy(pw, resp.Body); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	f.Close()
	return os.Rename(tmp, dst)
}

func attachModelDownloadAuth(req *http.Request, url string) {
	if req == nil || !isHuggingFaceURL(url) {
		return
	}
	token := huggingFaceToken()
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
}

func redirectSafeDownloadClient(base *http.Client) *http.Client {
	if base == nil {
		base = &http.Client{Timeout: 30 * time.Minute}
	}
	clone := *base
	prior := clone.CheckRedirect
	clone.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		if len(via) > 0 && !sameDownloadHost(req.URL, via[len(via)-1].URL) {
			req.Header.Del(nodeTokenHeader)
		}
		if prior != nil {
			return prior(req, via)
		}
		return nil
	}
	return &clone
}

func sameDownloadHost(a, b *url.URL) bool {
	if a == nil || b == nil {
		return false
	}
	return strings.EqualFold(a.Scheme, b.Scheme) && strings.EqualFold(a.Host, b.Host)
}

func huggingFaceToken() string {
	token := strings.TrimSpace(os.Getenv("HF_TOKEN"))
	if token == "" {
		token = strings.TrimSpace(os.Getenv("HUGGINGFACE_TOKEN"))
	}
	return token
}

func platformManagedGatedModelsEnabled() bool {
	return strings.TrimSpace(os.Getenv("RYV_DISABLE_PLATFORM_MODEL_DOWNLOADS")) != "1"
}

func platformModelCacheDownloadsEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("RYV_DISABLE_PLATFORM_MODEL_CACHE"))) {
	case "1", "true", "yes", "on":
		return false
	}
	switch strings.ToLower(strings.TrimSpace(os.Getenv("RYV_USE_PLATFORM_MODEL_CACHE"))) {
	case "0", "false", "no", "off":
		return false
	default:
		return true
	}
}

func isHuggingFaceURL(url string) bool {
	return strings.Contains(strings.ToLower(strings.TrimSpace(url)), "huggingface.co/")
}

func isRyvionModelArtifactURL(raw string) bool {
	lower := strings.ToLower(strings.TrimSpace(raw))
	return strings.Contains(lower, "/api/v1/node/models/") && strings.Contains(lower, "/download")
}

func redactDownloadURL(raw string) string {
	if strings.Contains(raw, "?") {
		return strings.SplitN(raw, "?", 2)[0] + "?..."
	}
	return raw
}

type progressWriter struct {
	dst        io.Writer
	total      int64
	written    int64
	label      string
	lastLog    time.Time
	lastReport time.Time
	onProgress func(bytesDone, bytesTotal int64)
}

func (pw *progressWriter) Write(p []byte) (int, error) {
	n, err := pw.dst.Write(p)
	pw.written += int64(n)
	now := time.Now()
	if time.Since(pw.lastLog) > 5*time.Second {
		pw.lastLog = now
		if pw.total > 0 {
			pct := float64(pw.written) / float64(pw.total) * 100
			slog.Info("downloading", "file", pw.label, "progress", fmt.Sprintf("%.1f%%", pct),
				"downloaded_mb", pw.written/(1024*1024), "total_mb", pw.total/(1024*1024))
		} else {
			slog.Info("downloading", "file", pw.label, "downloaded_mb", pw.written/(1024*1024))
		}
	}
	// Surface progress to subscribers (Manager.activeDownloads) at ~1 Hz —
	// fast enough to feel live in the buyer UI, slow enough to avoid lock
	// contention on the activeDownloads mutex during the inner copy loop.
	if pw.onProgress != nil && time.Since(pw.lastReport) > time.Second {
		pw.lastReport = now
		pw.onProgress(pw.written, pw.total)
	}
	return n, err
}

func serverBinaryName() string {
	if runtime.GOOS == "windows" {
		return "llama-server.exe"
	}
	return "llama-server"
}

const serverSourceMarkerName = ".llama-server-source"
const windowsCUDARuntimeURL = "https://github.com/ggml-org/llama.cpp/releases/download/b9180/cudart-llama-bin-win-cuda-12.4-x64.zip"

func serverSourceMarkerPath(serverPath string) string {
	return filepath.Join(filepath.Dir(serverPath), serverSourceMarkerName)
}

func shouldInstallServer(serverPath, markerPath, sourceURL string, sourceExplicit bool) (bool, string) {
	sourceURL = strings.TrimSpace(sourceURL)
	if sourceURL == "" {
		return false, ""
	}
	if _, err := os.Stat(serverPath); err != nil {
		if os.IsNotExist(err) {
			return true, "missing_binary"
		}
		return false, ""
	}
	if sourceExplicit {
		return false, ""
	}
	marker, err := os.ReadFile(markerPath)
	if err != nil {
		if isWindowsAcceleratedLlamaReleaseURL(sourceURL) {
			return true, "missing_source_marker_for_windows_gpu_runtime"
		}
		return false, ""
	}
	if strings.TrimSpace(string(marker)) != expectedServerSourceMarker(sourceURL) {
		return true, "source_url_changed"
	}
	return false, ""
}

func isWindowsCUDALlamaReleaseURL(sourceURL string) bool {
	lower := strings.ToLower(strings.TrimSpace(sourceURL))
	return strings.Contains(lower, "github.com/ggml-org/llama.cpp/releases/download/") &&
		strings.Contains(lower, "-bin-win-cuda-") &&
		strings.HasSuffix(lower, ".zip")
}

func isWindowsVulkanLlamaReleaseURL(sourceURL string) bool {
	lower := strings.ToLower(strings.TrimSpace(sourceURL))
	return strings.Contains(lower, "github.com/ggml-org/llama.cpp/releases/download/") &&
		strings.Contains(lower, "-bin-win-vulkan-") &&
		strings.HasSuffix(lower, ".zip")
}

func isWindowsAcceleratedLlamaReleaseURL(sourceURL string) bool {
	return isWindowsCUDALlamaReleaseURL(sourceURL) || isWindowsVulkanLlamaReleaseURL(sourceURL)
}

func writeServerSourceMarker(markerPath, sourceURL string) error {
	marker := expectedServerSourceMarker(sourceURL)
	if marker == "" {
		return nil
	}
	return os.WriteFile(markerPath, []byte(marker+"\n"), 0o644)
}

func expectedServerSourceMarker(sourceURL string) string {
	sourceURL = strings.TrimSpace(sourceURL)
	if sourceURL == "" {
		return ""
	}
	if isWindowsCUDALlamaReleaseURL(sourceURL) {
		return sourceURL + "\ncuda_runtime_url=" + windowsCUDARuntimeURL + "\ninstaller=ryvion-managed-llama-v3"
	}
	if isWindowsVulkanLlamaReleaseURL(sourceURL) {
		return sourceURL + "\nwindows_accelerator=vulkan\ninstaller=ryvion-managed-llama-v4"
	}
	return sourceURL + "\ninstaller=ryvion-managed-llama-v2"
}

// EnsureBundledServerPath refreshes the managed llama-server bundle for the
// current platform when the existing file is missing or known stale.
func EnsureBundledServerPath(ctx context.Context, serverPath string) (bool, string, error) {
	serverPath = strings.TrimSpace(serverPath)
	if serverPath == "" {
		return false, "", nil
	}
	sourceURL := platformServerURL()
	installServer, installReason := shouldInstallServer(serverPath, serverSourceMarkerPath(serverPath), sourceURL, false)
	if !installServer {
		return false, "", nil
	}
	dataDir := filepath.Dir(filepath.Dir(serverPath))
	if err := os.MkdirAll(filepath.Dir(serverPath), 0o755); err != nil {
		return false, installReason, fmt.Errorf("create bin dir: %w", err)
	}
	if err := checkDiskSpace(dataDir); err != nil {
		return false, installReason, fmt.Errorf("disk space check: %w", err)
	}
	if err := downloadAndExtractServer(ctx, sourceURL, serverPath); err != nil {
		return false, installReason, fmt.Errorf("download llama-server: %w", err)
	}
	if err := writeServerSourceMarker(serverSourceMarkerPath(serverPath), sourceURL); err != nil {
		slog.Warn("failed to write llama-server source marker", "path", serverSourceMarkerPath(serverPath), "error", err)
	}
	return true, installReason, nil
}

// downloadAndExtractServer downloads a llama.cpp release and extracts
// llama-server plus required shared libraries.
func downloadAndExtractServer(ctx context.Context, url, dst string) error {
	if strings.HasSuffix(strings.ToLower(url), ".zip") {
		return downloadAndExtractServerZip(ctx, url, dst)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: HTTP %d", url, resp.StatusCode)
	}

	gz, err := gzip.NewReader(resp.Body)
	if err != nil {
		return fmt.Errorf("gzip: %w", err)
	}
	defer gz.Close()

	binDir := filepath.Dir(dst)
	foundServer := false

	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("tar: %w", err)
		}

		name := filepath.Base(hdr.Name)
		isServer := name == serverBinaryName()
		isLib := strings.HasSuffix(name, ".dylib") || strings.HasSuffix(name, ".so") ||
			strings.Contains(name, ".so.")

		if !isServer && !isLib {
			continue
		}

		outPath := filepath.Join(binDir, name)

		switch hdr.Typeflag {
		case tar.TypeSymlink:
			// Recreate symlink (e.g. libmtmd.0.dylib → libmtmd.0.0.8106.dylib)
			os.Remove(outPath)
			target := filepath.Base(hdr.Linkname)
			if err := os.Symlink(target, outPath); err != nil {
				slog.Warn("failed to create symlink", "name", name, "target", target, "error", err)
			}
		case tar.TypeReg:
			perm := os.FileMode(0o644)
			if isServer {
				perm = 0o755
			}
			f, err := os.OpenFile(outPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, perm)
			if err != nil {
				return err
			}
			if _, err := io.Copy(f, tr); err != nil {
				f.Close()
				return err
			}
			f.Close()
			if isServer {
				foundServer = true
				slog.Info("extracted llama-server", "path", outPath, "size", hdr.Size)
			}
		}
	}
	if !foundServer {
		return fmt.Errorf("llama-server binary not found in archive")
	}
	return nil
}

func downloadAndExtractServerZip(ctx context.Context, url, dst string) error {
	if err := downloadAndExtractServerZipArchive(ctx, url, dst, true); err != nil {
		return err
	}
	if isWindowsCUDALlamaReleaseURL(url) {
		if err := downloadAndExtractServerZipArchive(ctx, windowsCUDARuntimeURL, dst, false); err != nil {
			return fmt.Errorf("download CUDA runtime: %w", err)
		}
	}
	return nil
}

func downloadAndExtractServerZipArchive(ctx context.Context, url, dst string, requireServer bool) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: HTTP %d", url, resp.StatusCode)
	}

	tmpZip, err := os.CreateTemp("", "ryv-llama-*.zip")
	if err != nil {
		return err
	}
	tmpPath := tmpZip.Name()
	defer os.Remove(tmpPath)
	if _, err := io.Copy(tmpZip, resp.Body); err != nil {
		tmpZip.Close()
		return err
	}
	if err := tmpZip.Close(); err != nil {
		return err
	}

	stat, err := os.Stat(tmpPath)
	if err != nil {
		return err
	}

	zr, err := zip.OpenReader(tmpPath)
	if err != nil {
		return fmt.Errorf("zip: %w", err)
	}
	defer zr.Close()

	binDir := filepath.Dir(dst)
	foundServer := false

	for _, f := range zr.File {
		if f.FileInfo().IsDir() {
			continue
		}

		name := filepath.Base(f.Name)
		lc := strings.ToLower(name)
		isServer := lc == strings.ToLower(serverBinaryName())
		isLib := strings.HasSuffix(lc, ".dll") || strings.HasSuffix(lc, ".so") || strings.HasSuffix(lc, ".dylib") ||
			strings.Contains(lc, ".so.")
		if !isServer && !isLib {
			continue
		}

		rc, err := f.Open()
		if err != nil {
			return err
		}

		outPath := filepath.Join(binDir, name)
		perm := os.FileMode(0o644)
		if isServer {
			perm = 0o755
		}
		out, err := os.OpenFile(outPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, perm)
		if err != nil {
			rc.Close()
			return err
		}
		if _, err := io.Copy(out, rc); err != nil {
			out.Close()
			rc.Close()
			return err
		}
		out.Close()
		rc.Close()

		if isServer {
			foundServer = true
			slog.Info("extracted llama-server", "path", outPath, "size", f.UncompressedSize64, "zip_size", stat.Size())
		}
	}

	if requireServer && !foundServer {
		return fmt.Errorf("llama-server binary not found in zip archive")
	}
	return nil
}

func envOr(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

func envOrExplicit(key, fallback string) (string, bool) {
	value, ok := os.LookupEnv(key)
	value = strings.TrimSpace(value)
	if ok && value != "" {
		return value, true
	}
	return fallback, false
}

func envBool(key string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(key))) {
	case "1", "true", "yes", "on", "enabled":
		return true
	default:
		return false
	}
}

// runAMDSmokeTest downloads a tiny model and runs a quick inference to verify ROCm works.
// If the gfx architecture is incompatible, llama-server will segfault — we catch that here.
func (m *Manager) runAMDSmokeTest(ctx context.Context, modelDir string) error {
	// Use tinyllama as the smoke test model — small enough to download quickly
	testModel := filepath.Join(modelDir, "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
	if _, err := os.Stat(testModel); os.IsNotExist(err) {
		cfg, ok := NativeModels["tinyllama"]
		if !ok {
			return fmt.Errorf("tinyllama not in native registry for smoke test")
		}
		if err := checkDiskSpace(m.dataDir); err != nil {
			return fmt.Errorf("disk space check: %w", err)
		}
		slog.Info("downloading smoke test model", "model", cfg.FileName)
		if err := downloadFile(ctx, cfg.URL, testModel); err != nil {
			return fmt.Errorf("download smoke test model: %w", err)
		}
	}

	// Try running llama-server with the test model for a quick health check
	testCtx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()

	port := "18081" // Use a different port to not conflict with main server
	args := []string{
		"--model", testModel,
		"--port", port,
		"--host", "127.0.0.1",
		"--threads", "2",
		"--ctx-size", "512",
		"--n-gpu-layers", "99",
	}

	// For older RDNA2 cards, try injecting HSA_OVERRIDE_GFX_VERSION
	gfxVersion := os.Getenv("HSA_OVERRIDE_GFX_VERSION")

	cmd := exec.CommandContext(testCtx, m.serverPath, args...)
	cmd.Env = append(os.Environ(), "CUDA_VISIBLE_DEVICES=0")
	if gfxVersion != "" {
		cmd.Env = append(cmd.Env, "HSA_OVERRIDE_GFX_VERSION="+gfxVersion)
	}
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start smoke test server: %w", err)
	}

	// Wait for health check
	healthURL := "http://127.0.0.1:" + port + "/health"
	passed := false
	for i := 0; i < 30; i++ {
		time.Sleep(2 * time.Second)
		resp, err := http.Get(healthURL)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				passed = true
				break
			}
		}
		// Check if process died (segfault)
		if cmd.ProcessState != nil && cmd.ProcessState.Exited() {
			break
		}
	}

	// Kill the test server
	if cmd.Process != nil {
		cmd.Process.Kill()
	}
	cmd.Wait()

	if !passed {
		return fmt.Errorf("smoke test failed — llama-server did not become healthy (possible gfx incompatibility)")
	}
	return nil
}

// findVisionProjectorFor returns the path to the multimodal projector
// (mmproj) file that matches the given GGUF model, if one exists. The
// projector is what makes a vision-capable model like Gemma 4 26B
// actually consume image_url parts in OpenAI multimodal messages —
// without it llama-server runs in text-only mode and silently drops
// the image content.
//
// Convention check (in order, first hit wins):
//
//  1. <model_dir>/mmproj-<base>.gguf      — common ggml-org naming
//  2. <model_dir>/<base>.mmproj.gguf      — bartowski / Hugging Face
//  3. <model_dir>/<base>-mmproj.gguf      — older / unofficial
//  4. <model_dir>/mmproj.gguf             — single-projector dirs
//  5. <model_dir>/<base-with-mmproj-suffix>-Q*_K_M.gguf  — quantized variants
//
// Returns "" when no projector is found, which is the right signal
// for callers — they simply omit `--mmproj` and the model continues
// working in text-only mode for that session.
func findVisionProjectorFor(modelPath string) string {
	if modelPath == "" {
		return ""
	}
	dir := filepath.Dir(modelPath)
	base := filepath.Base(modelPath)
	// Strip extension + common quant suffixes so naming like
	// 'gemma-4-26B-A4B-it-Q4_K_M.gguf' becomes the canonical
	// 'gemma-4-26B-A4B-it' for projector matching.
	stem := strings.TrimSuffix(base, filepath.Ext(base))
	for _, suffix := range []string{
		"-Q4_K_M", "-Q5_K_M", "-Q6_K", "-Q8_0", "-F16", "-BF16",
		"-q4_k_m", "-q5_k_m", "-q6_k", "-q8_0", "-f16", "-bf16",
	} {
		stem = strings.TrimSuffix(stem, suffix)
	}

	candidates := []string{
		filepath.Join(dir, "mmproj-"+stem+".gguf"),
		filepath.Join(dir, stem+".mmproj.gguf"),
		filepath.Join(dir, stem+"-mmproj.gguf"),
		filepath.Join(dir, "mmproj.gguf"),
		// Quantized projector variants — some HF mirrors ship multiple.
		filepath.Join(dir, "mmproj-"+stem+"-Q4_K_M.gguf"),
		filepath.Join(dir, "mmproj-"+stem+"-F16.gguf"),
	}
	for _, p := range candidates {
		if info, err := os.Stat(p); err == nil && !info.IsDir() {
			return p
		}
	}
	return ""
}
