package runner

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// emJobSpec is the subset of the em.job.v1 contract the node needs to run the
// engine natively. Geometry/params are opaque to the node and carried verbatim
// into /work/job.json — the runner bundle interprets them.
type emJobSpec struct {
	SchemaVersion string          `json:"schema_version"`
	Task          string          `json:"task"`
	Engine        string          `json:"engine"`
	EngineVersion string          `json:"engine_version"`
	InputURL      string          `json:"input_url"`
	Budget        emBudget        `json:"budget"`
	Runtime       emRuntimeBundle `json:"runtime"` // optional explicit bundle manifest
}

type emBudget struct {
	EstVRAMMB   int `json:"est_vram_mb"`
	EstRuntimeS int `json:"est_runtime_s"`
	MaxCells    int `json:"max_cells"`
}

// emRuntimeBundle lets the hub pin a specific signed bundle in the spec. When
// absent, the node derives the manifest from engine/engine_version + GOOS/GOARCH
// and the env-configured bundle base URL.
type emRuntimeBundle struct {
	BundleURL    string `json:"bundle_url"`
	BundleSHA256 string `json:"bundle_sha256"`
	Entrypoint   string `json:"entrypoint"`
	Signature    string `json:"signature"`
}

// nativeEMDefaultTimeout is used when the spec carries no est_runtime_s budget.
const nativeEMDefaultTimeout = 90 * time.Minute

// RunNativeEM executes a single FDTD simulation natively (no Docker). It mirrors
// the OCI happy path (stage SpecJSON -> /work/job.json, prefetch input_url, run
// offline against /work, read receipt.json + metrics.json, copy the small
// artifact) but the sandboxing OCI gave for free (working-dir jail, mem/cpu/time
// limits, kill-on-timeout) is re-created here for the native process.
func RunNativeEM(ctx context.Context, specJSON, gpus, nodeToken string) (*Result, error) {
	if strings.TrimSpace(specJSON) == "" {
		specJSON = `{}`
	}
	var spec emJobSpec
	if err := json.Unmarshal([]byte(specJSON), &spec); err != nil {
		return nil, fmt.Errorf("parse em job spec: %w", err)
	}

	// Build the bundle manifest: prefer an explicit one in the spec, else derive
	// from engine + version + this host's OS/arch and the configured base URL.
	manifest := emBundleManifest{
		Engine:        strings.TrimSpace(spec.Engine),
		EngineVersion: strings.TrimSpace(spec.EngineVersion),
		BundleURL:     strings.TrimSpace(spec.Runtime.BundleURL),
		BundleSHA256:  strings.TrimSpace(spec.Runtime.BundleSHA256),
		Entrypoint:    strings.TrimSpace(spec.Runtime.Entrypoint),
		Signature:     strings.TrimSpace(spec.Runtime.Signature),
		OS:            runtime.GOOS,
		Arch:          runtime.GOARCH,
	}
	if manifest.BundleURL == "" {
		manifest.BundleURL = deriveEMBundleURL(manifest)
	}
	if manifest.Entrypoint == "" {
		manifest.Entrypoint = defaultEMEntrypoint()
	}

	entrypoint, err := ensureEMBundle(ctx, manifest, nodeToken)
	if err != nil {
		return nil, fmt.Errorf("ensure EM runtime bundle: %w", err)
	}

	// Working-dir jail: a fresh temp dir under the work base, removed on exit.
	workBase := resolveWorkBase(runtime.GOOS, os.Getenv)
	if workBase != "" {
		if err := os.MkdirAll(workBase, 0o755); err != nil {
			return nil, fmt.Errorf("create work dir: %w", err)
		}
	}
	workDir, err := os.MkdirTemp(workBase, "ryv_em_*")
	if err != nil {
		return nil, fmt.Errorf("create temp work dir: %w", err)
	}
	defer os.RemoveAll(workDir)

	if err := os.WriteFile(filepath.Join(workDir, "job.json"), []byte(specJSON), 0o644); err != nil {
		return nil, fmt.Errorf("write job.json: %w", err)
	}
	// Prefetch any input_url BEFORE running so the engine runs offline. Reuses
	// the OCI prefetch map (input_url -> input.bin). After this point the
	// process needs no network ("--network=none" becomes "no network needed").
	if err := prefetchPayloadURL(ctx, specJSON, workDir); err != nil {
		slog.Warn("EM payload prefetch failed (non-fatal)", "error", err)
	}

	// Time limit + kill-on-timeout: derive from budget, clamp to a hard cap.
	timeout := nativeEMTimeout(spec.Budget.EstRuntimeS)
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	bundleRoot := emBundleRoot(entrypoint)
	cmd := buildNativeEMCommand(runCtx, entrypoint, workDir)
	cmd.Dir = workDir // jail the process CWD to the working dir
	cmd.Env = nativeEMProcessEnv(workDir, bundleRoot, gpus, spec.Budget)
	// OS-level hard caps (cgroup v2 on Linux / Job Object on Windows). Falls back
	// to the process-group containment + kill-on-timeout path when the host is
	// unprivileged or the OS feature is unavailable. ctrl is never nil.
	ctrl := applyNativeEMResourceLimits(cmd, spec.Budget)
	defer ctrl.Close()

	var out cappedBuffer
	cmd.Stdout = &out
	cmd.Stderr = &out

	start := time.Now()
	runErr := cmd.Start()
	if runErr == nil {
		// Hook in after the child exists but (on Windows) before it spawns its own
		// children, so the whole tree is captured by the Job Object. On Linux the
		// cgroup is applied at clone time via SysProcAttr, so this is a no-op.
		if err := ctrl.AfterStart(cmd); err != nil {
			slog.Warn("EM hard resource cap setup degraded to process-group fallback", "error", err)
		}
		runErr = cmd.Wait()
	}
	duration := time.Since(start)

	// Kill-on-timeout: CommandContext kills the process when runCtx expires; the
	// OS-level controller (cgroup/Job Object) and process-group teardown below
	// catch any stragglers it spawned.
	ctrl.Kill()
	killNativeEMProcessGroup(cmd)

	exitCode := 0
	if runErr != nil {
		if ee, ok := runErr.(*exec.ExitError); ok {
			exitCode = ee.ExitCode()
		} else {
			exitCode = -1
		}
	}
	if runCtx.Err() == context.DeadlineExceeded {
		slog.Warn("EM native run hit time limit; killed", "timeout", timeout, "job_dir", workDir)
	}

	// Same result contract as the OCI lane: receipt.json output_hash,
	// metrics.json (output_name locates the artifact), result.npz artifact.
	receiptHash := readReceiptHash(
		filepath.Join(workDir, "receipt.json"),
		filepath.Join(workDir, "receipt.partial.json"),
	)
	receiptComplete := receiptFileHasHash(filepath.Join(workDir, "receipt.json"))
	metrics := readMetrics(filepath.Join(workDir, "metrics.json"), duration)
	artifactPath, _ := copyArtifact(workDir, workBase)

	hash := receiptHash
	if hash == "" {
		// No (or partial) receipt: fall back to hashing logs so the hub still
		// gets a non-empty, deterministic hash for a clean fail+refund.
		sum := sha256.Sum256(out.Bytes())
		hash = hex.EncodeToString(sum[:])
	}

	metadata := map[string]any{
		"em_engine":         manifest.Engine,
		"em_engine_version": manifest.EngineVersion,
		"em_bundle_entry":   filepath.Base(entrypoint),
		"em_timeout_s":      int(timeout.Seconds()),
	}
	if recMeta := readEMReceiptMetadata(filepath.Join(workDir, "receipt.json")); len(recMeta) > 0 {
		metadata["em_receipt"] = recMeta
	}

	result := &Result{
		Hash:            hash,
		ExitCode:        exitCode,
		Logs:            out.Tail(32768),
		OutputPath:      artifactPath,
		Duration:        duration,
		Metrics:         metrics,
		Metadata:        metadata,
		ReceiptComplete: receiptComplete,
	}
	return result, runErr
}

// nativeEMTimeout clamps the budgeted runtime to a sane window (kill-on-timeout
// guard). A 2x safety margin is applied to the estimate.
func nativeEMTimeout(estRuntimeS int) time.Duration {
	if estRuntimeS <= 0 {
		return nativeEMDefaultTimeout
	}
	d := time.Duration(estRuntimeS) * 2 * time.Second
	const hardCap = 4 * time.Hour
	if d < time.Minute {
		d = time.Minute
	}
	if d > hardCap {
		d = hardCap
	}
	return d
}

// buildNativeEMCommand constructs the exec.Cmd for the bundle entrypoint. Python
// scripts are run through python; native executables run directly. job.json is
// at <workDir>/job.json and the engine writes its result contract into workDir.
func buildNativeEMCommand(ctx context.Context, entrypoint, workDir string) *exec.Cmd {
	lower := strings.ToLower(entrypoint)
	jobArg := filepath.Join(workDir, "job.json")
	switch {
	case strings.HasSuffix(lower, ".py"):
		// Prefer the bundle's OWN embedded interpreter (which has the FDTD engine
		// + GPU bindings installed); fall back to the host python (e.g. a
		// runner-only bundle where the operator's python carries gprMax via
		// RYV_EM_PYTHON). Without this the host python lacks gprMax and the runner
		// silently degrades to the analytic solver.
		python := emEmbeddedPython(emBundleRoot(entrypoint))
		if python == "" {
			python = nativeEMPython()
		}
		return exec.CommandContext(ctx, python, entrypoint, "--job", jobArg, "--work", workDir)
	case strings.HasSuffix(lower, ".ps1"):
		args := []string{"-NoProfile", "-ExecutionPolicy", "Bypass", "-File", entrypoint, "--job", jobArg, "--work", workDir}
		return exec.CommandContext(ctx, "powershell", args...)
	default:
		return exec.CommandContext(ctx, entrypoint, "--job", jobArg, "--work", workDir)
	}
}

func nativeEMPython() string {
	if p := strings.TrimSpace(os.Getenv("RYV_EM_PYTHON")); p != "" {
		return p
	}
	if runtime.GOOS == "windows" {
		return "python"
	}
	return "python3"
}

// emBundleRoot returns the bundle root dir for a resolved entrypoint. The bundle
// stages the runner under <root>/runner/run.py, so the root is the parent of the
// runner/ dir; a flat layout (entrypoint directly in the bundle dir) falls back
// to the entrypoint's own dir.
func emBundleRoot(entrypoint string) string {
	dir := filepath.Dir(entrypoint)
	if strings.EqualFold(filepath.Base(dir), "runner") {
		return filepath.Dir(dir)
	}
	return dir
}

// emBundlePATH prepends the bundle's interpreter bin dir (python/bin, or
// python/ on Windows) to PATH so a self-contained bundle's nvcc + python tools
// resolve without anything installed on the host. Falls back to the host PATH.
func emBundlePATH(bundleRoot string) string {
	host := os.Getenv("PATH")
	if strings.TrimSpace(bundleRoot) == "" {
		return host
	}
	candidates := []string{filepath.Join(bundleRoot, "python", "bin")}
	if runtime.GOOS == "windows" {
		candidates = []string{
			filepath.Join(bundleRoot, "python", "Scripts"),
			filepath.Join(bundleRoot, "python"),
		}
	}
	for _, c := range candidates {
		if st, err := os.Stat(c); err == nil && st.IsDir() {
			if host == "" {
				return c
			}
			return c + string(os.PathListSeparator) + host
		}
	}
	return host
}

// emEmbeddedPython returns the bundle's self-contained interpreter if present,
// else "" so the caller falls back to the host python. The bundle ships its own
// CPython (with the FDTD engine + GPU bindings) precisely so the real solve can
// run without anything installed on the operator host.
func emEmbeddedPython(bundleRoot string) string {
	if strings.TrimSpace(bundleRoot) == "" {
		return ""
	}
	var candidates []string
	if runtime.GOOS == "windows" {
		candidates = []string{
			filepath.Join(bundleRoot, "python", "python.exe"),
			filepath.Join(bundleRoot, "python", "Scripts", "python.exe"),
		}
	} else {
		candidates = []string{
			filepath.Join(bundleRoot, "python", "bin", "python3"),
			filepath.Join(bundleRoot, "python", "bin", "python"),
		}
	}
	for _, c := range candidates {
		if st, err := os.Stat(c); err == nil && !st.IsDir() {
			return c
		}
	}
	return ""
}

// nativeEMProcessEnv builds a minimal environment for the EM process: the result
// contract paths (mirroring the OCI RYV_* env), a GPU selector hint, and the
// work dir. The native process is expected to honor these and stay offline.
func nativeEMProcessEnv(workDir, bundleRoot, gpus string, budget emBudget) []string {
	env := []string{
		// RYVION_WORK_DIR is the name the runner (run.py / engine_gprmax) actually
		// reads; RYV_WORK_DIR is kept as a back-compat alias.
		"RYVION_WORK_DIR=" + workDir,
		"RYV_WORK_DIR=" + workDir,
		"RYV_RECEIPT_PATH=" + filepath.Join(workDir, "receipt.json"),
		"RYV_PARTIAL_RECEIPT_PATH=" + filepath.Join(workDir, "receipt.partial.json"),
		"RYV_METRICS_PATH=" + filepath.Join(workDir, "metrics.json"),
		"RYV_EM_OFFLINE=1",
		"RYV_EM_GPUS=" + strings.TrimSpace(gpus),
		"RYV_EM_MAX_CELLS=" + strconv.Itoa(budget.MaxCells),
		"RYV_EM_VRAM_MB=" + strconv.Itoa(budget.EstVRAMMB),
		// Prepend the bundle's own bin (its python + nvcc for pycuda's runtime
		// kernel compile) to PATH so a self-contained bundle needs nothing on the
		// host; falls back to the host PATH when no bundle bin exists.
		"PATH=" + emBundlePATH(bundleRoot),
		"HOME=" + os.Getenv("HOME"),
	}
	// Pin the assigned GPU so the EM job uses the scheduler-chosen device and does
	// not collide with a concurrent inference job. With CUDA_VISIBLE_DEVICES set,
	// the engine's "-gpu 0" addresses the (now sole-visible) assigned device.
	if cvd := emCudaVisibleDevices(gpus); cvd != "" {
		env = append(env, "CUDA_VISIBLE_DEVICES="+cvd, "HIP_VISIBLE_DEVICES="+cvd)
	}
	// LD_LIBRARY_PATH: prepend the bundle's own runtime libs (when a fully
	// self-contained bundle ships them), then the host's, so a bundled CUDA/ROCm
	// runtime wins but the operator's toolkit is still found.
	var ldParts []string
	if strings.TrimSpace(bundleRoot) != "" {
		for _, sub := range []string{
			filepath.Join("engine", "cuda"), filepath.Join("engine", "rocm"),
			"engine", filepath.Join("python", "lib"),
		} {
			p := filepath.Join(bundleRoot, sub)
			if st, err := os.Stat(p); err == nil && st.IsDir() {
				ldParts = append(ldParts, p)
			}
		}
	}
	if host := os.Getenv("LD_LIBRARY_PATH"); host != "" {
		ldParts = append(ldParts, host)
	}
	if len(ldParts) > 0 {
		env = append(env, "LD_LIBRARY_PATH="+strings.Join(ldParts, string(os.PathListSeparator)))
	}
	// Pass through the operator's CUDA/ROCm toolchain + any explicit engine GPU
	// preference so the native solver reaches the GPU at runtime (v1 relies on the
	// host toolkit; a fully self-contained bundle is a later optimization).
	for _, key := range []string{
		"CUDA_HOME", "CUDA_PATH", "NVIDIA_VISIBLE_DEVICES",
		"ROCR_VISIBLE_DEVICES", "ROCM_PATH", "RYVION_EM_GPU",
	} {
		if v := os.Getenv(key); v != "" {
			env = append(env, key+"="+v)
		}
	}
	if tmp := os.Getenv("TMPDIR"); tmp != "" {
		env = append(env, "TMPDIR="+tmp)
	}
	if sysroot := os.Getenv("SystemRoot"); sysroot != "" {
		env = append(env, "SystemRoot="+sysroot)
	}
	return env
}

// emCudaVisibleDevices maps the node's --gpus selection (auto|all|none|device
// list) to a CUDA_VISIBLE_DEVICES value: "auto"/"all"/"" -> "" (inherit / engine
// picks); "none" -> "-1" (force CPU); a device list ("0", "0,1", "device=0") ->
// bare indices.
func emCudaVisibleDevices(gpus string) string {
	g := strings.TrimSpace(strings.ToLower(gpus))
	switch g {
	case "", "auto", "all":
		return ""
	case "none":
		return "-1"
	}
	return strings.ReplaceAll(g, "device=", "")
}

func readEMReceiptMetadata(path string) map[string]any {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var rec map[string]any
	if json.Unmarshal(b, &rec) != nil {
		return nil
	}
	out := map[string]any{}
	for _, k := range []string{"variant_id", "study_id", "converged", "engine", "engine_version", "mesh_cells", "duration_ms", "error"} {
		if v, ok := rec[k]; ok {
			out[k] = v
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// deriveEMBundleURL builds the bundle download URL from the configured base when
// the spec does not pin one. Layout: <base>/<engine>/<version>/<os>-<arch>.<ext>
func deriveEMBundleURL(m emBundleManifest) string {
	base := strings.TrimRight(strings.TrimSpace(os.Getenv("RYV_EM_BUNDLE_BASE_URL")), "/")
	if base == "" || m.Engine == "" || m.EngineVersion == "" {
		return ""
	}
	ext := "tar.gz"
	if m.OS == "windows" {
		ext = "zip"
	}
	return fmt.Sprintf("%s/%s/%s/%s-%s.%s", base, m.Engine, m.EngineVersion, m.OS, m.Arch, ext)
}

func defaultEMEntrypoint() string {
	// The signed bundle stages the runner under runner/ (see build_bundle.py), so
	// the entrypoint is runner/run.py on every OS. ensureEMBundle converts the
	// slash to the OS separator; buildNativeEMCommand resolves the interpreter.
	return "runner/run.py"
}
