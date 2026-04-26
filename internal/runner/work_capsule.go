package runner

import (
	"archive/zip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"
)

type WorkCapsuleSpec struct {
	Task           string   `json:"task"`
	WorkType       string   `json:"work_type"`
	WorkOrderID    string   `json:"work_order_id"`
	RepositoryID   string   `json:"repository_connection_id"`
	Goal           string   `json:"goal"`
	RepoURL        string   `json:"repo_url"`
	BaseBranch     string   `json:"base_branch"`
	WorkBranch     string   `json:"work_branch"`
	SiteURL        string   `json:"site_url"`
	Adapter        string   `json:"adapter"`
	PolicyPreset   string   `json:"policy_preset"`
	Commands       []string `json:"commands"`
	TestCommands   []string `json:"test_commands"`
	AllowedPaths   []string `json:"allowed_paths"`
	ForbiddenPaths []string `json:"forbidden_paths"`
	RequiredChecks []string `json:"required_checks"`
	GitPush        bool     `json:"git_push"`
	CommitMessage  string   `json:"commit_message"`
	ProfileOnly    bool     `json:"profile_only"`
}

type commandRecord struct {
	Phase      string `json:"phase"`
	Command    string `json:"command"`
	ExitCode   int    `json:"exit_code"`
	DurationMS int64  `json:"duration_ms"`
	OutputTail string `json:"output_tail"`
}

type repositoryProfile struct {
	Languages               []string `json:"languages"`
	Frameworks              []string `json:"frameworks"`
	PackageManager          string   `json:"package_manager"`
	PackageFiles            []string `json:"package_files"`
	SuggestedChecks         []string `json:"suggested_checks"`
	SuggestedForbiddenPaths []string `json:"suggested_forbidden_paths"`
	RouteFiles              []string `json:"route_files"`
	ConfigFiles             []string `json:"config_files"`
	HighRiskPaths           []string `json:"high_risk_paths"`
	ChangedHighRiskPaths    []string `json:"changed_high_risk_paths"`
}

// RunWorkCapsule executes a certified-change work capsule in a temporary workspace.
// The capsule controls existing coding agents through explicit adapter/command
// wrappers and produces a zip bundle with diff, logs, check results, and metadata.
func RunWorkCapsule(ctx context.Context, specJSON string) (*Result, error) {
	start := time.Now()
	var spec WorkCapsuleSpec
	if strings.TrimSpace(specJSON) == "" {
		specJSON = `{}`
	}
	if err := json.Unmarshal([]byte(specJSON), &spec); err != nil {
		return nil, fmt.Errorf("parse work capsule spec: %w", err)
	}
	spec.Adapter = strings.TrimSpace(spec.Adapter)
	if spec.Adapter == "" {
		spec.Adapter = "custom-command"
	}
	workBase := resolveWorkBase(runtime.GOOS, os.Getenv)
	if workBase != "" {
		if err := os.MkdirAll(workBase, 0o755); err != nil {
			return nil, fmt.Errorf("create work base: %w", err)
		}
	}
	workDir, err := os.MkdirTemp(workBase, "ryv_work_capsule_*")
	if err != nil {
		return nil, fmt.Errorf("create work dir: %w", err)
	}
	defer os.RemoveAll(workDir)

	repoDir := workDir
	var log cappedBuffer
	records := []commandRecord{}
	exitCode := 0
	baseCommit := ""
	headCommit := ""

	if err := validateWorkCapsuleCommands(spec); err != nil {
		_, _ = log.Write([]byte(err.Error() + "\n"))
		return workCapsuleResult(workDir, repoDir, spec, records, baseCommit, headCommit, 2, start, log.Tail(32768), evaluateDelivery(spec), err)
	}
	if !supportedWorkCapsuleAdapter(spec.Adapter) {
		err := fmt.Errorf("unsupported work capsule adapter: %s", spec.Adapter)
		_, _ = log.Write([]byte(err.Error() + "\n"))
		return workCapsuleResult(workDir, repoDir, spec, records, baseCommit, headCommit, 2, start, log.Tail(32768), evaluateDelivery(spec), err)
	}
	if err := validateWorkCapsuleBranchPolicy(spec); err != nil {
		_, _ = log.Write([]byte(err.Error() + "\n"))
		return workCapsuleResult(workDir, repoDir, spec, records, baseCommit, headCommit, 2, start, log.Tail(32768), evaluateDelivery(spec), err)
	}

	if strings.TrimSpace(spec.RepoURL) != "" {
		repoDir = filepath.Join(workDir, "repo")
		args := []string{"clone"}
		if strings.TrimSpace(spec.BaseBranch) != "" {
			args = append(args, "--branch", spec.BaseBranch)
		}
		args = append(args, githubCloneRemote(spec.RepoURL, firstNonEmpty(os.Getenv("RYV_WORK_GITHUB_TOKEN"), os.Getenv("GITHUB_TOKEN"))), repoDir)
		rec := runCommand(ctx, workDir, "git", args, "repo", &log)
		records = append(records, rec)
		if rec.ExitCode != 0 {
			exitCode = rec.ExitCode
			return workCapsuleResult(workDir, repoDir, spec, records, baseCommit, headCommit, exitCode, start, log.Tail(32768), evaluateDelivery(spec), fmt.Errorf("git clone failed"))
		}
		baseCommit = gitOutput(ctx, repoDir, "rev-parse", "HEAD")
		if strings.TrimSpace(spec.WorkBranch) != "" && !spec.ProfileOnly {
			rec = runCommand(ctx, repoDir, "git", []string{"checkout", "-B", spec.WorkBranch}, "repo", &log)
			records = append(records, rec)
			if rec.ExitCode != 0 && exitCode == 0 {
				exitCode = rec.ExitCode
			}
		}
	}

	if spec.ProfileOnly {
		_, _ = log.Write([]byte("profile-only preflight: agent commands skipped\n"))
	} else if len(spec.Commands) == 0 && !adapterRunsWithoutCommands(spec.Adapter) {
		msg := "adapter " + spec.Adapter + " requires explicit commands in this MVP"
		_, _ = log.Write([]byte(msg + "\n"))
		exitCode = 2
	} else {
		for _, command := range spec.Commands {
			rec := runShellCommand(ctx, repoDir, command, "agent", &log)
			records = append(records, rec)
			if rec.ExitCode != 0 && exitCode == 0 {
				exitCode = rec.ExitCode
			}
		}
	}

	if !spec.ProfileOnly {
		for _, command := range spec.TestCommands {
			rec := runShellCommand(ctx, repoDir, command, "check", &log)
			records = append(records, rec)
			if rec.ExitCode != 0 && exitCode == 0 {
				exitCode = rec.ExitCode
			}
		}
	}
	delivery := evaluateDelivery(spec)
	if strings.TrimSpace(spec.RepoURL) != "" && spec.GitPush && !spec.ProfileOnly {
		delivery = pushWorkBranch(ctx, repoDir, spec, &records, &log)
		if status, _ := delivery["git_push_status"].(string); status == "failed" && exitCode == 0 {
			exitCode = 3
		}
	}
	if strings.TrimSpace(spec.RepoURL) != "" {
		headCommit = gitOutput(ctx, repoDir, "rev-parse", "HEAD")
	}
	return workCapsuleResult(workDir, repoDir, spec, records, baseCommit, headCommit, exitCode, start, log.Tail(32768), delivery, nil)
}

func workCapsuleResult(workDir, repoDir string, spec WorkCapsuleSpec, records []commandRecord, baseCommit, headCommit string, exitCode int, start time.Time, logs string, delivery map[string]any, runErr error) (*Result, error) {
	diff := ""
	changedFiles := []string{}
	if strings.TrimSpace(spec.RepoURL) != "" {
		_ = gitOutput(context.Background(), repoDir, "add", "-N", ".")
		if strings.TrimSpace(baseCommit) != "" && strings.TrimSpace(headCommit) != "" && baseCommit != headCommit {
			diff = gitOutput(context.Background(), repoDir, "diff", "--binary", baseCommit+".."+headCommit)
			changedFiles = splitLines(gitOutput(context.Background(), repoDir, "diff", "--name-only", baseCommit+".."+headCommit))
		}
		workingDiff := gitOutput(context.Background(), repoDir, "diff", "--binary")
		if strings.TrimSpace(workingDiff) != "" {
			if strings.TrimSpace(diff) != "" {
				diff += "\n"
			}
			diff += workingDiff
		}
		changedSet := map[string]bool{}
		for _, file := range changedFiles {
			if file != "" {
				changedSet[file] = true
			}
		}
		for _, file := range splitLines(gitOutput(context.Background(), repoDir, "diff", "--name-only")) {
			if file != "" && !changedSet[file] {
				changedFiles = append(changedFiles, file)
				changedSet[file] = true
			}
		}
	}
	diffHash := sha256Hex([]byte(diff))
	logHash := sha256Hex([]byte(logs))
	checksStatus := "not_run"
	for _, rec := range records {
		if rec.Phase != "check" {
			continue
		}
		if checksStatus == "not_run" {
			checksStatus = "passed"
		}
		if rec.ExitCode != 0 {
			checksStatus = "failed"
		}
	}
	architecture := evaluateArchitectureImpact(spec, changedFiles, records)
	repoProfile := analyzeRepositoryProfile(repoDir, changedFiles)
	riskLevel, riskSummary := summarizeWorkRisk(spec, changedFiles, exitCode, architecture, repoProfile)
	if delivery == nil {
		delivery = evaluateDelivery(spec)
	}
	workType := defaultString(strings.TrimSpace(spec.WorkType), "certified_change")
	resultDoc := map[string]any{
		"work_order_id":            spec.WorkOrderID,
		"repository_connection_id": spec.RepositoryID,
		"work_type":                workType,
		"goal":                     spec.Goal,
		"repo_url":                 spec.RepoURL,
		"base_branch":              spec.BaseBranch,
		"work_branch":              spec.WorkBranch,
		"base_commit":              baseCommit,
		"head_commit":              headCommit,
		"adapter":                  spec.Adapter,
		"policy_preset":            spec.PolicyPreset,
		"profile_only":             spec.ProfileOnly,
		"commands":                 commandStrings(records),
		"command_count":            len(records),
		"commands_hash":            logHash,
		"changed_files":            changedFiles,
		"diff_hash":                diffHash,
		"diff_bytes":               len(diff),
		"checks":                   records,
		"checks_status":            checksStatus,
		"delivery":                 delivery,
		"architecture":             architecture,
		"repo_profile":             repoProfile,
		"test_output_hash":         logHash,
		"risk_level":               riskLevel,
		"risk_summary":             riskSummary,
		"execution_isolation":      "host_process_enterprise_only",
		"isolation_notes":          "Commands executed on an enterprise-trusted node host workspace; managed OCI capsule execution is required before enabling third-party operator pools.",
	}
	resultJSON, _ := json.MarshalIndent(resultDoc, "", "  ")
	diffPath := filepath.Join(workDir, "diff.patch")
	logPath := filepath.Join(workDir, "commands.log")
	resultPath := filepath.Join(workDir, "work-capsule-result.json")
	_ = os.WriteFile(diffPath, []byte(diff), 0o644)
	_ = os.WriteFile(logPath, []byte(logs), 0o644)
	_ = os.WriteFile(resultPath, resultJSON, 0o644)
	zipPath := filepath.Join(workDir, "work-capsule-bundle.zip")
	if err := writeZip(zipPath, map[string]string{
		"diff.patch":               diffPath,
		"commands.log":             logPath,
		"work-capsule-result.json": resultPath,
	}); err != nil {
		return nil, err
	}
	outPath, err := copyOutOfWorkDir(zipPath)
	if err != nil {
		return nil, err
	}
	hash := fileSHA256(outPath)
	metadata := resultDoc
	metadata["stdout_tail"] = redactSecrets(logs)
	metadata["stderr_tail"] = redactSecrets(logs)
	return &Result{
		Hash:       hash,
		ExitCode:   exitCode,
		Logs:       redactSecrets(logs),
		OutputPath: outPath,
		Duration:   time.Since(start),
		Metrics:    map[string]any{"records": len(records), "changed_files": len(changedFiles)},
		Metadata:   metadata,
	}, runErr
}

func adapterRunsWithoutCommands(adapter string) bool {
	switch strings.ToLower(strings.TrimSpace(adapter)) {
	case "", "custom-command", "noop", "none":
		return true
	default:
		return false
	}
}

func supportedWorkCapsuleAdapter(adapter string) bool {
	switch strings.ToLower(strings.TrimSpace(adapter)) {
	case "", "custom-command", "noop", "none", "codex", "claude-code", "gemini-cli":
		return true
	default:
		return false
	}
}

func validateWorkCapsuleCommands(spec WorkCapsuleSpec) error {
	if spec.ProfileOnly {
		return nil
	}
	for _, command := range append(append([]string{}, spec.Commands...), spec.TestCommands...) {
		if reason := blockedWorkCommandReason(command); reason != "" {
			return fmt.Errorf("work capsule command blocked: %s", reason)
		}
	}
	return nil
}

func validateWorkCapsuleBranchPolicy(spec WorkCapsuleSpec) error {
	if strings.TrimSpace(spec.RepoURL) == "" {
		return nil
	}
	if err := validateWorkCapsuleGitRef(spec.BaseBranch, false); err != nil {
		return fmt.Errorf("invalid base branch: %w", err)
	}
	if spec.ProfileOnly {
		return nil
	}
	if strings.EqualFold(strings.TrimSpace(spec.BaseBranch), strings.TrimSpace(spec.WorkBranch)) {
		return fmt.Errorf("work branch must differ from base branch")
	}
	if err := validateWorkCapsuleGitRef(spec.WorkBranch, true); err != nil {
		return fmt.Errorf("invalid work branch: %w", err)
	}
	return nil
}

func validateWorkCapsuleGitRef(branch string, requireRyvionPrefix bool) error {
	branch = strings.TrimSpace(branch)
	if branch == "" {
		return fmt.Errorf("branch is required")
	}
	if len(branch) > 128 {
		return fmt.Errorf("branch is too long")
	}
	if requireRyvionPrefix && !strings.HasPrefix(strings.ToLower(branch), "ryvion/") {
		return fmt.Errorf("work branches must use the ryvion/ prefix")
	}
	if strings.HasPrefix(branch, "/") || strings.HasSuffix(branch, "/") || strings.HasPrefix(branch, "-") {
		return fmt.Errorf("branch has an unsafe boundary")
	}
	if strings.Contains(branch, "..") || strings.Contains(branch, "//") || strings.Contains(branch, "@{") ||
		strings.HasSuffix(branch, ".") || strings.HasSuffix(branch, ".lock") {
		return fmt.Errorf("branch contains an unsafe git ref sequence")
	}
	for _, r := range branch {
		if r <= 32 || r == 127 {
			return fmt.Errorf("branch contains control or whitespace characters")
		}
		switch r {
		case '\\', '~', '^', ':', '?', '*', '[', ']':
			return fmt.Errorf("branch contains unsupported git ref characters")
		}
	}
	return nil
}

func blockedWorkCommandReason(command string) string {
	normalized := strings.ToLower(strings.Join(strings.Fields(command), " "))
	if normalized == "" {
		return ""
	}
	blocked := []string{
		"vercel deploy", "netlify deploy", "fly deploy", "railway up", "firebase deploy",
		"kubectl apply", "kubectl delete", "helm install", "helm upgrade",
		"terraform apply", "tofu apply", "pulumi up", "aws cloudformation deploy",
		"npm publish", "pnpm publish", "yarn publish", "cargo publish",
		"git push", "docker push", "gh release create",
		"git reset --hard", "git clean -fd", "git clean -xdf",
		"sudo ", "chmod -r 777", "rm -rf /", "rm -fr /", "rm -rf .git", "rm -fr .git",
	}
	for _, needle := range blocked {
		if strings.Contains(normalized, needle) {
			return strings.TrimSpace(needle) + " is not allowed in a certified capsule; use PR-only release"
		}
	}
	if (strings.Contains(normalized, "curl ") || strings.Contains(normalized, "wget ")) &&
		(strings.Contains(normalized, "| sh") || strings.Contains(normalized, "| bash")) {
		return "piped remote shell install is not allowed"
	}
	return ""
}

func runShellCommand(ctx context.Context, dir, command, phase string, log *cappedBuffer) commandRecord {
	if runtime.GOOS == "windows" {
		return runCommand(ctx, dir, "cmd", []string{"/C", command}, phase, log)
	}
	return runCommand(ctx, dir, "sh", []string{"-lc", command}, phase, log)
}

func runCommand(ctx context.Context, dir, name string, args []string, phase string, log *cappedBuffer) commandRecord {
	start := time.Now()
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = dir
	cmd.Env = workCapsuleCommandEnv(os.Environ())
	var out cappedBuffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	err := cmd.Run()
	exitCode := 0
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			exitCode = ee.ExitCode()
		} else {
			exitCode = -1
		}
	}
	text := redactSecrets(out.Tail(32768))
	commandText := redactSecrets(name + " " + strings.Join(args, " "))
	_, _ = log.Write([]byte("$ " + commandText + "\n"))
	_, _ = log.Write([]byte(text + "\n"))
	return commandRecord{
		Phase:      phase,
		Command:    commandText,
		ExitCode:   exitCode,
		DurationMS: time.Since(start).Milliseconds(),
		OutputTail: text,
	}
}

func gitOutput(ctx context.Context, dir string, args ...string) string {
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func splitLines(s string) []string {
	lines := []string{}
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			lines = append(lines, line)
		}
	}
	return lines
}

func commandStrings(records []commandRecord) []string {
	out := make([]string, 0, len(records))
	for _, record := range records {
		out = append(out, record.Command)
	}
	return out
}

func summarizeWorkRisk(spec WorkCapsuleSpec, changed []string, exitCode int, architecture map[string]any, profile repositoryProfile) (string, string) {
	if spec.ProfileOnly {
		if exitCode != 0 {
			return "high", "Repository preflight failed; profile evidence is incomplete."
		}
		return "low", "Repository preflight captured architecture profile without running a code agent or changing files."
	}
	if exitCode != 0 {
		return "high", "One or more commands failed; human review is required before PR creation."
	}
	if status, _ := architecture["status"].(string); status == "failed" {
		if summary, _ := architecture["summary"].(string); strings.TrimSpace(summary) != "" {
			return "high", summary
		}
		return "high", "The change violated one or more architecture boundaries."
	}
	if len(profile.ChangedHighRiskPaths) > 0 {
		return "medium", "The diff touches security, billing, data, deployment, or configuration-sensitive paths."
	}
	for _, file := range changed {
		lower := strings.ToLower(file)
		if strings.Contains(lower, "auth") || strings.Contains(lower, "billing") ||
			strings.Contains(lower, "payment") || strings.Contains(lower, "database") ||
			strings.Contains(lower, "migration") || strings.Contains(lower, ".env") ||
			strings.Contains(lower, "deploy") || strings.Contains(lower, "security") {
			return "medium", "The diff touches security, billing, data, or deployment-sensitive paths."
		}
	}
	if len(changed) == 0 {
		return "review", "No repository diff was produced; review the command log before approval."
	}
	return "low", "The capsule completed with no failed checks and no high-risk paths detected."
}

func analyzeRepositoryProfile(repoDir string, changed []string) repositoryProfile {
	profile := repositoryProfile{}
	if strings.TrimSpace(repoDir) == "" {
		return profile
	}
	info, err := os.Stat(repoDir)
	if err != nil || !info.IsDir() {
		return profile
	}
	files := walkRepositoryFiles(repoDir, 5000)
	languageSet := map[string]bool{}
	frameworkSet := map[string]bool{}
	checkSet := map[string]bool{}
	packageFiles := []string{}
	routeFiles := []string{}
	configFiles := []string{}
	highRiskPaths := []string{}
	for _, file := range files {
		lower := strings.ToLower(file)
		addLanguageForPath(languageSet, lower)
		switch {
		case isPackageFile(lower):
			packageFiles = append(packageFiles, file)
		case isConfigFile(lower):
			configFiles = append(configFiles, file)
		}
		if isRouteFile(lower) {
			routeFiles = append(routeFiles, file)
		}
		if isHighRiskPath(lower) {
			highRiskPaths = append(highRiskPaths, file)
		}
		detectFrameworkMarkers(frameworkSet, lower)
	}
	pkg := readPackageJSON(repoDir)
	for name := range pkg.Dependencies {
		detectFrameworkPackage(frameworkSet, strings.ToLower(name))
	}
	for name := range pkg.DevDependencies {
		detectFrameworkPackage(frameworkSet, strings.ToLower(name))
	}
	profile.PackageManager = detectPackageManager(files)
	if profile.PackageManager == "" && len(pkg.Scripts) > 0 {
		profile.PackageManager = "npm"
	}
	for _, script := range []string{"lint", "typecheck", "test", "build"} {
		if _, ok := pkg.Scripts[script]; ok {
			addString(checkSet, packageScriptCommand(profile.PackageManager, script))
		}
	}
	if hasFile(files, "go.mod") {
		addString(checkSet, "go test ./...")
		addString(checkSet, "go vet ./...")
		addString(checkSet, "go build ./...")
	}
	if hasFile(files, "pyproject.toml") || hasFile(files, "requirements.txt") {
		addString(checkSet, "pytest")
	}
	if hasFile(files, "Cargo.toml") {
		addString(checkSet, "cargo test")
		addString(checkSet, "cargo build")
	}
	profile.Languages = sortedKeys(languageSet)
	profile.Frameworks = sortedKeys(frameworkSet)
	profile.PackageFiles = limitStrings(packageFiles, 40)
	profile.SuggestedChecks = sortedKeys(checkSet)
	profile.SuggestedForbiddenPaths = suggestedForbiddenPathRules(files)
	profile.RouteFiles = limitStrings(routeFiles, 60)
	profile.ConfigFiles = limitStrings(configFiles, 60)
	profile.HighRiskPaths = limitStrings(highRiskPaths, 80)
	profile.ChangedHighRiskPaths = changedHighRiskPaths(changed)
	return profile
}

func suggestedForbiddenPathRules(files []string) []string {
	rules := []string{".env*"}
	addRule := func(rule string) {
		rule = normalizeRepoPath(rule)
		if rule != "" {
			rules = append(rules, rule)
		}
	}
	for _, file := range files {
		path := strings.ToLower(normalizeRepoPath(file))
		switch {
		case strings.HasPrefix(path, ".github/workflows/"):
			addRule(".github/workflows/**")
		case strings.Contains(path, "/auth/") || strings.HasPrefix(path, "auth/") || strings.Contains(path, "auth."):
			addRule(firstSensitiveDir(path, "auth") + "/**")
		case strings.Contains(path, "/billing/") || strings.HasPrefix(path, "billing/"):
			addRule(firstSensitiveDir(path, "billing") + "/**")
		case strings.Contains(path, "/payment/") || strings.HasPrefix(path, "payment/") || strings.Contains(path, "/payments/") || strings.HasPrefix(path, "payments/"):
			addRule(firstSensitiveDir(path, "payment") + "/**")
		case strings.HasPrefix(path, "prisma/"):
			addRule("prisma/**")
		case strings.Contains(path, "migration"):
			addRule(firstPathArea(path) + "/**")
		case strings.HasPrefix(path, "supabase/"):
			addRule("supabase/**")
		case strings.HasPrefix(path, "infra/"):
			addRule("infra/**")
		case strings.HasPrefix(path, "deployment/"):
			addRule("deployment/**")
		case strings.HasPrefix(path, "k8s/"):
			addRule("k8s/**")
		case strings.HasPrefix(path, "terraform/"):
			addRule("terraform/**")
		}
	}
	return limitStrings(rules, 40)
}

func firstSensitiveDir(path, token string) string {
	parts := strings.Split(normalizeRepoPath(path), "/")
	for i, part := range parts {
		if strings.Contains(part, token) {
			return strings.Join(parts[:i+1], "/")
		}
	}
	return firstPathArea(path)
}

func firstPathArea(path string) string {
	path = normalizeRepoPath(path)
	if idx := strings.Index(path, "/"); idx >= 0 {
		return path[:idx]
	}
	return path
}

func walkRepositoryFiles(root string, limit int) []string {
	files := []string{}
	_ = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if path == root {
			return nil
		}
		name := d.Name()
		if d.IsDir() {
			if shouldSkipProfileDir(name) {
				return filepath.SkipDir
			}
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return nil
		}
		files = append(files, normalizeRepoPath(rel))
		if limit > 0 && len(files) >= limit {
			return filepath.SkipAll
		}
		return nil
	})
	sort.Strings(files)
	return files
}

func shouldSkipProfileDir(name string) bool {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case ".git", "node_modules", ".next", "dist", "build", "out", "coverage", "vendor", "target", ".turbo", ".cache", "__pycache__":
		return true
	default:
		return false
	}
}

func addLanguageForPath(set map[string]bool, path string) {
	switch {
	case strings.HasSuffix(path, ".go"):
		set["Go"] = true
	case strings.HasSuffix(path, ".ts") || strings.HasSuffix(path, ".tsx"):
		set["TypeScript"] = true
	case strings.HasSuffix(path, ".js") || strings.HasSuffix(path, ".jsx") || strings.HasSuffix(path, ".mjs") || strings.HasSuffix(path, ".cjs"):
		set["JavaScript"] = true
	case strings.HasSuffix(path, ".py"):
		set["Python"] = true
	case strings.HasSuffix(path, ".rs"):
		set["Rust"] = true
	case strings.HasSuffix(path, ".java"):
		set["Java"] = true
	case strings.HasSuffix(path, ".kt") || strings.HasSuffix(path, ".kts"):
		set["Kotlin"] = true
	case strings.HasSuffix(path, ".swift"):
		set["Swift"] = true
	case strings.HasSuffix(path, ".cs"):
		set["C#"] = true
	case strings.HasSuffix(path, ".php"):
		set["PHP"] = true
	case strings.HasSuffix(path, ".rb"):
		set["Ruby"] = true
	case strings.HasSuffix(path, ".tf") || strings.HasSuffix(path, ".tfvars"):
		set["Terraform"] = true
	case strings.HasSuffix(path, ".sql"):
		set["SQL"] = true
	}
}

func isPackageFile(path string) bool {
	switch path {
	case "package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "bun.lockb", "bun.lock", "go.mod", "go.sum", "pyproject.toml", "requirements.txt", "cargo.toml", "cargo.lock":
		return true
	default:
		return strings.HasSuffix(path, ".csproj") || strings.HasSuffix(path, ".fsproj")
	}
}

func isConfigFile(path string) bool {
	name := repoPathBase(path)
	switch name {
	case "next.config.js", "next.config.mjs", "next.config.ts", "vite.config.js", "vite.config.ts", "astro.config.mjs", "nuxt.config.ts", "svelte.config.js", "tailwind.config.js", "tailwind.config.ts", "dockerfile", "docker-compose.yml", "docker-compose.yaml", "vercel.json", "netlify.toml", "wrangler.toml", "tsconfig.json", "eslint.config.js", ".eslintrc.json":
		return true
	default:
		return strings.HasPrefix(path, ".github/workflows/") || strings.HasPrefix(name, ".env")
	}
}

func isRouteFile(path string) bool {
	name := repoPathBase(path)
	if name == "page.tsx" || name == "page.ts" || name == "page.jsx" || name == "page.js" ||
		name == "route.ts" || name == "route.js" || name == "layout.tsx" || name == "layout.ts" {
		return strings.HasPrefix(path, "app/")
	}
	return strings.HasPrefix(path, "pages/") || strings.HasPrefix(path, "src/pages/") ||
		strings.HasPrefix(path, "routes/") || strings.HasPrefix(path, "src/routes/")
}

func isHighRiskPath(path string) bool {
	path = strings.ToLower(normalizeRepoPath(path))
	name := repoPathBase(path)
	if strings.HasPrefix(name, ".env") {
		return true
	}
	riskNeedles := []string{
		"auth", "billing", "payment", "stripe", "checkout", "invoice", "secret", "token",
		"password", "credential", "database", "migration", "schema", "prisma", "drizzle",
		"supabase", "deploy", "terraform", ".github/workflows", "dockerfile", "docker-compose",
		"kubernetes", "k8s", "security", "permissions", "rbac", "oauth",
	}
	for _, needle := range riskNeedles {
		if strings.Contains(path, needle) {
			return true
		}
	}
	return false
}

func repoPathBase(path string) string {
	path = normalizeRepoPath(path)
	if idx := strings.LastIndex(path, "/"); idx >= 0 {
		return path[idx+1:]
	}
	return path
}

func detectFrameworkMarkers(set map[string]bool, path string) {
	switch {
	case strings.HasPrefix(path, "app/") && (strings.HasSuffix(path, "/page.tsx") || strings.HasSuffix(path, "/route.ts")):
		set["Next.js"] = true
	case strings.HasPrefix(path, "pages/") || strings.HasPrefix(path, "src/pages/"):
		set["Next.js"] = true
	case strings.HasPrefix(path, "src/routes/"):
		set["SvelteKit"] = true
	case strings.Contains(path, "vite.config."):
		set["Vite"] = true
	case strings.Contains(path, "astro.config."):
		set["Astro"] = true
	case strings.Contains(path, "nuxt.config."):
		set["Nuxt"] = true
	case strings.Contains(path, "go.mod"):
		set["Go modules"] = true
	}
}

func detectFrameworkPackage(set map[string]bool, name string) {
	switch name {
	case "next":
		set["Next.js"] = true
	case "react":
		set["React"] = true
	case "vite":
		set["Vite"] = true
	case "vue":
		set["Vue"] = true
	case "nuxt":
		set["Nuxt"] = true
	case "svelte", "@sveltejs/kit":
		set["SvelteKit"] = true
	case "astro":
		set["Astro"] = true
	case "@remix-run/react", "@remix-run/node":
		set["Remix"] = true
	case "express":
		set["Express"] = true
	case "fastify":
		set["Fastify"] = true
	}
}

type packageJSONProfile struct {
	Scripts         map[string]string `json:"scripts"`
	Dependencies    map[string]any    `json:"dependencies"`
	DevDependencies map[string]any    `json:"devDependencies"`
}

func readPackageJSON(repoDir string) packageJSONProfile {
	var pkg packageJSONProfile
	b, err := os.ReadFile(filepath.Join(repoDir, "package.json"))
	if err != nil {
		return pkg
	}
	_ = json.Unmarshal(b, &pkg)
	return pkg
}

func detectPackageManager(files []string) string {
	switch {
	case hasFile(files, "pnpm-lock.yaml"):
		return "pnpm"
	case hasFile(files, "yarn.lock"):
		return "yarn"
	case hasFile(files, "bun.lockb") || hasFile(files, "bun.lock"):
		return "bun"
	case hasFile(files, "package-lock.json") || hasFile(files, "package.json"):
		return "npm"
	default:
		return ""
	}
}

func packageScriptCommand(manager, script string) string {
	manager = strings.TrimSpace(manager)
	if manager == "" {
		manager = "npm"
	}
	switch manager {
	case "yarn":
		return "yarn " + script
	case "pnpm":
		return "pnpm run " + script
	case "bun":
		return "bun run " + script
	default:
		return "npm run " + script
	}
}

func hasFile(files []string, want string) bool {
	want = strings.ToLower(normalizeRepoPath(want))
	for _, file := range files {
		if strings.ToLower(normalizeRepoPath(file)) == want {
			return true
		}
	}
	return false
}

func changedHighRiskPaths(changed []string) []string {
	out := []string{}
	seen := map[string]bool{}
	for _, file := range changed {
		file = normalizeRepoPath(file)
		if file == "" || seen[file] || !isHighRiskPath(file) {
			continue
		}
		seen[file] = true
		out = append(out, file)
	}
	sort.Strings(out)
	return out
}

func addString(set map[string]bool, value string) {
	value = strings.TrimSpace(value)
	if value != "" {
		set[value] = true
	}
}

func sortedKeys(set map[string]bool) []string {
	out := make([]string, 0, len(set))
	for value := range set {
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func limitStrings(values []string, limit int) []string {
	clean := make([]string, 0, len(values))
	seen := map[string]bool{}
	for _, value := range values {
		value = normalizeRepoPath(value)
		if value == "" || seen[value] {
			continue
		}
		seen[value] = true
		clean = append(clean, value)
	}
	sort.Strings(clean)
	if limit > 0 && len(clean) > limit {
		return clean[:limit]
	}
	return clean
}

func evaluateArchitectureImpact(spec WorkCapsuleSpec, changed []string, records []commandRecord) map[string]any {
	allowed := cleanRules(spec.AllowedPaths)
	forbidden := cleanRules(spec.ForbiddenPaths)
	required := cleanRules(spec.RequiredChecks)
	violations := []string{}
	for _, file := range changed {
		normalized := normalizeRepoPath(file)
		if normalized == "" {
			continue
		}
		if len(allowed) > 0 && !matchesAnyRule(normalized, allowed) {
			violations = append(violations, "changed path outside allowed scope: "+normalized)
		}
		if matchesAnyRule(normalized, forbidden) {
			violations = append(violations, "changed forbidden path: "+normalized)
		}
	}
	missingChecks := []string{}
	for _, check := range required {
		if !commandWasRun(check, records) {
			missingChecks = append(missingChecks, check)
		}
	}
	status := "passed"
	summary := "Changed files stayed inside declared architecture boundaries."
	if len(allowed) == 0 && len(forbidden) == 0 && len(required) == 0 {
		status = "not_configured"
		summary = "No architecture contract was configured for this work order."
	} else if len(violations) > 0 || len(missingChecks) > 0 {
		status = "failed"
		summary = "Architecture policy failed; approval is blocked until the change is narrowed or required checks run."
	}
	return map[string]any{
		"status":                  status,
		"summary":                 summary,
		"policy_preset":           strings.TrimSpace(spec.PolicyPreset),
		"allowed_paths":           allowed,
		"forbidden_paths":         forbidden,
		"required_checks":         required,
		"changed_areas":           changedAreas(changed),
		"violations":              violations,
		"missing_required_checks": missingChecks,
	}
}

func cleanRules(in []string) []string {
	out := []string{}
	seen := map[string]bool{}
	for _, rule := range in {
		rule = normalizeRepoPath(rule)
		if rule == "" || seen[rule] {
			continue
		}
		seen[rule] = true
		out = append(out, rule)
	}
	return out
}

func normalizeRepoPath(path string) string {
	path = strings.TrimSpace(strings.ReplaceAll(path, "\\", "/"))
	path = strings.TrimPrefix(path, "./")
	path = strings.TrimPrefix(path, "/")
	return path
}

func matchesAnyRule(file string, rules []string) bool {
	for _, rule := range rules {
		if pathRuleMatches(file, rule) {
			return true
		}
	}
	return false
}

func pathRuleMatches(file, rule string) bool {
	file = normalizeRepoPath(file)
	rule = normalizeRepoPath(rule)
	if file == "" || rule == "" {
		return false
	}
	if strings.HasSuffix(rule, "/**") {
		prefix := strings.TrimSuffix(rule, "/**")
		return file == prefix || strings.HasPrefix(file, prefix+"/")
	}
	if strings.HasSuffix(rule, "/*") {
		prefix := strings.TrimSuffix(rule, "/*")
		if !strings.HasPrefix(file, prefix+"/") {
			return false
		}
		rest := strings.TrimPrefix(file, prefix+"/")
		return !strings.Contains(rest, "/")
	}
	if strings.Contains(rule, "*") {
		return wildcardMatch(file, rule)
	}
	return file == rule || strings.HasPrefix(file, strings.TrimSuffix(rule, "/")+"/")
}

func wildcardMatch(file, rule string) bool {
	parts := strings.Split(rule, "*")
	pos := 0
	for i, part := range parts {
		if part == "" {
			continue
		}
		idx := strings.Index(file[pos:], part)
		if idx < 0 {
			return false
		}
		if i == 0 && !strings.HasPrefix(rule, "*") && idx != 0 {
			return false
		}
		pos += idx + len(part)
	}
	return strings.HasSuffix(rule, "*") || pos == len(file)
}

func commandWasRun(required string, records []commandRecord) bool {
	required = strings.ToLower(strings.TrimSpace(required))
	if required == "" {
		return true
	}
	for _, record := range records {
		if record.Phase != "check" {
			continue
		}
		if strings.Contains(strings.ToLower(record.Command), required) {
			return true
		}
	}
	return false
}

func changedAreas(changed []string) []string {
	out := []string{}
	seen := map[string]bool{}
	for _, file := range changed {
		file = normalizeRepoPath(file)
		if file == "" {
			continue
		}
		area := file
		if idx := strings.Index(area, "/"); idx >= 0 {
			area = area[:idx]
		}
		if !seen[area] {
			seen[area] = true
			out = append(out, area)
		}
	}
	return out
}

func evaluateDelivery(spec WorkCapsuleSpec) map[string]any {
	return map[string]any{
		"git_push_requested": spec.GitPush,
		"git_push_status":    "not_requested",
		"pushed_branch":      "",
		"pushed_commit":      "",
	}
}

func pushWorkBranch(ctx context.Context, repoDir string, spec WorkCapsuleSpec, records *[]commandRecord, log *cappedBuffer) map[string]any {
	delivery := map[string]any{
		"git_push_requested": true,
		"git_push_status":    "failed",
		"pushed_branch":      strings.TrimSpace(spec.WorkBranch),
		"pushed_commit":      "",
	}
	if strings.TrimSpace(spec.WorkBranch) == "" {
		delivery["error"] = "missing work branch"
		return delivery
	}
	for _, rec := range []commandRecord{
		runCommand(ctx, repoDir, "git", []string{"config", "user.name", defaultString(os.Getenv("RYV_GIT_AUTHOR_NAME"), "Ryvion Work Capsule")}, "delivery", log),
		runCommand(ctx, repoDir, "git", []string{"config", "user.email", defaultString(os.Getenv("RYV_GIT_AUTHOR_EMAIL"), "work-capsule@ryvion.ai")}, "delivery", log),
		runCommand(ctx, repoDir, "git", []string{"add", "-A"}, "delivery", log),
	} {
		*records = append(*records, rec)
		if rec.ExitCode != 0 {
			delivery["error"] = "git preparation failed"
			return delivery
		}
	}
	if hasStagedChanges(ctx, repoDir) {
		message := strings.TrimSpace(spec.CommitMessage)
		if message == "" {
			message = "Ryvion certified change"
			if strings.TrimSpace(spec.WorkOrderID) != "" {
				message += " " + strings.TrimSpace(spec.WorkOrderID)
			}
		}
		rec := runCommand(ctx, repoDir, "git", []string{"commit", "-m", message}, "delivery", log)
		*records = append(*records, rec)
		if rec.ExitCode != 0 {
			delivery["error"] = "git commit failed"
			return delivery
		}
	}
	token := strings.TrimSpace(os.Getenv("RYV_WORK_GITHUB_TOKEN"))
	if token == "" {
		token = strings.TrimSpace(os.Getenv("GITHUB_TOKEN"))
	}
	if token != "" {
		remote := githubPushRemote(spec.RepoURL, token)
		if remote == "" {
			delivery["error"] = "repository URL is not a supported GitHub remote"
			return delivery
		}
		rec := runCommand(ctx, repoDir, "git", []string{"remote", "set-url", "origin", remote}, "delivery", log)
		*records = append(*records, rec)
		if rec.ExitCode != 0 {
			delivery["error"] = "git remote auth configuration failed"
			return delivery
		}
	}
	rec := runCommand(ctx, repoDir, "git", []string{"push", "-u", "origin", "HEAD:refs/heads/" + spec.WorkBranch}, "delivery", log)
	*records = append(*records, rec)
	if rec.ExitCode != 0 {
		delivery["error"] = "git push failed"
		return delivery
	}
	delivery["git_push_status"] = "pushed"
	delivery["pushed_commit"] = gitOutput(ctx, repoDir, "rev-parse", "HEAD")
	return delivery
}

func hasStagedChanges(ctx context.Context, repoDir string) bool {
	cmd := exec.CommandContext(ctx, "git", "diff", "--cached", "--quiet")
	cmd.Dir = repoDir
	return cmd.Run() != nil
}

func githubPushRemote(repoURL, token string) string {
	repoURL = strings.TrimSpace(strings.TrimSuffix(repoURL, ".git"))
	token = strings.TrimSpace(token)
	if repoURL == "" || token == "" {
		return ""
	}
	repo := ""
	switch {
	case strings.HasPrefix(repoURL, "https://github.com/"):
		repo = strings.TrimPrefix(repoURL, "https://github.com/")
	case strings.HasPrefix(repoURL, "http://github.com/"):
		repo = strings.TrimPrefix(repoURL, "http://github.com/")
	case strings.HasPrefix(repoURL, "git@github.com:"):
		repo = strings.TrimPrefix(repoURL, "git@github.com:")
	default:
		return ""
	}
	if strings.Count(repo, "/") < 1 {
		return ""
	}
	return "https://x-access-token:" + token + "@github.com/" + repo + ".git"
}

func githubCloneRemote(repoURL, token string) string {
	token = strings.TrimSpace(token)
	if token == "" {
		return repoURL
	}
	remote := githubPushRemote(repoURL, token)
	if remote == "" {
		return repoURL
	}
	return remote
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func fileSHA256(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	h := sha256.New()
	_, _ = io.Copy(h, f)
	return hex.EncodeToString(h.Sum(nil))
}

func copyOutOfWorkDir(path string) (string, error) {
	targetDir := resolveWorkBase(runtime.GOOS, os.Getenv)
	if targetDir == "" {
		targetDir = os.TempDir()
	}
	dst, err := os.CreateTemp(targetDir, "ryv_work_capsule_artifact_*")
	if err != nil {
		return "", err
	}
	defer dst.Close()
	src, err := os.Open(path)
	if err != nil {
		_ = os.Remove(dst.Name())
		return "", err
	}
	defer src.Close()
	if _, err := io.Copy(dst, src); err != nil {
		_ = os.Remove(dst.Name())
		return "", err
	}
	return dst.Name(), nil
}

func writeZip(dest string, files map[string]string) error {
	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer f.Close()
	zw := zip.NewWriter(f)
	defer zw.Close()
	for name, path := range files {
		if err := addZipFile(zw, name, path); err != nil {
			return err
		}
	}
	return nil
}

func addZipFile(zw *zip.Writer, name, path string) error {
	info, err := os.Stat(path)
	if err != nil || info.IsDir() {
		return err
	}
	h, err := zip.FileInfoHeader(info)
	if err != nil {
		return err
	}
	h.Name = name
	h.Method = zip.Deflate
	w, err := zw.CreateHeader(h)
	if err != nil {
		return err
	}
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(w, f)
	return err
}

func redactSecrets(s string) string {
	replacements := []string{"token=", "api_key=", "apikey=", "authorization:", "bearer ", "password=", "secret=", "github_token=", "gh_token="}
	out := s
	lower := strings.ToLower(out)
	for _, needle := range replacements {
		searchFrom := 0
		for {
			idx := strings.Index(lower[searchFrom:], needle)
			if idx < 0 {
				break
			}
			idx += searchFrom
			start := idx + len(needle)
			for start < len(out) && isWorkCapsuleSpace(out[start]) {
				start++
			}
			if strings.HasPrefix(strings.ToLower(out[start:]), "[redacted]") {
				searchFrom = start + len("[redacted]")
				continue
			}
			end := start
			if needle == "authorization:" && strings.HasPrefix(strings.ToLower(out[start:]), "bearer") {
				end = start + len("bearer")
				for end < len(out) && isWorkCapsuleSpace(out[end]) {
					end++
				}
			}
			for end < len(out) && out[end] != ' ' && out[end] != '\n' && out[end] != '\r' && out[end] != '\t' {
				end++
			}
			if end == start {
				searchFrom = start
				continue
			}
			out = out[:start] + "[redacted]" + out[end:]
			lower = strings.ToLower(out)
			searchFrom = start + len("[redacted]")
		}
	}
	searchFrom := 0
	for {
		idx := strings.Index(strings.ToLower(out[searchFrom:]), "x-access-token:")
		if idx < 0 {
			break
		}
		idx += searchFrom
		start := idx + len("x-access-token:")
		end := start
		for end < len(out) && out[end] != '@' && out[end] != ' ' && out[end] != '\n' && out[end] != '\r' && out[end] != '\t' {
			end++
		}
		if end == start {
			searchFrom = start
			continue
		}
		out = out[:start] + "[redacted]" + out[end:]
		searchFrom = start + len("[redacted]")
	}
	return out
}

func isWorkCapsuleSpace(b byte) bool {
	return b == ' ' || b == '\n' || b == '\r' || b == '\t'
}

func workCapsuleCommandEnv(env []string) []string {
	out := make([]string, 0, len(env))
	for _, entry := range env {
		name, _, ok := strings.Cut(entry, "=")
		if !ok {
			continue
		}
		if blockedWorkCapsuleEnvName(name) {
			continue
		}
		out = append(out, entry)
	}
	return out
}

func blockedWorkCapsuleEnvName(name string) bool {
	normalized := strings.ToUpper(strings.TrimSpace(name))
	switch normalized {
	case "GITHUB_TOKEN", "GH_TOKEN", "RYV_WORK_GITHUB_TOKEN", "RYVION_NODE_TOKEN", "RYVION_API_KEY",
		"HUB_TOKEN", "NODE_PRIVATE_KEY", "ED25519_PRIVATE_KEY":
		return true
	default:
		return strings.HasPrefix(normalized, "RYVION_SECRET_") || strings.HasPrefix(normalized, "RYV_SECRET_")
	}
}

func defaultString(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
}
