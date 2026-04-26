package runner

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestEvaluateArchitectureImpactBlocksForbiddenPathsAndMissingChecks(t *testing.T) {
	impact := evaluateArchitectureImpact(WorkCapsuleSpec{
		AllowedPaths:   []string{"app/**"},
		ForbiddenPaths: []string{"app/secrets/**"},
		RequiredChecks: []string{"npm run lint"},
	}, []string{"app/secrets/token.ts", "internal/db.go"}, []commandRecord{
		{Phase: "agent", Command: "sh -lc npm run lint", ExitCode: 0},
		{Phase: "check", Command: "sh -lc npm test", ExitCode: 0},
	})

	if got := impact["status"]; got != "failed" {
		t.Fatalf("status = %v, want failed", got)
	}
	violations := impact["violations"].([]string)
	if len(violations) != 2 {
		t.Fatalf("violations = %v, want forbidden and outside-allowed violations", violations)
	}
	missing := impact["missing_required_checks"].([]string)
	if len(missing) != 1 || missing[0] != "npm run lint" {
		t.Fatalf("missing_required_checks = %v, want npm run lint", missing)
	}
}

func TestEvaluateArchitectureImpactPassesDeclaredBoundaries(t *testing.T) {
	impact := evaluateArchitectureImpact(WorkCapsuleSpec{
		PolicyPreset:   "frontend_ui",
		AllowedPaths:   []string{"app/dashboard/**", "components/dashboard/**"},
		ForbiddenPaths: []string{"auth/**", ".env*"},
		RequiredChecks: []string{"npm run build"},
	}, []string{"app/dashboard/page.tsx", "components/dashboard/chart.tsx"}, []commandRecord{
		{Phase: "check", Command: "sh -lc npm run build", ExitCode: 0},
	})

	if got := impact["status"]; got != "passed" {
		t.Fatalf("status = %v, want passed", got)
	}
	if got := impact["policy_preset"]; got != "frontend_ui" {
		t.Fatalf("policy_preset = %v, want frontend_ui", got)
	}
	if got := impact["violations"].([]string); len(got) != 0 {
		t.Fatalf("violations = %v, want none", got)
	}
	if got := impact["missing_required_checks"].([]string); len(got) != 0 {
		t.Fatalf("missing_required_checks = %v, want none", got)
	}
}

func TestGitHubPushRemoteAndRedaction(t *testing.T) {
	remote := githubPushRemote("git@github.com:org/repo.git", "ghp_secret")
	if remote != "https://x-access-token:ghp_secret@github.com/org/repo.git" {
		t.Fatalf("remote = %q", remote)
	}
	redacted := redactSecrets("git remote set-url origin " + remote)
	if redacted == "" || redacted == remote || redactedContains(redacted, "ghp_secret") {
		t.Fatalf("redacted command leaked token: %q", redacted)
	}
}

func TestValidateWorkCapsuleCommandsBlocksDirectDeploy(t *testing.T) {
	err := validateWorkCapsuleCommands(WorkCapsuleSpec{Commands: []string{"npm run build", "vercel deploy --prod"}})
	if err == nil || !strings.Contains(err.Error(), "vercel deploy") {
		t.Fatalf("err = %v, want vercel deploy block", err)
	}
	err = validateWorkCapsuleCommands(WorkCapsuleSpec{Commands: []string{"git reset --hard HEAD"}})
	if err == nil || !strings.Contains(err.Error(), "git reset --hard") {
		t.Fatalf("err = %v, want reset block", err)
	}
	if err := validateWorkCapsuleCommands(WorkCapsuleSpec{ProfileOnly: true, Commands: []string{"vercel deploy --prod"}}); err != nil {
		t.Fatalf("profile-only preflight should skip command policy, got %v", err)
	}
}

func TestSupportedWorkCapsuleAdapterAllowlist(t *testing.T) {
	for _, adapter := range []string{"custom-command", "codex", "claude-code", "gemini-cli", "noop", "none"} {
		if !supportedWorkCapsuleAdapter(adapter) {
			t.Fatalf("adapter %q should be supported", adapter)
		}
	}
	if supportedWorkCapsuleAdapter("random-agent") {
		t.Fatal("random-agent should be rejected")
	}
}

func TestSummarizeWorkRiskProfileOnly(t *testing.T) {
	level, summary := summarizeWorkRisk(WorkCapsuleSpec{ProfileOnly: true}, nil, 0, nil, repositoryProfile{})
	if level != "low" || !strings.Contains(summary, "preflight") {
		t.Fatalf("risk = %s/%q, want low preflight", level, summary)
	}
}

func TestWorkCapsuleCommandEnvDropsControlPlaneSecrets(t *testing.T) {
	env := workCapsuleCommandEnv([]string{
		"PATH=/usr/bin",
		"GITHUB_TOKEN=secret",
		"RYV_WORK_GITHUB_TOKEN=secret",
		"OPENAI_API_KEY=model-key",
		"RYVION_SECRET_INTERNAL=secret",
	})
	joined := strings.Join(env, "\n")
	if !strings.Contains(joined, "PATH=/usr/bin") || !strings.Contains(joined, "OPENAI_API_KEY=model-key") {
		t.Fatalf("env = %v, want safe runtime vars preserved", env)
	}
	for _, forbidden := range []string{"GITHUB_TOKEN", "RYV_WORK_GITHUB_TOKEN", "RYVION_SECRET_INTERNAL"} {
		if strings.Contains(joined, forbidden) {
			t.Fatalf("env leaked %s: %v", forbidden, env)
		}
	}
}

func TestRedactSecretsCoversBearerAndGitHubToken(t *testing.T) {
	redacted := redactSecrets("Authorization: Bearer abc123 github_token=ghp_secret")
	if strings.Contains(redacted, "abc123") || strings.Contains(redacted, "ghp_secret") {
		t.Fatalf("redacted output leaked secret: %q", redacted)
	}
}

func TestAnalyzeRepositoryProfileDetectsFrameworkChecksAndRisk(t *testing.T) {
	dir := t.TempDir()
	writeTestFile(t, dir, "package.json", `{
		"scripts": {
			"build": "next build",
			"lint": "next lint",
			"test": "vitest run"
		},
		"dependencies": {
			"next": "16.0.0",
			"react": "19.0.0"
		},
		"devDependencies": {
			"vite": "7.0.0"
		}
	}`)
	writeTestFile(t, dir, "pnpm-lock.yaml", "lockfileVersion: '9.0'\n")
	writeTestFile(t, dir, "app/dashboard/page.tsx", "export default function Page() { return null }\n")
	writeTestFile(t, dir, "app/api/auth/route.ts", "export function GET() {}\n")
	writeTestFile(t, dir, ".github/workflows/deploy.yml", "name: deploy\n")

	profile := analyzeRepositoryProfile(dir, []string{"app/api/auth/route.ts", "app/dashboard/page.tsx"})
	if profile.PackageManager != "pnpm" {
		t.Fatalf("package manager = %q, want pnpm", profile.PackageManager)
	}
	if !contains(profile.Frameworks, "Next.js") || !contains(profile.Frameworks, "React") || !contains(profile.Frameworks, "Vite") {
		t.Fatalf("frameworks = %v, want Next.js, React, and Vite", profile.Frameworks)
	}
	if !contains(profile.SuggestedChecks, "pnpm run build") || !contains(profile.SuggestedChecks, "pnpm run lint") || !contains(profile.SuggestedChecks, "pnpm run test") {
		t.Fatalf("suggested checks = %v, want package scripts", profile.SuggestedChecks)
	}
	if !contains(profile.RouteFiles, "app/dashboard/page.tsx") || !contains(profile.RouteFiles, "app/api/auth/route.ts") {
		t.Fatalf("route files = %v, want app routes", profile.RouteFiles)
	}
	if !contains(profile.ChangedHighRiskPaths, "app/api/auth/route.ts") {
		t.Fatalf("changed high risk paths = %v, want auth route", profile.ChangedHighRiskPaths)
	}
	if !contains(profile.SuggestedForbiddenPaths, "app/api/auth/**") || !contains(profile.SuggestedForbiddenPaths, ".github/workflows/**") {
		t.Fatalf("suggested forbidden paths = %v, want auth and workflow rules", profile.SuggestedForbiddenPaths)
	}
}

func redactedContains(s, needle string) bool {
	return strings.Contains(s, needle)
}

func writeTestFile(t *testing.T, root, rel, content string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", rel, err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", rel, err)
	}
}

func contains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
