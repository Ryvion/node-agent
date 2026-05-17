package onboarding

import (
	"fmt"
	"strings"
)

func FormatOnboardingReport(report OnboardingReport) string {
	var b strings.Builder
	hardErrors := report.HardErrorCount()
	warnings := report.WarningCount()
	result := "ok"
	if hardErrors > 0 {
		result = "hard_error"
	} else if warnings > 0 {
		result = "warnings"
	}

	fmt.Fprintf(&b, "Ryvion node onboarding report\n")
	fmt.Fprintf(&b, "Platform: %s/%s\n", cleanToken(report.OS), cleanToken(report.Arch))
	fmt.Fprintf(&b, "Result: %s (%d hard errors, %d warnings)\n", result, hardErrors, warnings)
	for _, check := range report.Checks {
		fmt.Fprintf(&b, "- [%s] %s: %s\n", check.Status, cleanText(check.Title), cleanText(check.Message))
	}
	return strings.TrimRight(b.String(), "\n")
}
