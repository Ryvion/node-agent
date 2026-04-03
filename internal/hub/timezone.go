package hub

import (
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"
)

// detectIANATimezone returns the system's IANA timezone (e.g., "Europe/Moscow").
// Falls back to time.Now().Location().String() if detection fails.
func detectIANATimezone() string {
	// 1. Check TZ environment variable first (works on all platforms).
	if tz := strings.TrimSpace(os.Getenv("TZ")); tz != "" && strings.Contains(tz, "/") {
		return tz
	}

	// 2. Platform-specific detection.
	switch runtime.GOOS {
	case "linux":
		return detectLinuxTimezone()
	case "windows":
		return detectWindowsTimezone()
	}

	// 3. Fallback to Go's location name.
	return time.Now().Location().String()
}

func detectLinuxTimezone() string {
	// Try /etc/timezone (Debian/Ubuntu).
	if data, err := os.ReadFile("/etc/timezone"); err == nil {
		if tz := strings.TrimSpace(string(data)); tz != "" && strings.Contains(tz, "/") {
			return tz
		}
	}

	// Try /etc/localtime symlink target (most distros).
	if target, err := os.Readlink("/etc/localtime"); err == nil {
		if idx := strings.Index(target, "zoneinfo/"); idx >= 0 {
			return target[idx+len("zoneinfo/"):]
		}
	}

	// Try timedatectl (systemd).
	if out, err := exec.Command("timedatectl", "show", "--property=Timezone", "--value").Output(); err == nil {
		if tz := strings.TrimSpace(string(out)); tz != "" && strings.Contains(tz, "/") {
			return tz
		}
	}

	return time.Now().Location().String()
}

// windowsToIANA maps common Windows timezone names to IANA equivalents.
var windowsToIANA = map[string]string{
	"Russian Standard Time":           "Europe/Moscow",
	"Kaliningrad Standard Time":       "Europe/Kaliningrad",
	"Ekaterinburg Standard Time":      "Asia/Yekaterinburg",
	"N. Central Asia Standard Time":   "Asia/Novosibirsk",
	"North Asia Standard Time":        "Asia/Krasnoyarsk",
	"North Asia East Standard Time":   "Asia/Irkutsk",
	"Yakutsk Standard Time":           "Asia/Yakutsk",
	"Vladivostok Standard Time":       "Asia/Vladivostok",
	"Magadan Standard Time":           "Asia/Magadan",
	"Kamchatka Standard Time":         "Asia/Kamchatka",
	"Belarus Standard Time":           "Europe/Minsk",
	"FLE Standard Time":               "Europe/Kyiv",
	"Eastern Standard Time":           "America/New_York",
	"Central Standard Time":           "America/Chicago",
	"Mountain Standard Time":          "America/Denver",
	"Pacific Standard Time":           "America/Los_Angeles",
	"Alaskan Standard Time":           "America/Anchorage",
	"Hawaiian Standard Time":          "Pacific/Honolulu",
	"US Mountain Standard Time":       "America/Phoenix",
	"Eastern Standard Time (Mexico)":  "America/Mexico_City",
	"Central European Standard Time":  "Europe/Warsaw",
	"W. Europe Standard Time":         "Europe/Berlin",
	"Romance Standard Time":           "Europe/Paris",
	"GMT Standard Time":               "Europe/London",
	"E. Europe Standard Time":         "Europe/Bucharest",
	"GTB Standard Time":               "Europe/Athens",
	"Central Europe Standard Time":    "Europe/Budapest",
	"Turkey Standard Time":            "Asia/Istanbul",
	"Finland Standard Time":           "Europe/Helsinki",
	"China Standard Time":             "Asia/Shanghai",
	"Tokyo Standard Time":             "Asia/Tokyo",
	"Korea Standard Time":             "Asia/Seoul",
	"India Standard Time":             "Asia/Kolkata",
	"Singapore Standard Time":         "Asia/Singapore",
	"SE Asia Standard Time":           "Asia/Bangkok",
	"Arab Standard Time":              "Asia/Riyadh",
	"Arabian Standard Time":           "Asia/Dubai",
	"AUS Eastern Standard Time":       "Australia/Sydney",
	"W. Australia Standard Time":      "Australia/Perth",
	"E. Australia Standard Time":      "Australia/Brisbane",
	"New Zealand Standard Time":       "Pacific/Auckland",
	"E. South America Standard Time":  "America/Sao_Paulo",
	"Argentina Standard Time":         "America/Argentina/Buenos_Aires",
	"Pacific SA Standard Time":        "America/Santiago",
	"SA Pacific Standard Time":        "America/Bogota",
	"Atlantic Standard Time":          "America/Halifax",
	"Newfoundland Standard Time":      "America/St_Johns",
	"Canada Central Standard Time":    "America/Regina",
}

func detectWindowsTimezone() string {
	// Try PowerShell to get Windows timezone name.
	out, err := exec.Command("powershell", "-NoProfile", "-Command",
		"[System.TimeZoneInfo]::Local.Id").Output()
	if err != nil {
		return time.Now().Location().String()
	}
	winTZ := strings.TrimSpace(string(out))
	if iana, ok := windowsToIANA[winTZ]; ok {
		return iana
	}
	// Return the Windows name as-is — the hub will treat unknown names with benefit of the doubt.
	return winTZ
}
