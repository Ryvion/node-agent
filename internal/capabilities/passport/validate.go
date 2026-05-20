package capability

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
)

func ValidateCapabilityPassport(passport CapabilityPassport) error {
	var errs []error
	if strings.TrimSpace(passport.SchemaVersion) == "" {
		errs = append(errs, fmt.Errorf("schema version required"))
	}
	if strings.TrimSpace(passport.AgentVersion) == "" {
		errs = append(errs, fmt.Errorf("agent version required"))
	}
	if strings.TrimSpace(passport.OS) == "" {
		errs = append(errs, fmt.Errorf("os required"))
	}
	if strings.TrimSpace(passport.Arch) == "" {
		errs = append(errs, fmt.Errorf("arch required"))
	}
	if passport.HardwareProfile.CPUCores == 0 {
		errs = append(errs, fmt.Errorf("cpu cores must be greater than zero"))
	}
	if passport.HardwareProfile.RAMBytes == 0 {
		errs = append(errs, fmt.Errorf("ram bytes must be greater than zero"))
	}
	if err := validateNetworkMetrics(passport.NetworkCapabilitySummary); err != nil {
		errs = append(errs, err)
	}
	for _, format := range passport.ModelCapabilitySummary.SupportedModelFormats {
		if unsafeModelFormat(format) {
			errs = append(errs, fmt.Errorf("unsafe model format %q is not allowed by default", format))
		}
	}
	if passport.ModelCapabilitySummary.SupportsModelLease &&
		strings.TrimSpace(passport.HardwareProfile.GPUModel) == "" &&
		passport.HardwareProfile.VRAMBytes == 0 {
		errs = append(errs, fmt.Errorf("model lease support requires a gpu model or vram bytes"))
	}
	if err := validateNoObviousSecrets(passport); err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

func validateNetworkMetrics(summary NetworkCapabilitySummary) error {
	metrics := map[string]float64{
		"upload Mbps p50":   summary.UploadMbpsP50,
		"upload Mbps p95":   summary.UploadMbpsP95,
		"download Mbps p50": summary.DownloadMbpsP50,
		"download Mbps p95": summary.DownloadMbpsP95,
		"RTT ms p50":        summary.RTTMsP50,
		"RTT ms p95":        summary.RTTMsP95,
		"jitter ms p95":     summary.JitterMsP95,
		"loss rate p95":     summary.LossRateP95,
	}
	var errs []error
	for name, value := range metrics {
		if value < 0 {
			errs = append(errs, fmt.Errorf("%s must not be negative", name))
		}
	}
	if summary.LossRateP95 > 1 {
		errs = append(errs, fmt.Errorf("loss rate p95 must not exceed 1"))
	}
	return errors.Join(errs...)
}

func unsafeModelFormat(format string) bool {
	format = normalizedModelFormat(format)
	if format == "" {
		return false
	}
	switch format {
	case "pickle", "pkl":
		return true
	default:
		return strings.Contains(format, "pickle")
	}
}

func normalizedModelFormat(format string) string {
	format = strings.ToLower(strings.TrimSpace(format))
	format = strings.TrimPrefix(format, ".")
	format = strings.ReplaceAll(format, "_", "-")
	return format
}

func validateNoObviousSecrets(passport CapabilityPassport) error {
	var found []string
	collectSecretPaths(reflect.ValueOf(passport), "passport", &found)
	if len(found) == 0 {
		return nil
	}
	return fmt.Errorf("passport contains secret-like values at %s", strings.Join(found, ", "))
}

func collectSecretPaths(value reflect.Value, path string, found *[]string) {
	if !value.IsValid() {
		return
	}
	if value.Kind() == reflect.Pointer || value.Kind() == reflect.Interface {
		if value.IsNil() {
			return
		}
		collectSecretPaths(value.Elem(), path, found)
		return
	}
	switch value.Kind() {
	case reflect.Struct:
		valueType := value.Type()
		for i := 0; i < value.NumField(); i++ {
			collectSecretPaths(value.Field(i), path+"."+valueType.Field(i).Name, found)
		}
	case reflect.Slice, reflect.Array:
		for i := 0; i < value.Len(); i++ {
			collectSecretPaths(value.Index(i), fmt.Sprintf("%s[%d]", path, i), found)
		}
	case reflect.String:
		if looksLikeSecret(value.String()) {
			*found = append(*found, path)
		}
	}
}

func looksLikeSecret(value string) bool {
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "" {
		return false
	}
	needles := []string{
		"-----begin private key",
		"authorization:",
		"bearer ",
		"api_key",
		"apikey",
		"access_token",
		"refresh_token",
		"private_key",
		"client_secret",
		"password=",
		"passwd=",
		"secret=",
	}
	for _, needle := range needles {
		if strings.Contains(value, needle) {
			return true
		}
	}
	return strings.HasPrefix(value, "sk-") || strings.Contains(value, " hf_")
}
