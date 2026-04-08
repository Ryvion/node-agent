package runner

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

var blockedDownloadCIDRs = mustParseCIDRs(
	"100.64.0.0/10",
	"198.18.0.0/15",
)

func validateDownloadURL(raw string, allowLoopback bool) error {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	u, err := url.Parse(raw)
	if err != nil || u == nil || strings.TrimSpace(u.Hostname()) == "" {
		return fmt.Errorf("invalid download url")
	}
	switch strings.ToLower(strings.TrimSpace(u.Scheme)) {
	case "https":
	case "http":
		if !allowLoopback || !isLoopbackHost(u.Hostname()) {
			return fmt.Errorf("download url must use https")
		}
	default:
		return fmt.Errorf("download url must use https")
	}
	return validateRemoteHost(context.Background(), u.Hostname(), allowLoopback)
}

func validateRemoteHost(ctx context.Context, host string, allowLoopback bool) error {
	host = normalizeHost(host)
	if host == "" {
		return fmt.Errorf("host required")
	}
	if host == "localhost" {
		if allowLoopback {
			return nil
		}
		return fmt.Errorf("host resolves to local/private address")
	}
	if ip := net.ParseIP(host); ip != nil {
		return validatePublicIP(ip, allowLoopback)
	}
	lookupCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	addrs, err := net.DefaultResolver.LookupIPAddr(lookupCtx, host)
	if err != nil {
		return fmt.Errorf("host lookup failed")
	}
	if len(addrs) == 0 {
		return fmt.Errorf("host lookup returned no addresses")
	}
	for _, addr := range addrs {
		if err := validatePublicIP(addr.IP, allowLoopback); err != nil {
			return err
		}
	}
	return nil
}

func restrictedHTTPClient(timeout time.Duration, allowLoopback bool) *http.Client {
	transport := http.DefaultTransport.(*http.Transport).Clone()
	dialer := &net.Dialer{
		Timeout:   10 * time.Second,
		KeepAlive: 30 * time.Second,
	}
	transport.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
		host, _, err := net.SplitHostPort(addr)
		if err != nil {
			host = addr
		}
		if err := validateRemoteHost(ctx, host, allowLoopback); err != nil {
			return nil, err
		}
		return dialer.DialContext(ctx, network, addr)
	}
	return &http.Client{
		Timeout:   timeout,
		Transport: transport,
	}
}

func validateAgentImageRef(image string) error {
	image = strings.TrimSpace(strings.ToLower(image))
	if image == "" {
		return fmt.Errorf("image required")
	}
	if strings.Contains(image, "@sha256:") {
		return nil
	}

	for _, prefix := range allowedAgentImagePrefixes() {
		if !strings.HasPrefix(image, prefix) {
			continue
		}
		name := image[strings.LastIndex(image, "/")+1:]
		if !strings.Contains(name, ":") {
			return fmt.Errorf("managed agent images must use a versioned tag or digest")
		}
		if strings.HasSuffix(name, ":latest") {
			return fmt.Errorf("managed agent images must not use :latest")
		}
		return nil
	}
	return fmt.Errorf("agent image must use a pinned digest or an approved managed prefix")
}

func verifyAgentImageSignature(ctx context.Context, image string) error {
	if !shouldRequireAgentSignature(image) {
		return nil
	}

	cosignBin, err := exec.LookPath("cosign")
	if err != nil {
		return fmt.Errorf("cosign not found in PATH")
	}

	verifyCtx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()

	args := []string{"verify", "--output", "json"}
	if keyPath := strings.TrimSpace(os.Getenv("RYV_AGENT_COSIGN_PUBLIC_KEY")); keyPath != "" {
		args = append(args, "--key", keyPath)
	} else {
		args = append(
			args,
			"--certificate-identity-regexp", agentCosignIdentityRegex(),
			"--certificate-oidc-issuer", agentCosignOIDCIssuer(),
		)
	}
	args = append(args, image)

	cmd := exec.CommandContext(verifyCtx, cosignBin, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		msg := strings.TrimSpace(string(out))
		if msg == "" {
			msg = err.Error()
		}
		return fmt.Errorf("cosign verify failed: %s", msg)
	}
	return nil
}

func allowLoopbackDownloads() bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv("RYV_ALLOW_LOOPBACK_DOWNLOADS")))
	return v == "1" || v == "true"
}

func allowedAgentImagePrefixes() []string {
	prefixes := []string{"ghcr.io/ryvion/"}
	if raw := strings.TrimSpace(os.Getenv("RYV_ALLOWED_AGENT_IMAGE_PREFIXES")); raw != "" {
		prefixes = nil
		for _, entry := range strings.Split(raw, ",") {
			entry = strings.TrimSpace(strings.ToLower(entry))
			if entry != "" {
				prefixes = append(prefixes, entry)
			}
		}
	}
	return prefixes
}

func shouldRequireAgentSignature(image string) bool {
	if raw := strings.TrimSpace(os.Getenv("RYV_REQUIRE_AGENT_SIGNATURES")); raw != "" {
		return isTruthyEnv(raw)
	}
	if strings.TrimSpace(os.Getenv("RYV_AGENT_COSIGN_PUBLIC_KEY")) != "" {
		return true
	}
	if isManagedAgentImage(image) {
		return managedImageSignatureRequiredByDefault(image)
	}
	return false
}

func agentCosignIdentityRegex() string {
	if raw := strings.TrimSpace(os.Getenv("RYV_AGENT_COSIGN_IDENTITY_REGEX")); raw != "" {
		return raw
	}
	return `^https://github\.com/Ryvion/runners/\.github/workflows/build\.yml@refs/(heads/main|tags/.+)$`
}

func agentCosignOIDCIssuer() string {
	if raw := strings.TrimSpace(os.Getenv("RYV_AGENT_COSIGN_OIDC_ISSUER")); raw != "" {
		return raw
	}
	return "https://token.actions.githubusercontent.com"
}

func managedImageSignatureRequiredByDefault(image string) bool {
	image = strings.ToLower(strings.TrimSpace(image))
	if strings.Contains(image, "@sha256:") {
		return true
	}
	tag := imageTag(image)
	if tag == "" {
		return false
	}
	minTag := strings.TrimSpace(os.Getenv("RYV_MANAGED_AGENT_SIGNATURE_MIN_TAG"))
	if minTag == "" {
		minTag = "0.1.1"
	}
	return compareSemverTags(tag, minTag) >= 0
}

func isManagedAgentImage(image string) bool {
	image = strings.ToLower(strings.TrimSpace(image))
	for _, prefix := range allowedAgentImagePrefixes() {
		if strings.HasPrefix(image, prefix) {
			return true
		}
	}
	return false
}

func imageTag(image string) string {
	image = strings.TrimSpace(image)
	if image == "" || strings.Contains(image, "@sha256:") {
		return ""
	}
	lastSlash := strings.LastIndex(image, "/")
	lastColon := strings.LastIndex(image, ":")
	if lastColon <= lastSlash {
		return ""
	}
	return image[lastColon+1:]
}

func compareSemverTags(left, right string) int {
	lv, lok := parseSemverTag(left)
	rv, rok := parseSemverTag(right)
	if !lok || !rok {
		return strings.Compare(left, right)
	}
	for i := 0; i < 3; i++ {
		switch {
		case lv[i] < rv[i]:
			return -1
		case lv[i] > rv[i]:
			return 1
		}
	}
	return 0
}

func parseSemverTag(raw string) ([3]int, bool) {
	var out [3]int
	raw = strings.TrimSpace(raw)
	parts := strings.Split(raw, ".")
	if len(parts) != 3 {
		return out, false
	}
	for i, part := range parts {
		n, err := strconv.Atoi(part)
		if err != nil {
			return out, false
		}
		out[i] = n
	}
	return out, true
}

func isTruthyEnv(raw string) bool {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func normalizeHost(host string) string {
	host = strings.ToLower(strings.TrimSpace(host))
	if strings.HasPrefix(host, "[") && strings.HasSuffix(host, "]") {
		host = strings.TrimPrefix(strings.TrimSuffix(host, "]"), "[")
	}
	if parsedHost, _, err := net.SplitHostPort(host); err == nil {
		host = parsedHost
	}
	return strings.TrimSuffix(host, ".")
}

func isLoopbackHost(host string) bool {
	host = normalizeHost(host)
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func validatePublicIP(ip net.IP, allowLoopback bool) error {
	if ip == nil {
		return fmt.Errorf("host resolves to local/private address")
	}
	if allowLoopback && ip.IsLoopback() {
		return nil
	}
	if ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() || ip.IsMulticast() || ip.IsUnspecified() {
		return fmt.Errorf("host resolves to local/private address")
	}
	for _, cidr := range blockedDownloadCIDRs {
		if cidr.Contains(ip) {
			return fmt.Errorf("host resolves to local/private address")
		}
	}
	return nil
}

func mustParseCIDRs(raw ...string) []*net.IPNet {
	out := make([]*net.IPNet, 0, len(raw))
	for _, entry := range raw {
		_, network, err := net.ParseCIDR(entry)
		if err != nil {
			panic(err)
		}
		out = append(out, network)
	}
	return out
}
