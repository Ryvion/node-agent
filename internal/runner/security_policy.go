package runner

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
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

func allowLoopbackDownloads() bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv("RYV_ALLOW_LOOPBACK_DOWNLOADS")))
	return v == "1" || v == "true"
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
