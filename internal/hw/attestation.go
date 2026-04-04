package hw

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
)

// AttestationReport contains platform attestation evidence.
type AttestationReport struct {
	Method    uint32 // 0=none, 1=sev-snp, 2=tdx, 3=sgx
	TEEType   string
	ReportB64 string // Base64 platform report
	NonceHex  string // Nonce the report was generated for
	CertChain string // VCEK cert chain (PEM) if available
}

// GenerateNonce creates a random 32-byte nonce for attestation challenges.
func GenerateNonce() ([]byte, error) {
	nonce := make([]byte, 32)
	_, err := rand.Read(nonce)
	return nonce, err
}

// GenerateAttestationReport produces a TEE attestation report for the given nonce.
// Returns a zero report if no TEE is available.
func GenerateAttestationReport(nonce []byte) AttestationReport {
	if runtime.GOOS != "linux" {
		return AttestationReport{}
	}

	// AMD SEV-SNP
	if _, err := os.Stat("/dev/sev-guest"); err == nil {
		return generateSEVSNPReport(nonce)
	}

	// Intel TDX via configfs-tsm
	if _, err := os.Stat("/sys/kernel/config/tsm/report"); err == nil {
		return generateTDXReport(nonce)
	}

	return AttestationReport{}
}

func generateSEVSNPReport(nonce []byte) AttestationReport {
	nonceHex := hex.EncodeToString(nonce)
	tmpDir, err := os.MkdirTemp("", "snp-attest-*")
	if err != nil {
		return AttestationReport{Method: 1, TEEType: "sev-snp"}
	}
	defer os.RemoveAll(tmpDir)

	reportPath := tmpDir + "/report.bin"
	cmd := exec.Command("snpguest", "report", reportPath, nonceHex)
	if err := cmd.Run(); err != nil {
		// Fallback: try direct ioctl via sev-guest device
		report, err := readSEVReportDirect(nonce)
		if err != nil {
			return AttestationReport{Method: 1, TEEType: "sev-snp"}
		}
		return AttestationReport{
			Method:    1,
			TEEType:   "sev-snp",
			ReportB64: base64.StdEncoding.EncodeToString(report),
			NonceHex:  nonceHex,
		}
	}

	reportBytes, err := os.ReadFile(reportPath)
	if err != nil {
		return AttestationReport{Method: 1, TEEType: "sev-snp"}
	}

	certChain := fetchVCEKCertChain()

	return AttestationReport{
		Method:    1,
		TEEType:   "sev-snp",
		ReportB64: base64.StdEncoding.EncodeToString(reportBytes),
		NonceHex:  nonceHex,
		CertChain: certChain,
	}
}

func readSEVReportDirect(nonce []byte) ([]byte, error) {
	// Direct /dev/sev-guest ioctl for report generation
	// SNP_GET_REPORT ioctl number: 0xc0105300
	// For now, return error — snpguest tool is the supported path
	_ = nonce
	return nil, fmt.Errorf("direct sev-guest ioctl not yet implemented; install snpguest")
}

func fetchVCEKCertChain() string {
	tmpDir, err := os.MkdirTemp("", "snp-certs-*")
	if err != nil {
		return ""
	}
	defer os.RemoveAll(tmpDir)

	cmd := exec.Command("snpguest", "fetch", "vcek", "pem", tmpDir)
	if err := cmd.Run(); err != nil {
		return ""
	}

	cert, err := os.ReadFile(tmpDir + "/vcek.pem")
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(cert))
}

func generateTDXReport(nonce []byte) AttestationReport {
	nonceHex := hex.EncodeToString(nonce)

	// Intel TDX via configfs-tsm interface
	tsmPath := "/sys/kernel/config/tsm/report/ryvion"
	os.MkdirAll(tsmPath, 0700)
	defer os.RemoveAll(tsmPath)

	if err := os.WriteFile(tsmPath+"/inblob", nonce, 0600); err != nil {
		return AttestationReport{Method: 2, TEEType: "tdx"}
	}

	quote, err := os.ReadFile(tsmPath + "/outblob")
	if err != nil {
		return AttestationReport{Method: 2, TEEType: "tdx"}
	}

	return AttestationReport{
		Method:    2,
		TEEType:   "tdx",
		ReportB64: base64.StdEncoding.EncodeToString(quote),
		NonceHex:  nonceHex,
	}
}
