package modelpolicy

import (
	"strings"

	v7hardware "github.com/Ryvion/node-agent/internal/v7/hardware"
)

type DerivedPolicyInput struct {
	BasePolicy Policy
	Hardware   v7hardware.CapacityInventory
}

func BuildDerivedPolicy(input DerivedPolicyInput) Policy {
	policy := NormalizePolicy(input.BasePolicy)
	hardware := v7hardware.NormalizeInventory(input.Hardware)

	if isDarwinApple(hardware) {
		policy.RuntimePolicy.DenyFamilies = appendIfMissing(policy.RuntimePolicy.DenyFamilies, "phi")
	}

	if capBytes := runtimeHardwareByteCap(hardware); capBytes > 0 && capBytes < policy.RuntimePolicy.MaxRuntimeModelBytes {
		policy.RuntimePolicy.MaxRuntimeModelBytes = capBytes
	}
	if hardware.GPUVRAMBytes >= 24*bytesPerGiB || hardware.SystemRAMBytes >= 96*bytesPerGiB {
		if policy.RuntimePolicy.MaxConcurrentInferenceJobs < 2 {
			policy.RuntimePolicy.MaxConcurrentInferenceJobs = 2
		}
		if policy.RuntimePolicy.MaxWarmModels < 2 {
			policy.RuntimePolicy.MaxWarmModels = 2
		}
	}
	return NormalizePolicy(policy)
}

func runtimeHardwareByteCap(hardware v7hardware.CapacityInventory) uint64 {
	if hardware.GPUVRAMBytes > 0 {
		return (hardware.GPUVRAMBytes * 3) / 4
	}
	if hardware.UnifiedMemory && hardware.SystemRAMBytes > 0 {
		return hardware.SystemRAMBytes / 2
	}
	if hardware.SystemRAMBytes > 0 {
		return hardware.SystemRAMBytes / 2
	}
	return 0
}

func isDarwinApple(hardware v7hardware.CapacityInventory) bool {
	return strings.EqualFold(strings.TrimSpace(hardware.OS), "darwin") &&
		(strings.EqualFold(strings.TrimSpace(hardware.GPUVendor), v7hardware.GPUVendorApple) ||
			hardware.MetalAvailable ||
			hardware.UnifiedMemory)
}

func appendIfMissing(values []string, value string) []string {
	value = cleanPolicyText(strings.ToLower(value), maxPolicyCompactLen)
	if value == "" {
		return values
	}
	for _, existing := range values {
		if strings.EqualFold(strings.TrimSpace(existing), value) {
			return values
		}
	}
	return append(values, value)
}
