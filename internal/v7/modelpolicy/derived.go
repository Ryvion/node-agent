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
	if isWindowsHardware(hardware) {
		policy = deriveWindowsRuntimePolicy(policy, hardware)
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

func deriveWindowsRuntimePolicy(policy Policy, hardware v7hardware.CapacityInventory) Policy {
	if !hasUsableWindowsAcceleration(hardware) {
		return policy
	}

	capBytes := runtimeHardwareByteCap(hardware)
	if capBytes > policy.RuntimePolicy.MaxRuntimeModelBytes &&
		policy.RuntimePolicy.MaxRuntimeModelBytes == DefaultRuntimeMaxModelGB*bytesPerGiB {
		policy.RuntimePolicy.MaxRuntimeModelBytes = capBytes
	}

	if capBytes >= 10*bytesPerGiB && hardware.SystemRAMBytes >= 32*bytesPerGiB {
		policy.RuntimePolicy.AllowFamilies = mergeFamilies(policy.RuntimePolicy.AllowFamilies, policy.AllowedFamilies)
		if policy.RuntimePolicy.MaxRuntimeParameterCountBillions < 14 {
			policy.RuntimePolicy.MaxRuntimeParameterCountBillions = 14
		}
		policy.RuntimePolicy.AllowLargeModels = true
		policy.RuntimePolicy.RequireExplicitAllowForLargeModels = false
	}
	if capBytes >= 18*bytesPerGiB && hardware.SystemRAMBytes >= 48*bytesPerGiB {
		if policy.RuntimePolicy.MaxRuntimeParameterCountBillions < 32 {
			policy.RuntimePolicy.MaxRuntimeParameterCountBillions = 32
		}
		if policy.RuntimePolicy.MaxConcurrentInferenceJobs < 2 {
			policy.RuntimePolicy.MaxConcurrentInferenceJobs = 2
		}
		if policy.RuntimePolicy.MaxWarmModels < 2 {
			policy.RuntimePolicy.MaxWarmModels = 2
		}
	}
	return policy
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

func isWindowsHardware(hardware v7hardware.CapacityInventory) bool {
	return strings.EqualFold(strings.TrimSpace(hardware.OS), "windows")
}

func hasUsableWindowsAcceleration(hardware v7hardware.CapacityInventory) bool {
	if !isWindowsHardware(hardware) || !hardware.GPUDetected || hardware.GPUVRAMBytes == 0 {
		return false
	}
	return hardware.CUDAAvailable || hardware.VulkanAvailable || hardware.DirectMLAvailable
}

func mergeFamilies(left, right []string) []string {
	out := cloneStrings(left)
	for _, family := range right {
		out = appendIfMissing(out, family)
	}
	return out
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
