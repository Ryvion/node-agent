package modelpolicy

import (
	"strings"

	v7hardware "github.com/Ryvion/ryvion-node/internal/v7/hardware"
)

type DerivedPolicyInput struct {
	BasePolicy Policy
	Hardware   v7hardware.CapacityInventory
}

func BuildDerivedPolicy(input DerivedPolicyInput) Policy {
	policy := NormalizePolicy(input.BasePolicy)
	hardware := v7hardware.NormalizeInventory(input.Hardware)

	if isConstrainedAppleUnifiedMemory(hardware) {
		policy.RuntimePolicy.DenyFamilies = appendIfMissing(policy.RuntimePolicy.DenyFamilies, "phi")
	}
	if hasUsableAcceleratedRuntime(hardware) {
		policy = deriveAcceleratedRuntimePolicy(policy, hardware)
	}

	if capBytes := runtimePolicyByteCap(policy, hardware); capBytes > 0 && capBytes < policy.RuntimePolicy.MaxRuntimeModelBytes {
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

func deriveAcceleratedRuntimePolicy(policy Policy, hardware v7hardware.CapacityInventory) Policy {
	capBytes := runtimePolicyByteCap(policy, hardware)
	if capBytes > policy.RuntimePolicy.MaxRuntimeModelBytes &&
		policy.RuntimePolicy.MaxRuntimeModelBytes == DefaultRuntimeMaxModelGB*bytesPerGiB {
		policy.RuntimePolicy.MaxRuntimeModelBytes = capBytes
	}
	if capBytes > policy.MaxSingleModelBytes &&
		policy.MaxSingleModelBytes == DefaultMaxSingleModelGB*bytesPerGiB {
		policy.MaxSingleModelBytes = capBytes
	}
	if !policy.AutoDownload {
		policy.AllowLicenseRestricted = true
	}

	if capBytes >= 10*bytesPerGiB && hardware.SystemRAMBytes >= 30*bytesPerGiB {
		policy.RuntimePolicy.AllowFamilies = mergeFamilies(policy.RuntimePolicy.AllowFamilies, policy.AllowedFamilies)
		policy.RuntimePolicy.AllowFamilies = appendIfMissing(policy.RuntimePolicy.AllowFamilies, "phi")
		if policy.RuntimePolicy.MaxRuntimeParameterCountBillions < 16 {
			policy.RuntimePolicy.MaxRuntimeParameterCountBillions = 16
		}
		policy.RuntimePolicy.AllowLargeModels = true
		policy.RuntimePolicy.RequireExplicitAllowForLargeModels = false
	}
	if capBytes >= 18*bytesPerGiB && hardware.SystemRAMBytes >= 30*bytesPerGiB {
		if policy.RuntimePolicy.MaxRuntimeParameterCountBillions < 32 {
			policy.RuntimePolicy.MaxRuntimeParameterCountBillions = 32
		}
	}
	if capBytes >= 18*bytesPerGiB && hardware.SystemRAMBytes >= 48*bytesPerGiB {
		if policy.RuntimePolicy.MaxConcurrentInferenceJobs < 2 {
			policy.RuntimePolicy.MaxConcurrentInferenceJobs = 2
		}
		if policy.RuntimePolicy.MaxWarmModels < 2 {
			policy.RuntimePolicy.MaxWarmModels = 2
		}
	}
	return policy
}

func runtimePolicyByteCap(policy Policy, hardware v7hardware.CapacityInventory) uint64 {
	capBytes := runtimeHardwareByteCap(hardware)
	if policy.RuntimePolicy.AllowCPUOffload && hasDiscreteAcceleratedGPU(hardware) && hardware.SystemRAMBytes > 0 {
		if offloadCap := (hardware.SystemRAMBytes * 3) / 5; offloadCap > capBytes {
			capBytes = offloadCap
		}
	}
	return capBytes
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

func hasDiscreteAcceleratedGPU(hardware v7hardware.CapacityInventory) bool {
	if !hardware.GPUDetected || hardware.GPUVRAMBytes == 0 || hardware.UnifiedMemory {
		return false
	}
	return hardware.CUDAAvailable || hardware.VulkanAvailable || hardware.DirectMLAvailable
}

func isDarwinApple(hardware v7hardware.CapacityInventory) bool {
	return strings.EqualFold(strings.TrimSpace(hardware.OS), "darwin") &&
		(strings.EqualFold(strings.TrimSpace(hardware.GPUVendor), v7hardware.GPUVendorApple) ||
			hardware.MetalAvailable ||
			hardware.UnifiedMemory)
}

func isConstrainedAppleUnifiedMemory(hardware v7hardware.CapacityInventory) bool {
	if !isDarwinApple(hardware) {
		return false
	}
	return runtimeHardwareByteCap(hardware) < 10*bytesPerGiB || hardware.SystemRAMBytes < 30*bytesPerGiB
}

func hasUsableAcceleratedRuntime(hardware v7hardware.CapacityInventory) bool {
	if isDarwinApple(hardware) {
		return hardware.MetalAvailable && hardware.UnifiedMemory && runtimeHardwareByteCap(hardware) >= 10*bytesPerGiB
	}
	if !hardware.GPUDetected || hardware.GPUVRAMBytes == 0 {
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
