#pragma once

#include "saga/backend/contracts.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace saga::backend {

// Facts must be populated by the host/backend from observed runtime properties. Product
// identity is separate from caps.device_identity (the physical UUID); a reported name can
// scope a named profile but is not hardware attestation or performance qualification.
struct DeviceFacts {
    DeviceCaps caps;
    std::string product_identity;
    std::optional<std::uint32_t> warp_size;
    std::optional<std::uint32_t> multiprocessor_count;
    std::optional<std::uint64_t> device_memory_budget_bytes;
    std::optional<std::uint64_t> workspace_budget_bytes;
};

struct DeviceProfileSpec {
    std::string identity;
    DeviceKind kind{DeviceKind::cpu};
    std::string architecture;
    std::uint32_t architecture_version{0};
    std::string physical_device_identity;
    std::string product_identity;
    std::optional<std::uint32_t> warp_size;
    std::optional<std::uint32_t> multiprocessor_count;
    std::optional<std::uint64_t> minimum_device_memory_budget_bytes;
    std::optional<std::uint64_t> minimum_workspace_budget_bytes;
    std::vector<std::string> preferred_implementations;
};

class DeviceProfile {
public:
    explicit DeviceProfile(DeviceProfileSpec spec);

    const std::string& identity() const noexcept { return identity_; }
    DeviceKind kind() const noexcept { return kind_; }
    const std::string& architecture() const noexcept { return architecture_; }
    std::uint32_t architecture_version() const noexcept { return architecture_version_; }
    const std::string& physical_device_identity() const noexcept { return physical_device_identity_; }
    const std::string& product_identity() const noexcept { return product_identity_; }
    const std::optional<std::uint32_t>& warp_size() const noexcept { return warp_size_; }
    const std::optional<std::uint32_t>& multiprocessor_count() const noexcept { return multiprocessor_count_; }
    const std::optional<std::uint64_t>& minimum_device_memory_budget_bytes() const noexcept {
        return minimum_device_memory_budget_bytes_;
    }
    const std::optional<std::uint64_t>& minimum_workspace_budget_bytes() const noexcept {
        return minimum_workspace_budget_bytes_;
    }
    const std::vector<std::string>& preferred_implementations() const noexcept {
        return preferred_implementations_;
    }
    bool matches(const DeviceFacts& facts) const noexcept;

private:
    const std::string identity_;
    const DeviceKind kind_;
    const std::string architecture_;
    const std::uint32_t architecture_version_;
    const std::string physical_device_identity_;
    const std::string product_identity_;
    const std::optional<std::uint32_t> warp_size_;
    const std::optional<std::uint32_t> multiprocessor_count_;
    const std::optional<std::uint64_t> minimum_device_memory_budget_bytes_;
    const std::optional<std::uint64_t> minimum_workspace_budget_bytes_;
    const std::vector<std::string> preferred_implementations_;
};

// This named constraint set is limited to the original RTX 4080's observed product,
// SM89/version 89, 76-SM, warp-32 topology. It is not qualification evidence.
DeviceProfile make_rtx4080_profile(const DeviceFacts& observed_facts);

struct PairProfileSpec {
    std::string model_identity;
    std::string device_profile_identity;
    std::string physical_device_identity;
    std::vector<std::string> preferred_implementations;
};

// Pair preferences rank only implementations that pass KernelRegistry's complete guards.
class PairProfile {
public:
    explicit PairProfile(PairProfileSpec spec);

    const std::string& model_identity() const noexcept { return model_identity_; }
    const std::string& device_profile_identity() const noexcept { return device_profile_identity_; }
    const std::string& physical_device_identity() const noexcept { return physical_device_identity_; }
    const std::vector<std::string>& preferred_implementations() const noexcept {
        return preferred_implementations_;
    }

private:
    const std::string model_identity_;
    const std::string device_profile_identity_;
    const std::string physical_device_identity_;
    const std::vector<std::string> preferred_implementations_;
};

class PreparedProgram {
public:
    std::span<const PreparedKernel> kernels() const noexcept { return kernels_; }
    std::size_t size() const noexcept { return kernels_.size(); }
    const std::string& selected_alternative_identity() const noexcept { return selected_alternative_identity_; }

private:
    friend PreparedProgram resolve_program(std::string_view, const ModelProgram&, const DeviceFacts&,
                                           const DeviceProfile&, const KernelRegistry&,
                                           const PairProfile*);
    PreparedProgram(std::string identity, std::vector<PreparedKernel> kernels)
        : selected_alternative_identity_(std::move(identity)), kernels_(std::move(kernels)) {}
    std::string selected_alternative_identity_;
    std::vector<PreparedKernel> kernels_;
};

// Startup-only binding for a finite native call list. This prepares each call once;
// it does not interpret a graph or launch kernels.
PreparedProgram resolve_program(std::string_view model_identity, const ModelProgram& program,
                                const DeviceFacts& facts, const DeviceProfile& device_profile,
                                const KernelRegistry& registry,
                                const PairProfile* pair_profile = nullptr);

} // namespace saga::backend
