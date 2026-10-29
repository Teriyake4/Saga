#include "saga/backend/device_profile.h"

#include <algorithm>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace saga::backend {
namespace {
constexpr std::string_view rtx4080_profile_identity = "rtx4080-sm89-76sm-warp32";
constexpr std::string_view rtx4080_product_name = "NVIDIA GeForce RTX 4080";
constexpr std::string_view rtx4080_product_alias = "GeForce RTX 4080";

bool valid_device_kind(DeviceKind kind) {
    return kind == DeviceKind::cpu || kind == DeviceKind::accelerator;
}

void validate_preferences(const std::vector<std::string>& preferences, std::string_view owner) {
    std::vector<std::string_view> seen;
    seen.reserve(preferences.size());
    for (const auto& id : preferences) {
        if (id.empty()) throw std::invalid_argument(std::string(owner) + " contains an empty implementation preference");
        if (std::find(seen.begin(), seen.end(), id) != seen.end())
            throw std::invalid_argument(std::string(owner) + " contains a duplicate implementation preference: " + id);
        seen.push_back(id);
    }
}

void append_unique(std::vector<std::string>& target, const std::vector<std::string>& source) {
    for (const auto& id : source) {
        if (std::find(target.begin(), target.end(), id) == target.end()) target.push_back(id);
    }
}
} // namespace

DeviceProfile::DeviceProfile(DeviceProfileSpec spec)
    : identity_(std::move(spec.identity)), kind_(spec.kind), architecture_(std::move(spec.architecture)),
      architecture_version_(spec.architecture_version),
      physical_device_identity_(std::move(spec.physical_device_identity)),
      product_identity_(std::move(spec.product_identity)), warp_size_(spec.warp_size),
      multiprocessor_count_(spec.multiprocessor_count),
      minimum_device_memory_budget_bytes_(spec.minimum_device_memory_budget_bytes),
      minimum_workspace_budget_bytes_(spec.minimum_workspace_budget_bytes),
      preferred_implementations_(std::move(spec.preferred_implementations)) {
    if (identity_.empty() || architecture_.empty() || !valid_device_kind(kind_))
        throw std::invalid_argument("device profile requires nonempty identity/architecture and a valid device kind");
    if ((warp_size_ && *warp_size_ == 0) ||
        (multiprocessor_count_ && *multiprocessor_count_ == 0) ||
        (minimum_device_memory_budget_bytes_ && *minimum_device_memory_budget_bytes_ == 0) ||
        (minimum_workspace_budget_bytes_ && *minimum_workspace_budget_bytes_ == 0))
        throw std::invalid_argument("device profile limits must be positive when specified");
    validate_preferences(preferred_implementations_, "device profile");

    if (identity_ == rtx4080_profile_identity &&
        (kind_ != DeviceKind::accelerator ||
         (product_identity_ != rtx4080_product_name && product_identity_ != rtx4080_product_alias) ||
         physical_device_identity_.empty() || physical_device_identity_ == product_identity_ ||
         architecture_ != "sm89" || architecture_version_ != 89 || warp_size_ != 32 ||
         multiprocessor_count_ != 76)) {
        throw std::invalid_argument(
            "reserved RTX 4080 profile identity requires accepted product identity, physical device identity, "
            "accelerator, SM89/version 89, warp size 32, and 76 SMs");
    }
}

bool DeviceProfile::matches(const DeviceFacts& facts) const noexcept {
    if (facts.caps.kind != kind_ || facts.caps.architecture != architecture_ ||
        facts.caps.architecture_version != architecture_version_) return false;
    if (!physical_device_identity_.empty() && facts.caps.device_identity != physical_device_identity_) return false;
    if (!product_identity_.empty() && facts.product_identity != product_identity_) return false;
    if (warp_size_ && facts.warp_size != warp_size_) return false;
    if (multiprocessor_count_ && facts.multiprocessor_count != multiprocessor_count_) return false;
    if (minimum_device_memory_budget_bytes_ &&
        (!facts.device_memory_budget_bytes ||
         *facts.device_memory_budget_bytes < *minimum_device_memory_budget_bytes_)) return false;
    if (minimum_workspace_budget_bytes_ &&
        (!facts.workspace_budget_bytes ||
         *facts.workspace_budget_bytes < *minimum_workspace_budget_bytes_)) return false;
    return true;
}

DeviceProfile make_rtx4080_profile(const DeviceFacts& observed_facts) {
    DeviceProfileSpec spec;
    spec.identity = rtx4080_profile_identity;
    spec.kind = observed_facts.caps.kind;
    spec.architecture = observed_facts.caps.architecture;
    spec.architecture_version = observed_facts.caps.architecture_version;
    spec.physical_device_identity = observed_facts.caps.device_identity;
    spec.product_identity = observed_facts.product_identity;
    spec.warp_size = observed_facts.warp_size;
    spec.multiprocessor_count = observed_facts.multiprocessor_count;
    return DeviceProfile(std::move(spec));
}

PairProfile::PairProfile(PairProfileSpec spec)
    : model_identity_(std::move(spec.model_identity)),
      device_profile_identity_(std::move(spec.device_profile_identity)),
      physical_device_identity_(std::move(spec.physical_device_identity)),
      preferred_implementations_(std::move(spec.preferred_implementations)) {
    if (model_identity_.empty() || device_profile_identity_.empty() || physical_device_identity_.empty())
        throw std::invalid_argument("pair profile requires model, device-profile, and physical-device identities");
    validate_preferences(preferred_implementations_, "pair profile");
}

PreparedProgram resolve_program(std::string_view model_identity, const ModelProgram& program,
                                const DeviceFacts& facts, const DeviceProfile& device_profile,
                                const KernelRegistry& registry, const PairProfile* pair_profile) {
    if (model_identity.empty()) throw std::invalid_argument("model identity must be nonempty");
    if (program.alternatives.size() > maximum_program_alternatives)
        throw std::invalid_argument("model program exceeds maximum approved alternative count");

    const auto valid_phase = [](Phase phase) {
        return phase == Phase::prefill || phase == Phase::decode || phase == Phase::verify;
    };
    std::size_t total_calls = 0;
    const auto validate_calls = [&](const std::vector<KernelRequest>& calls, std::string_view identity) {
        if (calls.size() > maximum_program_calls - total_calls)
            throw std::invalid_argument("model program exceeds maximum total call count");
        total_calls += calls.size();
        if (!valid_phase(program.phase)) throw std::invalid_argument("model program has unknown phase");
        for (std::size_t i = 0; i < calls.size(); ++i) {
            if (calls[i].phase != program.phase)
                throw std::invalid_argument("program call phase mismatch in candidate: " + std::string(identity));
            if (auto error = validate_request(calls[i]))
                throw std::invalid_argument("invalid program call " + std::to_string(i) + " in " +
                                            std::string(identity) + ": " + *error);
        }
    };

    if (program.calls.empty()) throw std::invalid_argument("primary model program must contain at least one call");
    validate_calls(program.calls, "primary");
    std::vector<std::string_view> alternative_ids;
    alternative_ids.reserve(program.alternatives.size());
    for (const auto& alternative : program.alternatives) {
        if (alternative.identity.empty() || alternative.identity == "primary")
            throw std::invalid_argument("program alternative identity must be nonempty and not 'primary'");
        if (std::find(alternative_ids.begin(), alternative_ids.end(), alternative.identity) != alternative_ids.end())
            throw std::invalid_argument("duplicate program alternative identity: " + alternative.identity);
        if (alternative.calls.empty())
            throw std::invalid_argument("program alternative must contain at least one call: " + alternative.identity);
        alternative_ids.push_back(alternative.identity);
        validate_calls(alternative.calls, alternative.identity);
    }

    if (!device_profile.matches(facts))
        throw std::invalid_argument("observed device facts do not match device profile: " + device_profile.identity());
    if (pair_profile && (pair_profile->model_identity() != model_identity ||
                         pair_profile->device_profile_identity() != device_profile.identity() ||
                         pair_profile->physical_device_identity() != facts.caps.device_identity)) {
        throw std::invalid_argument("pair profile identity mismatch for model/device profile/physical device");
    }

    PairTuningProfile merged_preferences;
    if (pair_profile) append_unique(merged_preferences.preferred_implementations,
                                    pair_profile->preferred_implementations());
    append_unique(merged_preferences.preferred_implementations,
                  device_profile.preferred_implementations());

    const auto prepare_candidate = [&](std::string_view identity,
                                       const std::vector<KernelRequest>& calls) {
        std::vector<PreparedKernel> kernels;
        kernels.reserve(calls.size());
        for (const auto& request : calls)
            kernels.push_back(registry.resolve(request, facts.caps, &merged_preferences));
        return PreparedProgram(std::string(identity), std::move(kernels));
    };

    std::string unavailable;
    try {
        return prepare_candidate("primary", program.calls);
    } catch (const KernelUnavailable& error) {
        unavailable = "primary: " + std::string(error.what());
    }
    for (const auto& alternative : program.alternatives) {
        try {
            return prepare_candidate(alternative.identity, alternative.calls);
        } catch (const KernelUnavailable& error) {
            unavailable += "; " + alternative.identity + ": " + error.what();
        }
    }
    throw KernelUnavailable("no compatible primary or approved program alternative: " + unavailable);
}

} // namespace saga::backend
