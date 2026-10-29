#pragma once

#include "saga/backend/device_profile.h"
#include "saga/runtime/model_contracts.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace saga::runtime {

// Immutable, owning snapshot passed to backend construction. Identity is an explicit canonical
// serialization (not a hash or qualification claim) of the selected startup inputs.
class ExecutionPlan final {
public:
    const ModelDescription& description() const noexcept { return description_; }
    const RuntimeOptions& options() const noexcept { return options_; }
    const backend::DeviceFacts& device_facts() const noexcept { return facts_; }
    const backend::DeviceProfile& device_profile() const noexcept { return device_profile_; }
    const std::optional<backend::PairProfile>& pair_profile() const noexcept { return pair_profile_; }
    const std::vector<backend::PreparedProgram>& programs() const noexcept { return programs_; }
    const memory::StaticCandidate& candidate() const noexcept { return candidate_; }
    const std::string& identity() const noexcept { return identity_; }
    const std::vector<std::string>& context_attempt_diagnostics() const noexcept { return context_attempt_diagnostics_; }
private:
    friend class StartupCoordinator;
    ExecutionPlan(ModelDescription, RuntimeOptions, backend::DeviceFacts, backend::DeviceProfile,
                  std::optional<backend::PairProfile>, std::vector<backend::PreparedProgram>,
                  memory::StaticCandidate, std::string, std::vector<std::string>);
    ModelDescription description_;
    RuntimeOptions options_;
    backend::DeviceFacts facts_;
    backend::DeviceProfile device_profile_;
    std::optional<backend::PairProfile> pair_profile_;
    std::vector<backend::PreparedProgram> programs_;
    memory::StaticCandidate candidate_;
    std::string identity_;
    std::vector<std::string> context_attempt_diagnostics_;
};

struct StartupRequest {
    std::string model_id;
    std::string backend_id;
    ArtifactMetadata artifact;
    RuntimeOptions options; // context_tokens is required and is never silently reduced
    memory::Budgets budgets{};
    memory::GraphMode graphs{memory::GraphMode::disabled};
    backend::DeviceProfile device_profile;
    std::optional<backend::PairProfile> pair_profile;
    const backend::KernelRegistry* kernels{};
    // Used only when context_tokens == 0. Startup tries unique values in descending order and
    // selects the highest statically eligible candidate. Warmup/capture/reconcile failures are
    // propagated; they do not trigger a lower-context retry without a typed capacity diagnosis.
    std::vector<std::uint64_t> auto_contexts;
};

class ApprovedRuntime;

// Synchronous, externally serialized startup composition. The coordinator, registries, grant,
// approval and returned owner are single-owner/thread-serialized; do not destroy them from a
// session callback. No timeout is a physical fence: backends must implement drain() as a blocking
// barrier before returning from any failure path and retain/drain work if construction throws.
class StartupCoordinator final {
public:
    StartupCoordinator(const ModelRegistry&, const BackendRegistry&);
    ApprovedRuntime start(const StartupRequest&) const;
private:
    const ModelRegistry& models_;
    const BackendRegistry& backends_;
};

class ApprovedRuntime final {
public:
    ApprovedRuntime(const ApprovedRuntime&) = delete;
    ApprovedRuntime& operator=(const ApprovedRuntime&) = delete;
    ApprovedRuntime(ApprovedRuntime&&) noexcept;
    ApprovedRuntime& operator=(ApprovedRuntime&&) noexcept;
    ~ApprovedRuntime() noexcept;
    IModelRuntime& runtime() const;
    const ExecutionPlan& plan() const;
    const memory::ReadinessApproval& approval() const;
private:
    friend class StartupCoordinator;
    ApprovedRuntime(std::shared_ptr<const IModelDefinition>, std::shared_ptr<IGpuBackend>,
                    std::unique_ptr<IExecutionSession>, std::unique_ptr<IModelRuntime>,
                    std::shared_ptr<const ExecutionPlan>, memory::ReadinessApproval);
    void reset() noexcept;
    std::shared_ptr<const IModelDefinition> model_;
    std::shared_ptr<IGpuBackend> backend_;
    std::unique_ptr<IExecutionSession> session_;
    std::unique_ptr<IModelRuntime> runtime_;
    std::shared_ptr<const ExecutionPlan> plan_;
    std::optional<memory::ReadinessApproval> approval_;
};

} // namespace saga::runtime
