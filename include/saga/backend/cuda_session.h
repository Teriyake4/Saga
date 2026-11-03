#pragma once

#include "saga/backend/cuda.h"
#include "saga/runtime/startup.h"


#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stop_token>
#include <thread>
#include <unordered_map>
#include <vector>

namespace saga::backend::cuda {

class CudaSession;

// Callback-scoped capability for copies of exact startup-selected preparations. References must
// not escape callbacks; an inactive or non-owner-thread use is rejected. There is deliberately no
// allocation, registry, launch-configuration, or raw-executor access.
class SessionExecution final {
public:
    SessionExecution(const SessionExecution&) = delete;
    SessionExecution& operator=(const SessionExecution&) = delete;
    SessionExecution(SessionExecution&&) = delete;
    SessionExecution& operator=(SessionExecution&&) = delete;
    void launch(const backend::PreparedKernel&, const ProjectionRequest&, const Buffer& weights,
                const Buffer& activations, const Buffer& output);
    void copy_to_device(const Buffer&, std::uint64_t, std::span<const std::byte>);
    void copy_to_device(const Buffer&, std::uint64_t, const Buffer&, std::uint64_t, std::uint64_t);
    void copy_to_host(std::span<std::byte>, const Buffer&, std::uint64_t = 0);
    void copy_to_host(const Buffer&, std::uint64_t, const Buffer&, std::uint64_t, std::uint64_t);
    void synchronize();

private:
    friend class CudaBackend;
    friend class CudaSession;
    class CallbackScope final {
    public:
        explicit CallbackScope(SessionExecution&);
        CallbackScope(const CallbackScope&) = delete;
        CallbackScope& operator=(const CallbackScope&) = delete;
        ~CallbackScope();
    private:
        SessionExecution& execution_;
    };
    SessionExecution(Executor&, std::vector<backend::PreparedKernel>);
    CallbackScope activate_for_callback();
    void require_active() const;
    Executor* executor_{};
    std::vector<backend::PreparedKernel> selected_;
    const std::thread::id owner_thread_;
    std::atomic<bool> active_{false};
};

// Contains canonical granted allocation IDs only; zero-sized declarations have an empty Buffer.
// Binder/callback implementations must not retain Buffer copies for bringup-lifetime entries.
using SessionResources = std::unordered_map<std::string, Buffer>;

struct PreparedSession {
    std::shared_ptr<const void> metadata;
    std::function<void(SessionExecution&, const SessionResources&, std::stop_token)> warmup;
    std::function<runtime::OwningStepResult(SessionExecution&, const SessionResources&,
        const runtime::PrefillRequest&, std::stop_token)> prefill;
    std::function<runtime::OwningStepResult(SessionExecution&, const SessionResources&,
        const runtime::DecodeRequest&, std::stop_token)> decode;
};

// A startup-only binder. It receives the immutable selected plan and the exact, declared
// project allocations. It cannot submit work; callbacks receive only SessionExecution.
// Retained plan-derived state must be copied into PreparedSession metadata/callback captures;
// neither the plan, grant, nor borrowed resource table may be retained by reference.
using SessionBinder = std::function<PreparedSession(
    const runtime::ExecutionPlan&, const SessionResources&)>;

class CudaBackend final : public runtime::IGpuBackend {
public:
    CudaBackend(int device_ordinal, SessionBinder binder, ExecutorOptions options = {});
    ~CudaBackend() override;
    CudaBackend(const CudaBackend&) = delete;
    CudaBackend& operator=(const CudaBackend&) = delete;

    backend::DeviceFacts device_facts() const override;
    std::unique_ptr<runtime::IExecutionSession> instantiate(
        const runtime::ExecutionPlan&, memory::BringupGrant&) override;
    bool healthy() const noexcept;
    std::optional<FenceFailureReport> failure_report() const;
    void request_cancel() noexcept;

private:
    friend class CudaSession;
    struct State;
    std::shared_ptr<State> state_;
};

} // namespace saga::backend::cuda
