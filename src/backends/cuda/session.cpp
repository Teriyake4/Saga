#include "saga/backend/cuda_session.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace saga::backend::cuda {
namespace {

struct QuarantineRoot {
    std::shared_ptr<void> retained;
    QuarantineRoot* next{};
};
std::mutex& quarantine_mutex() {
    static auto* mutex = new std::mutex;
    return *mutex;
}
QuarantineRoot*& quarantine_roots() {
    static QuarantineRoot* roots{};
    return roots;
}

QuarantineRoot* register_quarantine_root() {
    auto* root = new QuarantineRoot;
    std::lock_guard lock(quarantine_mutex());
    root->next = quarantine_roots();
    quarantine_roots() = root;
    return root;
}

void unregister_quarantine_root(QuarantineRoot* root) noexcept {
    if (!root) return;
    std::lock_guard lock(quarantine_mutex());
    if (root->retained) return;
    auto** cursor = &quarantine_roots();
    while (*cursor && *cursor != root) cursor = &(*cursor)->next;
    if (*cursor == root) {
        *cursor = root->next;
        delete root;
    }
}

runtime::Completion completed_completion() {
    runtime::CompletionSource source;
    auto result = source.completion();
    source.succeed();
    return result;
}

std::uint64_t step_frontier(const std::vector<runtime::StepRow>& rows) {
    std::uint64_t frontier = 0;
    for (const auto& row : rows) {
        if (row.positions.size() != row.token_ids.size())
            throw std::invalid_argument("CUDA session request positions do not match token IDs");
        for (auto position : row.positions) {
            if (position == UINT64_MAX) throw std::overflow_error("CUDA session token frontier overflows");
            frontier = std::max(frontier, position + 1);
        }
    }
    return frontier;
}

bool same_device(const backend::DeviceFacts& a, const backend::DeviceFacts& b) {
    return a.caps.kind == b.caps.kind && a.caps.architecture == b.caps.architecture &&
        a.caps.architecture_version == b.caps.architecture_version &&
        a.caps.features == b.caps.features && a.caps.device_identity == b.caps.device_identity &&
        a.product_identity == b.product_identity && a.warp_size == b.warp_size &&
        a.multiprocessor_count == b.multiprocessor_count &&
        a.device_memory_budget_bytes == b.device_memory_budget_bytes &&
        a.workspace_budget_bytes == b.workspace_budget_bytes;
}

struct RetainedStep {
    std::shared_ptr<void> session;
    runtime::ResourceLease caller;
    runtime::PrefillRequest prefill;
    runtime::DecodeRequest decode;
};

} // namespace

struct CudaBackend::State {
    State(int ordinal, SessionBinder bind, ExecutorOptions opts)
        : executor(std::make_shared<Executor>(ordinal, opts)), binder(std::move(bind)),
          owner_thread(std::this_thread::get_id()) {}

    std::shared_ptr<Executor> executor;
    SessionBinder binder;
    std::atomic<bool> poisoned{false};
    const std::thread::id owner_thread;
    mutable std::mutex cancel_mutex;
    std::stop_source active_stop;
    bool step_active{};
    bool cancel_requested{};
    std::mutex session_mutex;
    std::weak_ptr<void> active_session;
    bool constructing_session{};
};

SessionExecution::SessionExecution(Executor& executor, std::vector<backend::PreparedKernel> selected)
    : executor_(&executor), selected_(std::move(selected)), owner_thread_(std::this_thread::get_id()) {}

SessionExecution::CallbackScope::CallbackScope(SessionExecution& execution) : execution_(execution) {
    if (std::this_thread::get_id() != execution_.owner_thread_)
        throw std::logic_error("CUDA callback capability activated from a non-owner thread");
    bool inactive = false;
    if (!execution_.active_.compare_exchange_strong(inactive, true, std::memory_order_release,
                                                     std::memory_order_relaxed))
        throw std::logic_error("CUDA callback capability is already active");
}
SessionExecution::CallbackScope::~CallbackScope() {
    execution_.active_.store(false, std::memory_order_release);
}
SessionExecution::CallbackScope SessionExecution::activate_for_callback() {
    return CallbackScope(*this);
}
void SessionExecution::require_active() const {
    if (std::this_thread::get_id() != owner_thread_)
        throw std::logic_error("CUDA callback capability used from a non-owner thread");
    if (!active_.load(std::memory_order_acquire))
        throw std::logic_error("CUDA callback capability used outside its callback scope");
}

void SessionExecution::launch(const backend::PreparedKernel& kernel, const ProjectionRequest& request,
                              const Buffer& weights, const Buffer& activations, const Buffer& output) {
    require_active();
    const auto selected = std::find_if(selected_.begin(), selected_.end(),
        [&](const auto& prepared) { return prepared.same_preparation(kernel); });
    if (selected == selected_.end())
        throw std::invalid_argument("CUDA callback attempted to launch a kernel outside its startup-selected calls");
    executor_->launch(*selected, request, weights, activations, output);
}

void SessionExecution::copy_to_device(const Buffer& destination, std::uint64_t offset,
                                      std::span<const std::byte> source) {
    require_active();
    executor_->copy_to_device(destination, offset, source);
}
void SessionExecution::copy_to_device(const Buffer& destination, std::uint64_t destination_offset,
                                      const Buffer& source, std::uint64_t source_offset, std::uint64_t bytes) {
    require_active();
    executor_->copy_to_device(destination, destination_offset, source, source_offset, bytes);
}
void SessionExecution::copy_to_host(std::span<std::byte> destination, const Buffer& source,
                                    std::uint64_t offset) {
    require_active();
    executor_->copy_to_host(destination, source, offset);
}
void SessionExecution::copy_to_host(const Buffer& destination, std::uint64_t destination_offset,
                                    const Buffer& source, std::uint64_t source_offset, std::uint64_t bytes) {
    require_active();
    executor_->copy_to_host(destination, destination_offset, source, source_offset, bytes);
}
void SessionExecution::synchronize() {
    require_active();
    executor_->synchronize();
}

class CudaSession final : public runtime::IExecutionSession {
public:
    struct OwnedState {
        OwnedState() : quarantine_root(register_quarantine_root()) {}
        ~OwnedState() { unregister_quarantine_root(quarantine_root); }
        std::shared_ptr<CudaBackend::State> backend;
        SessionResources resources;
        PreparedSession prepared;
        std::vector<backend::PreparedProgram> programs;
        runtime::ModelInspection inspection;
        std::optional<memory::StaticCandidate> candidate;
        std::uint64_t max_context{};
        std::string overhead_id;
        std::vector<std::string> zero_overhead_ids;
        std::vector<std::string> bringup_ids;
        std::shared_ptr<RetainedStep> active_step;
        std::unique_ptr<SessionExecution> warmup_execution;
        std::unique_ptr<SessionExecution> prefill_execution;
        std::unique_ptr<SessionExecution> decode_execution;
        QuarantineRoot* quarantine_root{};
        std::atomic<bool> settled{true};
        bool warmed{};
    };

    explicit CudaSession(std::shared_ptr<OwnedState> state) : state_(std::move(state)) {}
    ~CudaSession() override { drain(); }

    runtime::PendingStep prefill(runtime::PrefillRequest request,
                                 runtime::ResourceLease lease) override {
        require_owner();
        if (!state_->prepared.prefill) throw std::runtime_error("CUDA session has no bound prefill callback");
        runtime::validate_step_request(request, state_->inspection.frontend, state_->max_context);
        return run<runtime::PrefillRequest>(std::move(request), std::move(lease), *state_->prefill_execution,
            [&](const auto& input, std::stop_token token) {
                return state_->prepared.prefill(*state_->prefill_execution, state_->resources, input, token);
            });
    }

    runtime::PendingStep decode(runtime::DecodeRequest request,
                                runtime::ResourceLease lease) override {
        require_owner();
        if (!state_->prepared.decode) throw std::runtime_error("CUDA session has no bound decode callback");
        runtime::validate_step_request(request, state_->inspection.frontend, state_->max_context);
        return run<runtime::DecodeRequest>(std::move(request), std::move(lease), *state_->decode_execution,
            [&](const auto& input, std::stop_token token) {
                return state_->prepared.decode(*state_->decode_execution, state_->resources, input, token);
            });
    }

    runtime::Completion warmup(memory::BringupGrant& grant) override {
        require_owner();
        require_grant(grant);
        if (!state_->prepared.warmup) throw std::runtime_error("CUDA session has no bound warmup callback");
        state_->settled.store(false, std::memory_order_release);
        try {
            {
                auto callback_scope = state_->warmup_execution->activate_for_callback();
                state_->prepared.warmup(*state_->warmup_execution, state_->resources, {});
            }
            if (!state_->backend->executor->drain()) poison();
            else if (!state_->backend->poisoned.load(std::memory_order_acquire))
                state_->settled.store(true, std::memory_order_release);
            if (!physically_settled()) throw std::runtime_error("CUDA warmup physical fence failed");
            require_grant(grant);
            for (const auto& id : state_->bringup_ids) state_->resources.erase(id);
            state_->warmed = true;
            return completed_completion();
        } catch (...) {
            if (!state_->backend->executor->drain()) poison();
            else if (!state_->backend->poisoned.load(std::memory_order_acquire))
                state_->settled.store(true, std::memory_order_release);
            throw;
        }
    }

    runtime::Completion capture(memory::BringupGrant&) override {
        require_owner();
        throw std::runtime_error("CUDA session graph capture is unsupported");
    }

    std::vector<std::pair<std::string, std::uint64_t>> overhead_measurements(
        memory::BringupGrant& grant) override {
        require_owner();
        require_grant(grant);
        if (!state_->warmed) throw std::runtime_error("CUDA overhead requires completed warmup");
        if (!physically_settled()) throw std::runtime_error("poisoned CUDA session cannot report overhead");
        std::vector<std::pair<std::string, std::uint64_t>> result;
        result.reserve(1 + state_->zero_overhead_ids.size());
        result.emplace_back(state_->overhead_id, state_->backend->executor->observe_runtime_overhead());
        for (const auto& id : state_->zero_overhead_ids) result.emplace_back(id, 0);
        return result;
    }

    void drain() noexcept override {
        if (!state_) return;
        state_->settled.store(false, std::memory_order_release);
        if (std::this_thread::get_id() != state_->backend->owner_thread ||
            !state_->backend->executor->drain()) poison();
        else if (!state_->backend->poisoned.load(std::memory_order_acquire))
            state_->settled.store(true, std::memory_order_release);
    }

    bool physically_settled() const noexcept override {
        return state_ && state_->settled.load(std::memory_order_acquire) &&
               !state_->backend->poisoned.load(std::memory_order_acquire) &&
               state_->backend->executor->healthy();
    }

private:
    void require_owner() const {
        if (std::this_thread::get_id() != state_->backend->owner_thread)
            throw std::logic_error("CUDA session operation called from a non-owner thread");
        if (state_->backend->poisoned.load(std::memory_order_acquire) ||
            !state_->backend->executor->healthy())
            throw std::runtime_error("CUDA backend is poisoned");
    }

    void require_grant(const memory::BringupGrant& grant) const {
        if (!state_->candidate || !grant.matches(*state_->candidate))
            throw std::invalid_argument("CUDA session grant no longer matches its selected candidate");
    }

    template<class Request, class Callback>
    runtime::PendingStep run(Request request, runtime::ResourceLease caller,
                             SessionExecution& execution, Callback&& callback) {
        require_owner();
        const auto frontier = step_frontier(request.rows);
        auto retained = std::make_shared<RetainedStep>();
        retained->session = state_;
        retained->caller = std::move(caller);
        if constexpr (std::is_same_v<Request, runtime::PrefillRequest>) retained->prefill = std::move(request);
        else retained->decode = std::move(request);
        const auto& input = [&]() -> const Request& {
            if constexpr (std::is_same_v<Request, runtime::PrefillRequest>) return retained->prefill;
            else return retained->decode;
        }();
        runtime::StepSource source(runtime::ResourceLease(retained),
            runtime::StepResultBounds(frontier, state_->inspection.frontend.vocabulary_size
                ? std::optional<std::size_t>(state_->inspection.frontend.vocabulary_size) : std::nullopt));
        auto pending = source.pending();
        std::stop_token token;
        {
            std::lock_guard lock(state_->backend->cancel_mutex);
            if (state_->backend->step_active)
                throw std::logic_error("CUDA session already has an active physical step");
            state_->backend->active_stop = std::stop_source{};
            state_->backend->cancel_requested = false;
            token = state_->backend->active_stop.get_token();
            state_->backend->step_active = true;
            state_->active_step = retained;
        }

        const auto finish_settled = [&](runtime::OwningStepResult* result, std::exception_ptr error) {
            std::lock_guard lock(state_->backend->cancel_mutex);
            if (state_->backend->cancel_requested) source.cancel();
            else if (error) source.fail(std::move(error));
            else if (!source.succeed(std::move(*result)))
                source.fail(std::make_exception_ptr(std::runtime_error(
                    "CUDA callback returned an invalid bounded step result")));
            state_->active_step.reset();
            state_->backend->step_active = false;
            state_->backend->cancel_requested = false;
            state_->backend->active_stop = std::stop_source{};
        };
        const auto finish_unsettled = [&] {
            poison();
            std::lock_guard lock(state_->backend->cancel_mutex);
            source.fail(std::make_exception_ptr(std::runtime_error("CUDA step physical fence failed")));
            state_->backend->step_active = false;
            state_->backend->cancel_requested = false;
            state_->backend->active_stop = std::stop_source{};
        };

        state_->settled.store(false, std::memory_order_release);
        try {
            if (token.stop_requested()) {
                if (!state_->backend->executor->drain()) finish_unsettled();
                else {
                    if (!state_->backend->poisoned.load(std::memory_order_acquire))
                        state_->settled.store(true, std::memory_order_release);
                    finish_settled(nullptr, {});
                }
                return pending;
            }
            auto result = [&] {
                auto callback_scope = execution.activate_for_callback();
                return callback(input, token);
            }();
            if (!state_->backend->executor->drain()) finish_unsettled();
            else {
                if (!state_->backend->poisoned.load(std::memory_order_acquire))
                    state_->settled.store(true, std::memory_order_release);
                finish_settled(&result, {});
            }
        } catch (...) {
            const auto error = std::current_exception();
            if (!state_->backend->executor->drain()) finish_unsettled();
            else {
                if (!state_->backend->poisoned.load(std::memory_order_acquire))
                    state_->settled.store(true, std::memory_order_release);
                finish_settled(nullptr, error);
            }
        }
        return pending;
    }

    void poison() noexcept {
        state_->settled.store(false, std::memory_order_release);
        state_->backend->poisoned.store(true, std::memory_order_release);
        std::lock_guard lock(quarantine_mutex());
        state_->quarantine_root->retained = state_;
    }

    std::shared_ptr<OwnedState> state_;
};

CudaBackend::CudaBackend(int ordinal, SessionBinder binder, ExecutorOptions options)
    : state_(std::make_shared<State>(ordinal, std::move(binder), options)) {
    if (!state_->binder) throw std::invalid_argument("CUDA backend requires a startup session binder");
}
CudaBackend::~CudaBackend() = default;

backend::DeviceFacts CudaBackend::device_facts() const {
    if (std::this_thread::get_id() != state_->owner_thread)
        throw std::logic_error("CUDA device facts requested from a non-owner thread");
    return state_->executor->device().facts;
}

bool CudaBackend::healthy() const noexcept {
    return !state_->poisoned.load(std::memory_order_acquire) && state_->executor->healthy();
}

std::optional<FenceFailureReport> CudaBackend::failure_report() const {
    return state_->executor->failure_report();
}

void CudaBackend::request_cancel() noexcept {
    std::stop_source active;
    {
        std::lock_guard lock(state_->cancel_mutex);
        if (!state_->step_active) return;
        state_->cancel_requested = true;
        active = state_->active_stop;
    }
    active.request_stop();
}

std::unique_ptr<runtime::IExecutionSession> CudaBackend::instantiate(
    const runtime::ExecutionPlan& plan, memory::BringupGrant& grant) {
    if (std::this_thread::get_id() != state_->owner_thread)
        throw std::logic_error("CUDA session construction called from a non-owner thread");
    if (!healthy()) throw std::runtime_error("CUDA backend is poisoned");
    if (!grant.matches(plan.candidate())) throw std::invalid_argument("CUDA grant does not match selected execution plan");
    if (plan.options().graphs != memory::GraphMode::disabled || plan.candidate().graphs() != memory::GraphMode::disabled)
        throw std::invalid_argument("CUDA session does not support graph capture");
    const auto actual = state_->executor->device().facts;
    if (!same_device(actual, plan.device_facts()) || !plan.device_profile().matches(actual))
        throw std::invalid_argument("CUDA executor device does not match selected plan device facts/profile");

    {
        std::lock_guard lock(state_->session_mutex);
        if (state_->constructing_session || !state_->active_session.expired())
            throw std::logic_error("CUDA backend permits only one live execution session");
        state_->constructing_session = true;
    }
    struct ConstructionGuard {
        std::shared_ptr<State> state;
        bool committed{};
        ~ConstructionGuard() {
            if (!committed) {
                std::lock_guard lock(state->session_mutex);
                state->constructing_session = false;
            }
        }
    } construction{state_};

    auto owned = std::make_shared<CudaSession::OwnedState>();
    owned->backend = state_;
    owned->inspection = plan.description().inspection;
    owned->max_context = plan.options().context_tokens;
    owned->candidate.emplace(plan.candidate());
    owned->programs = plan.programs();

    if (plan.programs().size() != plan.description().programs.size())
        throw std::invalid_argument("CUDA prepared program count does not match plan description");
    bool has_prefill = false, has_decode = false;
    std::vector<backend::PreparedKernel> all_selected;
    std::vector<backend::PreparedKernel> prefill_selected;
    std::vector<backend::PreparedKernel> decode_selected;
    for (std::size_t i = 0; i < plan.description().programs.size(); ++i) {
        const auto& declared = plan.description().programs[i];
        const auto& prepared = plan.programs()[i];
        has_prefill |= declared.phase == backend::Phase::prefill;
        has_decode |= declared.phase == backend::Phase::decode;
        if (declared.phase == backend::Phase::verify)
            throw std::invalid_argument("CUDA session does not implement verify");
        const std::vector<backend::KernelRequest>* calls = &declared.calls;
        if (prepared.selected_alternative_identity() != "primary") {
            const auto alt = std::find_if(declared.alternatives.begin(), declared.alternatives.end(),
                [&](const auto& item) { return item.identity == prepared.selected_alternative_identity(); });
            if (alt == declared.alternatives.end()) throw std::invalid_argument("selected CUDA program alternative is undeclared");
            calls = &alt->calls;
        }
        if (calls->empty() || prepared.kernels().size() != calls->size())
            throw std::invalid_argument("CUDA session requires an explicit nonempty bound call list");
        for (std::size_t j = 0; j < calls->size(); ++j) {
            const auto& binding = prepared.kernels()[j].bindings();
            if (binding != (*calls)[j] || binding.operation != backend::Operation::projection ||
                binding.phase != declared.phase)
                throw std::invalid_argument("CUDA session supports only the exact selected projection call list");
            all_selected.push_back(prepared.kernels()[j]);
            if (declared.phase == backend::Phase::prefill) prefill_selected.push_back(prepared.kernels()[j]);
            else if (declared.phase == backend::Phase::decode) decode_selected.push_back(prepared.kernels()[j]);
        }
    }
    owned->warmup_execution = std::unique_ptr<SessionExecution>(
        new SessionExecution(*state_->executor, std::move(all_selected)));
    if (has_prefill)
        owned->prefill_execution = std::unique_ptr<SessionExecution>(
            new SessionExecution(*state_->executor, std::move(prefill_selected)));
    if (has_decode)
        owned->decode_execution = std::unique_ptr<SessionExecution>(
            new SessionExecution(*state_->executor, std::move(decode_selected)));

    std::optional<std::string> overhead_id;
    for (const auto& line : plan.candidate().unique_allocations()) {
        const auto& record = line.allocation;
        if (!record.enabled) continue;
        if (record.source != memory::AllocationSource::project) {
            if (record.source == memory::AllocationSource::runtime_overhead) {
                if (record.location == memory::Location::device && record.overhead_bound) {
                    if (overhead_id) throw std::invalid_argument("CUDA backend requires exactly one device runtime overhead reservation");
                    overhead_id = record.id;
                } else if (record.location == memory::Location::host && record.overhead_bound == 0 &&
                           line.padded_bytes == 0) {
                    owned->zero_overhead_ids.push_back(record.id);
                } else {
                    throw std::invalid_argument("unsupported enabled CUDA runtime overhead reservation");
                }
            } else {
                throw std::invalid_argument("CUDA session does not support enabled non-project overhead sources");
            }
            continue;
        }

        const auto contract = grant.allocation_contract(record.id);
        const auto bytes = line.padded_bytes.value_or(contract.logical_bytes.value_or(0));
        if (!bytes) {
            auto lease = grant.allocate(record.id, 0, [](std::uint64_t cap) {
                if (cap != 0) throw std::invalid_argument("zero CUDA allocation received a nonzero cap");
                return memory::MemoryBacking{};
            });
            (void)lease;
            owned->resources.emplace(record.id, Buffer{});
            continue;
        }
        BufferDomain domain;
        switch (record.location) {
        case memory::Location::host: domain = BufferDomain::host; break;
        case memory::Location::pinned: domain = BufferDomain::pinned; break;
        case memory::Location::device: domain = BufferDomain::device; break;
        default: throw std::invalid_argument("unknown CUDA allocation location");
        }
        auto buffer = state_->executor->allocate(grant, record.id, bytes, domain, record.alignment);
        owned->resources.emplace(record.id, std::move(buffer));
        if (record.lifetime == memory::Lifetime::bringup) owned->bringup_ids.push_back(record.id);
    }
    if (!overhead_id) throw std::invalid_argument("CUDA backend requires one enabled device runtime overhead reservation");
    owned->overhead_id = *overhead_id;
    grant.observe_overhead(*overhead_id, state_->executor->observe_runtime_overhead());
    for (const auto& id : owned->zero_overhead_ids) grant.observe_overhead(id, 0);

    auto prepared = state_->binder(plan, owned->resources);
    if (!prepared.metadata) throw std::invalid_argument("CUDA binder must return owning prepared metadata");
    if (!prepared.warmup) throw std::invalid_argument("CUDA binder must provide warmup");
    if (has_prefill != static_cast<bool>(prepared.prefill) || has_decode != static_cast<bool>(prepared.decode))
        throw std::invalid_argument("CUDA binder callbacks must exactly match selected prefill/decode programs");
    owned->prepared = std::move(prepared);

    auto session = std::make_unique<CudaSession>(owned);
    {
        std::lock_guard lock(state_->session_mutex);
        state_->active_session = owned;
        state_->constructing_session = false;
    }
    construction.committed = true;
    return session;
}

} // namespace saga::backend::cuda
