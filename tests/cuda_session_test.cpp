#include "saga/backend/cuda_session.h"
#include "saga/runtime/startup.h"
#include "control.h"

#include <algorithm>
#include <atomic>

#include <condition_variable>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <stop_token>
#include <thread>
#include <type_traits>

namespace {
using namespace saga;
using namespace saga::backend;
using namespace saga::backend::cuda;
using namespace saga::runtime;
using namespace saga::memory;

static_assert(!std::is_copy_constructible_v<SessionExecution>);
static_assert(!std::is_copy_assignable_v<SessionExecution>);
static_assert(!std::is_move_constructible_v<SessionExecution>);
static_assert(!std::is_move_assignable_v<SessionExecution>);

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template<class F> void must_throw(F&& function, const char* message) {
    try { function(); }
    catch (const std::exception&) { return; }
    throw std::runtime_error(message);
}

struct Behavior {
    bool throw_warmup{};
    bool omit_decode{};
    bool fail_decode{};
    bool throw_decode_after_submit{};
    bool warmup_submit_then_throw{};
    bool cancel_wait{};
    bool recursive_cancel_callback{};
    bool launch_unselected{};
    bool cancel_before_return{};
    bool retain_capability{};
    IExecutionSession* observed_session{};
    bool unsettled_in_warmup{};
    bool unsettled_in_decode{};
    SessionExecution* retained_execution{};
    const PreparedKernel* retained_kernel{};
    bool verify_phase{};
    int binder_calls{};
    std::atomic<bool> entered{false};
    std::mutex mutex;
    std::condition_variable condition;
};

struct SyntheticKernel final : IKernelImplementation {
    KernelDescriptor desc;
    SyntheticKernel(const DeviceFacts& facts, const KernelRequest& r) {
        desc.id = "session-fake-projection";
        desc.operation = r.operation;
        desc.phase = r.phase;
        desc.weights = r.weights;
        desc.activation = r.activation;
        desc.layout = r.layout;
        desc.numerics = r.numerics;
        desc.shape = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
        desc.required_physical = r.physical;
        desc.device = {facts.caps.kind, facts.caps.architecture,
                       facts.caps.architecture_version, facts.caps.architecture_version, 0};
    }
    KernelDescriptor descriptor() const override { return desc; }
    bool supports(const KernelRequest&, const DeviceCaps&) const noexcept override { return true; }
    PreparedKernel prepare(const KernelRequest& request, KernelPreparationContext&) const override {
        return {&launch_noop, std::make_shared<int>(1), request, {}};
    }
    static void launch_noop(const void*, const LaunchContext&) {}
};

Profile profile() {
    Profile p;
    p.identity = "fake-cuda-session-profile";
    p.model_max_context = 64;
    p.main_cache = CacheSpec{kv::Geometry{1, 1, 1}, kv::Pair{kv::Codec::fp16, kv::Codec::fp16},
                             kv::CacheLayout{1, 1, 0}, std::nullopt, std::nullopt, 1, 1};
    const std::vector<std::pair<Category, Location>> terms{
        {Category::text_weights, Location::device}, {Category::fp32_recurrence, Location::device},
        {Category::convolution, Location::device}, {Category::continuation, Location::device},
        {Category::provisional, Location::device}, {Category::rollback_records, Location::device},
        {Category::activations, Location::device}, {Category::kernel_workspace, Location::host},
        {Category::logits, Location::device}, {Category::distributions, Location::device},
        {Category::acceptance, Location::device}, {Category::host_mappings, Location::host},
        {Category::staging, Location::host}, {Category::request_buffers, Location::host},
        {Category::pinned_transfer, Location::pinned}, {Category::pinned_control, Location::pinned}
    };
    std::size_t n = 0;
    for (const auto [category, location] : terms) {
        Term t;
        t.id = category == Category::text_weights ? "weights" : "zero-" + std::to_string(n);
        t.owner = "cuda-session-test";
        t.category = category;
        t.location = location;
        t.lifetime = Lifetime::persistent;
        t.bytes = category == Category::text_weights ? 256 : 0;
        if (category == Category::activations) { t.id = "activations"; t.bytes = 256; t.lifetime = Lifetime::execution; }
        if (category == Category::logits) { t.id = "logits"; t.bytes = 256; t.lifetime = Lifetime::execution; }
        p.terms.push_back(std::move(t));
        ++n;
    }
    Term overhead;
    overhead.id = "cuda-runtime";
    overhead.owner = "cuda-runtime";
    overhead.category = Category::runtime_overhead;
    overhead.location = Location::device;
    overhead.lifetime = Lifetime::bringup;
    overhead.bytes = 0;
    overhead.overhead_bound = 1U << 20;
    p.terms.push_back(std::move(overhead));
    Term warmup;
    warmup.id = "warmup-scratch";
    warmup.owner = "cuda-session-test";
    warmup.category = Category::request_buffers;
    warmup.location = Location::device;
    warmup.lifetime = Lifetime::bringup;
    warmup.bytes = 256;
    p.terms.push_back(std::move(warmup));
    return p;
}

struct Definition final : IModelDefinition {
    Behavior& behavior;
    bool graphs{};
    explicit Definition(Behavior& b, bool graph_mode = false) : behavior(b), graphs(graph_mode) {}
    ModelInspection inspect(const ArtifactMetadata& artifact) const override {
        ModelSpec spec{"synthetic-cuda-session", 64, {1, 1, 1},
            {{"weight", "weights", WeightEncoding::none, {1, 1, 1}, 0, 2, "weights-file"}}};
        return {artifact, std::move(spec), {"fake-frontend", FrontendPolicy::caller_token_ids, 4, {0}}, {}, {}};
    }
    std::vector<ModelProgram> describe_programs(const ModelInspection&, const RuntimeOptions&) const override {
        KernelRequest request;
        const auto phase = behavior.verify_phase ? Phase::verify : Phase::decode;
        request.phase = phase;
        return {{phase, {request}, {}}};
    }
    Profile describe_resources(const ModelInspection&, const RuntimeOptions& options) const override {
        (void)options;
        auto result = profile();
        result.graphs = graphs ? GraphMode::enabled : GraphMode::disabled;
        return result;
    }
    struct ModelRuntime final : IModelRuntime {
        IExecutionSession& session;
        SequenceSlots slots{1};
        explicit ModelRuntime(IExecutionSession& s) : session(s) {}
        SequenceHandle create_sequence(std::uint64_t context) override {
            if (!context) throw std::invalid_argument("context must be positive");
            return slots.acquire();
        }
        PendingStep prefill(PrefillRequest request, ResourceLease lease) override {
            return session.prefill(std::move(request), std::move(lease));
        }
        PendingStep decode(DecodeRequest request, ResourceLease lease) override {
            return session.decode(std::move(request), std::move(lease));
        }
        Completion release_sequence(SequenceHandle) override {
            CompletionSource source; auto completion = source.completion(); source.succeed(); return completion;
        }
    };
    std::unique_ptr<IModelRuntime> create_runtime(const ModelDescription&, IExecutionSession& session) const override {
        behavior.observed_session = &session;
        return std::make_unique<ModelRuntime>(session);
    }
};

struct Fixture {
    Behavior behavior;
    std::shared_ptr<CudaBackend> backend;
    ModelRegistry models;
    BackendRegistry backends;
    KernelRegistry kernels;
    std::unique_ptr<StartupCoordinator> coordinator;
    std::optional<StartupRequest> request;

    explicit Fixture(bool graphs = false) {
        backend = std::make_shared<CudaBackend>(0, [this](const ExecutionPlan& plan,
                                                          const SessionResources&) {
            ++behavior.binder_calls;
            PreparedSession prepared;
            prepared.metadata = std::make_shared<int>(42);
            const auto kernel = plan.programs().front().kernels().front();
            auto unselected = PreparedKernel(&SyntheticKernel::launch_noop, std::make_shared<int>(2),
                                             kernel.bindings(), kernel.workspace());
            prepared.warmup = [this, kernel](SessionExecution& execution, const SessionResources& resources, std::stop_token) {
                behavior.unsettled_in_warmup = !behavior.observed_session->physically_settled();
                if (behavior.warmup_submit_then_throw) {
                    ProjectionRequest projection;
                    projection.shape = {1, 1, 1};
                    projection.strides = {1, 1, 1};
                    projection.phase = Phase::decode;
                    execution.launch(kernel, projection, resources.at("weights"),
                                     resources.at("activations"), resources.at("logits"));
                    fake::fail_query = true;
                }
                if (behavior.throw_warmup || behavior.warmup_submit_then_throw)
                    throw std::runtime_error("injected warmup failure");
            };
            if (!behavior.omit_decode) {
                prepared.decode = [this, kernel, unselected](SessionExecution& execution,
                    const SessionResources& resources, const DecodeRequest& request, std::stop_token token) {
                    behavior.unsettled_in_decode = !behavior.observed_session->physically_settled();
                    if (behavior.retain_capability) {
                        behavior.retained_execution = &execution;
                        behavior.retained_kernel = &kernel;
                    }
                    std::unique_ptr<std::stop_callback<std::function<void()>>> recursive_callback;
                    if (behavior.recursive_cancel_callback) {
                        recursive_callback = std::make_unique<std::stop_callback<std::function<void()>>>(
                            token, [this] { backend->request_cancel(); });
                    }
                    if (behavior.cancel_wait) {
                        behavior.entered.store(true, std::memory_order_release);
                        behavior.condition.notify_all();
                        while (!token.stop_requested()) std::this_thread::yield();
                    }
                    ProjectionRequest projection;
                    projection.shape = {1, 1, 1};
                    projection.strides = {1, 1, 1};
                    projection.phase = Phase::decode;
                    if (behavior.launch_unselected)
                        execution.launch(unselected, projection, resources.at("weights"),
                                         resources.at("activations"), resources.at("logits"));
                    execution.launch(kernel, projection, resources.at("weights"),
                                     resources.at("activations"), resources.at("logits"));
                    if (behavior.cancel_before_return) backend->request_cancel();
                    if (behavior.fail_decode || behavior.throw_decode_after_submit) fake::fail_query = true;
                    if (behavior.throw_decode_after_submit)
                        throw std::runtime_error("injected callback failure after submission");
                    StepRowResult row;
                    row.consumed_frontier = request.rows.front().positions.front() + 1;
                    row.logits.assign(4, 0.0F);
                    return OwningStepResult{{std::move(row)}};
                };
            }
            return prepared;
        });
        const auto facts = backend->device_facts();
        models.add("synthetic", std::make_shared<Definition>(behavior, graphs));
        backends.add("cuda", backend);
        KernelRequest kernel_request;
        kernel_request.phase = behavior.verify_phase ? Phase::verify : Phase::decode;
        kernels.add(std::make_shared<SyntheticKernel>(facts, kernel_request));
        DeviceProfile device({"fake-cuda", facts.caps.kind, facts.caps.architecture,
            facts.caps.architecture_version, facts.caps.device_identity, facts.product_identity,
            facts.warp_size, facts.multiprocessor_count, std::nullopt, std::nullopt, {}});
        Budgets budgets{};
        for (auto& budget : budgets) budget = {4U << 20, 4U << 20, 0};
        request.emplace(StartupRequest{"synthetic", "cuda", {"artifact", {{"weights-file", 16}}, {}, {}},
            {16, false, false, graphs ? GraphMode::enabled : GraphMode::disabled}, budgets,
            graphs ? GraphMode::enabled : GraphMode::disabled, std::move(device), std::nullopt,
            &kernels, {}});
        coordinator = std::make_unique<StartupCoordinator>(models, backends);
    }

    ApprovedRuntime start() { return coordinator->start(*request); }
};

struct LeaseProbe {
    int* destroyed{};
    explicit LeaseProbe(int& count) : destroyed(&count) {}
    LeaseProbe(const LeaseProbe&) = delete;
    LeaseProbe& operator=(const LeaseProbe&) = delete;
    LeaseProbe(LeaseProbe&& other) noexcept : destroyed(std::exchange(other.destroyed, nullptr)) {}
    ~LeaseProbe() noexcept { if (destroyed) ++*destroyed; }
};

DecodeRequest request_for(SequenceHandle sequence) {
    StepRow row;
    row.token_ids = {1};
    row.positions = {0};
    row.context_tokens = 16;
    row.chunk_tokens = 1;
    return {sequence, {std::move(row)}};
}

void success_and_consumer_lease() {
    Fixture fixture;
    auto approved = fixture.start();
    const auto sequence = approved.runtime().create_sequence(16);
    int destroyed = 0;
    auto pending = approved.runtime().decode(request_for(sequence), ResourceLease(LeaseProbe(destroyed)));
    require(fixture.behavior.unsettled_in_warmup, "warmup callback must not claim physical settlement");
    require(fixture.behavior.unsettled_in_decode, "step callback must not claim physical settlement");
    require(fixture.behavior.observed_session->physically_settled(), "successful drain must prove physical settlement");
    require(pending.status() == CompletionStatus::success, "decode must settle after its physical fence");
    require(pending.result().rows.size() == 1, "decode must publish one bounded result row");
    require(destroyed == 0, "caller lease must remain retained by a live pending-step consumer");
    pending = PendingStep{};
    require(destroyed == 1, "caller lease must release after the final pending-step consumer");
    require(fixture.backend->healthy(), "successful physical work must leave backend healthy");
    const auto report = approved.approval().report();
    const auto scratch = std::find_if(report.allocations.begin(), report.allocations.end(),
        [](const auto& item) { return item.id == "warmup-scratch"; });
    require(scratch != report.allocations.end() && scratch->live_bytes == 0,
            "bringup-only CUDA resources must release after successful warmup settlement");
    auto duplicate_grant = approved.plan().candidate().grant(fixture.request->budgets);
    must_throw([&] { (void)fixture.backend->instantiate(approved.plan(), duplicate_grant); },
               "backend must reject a second live session");
}

void cancellation_and_owner_thread() {
    Fixture fixture;
    fixture.behavior.cancel_wait = true;
    fixture.behavior.recursive_cancel_callback = true;
    auto approved = fixture.start();
    const auto sequence = approved.runtime().create_sequence(16);
    std::jthread cancel([&] {
        std::unique_lock lock(fixture.behavior.mutex);
        fixture.behavior.condition.wait(lock, [&] { return fixture.behavior.entered.load(std::memory_order_acquire); });
        lock.unlock();
        for (int i = 0; i < 8; ++i) fixture.backend->request_cancel();
    });
    auto pending = approved.runtime().decode(request_for(sequence), {});
    cancel.join();
    require(pending.status() == CompletionStatus::cancelled, "cancellation settles only after callback and physical drain");

    bool wrong_thread_rejected = false;
    std::thread wrong([&] {
        try { (void)approved.runtime().decode(request_for(sequence), {}); }
        catch (const std::logic_error&) { wrong_thread_rejected = true; }
    });
    wrong.join();
    require(wrong_thread_rejected, "session operation must reject a non-owner thread before callback execution");
    bool facts_thread_rejected = false;
    std::thread wrong_facts([&] {
        try { (void)fixture.backend->device_facts(); }
        catch (const std::logic_error&) { facts_thread_rejected = true; }
    });
    wrong_facts.join();
    require(facts_thread_rejected, "device facts access must reject a non-owner thread");
    require(fixture.backend->healthy(), "wrong-thread rejection before device work must not poison a healthy executor");
}

void callback_capability_expires_after_callback() {
    Fixture fixture;
    fixture.behavior.retain_capability = true;
    auto approved = fixture.start();
    const auto sequence = approved.runtime().create_sequence(16);
    auto pending = approved.runtime().decode(request_for(sequence), {});
    require(pending.status() == CompletionStatus::success, "retained-capability decode must succeed");
    require(fixture.behavior.retained_execution && fixture.behavior.retained_kernel,
            "callback must expose the capability only by reference");
    must_throw([&] { fixture.behavior.retained_execution->synchronize(); },
               "retained callback capability must reject synchronize after callback return");
    ProjectionRequest projection;
    projection.shape = {1, 1, 1};
    projection.strides = {1, 1, 1};
    Buffer empty;
    must_throw([&] { fixture.behavior.retained_execution->launch(*fixture.behavior.retained_kernel,
        projection, empty, empty, empty); },
        "retained callback capability must reject launch after callback return");
    require(fixture.backend->healthy(), "inactive capability rejection must not affect backend health");
}

void cancellation_recorded_before_result_publish_wins() {
    Fixture fixture;
    fixture.behavior.cancel_before_return = true;
    auto approved = fixture.start();
    const auto sequence = approved.runtime().create_sequence(16);
    auto pending = approved.runtime().decode(request_for(sequence), {});
    require(pending.status() == CompletionStatus::cancelled,
            "cancellation recorded during callback must win before terminal result publication");
    require(fixture.backend->healthy(), "settled cancellation must leave backend healthy");
}

void failed_fence_quarantines_caller_and_rejects_reuse() {
    Fixture fixture;
    fixture.behavior.throw_decode_after_submit = true;
    auto approved = fixture.start();
    const auto sequence = approved.runtime().create_sequence(16);
    int destroyed = 0;
    auto pending = approved.runtime().decode(request_for(sequence), ResourceLease(LeaseProbe(destroyed)));
    require(pending.status() == CompletionStatus::failure, "failed fence must fail the pending step");
    require(!fixture.backend->healthy() && fixture.backend->failure_report().has_value(),
            "failed fence must poison backend and expose executor failure report");
    pending = PendingStep{};
    require(destroyed == 0, "failed-fence quarantine must retain caller/request owners after consumer release");
    auto valid_grant = approved.plan().candidate().grant(fixture.request->budgets);
    must_throw([&] { (void)fixture.backend->instantiate(approved.plan(), valid_grant); },
               "poisoned backend must reject new session construction");
    fake::fail_query = false;
}

void startup_rejections_and_exact_grant() {
    {
        Fixture fixture(true);
        must_throw([&] { auto ignored = fixture.start(); (void)ignored; },
                   "graph-enabled startup must be rejected before binding");
        require(fixture.behavior.binder_calls == 0,
                "graph-enabled plan must be rejected before allocation binder invocation");
    }
    {
        Fixture fixture;
        fixture.behavior.omit_decode = true;
        must_throw([&] { auto ignored = fixture.start(); (void)ignored; },
                   "selected decode program without callback must be rejected");
        require(fixture.backend->healthy(), "missing callback rejection must not poison backend");
    }
    {
        Fixture fixture;
        fixture.behavior.verify_phase = true;
        must_throw([&] { auto ignored = fixture.start(); (void)ignored; },
                   "verify phase must be rejected before allocation/binding");
        require(fixture.behavior.binder_calls == 0,
                "unsupported phase must be rejected before startup binder invocation");
    }
    {
        Fixture fixture;
        fixture.behavior.launch_unselected = true;
        auto approved = fixture.start();
        const auto sequence = approved.runtime().create_sequence(16);
        const auto launches_before = fake::launch_checks;
        auto pending = approved.runtime().decode(request_for(sequence), {});
        require(pending.status() == CompletionStatus::failure,
                "unselected prepared kernel must fail the step");
        require(fake::launch_checks == launches_before,
                "unselected kernel must be rejected before entering the CUDA executor");
        require(fixture.backend->healthy(), "rejected unselected call with successful drain remains physically healthy");
    }
    {
        Fixture fixture;
        auto approved = fixture.start();
        CudaBackend foreign_device(1, [](const ExecutionPlan&, const SessionResources&) {
            return PreparedSession{};
        });
        auto grant = approved.plan().candidate().grant(fixture.request->budgets);
        must_throw([&] { (void)foreign_device.instantiate(approved.plan(), grant); },
                   "backend must reject a plan bound to a different physical device identity");
    }
    {
        Fixture fixture;
        auto approved = fixture.start();
        std::vector<AllocationRecord> records;
        auto lines = approved.plan().candidate().unique_allocations();
        for (const auto& line : lines) records.push_back(line.allocation);
        records.front().owner += "-mismatch";
        StaticCandidate altered(approved.plan().candidate().profile_identity(),
            approved.plan().candidate().graphs(), std::move(records));
        auto grant = altered.grant(fixture.request->budgets);
        const auto binder_calls = fixture.behavior.binder_calls;
        must_throw([&] { (void)fixture.backend->instantiate(approved.plan(), grant); },
                   "same-profile grant with altered allocation metadata must be rejected");
        require(fixture.behavior.binder_calls == binder_calls,
                "exact grant mismatch must be rejected before resource binding");
    }
}

void warmup_exception_drains_before_propagating() {
    {
        Fixture fixture;
        fixture.behavior.throw_warmup = true;
        const auto before = fake::event_queries;
        must_throw([&] { auto ignored = fixture.start(); (void)ignored; },
                   "warmup callback exception must propagate to startup");
        require(fake::event_queries > before, "warmup exception path must attempt a bounded drain");
        require(fixture.backend->healthy(), "successful cleanup fence after warmup exception must preserve backend health");
    }
    {
        Fixture fixture;
        fixture.behavior.warmup_submit_then_throw = true;
        must_throw([&] { auto ignored = fixture.start(); (void)ignored; },
                   "warmup exception after submission must propagate");
        require(!fixture.backend->healthy(), "unfenced warmup owners must poison the backend");
        fake::fail_query = false;
    }
}

} // namespace

int main() {
    try {
        success_and_consumer_lease();
        cancellation_and_owner_thread();
        callback_capability_expires_after_callback();
        cancellation_recorded_before_result_publish_wins();
        failed_fence_quarantines_caller_and_rejects_reuse();
        startup_rejections_and_exact_grant();
        warmup_exception_drains_before_propagating();
        std::cout << "cuda session lifecycle tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cuda session lifecycle test failure: " << error.what() << '\n';
        fake::fail_query = false;
        return 1;
    }
}
