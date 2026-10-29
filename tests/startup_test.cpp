#include "saga/runtime/startup.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using namespace saga;
int failures = 0;
void check(bool condition, const std::string& message) {
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
template <class Exception, class Function>
void throws(Function&& function, const std::string& message) {
    try { function(); check(false, message + " (did not throw)"); }
    catch (const Exception&) {}
    catch (const std::exception& error) { check(false, message + " (wrong exception: " + error.what() + ")"); }
}
template <class Function>
void throws_any(Function&& function, const std::string& message) {
    try { function(); check(false, message + " (did not throw)"); }
    catch (const std::exception&) {}
}

struct Counts {
    int inspections{}, instantiations{}, runtime_destroyed{}, session_destroyed{};
    int backing_destroyed{}, weight_backing_destroyed{}, drain_calls{}, bad_drain_order{}, runtime_saw_session{};
    int allocations{}, warmups{}, captures{}, borrowed_description_copies{};
    bool fail_after_allocation{};
    bool runtime_factory_throws{}, backend_throws{}, capture_fails{}, omit_measurements{};
    bool exceed_overhead{}, drop_persistent_owner{};
    int warmup_mode{};
    bool scaled_resource{}, substitute_artifact{}, invalid_state{}, resource_graph_mismatch{};
    bool facts_workspace_present{true};
    bool decomposition{}, missing_workspace_term{}, alias_allocations{};
    bool throw_after_submit{}, kernel_unavailable_high{};
    std::uint64_t workspace_requirement{};
    int workspace_profile_mode{};
    std::uint64_t selected_context{};
    std::string selected_alternative;
    std::vector<std::string> plan_identities;
    int alias_allocations_seen{};
    bool unknown_alias_rejected{}, shared_alias_cap_rejected{}, alias_factory_called{};
    std::mutex gate_mutex;
    std::condition_variable gate_cv;
    bool warmup_entered{};
    bool release_warmup{};
    bool submitted_work{}, completion_notified{}, settled_work{};
};
struct BackingDeleter {
    Counts* counts{};
    bool weight{};
    void operator()(std::vector<std::byte>* bytes) const noexcept {
        delete bytes;
        ++counts->backing_destroyed;
        if (weight) ++counts->weight_backing_destroyed;
        if (counts->drain_calls == 0) ++counts->bad_drain_order;
    }
};

backend::DeviceFacts cpu_facts() {
    backend::DeviceFacts facts;
    facts.caps = {backend::DeviceKind::cpu, "host", 0, 0, "cpu-1"};
    facts.product_identity = "test-cpu";
    facts.workspace_budget_bytes = 0;
    return facts;
}
backend::DeviceProfile cpu_profile(std::optional<std::uint64_t> minimum_workspace = std::nullopt,
                                   std::string identity = "synthetic-cpu-profile") {
    return backend::DeviceProfile({std::move(identity), backend::DeviceKind::cpu,
        "host", 0, "cpu-1", "test-cpu", std::nullopt, std::nullopt,
        std::nullopt, minimum_workspace, {}});
}
backend::KernelRequest request_call(std::uint64_t workspace_available = 0) {
    backend::KernelRequest r;
    r.operation = backend::Operation::projection;
    r.phase = backend::Phase::decode;
    r.shape = {1, 1, 1};
    r.workspace_available = workspace_available;
    return r;
}
class SyntheticKernel final : public backend::IKernelImplementation {
public:
    explicit SyntheticKernel(Counts& counts) {
        const auto r = request_call();
        descriptor_.id = "synthetic-cpu-projection";
        descriptor_.operation = r.operation;
        descriptor_.phase = r.phase;
        descriptor_.weights = r.weights;
        descriptor_.activation = r.activation;
        descriptor_.layout = r.layout;
        descriptor_.numerics = r.numerics;
        descriptor_.shape = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
        descriptor_.required_physical = r.physical;
        descriptor_.workspace = {counts.workspace_requirement, counts.workspace_requirement ? 1U : 1U};
        descriptor_.device = {backend::DeviceKind::cpu, "host", 0, 0, 0};
    }
    backend::KernelDescriptor descriptor() const override { return descriptor_; }
    bool supports(const backend::KernelRequest&, const backend::DeviceCaps&) const noexcept override { return true; }
    backend::PreparedKernel prepare(const backend::KernelRequest& r,
        backend::KernelPreparationContext&) const override {
        return {&launch, std::make_shared<const int>(1), r, descriptor_.workspace};
    }
private:
    static void launch(const void*, const backend::LaunchContext&) {}
    backend::KernelDescriptor descriptor_;
};

memory::Profile full_profile(const Counts& counts, bool graphs) {
    memory::Profile p;
    p.identity = "synthetic-resource-profile";
    p.model_max_context = 64;
    p.graphs = graphs ? memory::GraphMode::enabled : memory::GraphMode::disabled;
    p.main_cache = memory::CacheSpec{kv::Geometry{1, 1, 1},
        kv::Pair{kv::Codec::fp16, kv::Codec::fp16}, kv::CacheLayout{1, 1, 0},
        std::nullopt, std::nullopt, 1, 1};
    for (auto& b : p.budgets) b = {1U << 20, 1U << 20, 0};
    const std::vector<std::pair<memory::Category, memory::Location>> required{
        {memory::Category::text_weights, memory::Location::device},
        {memory::Category::fp32_recurrence, memory::Location::device},
        {memory::Category::convolution, memory::Location::device},
        {memory::Category::continuation, memory::Location::device},
        {memory::Category::provisional, memory::Location::device},
        {memory::Category::rollback_records, memory::Location::device},
        {memory::Category::activations, memory::Location::device},
        {memory::Category::kernel_workspace, memory::Location::host},
        {memory::Category::logits, memory::Location::device},
        {memory::Category::distributions, memory::Location::device},
        {memory::Category::acceptance, memory::Location::device},
        {memory::Category::host_mappings, memory::Location::host},
        {memory::Category::staging, memory::Location::host},
        {memory::Category::request_buffers, memory::Location::host},
        {memory::Category::pinned_transfer, memory::Location::pinned},
        {memory::Category::pinned_control, memory::Location::pinned}
    };
    std::size_t n = 0;
    for (const auto [category, location] : required) {
        memory::Term t;
        t.id = category == memory::Category::text_weights ? "weights" : "zero-" + std::to_string(n);
        t.owner = "synthetic";
        t.category = category;
        t.location = location;
        t.lifetime = memory::Lifetime::persistent;
        t.bytes = category == memory::Category::text_weights ? 2 : 0;
        if (category == memory::Category::activations && counts.scaled_resource) {
            t.bytes.reset();
            t.bytes_per_context = 1;
        }
        if (category == memory::Category::kernel_workspace) {
            if (counts.missing_workspace_term) continue;
            if (counts.workspace_profile_mode == 1) t.bytes = 4;
            else if (counts.workspace_profile_mode == 2) { t.bytes.reset(); t.bytes_per_context = 1; }
        }
        p.terms.push_back(std::move(t));
        ++n;
    }
    memory::Term overhead;
    overhead.id = "runtime-zero"; overhead.owner = "runtime";
    overhead.category = memory::Category::runtime_overhead;
    overhead.location = memory::Location::host;
    overhead.lifetime = memory::Lifetime::bringup;
    overhead.bytes = 0; overhead.overhead_bound = 0;
    p.terms.push_back(overhead);
    if (counts.resource_graph_mismatch) p.graphs = graphs ? memory::GraphMode::disabled : memory::GraphMode::enabled;
    if (graphs) {
        overhead.id = "graph-zero";
        overhead.category = memory::Category::graph_overhead;
        overhead.overhead_bound = 8;
        p.terms.push_back(overhead);
    }
    return p;
}

struct Session final : runtime::IExecutionSession {
    explicit Session(Counts& counts) : counts(counts) {}
    void initialize(const runtime::ExecutionPlan& plan, memory::BringupGrant& grant,
                    bool fail_after_allocation) {
        candidate_lines = plan.candidate().unique_allocations();
        if (!plan.description().programs.empty()) {
            copied_program = plan.description().programs.front();
            ++counts.borrowed_description_copies;
        }
        for (const auto& line : candidate_lines) {
            if (line.allocation.source != memory::AllocationSource::project) continue;
            const auto cap = line.bringup_cap.value_or(0);
            const auto allocation_id = counts.alias_allocations && line.allocation.id == "weights"
                ? std::string("weights-alias") : line.allocation.id;
            auto lease = grant.allocate(allocation_id, cap, [&](std::uint64_t exact_cap) {
                ++counts.allocations;
                const bool is_weight = line.allocation.id == "weights";
                std::unique_ptr<std::vector<std::byte>, BackingDeleter> owner(
                    new std::vector<std::byte>(static_cast<std::size_t>(exact_cap)), {&counts, is_weight});
                return memory::MemoryBacking{memory::OwnedBacking(std::move(owner)), exact_cap};
            });
            if (line.allocation.id == "weights") weight_lease = lease;
            leases.push_back(std::move(lease));
            if (fail_after_allocation) throw std::runtime_error("failure after exact allocation");
        }
        if (counts.alias_allocations) {
            bool unknown_rejected = false;
            try { (void)grant.allocate("not-a-candidate-id", 0, [](std::uint64_t) -> memory::MemoryBacking {
                throw std::runtime_error("unknown allocation must not invoke allocator");
            }); } catch (const std::invalid_argument&) { unknown_rejected = true; }
            bool cap_rejected = false;
            bool cap_factory_called = false;
            try { (void)grant.allocate("weights", 1, [&](std::uint64_t) -> memory::MemoryBacking {
                cap_factory_called = true;
                throw std::runtime_error("shared cap must reject before allocator");
            }); } catch (const std::runtime_error&) { cap_rejected = true; }
            counts.unknown_alias_rejected = unknown_rejected;
            counts.shared_alias_cap_rejected = cap_rejected;
            counts.alias_factory_called = cap_factory_called;
            ++counts.alias_allocations_seen;
        }
        if (counts.throw_after_submit) {
            launch_gated_work({}, 10);
            throw std::runtime_error("construction failed after submitting physical work");
        }
    }
    ~Session() override {
        if (!drained) ++counts.bad_drain_order;
        ++counts.session_destroyed;
        step_source.reset();
    }
    runtime::PendingStep prefill(runtime::PrefillRequest, runtime::ResourceLease resources) override {
        return pending(std::move(resources));
    }
    runtime::PendingStep decode(runtime::DecodeRequest, runtime::ResourceLease resources) override {
        return pending(std::move(resources));
    }
    std::optional<runtime::PendingStep> verify(runtime::VerifyRequest, runtime::ResourceLease) override {
        return std::nullopt;
    }
    runtime::Completion warmup(memory::BringupGrant&) override {
        ++counts.warmups;
        if (counts.warmup_mode == 4) throw std::runtime_error("synthetic warmup throw");
        if (counts.warmup_mode == 5) {
            runtime::CompletionSource abandoned;
            return abandoned.completion();
        }
        if (counts.warmup_mode == 6) {
            auto source = std::make_shared<runtime::CompletionSource>();
            auto completion = source->completion();
            launch_gated_work(source, 6);
            return completion;
        }
        if (counts.warmup_mode >= 7) {
            runtime::Completion completion;
            {
                runtime::CompletionSource source;
                completion = source.completion();
                launch_gated_work({}, counts.warmup_mode);
                if (counts.warmup_mode == 7)
                    source.fail(std::make_exception_ptr(std::runtime_error("synthetic early warmup failure")));
                else if (counts.warmup_mode == 8) source.cancel();
                else if (counts.warmup_mode == 10)
                    throw std::runtime_error("synthetic warmup throw with work pending");
            }
            if (counts.warmup_mode != 10) {
                std::lock_guard lock(counts.gate_mutex);
                counts.completion_notified = true;
                counts.gate_cv.notify_all();
            }
            return completion;
        }
        runtime::CompletionSource source;
        auto completion = source.completion();
        if (counts.warmup_mode == 1) source.fail(std::make_exception_ptr(std::runtime_error("synthetic warmup failure")));
        else if (counts.warmup_mode == 2) source.cancel();
        else source.succeed();
        return completion;
    }
    runtime::Completion capture(memory::BringupGrant&) override {
        ++counts.captures;
        runtime::CompletionSource source;
        if (counts.capture_fails) source.fail(std::make_exception_ptr(std::runtime_error("synthetic capture failure")));
        else source.succeed();
        return source.completion();
    }
    std::vector<std::pair<std::string, std::uint64_t>> overhead_measurements(
        memory::BringupGrant& grant) override {
        (void)grant;
        std::vector<std::pair<std::string, std::uint64_t>> all;
        if (counts.drop_persistent_owner) { leases.clear(); weight_lease = {}; }
        if (counts.omit_measurements) return all;
        for (const auto& line : candidate_lines) {
            if (line.allocation.source != memory::AllocationSource::project)
                all.emplace_back(line.allocation.id, counts.exceed_overhead ? 1 : 0);
        }
        return all;
    }
    void drain() noexcept override {
        if (worker.joinable()) worker.join();
        ++counts.drain_calls;
        drained = true;
    }
    void launch_gated_work(std::shared_ptr<runtime::CompletionSource> completion, int mode) {
        worker = std::thread([this, completion = std::move(completion), mode] {
            {
                std::unique_lock lock(counts.gate_mutex);
                counts.submitted_work = true;
                if (completion) counts.warmup_entered = true;
                counts.gate_cv.notify_all();
                counts.gate_cv.wait(lock, [&] { return counts.release_warmup; });
                counts.settled_work = true;
            }
            counts.gate_cv.notify_all();
            if (completion && mode == 6) completion->succeed();
        });
        std::unique_lock lock(counts.gate_mutex);
        if (!counts.gate_cv.wait_for(lock, std::chrono::seconds(5),
                [&] { return counts.submitted_work; }))
            throw std::runtime_error("synthetic worker did not reach submission gate");
    }
    runtime::PendingStep pending(runtime::ResourceLease resources) {
        step_source = std::make_unique<runtime::StepSource>(
            runtime::ResourceLease(LeaseBundle{std::move(resources), weight_lease}),
            runtime::StepResultBounds{1, std::nullopt});
        return step_source->pending();
    }
    void bind_candidate(const runtime::ExecutionPlan& plan) { candidate_lines = plan.candidate().unique_allocations(); }
    void runtime_destroyed_now() noexcept { runtime_destroyed = true; }
    struct LeaseBundle { runtime::ResourceLease request; memory::AllocationLease weight; };
    Counts& counts;
    bool runtime_destroyed{}, drained{};
    memory::AllocationLease weight_lease;
    std::vector<memory::AllocationLease> leases;
    std::vector<memory::LedgerLine> candidate_lines;
    backend::ModelProgram copied_program;

    std::unique_ptr<runtime::StepSource> step_source;
    std::thread worker;
};

struct Runtime final : runtime::IModelRuntime {
    Runtime(Session& session, Counts& counts) : session(session), counts(counts), slots(1) {}
    ~Runtime() override { ++counts.runtime_destroyed; session.runtime_destroyed_now(); }
    runtime::SequenceHandle create_sequence(std::uint64_t context) override {
        if (!context) throw std::invalid_argument("positive context required");
        return slots.acquire();
    }
    runtime::PendingStep prefill(runtime::PrefillRequest r, runtime::ResourceLease l) override {
        return session.prefill(std::move(r), std::move(l));
    }
    runtime::PendingStep decode(runtime::DecodeRequest r, runtime::ResourceLease l) override {
        return session.decode(std::move(r), std::move(l));
    }
    runtime::Completion release_sequence(runtime::SequenceHandle h) override {
        auto token = slots.begin_release(h);
        runtime::CompletionSource source; auto completion = source.completion(); source.succeed();
        slots.finish_release(std::move(token), completion); return completion;
    }
    Session& session; Counts& counts; runtime::SequenceSlots slots;
};

struct Definition final : runtime::IModelDefinition {
    explicit Definition(Counts& counts) : counts(counts) {}
    runtime::ModelInspection inspect(const runtime::ArtifactMetadata& artifact) const override {
        ++counts.inspections;
        backend::ModelSpec spec{"synthetic-model", 64, {1, 1, 1},
            {{"weight", "weights", backend::WeightEncoding::fp16, {1, 1, 1}, 0, 2, "file"}}};
        auto inspected_artifact = artifact;
        if (counts.substitute_artifact) inspected_artifact.identity += "-substituted";
        std::vector<runtime::StateSchema> state;
        if (counts.invalid_state) state.push_back({"bad-state", runtime::StateKind::recurrent,
            runtime::StateDType::f32, {1}, memory::Location::device,
            runtime::StateLifetime::sequence, "missing-state-allocation"});
        return {std::move(inspected_artifact), std::move(spec),
            {"front", runtime::FrontendPolicy::caller_token_ids, 4, {0}}, std::move(state), {}};
    }
    std::vector<backend::ModelProgram> describe_programs(
        const runtime::ModelInspection&, const runtime::RuntimeOptions& options) const override {
        auto primary = request_call(counts.workspace_requirement);
        if (counts.kernel_unavailable_high && options.context_tokens >= 64) primary.shape.outputs = 2;
        if (counts.decomposition) {
            primary.operation = backend::Operation::attention;
            auto alternative = request_call(counts.workspace_requirement);
            return {{backend::Phase::decode, {primary}, {{"synthetic-decomposition", {alternative}}}}};
        }
        return {{backend::Phase::decode, {primary}, {}}};
    }
    memory::Profile describe_resources(const runtime::ModelInspection&,
        const runtime::RuntimeOptions& options) const override {
        auto profile = full_profile(counts, options.graphs == memory::GraphMode::enabled && requested_graphs);
        if (counts.alias_allocations) {
            auto alias = profile.terms.front();
            alias.id = "weights-alias";
            alias.physical_identity = "weights";
            profile.terms.front().physical_identity = "weights";
            profile.terms.push_back(std::move(alias));
        }
        return profile;
    }
    std::unique_ptr<runtime::IModelRuntime> create_runtime(
        const runtime::ModelDescription&, runtime::IExecutionSession& session) const override {
        auto& fake = dynamic_cast<Session&>(session);
        if (counts.runtime_factory_throws) throw std::runtime_error("synthetic runtime factory failure");
        return std::make_unique<Runtime>(fake, counts);
    }
    Counts& counts;
    bool requested_graphs{};
};

struct Backend final : runtime::IGpuBackend {
    explicit Backend(Counts& counts) : counts(counts) {}
    backend::DeviceFacts device_facts() const override {
        auto facts = cpu_facts();
        if (!counts.facts_workspace_present) facts.workspace_budget_bytes.reset();
        return facts;
    }
    std::unique_ptr<runtime::IExecutionSession> instantiate(
        const runtime::ExecutionPlan& plan, memory::BringupGrant& grant) override {
        ++counts.instantiations;
        counts.selected_context = plan.options().context_tokens;
        counts.selected_alternative = plan.programs().empty() ? "" : plan.programs().front().selected_alternative_identity();
        counts.plan_identities.push_back(plan.identity());
        if (counts.backend_throws) throw std::runtime_error("synthetic backend construction failure");
        auto session = std::make_unique<Session>(counts);
        try {
            session->initialize(plan, grant, counts.fail_after_allocation);
            return session;
        } catch (...) {
            session->drain();
            throw;
        }
    }
    Counts& counts;
};

struct Fixture {
    explicit Fixture(bool graphs = false, std::uint64_t workspace = 0) : model(std::make_shared<Definition>(counts)),
        backend(std::make_shared<Backend>(counts)) {
        counts.workspace_requirement = workspace;
        model->requested_graphs = graphs;
        models.add("registered-model", model);
        backends.add("fake-backend", backend);
        coordinator = std::make_unique<runtime::StartupCoordinator>(models, backends);
        kernels.add(std::make_shared<SyntheticKernel>(counts));
        request = std::make_unique<runtime::StartupRequest>(runtime::StartupRequest{
            "registered-model", "fake-backend", {"artifact", {{"file", 16}}, {}, {}},
            {32, false, false, graphs ? memory::GraphMode::enabled : memory::GraphMode::disabled},
            {}, graphs ? memory::GraphMode::enabled : memory::GraphMode::disabled,
            cpu_profile(), std::nullopt, &kernels, {}});
        for (auto& budget : request->budgets) budget = {1U << 20, 1U << 20, 0};
    }
    Counts counts;
    runtime::ModelRegistry models;
    runtime::BackendRegistry backends;
    std::shared_ptr<Definition> model;
    std::shared_ptr<Backend> backend;
    backend::KernelRegistry kernels;
    std::unique_ptr<runtime::StartupCoordinator> coordinator;
    std::unique_ptr<runtime::StartupRequest> request;
};

void coordinator_rejects_invalid_request_before_allocation() {
    runtime::ModelRegistry models; runtime::BackendRegistry backends;
    struct NeverModel final : runtime::IModelDefinition {
        runtime::ModelInspection inspect(const runtime::ArtifactMetadata&) const override {
            ++called; throw std::runtime_error("unexpected inspect");
        }
        std::vector<backend::ModelProgram> describe_programs(const runtime::ModelInspection&,
            const runtime::RuntimeOptions&) const override { return {}; }
        memory::Profile describe_resources(const runtime::ModelInspection&,
            const runtime::RuntimeOptions&) const override { return {}; }
        std::unique_ptr<runtime::IModelRuntime> create_runtime(const runtime::ModelDescription&,
            runtime::IExecutionSession&) const override { return {}; }
        mutable int called{};
    };
    struct NeverBackend final : runtime::IGpuBackend {
        backend::DeviceFacts device_facts() const override { return cpu_facts(); }
        std::unique_ptr<runtime::IExecutionSession> instantiate(const runtime::ExecutionPlan&,
            memory::BringupGrant&) override { ++called; return {}; }
        int called{};
    };
    auto model = std::make_shared<NeverModel>(); auto backend = std::make_shared<NeverBackend>();
    models.add("model", model); backends.add("backend", backend);
    runtime::StartupCoordinator coordinator(models, backends);
    backend::KernelRegistry kernels;
    runtime::StartupRequest mismatch{"model", "backend", {"artifact", {}, {}, {}}, {4, false, false}, {},
        memory::GraphMode::disabled,
        backend::DeviceProfile({"wrong", backend::DeviceKind::cpu, "host", 0, "other",
            "test-cpu", {}, {}, {}, {}, {}}), std::nullopt, &kernels, {}};
    throws<std::invalid_argument>([&] { (void)coordinator.start(mismatch); }, "device profile mismatch rejected");
    check(model->called == 0 && backend->called == 0, "mismatch rejected before inspection and allocation");
    mismatch.kernels = nullptr;
    throws<std::invalid_argument>([&] { (void)coordinator.start(mismatch); }, "missing kernel registry rejected");
}

void successful_bringup(bool graphs) {
    Fixture f(graphs);
    auto owner = f.coordinator->start(*f.request);
    check(owner.plan().description().inspection.spec.model_identity == "synthetic-model",
          "plan preserves inspected model identity independently of registry key");
    check(owner.plan().identity() == owner.approval().profile_identity(),
          "approval is bound to the exact immutable execution plan candidate");
    check(owner.approval().graphs() == f.request->graphs, "approval records the requested graph mode");
    check(f.counts.allocations > 0 && f.counts.warmups == 1, "all planned project allocations and warmup ran");
    check(f.counts.borrowed_description_copies == 1,
          "backend snapshots borrowed program metadata into its owning session");
    check(f.counts.captures == (graphs ? 1 : 0), "capture runs only when graph mode is enabled");
    const auto report = owner.approval().report();
    check(std::all_of(report.allocations.begin(), report.allocations.end(),
        [](const memory::AllocationMeasurement& a) { return a.measured; }),
        "every bounded overhead and project allocation has an observation");
}

void construction_failure_unwinds_and_drains() {
    Fixture f;
    f.counts.fail_after_allocation = true;
    throws<std::runtime_error>([&] { (void)f.coordinator->start(*f.request); },
                               "session initialization failure after exact allocation propagates");
    check(f.counts.drain_calls == 1, "partial session is physically drained before teardown");
    check(f.counts.runtime_destroyed == 0 && f.counts.session_destroyed == 1,
          "failed session initialization destroys only its partial session");
    check(f.counts.backing_destroyed > 0, "failed startup releases already-acquired backing");
    check(f.counts.bad_drain_order == 0, "failed startup preserves drain-before-backing/session destruction");
}

void startup_failure_matrix() {
    for (const int mode : {1, 2, 4, 5}) {
        Fixture f;
        f.counts.warmup_mode = mode;
        throws_any([&] { (void)f.coordinator->start(*f.request); },
            "warmup failure/cancel/throw/abandon fails startup mode " + std::to_string(mode));
        check(f.counts.drain_calls == 1 && f.counts.session_destroyed == 1,
              "warmup failure drains before session destruction");
        check(f.counts.runtime_destroyed == 1 && f.counts.bad_drain_order == 0,
              "warmup failure destroys runtime after physical drain");
        check(f.counts.weight_backing_destroyed == 1,
              "warmup failure releases persistent backing after drain");
    }
    {
        Fixture f;
        f.request->graphs = memory::GraphMode::enabled;
        f.request->options.graphs = memory::GraphMode::enabled;
        f.model->requested_graphs = true;
        f.counts.capture_fails = true;
        throws<std::runtime_error>([&] { (void)f.coordinator->start(*f.request); },
                                   "capture failure prevents startup approval");
        check(f.counts.captures == 1 && f.counts.drain_calls == 1,
              "capture failure drains the submitted preparation");
        check(f.counts.runtime_destroyed == 1 && f.counts.bad_drain_order == 0,
              "capture failure drains before runtime/session/backing teardown");
    }
    for (int kind = 0; kind < 2; ++kind) {
        Fixture f;
        if (kind == 0) f.counts.runtime_factory_throws = true;
        else f.counts.fail_after_allocation = true;
        throws<std::runtime_error>([&] { (void)f.coordinator->start(*f.request); },
            kind == 0 ? "runtime factory failure propagates" : "backend allocation initialization failure propagates");
        check(f.counts.drain_calls == 1 && f.counts.session_destroyed == 1,
              "construction failure drains before session/backing destruction");
        check(f.counts.bad_drain_order == 0,
              "construction failure never releases backing before physical drain");
    }
    {
        Fixture f;
        f.counts.backend_throws = true;
        throws<std::runtime_error>([&] { (void)f.coordinator->start(*f.request); },
                                   "backend construction exception propagates");
        check(f.counts.instantiations == 1 && f.counts.allocations == 0,
              "backend construction throw occurs before allocation");
    }
    for (int kind = 0; kind < 3; ++kind) {
        Fixture f;
        f.counts.omit_measurements = kind == 0;
        f.counts.exceed_overhead = kind == 1;
        f.counts.drop_persistent_owner = kind == 2;
        throws_any([&] { (void)f.coordinator->start(*f.request); },
            "readiness reconciliation rejects incomplete/invalid observation case " + std::to_string(kind));
        check(f.counts.drain_calls >= 1 && f.counts.bad_drain_order == 0,
              "readiness rejection drains before resources are released (drains=" +
              std::to_string(f.counts.drain_calls) + ", order=" + std::to_string(f.counts.bad_drain_order) + ")");
    }
}

void construction_throw_drains_submitted_work() {
    Fixture f;
    f.counts.throw_after_submit = true;
    std::atomic<bool> finished{false};
    std::exception_ptr error;
    std::thread startup([&] {
        try { (void)f.coordinator->start(*f.request); }
        catch (...) { error = std::current_exception(); }
        finished.store(true);
    });
    bool submitted = false;
    {
        std::unique_lock lock(f.counts.gate_mutex);
        submitted = f.counts.gate_cv.wait_for(lock, std::chrono::seconds(5),
            [&] { return f.counts.submitted_work; });
    }
    check(submitted, "constructor work reached its externally gated worker");
    check(f.counts.allocations > 0 && !finished.load(),
          "throwing backend constructor has allocations and outstanding physical work");
    {
        std::lock_guard lock(f.counts.gate_mutex);
        f.counts.release_warmup = true;
    }
    f.counts.gate_cv.notify_all();
    startup.join();
    check(error && f.counts.settled_work && f.counts.drain_calls >= 1,
          "backend catches construction throw and drains worker to its physical fence");
    check(f.counts.session_destroyed == 1 && f.counts.weight_backing_destroyed == 1 &&
          f.counts.bad_drain_order == 0,
          "partial session and backing release only after submitted work settles");
}

void pending_warmup_waits_for_external_fence() {
    Fixture f;
    f.counts.warmup_mode = 6;
    std::atomic<bool> finished{false};
    std::exception_ptr error;
    std::thread startup([&] {
        try { auto owner = f.coordinator->start(*f.request); }
        catch (...) { error = std::current_exception(); }
        finished.store(true);
    });
    bool entered = false;
    {
        std::unique_lock lock(f.counts.gate_mutex);
        entered = f.counts.gate_cv.wait_for(lock, std::chrono::seconds(5),
            [&] { return f.counts.warmup_entered; });
    }
    check(entered, "warmup worker reached its externally gated completion");
    check(!finished.load(), "coordinator waits on the pending completion returned by warmup");
    {
        std::lock_guard lock(f.counts.gate_mutex);
        f.counts.release_warmup = true;
    }
    f.counts.gate_cv.notify_all();
    startup.join();
    check(!error && finished.load() && f.counts.settled_work,
          "startup proceeds only after worker signals the external warmup fence");
    check(f.counts.drain_calls == 2 && f.counts.bad_drain_order == 0,
          "gated successful startup drains before owner teardown (drains=" +
          std::to_string(f.counts.drain_calls) + ", order=" + std::to_string(f.counts.bad_drain_order) + ")");
}

void failed_gated_warmup_drains_before_unwind() {
    for (const int mode : {7, 8, 9, 10}) {
        Fixture f;
        f.counts.warmup_mode = mode;
        std::atomic<bool> finished{false};
        std::exception_ptr error;
        std::thread startup([&] {
            try { (void)f.coordinator->start(*f.request); }
            catch (...) { error = std::current_exception(); }
            finished.store(true);
        });
        bool submitted = false;
        bool notified = mode == 10;
        {
            std::unique_lock lock(f.counts.gate_mutex);
            submitted = f.counts.gate_cv.wait_for(lock, std::chrono::seconds(5),
                [&] { return f.counts.submitted_work; });
            if (submitted && mode != 10)
                notified = f.counts.gate_cv.wait_for(lock, std::chrono::seconds(5),
                    [&] { return f.counts.completion_notified; });
        }
        check(submitted, "failed warmup worker reaches gate for mode " + std::to_string(mode));
        check(notified, "warmup terminal notification occurs for mode " + std::to_string(mode));
        if (submitted) {
            bool unsettled;
            { std::lock_guard lock(f.counts.gate_mutex); unsettled = !f.counts.settled_work; }
            check(!finished.load() && unsettled && f.counts.allocations > 0 &&
                  f.counts.runtime_destroyed == 0 && f.counts.session_destroyed == 0 &&
                  f.counts.weight_backing_destroyed == 0,
                  "terminal notification precedes physical settlement while startup owners stay live");
        }
        {
            std::lock_guard lock(f.counts.gate_mutex);
            f.counts.release_warmup = true;
        }
        f.counts.gate_cv.notify_all();
        startup.join();
        check(error != nullptr && finished.load(),
              "failed/cancelled/abandoned/thrown warmup prevents approval after fence release");
        check(f.counts.settled_work && f.counts.drain_calls >= 1 &&
              f.counts.runtime_destroyed == 1 && f.counts.session_destroyed == 1 &&
              f.counts.weight_backing_destroyed == 1 && f.counts.bad_drain_order == 0,
              "failed warmup drains worker before runtime/session/backing unwind");
    }
}

void moved_runtime_teardown_and_moved_from_access() {
    Fixture a, b;
    auto first = a.coordinator->start(*a.request);
    auto second = b.coordinator->start(*b.request);
    first = std::move(second);
    check(a.counts.drain_calls == 2 && a.counts.runtime_destroyed == 1 &&
          a.counts.session_destroyed == 1 && a.counts.bad_drain_order == 0,
          "move assignment drains and destroys the replaced owner in order (drains=" +
          std::to_string(a.counts.drain_calls) + ", runtime=" +
          std::to_string(a.counts.runtime_destroyed) + ", session=" +
          std::to_string(a.counts.session_destroyed) + ", order=" +
          std::to_string(a.counts.bad_drain_order) + ")");
    throws<std::logic_error>([&] { (void)second.runtime(); },
                             "moved-from runtime accessor rejects empty owner");
    throws<std::logic_error>([&] { (void)second.plan(); },
                             "moved-from plan accessor rejects empty owner");
    throws<std::logic_error>([&] { (void)second.approval(); },
                             "moved-from approval accessor rejects empty owner");
    check(b.counts.drain_calls == 1 && b.counts.bad_drain_order == 0,
          "moved-to owner remains live and drains when finally destroyed");
}

struct CallerResource {
    explicit CallerResource(int& count) : destroyed(&count) {}
    CallerResource(const CallerResource&) = delete;
    CallerResource& operator=(const CallerResource&) = delete;
    CallerResource(CallerResource&& other) noexcept : destroyed(std::exchange(other.destroyed, nullptr)) {}
    ~CallerResource() noexcept { if (destroyed) ++*destroyed; }
    int* destroyed;
};
void selection_kernel_and_workspace_matrix() {
    {
        Fixture f;
        f.counts.scaled_resource = true;
        auto profile = full_profile(f.counts, false);
        const auto low = memory::evaluate(profile, 32).known_bytes[static_cast<std::size_t>(memory::Location::device)];
        const auto high = memory::evaluate(profile, 64).known_bytes[static_cast<std::size_t>(memory::Location::device)];
        check(high > low, "context-scaled fixture grows with context");
        f.request->budgets[static_cast<std::size_t>(memory::Location::device)] = {low, low, 0};
        f.request->options.context_tokens = 64;
        throws<std::runtime_error>([&] { (void)f.coordinator->start(*f.request); },
                                   "explicit infeasible context rejects rather than shrinking");
        check(f.counts.instantiations == 0, "explicit infeasible context never constructs backend session");
    }
    {
        Fixture f;
        f.counts.scaled_resource = true;
        auto profile = full_profile(f.counts, false);
        const auto low = memory::evaluate(profile, 32).known_bytes[static_cast<std::size_t>(memory::Location::device)];
        f.request->budgets[static_cast<std::size_t>(memory::Location::device)] = {low, low, 0};
        f.request->options.context_tokens = 0;
        f.request->auto_contexts = {32, 64};
        auto owner = f.coordinator->start(*f.request);
        check(f.counts.selected_context == 32, "auto startup selects highest statically feasible context");
        const auto& attempts = owner.plan().context_attempt_diagnostics();
        check(attempts.size() == 2 && attempts[0].find("context=64 statically rejected") != std::string::npos &&
              attempts[1].find("context=32 selected") != std::string::npos,
              "auto plan preserves rejection and selection diagnostics in descending order");
    }
    {
        Fixture f;
        f.counts.kernel_unavailable_high = true;
        f.counts.scaled_resource = true;
        auto profile = full_profile(f.counts, false);
        const auto lower_bytes = memory::evaluate(profile, 32).known_bytes[
            static_cast<std::size_t>(memory::Location::device)];
        f.request->budgets[static_cast<std::size_t>(memory::Location::device)] =
            {lower_bytes, lower_bytes, 0};
        f.request->options.context_tokens = 0;
        f.request->auto_contexts = {64, 32};
        try {
            auto owner = f.coordinator->start(*f.request);
            check(f.counts.selected_context == 32,
                  "auto startup skips high context that is statically infeasible and kernel-unavailable");
            const auto& attempts = owner.plan().context_attempt_diagnostics();
            check(attempts.size() == 2 && attempts[0].find("context=64 statically rejected") != std::string::npos &&
                  attempts[1].find("context=32 selected") != std::string::npos,
                  "static rejection diagnostic is retained before the lower-context selection");
        } catch (const std::exception& error) {
            check(false, std::string("auto startup failed to skip statically infeasible high context: ") + error.what());
        }
    }
    {
        Fixture f;
        f.counts.kernel_unavailable_high = true;
        f.request->options.context_tokens = 0;
        f.request->auto_contexts = {64, 32};
        throws<backend::KernelUnavailable>([&] { (void)f.coordinator->start(*f.request); },
            "statically eligible high-context kernel unavailability propagates without retry");
        check(f.counts.instantiations == 0,
              "kernel preparation failure occurs before backend construction and is not retried");
    }
    {
        Fixture f;
        f.request->options.context_tokens = 0;
        f.request->auto_contexts = {64, 32};
        f.counts.backend_throws = true;
        throws<std::runtime_error>([&] { (void)f.coordinator->start(*f.request); },
                                   "auto backend failure propagates without lower-context retry");
        check(f.counts.instantiations == 1 && f.counts.selected_context == 64,
              "auto startup does not retry backend construction errors at smaller context");
    }
    {
        Fixture f;
        f.counts.decomposition = true;
        auto owner = f.coordinator->start(*f.request);
        check(f.counts.selected_alternative == "synthetic-decomposition",
              "missing primary kernel resolves through the explicit decomposition alternative");
    }
    {
        Fixture f(false, 8);
        f.counts.missing_workspace_term = true;
        throws_any([&] { (void)f.coordinator->start(*f.request); },
                                      "missing selected kernel workspace metadata is rejected");
        check(f.counts.instantiations == 0, "missing workspace rejection precedes allocation");
    }
    {
        Fixture f(false, 8);
        f.counts.workspace_profile_mode = 1;
        throws<std::invalid_argument>([&] { (void)f.coordinator->start(*f.request); },
                                      "undersized selected kernel workspace is rejected");
        check(f.counts.instantiations == 0, "undersized workspace rejection precedes allocation");
    }
    {
        Fixture f(false, 8);
        f.counts.workspace_profile_mode = 2;
        f.request->options.context_tokens = 8;
        auto owner = f.coordinator->start(*f.request);
        check(f.counts.instantiations == 1, "context-scaled workspace metadata covers selected workspace");
    }
}

void preallocation_mismatch_matrix() {
    for (int kind = 0; kind < 4; ++kind) {
        Fixture f;
        if (kind == 0) f.counts.substitute_artifact = true;
        else if (kind == 1) f.counts.invalid_state = true;
        else if (kind == 2) f.counts.resource_graph_mismatch = true;
        else {
            f.request->pair_profile.emplace(backend::PairProfileSpec{
                "wrong-inspected-model", "synthetic-cpu-profile", "cpu-1", {}});
        }
        if (kind == 2) {
            f.request->graphs = memory::GraphMode::enabled;
            f.request->options.graphs = memory::GraphMode::enabled;
            f.model->requested_graphs = true;
        }
        throws_any([&] { (void)f.coordinator->start(*f.request); },
                   "artifact/state/graph/profile mismatch rejected case " + std::to_string(kind));
        check(f.counts.instantiations == 0 && f.counts.allocations == 0,
              "metadata/profile mismatch is rejected before backend allocation");
    }
    {
        Fixture f;
        f.request->pair_profile.emplace(backend::PairProfileSpec{
            "synthetic-model", "synthetic-cpu-profile", "cpu-1", {}});
        auto owner = f.coordinator->start(*f.request);
        check(owner.plan().description().inspection.spec.model_identity == "synthetic-model" &&
              f.request->model_id == "registered-model",
              "pair profile and execution plan use inspected identity, not registry key");
    }
}

void semantic_plan_identity_matrix() {
    Fixture base;
    auto original = base.coordinator->start(*base.request);
    const auto identity = original.plan().identity();
    original = runtime::ApprovedRuntime(std::move(original));
    {
        Fixture changed;
        changed.request->artifact.identity = "artifact|registry.model_id=registered-model";
        auto owner = changed.coordinator->start(*changed.request);
        check(owner.plan().identity() != identity, "delimiter-containing artifact identity changes plan identity");
    }
    {
        Fixture absent;
        auto owner_a = absent.coordinator->start(*absent.request);
        const auto absent_identity = owner_a.plan().identity();
        Fixture explicit_zero;
        explicit_zero.request = std::make_unique<runtime::StartupRequest>(runtime::StartupRequest{
            "registered-model", "fake-backend", {"artifact", {{"file", 16}}, {}, {}},
            {32, false, false, memory::GraphMode::disabled}, {}, memory::GraphMode::disabled,
            cpu_profile(), std::nullopt, &explicit_zero.kernels, {}});
        for (auto& budget : explicit_zero.request->budgets) budget = {1U << 20, 1U << 20, 0};
        auto owner_b = explicit_zero.coordinator->start(*explicit_zero.request);
        explicit_zero.counts.facts_workspace_present = false;
        auto owner_c = explicit_zero.coordinator->start(*explicit_zero.request);
        check(owner_b.plan().identity() != owner_c.plan().identity() &&
              owner_c.plan().identity() != absent_identity,
              "unknown device workspace fact and explicit zero remain semantically distinct");
    }
    {
        Fixture alias;
        alias.counts.alias_allocations = true;
        throws_any([&] { (void)alias.coordinator->start(*alias.request); },
                   "intentional second alias allocation exhausts the shared candidate cap");
        check(alias.counts.alias_allocations_seen == 1,
              "candidate alias ID is accepted for its initial allocation after plan reidentity");
        check(alias.counts.unknown_alias_rejected,
              "grant rejects unknown allocation ID even after candidate reidentity");
        check(alias.counts.shared_alias_cap_rejected && !alias.counts.alias_factory_called,
              "known alias shares original cap and rejects over-cap allocation before factory");
    }
}

void pending_consumer_retains_request_and_allocation() {
    Fixture f;
    std::optional<runtime::ApprovedRuntime> owner;
    owner.emplace(f.coordinator->start(*f.request));
    auto sequence = owner->runtime().create_sequence(32);
    int caller_destroyed = 0;
    runtime::DecodeRequest request;
    request.sequence = sequence;
    request.rows.push_back({{1}, {0}, 1, 1});
    auto pending = owner->runtime().decode(std::move(request),
        runtime::ResourceLease(CallerResource(caller_destroyed)));
    check(pending.valid(), "typed decode returns a valid pending consumer");
    auto pending_copy = pending;
    check(pending_copy.valid(), "pending step consumer can be independently copied");
    auto& fake_runtime = dynamic_cast<Runtime&>(owner->runtime());
    runtime::OwningStepResult completed;
    completed.rows.push_back({1, {}, {}});
    check(fake_runtime.session.step_source->succeed(std::move(completed)),
          "typed decode source settles physical work before teardown");
    check(pending.status() == runtime::CompletionStatus::success &&
          pending_copy.status() == runtime::CompletionStatus::success,
          "copied pending consumers observe completed typed request");
    check(f.counts.backing_destroyed == 0 && caller_destroyed == 0,
          "live owner and pending consumer retain caller and allocation backing");
    owner.reset();
    check(f.counts.session_destroyed == 1 && f.counts.runtime_destroyed == 1,
          "runtime owner teardown drains then destroys runtime and session");
    pending = runtime::PendingStep{};
    check(f.counts.weight_backing_destroyed == 0,
          "last copied pending consumer retains allocation backing after session teardown");
    check(caller_destroyed == 0,
          "last copied pending consumer retains caller resources after session teardown");
    f.coordinator.reset();
    f.models = {};
    f.backends = {};
    f.backend.reset(); f.model.reset();
    check(pending_copy.valid(), "pending consumer survives coordinator, registries, and producer handles");
    pending_copy = runtime::PendingStep{};
    check(caller_destroyed == 1, "caller resource releases with final copied pending consumer");
    check(f.counts.weight_backing_destroyed == 1,
          "persistent allocation backing releases after final pending consumer");
}
}

int main() {
    static_assert(!std::is_copy_constructible_v<saga::runtime::ApprovedRuntime>);
    static_assert(std::is_move_constructible_v<saga::runtime::ApprovedRuntime>);
    coordinator_rejects_invalid_request_before_allocation();
    successful_bringup(false);
    successful_bringup(true);
    construction_failure_unwinds_and_drains();
    startup_failure_matrix();
    pending_warmup_waits_for_external_fence();
    construction_throw_drains_submitted_work();
    failed_gated_warmup_drains_before_unwind();
    moved_runtime_teardown_and_moved_from_access();
    selection_kernel_and_workspace_matrix();
    preallocation_mismatch_matrix();
    semantic_plan_identity_matrix();
    pending_consumer_retains_request_and_allocation();
    if (failures != 0) std::cerr << failures << " startup behavioral check(s) failed\n";
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
