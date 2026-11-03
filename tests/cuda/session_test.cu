#include "saga/backend/cuda_session.h"
#include "saga/runtime/startup.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
using namespace saga;
using namespace saga::backend;
using namespace saga::backend::cuda;
using namespace saga::memory;
using namespace saga::runtime;
using Json = nlohmann::json;

constexpr std::uint64_t outputs = 19, inputs = 23, tokens = 1;
constexpr std::uint64_t weight_stride = 48, activation_stride = 50, output_stride = 80;
constexpr std::uint64_t weight_extent = (outputs - 1) * weight_stride + inputs * 2;
constexpr std::uint64_t activation_extent = inputs * 2;
constexpr std::uint64_t output_extent = outputs * sizeof(float);

constexpr std::uint64_t runtime_cap = 1ULL << 30;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct LeaseProbe {
    int* destroyed{};
    explicit LeaseProbe(int& count) : destroyed(&count) {}
    LeaseProbe(const LeaseProbe&) = delete;
    LeaseProbe& operator=(const LeaseProbe&) = delete;
    LeaseProbe(LeaseProbe&& other) noexcept : destroyed(std::exchange(other.destroyed, nullptr)) {}
    ~LeaseProbe() noexcept { if (destroyed) ++*destroyed; }
};

struct FixtureData {
    std::vector<__nv_bfloat16> weights = std::vector<__nv_bfloat16>(
        static_cast<std::size_t>(weight_extent / 2));
    std::vector<__nv_bfloat16> activation = std::vector<__nv_bfloat16>(
        static_cast<std::size_t>(activation_extent / 2));
    std::vector<float> expected = std::vector<float>(static_cast<std::size_t>(outputs));

    FixtureData() {
        for (std::uint64_t row = 0; row < outputs; ++row) {
            for (std::uint64_t col = 0; col < inputs; ++col) {
                const int code = static_cast<int>((row * 7 + col * 3) % 17) - 8;
                weights[row * weight_stride / 2 + col] = __float2bfloat16(code * 0.0625F);
            }
        }
        for (std::uint64_t col = 0; col < inputs; ++col) {
            const int code = static_cast<int>((col * 5 + 2) % 13) - 6;
            activation[col] = __float2bfloat16(code * 0.125F);
        }
        for (std::uint64_t row = 0; row < outputs; ++row) {
            float sum = 0.0F;
            for (std::uint64_t col = 0; col < inputs; ++col)
                sum = std::fma(__bfloat162float(weights[row * weight_stride / 2 + col]),
                               __bfloat162float(activation[col]), sum);
            expected[row] = sum;
        }
    }
};

Profile make_profile() {
    Profile profile;
    profile.identity = "cuda-physical-session-fixture-v1";
    profile.model_max_context = 64;
    profile.main_cache = CacheSpec{kv::Geometry{1, 1, 1}, kv::Pair{kv::Codec::fp16, kv::Codec::fp16},
                                   kv::CacheLayout{1, 1, 0}, std::nullopt, std::nullopt, 1, 1};
    const std::vector<std::pair<Category, Location>> required{
        {Category::text_weights, Location::device}, {Category::fp32_recurrence, Location::device},
        {Category::convolution, Location::device}, {Category::continuation, Location::device},
        {Category::provisional, Location::device}, {Category::rollback_records, Location::device},
        {Category::activations, Location::device}, {Category::kernel_workspace, Location::host},
        {Category::logits, Location::device}, {Category::distributions, Location::device},
        {Category::acceptance, Location::device}, {Category::host_mappings, Location::host},
        {Category::staging, Location::host}, {Category::request_buffers, Location::host},
        {Category::pinned_transfer, Location::pinned}, {Category::pinned_control, Location::pinned},
    };
    std::size_t index = 0;
    for (const auto [category, location] : required) {
        Term term;
        term.id = "zero-" + std::to_string(index++);
        term.owner = "cuda-session-physical-test";
        term.category = category;
        term.location = location;
        term.lifetime = Lifetime::persistent;
        term.bytes = 0;
        profile.terms.push_back(std::move(term));
    }
    auto find = [&](Category category) -> Term& {
        return *std::find_if(profile.terms.begin(), profile.terms.end(),
            [&](const Term& term) { return term.category == category; });
    };
    auto& weights = find(Category::text_weights);
    weights.id = "weights";
    weights.bytes = weight_extent;
    weights.alignment = 256;
    auto& activations = find(Category::activations);
    activations.id = "activations";
    activations.bytes = activation_extent;
    activations.alignment = 256;
    activations.lifetime = Lifetime::persistent;
    auto& logits = find(Category::logits);
    logits.id = "output";
    logits.bytes = output_extent;
    logits.alignment = 256;
    logits.lifetime = Lifetime::persistent;
    auto& staging = find(Category::staging);
    staging.id = "staging-host";
    staging.bytes = activation_extent;
    staging.alignment = 256;
    staging.lifetime = Lifetime::bringup;
    auto& pinned = find(Category::pinned_transfer);
    pinned.id = "staging-pinned";
    pinned.bytes = weight_extent;
    pinned.alignment = 256;
    pinned.page_size = 4096;
    pinned.lifetime = Lifetime::bringup;
    auto& host_logits = find(Category::host_mappings);
    host_logits.id = "host-output";
    host_logits.bytes = output_extent;
    host_logits.alignment = 256;
    host_logits.lifetime = Lifetime::persistent;
    Term overhead;
    overhead.id = "runtime";
    overhead.owner = "cuda-runtime";
    overhead.category = Category::runtime_overhead;
    overhead.location = Location::device;
    overhead.lifetime = Lifetime::bringup;
    overhead.bytes = 0;
    overhead.overhead_bound = runtime_cap;
    profile.terms.push_back(std::move(overhead));
    Term scratch;
    scratch.id = "warmup-scratch";
    scratch.owner = "cuda-session-physical-test";
    scratch.category = Category::request_buffers;
    scratch.location = Location::device;
    scratch.lifetime = Lifetime::bringup;
    scratch.bytes = 256;
    profile.terms.push_back(std::move(scratch));
    return profile;
}

struct Model final : IModelDefinition {
    std::shared_ptr<FixtureData> data;
    bool fail_warmup_after_submit{};
    bool fail_before_submit{};
    bool fail_decode_after_submit{};
    bool cancel_wait{};
    bool launch_unselected{};
    std::mutex mutex;
    std::condition_variable condition;
    bool submitted{};
    explicit Model(std::shared_ptr<FixtureData> fixture) : data(std::move(fixture)) {}

    ModelInspection inspect(const ArtifactMetadata& artifact) const override {
        ModelSpec spec{"synthetic-bf16-projection-session", 64, {1, 1, 1},
            {{"fixture-weight", "weights", WeightEncoding::bf16, {outputs, inputs, 1},
              0, weight_extent, "fixture-bf16-weights"}}};
        return {artifact, std::move(spec), {"fixture-token-ids", FrontendPolicy::caller_token_ids,
                static_cast<std::uint32_t>(outputs), {}}, {}, {}};
    }
    std::vector<ModelProgram> describe_programs(const ModelInspection&, const RuntimeOptions&) const override {
        ProjectionRequest projection{{outputs, inputs, tokens},
            {weight_stride, activation_stride, output_stride}, Phase::decode, 2};
        return {{Phase::decode, {make_bf16_projection_request(projection)}, {}}};
    }
    Profile describe_resources(const ModelInspection&, const RuntimeOptions&) const override {
        return make_profile();
    }
    struct Runtime final : IModelRuntime {
        IExecutionSession& session;
        SequenceSlots slots{1};
        explicit Runtime(IExecutionSession& execution) : session(execution) {}
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
            CompletionSource source;
            auto result = source.completion();
            source.succeed();
            return result;
        }
    };
    std::unique_ptr<IModelRuntime> create_runtime(const ModelDescription&, IExecutionSession& session) const override {
        return std::make_unique<Runtime>(session);
    }
};

struct TestRun {
    std::shared_ptr<FixtureData> data = std::make_shared<FixtureData>();
    std::shared_ptr<Model> model = std::make_shared<Model>(data);
    KernelRegistry kernels;
    ModelRegistry models;
    BackendRegistry backends;
    std::shared_ptr<CudaBackend> backend;
    std::unique_ptr<StartupCoordinator> coordinator;
    std::unique_ptr<StartupRequest> request;
    std::vector<std::string> checks;

    explicit TestRun(int ordinal) {
        backend = std::make_shared<CudaBackend>(ordinal, [this](const ExecutionPlan& plan,
                                                                const SessionResources&) {
            if (model->fail_before_submit) throw std::runtime_error("injected binder failure before submission");
            const auto selected = plan.programs().front().kernels().front();
            const auto foreign = kernels.resolve(selected.bindings(), plan.device_facts().caps);
            PreparedSession session;
            session.metadata = std::make_shared<int>(1);
            session.warmup = [this, selected](SessionExecution& execution,
                    const SessionResources& resources, std::stop_token) {
                if (model->fail_before_submit) throw std::runtime_error("injected binder/warmup rejection");
                std::memcpy(resources.at("staging-pinned").address(), data->weights.data(),
                            static_cast<std::size_t>(weight_extent));
                std::memcpy(resources.at("staging-host").address(), data->activation.data(),
                            static_cast<std::size_t>(activation_extent));
                execution.copy_to_device(resources.at("weights"), 0, resources.at("staging-pinned"), 0,
                                         weight_extent);
                execution.copy_to_device(resources.at("activations"), 0, resources.at("staging-host"), 0,
                                         activation_extent);
                ProjectionRequest projection{{outputs, inputs, tokens},
                    {weight_stride, activation_stride, output_stride}, Phase::decode, 2};
                execution.launch(selected, projection, resources.at("weights"),
                                 resources.at("activations"), resources.at("output"));
                if (model->fail_warmup_after_submit) throw std::runtime_error("injected warmup failure after submit");
            };
            session.decode = [this, selected, foreign](SessionExecution& execution,
                    const SessionResources& resources, const DecodeRequest& request, std::stop_token token) {
                ProjectionRequest projection{{outputs, inputs, tokens},
                    {weight_stride, activation_stride, output_stride}, Phase::decode, 2};
                if (model->launch_unselected)
                    execution.launch(foreign, projection, resources.at("weights"),
                                     resources.at("activations"), resources.at("output"));
                execution.launch(selected, projection, resources.at("weights"),
                                 resources.at("activations"), resources.at("output"));
                {
                    std::lock_guard lock(model->mutex);
                    model->submitted = true;
                }
                model->condition.notify_all();
                while (token.stop_requested() == false && model->cancel_wait)
                    std::this_thread::yield();
                if (model->fail_decode_after_submit)
                    throw std::runtime_error("injected decode callback failure after physical submission");
                execution.copy_to_host(resources.at("host-output"), 0, resources.at("output"), 0,
                                       output_extent);
                std::vector<float> logits(static_cast<std::size_t>(outputs));
                std::memcpy(logits.data(), resources.at("host-output").address(),
                            static_cast<std::size_t>(output_extent));
                StepRowResult row;
                row.consumed_frontier = request.rows.front().positions.front() + 1;
                row.logits = std::move(logits);
                return OwningStepResult{{std::move(row)}};
            };
            return session;
        });
        register_bf16_projection_kernels(kernels);
        models.add("fixture", model);
        backends.add("cuda", backend);
        const auto facts = backend->device_facts();
        DeviceProfile device({"native-cuda-session-test", facts.caps.kind, facts.caps.architecture,
            facts.caps.architecture_version, facts.caps.device_identity, facts.product_identity,
            facts.warp_size, facts.multiprocessor_count, facts.device_memory_budget_bytes,
            facts.workspace_budget_bytes, {}});
        Budgets budgets{};
        for (auto& budget : budgets) budget = {1ULL << 32, 1ULL << 32, 0};
        request = std::make_unique<StartupRequest>(StartupRequest{
            "fixture", "cuda", {"fixture-artifact", {{"fixture-bf16-weights", weight_extent}}, {}, {}},
            {16, false, false, GraphMode::disabled}, budgets, GraphMode::disabled,
            std::move(device), std::nullopt, &kernels, {}});
        coordinator = std::make_unique<StartupCoordinator>(models, backends);
    }
    ApprovedRuntime start() { return coordinator->start(*request); }
};

DecodeRequest decode_request(SequenceHandle sequence) {
    StepRow row;
    row.token_ids = {1};
    row.positions = {0};
    row.context_tokens = 16;
    row.chunk_tokens = 1;
    return {sequence, {std::move(row)}};
}

std::string location_name(Location value) {
    switch (value) {
    case Location::host: return "host";
    case Location::pinned: return "pinned";
    case Location::device: return "device";
    }
    throw std::runtime_error("unknown allocation location");
}
std::string lifetime_name(Lifetime value) {
    switch (value) {
    case Lifetime::persistent: return "persistent";
    case Lifetime::bringup: return "bringup";
    case Lifetime::execution: return "execution";
    }
    throw std::runtime_error("unknown allocation lifetime");
}
std::string source_name(AllocationSource value) {
    switch (value) {
    case AllocationSource::project: return "project";
    case AllocationSource::runtime_overhead: return "runtime_overhead";
    case AllocationSource::graph_overhead: return "graph_overhead";
    }
    throw std::runtime_error("unknown allocation source");
}

Json allocation_report(const ApprovedRuntime& approved) {
    const auto& candidate = approved.plan().candidate();
    const auto report = approved.approval().report();
    Json rows = Json::array();
    for (const auto& measured : report.allocations) {
        const auto line = std::find_if(candidate.unique_allocations().begin(), candidate.unique_allocations().end(),
            [&](const auto& item) { return item.allocation.id == measured.id; });
        require(line != candidate.unique_allocations().end(), "measured allocation absent from candidate");
        rows.push_back({{"id", measured.id}, {"cap_bytes", measured.cap}, {"live_bytes", measured.live_bytes},
            {"measured_peak_bytes", measured.measured_peak}, {"measured", measured.measured},
            {"location", location_name(line->allocation.location)},
            {"lifetime", lifetime_name(line->allocation.lifetime)},
            {"source", source_name(line->allocation.source)}});
    }
    return {{"profile", candidate.profile_identity()},
        {"live_bytes", {{"host", report.live_bytes[0]}, {"pinned", report.live_bytes[1]},
                        {"device", report.live_bytes[2]}}}, {"allocations", std::move(rows)}};
}

Json run_success(int ordinal, int iterations) {
    TestRun test(ordinal);
    auto approved = test.start();
    test.checks.push_back("startup_selected_real_registered_bf16_decode_kernel");
    const auto sequence = approved.runtime().create_sequence(16);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        int destroyed = 0;
        auto pending = approved.runtime().decode(decode_request(sequence),
            ResourceLease(LeaseProbe(destroyed)));
        require(pending.status() == CompletionStatus::success, "decode must settle after a physical fence");
        auto result = pending.result();
        require(result.rows.size() == 1 && result.rows.front().logits.size() == outputs,
                "decode must return one row of exactly fixture-vocabulary logits");
        for (std::size_t i = 0; i < test.data->expected.size(); ++i) {
            const float actual = result.rows.front().logits[i];
            const float expected = test.data->expected[i];
            require(std::isfinite(actual) && std::isfinite(expected) && std::isfinite(std::abs(actual - expected)),
                    "non-finite session output/reference/difference");
            require(std::abs(actual - expected) <= 0.02F,
                    "session output differs from independent scalar reference");
        }
        require(destroyed == 0, "caller resource lease released while PendingStep consumer remains alive");
        pending = PendingStep{};
        require(destroyed == 1, "caller resource lease was not released after final PendingStep consumer");
    }
    test.checks.push_back("independent_scalar_reference_matches_all_logits");
    test.checks.push_back("decode_returns_owned_bounded_logits");
    test.checks.push_back("caller_lease_retained_until_pending_consumer_released");
    require(test.backend->healthy(), "successful session must leave CUDA backend healthy");
    const auto facts = test.backend->device_facts();
    int driver_version{}, runtime_version{};
    if (cudaDriverGetVersion(&driver_version) != cudaSuccess ||
        cudaRuntimeGetVersion(&runtime_version) != cudaSuccess)
        throw std::runtime_error("could not query CUDA driver/runtime versions");
    const auto ledger = allocation_report(approved);
    return {{"status", "physical_session_test_passed"}, {"qualification", "not_claimed"},
        {"serving_readiness", false}, {"readiness_approval", false}, {"ledger_reconciliation", "passed"},
        {"device", {{"name", facts.product_identity}, {"architecture", facts.caps.architecture},
            {"architecture_version", facts.caps.architecture_version}, {"identity", facts.caps.device_identity},
            {"ordinal", ordinal}, {"driver", driver_version}, {"runtime", runtime_version}}},
        {"iterations", iterations}, {"checks", test.checks},
        {"projection_scope", "synthetic_fixture_only"}, {"tuning_claim", false},
        {"device_loss_injection", "not_run"}, {"ledger", ledger}};
}

void run_failure_paths(int ordinal, std::vector<std::string>& checks) {
    {
        TestRun test(ordinal);
        test.model->fail_warmup_after_submit = true;
        bool threw = false;
        try { auto ignored = test.start(); (void)ignored; }
        catch (const std::exception&) { threw = true; }
        require(threw && test.backend->healthy(), "warmup exception after submission must drain and remain healthy");
        test.model->fail_warmup_after_submit = false;
        { auto recovered = test.start(); }
        require(test.backend->healthy(), "warmup failure cleanup must release session resources for reuse");
        checks.push_back("warmup_callback_failure_after_submit_drains_and_releases");
    }
    {
        TestRun test(ordinal);
        test.model->fail_decode_after_submit = true;
        auto approved = test.start();
        const auto sequence = approved.runtime().create_sequence(16);
        int destroyed = 0;
        auto pending = approved.runtime().decode(decode_request(sequence),
            ResourceLease(LeaseProbe(destroyed)));
        require(pending.status() == CompletionStatus::failure,
                "decode callback failure after submission must not publish a result");
        bool result_rejected = false;
        try { (void)pending.result(); } catch (const std::exception&) { result_rejected = true; }
        require(result_rejected && destroyed == 0,
                "failed callback must publish no borrowed result and retain consumer resources");
        pending = PendingStep{};
        require(destroyed == 1, "failed-step caller lease was not released after final consumer");
        require(test.backend->healthy(), "callback exception with successful physical drain must remain healthy");
        checks.push_back("decode_callback_failure_after_submit_drains_without_result");
    }
    {
        TestRun test(ordinal);
        auto approved = test.start();
        const auto sequence = approved.runtime().create_sequence(16);
        bool rejected = false;
        std::thread wrong([&] {
            try { (void)approved.runtime().decode(decode_request(sequence), {}); }
            catch (const std::logic_error&) { rejected = true; }
        });
        wrong.join();
        require(rejected && test.backend->healthy(), "wrong-thread decode must reject before physical work");
        checks.push_back("wrong_thread_rejected_before_work");
    }
    {
        TestRun test(ordinal);
        test.model->cancel_wait = true;
        auto approved = test.start();
        const auto sequence = approved.runtime().create_sequence(16);
        std::jthread cancel([&] {
            std::unique_lock lock(test.model->mutex);
            test.model->condition.wait(lock, [&] { return test.model->submitted; });
            lock.unlock();
            test.backend->request_cancel();
        });
        auto pending = approved.runtime().decode(decode_request(sequence), {});
        cancel.join();
        require(pending.status() == CompletionStatus::cancelled,
                "post-submit cancellation must settle only after callback and physical fence");
        require(test.backend->healthy(), "physically settled cancellation must retain backend health");
        checks.push_back("post_submit_cancellation_physically_fenced");
    }
    {
        TestRun test(ordinal);
        test.model->launch_unselected = true;
        auto approved = test.start();
        const auto sequence = approved.runtime().create_sequence(16);
        auto pending = approved.runtime().decode(decode_request(sequence), {});
        require(pending.status() == CompletionStatus::failure && test.backend->healthy(),
                "unselected preparation must be rejected without poisoning backend");
        checks.push_back("unselected_preparation_rejected_before_cuda_launch");
    }
    {
        TestRun test(ordinal);
        test.model->fail_before_submit = true;
        bool rejected = false;
        try { auto ignored = test.start(); (void)ignored; }
        catch (const std::exception&) { rejected = true; }
        require(rejected && test.backend->healthy(), "binder failure before submission must remain healthy");
        checks.push_back("binder_failure_before_submit");
    }
    {
        TestRun test(ordinal);
        for (int cycle = 0; cycle < 2; ++cycle) {
            {
                auto approved = test.start();
                test.checks.clear();
            }
            require(test.backend->healthy(), "healthy teardown must allow a later session startup");
        }
        checks.push_back("repeated_startup_after_healthy_teardown");
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 2) throw std::invalid_argument("usage: saga_cuda_session_test_device [iterations]");
        int iterations = 30;
        if (argc == 2) {
            char* end{};
            const auto parsed = std::strtol(argv[1], &end, 10);
            if (!end || *end || parsed < 1 || parsed > 10000)
                throw std::invalid_argument("iterations must be in 1..10000");
            iterations = static_cast<int>(parsed);
        }
        Json success = run_success(0, iterations);
        auto checks = success.at("checks").get<std::vector<std::string>>();
        run_failure_paths(0, checks);
        const std::vector<std::string> expected{
            "startup_selected_real_registered_bf16_decode_kernel",
            "independent_scalar_reference_matches_all_logits",
            "decode_returns_owned_bounded_logits",
            "caller_lease_retained_until_pending_consumer_released",
            "warmup_callback_failure_after_submit_drains_and_releases",
            "decode_callback_failure_after_submit_drains_without_result",
            "wrong_thread_rejected_before_work", "post_submit_cancellation_physically_fenced",
            "unselected_preparation_rejected_before_cuda_launch", "binder_failure_before_submit",
            "repeated_startup_after_healthy_teardown"};
        auto actual = checks;
        std::sort(actual.begin(), actual.end());
        auto sorted_expected = expected;
        std::sort(sorted_expected.begin(), sorted_expected.end());
        require(actual == sorted_expected, "physical session check set differs from the required executed checks");
        success["checks"] = checks;
        success["all_checks_passed"] = true;
        success["expected_checks"] = expected;
        success["device_loss_injection"] = "not_run";
        std::cout << success.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CUDA session physical test failed: " << error.what() << '\n';
        return 1;
    }
}
