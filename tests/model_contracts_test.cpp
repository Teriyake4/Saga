#include "saga/runtime/model_contracts.h"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace saga;
void check(bool ok, const char* message) { if (!ok) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); } }
template<class E, class F> void throws_exact(F&& f, const std::string& message) {
    try { f(); } catch (const E& error) {
            if (error.what() != message) { std::cerr << "expected: " << message << " got: " << error.what() << std::endl; check(false, "unexpected exception diagnostic"); }
            return;
        }
    check(false, "expected exact exception");
}
struct Sentinel { explicit Sentinel(int& count) : count(count) {} ~Sentinel() noexcept { ++count; } int& count; };

runtime::ArtifactMetadata metadata() { return {"artifact", {{"file-A", 64}, {"file-B", 16}}}; }
backend::LogicalWeightBinding binding(std::string logical, std::string physical, std::string component,
                                      std::uint64_t start, std::uint64_t size) {
    return {std::move(logical), std::move(physical), backend::WeightEncoding::fp16,
            {2, 3, 1}, start, size, std::move(component)};
}
backend::ModelSpec spec() {
    return {"synthetic", 128, {1, 2, 4}, {binding("w", "tensor-w", "file-A", 8, 16)}};
}
backend::ModelProgram program() {
    backend::KernelRequest request;
    request.operation = backend::Operation::projection;
    request.phase = backend::Phase::decode;
    request.shape = {2, 3, 1};
    return {backend::Phase::decode, {request}};
}
runtime::RuntimeOptions options() { return {32, false, false}; }
void valid(const runtime::ArtifactMetadata& a, const backend::ModelSpec& s, const backend::ModelProgram& p,
           const runtime::RuntimeOptions& o, runtime::ModelCapabilities c = {}) {
    runtime::validate_model_contract(a, s, p, o, c);
}
void metadata_tests() {
    const auto a = metadata(); auto s = spec(); const auto p = program(); auto o = options();
    valid(a, s, p, o);
    auto aliases = s;
    aliases.weights.push_back(binding("w-alias", "tensor-w", "file-A", 8, 16));
    valid(a, aliases, p, o);
    auto disjoint = s;
    disjoint.weights.push_back(binding("w2", "tensor-w2", "file-A", 24, 8));
    disjoint.weights.push_back(binding("w3", "tensor-w3", "file-B", 0, 16));
    valid(a, disjoint, p, o);
    auto near_max = runtime::ArtifactMetadata{"edge", {{"edge-file", UINT64_MAX}}};
    auto edge = backend::ModelSpec{"edge-model", UINT64_MAX, {1,1,1},
        {binding("edge-a", "edge-a-id", "edge-file", UINT64_MAX - 8, 4),
         binding("edge-b", "edge-b-id", "edge-file", UINT64_MAX - 4, 4)}};
    valid(near_max, edge, p, runtime::RuntimeOptions{UINT64_MAX, false, false});
    auto feature_options = o; feature_options.request_mtp = true; feature_options.request_vision = true;
    valid(a, s, p, feature_options, {true, true});

    const std::vector<std::pair<std::string, std::function<void()>>> invalid = {
        {"empty or duplicate artifact component identity", [&] { auto x = a; x.components.push_back({"file-A", 1}); valid(x,s,p,o); }},
        {"empty or duplicate artifact component identity", [&] { auto x = a; x.components[0].identity.clear(); valid(x,s,p,o); }},
        {"empty or duplicate logical weight name", [&] { auto x = s; x.weights.push_back(binding("w", "tensor-z", "file-A", 32, 1)); valid(a,x,p,o); }},
        {"conflicting aliases for physical tensor", [&] { auto x = aliases; x.weights.back().byte_offset = 9; valid(a,x,p,o); }},
        {"conflicting aliases for physical tensor", [&] { auto x = aliases; x.weights.back().component_identity = "file-B"; x.weights.back().byte_offset = 0; valid(a,x,p,o); }},
        {"conflicting aliases for physical tensor", [&] { auto x = aliases; x.weights.back().byte_extent = 15; valid(a,x,p,o); }},
        {"conflicting aliases for physical tensor", [&] { auto x = aliases; x.weights.back().encoding = backend::WeightEncoding::bf16; valid(a,x,p,o); }},
        {"conflicting aliases for physical tensor", [&] { auto x = aliases; x.weights.back().shape.tokens = 2; valid(a,x,p,o); }},
        {"overlapping ranges have different physical identities", [&] { auto x = s; x.weights.push_back(binding("w2", "tensor-z", "file-A", 16, 8)); valid(a,x,p,o); }},
        {"overlapping ranges have different physical identities", [&] { auto x = s; x.weights.push_back(binding("w2", "tensor-z", "file-A", 4, 8)); valid(a,x,p,o); }},
        {"empty physical tensor identity", [&] { auto x = s; x.weights[0].physical_identity.clear(); valid(a,x,p,o); }},
        {"artifact and model identities are required", [&] { auto x = a; x.identity.clear(); valid(x,s,p,o); }},
        {"artifact and model identities are required", [&] { auto x = s; x.model_identity.clear(); valid(a,x,p,o); }},
        {"empty or duplicate logical weight name", [&] { auto x = s; x.weights[0].logical_name.clear(); valid(a,x,p,o); }},
        {"weight references unknown artifact component", [&] { auto x = s; x.weights[0].component_identity.clear(); valid(a,x,p,o); }},
        {"invalid requested context", [&] { auto x = s; x.maximum_context_tokens=0; valid(a,x,p,o); }},
        {"model dimensions must be positive", [&] { auto x = s; x.main_attention.layers=0; valid(a,x,p,o); }},
        {"model dimensions must be positive", [&] { auto x = s; x.main_attention.heads=0; valid(a,x,p,o); }},
        {"model dimensions must be positive", [&] { auto x = s; x.main_attention.head_channels=0; valid(a,x,p,o); }},
        {"weight references unknown artifact component", [&] { auto x = s; x.weights[0].component_identity = "missing"; valid(a,x,p,o); }},
        {"weight extent must be positive", [&] { auto x = s; x.weights[0].byte_extent = 0; valid(a,x,p,o); }},
        {"weight extent outside artifact component", [&] { auto x = s; x.weights[0].byte_offset = UINT64_MAX; x.weights[0].byte_extent = 2; valid(a,x,p,o); }},
        {"weight extent outside artifact component", [&] { auto x = s; x.weights[0].byte_offset = UINT64_MAX - 1; x.weights[0].byte_extent = UINT64_MAX; valid(a,x,p,o); }},
        {"unknown weight encoding", [&] { auto x = s; x.weights[0].encoding = static_cast<backend::WeightEncoding>(255); valid(a,x,p,o); }},
        {"weight dimensions must be positive", [&] { auto x = s; x.weights[0].shape.inputs = 0; valid(a,x,p,o); }},
        {"requested MTP capability unavailable", [&] { auto x=o; x.request_mtp=true; valid(a,s,p,x); }},
        {"requested vision capability unavailable", [&] { auto x=o; x.request_vision=true; valid(a,s,p,x); }},
        {"invalid requested context", [&] { auto x=o; x.context_tokens=0; valid(a,s,p,x); }},
        {"invalid requested context", [&] { auto x=o; x.context_tokens=129; valid(a,s,p,x); }},
        {"unknown model program phase", [&] { auto x=p; x.phase=static_cast<backend::Phase>(255); valid(a,s,x,o); }},
        {"kernel phase mismatches model program", [&] { auto x=p; x.calls[0].phase=backend::Phase::prefill; valid(a,s,x,o); }},
        {"model contract requires weight bindings", [&] { auto x=s; x.weights.clear(); valid(a,x,p,o); }},
        {"model contract requires program calls", [&] { auto x=p; x.calls.clear(); valid(a,s,x,o); }}
    };
    for (const auto& [message, run] : invalid) throws_exact<std::invalid_argument>(run, message);

    const std::vector<std::pair<std::string, std::function<void(backend::KernelRequest&)>>> structural = {
        {"unknown KV codec", [](auto& x) { x.kv_pair = {static_cast<kv::Codec>(255), kv::Codec::fp16}; }},
        {"unknown KV codec", [](auto& x) { x.kv_pair = {kv::Codec::fp16, static_cast<kv::Codec>(255)}; }},
        {"base alignment must be a power of two", [](auto& x) { x.physical.alignment = 3; }},
        {"byte strides must be positive", [](auto& x) { x.physical.weight_row_stride = 0; }},
        {"byte strides must be positive", [](auto& x) { x.physical.activation_row_stride = 0; }},
        {"byte strides must be positive", [](auto& x) { x.physical.output_row_stride = 0; }},
        {"gsq3_group128_v1 requires quantization group 128", [](auto& x) { x.weights = backend::WeightEncoding::gsq3_group128_v1; }},
        {"workspace alignment must be a power of two", [](auto& x) { x.workspace_alignment = 3; }},
        {"unknown operation", [](auto& x) { x.operation = static_cast<backend::Operation>(255); }},

        {"unknown weight encoding", [](auto& x) { x.weights = static_cast<backend::WeightEncoding>(255); }},
        {"unknown activation dtype", [](auto& x) { x.activation = static_cast<backend::ActivationFormat>(255); }},
        {"unknown physical layout", [](auto& x) { x.layout = static_cast<backend::Layout>(255); }},
        {"unknown numerical contract", [](auto& x) { x.numerics = static_cast<backend::NumericalContract>(255); }},
        {"shape/token width must be positive", [](auto& x) { x.shape.tokens = 0; }}
    };
    for (const auto& [expected, mutate] : structural) {
        auto request = program().calls[0]; mutate(request);
        auto model_program = backend::ModelProgram{backend::Phase::decode, {request}};
        throws_exact<std::invalid_argument>([&] { valid(a,s,model_program,o); },
            "invalid model kernel request: " + expected);
    }

    auto malformed = p; malformed.calls[0].physical.weight_row_stride = 0;
    throws_exact<std::invalid_argument>([&] { valid(a,s,malformed,o); }, "invalid model kernel request: byte strides must be positive");
}

struct TrackedBackingDeleter {
    int* destroyed{};
    void operator()(std::vector<std::byte>* value) const noexcept { delete value; ++*destroyed; }
};
struct ExecutionResources {
    runtime::ResourceLease caller;
    memory::AllocationLease allocation;
};
struct FakeSession final : runtime::IExecutionSession {
    FakeSession(const backend::ModelProgram& borrowed, memory::AllocationLease allocation,
                int& copied, int& destroyed, bool fail_after_take = false)
        : prebound(borrowed), allocation(std::move(allocation)), destroyed(destroyed) {
        if (fail_after_take) throw std::runtime_error("test failure after allocation transfer");
        ++copied;
    }
    ~FakeSession() override { ++destroyed; }
    runtime::Completion execute(runtime::ResourceLease resources) override {
        source = std::make_unique<runtime::CompletionSource>(std::move(resources));
        return source->completion();
    }
    bool bound() const { return !prebound.calls.empty(); }
    backend::ModelProgram prebound;
    memory::AllocationLease allocation;
    std::unique_ptr<runtime::CompletionSource> source;
    int& destroyed;
};
struct FakeRuntime final : runtime::IModelRuntime {
    FakeRuntime(runtime::IExecutionSession& session, int& destroyed, int& observed)
        : slots(1), session(session), destroyed(destroyed), observed(observed) {}
    ~FakeRuntime() override {
        const auto* fake = dynamic_cast<const FakeSession*>(&session);
        if (fake && fake->bound()) ++observed;
        ++destroyed;
    }
    runtime::SequenceHandle create_sequence(std::uint64_t context) override {
        if (!context) throw std::invalid_argument("positive context required");
        return slots.acquire();
    }
    runtime::Completion release_sequence(runtime::SequenceHandle h) override {
        auto token = slots.begin_release(h);
        runtime::CompletionSource fence; auto result = fence.completion(); fence.succeed();
        slots.finish_release(std::move(token), result); return result;
    }
    runtime::SequenceSlots slots;
    runtime::IExecutionSession& session;
    int& destroyed;
    int& observed;
};
struct FakeDefinition final : runtime::IModelDefinition {
    explicit FakeDefinition(int& copies, int& runtime_destroyed, int& runtime_observed)
        : copies(copies), runtime_destroyed(runtime_destroyed), runtime_observed(runtime_observed) {}
    runtime::ModelDescription describe(const runtime::ArtifactMetadata& a, const runtime::RuntimeOptions& o) const override {
        auto result = runtime::ModelDescription{spec(), program(), {}};
        valid(a, result.spec, result.program, o);
        return result;
    }
    std::unique_ptr<runtime::IModelRuntime> create_runtime(const runtime::ModelDescription& d,
                                                            runtime::IExecutionSession& session) const override {
        if (d.program.calls.empty()) throw std::invalid_argument("description needs calls");
        return std::make_unique<FakeRuntime>(session, runtime_destroyed, runtime_observed);
    }
    int& copies; int& runtime_destroyed; int& runtime_observed;
};
struct FakeBackend final : runtime::IGpuBackend {
    FakeBackend(int& copies, int& session_destroyed, int& backing_destroyed)
        : copies(copies), session_destroyed(session_destroyed), backing_destroyed(backing_destroyed) {}
    backend::DeviceCaps capabilities() const override { return {}; }
    std::unique_ptr<runtime::IExecutionSession> instantiate(const runtime::ModelDescription& d,
                                                              memory::BringupGrant& grant) override {
        auto allocation = grant.allocate("weights", 16, [&](std::uint64_t cap) {
            std::unique_ptr<std::vector<std::byte>, TrackedBackingDeleter> owner(
                new std::vector<std::byte>(static_cast<std::size_t>(cap)), {&backing_destroyed});
            return memory::MemoryBacking{memory::OwnedBacking(std::move(owner)), cap};
        });
        return std::make_unique<FakeSession>(d.program, std::move(allocation), copies, session_destroyed,
                                             fail_session_construction);
    }
    int& copies; int& session_destroyed; int& backing_destroyed; bool fail_session_construction{false};
};
struct Counts { int sentinel_destroyed{}; int description_copies{}; int runtime_destroyed{};
    int runtime_observed{}; int session_destroyed{}; int backing_destroyed{}; };
memory::StaticCandidate candidate() {
    memory::AllocationRecord weights; weights.id="weights"; weights.owner="fake"; weights.logical_bytes=16;
    memory::AllocationRecord runtime; runtime.id="runtime"; runtime.owner="fake";
    runtime.source=memory::AllocationSource::runtime_overhead; runtime.logical_bytes=0; runtime.overhead_bound=0;
    return {"test-bringup", memory::GraphMode::disabled, {weights, runtime}};
}
memory::Budgets budgets() { return {{{1024,1024,0},{1024,1024,0},{1024,1024,0}}}; }
void registry_and_composition_tests() {
    Counts counts;
    runtime::ModelRegistry model_registry;
    runtime::BackendRegistry backend_registry;
    auto definition = std::make_shared<FakeDefinition>(counts.description_copies, counts.runtime_destroyed,
                                                        counts.runtime_observed);
    auto backend = std::make_shared<FakeBackend>(counts.description_copies, counts.session_destroyed,
                                                  counts.backing_destroyed);
    std::weak_ptr<const runtime::IModelDefinition> weak_definition = definition;
    std::weak_ptr<runtime::IGpuBackend> weak_backend = backend;
    model_registry.add("same-id", definition); backend_registry.add("same-id", backend);
    throws_exact<std::invalid_argument>([&] { model_registry.add("", definition); }, "model registration requires ID and definition");
    throws_exact<std::invalid_argument>([&] { model_registry.add("null", {}); }, "model registration requires ID and definition");
    throws_exact<std::invalid_argument>([&] { model_registry.add("same-id", definition); }, "duplicate model ID: same-id");
    throws_exact<std::out_of_range>([&] { model_registry.find("missing"); }, "unknown model ID: missing");
    throws_exact<std::invalid_argument>([&] { backend_registry.add("", backend); }, "backend registration requires ID and backend");
    throws_exact<std::invalid_argument>([&] { backend_registry.add("null", {}); }, "backend registration requires ID and backend");
    throws_exact<std::invalid_argument>([&] { backend_registry.add("same-id", backend); }, "duplicate backend ID: same-id");
    throws_exact<std::out_of_range>([&] { backend_registry.find("missing"); }, "unknown backend ID: missing");
    auto selected_model = model_registry.find("same-id"); auto selected_backend = backend_registry.find("same-id");
    check(selected_model && selected_backend, "independent same-ID registry selection");
    definition.reset(); backend.reset(); model_registry = {}; backend_registry = {};
    check(!weak_definition.expired() && !weak_backend.expired(), "registry-selected owners outlive registries");

    auto failed_grant = candidate().grant(budgets());
    auto failing_backend = std::make_shared<FakeBackend>(counts.description_copies, counts.session_destroyed,
                                                          counts.backing_destroyed);
    failing_backend->fail_session_construction = true;
    auto description = selected_model->describe(metadata(), options());
    throws_exact<std::runtime_error>([&] { (void)failing_backend->instantiate(description, failed_grant); },
                                     "test failure after allocation transfer");
    check(failed_grant.report().live_bytes[static_cast<std::size_t>(memory::Location::device)] == 0 &&
          counts.backing_destroyed == 1, "failed session construction unwinds grant-backed storage");
    const int backing_releases_before_session = counts.backing_destroyed;

    auto grant = candidate().grant(budgets());
    auto session = selected_backend->instantiate(description, grant);
    check(session && counts.description_copies == 1, "registry-selected backend instantiated/prebound program");
    auto* fake_session = dynamic_cast<FakeSession*>(session.get());
    check(fake_session && fake_session->bound(), "session has immutable prebound work");
    auto model_runtime = selected_model->create_runtime(description, *session);
    check(model_runtime != nullptr, "definition creates runtime");
    description.program.calls.clear(); description.spec.model_identity.clear();
    check(fake_session->bound(), "session owns program independently of borrowed description");
    auto sequence = model_runtime->create_sequence(32);
    check(static_cast<bool>(sequence), "runtime creates generation-tagged sequence");
    auto release = model_runtime->release_sequence(sequence); release.result();

    auto caller = runtime::ResourceLease(std::make_unique<Sentinel>(counts.sentinel_destroyed));
    runtime::ResourceLease resources(ExecutionResources{std::move(caller), fake_session->allocation});
    auto first_consumer = session->execute(std::move(resources));
    auto last_consumer = first_consumer;
    check(counts.sentinel_destroyed == 0 && !first_consumer.is_ready(), "pending work retains caller resource");
    check(fake_session->source->succeed(), "physical work settles once");
    fake_session->source.reset(); // producer is gone; only consumer handles retain CompletionState.
    model_runtime.reset();
    check(counts.runtime_destroyed == 1 && counts.runtime_observed == 1 && counts.session_destroyed == 0,
          "runtime uses session and is destroyed first");
    session.reset();
    check(counts.session_destroyed == 1 && counts.backing_destroyed == backing_releases_before_session,
          "session destruction releases only its allocation share");
    check(grant.report().live_bytes[static_cast<std::size_t>(memory::Location::device)] == 16 &&
          counts.sentinel_destroyed == 0, "completion consumers retain caller resource and grant allocation");
    first_consumer = {};
    check(counts.sentinel_destroyed == 0 && grant.report().live_bytes[2] == 16,
          "one remaining completion consumer retains both resources");
    last_consumer = {};
    check(counts.sentinel_destroyed == 1 && counts.backing_destroyed == backing_releases_before_session + 1 &&
          grant.report().live_bytes[2] == 0, "last completion release frees retained resources");
    selected_model.reset(); selected_backend.reset();
    check(weak_definition.expired() && weak_backend.expired(), "selected owners expire after final handle release");
}
}
int main() { metadata_tests(); registry_and_composition_tests(); return 0; }
