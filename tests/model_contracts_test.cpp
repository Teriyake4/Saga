#include "saga/runtime/model_contracts.h"
#include "saga/runtime/startup.h"

#include <algorithm>
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

runtime::ArtifactMetadata metadata() { return {"artifact", {{"file-A", 64}, {"file-B", 16}}, {}, {}}; }
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
    return {backend::Phase::decode, {request}, {}};
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
    auto near_max = runtime::ArtifactMetadata{"edge", {{"edge-file", UINT64_MAX}}, {}, {}};
    auto edge = backend::ModelSpec{"edge-model", UINT64_MAX, {1,1,1},
        {binding("edge-a", "edge-a-id", "edge-file", UINT64_MAX - 8, 4),
         binding("edge-b", "edge-b-id", "edge-file", UINT64_MAX - 4, 4)}};
    valid(near_max, edge, p, runtime::RuntimeOptions{UINT64_MAX, false, false});
    auto feature_options = o; feature_options.request_mtp = true; feature_options.request_vision = true;
    valid(a, s, p, feature_options, {true, true});
    auto optional_unavailable = a;
    optional_unavailable.optional_capabilities = {runtime::Capability::vision};
    valid(optional_unavailable, s, p, o, {});

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
        {"unknown artifact capability", [&] { auto x=a; x.required_capabilities = {static_cast<runtime::Capability>(255)}; valid(x,s,p,o); }},
        {"duplicate artifact capability declaration", [&] { auto x=a; x.required_capabilities = {runtime::Capability::mtp, runtime::Capability::mtp}; valid(x,s,p,o,{true,false}); }},
        {"artifact capability is both required and optional", [&] { auto x=a; x.required_capabilities = {runtime::Capability::mtp}; x.optional_capabilities = {runtime::Capability::mtp}; valid(x,s,p,o,{true,false}); }},
        {"required artifact MTP capability unavailable", [&] { auto x=a; x.required_capabilities = {runtime::Capability::mtp}; valid(x,s,p,o); }},
        {"requested vision capability unavailable", [&] { auto x=o; x.request_vision=true; valid(optional_unavailable,s,p,x); }},
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
        auto model_program = backend::ModelProgram{backend::Phase::decode, {request}, {}};
        throws_exact<std::invalid_argument>([&] { valid(a,s,model_program,o); },
            "invalid model kernel request: " + expected);
    }

    auto malformed = p; malformed.calls[0].physical.weight_row_stride = 0;
    throws_exact<std::invalid_argument>([&] { valid(a,s,malformed,o); }, "invalid model kernel request: byte strides must be positive");

    auto alternatives = p;
    alternatives.alternatives.push_back({"small-decomposition", alternatives.calls});
    valid(a, s, alternatives, o);
    alternatives.alternatives[0].identity.clear();
    throws_exact<std::invalid_argument>([&] { valid(a,s,alternatives,o); },
                                        "empty, reserved, or duplicate program alternative identity");
    alternatives = p;
    alternatives.alternatives = {{"branch-a", p.calls}, {"branch-a", p.calls}};
    throws_exact<std::invalid_argument>([&] { valid(a,s,alternatives,o); },
                                        "empty, reserved, or duplicate program alternative identity");
    alternatives = p;
    auto wrong_phase_call = p.calls.front(); wrong_phase_call.phase = backend::Phase::prefill;
    alternatives.alternatives.push_back({"wrong-phase", {wrong_phase_call}});
    throws_exact<std::invalid_argument>([&] { valid(a,s,alternatives,o); },
                                        "kernel phase mismatches model program");
    alternatives = p;
    alternatives.alternatives.resize(backend::maximum_program_alternatives + 1,
                                      backend::ProgramAlternative{"branch", p.calls});
    for (std::size_t i = 0; i < alternatives.alternatives.size(); ++i)
        alternatives.alternatives[i].identity = "branch-" + std::to_string(i);
    throws_exact<std::invalid_argument>([&] { valid(a,s,alternatives,o); },
                                        "model program alternative limit exceeded");
    alternatives = p;
    alternatives.calls.resize(65, p.calls.front());
    valid(a, s, alternatives, o);
    alternatives = p;
    alternatives.alternatives = {{"primary", p.calls}};
    throws_exact<std::invalid_argument>([&] { valid(a,s,alternatives,o); },
                                        "empty, reserved, or duplicate program alternative identity");
    alternatives = p;
    alternatives.alternatives.push_back({"branch", std::vector<backend::KernelRequest>(
        backend::maximum_program_calls, p.calls.front())});
    throws_exact<std::invalid_argument>([&] { valid(a,s,alternatives,o); },
                                        "model program total call limit exceeded");
}

void full_contract_tests() {
    runtime::ModelDescription description;
    description.inspection.artifact = metadata();
    description.inspection.spec = spec();
    description.inspection.frontend = {"frontend-v1", runtime::FrontendPolicy::caller_token_ids, 32, {0, 1}};
    description.inspection.state.push_back({"rnn-state", runtime::StateKind::recurrent, runtime::StateDType::f32,
        {64}, memory::Location::device, runtime::StateLifetime::sequence, "state-buffer"});
    description.inspection.state.push_back({"main-kv-state", runtime::StateKind::key_value, runtime::StateDType::f16,
        {1, 1, 1}, memory::Location::device, runtime::StateLifetime::sequence, "main-kv"});
    description.resources.identity = "profile-v1";
    description.resources.model_max_context = 128;
    memory::Term weights; weights.id = "tensor-w"; weights.owner = "synthetic";
    weights.category = memory::Category::text_weights; weights.location = memory::Location::device;
    weights.lifetime = memory::Lifetime::persistent; weights.bytes = 16;
    memory::Term state; state.id = "state-buffer"; state.owner = "synthetic";
    state.category = memory::Category::fp32_recurrence; state.location = memory::Location::device;
    state.lifetime = memory::Lifetime::execution; state.bytes = 128; state.bytes_per_context = 4;
    description.resources.terms = {weights, state};
    description.programs = {program()};
    runtime::validate_model_description(description, options());
    auto invalid = description;
    invalid.resources.terms[0].bytes = 8;
    throws_exact<std::invalid_argument>([&] { runtime::validate_model_description(invalid, options()); },
                                        "weight resource allocation is zero or smaller than binding");
    invalid = description;
    invalid.resources.terms[0].bytes = 8;
    invalid.resources.terms[0].bytes_per_context = 1;
    runtime::validate_model_description(invalid, options());
    invalid = description;
    invalid.inspection.frontend.vocabulary_size = 1;
    throws_exact<std::invalid_argument>([&] { runtime::validate_model_description(invalid, options()); },
                                        "invalid or duplicate special token ID");
    invalid = description;
    invalid.inspection.state[0].dimensions[0] = 0;
    throws_exact<std::invalid_argument>([&] { runtime::validate_model_description(invalid, options()); },
                                        "state dimensions are invalid or overflow byte size");
    invalid = description;
    invalid.inspection.state[0].dimensions = {UINT64_MAX, 2};
    throws_exact<std::invalid_argument>([&] { runtime::validate_model_description(invalid, options()); },
                                        "state dimensions are invalid or overflow byte size");
    invalid = description;
    invalid.inspection.state[0].dtype = static_cast<runtime::StateDType>(255);
    throws_exact<std::invalid_argument>([&] { runtime::validate_model_description(invalid, options()); },
                                        "unknown state dtype");
    invalid = description;
    invalid.resources.model_max_context = 64;
    throws_exact<std::invalid_argument>([&] { runtime::validate_model_description(invalid, options()); },
                                        "resource profile model context mismatches model specification");
    invalid = description;
    invalid.resources.terms[0].id = "unrelated";
    throws_exact<std::invalid_argument>([&] { runtime::validate_model_description(invalid, options()); },
                                        "weight resource binding is missing");
    invalid = description;
    invalid.inspection.frontend.policy = static_cast<runtime::FrontendPolicy>(255);
    throws_exact<std::invalid_argument>([&] { runtime::validate_model_description(invalid, options()); },
                                        "unknown frontend policy");
    invalid = description;
    invalid.inspection.state[0].dtype = runtime::StateDType::f16;
    throws_exact<std::invalid_argument>([&] { runtime::validate_model_description(invalid, options()); },
                                        "recurrent state must use f32 dtype");
    invalid = description;
    invalid.resources.terms[1].bytes = 64;
    throws_exact<std::invalid_argument>([&] { runtime::validate_model_description(invalid, options()); },
                                        "state resource allocation is zero or smaller than state schema");
    invalid = description;
    invalid.resources.terms[1].bytes_per_context = 8;
    runtime::validate_model_description(invalid, options());
    invalid = description;
    invalid.resources.terms[1].bytes = 0;
    invalid.resources.terms[1].bytes_per_context.reset();
    throws_exact<std::invalid_argument>([&] { runtime::validate_model_description(invalid, options()); },
                                        "state resource allocation is zero or smaller than state schema");
    invalid = description;
    invalid.resources.terms[1].category = memory::Category::text_weights;
    throws_exact<std::invalid_argument>([&] { runtime::validate_model_description(invalid, options()); },
                                        "state resource category mismatches schema kind");
    invalid = description;
    invalid.resources.terms[0].category = memory::Category::main_kv;
    throws_exact<std::invalid_argument>([&] { runtime::validate_model_description(invalid, options()); },
                                        "weight resource category is not a weight category");
    invalid = description;
    invalid.inspection.state[0].resource_id = "tensor-w";
    throws_exact<std::invalid_argument>([&] { runtime::validate_model_description(invalid, options()); },
                                        "state resource aliases model weights");
    invalid = description;
    invalid.resources.terms[1].lifetime = memory::Lifetime::persistent;
    runtime::validate_model_description(invalid, options());
    invalid.resources.terms[1].lifetime = memory::Lifetime::bringup;
    throws_exact<std::invalid_argument>([&] { runtime::validate_model_description(invalid, options()); },
                                        "state resource must have persistent or execution lifetime");
    invalid = description;
    invalid.inspection.state[1].resource_id = "state-buffer";
    throws_exact<std::invalid_argument>([&] { runtime::validate_model_description(invalid, options()); },
                                        "empty or duplicate state identity or resource binding");
    invalid = description;
    invalid.inspection.artifact.optional_capabilities = {runtime::Capability::mtp,
                                                          runtime::Capability::mtp};
    throws_exact<std::invalid_argument>([&] { runtime::validate_model_description(invalid, options()); },
                                        "duplicate artifact capability declaration");
    invalid = description;
    invalid.inspection.artifact.required_capabilities = {runtime::Capability::vision};
    throws_exact<std::invalid_argument>([&] { runtime::validate_model_description(invalid, options()); },
                                        "required artifact vision capability unavailable");
    invalid = description;
    invalid.inspection.artifact.optional_capabilities = {runtime::Capability::vision};
    runtime::validate_model_description(invalid, options());
    invalid.inspection.capabilities.mtp = true;
    invalid.inspection.artifact.required_capabilities = {runtime::Capability::mtp};
    invalid.inspection.artifact.optional_capabilities = {runtime::Capability::mtp};
    throws_exact<std::invalid_argument>([&] { runtime::validate_model_description(invalid, options()); },
                                        "artifact capability is both required and optional");

    invalid = description;
    auto second_weight = binding("other", "tensor-other", "file-A", 32, 16);
    invalid.inspection.spec.weights.push_back(second_weight);
    invalid.resources.terms[0].physical_identity = "malicious-shared-key";
    memory::Term other_weight = invalid.resources.terms[0];
    other_weight.id = "tensor-other";
    invalid.resources.terms.push_back(other_weight);
    throws_exact<std::invalid_argument>([&] { runtime::validate_model_description(invalid, options()); },
                                        "weight resource physical identity mismatches model binding");

    invalid = description;
    invalid.resources.terms[1].bytes.reset();
    invalid.resources.terms[1].bytes_per_context.reset();
    runtime::validate_model_description(invalid, options());
    check(!invalid.resources.terms[1].bytes && !invalid.resources.terms[1].bytes_per_context,
          "unknown state bytes remain unknown in the owned profile");
    const auto unresolved = memory::evaluate(invalid.resources, options().context_tokens);
    check(std::find(unresolved.unresolved.begin(), unresolved.unresolved.end(),
                    "unknown project bytes: state-buffer") != unresolved.unresolved.end(),
          "planner keeps unknown state bytes unresolved");
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
    runtime::PendingStep prefill(runtime::PrefillRequest request, runtime::ResourceLease resources) override {
        const auto frontier = request.rows.empty() ? 0 : request.rows.front().context_tokens;
        source = std::make_unique<runtime::StepSource>(std::move(resources), runtime::StepResultBounds{frontier, std::nullopt});
        return source->pending();
    }
    runtime::Completion warmup(memory::BringupGrant&) override { runtime::CompletionSource s; s.succeed(); return s.completion(); }
    runtime::Completion capture(memory::BringupGrant&) override { runtime::CompletionSource s; s.succeed(); return s.completion(); }
    std::vector<std::pair<std::string, std::uint64_t>> overhead_measurements(memory::BringupGrant&) override { return {}; }
    void drain() noexcept override {}
    runtime::PendingStep decode(runtime::DecodeRequest request, runtime::ResourceLease resources) override {
        if (request.rows.size() != 1 || request.rows.front().token_ids.empty())
            throw std::invalid_argument("typed decode request required");
        source = std::make_unique<runtime::StepSource>(std::move(resources),
            runtime::StepResultBounds{request.rows.front().context_tokens, std::nullopt});
        return source->pending();
    }
    bool bound() const { return !prebound.calls.empty(); }
    backend::ModelProgram prebound;
    memory::AllocationLease allocation;
    std::unique_ptr<runtime::StepSource> source;
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
    runtime::PendingStep prefill(runtime::PrefillRequest request, runtime::ResourceLease resources) override {
        return session.prefill(std::move(request), std::move(resources));
    }
    runtime::PendingStep decode(runtime::DecodeRequest request, runtime::ResourceLease resources) override {
        return session.decode(std::move(request), std::move(resources));
    }
    std::optional<runtime::PendingStep> verify(runtime::VerifyRequest request,
                                               runtime::ResourceLease resources) override {
        return session.verify(std::move(request), std::move(resources));
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
    runtime::ModelInspection inspect(const runtime::ArtifactMetadata& a) const override {
        return {a, spec(), {"test-frontend", runtime::FrontendPolicy::caller_token_ids, 32000, {}}, {}, {}};
    }
    std::vector<backend::ModelProgram> describe_programs(const runtime::ModelInspection& i,
                                                          const runtime::RuntimeOptions& o) const override {
        valid(i.artifact, i.spec, program(), o);
        return {program()};
    }
    memory::Profile describe_resources(const runtime::ModelInspection& i,
                                       const runtime::RuntimeOptions&) const override {
        memory::Profile profile;
        profile.identity = "synthetic-test-profile";
        profile.model_max_context = i.spec.maximum_context_tokens;
        memory::Term weights;
        weights.id = "tensor-w"; weights.owner = "synthetic"; weights.category = memory::Category::text_weights;
        weights.location = memory::Location::device; weights.lifetime = memory::Lifetime::persistent;
        weights.bytes = 16;
        profile.terms.push_back(std::move(weights));
        return profile;
    }
    std::unique_ptr<runtime::IModelRuntime> create_runtime(const runtime::ModelDescription& d,
                                                            runtime::IExecutionSession& session) const override {
        if (d.programs.empty() || d.programs.front().calls.empty()) throw std::invalid_argument("description needs calls");
        return std::make_unique<FakeRuntime>(session, runtime_destroyed, runtime_observed);
    }
    int& copies; int& runtime_destroyed; int& runtime_observed;
};
struct FakeBackend final : runtime::IGpuBackend {
    FakeBackend(int& copies, int& session_destroyed, int& backing_destroyed)
        : copies(copies), session_destroyed(session_destroyed), backing_destroyed(backing_destroyed) {}
    backend::DeviceFacts device_facts() const override { return {}; }
    std::unique_ptr<runtime::IExecutionSession> instantiate(const runtime::ExecutionPlan& plan,
                                                              memory::BringupGrant& grant) override {
        const auto& d = plan.description();
        auto allocation = grant.allocate("weights", 16, [&](std::uint64_t cap) {
            std::unique_ptr<std::vector<std::byte>, TrackedBackingDeleter> owner(
                new std::vector<std::byte>(static_cast<std::size_t>(cap)), {&backing_destroyed});
            return memory::MemoryBacking{memory::OwnedBacking(std::move(owner)), cap};
        });
        return std::make_unique<FakeSession>(d.programs.front(), std::move(allocation), copies, session_destroyed,
                                             fail_session_construction);
    }
    int& copies; int& session_destroyed; int& backing_destroyed; bool fail_session_construction{false};
};
struct Counts { int sentinel_destroyed{}; int description_copies{}; int runtime_destroyed{};
    int runtime_observed{}; int session_destroyed{}; int backing_destroyed{}; };

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

    check(selected_model && selected_backend, "selected owners remain live after registry destruction");
    selected_model.reset(); selected_backend.reset();
    check(weak_definition.expired() && weak_backend.expired(), "selected owners expire after final handle release");
}
}
int main() { metadata_tests(); full_contract_tests(); registry_and_composition_tests(); return 0; }
