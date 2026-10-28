#include "saga/backend/contracts.h"
#include "saga/kv/catalog.h"
#include "saga/memory/checked.h"

#include <array>
#include <cstddef>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace saga::backend;

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void throws_with(F&& function, const std::string& fragment) {
    try { function(); }
    catch (const std::exception& error) {
        check(std::string(error.what()).find(fragment) != std::string::npos,
              "wrong exception: " + std::string(error.what()) + "; wanted " + fragment);
        return;
    }
    throw std::runtime_error("expected exception containing: " + fragment);
}

KernelRequest synthetic_request() {
    KernelRequest request;
    request.operation = Operation::gate_up_swiglu;
    request.weights = WeightEncoding::gsq3_group128_v1;
    request.layout = Layout::gsq3_tiled_v1;
    request.numerics = NumericalContract::bf16_intermediate_rounding;
    request.shape = {256, 512, 1};
    request.physical = {256, 1024, 512, 256, 128};
    request.workspace_available = 4096;
    request.workspace_alignment = 256;
    return request;
}
DeviceCaps synthetic_device() {
    return {DeviceKind::accelerator, "synthetic-sm", 89, 3, "fake-device-not-hardware-evidence"};
}
KernelDescriptor exact_descriptor(std::string id, const KernelRequest& r, const DeviceCaps& caps) {
    KernelDescriptor d;
    d.id = std::move(id);
    d.operation = r.operation;
    d.phase = r.phase;
    d.weights = r.weights;
    d.activation = r.activation;
    d.layout = r.layout;
    d.numerics = r.numerics;
    d.kv_pair = r.kv_pair;
    d.shape = {{r.shape.outputs, r.shape.outputs, 1}, {r.shape.inputs, r.shape.inputs, 1},
               {r.shape.tokens, r.shape.tokens, 1}};
    d.required_physical = r.physical;
    d.workspace = {r.workspace_available, r.workspace_alignment};
    d.device = {caps.kind, caps.architecture, caps.architecture_version, caps.architecture_version, caps.features};
    return d;
}
struct Counts {
    int supports{0};
    int prepares{0};
    int launches{0};
    bool metadata_destructor_saw_live_implementation{false};
};
struct LaunchMetadata {
    std::shared_ptr<Counts> counts;
    int result;
    std::weak_ptr<const IKernelImplementation> implementation;
    LaunchMetadata(std::shared_ptr<Counts> counters, int value,
                   std::weak_ptr<const IKernelImplementation> owner)
        : counts(std::move(counters)), result(value), implementation(std::move(owner)) {}
    ~LaunchMetadata() { counts->metadata_destructor_saw_live_implementation = !implementation.expired(); }
};
class FakeKernel final : public IKernelImplementation, public std::enable_shared_from_this<FakeKernel> {
public:
    FakeKernel(KernelDescriptor descriptor, std::shared_ptr<Counts> counts, int result = 7)
        : descriptor_(std::move(descriptor)), counts_(std::move(counts)), result_(result) {}
    KernelDescriptor descriptor() const override { return descriptor_; }
    bool supports(const KernelRequest&, const DeviceCaps&) const noexcept override {
        ++counts_->supports;
        return allow;
    }
    PreparedKernel prepare(const KernelRequest& request, KernelPreparationContext&) const override {
        ++counts_->prepares;
        auto bindings = request;
        if (corrupt_binding) bindings.numerics = NumericalContract::fp32_accumulate;
        return PreparedKernel(&launch_entry, std::make_shared<const LaunchMetadata>(counts_, result_, shared_from_this()),
                              bindings, descriptor_.workspace);
    }
    bool allow{true};
    bool corrupt_binding{false};
private:
    static void launch_entry(const void* opaque, const LaunchContext& context) {
        const auto& metadata = *static_cast<const LaunchMetadata*>(opaque);
        ++metadata.counts->launches;
        if (!context.bindings.empty()) *static_cast<int*>(context.bindings.front().address) = metadata.result;
    }
    KernelDescriptor descriptor_;
    std::shared_ptr<Counts> counts_;
    int result_;
};

void registry_tests() {
    const auto request = synthetic_request();
    const auto device = synthetic_device();
    PairTuningProfile forced{{"specialized", "unknown-profile-entry"}};
    KernelRegistry empty;
    throws_with([&] { empty.resolve(request, device); }, "missing compatible kernel: gate_up_swiglu/decode");
    throws_with([&] { empty.resolve(request, device); }, "weights=gsq3_group128_v1 activation=bf16");

    // Removing a profile or replacing a registered kernel changes no model or scheduler code.
    ModelProgram program{Phase::decode, {request}};
    auto baseline_counts = std::make_shared<Counts>();
    auto specialization_counts = std::make_shared<Counts>();
    KernelRegistry registry;
    auto default_descriptor = exact_descriptor("default", request, device);
    default_descriptor.default_priority = 10;
    registry.add(std::make_shared<FakeKernel>(default_descriptor, baseline_counts, 11));
    auto specialized = std::make_shared<FakeKernel>(exact_descriptor("specialized", request, device), specialization_counts, 19);
    registry.add(specialized);
    check(registry.resolve(program.calls.front(), device).implementation_id() == "default", "default without profile");
    check(registry.resolve(request, device, &forced).implementation_id() == "specialized", "compatible preference");
    specialized->allow = false;
    check(registry.resolve(request, device, &forced).implementation_id() == "default", "supports cannot be bypassed");
    specialized->allow = true;
    throws_with([&] { registry.add(specialized); }, "duplicate");
    throws_with([&] { registry.add(nullptr); }, "null");

    using Mutation = std::function<void(KernelRequest&, DeviceCaps&)>;
    const std::vector<std::pair<std::string, Mutation>> guards{
        {"operation", [](auto& r, auto&) { r.operation = Operation::projection; }},
        {"phase", [](auto& r, auto&) { r.phase = Phase::verify; }},
        {"weight encoding", [](auto& r, auto&) { r.weights = WeightEncoding::bf16; }},
        {"activation dtype", [](auto& r, auto&) { r.activation = ActivationFormat::fp16; }},
        {"physical layout", [](auto& r, auto&) { r.layout = Layout::gsq3_canonical_v1; }},
        {"numerical contract", [](auto& r, auto&) { r.numerics = NumericalContract::fp32_accumulate; }},
        {"KV pair", [](auto& r, auto&) { r.kv_pair = saga::kv::Pair{}; }},
        {"shape/token width", [](auto& r, auto&) { r.shape.outputs = 128; }},
        {"shape/token width", [](auto& r, auto&) { r.shape.inputs = 256; }},
        {"shape/token width", [](auto& r, auto&) { r.shape.tokens = 2; }},

        {"base alignment", [](auto& r, auto&) { r.physical.alignment = 128; }},
        {"byte strides", [](auto& r, auto&) { r.physical.weight_row_stride = 255; }},
        {"byte strides", [](auto& r, auto&) { r.physical.activation_row_stride = 1023; }},
        {"byte strides", [](auto& r, auto&) { r.physical.output_row_stride = 511; }},
        {"workspace size/alignment", [](auto& r, auto&) { r.workspace_available = 4095; }},
        {"workspace size/alignment", [](auto& r, auto&) { r.workspace_alignment = 128; }},
        {"device architecture/capabilities", [](auto&, auto& c) { c.kind = DeviceKind::cpu; }},
        {"device architecture/capabilities", [](auto&, auto& c) { c.architecture = "other"; }},
        {"device architecture/capabilities", [](auto&, auto& c) { c.architecture_version = 90; }},
        {"device architecture/capabilities", [](auto&, auto& c) { c.architecture_version = 88; }},
        {"device architecture/capabilities", [](auto&, auto& c) { c.features = 1; }}
    };
    for (const auto& [reason, mutate] : guards) {
        auto changed = request;
        auto changed_device = device;
        mutate(changed, changed_device);
        auto counts = std::make_shared<Counts>();
        KernelRegistry guarded;
        guarded.add(std::make_shared<FakeKernel>(exact_descriptor("specialized", request, device), counts));
        throws_with([&] { guarded.resolve(changed, changed_device, &forced); }, reason);
        check(counts->supports == 0 && counts->prepares == 0, "incompatible candidate reached implementation");
        // This is an explicitly registered fake alternative for the changed contract, not a silent conversion.
        guarded.add(std::make_shared<FakeKernel>(exact_descriptor("default", changed, changed_device), baseline_counts));
        check(guarded.resolve(changed, changed_device, &forced).implementation_id() == "default",
              "forced profile bypassed guard: " + reason);
    }
    auto bad_range = default_descriptor;
    bad_range.id = "invalid";
    bad_range.shape.tokens.multiple = 0;
    throws_with([&] { registry.add(std::make_shared<FakeKernel>(bad_range, baseline_counts)); }, "invalid kernel descriptor");
    auto strided = request;
    strided.physical.alignment = 3;
    check(incompatibility(default_descriptor, strided, device).has_value(), "non-power-of-two alignment accepted");
    auto multiple = default_descriptor;
    multiple.shape.inputs = {1, 1024, 128};
    strided = request;
    strided.shape.inputs = 513;
    check(incompatibility(multiple, strided, device) == "shape/token width", "shape multiple guard");

    KernelRegistry broken;
    auto liar = std::make_shared<FakeKernel>(exact_descriptor("liar", request, device), baseline_counts);
    liar->corrupt_binding = true;
    broken.add(liar);
    throws_with([&] { broken.resolve(request, device); }, "changed binding/workspace contract");
    throws_with([&] { PreparedKernel(nullptr, std::make_shared<int>(1), request, {}); }, "entry");

    auto launch_counts = std::make_shared<Counts>();
    std::weak_ptr<const IKernelImplementation> implementation_lifetime;
    {
        auto prepared = [&] {
            KernelRegistry local;
            auto implementation = std::make_shared<FakeKernel>(exact_descriptor("replacement", request, device), launch_counts, 42);
            implementation_lifetime = implementation;
            local.add(implementation);
            return local.resolve(request, device);
        }();
        check(!implementation_lifetime.expired(), "prepared launcher lost implementation lifetime");
        int output = 0;
        const std::array bindings{BufferView{&output, sizeof(output)}};
        alignas(256) std::array<std::byte, 4096 + 256> workspace{};
        const BufferView valid_workspace{workspace.data(), 4096};
        const auto reject_workspace = [&](BufferView supplied, const std::string& reason) {
            const auto calls_before = launch_counts->launches;
            const auto output_before = output;
            throws_with([&] { prepared.launch({bindings, supplied, 0}); }, reason);
            check(launch_counts->launches == calls_before && output == output_before,
                  "invalid workspace reached launch entry");
        };
        reject_workspace({}, "undersized");
        reject_workspace({nullptr, 4096}, "non-null");
        reject_workspace({workspace.data(), 4095}, "undersized");
        reject_workspace({workspace.data() + 1, 4096}, "misaligned");
        reject_workspace({workspace.data() + 128, 4096}, "misaligned");
        for (int token = 0; token < 16; ++token) prepared.launch({bindings, valid_workspace, 0});
        check(output == 42 && launch_counts->launches == 16, "owned immutable launch metadata");
        check(launch_counts->supports == 1 && launch_counts->prepares == 1, "per-token startup dispatch");
        auto moved = std::move(prepared);
        throws_with([&] { prepared.launch({}); }, "moved-from");
        moved.launch({bindings, {workspace.data(), workspace.size()}, 0});
        check(launch_counts->launches == 17, "valid larger workspace rejected after move");
    }
    check(implementation_lifetime.expired(), "prepared implementation leaked");
    check(launch_counts->metadata_destructor_saw_live_implementation,
          "implementation died before its launch metadata destructor");

    auto no_workspace_request = request;
    no_workspace_request.workspace_available = 0;
    auto no_workspace_descriptor = exact_descriptor("workspace-free", no_workspace_request, device);
    no_workspace_request.workspace_alignment = 1;
    auto no_workspace_counts = std::make_shared<Counts>();
    KernelRegistry no_workspace_registry;
    no_workspace_registry.add(std::make_shared<FakeKernel>(no_workspace_descriptor, no_workspace_counts));
    check(!incompatibility(no_workspace_descriptor, no_workspace_request, device),
          "unused workspace alignment rejected workspace-free candidate");
    auto no_workspace = no_workspace_registry.resolve(no_workspace_request, device);
    check(no_workspace.workspace() == WorkspaceRequirement{0, 256}, "workspace-free requirement changed");
    no_workspace.launch({});
    alignas(256) std::array<std::byte, 256> optional_workspace{};
    no_workspace.launch({{}, {optional_workspace.data(), 0}, 0});
    no_workspace.launch({{}, {optional_workspace.data(), optional_workspace.size()}, 0});
    for (const auto bytes : {0ULL, 1ULL}) {
        throws_with([&] {
            no_workspace.launch({{}, {optional_workspace.data() + 1, bytes}, 0});
        }, "misaligned");
    }
    check(no_workspace_counts->launches == 3, "workspace-free launch checks did not guard entry");
    no_workspace_request.workspace_available = optional_workspace.size();
    no_workspace_registry.resolve(no_workspace_request, device).launch({});
    check(no_workspace_counts->launches == 4, "unused available workspace alignment rejected candidate");
    for (const auto alignment : {0ULL, 3ULL}) {
        no_workspace_request.workspace_alignment = alignment;
        check(incompatibility(no_workspace_descriptor, no_workspace_request, device).has_value(),
              "workspace-free request accepted invalid alignment");
        throws_with([&] { no_workspace_registry.resolve(no_workspace_request, device); }, "invalid kernel request");
        auto malformed = no_workspace_descriptor;
        malformed.workspace.alignment = alignment;
        throws_with([&] {
            no_workspace_registry.add(std::make_shared<FakeKernel>(malformed, no_workspace_counts));
        }, "invalid kernel descriptor");
        for (const auto bytes : {0ULL, 4096ULL}) {
            throws_with([&] {
                PreparedKernel([](const void*, const LaunchContext&) {}, std::make_shared<int>(1), request,
                               {bytes, alignment});
            }, "workspace alignment");
        }
    }
}

void well_formedness_tests() {
    const auto request = synthetic_request();
    const auto device = synthetic_device();
    const auto descriptor = exact_descriptor("valid", request, device);
    auto counts = std::make_shared<Counts>();
    KernelRegistry registry;
    registry.add(std::make_shared<FakeKernel>(descriptor, counts));
    KernelRegistry empty;
    using Mutation = std::function<void(KernelRequest&, DeviceCaps&)>;
    const std::vector<std::pair<std::string, Mutation>> invalid_contracts{
        {"unknown operation", [](auto& r, auto&) { r.operation = static_cast<Operation>(255); }},
        {"unknown phase", [](auto& r, auto&) { r.phase = static_cast<Phase>(255); }},
        {"unknown weight encoding", [](auto& r, auto&) { r.weights = static_cast<WeightEncoding>(255); }},
        {"unknown activation dtype", [](auto& r, auto&) { r.activation = static_cast<ActivationFormat>(255); }},
        {"unknown physical layout", [](auto& r, auto&) { r.layout = static_cast<Layout>(255); }},
        {"unknown numerical contract", [](auto& r, auto&) { r.numerics = static_cast<NumericalContract>(255); }},
        {"unknown KV codec", [](auto& r, auto&) { r.kv_pair = saga::kv::Pair{static_cast<saga::kv::Codec>(255), saga::kv::Codec::fp16}; }},
        {"unknown KV codec", [](auto& r, auto&) { r.kv_pair = saga::kv::Pair{saga::kv::Codec::fp16, static_cast<saga::kv::Codec>(255)}; }},
        {"unknown device kind", [](auto&, auto& c) { c.kind = static_cast<DeviceKind>(255); }},
        {"quantization group 128", [](auto& r, auto&) { r.physical.quantization_group = 0; }},
        {"quantization group 128", [](auto& r, auto&) { r.physical.quantization_group = 64; }},
        {"quantization group 128", [](auto& r, auto&) { r.physical.quantization_group = 256; }},
        {"shape/token", [](auto& r, auto&) { r.shape.outputs = 0; }},
        {"shape/token", [](auto& r, auto&) { r.shape.inputs = 0; }},
        {"shape/token", [](auto& r, auto&) { r.shape.tokens = 0; }},
        {"base alignment", [](auto& r, auto&) { r.physical.alignment = 0; }},
        {"base alignment", [](auto& r, auto&) { r.physical.alignment = 3; }},
        {"byte strides", [](auto& r, auto&) { r.physical.weight_row_stride = 0; }},
        {"byte strides", [](auto& r, auto&) { r.physical.activation_row_stride = 0; }},
        {"byte strides", [](auto& r, auto&) { r.physical.output_row_stride = 0; }},
        {"workspace alignment", [](auto& r, auto&) { r.workspace_alignment = 0; }},
        {"workspace alignment", [](auto& r, auto&) { r.workspace_alignment = 3; }},
        {"device architecture", [](auto&, auto& c) { c.architecture.clear(); }}
    };
    PairTuningProfile forced{{"invalid"}};
    for (const auto& [reason, mutate] : invalid_contracts) {
        auto changed = request;
        auto changed_device = device;
        mutate(changed, changed_device);
        const auto malformed = exact_descriptor("invalid", changed, changed_device);
        const auto request_reason = incompatibility(descriptor, changed, changed_device);
        check(request_reason && request_reason->find("invalid kernel request") != std::string::npos &&
              request_reason->find(reason) != std::string::npos, "missing request validation: " + reason);
        check(incompatibility(malformed, changed, changed_device).has_value(),
              "matching malformed contracts were compatible: " + reason);
        const auto descriptor_reason = incompatibility(malformed, request, device);
        check(descriptor_reason && descriptor_reason->find("invalid kernel descriptor") != std::string::npos,
              "missing descriptor validation: " + reason);
        throws_with([&] { registry.resolve(changed, changed_device, &forced); }, "invalid kernel request");
        throws_with([&] { empty.resolve(changed, changed_device); }, "invalid kernel request");
        throws_with([&] {
            registry.add(std::make_shared<FakeKernel>(malformed, counts));
        }, "invalid kernel descriptor");
        // Invalid KV codes must also be safe to include in diagnostics.
        check(!describe(changed, changed_device).empty(), "malformed request could not be described");
    }
    check(counts->supports == 0 && counts->prepares == 0, "malformed contract reached implementation");

    // Well-formedness does not invent operation-specific restrictions or forbid catalogued KV formats.
    for (const auto operation : {Operation::projection, Operation::gate_up_swiglu, Operation::attention,
                                 Operation::kv_append, Operation::recurrent_update}) {
        for (const auto& codec : saga::kv::catalog()) {
            auto future = request;
            future.operation = operation;
            future.weights = WeightEncoding::none;
            future.physical.quantization_group = 0;
            future.kv_pair = saga::kv::Pair{codec.id, codec.id};
            KernelRegistry explicit_contract;
            explicit_contract.add(std::make_shared<FakeKernel>(exact_descriptor("explicit", future, device), counts));
            check(explicit_contract.resolve(future, device).bindings() == future, "explicit contract overconstrained");
        }
    }
}

void catalog_tests() {
    using namespace saga::kv;
    using saga::memory::checked_add;
    using saga::memory::checked_multiply;
    constexpr std::uint64_t mib = 1024 * 1024;
    const std::array<std::pair<Codec, std::uint64_t>, 11> same_codec{{
        {Codec::f32, 1024}, {Codec::fp16, 512}, {Codec::bf16, 512}, {Codec::q8_0, 272},
        {Codec::q5_0, 176}, {Codec::q5_1, 192}, {Codec::q4_0, 144}, {Codec::q4_1, 160},
        {Codec::iq4_nl, 144}, {Codec::int8_g64, 264}, {Codec::int4_g64, 136}
    }};
    check(catalog().size() == same_codec.size(), "catalog size");
    for (const auto& [codec, expected_mib] : same_codec) {
        const auto& entry = descriptor(codec);
        check(parse_codec(entry.name) == codec && !entry.numerical_version.empty(), "codec identity");
        check(entry.rejects_nonfinite && entry.masks_zero_padded_lanes, "numeric tail/nonfinite policy");
        check(main_cache_bytes({}, {codec, codec}, 8192).total_bytes == expected_mib * mib,
              "proposal 8K same-codec fixture: " + std::string(entry.name));
    }
    check(parse_codec("f16") == Codec::fp16 && parse_codec("bf16") != Codec::fp16, "distinct dtype/alias contract");
    check(descriptor(Codec::q4_0).family != descriptor(Codec::iq4_nl).family, "nonlinear contract collapsed");
    check(descriptor(Codec::int4_g64).minimum_code == -7 && descriptor(Codec::int4_g64).maximum_code == 7,
          "reserved INT4 code policy");
    check(descriptor(Codec::int8_g64).scale_policy == ScalePolicy::normal_fp16_clamped, "custom scale clamp policy");
    check(descriptor(Codec::q4_0).scale_policy != ScalePolicy::normal_fp16_clamped, "GGML custom policy leakage");
    for (std::uint64_t multiplier : {1ULL, 2ULL, 4ULL}) {
        check(main_cache_bytes({}, {Codec::int8_g64, Codec::int8_g64}, 8192 * multiplier).total_bytes == 264 * multiplier * mib, "INT8 fixtures");
        check(main_cache_bytes({}, {Codec::int8_g64, Codec::int4_g64}, 8192 * multiplier).total_bytes == 200 * multiplier * mib, "mixed group64 fixtures");
        check(main_cache_bytes({}, {Codec::int4_g64, Codec::int4_g64}, 8192 * multiplier).total_bytes == 136 * multiplier * mib, "INT4 fixtures");
    }
    for (const auto& [value, expected_mib] : std::array<std::pair<Codec, std::uint64_t>, 3>{{
             {Codec::q4_0, 208}, {Codec::q5_0, 224}, {Codec::q5_1, 232}}})
        check(main_cache_bytes({}, {Codec::q8_0, value}, 8192).total_bytes == expected_mib * mib, "GGML mixed fixture");
    check(group64_numerics.minimum_scale == 0x1p-14f && group64_numerics.maximum_scale == 65504.0f &&
          group64_numerics.zero_group_scale == 1.0f && group64_numerics.quantize_using_stored_scale &&
          group64_numerics.scale_and_codes_nearest_even && group64_numerics.int4_lower_channel_in_low_nibble &&
          group64_numerics.int4_signed_twos_complement && group64_numerics.little_endian_fp16_scales,
          "proposal G64 numeric metadata");
    const auto group64 = main_cache_bytes({}, {Codec::int8_g64, Codec::int8_g64}, 8192);
    check(group64.key_bytes_per_token == 16896 && group64.value_bytes_per_token == 16896, "scales counted exactly once");
    const auto tail = main_cache_bytes({1, 1, 65}, {Codec::int8_g64, Codec::int4_g64}, 3, {2, 256, 8});
    check(tail.key_bytes_per_token == 132 && tail.value_bytes_per_token == 68, "head-local block ceil");
    check(tail.rounded_capacity_tokens == 4 && tail.canonical_block_bytes == 800, "token pages");
    check(tail.total_bytes == 2064 && tail.layout_padding_bytes == 1248 && tail.table_bytes == 16,
          "separate code and scale planes page alignment");
    const auto independent_heads = main_cache_bytes({1, 2, 33}, {Codec::q4_0, Codec::q4_0}, 1);
    check(independent_heads.total_bytes == 144, "blocks illegally shared across heads");
    const auto draft = main_cache_bytes({1, 4, 256}, {Codec::fp16, Codec::fp16}, 8192);
    check(draft.total_bytes == 32 * mib, "draft geometry must not use main layer count");
    throws_with([] { parse_codec("int4"); }, "unknown KV codec");
    throws_with([] { descriptor(static_cast<Codec>(255)); }, "unknown KV codec ID");
    throws_with([] { main_cache_bytes({}, {}, 0); }, "positive");
    throws_with([] { main_cache_bytes({0, 4, 256}, {}, 1); }, "positive");
    throws_with([] { main_cache_bytes({}, {}, 1, {0, 1, 0}); }, "positive");
    throws_with([] { main_cache_bytes({}, {}, 1, {1, 3, 0}); }, "power of two");
    constexpr auto maximum = std::numeric_limits<std::uint64_t>::max();
    throws_with([&] { checked_add(maximum, 1); }, "overflow");
    throws_with([&] { checked_multiply(maximum, 2); }, "overflow");
    check(checked_multiply(maximum, 0) == 0, "zero multiplication");
    check(saga::memory::ceil_divide(maximum, 64) == maximum / 64 + 1, "overflow-safe ceil");
    throws_with([&] { saga::memory::round_up(maximum, 256); }, "overflow");
    throws_with([] { saga::memory::ceil_divide(1, 0); }, "zero");
    throws_with([&] { main_cache_bytes({}, {}, maximum); }, "overflow");
    throws_with([&] { main_cache_bytes({maximum, 4, 256}, {}, 1); }, "overflow");
    throws_with([&] { main_cache_bytes({1, 1, 1}, {}, maximum, {2, 1, 0}); }, "overflow");
    throws_with([&] { main_cache_bytes({1, 1, 1}, {}, 2, {1, 1, maximum}); }, "overflow");
}
} // namespace

int main() {
    try {
        registry_tests();
        well_formedness_tests();
        catalog_tests();
        std::cout << "contracts_test: passed (synthetic host contracts only)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "contracts_test: " << error.what() << '\n';
        return 1;
    }
}
