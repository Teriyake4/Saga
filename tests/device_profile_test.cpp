#include "saga/backend/device_profile.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace {
using namespace saga::backend;

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template<class F> void throws_with(F&& function, const std::string& fragment) {
    try {
        function();
    } catch (const std::exception& error) {
        check(std::string(error.what()).find(fragment) != std::string::npos,
              "wrong exception: " + std::string(error.what()));
        return;
    }
    throw std::runtime_error("expected exception containing: " + fragment);
}

KernelRequest request() {
    KernelRequest value;
    value.operation = Operation::projection;
    value.phase = Phase::decode;
    value.weights = WeightEncoding::bf16;
    value.activation = ActivationFormat::bf16;
    value.layout = Layout::contiguous;
    value.numerics = NumericalContract::fp32_accumulate;
    value.shape = {8, 8, 1};
    value.physical = {16, 16, 16, 16, 0};
    value.workspace_available = 128;
    value.workspace_alignment = 16;
    return value;
}

DeviceFacts facts() {
    DeviceFacts result;
    result.caps = {DeviceKind::accelerator, "sm89", 89, 0, "physical-uuid-A"};
    result.product_identity = "NVIDIA GeForce RTX 4080";
    result.warp_size = 32;
    result.multiprocessor_count = 76;
    result.device_memory_budget_bytes = 1024;
    result.workspace_budget_bytes = 256;
    return result;
}

KernelDescriptor descriptor(std::string id, const KernelRequest& r, const DeviceCaps& caps,
                           int priority = 0, Phase phase = Phase::decode,
                           std::uint64_t workspace_bytes = 0) {
    KernelDescriptor d;
    d.id = std::move(id);
    d.operation = r.operation;
    d.phase = phase;
    d.weights = r.weights;
    d.activation = r.activation;
    d.layout = r.layout;
    d.numerics = r.numerics;
    d.shape = {{r.shape.outputs, r.shape.outputs, 1}, {r.shape.inputs, r.shape.inputs, 1},
               {r.shape.tokens, r.shape.tokens, 1}};
    d.required_physical = r.physical;
    d.workspace = {workspace_bytes, workspace_bytes == 0 ? 1U : 16U};
    d.device = {caps.kind, caps.architecture, caps.architecture_version,
                caps.architecture_version, caps.features};
    d.default_priority = priority;
    return d;
}

class TestKernel : public IKernelImplementation {
public:
    explicit TestKernel(KernelDescriptor descriptor) : descriptor_(std::move(descriptor)) {}
    KernelDescriptor descriptor() const override { return descriptor_; }
    bool supports(const KernelRequest&, const DeviceCaps&) const noexcept override { return true; }
    PreparedKernel prepare(const KernelRequest& r, KernelPreparationContext&) const override {
        return {&launch, std::make_shared<const int>(0), r, descriptor_.workspace};
    }
protected:
    static void launch(const void*, const LaunchContext&) {}
    KernelDescriptor descriptor_;
};

class FailingPrepareKernel final : public TestKernel {
public:
    using TestKernel::TestKernel;
    PreparedKernel prepare(const KernelRequest&, KernelPreparationContext&) const override {
        throw KernelUnavailable("synthetic preparation failure");
    }
};

std::shared_ptr<const IKernelImplementation> kernel(KernelDescriptor d) {
    return std::make_shared<const TestKernel>(std::move(d));
}

DeviceProfile profile_for(const DeviceFacts& observed) {
    return make_rtx4080_profile(observed);
}

PairProfile pair_profile(std::string model, std::string profile, std::string physical_device,
                         std::vector<std::string> preferences) {
    return PairProfile(PairProfileSpec{std::move(model), std::move(profile),
                                       std::move(physical_device), std::move(preferences)});
}

static_assert(!std::is_assignable_v<DeviceProfile&, DeviceProfile>);
static_assert(!std::is_assignable_v<PairProfile&, PairProfile>);
static_assert(!std::is_assignable_v<decltype(std::declval<const DeviceProfile&>().identity()), std::string>);
static_assert(!std::is_assignable_v<decltype(std::declval<const PairProfile&>().model_identity()), std::string>);

void immutable_profile_tests() {
    DeviceProfileSpec cpu_spec;
    cpu_spec.identity = "synthetic-host";
    cpu_spec.kind = DeviceKind::cpu;
    cpu_spec.architecture = "host";
    cpu_spec.preferred_implementations = {"host-default"};
    const DeviceProfile cpu_profile(cpu_spec);
    cpu_spec.identity = "mutated-input";
    cpu_spec.architecture = "mutated-architecture";
    cpu_spec.preferred_implementations[0] = "mutated-kernel";

    DeviceFacts host_facts;
    check(cpu_profile.matches(host_facts), "generic CPU profile should match synthetic host facts");
    check(cpu_profile.identity() == "synthetic-host" && cpu_profile.architecture() == "host" &&
          cpu_profile.preferred_implementations().front() == "host-default",
          "profile must own immutable values copied from its specification");
    auto accelerator_facts = facts();
    check(!cpu_profile.matches(accelerator_facts), "profile device kind must be checked against observed facts");

    DeviceProfileSpec accelerator_spec;
    accelerator_spec.identity = "synthetic-accelerator";
    accelerator_spec.kind = DeviceKind::accelerator;
    accelerator_spec.architecture = "test-accelerator";
    accelerator_spec.architecture_version = 7;
    accelerator_spec.physical_device_identity = "synthetic-uuid";
    const DeviceProfile accelerator_profile(accelerator_spec);
    accelerator_facts.caps.architecture = "test-accelerator";
    accelerator_facts.caps.architecture_version = 7;
    accelerator_facts.caps.device_identity = "synthetic-uuid";
    accelerator_facts.product_identity.clear();
    check(accelerator_profile.matches(accelerator_facts), "generic accelerator profile should be constructible");

    PairProfileSpec pair_spec{"model-copy", "profile-copy", "physical-copy", {"kernel-copy"}};
    const PairProfile pair(pair_spec);
    pair_spec.model_identity = "mutated-model";
    pair_spec.device_profile_identity = "mutated-profile";
    pair_spec.physical_device_identity = "mutated-physical";
    pair_spec.preferred_implementations.front() = "mutated-kernel";
    check(pair.model_identity() == "model-copy" && pair.device_profile_identity() == "profile-copy" &&
          pair.physical_device_identity() == "physical-copy" &&
          pair.preferred_implementations().front() == "kernel-copy",
          "pair profile must own immutable values copied from its specification");
    throws_with([&] { (void)PairProfile(PairProfileSpec{}); }, "requires model, device-profile, and physical-device identities");

    DeviceProfileSpec invalid_preferences;
    invalid_preferences.identity = "invalid-preferences";
    invalid_preferences.architecture = "host";
    invalid_preferences.preferred_implementations = {""};
    throws_with([&] { (void)DeviceProfile(invalid_preferences); }, "empty implementation preference");
}

void device_profile_tests() {
    const auto observed = facts();
    auto profile = profile_for(observed);
    check(profile.identity() == "rtx4080-sm89-76sm-warp32", "named profile identity");
    check(profile.matches(observed), "observed device facts should match profile");

    auto alias = observed;
    alias.product_identity = "GeForce RTX 4080";
    check(make_rtx4080_profile(alias).matches(alias), "probe product-name alias should be accepted");

    auto changed = observed;
    changed.caps.device_identity = "different-physical-uuid";
    check(!profile.matches(changed), "device identity mismatch must fail");
    changed = observed;
    changed.multiprocessor_count = 80;
    check(!profile.matches(changed), "topology mismatch must fail");
    changed = observed;
    changed.caps.architecture = "sm90";
    check(!profile.matches(changed), "architecture mismatch must fail");
    changed = observed;
    changed.product_identity = "NVIDIA GeForce RTX 4080 SUPER";
    throws_with([&] { (void)make_rtx4080_profile(changed); }, "reserved RTX 4080 profile identity");
    changed = observed;
    changed.product_identity = changed.caps.device_identity;
    throws_with([&] { (void)make_rtx4080_profile(changed); }, "reserved RTX 4080 profile identity");
    changed = observed;
    changed.caps.device_identity = "marketing-name-says-rtx-4080";
    changed.product_identity = "NVIDIA GeForce RTX 4080";
    changed.caps.architecture = "unknown";
    throws_with([&] { (void)make_rtx4080_profile(changed); }, "reserved RTX 4080 profile identity");

    DeviceProfileSpec reserved;
    reserved.identity = "rtx4080-sm89-76sm-warp32";
    reserved.kind = DeviceKind::accelerator;
    reserved.architecture = "sm89";
    reserved.architecture_version = 89;
    reserved.physical_device_identity = observed.caps.device_identity;
    reserved.product_identity = "NVIDIA GeForce RTX 4080";
    reserved.warp_size = 32;
    reserved.multiprocessor_count = 76;
    check(DeviceProfile(reserved).matches(observed), "reserved profile constructor accepts exact observed contract");

    auto forged = reserved;
    forged.product_identity = "GeForce RTX 4080 SUPER";
    throws_with([&] { (void)DeviceProfile(forged); }, "reserved RTX 4080 profile identity");
    forged = reserved;
    forged.kind = DeviceKind::cpu;
    throws_with([&] { (void)DeviceProfile(forged); }, "reserved RTX 4080 profile identity");
    forged = reserved;
    forged.architecture = "sm90";
    throws_with([&] { (void)DeviceProfile(forged); }, "reserved RTX 4080 profile identity");
    forged = reserved;
    forged.architecture_version = 90;
    throws_with([&] { (void)DeviceProfile(forged); }, "reserved RTX 4080 profile identity");
    forged = reserved;
    forged.warp_size = 64;
    throws_with([&] { (void)DeviceProfile(forged); }, "reserved RTX 4080 profile identity");
    forged = reserved;
    forged.multiprocessor_count = 80;
    throws_with([&] { (void)DeviceProfile(forged); }, "reserved RTX 4080 profile identity");
}

void alternative_resolution_tests() {
    const auto observed = facts();
    const auto device_profile = profile_for(observed);
    auto fused = request();
    fused.operation = Operation::gate_up_swiglu;
    auto first_projection = request();
    auto second_projection = request();
    second_projection.shape.outputs = 16;
    const ModelProgram program{Phase::decode, {fused},
                               {{"approved-two-projection-decomposition",
                                 {first_projection, second_projection}}}};

    KernelRegistry registry;
    auto invalid_fused = descriptor("fused-specialization", fused, observed.caps, 100);
    invalid_fused.shape.outputs = {16, 16, 1};
    registry.add(kernel(std::move(invalid_fused)));
    registry.add(kernel(descriptor("projection", first_projection, observed.caps)));
    registry.add(kernel(descriptor("projection-wide", second_projection, observed.caps)));
    const PairProfile forced_fused = pair_profile("model-A", device_profile.identity(),
                                                   observed.caps.device_identity, {"fused-specialization"});
    auto prepared = resolve_program("model-A", program, observed, device_profile, registry, &forced_fused);
    check(prepared.selected_alternative_identity() == "approved-two-projection-decomposition" &&
          prepared.size() == 2 && prepared.kernels()[0].implementation_id() == "projection" &&
          prepared.kernels()[1].implementation_id() == "projection-wide",
          "unavailable fused call should bind the explicit two-call decomposition");

    KernelRegistry empty;
    throws_with([&] { (void)resolve_program("model-A", program, observed, device_profile, empty); },
                "no compatible primary or approved program alternative");

    KernelRegistry failing_prepare;
    failing_prepare.add(std::make_shared<const FailingPrepareKernel>(descriptor("fused-prepare-fails", fused,
                                                                                 observed.caps, 100)));
    failing_prepare.add(kernel(descriptor("projection", first_projection, observed.caps)));
    failing_prepare.add(kernel(descriptor("projection-wide", second_projection, observed.caps)));
    throws_with([&] {
        (void)resolve_program("model-A", program, observed, device_profile, failing_prepare);
    }, "synthetic preparation failure");

    ModelProgram invalid = program;
    invalid.alternatives.front().calls.front().phase = Phase::prefill;
    throws_with([&] { (void)resolve_program("model-A", invalid, observed, device_profile, registry); },
                "phase mismatch");
    invalid = program;
    invalid.alternatives.push_back(invalid.alternatives.front());
    throws_with([&] { (void)resolve_program("model-A", invalid, observed, device_profile, registry); },
                "duplicate program alternative identity");
}

void prepared_lifetime_test() {
    const auto observed = facts();
    const auto device_profile = profile_for(observed);
    const auto r = request();
    const ModelProgram program{Phase::decode, {r}, {}};
    std::weak_ptr<const IKernelImplementation> implementation;
    {
        auto prepared = [&] {
            KernelRegistry local;
            auto owner = kernel(descriptor("owned-after-registry", r, observed.caps));
            implementation = owner;
            local.add(std::move(owner));
            return resolve_program("model-A", program, observed, device_profile, local);
        }();
        check(!implementation.expired(),
              "prepared program must retain kernel implementation after registry destruction");
        prepared.kernels()[0].launch({});
    }
    check(implementation.expired(), "kernel implementation should release with prepared program");
}

void startup_resolution_tests() {
    const auto observed = facts();
    const auto device_profile = profile_for(observed);
    const auto r = request();
    const ModelProgram program{Phase::decode, {r}, {}};
    const auto make_registry = [&](const std::string& id, int priority = 0) {
        KernelRegistry registry;
        registry.add(kernel(descriptor(id, r, observed.caps, priority)));
        return registry;
    };

    auto baseline_registry = make_registry("baseline", 2);
    auto prepared = resolve_program("model-A", program, observed, device_profile, baseline_registry);
    check(prepared.size() == 1 && prepared.kernels()[0].implementation_id() == "baseline",
          "absence of pair profile selects compatible default");

    // Replacing the registry implementation changes only startup binding; the finite
    // model call list remains identical.
    auto replacement_registry = make_registry("replacement", 1);
    auto replacement = resolve_program("model-A", program, observed, device_profile, replacement_registry);
    check(program.calls.size() == 1 && replacement.kernels()[0].implementation_id() == "replacement",
          "kernel replacement must not change model program");

    KernelRegistry ranked;
    ranked.add(kernel(descriptor("default", r, observed.caps, 100)));
    ranked.add(kernel(descriptor("card-choice", r, observed.caps, 0)));
    ranked.add(kernel(descriptor("pair-choice", r, observed.caps, -10)));
    DeviceProfileSpec card_spec{device_profile.identity(), device_profile.kind(), device_profile.architecture(),
                                device_profile.architecture_version(), device_profile.physical_device_identity(),
                                device_profile.product_identity(), device_profile.warp_size(),
                                device_profile.multiprocessor_count(),
                                device_profile.minimum_device_memory_budget_bytes(),
                                device_profile.minimum_workspace_budget_bytes(), {"card-choice"}};
    const DeviceProfile card_profile(std::move(card_spec));
    const PairProfile pair = pair_profile("model-A", card_profile.identity(),
                                          observed.caps.device_identity, {"pair-choice"});
    auto pair_selected = resolve_program("model-A", program, observed, card_profile, ranked, &pair);
    check(pair_selected.kernels()[0].implementation_id() == "pair-choice",
          "pair preference ranks before card preference and backend priority");

    const PairProfile wrong_model = pair_profile("model-B", card_profile.identity(),
                                                 observed.caps.device_identity, {"pair-choice"});
    throws_with([&] { (void)resolve_program("model-A", program, observed, card_profile, ranked, &wrong_model); },
                "pair profile identity mismatch");
    const PairProfile wrong_profile = pair_profile("model-A", "another-device-profile",
                                                   observed.caps.device_identity, {"pair-choice"});
    throws_with([&] { (void)resolve_program("model-A", program, observed, card_profile, ranked, &wrong_profile); },
                "pair profile identity mismatch");

    // An incompatible forced pair ID cannot bypass operation/phase/workspace guards;
    // the next compatible card preference is selected instead.
    KernelRegistry fallback_registry;
    fallback_registry.add(kernel(descriptor("compatible-default", r, observed.caps, 50)));
    fallback_registry.add(kernel(descriptor("wrong-phase", r, observed.caps, 100, Phase::prefill)));
    DeviceProfileSpec fallback_card_spec{device_profile.identity(), device_profile.kind(),
                                         device_profile.architecture(), device_profile.architecture_version(),
                                         device_profile.physical_device_identity(), device_profile.product_identity(),
                                         device_profile.warp_size(), device_profile.multiprocessor_count(),
                                         device_profile.minimum_device_memory_budget_bytes(),
                                         device_profile.minimum_workspace_budget_bytes(), {"compatible-default"}};
    const DeviceProfile fallback_card(std::move(fallback_card_spec));
    const PairProfile incompatible_pair = pair_profile("model-A", fallback_card.identity(),
                                                       observed.caps.device_identity, {"wrong-phase"});
    auto fallback = resolve_program("model-A", program, observed, fallback_card, fallback_registry,
                                    &incompatible_pair);
    check(fallback.kernels()[0].implementation_id() == "compatible-default",
          "incompatible pair preference falls back to compatible card/default candidate");

    KernelRegistry no_valid_candidate;
    no_valid_candidate.add(kernel(descriptor("too-much-workspace", r, observed.caps, 0,
                                             Phase::decode, 256)));
    throws_with([&] {
        (void)resolve_program("model-A", program, observed, device_profile, no_valid_candidate,
                              &incompatible_pair);
    }, "missing compatible kernel");

    KernelRegistry workspace_guard;
    workspace_guard.add(kernel(descriptor("workspace-guarded", r, observed.caps, 0,
                                           Phase::decode, 256)));
    throws_with([&] { (void)resolve_program("model-A", program, observed, device_profile, workspace_guard); },
                "missing compatible kernel");

    const PairProfile wrong_device = pair_profile("model-A", card_profile.identity(),
                                                  "another-physical-uuid", {"pair-choice"});
    throws_with([&] { (void)resolve_program("model-A", program, observed, card_profile, ranked, &wrong_device); },
                "physical device");

    auto mismatch_facts = observed;
    mismatch_facts.caps.device_identity = "different-physical-device";
    throws_with([&] {
        (void)resolve_program("model-A", program, mismatch_facts, device_profile, baseline_registry);
    }, "do not match device profile");
}
} // namespace

int main() {
    try {
        immutable_profile_tests();
        device_profile_tests();
        startup_resolution_tests();
        alternative_resolution_tests();
        prepared_lifetime_test();
        std::cout << "device profile policy tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
