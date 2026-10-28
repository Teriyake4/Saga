#include "saga/backend/contracts.h"
#include "saga/memory/checked.h"

#include <algorithm>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace saga::backend {
namespace {
bool matches(std::uint64_t value, DimensionRange range) {
    return range.multiple != 0 && value >= range.minimum && value <= range.maximum &&
           value % range.multiple == 0;
}
bool valid_range(DimensionRange range) {
    return range.minimum != 0 && range.minimum <= range.maximum && range.multiple != 0;
}
bool stride_matches(std::uint64_t value, std::uint64_t multiple) {
    return value != 0 && multiple != 0 && value % multiple == 0;
}
const char* operation_name(Operation operation) {
    switch (operation) {
    case Operation::projection: return "projection";
    case Operation::gate_up_swiglu: return "gate_up_swiglu";
    case Operation::attention: return "attention";
    case Operation::kv_append: return "kv_append";
    case Operation::recurrent_update: return "recurrent_update";
    }
    return "invalid";
}
const char* phase_name(Phase phase) {
    switch (phase) {
    case Phase::prefill: return "prefill";
    case Phase::decode: return "decode";
    case Phase::verify: return "verify";
    }
    return "invalid";
}
const char* weight_name(WeightEncoding weights) {
    switch (weights) {
    case WeightEncoding::none: return "none";
    case WeightEncoding::fp16: return "fp16";
    case WeightEncoding::bf16: return "bf16";
    case WeightEncoding::gsq3_group128_v1: return "gsq3_group128_v1";
    }
    return "invalid";
}
const char* activation_name(ActivationFormat activation) {
    switch (activation) {
    case ActivationFormat::fp16: return "fp16";
    case ActivationFormat::bf16: return "bf16";
    case ActivationFormat::fp32: return "fp32";
    }
    return "invalid";
}
const char* layout_name(Layout layout) {
    switch (layout) {
    case Layout::contiguous: return "contiguous";
    case Layout::gsq3_canonical_v1: return "gsq3_canonical_v1";
    case Layout::gsq3_tiled_v1: return "gsq3_tiled_v1";
    case Layout::paged_kv_v1: return "paged_kv_v1";
    }
    return "invalid";
}
const char* numerical_name(NumericalContract numerics) {
    switch (numerics) {
    case NumericalContract::fp32_accumulate: return "fp32_accumulate";
    case NumericalContract::bf16_intermediate_rounding: return "bf16_intermediate_rounding";
    case NumericalContract::fp16_intermediate_rounding: return "fp16_intermediate_rounding";
    }
    return "invalid";
}
template<class Enum>
bool known(Enum value, std::initializer_list<Enum> values) {
    return std::find(values.begin(), values.end(), value) != values.end();
}
std::string_view codec_name(kv::Codec codec) {
    for (const auto& entry : kv::catalog()) if (entry.id == codec) return entry.name;
    return "invalid";
}
bool valid_codec(kv::Codec codec) {
    for (const auto& entry : kv::catalog()) if (entry.id == codec) return true;
    return false;
}
bool valid_device_kind(DeviceKind kind) {
    return known(kind, {DeviceKind::cpu, DeviceKind::accelerator});
}
template<class Contract>
std::optional<std::string> format_error(const Contract& c, const PhysicalContract& physical) {
    if (!known(c.operation, {Operation::projection, Operation::gate_up_swiglu, Operation::attention,
                             Operation::kv_append, Operation::recurrent_update})) return "unknown operation";
    if (!known(c.phase, {Phase::prefill, Phase::decode, Phase::verify})) return "unknown phase";
    if (!known(c.weights, {WeightEncoding::none, WeightEncoding::fp16, WeightEncoding::bf16,
                           WeightEncoding::gsq3_group128_v1})) return "unknown weight encoding";
    if (!known(c.activation, {ActivationFormat::fp16, ActivationFormat::bf16, ActivationFormat::fp32}))
        return "unknown activation dtype";
    if (!known(c.layout, {Layout::contiguous, Layout::gsq3_canonical_v1, Layout::gsq3_tiled_v1,
                          Layout::paged_kv_v1})) return "unknown physical layout";
    if (!known(c.numerics, {NumericalContract::fp32_accumulate, NumericalContract::bf16_intermediate_rounding,
                            NumericalContract::fp16_intermediate_rounding})) return "unknown numerical contract";
    if (c.kv_pair && (!valid_codec(c.kv_pair->key) || !valid_codec(c.kv_pair->value))) return "unknown KV codec";
    if (c.weights == WeightEncoding::gsq3_group128_v1 && physical.quantization_group != 128)
        return "gsq3_group128_v1 requires quantization group 128";
    if (!memory::is_power_of_two(physical.alignment)) return "base alignment must be a power of two";
    if (physical.weight_row_stride == 0 || physical.activation_row_stride == 0 || physical.output_row_stride == 0)
        return "byte strides must be positive";
    return std::nullopt;
}

std::optional<std::string> descriptor_error(const KernelDescriptor& d) {
    if (auto error = format_error(d, d.required_physical)) return error;
    if (d.id.empty()) return "empty implementation ID";
    if (!valid_range(d.shape.outputs) || !valid_range(d.shape.inputs) || !valid_range(d.shape.tokens))
        return "invalid shape/token range";
    if (!memory::is_power_of_two(d.workspace.alignment)) return "workspace alignment must be a power of two";
    if (!valid_device_kind(d.device.kind)) return "unknown device kind";
    if (d.device.architecture.empty() || d.device.minimum_version > d.device.maximum_version)
        return "invalid device architecture/version range";
    return std::nullopt;
}
} // namespace

std::optional<std::string> validate_request(const KernelRequest& request) {
    if (auto error = format_error(request, request.physical)) return error;
    if (request.shape.outputs == 0 || request.shape.inputs == 0 || request.shape.tokens == 0)
        return "shape/token width must be positive";
    if (!memory::is_power_of_two(request.workspace_alignment)) return "workspace alignment must be a power of two";
    return std::nullopt;
}

std::optional<std::string> incompatibility(const KernelDescriptor& d, const KernelRequest& r,
                                         const DeviceCaps& caps) {
    if (auto error = validate_request(r)) return "invalid kernel request: " + *error;
    if (!valid_device_kind(caps.kind)) return "invalid kernel request: unknown device kind";
    if (caps.architecture.empty()) return "invalid kernel request: device architecture must be nonempty";
    if (auto error = descriptor_error(d)) return "invalid kernel descriptor: " + *error;
    if (d.operation != r.operation) return "operation";
    if (d.phase != r.phase) return "phase";
    if (d.weights != r.weights) return "weight encoding";
    if (d.activation != r.activation) return "activation dtype";
    if (d.layout != r.layout) return "physical layout";
    if (d.numerics != r.numerics) return "numerical contract";
    if (d.kv_pair != r.kv_pair) return "KV pair";
    if (!matches(r.shape.outputs, d.shape.outputs) || !matches(r.shape.inputs, d.shape.inputs) ||
        !matches(r.shape.tokens, d.shape.tokens)) return "shape/token width";
    if (d.required_physical.quantization_group != r.physical.quantization_group)
        return "quantization group";
    if (r.physical.alignment < d.required_physical.alignment) return "base alignment";
    if (!stride_matches(r.physical.weight_row_stride, d.required_physical.weight_row_stride) ||
        !stride_matches(r.physical.activation_row_stride, d.required_physical.activation_row_stride) ||
        !stride_matches(r.physical.output_row_stride, d.required_physical.output_row_stride)) return "byte strides";
    if (r.workspace_available < d.workspace.bytes ||
        (d.workspace.bytes != 0 && r.workspace_alignment < d.workspace.alignment)) return "workspace size/alignment";
    if (caps.kind != d.device.kind || caps.architecture != d.device.architecture ||
        caps.architecture_version < d.device.minimum_version ||
        caps.architecture_version > d.device.maximum_version ||
        (caps.features & d.device.required_features) != d.device.required_features)
        return "device architecture/capabilities";
    return std::nullopt;
}

std::string describe(const KernelRequest& r, const DeviceCaps& caps) {
    std::ostringstream text;
    text << operation_name(r.operation) << '/' << phase_name(r.phase)
         << " weights=" << weight_name(r.weights) << " activation=" << activation_name(r.activation)
         << " layout=" << layout_name(r.layout) << " numerics=" << numerical_name(r.numerics)
         << " N/K/T=" << r.shape.outputs << '/' << r.shape.inputs << '/' << r.shape.tokens
         << " group=" << r.physical.quantization_group
         << " alignment=" << r.physical.alignment << " strides=" << r.physical.weight_row_stride
         << '/' << r.physical.activation_row_stride << '/' << r.physical.output_row_stride
         << " workspace=" << r.workspace_available << '@' << r.workspace_alignment
         << " device=" << caps.architecture << ':' << caps.architecture_version
         << " features=" << caps.features;
    if (r.kv_pair) text << " KV=" << codec_name(r.kv_pair->key) << '/'
                        << codec_name(r.kv_pair->value);
    else text << " KV=none";
    return text.str();
}

PreparedKernel::PreparedKernel(LaunchFunction entry, std::shared_ptr<const void> metadata,
                               KernelRequest bindings, WorkspaceRequirement workspace)
    : entry_(entry), metadata_(std::move(metadata)), bindings_(std::move(bindings)), workspace_(workspace) {
    if (!entry_ || !metadata_) throw std::invalid_argument("prepared launcher requires owned metadata and entry");
    if (!memory::is_power_of_two(workspace_.alignment))
        throw std::invalid_argument("prepared workspace alignment must be a power of two");
}

void PreparedKernel::launch(const LaunchContext& context) const {
    if (!metadata_) throw std::logic_error("launch of moved-from prepared kernel");
    if (context.workspace.bytes < workspace_.bytes)
        throw std::invalid_argument("prepared kernel workspace is undersized");
    if (workspace_.bytes != 0 && !context.workspace.address)
        throw std::invalid_argument("prepared kernel workspace requires a non-null address");
    if (context.workspace.address &&
        reinterpret_cast<std::uintptr_t>(context.workspace.address) % workspace_.alignment != 0)
        throw std::invalid_argument("prepared kernel workspace address is misaligned");
    entry_(metadata_.get(), context);
}

void KernelRegistry::add(std::shared_ptr<const IKernelImplementation> implementation) {
    if (!implementation) throw std::invalid_argument("null kernel implementation");
    auto d = implementation->descriptor();
    if (auto error = descriptor_error(d))
        throw std::invalid_argument("invalid kernel descriptor: " + d.id + ": " + *error);
    for (const auto& entry : entries_) if (entry.descriptor.id == d.id)
        throw std::invalid_argument("duplicate kernel implementation: " + d.id);
    entries_.push_back({std::move(implementation), std::move(d)});
}

PreparedKernel KernelRegistry::resolve(const KernelRequest& request, const DeviceCaps& device,
                                      const PairTuningProfile* preferences) const {
    if (auto error = validate_request(request))
        throw std::invalid_argument("invalid kernel request: " + *error);
    if (!valid_device_kind(device.kind)) throw std::invalid_argument("invalid kernel request: unknown device kind");
    if (device.architecture.empty()) throw std::invalid_argument("invalid kernel request: device architecture must be nonempty");
    std::vector<const Entry*> compatible;
    std::ostringstream rejected;
    for (const auto& entry : entries_) {
        auto reason = incompatibility(entry.descriptor, request, device);
        if (!reason && !entry.implementation->supports(request, device)) reason = "implementation support predicate";
        if (reason) rejected << " [" << entry.descriptor.id << ": " << *reason << ']';
        else compatible.push_back(&entry);
    }
    if (compatible.empty())
        throw std::runtime_error("missing compatible kernel: " + describe(request, device) + rejected.str());
    const auto preference_rank = [preferences](const std::string& id) {
        if (!preferences) return std::size_t{0};
        const auto& ids = preferences->preferred_implementations;
        return static_cast<std::size_t>(std::find(ids.begin(), ids.end(), id) - ids.begin());
    };
    std::sort(compatible.begin(), compatible.end(), [&](const Entry* a, const Entry* b) {
        const auto ar = preference_rank(a->descriptor.id), br = preference_rank(b->descriptor.id);
        if (ar != br) return ar < br;
        if (a->descriptor.default_priority != b->descriptor.default_priority)
            return a->descriptor.default_priority > b->descriptor.default_priority;
        return a->descriptor.id < b->descriptor.id;
    });
    const auto& chosen = *compatible.front();
    KernelPreparationContext context{device};
    auto result = chosen.implementation->prepare(request, context);
    if (result.bindings() != request || result.workspace() != chosen.descriptor.workspace)
        throw std::runtime_error("prepared kernel changed binding/workspace contract: " + chosen.descriptor.id);
    result.implementation_id_ = chosen.descriptor.id;
    struct LaunchLifetime {
        std::shared_ptr<const IKernelImplementation> implementation;
        std::shared_ptr<const void> metadata; // destroyed before implementation, even on assignment
    };
    auto lifetime = std::make_shared<const LaunchLifetime>(LaunchLifetime{chosen.implementation, result.metadata_});
    result.metadata_ = std::shared_ptr<const void>(std::move(lifetime), result.metadata_.get());
    return result;
}

} // namespace saga::backend
