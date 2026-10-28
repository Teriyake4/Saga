#include "saga/runtime/model_contracts.h"

#include <algorithm>

#include <stdexcept>
#include <unordered_set>

namespace saga::runtime {
namespace {
bool valid(backend::WeightEncoding value) {
    return value == backend::WeightEncoding::none || value == backend::WeightEncoding::fp16 ||
           value == backend::WeightEncoding::bf16 || value == backend::WeightEncoding::gsq3_group128_v1;
}
bool valid(backend::Phase value) {
    return value == backend::Phase::prefill || value == backend::Phase::decode || value == backend::Phase::verify;
}
bool valid(backend::Operation value) {
    return value == backend::Operation::projection || value == backend::Operation::gate_up_swiglu ||
           value == backend::Operation::attention || value == backend::Operation::kv_append ||
           value == backend::Operation::recurrent_update;
}
bool valid(backend::ActivationFormat value) {
    return value == backend::ActivationFormat::fp16 || value == backend::ActivationFormat::bf16 || value == backend::ActivationFormat::fp32;
}
bool valid(backend::Layout value) {
    return value == backend::Layout::contiguous || value == backend::Layout::gsq3_canonical_v1 ||
           value == backend::Layout::gsq3_tiled_v1 || value == backend::Layout::paged_kv_v1;
}
bool valid(backend::NumericalContract value) {
    return value == backend::NumericalContract::fp32_accumulate ||
           value == backend::NumericalContract::bf16_intermediate_rounding ||
           value == backend::NumericalContract::fp16_intermediate_rounding;
}
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

}

void validate_model_contract(const ArtifactMetadata& artifact, const backend::ModelSpec& spec,
                             const backend::ModelProgram& program, const RuntimeOptions& options,
                             ModelCapabilities capabilities) {
    require(!artifact.identity.empty() && !spec.model_identity.empty(), "artifact and model identities are required");
    require(options.context_tokens > 0 && spec.maximum_context_tokens > 0 &&
            options.context_tokens <= spec.maximum_context_tokens, "invalid requested context");
    require(spec.main_attention.layers > 0 && spec.main_attention.heads > 0 &&
            spec.main_attention.head_channels > 0, "model dimensions must be positive");
    require(!options.request_mtp || capabilities.mtp, "requested MTP capability unavailable");
    require(!options.request_vision || capabilities.vision, "requested vision capability unavailable");
    std::unordered_set<std::string> component_ids;
    for (const auto& component : artifact.components)
        require(!component.identity.empty() && component_ids.insert(component.identity).second,
                "empty or duplicate artifact component identity");
    std::unordered_set<std::string> logical_ids, physical_ids;
    for (const auto& weight : spec.weights) {
        require(!weight.logical_name.empty() && logical_ids.insert(weight.logical_name).second,
                "empty or duplicate logical weight name");
        require(!weight.physical_identity.empty() && physical_ids.insert(weight.physical_identity).second,
                "empty or conflicting physical weight identity");
        require(valid(weight.encoding), "unknown weight encoding");
        require(weight.shape.outputs > 0 && weight.shape.inputs > 0 && weight.shape.tokens > 0,
                "weight dimensions must be positive");
        const auto component = std::find_if(artifact.components.begin(), artifact.components.end(),
            [&](const auto& item) { return item.identity == weight.physical_identity; });
        require(component != artifact.components.end(), "weight references unknown artifact component");
        require(weight.byte_offset <= component->byte_size &&
                weight.byte_extent <= component->byte_size - weight.byte_offset,
                "weight extent outside artifact component");
    }
    require(valid(program.phase), "unknown model program phase");
    for (const auto& call : program.calls) {
        require(valid(call.operation) && valid(call.phase) && call.phase == program.phase,
                "unknown or mismatched kernel operation phase");
        require(valid(call.weights) && valid(call.activation) && valid(call.layout) && valid(call.numerics),
                "unknown kernel format or numerical contract");
        require(call.shape.outputs > 0 && call.shape.inputs > 0 && call.shape.tokens > 0,
                "kernel dimensions must be positive");
    }

}

void ModelRegistry::add(std::string id, std::shared_ptr<const IModelDefinition> value) {
    if (id.empty() || !value) throw std::invalid_argument("model registration requires ID and definition");
    if (std::any_of(entries_.begin(), entries_.end(), [&](const auto& e) { return e.id == id; }))
        throw std::invalid_argument("duplicate model ID: " + id);
    entries_.push_back({std::move(id), std::move(value)});
}
std::shared_ptr<const IModelDefinition> ModelRegistry::find(const std::string& id) const {
    const auto it = std::find_if(entries_.begin(), entries_.end(), [&](const auto& e) { return e.id == id; });
    if (it == entries_.end()) throw std::out_of_range("unknown model ID: " + id);
    return it->value;
}
void BackendRegistry::add(std::string id, std::shared_ptr<const IGpuBackend> value) {
    if (id.empty() || !value) throw std::invalid_argument("backend registration requires ID and backend");
    if (std::any_of(entries_.begin(), entries_.end(), [&](const auto& e) { return e.id == id; }))
        throw std::invalid_argument("duplicate backend ID: " + id);
    entries_.push_back({std::move(id), std::move(value)});
}
std::shared_ptr<const IGpuBackend> BackendRegistry::find(const std::string& id) const {
    const auto it = std::find_if(entries_.begin(), entries_.end(), [&](const auto& e) { return e.id == id; });
    if (it == entries_.end()) throw std::out_of_range("unknown backend ID: " + id);
    return it->value;
}
} // namespace saga::runtime
