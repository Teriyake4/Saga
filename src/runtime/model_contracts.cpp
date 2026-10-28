#include "saga/runtime/model_contracts.h"

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace saga::runtime {
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
template<class Enum> bool one_of(Enum value, std::initializer_list<Enum> allowed) {
    return std::find(allowed.begin(), allowed.end(), value) != allowed.end();
}

bool same_tensor(const backend::LogicalWeightBinding& a, const backend::LogicalWeightBinding& b) {
    return a.component_identity == b.component_identity && a.byte_offset == b.byte_offset &&
           a.byte_extent == b.byte_extent && a.encoding == b.encoding && a.shape == b.shape;
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

    std::unordered_map<std::string, std::uint64_t> components;
    for (const auto& component : artifact.components)
        require(!component.identity.empty() && components.emplace(component.identity, component.byte_size).second,
                "empty or duplicate artifact component identity");
    require(!spec.weights.empty(), "model contract requires weight bindings");
    std::unordered_set<std::string> logical_names;
    std::unordered_map<std::string, const backend::LogicalWeightBinding*> physical;
    std::vector<const backend::LogicalWeightBinding*> ranges;
    for (const auto& weight : spec.weights) {
        require(!weight.logical_name.empty() && logical_names.insert(weight.logical_name).second,
                "empty or duplicate logical weight name");
        require(!weight.physical_identity.empty(), "empty physical tensor identity");
        require(one_of(weight.encoding, {backend::WeightEncoding::none, backend::WeightEncoding::fp16,
                backend::WeightEncoding::bf16, backend::WeightEncoding::gsq3_group128_v1}), "unknown weight encoding");
        require(weight.shape.outputs && weight.shape.inputs && weight.shape.tokens,
                "weight dimensions must be positive");
        require(weight.byte_extent > 0, "weight extent must be positive");
        const auto component = components.find(weight.component_identity);
        require(!weight.component_identity.empty() && component != components.end(),
                "weight references unknown artifact component");
        require(weight.byte_offset <= component->second &&
                weight.byte_extent <= component->second - weight.byte_offset,
                "weight extent outside artifact component");
        const auto [known, inserted] = physical.emplace(weight.physical_identity, &weight);
        if (!inserted) {
            require(same_tensor(*known->second, weight), "conflicting aliases for physical tensor");
            continue;
        }
        ranges.push_back(&weight);
    }
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        for (std::size_t j = i + 1; j < ranges.size(); ++j) {
            const auto& a = *ranges[i]; const auto& b = *ranges[j];
            if (a.component_identity != b.component_identity) continue;
            const bool overlap = a.byte_offset <= b.byte_offset
                ? b.byte_offset - a.byte_offset < a.byte_extent
                : a.byte_offset - b.byte_offset < b.byte_extent;
            require(!overlap, "overlapping ranges have different physical identities");
        }
    }

    require(one_of(program.phase, {backend::Phase::prefill, backend::Phase::decode, backend::Phase::verify}),
            "unknown model program phase");
    // An empty/no-op program is not a supported model contract convention here.
    require(!program.calls.empty(), "model contract requires program calls");
    for (const auto& call : program.calls) {
        require(call.phase == program.phase, "kernel phase mismatches model program");
        if (const auto problem = backend::validate_request(call))
            throw std::invalid_argument("invalid model kernel request: " + *problem);
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
void BackendRegistry::add(std::string id, std::shared_ptr<IGpuBackend> value) {
    if (id.empty() || !value) throw std::invalid_argument("backend registration requires ID and backend");
    if (std::any_of(entries_.begin(), entries_.end(), [&](const auto& e) { return e.id == id; }))
        throw std::invalid_argument("duplicate backend ID: " + id);
    entries_.push_back({std::move(id), std::move(value)});
}
std::shared_ptr<IGpuBackend> BackendRegistry::find(const std::string& id) const {
    const auto it = std::find_if(entries_.begin(), entries_.end(), [&](const auto& e) { return e.id == id; });
    if (it == entries_.end()) throw std::out_of_range("unknown backend ID: " + id);
    return it->value;
}
} // namespace saga::runtime
