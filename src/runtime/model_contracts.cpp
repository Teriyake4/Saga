#include "saga/runtime/model_contracts.h"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <limits>
#include <mutex>
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
void validate_artifact_capabilities(const ArtifactMetadata& artifact, ModelCapabilities available) {
    std::unordered_set<Capability> required;
    std::unordered_set<Capability> optional;
    for (auto capability : artifact.required_capabilities) {
        require(one_of(capability, {Capability::mtp, Capability::vision}), "unknown artifact capability");
        require(required.insert(capability).second, "duplicate artifact capability declaration");
        require(capability != Capability::mtp || available.mtp, "required artifact MTP capability unavailable");
        require(capability != Capability::vision || available.vision, "required artifact vision capability unavailable");
    }
    for (auto capability : artifact.optional_capabilities) {
        require(one_of(capability, {Capability::mtp, Capability::vision}), "unknown artifact capability");
        require(optional.insert(capability).second, "duplicate artifact capability declaration");
        require(!required.contains(capability), "artifact capability is both required and optional");
    }
}
std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const char* message) {
    require(a <= std::numeric_limits<std::uint64_t>::max() - b, message);
    return a + b;
}
std::uint64_t checked_multiply(std::uint64_t a, std::uint64_t b, const char* message) {
    require(a == 0 || b <= std::numeric_limits<std::uint64_t>::max() / a, message);
    return a * b;
}
void validate_call_set(backend::Phase phase, const std::vector<backend::KernelRequest>& calls) {
    require(!calls.empty(), "model contract requires program calls");

    for (const auto& call : calls) {
        require(call.phase == phase, "kernel phase mismatches model program");
        if (const auto problem = backend::validate_request(call))
            throw std::invalid_argument("invalid model kernel request: " + *problem);
    }
}
void validate_program(const backend::ModelProgram& program) {
    require(one_of(program.phase, {backend::Phase::prefill, backend::Phase::decode, backend::Phase::verify}),
            "unknown model program phase");
    require(program.alternatives.size() <= backend::maximum_program_alternatives,
            "model program alternative limit exceeded");
    std::size_t total_calls = program.calls.size();
    require(total_calls <= backend::maximum_program_calls, "model program total call limit exceeded");
    validate_call_set(program.phase, program.calls);
    std::unordered_set<std::string> identities{"primary"};
    for (const auto& alternative : program.alternatives) {
        require(!alternative.identity.empty() && identities.insert(alternative.identity).second,
                "empty, reserved, or duplicate program alternative identity");
        require(total_calls <= backend::maximum_program_calls &&
                alternative.calls.size() <= backend::maximum_program_calls - total_calls,
                "model program total call limit exceeded");
        total_calls += alternative.calls.size();
        validate_call_set(program.phase, alternative.calls);
    }
}
bool valid_category(memory::Category category) {
    return one_of(category, {memory::Category::text_weights, memory::Category::draft_weights,
        memory::Category::vision_weights, memory::Category::main_kv, memory::Category::draft_kv,
        memory::Category::fp32_recurrence, memory::Category::convolution, memory::Category::continuation,
        memory::Category::provisional, memory::Category::rollback_records, memory::Category::activations,
        memory::Category::kernel_workspace, memory::Category::logits, memory::Category::distributions,
        memory::Category::acceptance, memory::Category::media_weights, memory::Category::media_workspace,
        memory::Category::media_features, memory::Category::host_mappings, memory::Category::staging,
        memory::Category::request_buffers, memory::Category::pinned_transfer, memory::Category::pinned_control,
        memory::Category::runtime_overhead, memory::Category::graph_overhead});
}
bool weight_category(memory::Category category) {
    return one_of(category, {memory::Category::text_weights, memory::Category::draft_weights,
                             memory::Category::vision_weights});
}
bool state_category(StateKind kind, memory::Category category) {
    switch (kind) {
    case StateKind::key_value: return false;
    case StateKind::recurrent: return category == memory::Category::fp32_recurrence;
    case StateKind::convolution: return category == memory::Category::convolution;
    case StateKind::auxiliary:
        return one_of(category, {memory::Category::continuation, memory::Category::provisional,
                                 memory::Category::rollback_records, memory::Category::activations});
    }
    return false;
}
std::uint64_t dtype_bytes(StateDType dtype) {
    switch (dtype) {
    case StateDType::u8: case StateDType::i8: return 1;
    case StateDType::f16: case StateDType::bf16: return 2;
    case StateDType::f32: case StateDType::i32: return 4;
    case StateDType::i64: return 8;
    }
    throw std::invalid_argument("unknown state dtype");
}
void validate_rows(const SequenceHandle& sequence, const std::vector<StepRow>& rows,
                   const FrontendContract& frontend, std::uint64_t maximum_context_tokens) {
    require(static_cast<bool>(sequence), "step requires a valid sequence handle");
    require(rows.size() == 1, "initial step boundary supports exactly one row");
    const auto& row = rows.front();
    require(!row.token_ids.empty() && row.token_ids.size() == row.positions.size(),
            "step token and position counts must be equal and nonzero");
    require(maximum_context_tokens > 0 && row.context_tokens > 0 &&
            row.context_tokens <= maximum_context_tokens && row.token_ids.size() <= row.context_tokens &&
            row.chunk_tokens > 0 && row.chunk_tokens <= row.token_ids.size(),
            "invalid step context or chunk bounds");
    for (std::size_t i = 0; i < row.token_ids.size(); ++i) {
        require(row.token_ids[i] < frontend.vocabulary_size, "token ID is outside frontend vocabulary");
        require(row.positions[i] < row.context_tokens, "step position is outside context");
    }
}
}

void validate_model_contract(const ArtifactMetadata& artifact, const backend::ModelSpec& spec,
                             const backend::ModelProgram& program, const RuntimeOptions& options,
                             ModelCapabilities capabilities) {
    require(!artifact.identity.empty() && !spec.model_identity.empty(), "artifact and model identities are required");
    validate_artifact_capabilities(artifact, capabilities);
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
    validate_program(program);
}

namespace detail {
struct StepResultState {
    std::mutex mutex;
    bool claimed{false};
    std::optional<OwningStepResult> value;
};
}
PendingStep::PendingStep(Completion completion, std::shared_ptr<detail::StepResultState> result) noexcept
    : completion_(std::move(completion)), result_(std::move(result)) {}
OwningStepResult PendingStep::result() const {
    if (completion_.status() == CompletionStatus::pending)
        throw std::logic_error("pending step result is not available");
    completion_.result();
    if (!result_) throw std::logic_error("invalid pending step");
    std::lock_guard lock(result_->mutex);
    if (!result_->value) throw std::logic_error("successful step has no result");
    return *result_->value;
}
StepSource::StepSource(ResourceLease retained, StepResultBounds bounds)
    : completion_(std::move(retained)), result_(std::make_shared<detail::StepResultState>()),
      bounds_(std::move(bounds)) {
    if (bounds_.expected_logits_per_row && *bounds_.expected_logits_per_row > maximum_step_logits)
        throw std::invalid_argument("expected logits exceed step result limit");
}
PendingStep StepSource::pending() const { return PendingStep(completion_.completion(), result_); }
bool StepSource::succeed(OwningStepResult result) noexcept {
    if (!result_ || !completion_.valid() || result.rows.size() != maximum_step_rows) return false;
    for (const auto& row : result.rows) {
        if (row.consumed_frontier > bounds_.maximum_frontier || row.logits.size() > maximum_step_logits ||
            row.data.size() > maximum_step_result_bytes ||
            (bounds_.expected_logits_per_row && row.logits.size() != *bounds_.expected_logits_per_row)) return false;
    }
    std::lock_guard lock(result_->mutex);
    if (result_->claimed) return false;
    result_->claimed = true;
    result_->value.emplace(std::move(result));
    if (!completion_.succeed()) { result_->value.reset(); return false; }
    return true;
}
bool StepSource::fail(std::exception_ptr error) noexcept {
    if (!result_ || !completion_.valid()) return false;
    std::lock_guard lock(result_->mutex);
    if (result_->claimed) return false;
    result_->claimed = true;
    return completion_.fail(std::move(error));
}
bool StepSource::cancel() noexcept {
    if (!result_ || !completion_.valid()) return false;
    std::lock_guard lock(result_->mutex);
    if (result_->claimed) return false;
    result_->claimed = true;
    return completion_.cancel();
}

void validate_step_request(const PrefillRequest& request, const FrontendContract& frontend,
                           std::uint64_t maximum_context_tokens) {
    validate_rows(request.sequence, request.rows, frontend, maximum_context_tokens);
}
void validate_step_request(const DecodeRequest& request, const FrontendContract& frontend,
                           std::uint64_t maximum_context_tokens) {
    validate_rows(request.sequence, request.rows, frontend, maximum_context_tokens);
}
void validate_step_request(const VerifyRequest& request, const FrontendContract& frontend,
                           std::uint64_t maximum_context_tokens) {
    validate_rows(request.sequence, request.rows, frontend, maximum_context_tokens);
}

void validate_model_description(const ModelDescription& description, const RuntimeOptions& options) {
    const auto& inspection = description.inspection;
    const auto& artifact = inspection.artifact;
    const auto& spec = inspection.spec;
    require(!artifact.identity.empty() && !spec.model_identity.empty(), "artifact and model identities are required");
    validate_artifact_capabilities(artifact, inspection.capabilities);
    require(description.resources.model_max_context == spec.maximum_context_tokens,
            "resource profile model context mismatches model specification");
    require(options.context_tokens > 0 && options.context_tokens <= spec.maximum_context_tokens,
            "invalid requested context");
    require(!options.request_mtp || inspection.capabilities.mtp, "requested MTP capability unavailable");
    require(!options.request_vision || inspection.capabilities.vision, "requested vision capability unavailable");

    require(!options.request_mtp || description.resources.draft_enabled,
            "requested MTP capability is disabled in resource profile");
    require(!options.request_vision || description.resources.vision_enabled,
            "requested vision capability is disabled in resource profile");
    require(!inspection.frontend.identity.empty() && inspection.frontend.vocabulary_size > 0,
            "frontend identity and vocabulary are required");
    require(one_of(inspection.frontend.policy, {FrontendPolicy::caller_token_ids, FrontendPolicy::vocabulary_ids}),
            "unknown frontend policy");
    require(!description.resources.identity.empty(), "resource profile identity is required");
    require(one_of(description.resources.graphs, {memory::GraphMode::disabled, memory::GraphMode::enabled}),
            "unknown resource graph mode");
    std::unordered_map<std::string, const memory::Term*> terms;
    for (const auto& term : description.resources.terms) {
        require(!term.id.empty() && !term.owner.empty() && terms.emplace(term.id, &term).second,
                "empty or duplicate resource term identity or owner");
        require(valid_category(term.category) &&
                one_of(term.location, {memory::Location::host, memory::Location::pinned, memory::Location::device}) &&
                one_of(term.lifetime, {memory::Lifetime::persistent, memory::Lifetime::bringup,
                                       memory::Lifetime::execution}),
                "invalid resource term category, location, or lifetime");
    }
    std::unordered_set<std::uint32_t> special;
    for (auto id : inspection.frontend.special_token_ids)
        require(id < inspection.frontend.vocabulary_size && special.insert(id).second,
                "invalid or duplicate special token ID");

    std::unordered_set<std::string> weight_ids;
    for (const auto& weight : spec.weights) weight_ids.insert(weight.physical_identity);
    std::unordered_set<std::string> state_ids;
    std::unordered_set<std::string> state_resource_ids;
    for (const auto& state : inspection.state) {
        require(!state.id.empty() && state_ids.insert(state.id).second && !state.resource_id.empty() &&
                state_resource_ids.insert(state.resource_id).second,
                "empty or duplicate state identity or resource binding");
        require(one_of(state.kind, {StateKind::key_value, StateKind::recurrent, StateKind::convolution, StateKind::auxiliary}),
                "unknown state kind");
        require(one_of(state.lifetime, {StateLifetime::sequence, StateLifetime::request, StateLifetime::step}),
                "unknown state lifetime");
        require(one_of(state.location, {memory::Location::host, memory::Location::pinned, memory::Location::device}),
                "unknown state location");
        require(!weight_ids.contains(state.resource_id), "state resource aliases model weights");
        const auto element = dtype_bytes(state.dtype);
        require(!state.dimensions.empty(), "state dimensions are required");
        std::uint64_t schema_bytes = element;
        for (auto dimension : state.dimensions) {
            require(dimension > 0 && schema_bytes <= std::numeric_limits<std::uint64_t>::max() / dimension,
                    "state dimensions are invalid or overflow byte size");
            schema_bytes *= dimension;
        }
        if (state.kind == StateKind::recurrent)
            require(state.dtype == StateDType::f32, "recurrent state must use f32 dtype");

        if (state.kind == StateKind::key_value &&
            (state.resource_id == "main-kv" || state.resource_id == "draft-kv")) {
            require(state.location == memory::Location::device, "cache state must use device location");
            if (state.resource_id == "draft-kv")
                require(description.resources.draft_enabled, "draft cache state requires draft resources");
            continue;
        }

        const auto term_it = terms.find(state.resource_id);
        require(term_it != terms.end(), "state resource binding is missing");
        const auto& term = *term_it->second;
        require(term.location == state.location, "state resource domain mismatches schema");
        require(term.lifetime == memory::Lifetime::persistent || term.lifetime == memory::Lifetime::execution,
                "state resource must have persistent or execution lifetime");
        require(state_category(state.kind, term.category), "state resource category mismatches schema kind");
        require(term.category != memory::Category::runtime_overhead &&
                term.category != memory::Category::graph_overhead, "state resource must be a project allocation");
        require(term.physical_identity.empty() || term.physical_identity == state.resource_id,
                "state resource has an unexpected physical alias");
        for (const auto& weight : spec.weights)
            require(term.physical_identity.empty() || term.physical_identity != weight.physical_identity,
                    "state resource aliases model weights");
        const bool has_size = term.bytes.has_value() || term.bytes_per_context.has_value();
        if (has_size) {
            const auto fixed_bytes = term.bytes.value_or(0);
            const auto contextual_bytes = checked_multiply(term.bytes_per_context.value_or(0), options.context_tokens,
                                                            "state context resource size overflows");
            const auto required_bytes = checked_add(fixed_bytes, contextual_bytes,
                                                    "state resource size overflows");
            require(required_bytes >= schema_bytes,
                    "state resource allocation is zero or smaller than state schema");
        }
    }
    require(!description.programs.empty(), "model description requires at least one phase program");
    validate_model_contract(artifact, spec, description.programs.front(), options, inspection.capabilities);
    std::unordered_set<backend::Phase> phases;
    for (const auto& program : description.programs) {
        require(phases.insert(program.phase).second, "duplicate model program phase");
        validate_program(program);
    }
    if (description.resources.main_cache && description.resources.main_cache->geometry) {
        const auto& geometry = *description.resources.main_cache->geometry;
        require(geometry.layers == spec.main_attention.layers && geometry.heads == spec.main_attention.heads &&
                geometry.head_channels == spec.main_attention.head_channels,
                "resource profile main cache geometry mismatches model specification");
    }

    std::unordered_set<std::string> checked_weights;
    for (const auto& weight : spec.weights) {
        if (!checked_weights.insert(weight.physical_identity).second) continue;
        const auto term_it = terms.find(weight.physical_identity);
        require(term_it != terms.end(), "weight resource binding is missing");
        const auto& term = *term_it->second;
        require(weight_category(term.category), "weight resource category is not a weight category");
        require(term.category != memory::Category::draft_weights || description.resources.draft_enabled,
                "draft weight resource requires draft resources");
        require(term.category != memory::Category::vision_weights || description.resources.vision_enabled,
                "vision weight resource requires vision resources");
        require(term.lifetime == memory::Lifetime::persistent, "weight resource must have persistent lifetime");
        require(term.physical_identity.empty() || term.physical_identity == weight.physical_identity,
                "weight resource physical identity mismatches model binding");
        require(term.location == memory::Location::device || term.location == memory::Location::host ||
                term.location == memory::Location::pinned, "unknown weight resource location");
        const bool has_size = term.bytes.has_value() || term.bytes_per_context.has_value();
        if (has_size) {
            const auto fixed_bytes = term.bytes.value_or(0);
            const auto contextual_bytes = checked_multiply(term.bytes_per_context.value_or(0), options.context_tokens,
                                                            "weight context resource size overflows");
            const auto allocated_bytes = checked_add(fixed_bytes, contextual_bytes,
                                                     "weight resource size overflows");
            require(allocated_bytes > 0 && allocated_bytes >= weight.byte_extent,
                    "weight resource allocation is zero or smaller than binding");
        }
        for (const auto& state : inspection.state)
            require(state.resource_id != weight.physical_identity,
                    "state resource aliases model weights");
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
